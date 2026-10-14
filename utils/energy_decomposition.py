"""
Per-Residue Interaction Energy Decomposition & Protein-Ligand Interaction Fingerprints (PLIF)
Decomposes binding free energy into Van der Waals (vdW), electrostatic (Coulombic),
and hydrogen bonding contributions for each active site residue.
"""

import math
import numpy as np
import pandas as pd
try:
    import plotly.graph_objects as go
except ImportError:
    go = None
try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None
from typing import Dict, List, Tuple

# Standard partial charges for amino acid polar/charged sidechains (AMBER ff14SB approximation)
RESIDUE_NET_CHARGES = {
    "ARG": +1.0, "LYS": +1.0, "HIS": +0.1,
    "ASP": -1.0, "GLU": -1.0,
    "SER": 0.0, "THR": 0.0, "ASN": 0.0, "GLN": 0.0,
    "TYR": 0.0, "TRP": 0.0, "CYS": 0.0, "MET": 0.0,
    "ALA": 0.0, "VAL": 0.0, "LEU": 0.0, "ILE": 0.0,
    "PRO": 0.0, "PHE": 0.0, "GLY": 0.0
}

def parse_pdb_atoms(pdb_lines: List[str]) -> List[dict]:
    """Extracts atom records with coordinates, element, residue name, and residue number."""
    atoms = []
    for line in pdb_lines:
        if line.startswith(("ATOM", "HETATM")):
            try:
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain = line[21:22].strip() or "A"
                res_num = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                elem = line[76:78].strip() if len(line) >= 78 else atom_name[0]
                atoms.append({
                    "name": atom_name,
                    "res_name": res_name,
                    "chain": chain,
                    "res_num": res_num,
                    "res_id": f"{res_name}{res_num}:{chain}",
                    "coord": np.array([x, y, z]),
                    "elem": elem.upper()
                })
            except (ValueError, IndexError):
                continue
    return atoms

def compute_residue_decomposition(protein_pdb_lines: List[str], ligand_pdbqt_lines: List[str], cutoff_distance: float = 6.0) -> pd.DataFrame:
    """
    Computes per-residue binding energy contributions (vdW, Electrostatics, H-bonds)
    between protein binding site residues and the docked ligand.

    NOTE: These are QUALITATIVE estimates from a heuristic 12-6 Lennard-Jones +
    distance-binned H-bond scoring model — not rigorous free energies. Values
    are deliberately NOT clamped to cosmetic ranges; treat large magnitudes
    as flags of the model's approximations, not as precise measurements.
    """
    prot_atoms = parse_pdb_atoms(protein_pdb_lines)
    lig_atoms = parse_pdb_atoms(ligand_pdbqt_lines)

    if not prot_atoms or not lig_atoms:
        return pd.DataFrame()

    lig_coords = np.array([a["coord"] for a in lig_atoms if a["elem"] != "H"])
    if len(lig_coords) == 0:
        lig_coords = np.array([a["coord"] for a in lig_atoms])

    # Bounding-box pre-filter: only protein atoms within (cutoff + margin) of
    # the ligand bounding box can possibly interact. Replaces the previous
    # unfiltered O(protein x ligand) double loop.
    lig_min = lig_coords.min(axis=0) - (cutoff_distance + 2.0)
    lig_max = lig_coords.max(axis=0) + (cutoff_distance + 2.0)
    prot_atoms = [
        a for a in prot_atoms
        if np.all(a["coord"] >= lig_min) and np.all(a["coord"] <= lig_max)
    ]
    if not prot_atoms:
        return pd.DataFrame()

    # Pair discovery: cKDTree when scipy is available, bounded brute force otherwise.
    prot_coords = np.array([a["coord"] for a in prot_atoms])
    lig_all_coords = np.array([a["coord"] for a in lig_atoms])
    if cKDTree is not None:
        tree = cKDTree(lig_all_coords)
        neighbors = tree.query_ball_point(prot_coords, r=cutoff_distance)
        pair_indices = {
            (pi, li) for pi, lidx in enumerate(neighbors) for li in lidx
        }
    else:
        pair_indices = set()
        for pi, pa in enumerate(prot_atoms):
            for li, la in enumerate(lig_atoms):
                if float(np.linalg.norm(pa["coord"] - la["coord"])) <= cutoff_distance:
                    pair_indices.add((pi, li))

    # Group protein atoms by residue
    res_map = {}
    for a in prot_atoms:
        if a["elem"] == "H":
            continue
        rid = a["res_id"]
        if rid not in res_map:
            res_map[rid] = {
                "res_name": a["res_name"],
                "res_num": a["res_num"],
                "atoms": []
            }
        res_map[rid]["atoms"].append(a)
    atom_to_res = {id(a): rid for rid, data in res_map.items() for a in data["atoms"]}

    # Accumulate per-protein-atom pair contributions, then aggregate by residue
    atom_vdw = {}
    atom_elec = {}
    atom_hbond = {}
    atom_hydro = {}
    atom_min_dist = {}

    # Dielectric constant for active site interior
    epsilon_r = 4.0
    coulomb_const = 332.0637  # kcal*A / (mol*e^2)

    for pi, li in pair_indices:
        pa = prot_atoms[pi]
        la = lig_atoms[li]
        if pa["elem"] == "H":
            continue
        p_coord = pa["coord"]
        l_coord = la["coord"]
        d = float(np.linalg.norm(p_coord - l_coord))
        if d > cutoff_distance or d <= 0.8:
            continue

        atom_min_dist[pi] = min(atom_min_dist.get(pi, 999.0), d)

        # 1. 12-6 Lennard-Jones vdW approximation (sigma ~ 3.5 A, epsilon ~ 0.12)
        sigma = 3.6
        eps = 0.12
        ratio = sigma / d
        lj = 4.0 * eps * (math.pow(min(ratio, 1.8), 12) - math.pow(min(ratio, 1.8), 6))
        atom_vdw[pi] = atom_vdw.get(pi, 0.0) + lj * 0.15

        # 2. Polar / H-bond interactions
        p_is_polar = pa["elem"] in ["N", "O", "S"]
        l_is_polar = la["elem"] in ["N", "O", "S", "F"]
        if p_is_polar and l_is_polar and 2.5 <= d <= 3.5:
            atom_hbond[pi] = atom_hbond.get(pi, 0) + 1
            atom_elec[pi] = atom_elec.get(pi, 0.0) - 1.8  # ~1.8 kcal/mol per H-bond

        # 3. Hydrophobic contacts
        if pa["elem"] == "C" and la["elem"] == "C" and 3.0 <= d <= 4.5:
            atom_hydro[pi] = atom_hydro.get(pi, 0) + 1
            atom_vdw[pi] = atom_vdw.get(pi, 0.0) - 0.15

    results = []
    for rid, data in res_map.items():
        min_dist = 999.0
        vdw_energy = 0.0
        elec_energy = 0.0
        hbond_count = 0
        hydrophobic_contacts = 0

        res_charge = RESIDUE_NET_CHARGES.get(data["res_name"], 0.0)

        # Aggregate per-atom contributions for this residue
        for idx, a in enumerate(prot_atoms):
            if atom_to_res.get(id(a)) != rid:
                continue
            vdw_energy += atom_vdw.get(idx, 0.0)
            elec_energy += atom_elec.get(idx, 0.0)
            hbond_count += atom_hbond.get(idx, 0)
            hydrophobic_contacts += atom_hydro.get(idx, 0)
            if idx in atom_min_dist and atom_min_dist[idx] < min_dist:
                min_dist = atom_min_dist[idx]

        # If residue is within binding pocket interaction radius
        if min_dist <= cutoff_distance:
            # Net charge electrostatic interaction
            if abs(res_charge) > 0.01:
                # Approximate ligand center charge interaction
                elec_energy += (res_charge * -0.5 * coulomb_const) / (epsilon_r * max(min_dist, 2.5)) * 0.05

            # No cosmetic clamping: values are qualitative estimates (see docstring)
            total_energy = round(vdw_energy + elec_energy, 2)

            results.append({
                "Residue": rid,
                "Res_Name": data["res_name"],
                "Res_Num": data["res_num"],
                "Min_Distance_A": round(min_dist, 2),
                "vdW_Energy_kcal": round(vdw_energy, 2),
                "Elec_Energy_kcal": round(elec_energy, 2),
                "HBonds": hbond_count,
                "Hydrophobic": hydrophobic_contacts,
                "Total_Interaction_kcal": total_energy
            })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    # Sort by strongest binding contribution (most negative total energy)
    df = df.sort_values(by="Total_Interaction_kcal", ascending=True).reset_index(drop=True)
    return df

def generate_decomposition_plot(df: pd.DataFrame, top_n: int = 15):
    """
    Generates a publication-grade academic stacked bar chart
    illustrating per-residue vdW vs. electrostatic energy decomposition.
    """
    if go is None or df.empty:
        return None

    top_df = df.head(top_n)

    fig = go.Figure()

    # Van der Waals component
    fig.add_trace(go.Bar(
        name="Van der Waals / Steric",
        x=top_df["Residue"],
        y=top_df["vdW_Energy_kcal"],
        marker_color="#2563eb",  # Academic Royal Blue
        opacity=0.88,
        hovertemplate="<b>%{x}</b><br>vdW: %{y:.2f} kcal/mol<extra></extra>"
    ))

    # Electrostatic component
    fig.add_trace(go.Bar(
        name="Electrostatic & H-Bond",
        x=top_df["Residue"],
        y=top_df["Elec_Energy_kcal"],
        marker_color="#059669",  # Emerald Green
        opacity=0.88,
        hovertemplate="<b>%{x}</b><br>Electrostatic: %{y:.2f} kcal/mol<extra></extra>"
    ))

    # Total interaction line overlay
    fig.add_trace(go.Scatter(
        name="Total Delta G (Est.)",
        x=top_df["Residue"],
        y=top_df["Total_Interaction_kcal"],
        mode="lines+markers",
        line=dict(color="#dc2626", width=2.5, dash="dot"),
        marker=dict(size=7, color="#dc2626"),
        hovertemplate="<b>%{x}</b><br>Total: %{y:.2f} kcal/mol<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text="<b>Per-Residue Interaction Energy Decomposition (kcal/mol) — Qualitative Estimates</b>",
            font=dict(family="Inter, sans-serif", size=15, color="#0f172a")
        ),
        barmode="relative",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(
            title="<b>Active Site Residue</b>",
            tickangle=-45,
            showgrid=False,
            linecolor="#cbd5e1",
            tickfont=dict(family="JetBrains Mono, monospace", size=11, color="#334155")
        ),
        yaxis=dict(
            title="<b>Delta G Contribution (kcal/mol)</b>",
            gridcolor="#f1f5f9",
            zeroline=True,
            zerolinecolor="#94a3b8",
            zerolinewidth=1.5,
            linecolor="#cbd5e1",
            tickfont=dict(family="Inter, sans-serif", size=11, color="#334155")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family="Inter, sans-serif", size=11, color="#334155"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e2e8f0",
            borderwidth=1
        ),
        hovermode="x unified"
    )

    return fig
