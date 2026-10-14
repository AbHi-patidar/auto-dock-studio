"""
Protein Preparation Module for AutoDock Studio
Automates PDB downloading, cleaning, pocket detection, catalytic metal preservation,
blind docking grid calculation, and PDBQT conversion.
"""

import os
import re
import logging
import urllib.request
import urllib.error
import numpy as np

logger = logging.getLogger("autodock_studio")

# PDB IDs are 4 alphanumeric characters starting with a digit (e.g. 1HSG, 6LU7)
PDB_ID_PATTERN = re.compile(r"^[0-9][A-Za-z0-9]{3}$")

STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL", "HID", "HIE", "HIP", "CYX", "ASH", "GLH", "LYN"
}

SOLVENT_AND_SALTS = {
    "HOH", "WAT", "H2O", "DOD", "TIP", "CL", "NA", "SO4", "PO4",
    "K", "EDO", "PEG", "GOL", "DMS", "ACT", "FMT", "ACE", "NME", "BME", "MPD"
}

# Essential Catalytic Metal Ions and Co-factors
CATALYTIC_METALS = {"ZN", "MG", "MN", "FE", "CA", "CU", "NI", "CO", "CD"}
ESSENTIAL_COFACTORS = {"HEM", "NAD", "NAP", "FAD", "FMN", "ATP", "ADP", "GTP", "GDP", "PLP", "SAM"}

def fetch_pdb_from_rcsb(pdb_id: str, output_path: str) -> str:
    """Download standard PDB file from RCSB database."""
    pdb_id = pdb_id.strip().upper()
    if not PDB_ID_PATTERN.match(pdb_id):
        raise ValueError(
            f"Invalid PDB ID '{pdb_id}': must be 4 characters starting with a digit (e.g. '1HSG')."
        )
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    req = urllib.request.Request(url, headers={"User-Agent": "AutoDockStudio/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"RCSB download failed for PDB ID '{pdb_id}': HTTP {e.code} ({e.reason}). "
            "Check that the ID exists on rcsb.org."
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach the RCSB PDB server for '{pdb_id}': {e.reason}. "
            "Check your network connection and try again."
        ) from e
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path

def clean_protein_pdb(
    input_pdb: str,
    output_pdb: str,
    remove_water: bool = True,
    preserve_metals: bool = True,
    preserve_cofactors: bool = True
):
    """
    Cleans raw PDB: removes solvent/salts, preserves catalytic metals and cofactors,
    extracts co-crystallized ligand if present, and isolates receptor coordinates.
    """
    protein_atoms = []
    hetero_atoms = {}
    preserved_hetero = []

    with open(input_pdb, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            record = line[:6].strip()
            if record == "ATOM":
                resname = line[17:20].strip()
                if resname in STANDARD_RESIDUES:
                    protein_atoms.append(line)
            elif record == "HETATM":
                resname = line[17:20].strip()
                
                # Check for water
                if resname in {"HOH", "WAT", "H2O"}:
                    if not remove_water:
                        protein_atoms.append(line)
                    continue

                # Check for catalytic metals
                if preserve_metals and resname in CATALYTIC_METALS:
                    protein_atoms.append(line)
                    preserved_hetero.append(f"Metal: {resname}")
                    continue

                # Check for essential cofactors
                if preserve_cofactors and resname in ESSENTIAL_COFACTORS:
                    protein_atoms.append(line)
                    preserved_hetero.append(f"Cofactor: {resname}")
                    continue

                # Common crystallization salts
                if resname in SOLVENT_AND_SALTS:
                    continue

                # Track small molecule ligands for pocket detection
                if resname not in hetero_atoms:
                    hetero_atoms[resname] = []
                hetero_atoms[resname].append(line)

    # Write cleaned protein
    if not protein_atoms:
        raise ValueError(
            f"No protein ATOM records found in '{input_pdb}'. "
            "The file may be empty, malformed, or contain no standard residues."
        )
    with open(output_pdb, "w", encoding="utf-8") as f:
        for line in protein_atoms:
            f.write(line)
        f.write("END\n")

    # Calculate coordinates
    coords = []
    for line in protein_atoms:
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])
        except (ValueError, IndexError):
            continue

    coords = np.array(coords) if coords else np.zeros((1, 3))
    center = coords.mean(axis=0) if len(coords) > 0 else np.array([0.0, 0.0, 0.0])
    span = (coords.max(axis=0) - coords.min(axis=0)) if len(coords) > 0 else np.array([20.0, 20.0, 20.0])

    # Analyze hetero atoms for binding pocket identification
    detected_ligands = {}
    for lig_name, lig_lines in hetero_atoms.items():
        if len(lig_lines) >= 6:  # Reasonable minimum for a drug-like small molecule
            l_coords = []
            for l in lig_lines:
                try:
                    l_coords.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
                except (ValueError, IndexError):
                    pass
            if l_coords:
                l_arr = np.array(l_coords)
                detected_ligands[lig_name] = {
                    "num_atoms": len(l_coords),
                    "center": l_arr.mean(axis=0).round(2).tolist(),
                    "lines": lig_lines
                }

    return {
        "num_atoms": len(protein_atoms),
        "center": center.round(2).tolist(),
        "dimensions": span.round(2).tolist(),
        "detected_ligands": detected_ligands,
        "preserved_hetero": list(set(preserved_hetero))
    }

def compute_blind_docking_box(clean_pdb_path: str, padding: float = 6.0) -> dict:
    """
    Computes a search space box that envelops the entire protein surface
    for blind docking when binding pocket coordinates are completely unknown.
    """
    coords = []
    with open(clean_pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except (ValueError, IndexError):
                    pass
    if not coords:
        return {"center": [0, 0, 0], "size": [40, 40, 40]}

    c_arr = np.array(coords)
    center = c_arr.mean(axis=0).round(2).tolist()
    # Span with padding, clamped to max 90 A for Vina practical limits
    min_pt = c_arr.min(axis=0) - padding
    max_pt = c_arr.max(axis=0) + padding
    raw_size = max_pt - min_pt
    size = np.clip(raw_size, 20.0, 90.0).round(2).tolist()
    if np.any(raw_size > 90.0):
        # The box no longer envelops the whole protein: warn loudly.
        logger.warning(
            "Blind-docking box clamped to 90.0 A (raw span was %s A). "
            "The search box does NOT cover the full protein surface; "
            "consider docking specific domains separately.",
            np.round(raw_size, 1).tolist()
        )

    return {
        "center": center,
        "size": size
    }

def receptor_pdb_to_pdbqt(clean_pdb_path: str, pdbqt_path: str) -> str:
    """
    Converts cleaned protein PDB into PDBQT format required by AutoDock Vina.
    Adds polar charges and AD4 atom types (C, A, N, OA, SA, HD, Zn, Mg, etc.)
    """
    pdbqt_lines = []
    
    with open(clean_pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            record = line[:6].strip()
            if record in ("ATOM", "HETATM"):
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                element = line[76:78].strip() if len(line) >= 78 else ""
                if not element:
                    element = atom_name[0]
                
                # Exclude non-polar hydrogens (Vina requires only polar hydrogens)
                if element == "H":
                    if not (atom_name.startswith("H") and ("N" in atom_name or "O" in atom_name or atom_name in ("H", "H1", "H2", "H3", "HD1", "HE2"))):
                        continue

                # AutoDock atom typing
                ad_type = element
                partial_charge = 0.000

                # Handle Catalytic Metals
                if res_name in CATALYTIC_METALS or element in CATALYTIC_METALS:
                    ad_type = element.capitalize()
                    partial_charge = 2.000 if element in ("ZN", "MG", "CA", "FE", "CU", "NI", "CO", "CD") else 1.000
                elif element == "C":
                    if res_name in {"PHE", "TYR", "TRP", "HIS", "HID", "HIE"} and atom_name in {
                        "CG", "CD1", "CD2", "CE1", "CE2", "CE3", "CZ", "CZ2", "CZ3", "CH2", "ND1", "NE2"
                    }:
                        ad_type = "A"
                    else:
                        ad_type = "C"
                    partial_charge = 0.100
                elif element == "N":
                    ad_type = "NA" if atom_name in {"ND1", "NE2"} and res_name in {"HIS", "HID", "HIE"} else "N"
                    partial_charge = -0.400 if ad_type == "NA" else -0.350
                elif element == "O":
                    ad_type = "OA"
                    partial_charge = -0.550
                elif element == "S":
                    ad_type = "SA"
                    partial_charge = -0.200
                elif element == "H":
                    ad_type = "HD"
                    partial_charge = 0.350

                coord_part = line[:54].ljust(54)
                occ_b = "  1.00  0.00"
                charge_str = f"{partial_charge:+6.3f}"
                type_str = f"{ad_type:>2}"
                
                pdbqt_line = f"{coord_part}{occ_b}    {charge_str} {type_str}\n"
                pdbqt_lines.append(pdbqt_line)

    with open(pdbqt_path, "w", encoding="utf-8") as f:
        f.writelines(pdbqt_lines)
        f.write("TER\n")

    return pdbqt_path
