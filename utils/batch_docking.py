"""
High-Throughput Virtual Screening (HTVS) & Batch Docking Engine
Automates multi-ligand library docking, affinity scoring, Ligand Efficiency (LE),
Lipophilic Ligand Efficiency (LLE), and generates ranked hit leaderboards.
"""

import os
import math
import tempfile
import pandas as pd
from typing import List, Dict, Callable, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen

try:
    from utils.ligand_prep import prepare_ligand_from_smiles
except ImportError:
    from utils.ligand_prep import prepare_ligand_pdbqt
    def prepare_ligand_from_smiles(smiles: str, output_pdbqt_path: str):
        return prepare_ligand_pdbqt(None, output_pdbqt_path, is_smiles=True, smiles_str=smiles)
from utils.docking_runner import run_docking, parse_vina_output
from utils.admet_profiler import calculate_comprehensive_admet

def parse_batch_input(file_content: str, filename: str) -> List[Dict[str, str]]:
    """
    Parses CSV, TXT, or SDF file content into a standardized list of {name, smiles}.
    """
    compounds = []
    fname_lower = filename.lower()

    if fname_lower.endswith(".sdf"):
        # Parse multi-mol SDF
        supplier = Chem.SDMolSupplier()
        supplier.SetData(file_content)
        for idx, mol in enumerate(supplier):
            if mol is not None:
                name = mol.GetProp("_Name") if mol.HasProp("_Name") and mol.GetProp("_Name").strip() else f"CMPD_{idx+1:03d}"
                smiles = Chem.MolToSmiles(mol)
                compounds.append({"name": name, "smiles": smiles})
    else:
        # Parse CSV or TSV or TXT
        lines = [line.strip() for line in file_content.strip().splitlines() if line.strip()]
        if not lines:
            return []

        # Check for header
        first_line = lines[0].lower()
        has_header = any(keyword in first_line for keyword in ["smiles", "name", "id", "compound"])
        data_lines = lines[1:] if has_header else lines

        delim = "," if "," in lines[0] else ("\t" if "\t" in lines[0] else " ")

        for idx, line in enumerate(data_lines):
            parts = [p.strip() for p in line.split(delim) if p.strip()]
            if not parts:
                continue

            if len(parts) == 1:
                # Only smiles provided
                compounds.append({"name": f"CMPD_{idx+1:03d}", "smiles": parts[0]})
            else:
                # Could be (Name, SMILES) or (SMILES, Name)
                # Check which one is a valid SMILES
                cand1, cand2 = parts[0], parts[1]
                m1 = Chem.MolFromSmiles(cand1)
                m2 = Chem.MolFromSmiles(cand2)
                if m1 is not None and m2 is None:
                    compounds.append({"name": cand2, "smiles": cand1})
                elif m2 is not None:
                    compounds.append({"name": cand1, "smiles": cand2})
                else:
                    compounds.append({"name": f"CMPD_{idx+1:03d}", "smiles": cand1})

    return compounds

def calculate_ligand_efficiency(delta_g: float, heavy_atom_count: int) -> float:
    """
    Ligand Efficiency (LE) = -Delta G / N_heavy (kcal/mol per non-hydrogen atom).
    Standard criterion: LE >= 0.30 kcal/mol/atom indicates promising lead-likeness.
    """
    if heavy_atom_count <= 0:
        return 0.0
    return round(-delta_g / heavy_atom_count, 3)

def calculate_lle(affinity_kcal: float, logp: float, temp_k: float = 298.15) -> float:
    """
    Lipophilic Ligand Efficiency (LLE) = pKi - LogP
    where pKi = -log10(Ki) = -Delta G / (2.303 * R * T).
    Target: LLE >= 5.0 for clinical candidate quality.
    """
    r_const = 0.0019872  # kcal/(mol*K)
    delta_g = affinity_kcal
    # pKi = -delta_g / (2.303 * R * T)
    pki = -delta_g / (2.302585 * r_const * temp_k)
    lle = pki - logp
    return round(lle, 2)

def run_batch_virtual_screening(
    compounds: List[Dict[str, str]],
    receptor_pdbqt_path: str,
    grid_params: dict,
    exhaustiveness: int = 4,
    num_modes: int = 3,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> pd.DataFrame:
    """
    Executes automated batch virtual screening across compound collection.
    Returns ranked DataFrame with Affinity, LE, LLE, ADMET alerts, and pose paths.
    """
    results = []
    total = len(compounds)

    for i, cmpd in enumerate(compounds):
        name = cmpd["name"]
        smiles = cmpd["smiles"]

        if progress_callback:
            progress_callback(i + 1, total, f"Docking {name} ({i+1}/{total})...")

        # 1. Chemical validation & ADMET
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            results.append({
                "Rank": i + 1,
                "Compound": name,
                "SMILES": smiles,
                "Status": "Failed (Invalid SMILES)",
                "Affinity_kcal": None,
                "Est_Ki_nM": None,
                "LE": None,
                "LLE": None,
                "MW": None,
                "LogP": None,
                "Lipinski_Violations": None,
                "PAINS_Alerts": None,
                "Pose_PDBQT": None
            })
            continue

        heavy_atoms = mol.GetNumHeavyAtoms()
        logp = round(Crippen.MolLogP(mol), 2)
        mw = round(Descriptors.MolWt(mol), 2)

        try:
            admet = calculate_comprehensive_admet(smiles)
            lipinski_viol = admet["lipinski"]["violations"]
            pains_count = admet["pains"]["alert_count"]
        except Exception:
            lipinski_viol = 0
            pains_count = 0

        # 2. Ligand 3D & PDBQT preparation
        temp_dir = tempfile.mkdtemp(prefix=f"htvs_{name}_")
        lig_pdbqt_path = os.path.join(temp_dir, f"{name}.pdbqt")
        docked_out_path = os.path.join(temp_dir, f"{name}_docked.pdbqt")

        try:
            prepare_ligand_from_smiles(smiles, lig_pdbqt_path)
        except Exception as e:
            results.append({
                "Rank": i + 1,
                "Compound": name,
                "SMILES": smiles,
                "Status": f"Prep Failed ({str(e)[:30]})",
                "Affinity_kcal": None,
                "Est_Ki_nM": None,
                "LE": None,
                "LLE": None,
                "MW": mw,
                "LogP": logp,
                "Lipinski_Violations": lipinski_viol,
                "PAINS_Alerts": pains_count,
                "Pose_PDBQT": None
            })
            continue

        # 3. AutoDock Vina Execution
        try:
            dock_res = run_docking(
                receptor_pdbqt=receptor_pdbqt_path,
                ligand_pdbqt=lig_pdbqt_path,
                output_pdbqt=docked_out_path,
                center_x=grid_params["center_x"],
                center_y=grid_params["center_y"],
                center_z=grid_params["center_z"],
                size_x=grid_params["size_x"],
                size_y=grid_params["size_y"],
                size_z=grid_params["size_z"],
                exhaustiveness=exhaustiveness,
                num_modes=num_modes
            )
            docked_poses = dock_res.get("poses", [])
            if not docked_poses:
                raise ValueError("No docked modes returned by Vina")

            best_affinity = docked_poses[0]["affinity"]
            
            # Estimated Ki: Ki = exp(DeltaG / RT)
            # R = 1.9872e-3 kcal/(mol*K), T = 298.15 K -> RT = 0.5925 kcal/mol
            rt = 0.5925
            ki_molar = math.exp(best_affinity / rt)
            ki_nm = round(ki_molar * 1e9, 1)

            le = calculate_ligand_efficiency(best_affinity, heavy_atoms)
            lle = calculate_lle(best_affinity, logp)

            results.append({
                "Rank": 0,  # Will be assigned after sort
                "Compound": name,
                "SMILES": smiles,
                "Status": "Success",
                "Affinity_kcal": best_affinity,
                "Est_Ki_nM": ki_nm,
                "LE": le,
                "LLE": lle,
                "MW": mw,
                "LogP": logp,
                "Lipinski_Violations": lipinski_viol,
                "PAINS_Alerts": pains_count,
                "Pose_PDBQT": docked_out_path
            })
        except Exception as e:
            results.append({
                "Rank": i + 1,
                "Compound": name,
                "SMILES": smiles,
                "Status": f"Docking Failed ({str(e)[:30]})",
                "Affinity_kcal": None,
                "Est_Ki_nM": None,
                "LE": None,
                "LLE": None,
                "MW": mw,
                "LogP": logp,
                "Lipinski_Violations": lipinski_viol,
                "PAINS_Alerts": pains_count,
                "Pose_PDBQT": None
            })

    # Sort successful hits by binding affinity (most negative first)
    df = pd.DataFrame(results)
    if "Affinity_kcal" in df.columns:
        df_succ = df[df["Status"] == "Success"].sort_values(by="Affinity_kcal", ascending=True)
        df_fail = df[df["Status"] != "Success"]
        df_sorted = pd.concat([df_succ, df_fail], ignore_index=True)
        df_sorted["Rank"] = range(1, len(df_sorted) + 1)
        return df_sorted

    return df
