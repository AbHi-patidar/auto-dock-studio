"""
Crystallographic Re-docking & RMSD Validation Engine for AutoDock Studio
Performs self-docking of native co-crystallized ligands to calculate
Heavy-Atom Root-Mean-Square Deviation (RMSD) against experimental coordinates.
Standard journal protocol requirement for peer review (Nature, J. Med. Chem.).
"""

import os
import math
import logging
import numpy as np

logger = logging.getLogger("autodock_studio")

def extract_native_ligand_pdb(raw_pdb_path: str, ligand_resname: str, output_path: str) -> str:
    """
    Extracts all HETATM lines for the target co-crystallized ligand from raw PDB.
    """
    lig_lines = []
    with open(raw_pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("HETATM") and line[17:20].strip() == ligand_resname.strip():
                lig_lines.append(line)

    if not lig_lines:
        raise ValueError(f"No HETATM coordinates found for ligand '{ligand_resname}' in {raw_pdb_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lig_lines)
        f.write("END\n")

    return output_path

def calculate_heavy_atom_rmsd(crystal_pdb_lines: list, docked_pdb_lines: list) -> float:
    """
    Calculates the heavy-atom Root Mean Square Deviation (RMSD) in Angstroms
    between experimental crystal coordinates and docked binding pose.
    """
    def parse_coords(lines):
        coords = []
        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                # Element detection (robust to non-standard records):
                # 1) standard PDB columns 77-78;
                # 2) trailing token (parse_vina_output emits synthetic PDB lines
                #    with the element appended after the columns);
                # 3) first character of the atom name as a last resort.
                elem = line[76:78].strip() if len(line) >= 78 else ""
                if not elem or not elem.replace("+", "").replace("-", "").isalpha():
                    tail = line.split()
                    if tail and len(tail[-1]) <= 2 and tail[-1].isalpha():
                        elem = tail[-1]
                    else:
                        elem = line[12:14].strip()
                if elem.upper() == "H":
                    continue  # Ignore hydrogens for heavy-atom RMSD
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except (ValueError, IndexError):
                    pass
        return np.array(coords)

    c_arr = parse_coords(crystal_pdb_lines)
    d_arr = parse_coords(docked_pdb_lines)

    if len(c_arr) == 0 or len(d_arr) == 0:
        return 999.0

    # Atom order is assumed to match between crystal and docked ligand; if it
    # does not (different residue atom ordering), the RMSD is meaningless rather
    # than an error. Warn loudly instead of silently truncating on mismatch.
    if len(c_arr) != len(d_arr):
        logger.warning(
            "RMSD atom-count mismatch: crystal=%d vs docked=%d heavy atoms. "
            "Truncating to the shorter set; result is unreliable if atom "
            "orders differ between the two conformers.",
            len(c_arr), len(d_arr)
        )
    # Match counts if minor differences occur (e.g. partial crystallographic occupancy)
    min_len = min(len(c_arr), len(d_arr))
    c_arr = c_arr[:min_len]
    d_arr = d_arr[:min_len]

    diff = c_arr - d_arr
    rmsd = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
    return round(rmsd, 2)

def evaluate_validation_tier(rmsd: float) -> dict:
    """
    Categorizes docking protocol accuracy according to medicinal chemistry standards.
    """
    if rmsd <= 2.0:
        tier = "Validated (High Fidelity)"
        color = "#059669" # Emerald
        desc = "RMSD <= 2.0 A satisfies the gold-standard journal criteria. The docking protocol accurately reproduces the experimental crystallographic binding mode."
        passed = True
    elif rmsd <= 3.0:
        tier = "Marginal Agreement"
        color = "#d97706" # Amber
        desc = "RMSD between 2.0 A and 3.0 A represents acceptable qualitative agreement with slight rotational or substituent displacement."
        passed = False
    else:
        tier = "Divergent (Unvalidated)"
        color = "#dc2626" # Crimson
        desc = "RMSD > 3.0 A indicates substantial deviation from experimental coordinates. Consider enlarging the grid box or testing flexible sidechains."
        passed = False

    return {
        "rmsd": rmsd,
        "tier": tier,
        "color": color,
        "desc": desc,
        "passed": passed
    }
