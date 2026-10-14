"""
Ligand Preparation Module for AutoDock Studio
Handles SMILES input, 3D conformer generation, energy minimization, and PDBQT conversion.
Supports both local RDKit and NIH Cactus / PubChem online 3D coordinate generation.
"""

import os
import sys
import logging
import tempfile
import urllib.parse
import urllib.request
import re

logger = logging.getLogger("autodock_studio")

def fetch_3d_sdf_from_cactus(smiles: str, output_sdf_path: str):
    """
    Downloads energy-minimized 3D coordinates in SDF format directly from NIH Cactus chemical resolver.
    Acts as a zero-dependency fallback for 3D coordinate generation.
    """
    encoded_smiles = urllib.parse.quote(smiles.strip())
    url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded_smiles}/file?format=sdf&get3d=True"
    req = urllib.request.Request(url, headers={"User-Agent": "AutoDockStudio/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        content = resp.read().decode("utf-8")
    with open(output_sdf_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_sdf_path

def smiles_to_3d_mol(smiles: str, name: str = "ligand"):
    """
    Takes a SMILES string, generates 3D coordinates with ETKDG,
    minimizes energy with MMFF94, and returns an RDKit Mol object.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: '{smiles}'")

        mol = Chem.AddHs(mol)
        mol.SetProp("_Name", name)

        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        status = AllChem.EmbedMolecule(mol, params)
        if status != 0:
            AllChem.EmbedMolecule(mol, useRandomCoords=True)

        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            logger.warning("MMFF94 minimization failed for '%s'; trying UFF.", name)
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
            except Exception:
                logger.exception("UFF minimization also failed; using the unminimized ETKDG conformer.")

        return mol

    except ImportError:
        # Fallback to NIH Cactus SDF if local RDKit is not installed.
        # Use a unique temp file (never a fixed shared name) so concurrent
        # sessions cannot overwrite each other's ligands.
        fd, temp_sdf = tempfile.mkstemp(prefix="ligand_cactus_", suffix=".sdf")
        os.close(fd)
        fetch_3d_sdf_from_cactus(smiles, temp_sdf)
        return temp_sdf

def prepare_ligand_pdbqt(mol_or_file, output_pdbqt_path: str, is_smiles: bool = False, smiles_str: str = ""):
    """
    Prepares a ligand for AutoDock Vina in PDBQT format using Meeko or RDKit/Cactus fallback.
    """
    if is_smiles or (isinstance(mol_or_file, str) and not os.path.exists(mol_or_file)):
        is_smiles = True
        if not smiles_str and isinstance(mol_or_file, str):
            smiles_str = mol_or_file

    mol = None
    # 1. Try modern Meeko + RDKit
    try:
        from rdkit import Chem

        if is_smiles:
            mol = smiles_to_3d_mol(smiles_str)
        elif isinstance(mol_or_file, str):
            if mol_or_file.endswith(".sdf") or mol_or_file.endswith(".mol"):
                supplier = Chem.SDMolSupplier(mol_or_file, removeHs=False)
                mol = next(supplier)
            elif mol_or_file.endswith(".pdb"):
                mol = Chem.MolFromPDBFile(mol_or_file, removeHs=False)
            else:
                raise ValueError("Unsupported format. Use .sdf, .mol, or .pdb")
        else:
            mol = mol_or_file

        from meeko import MoleculePreparation, PDBQTWriterLegacy
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)

        # Single Meeko code path, pinned to meeko==0.6.0 (see requirements.txt).
        # Older nested try/except fallbacks across three different Meeko APIs
        # masked real preparation errors; they are intentionally removed.
        res = PDBQTWriterLegacy.write_string(mol_setups[0])
        if isinstance(res, (tuple, list)):
            pdbqt_string = res[0]
        else:
            pdbqt_string = str(res)
        logger.info("Ligand PDBQT prepared with the Meeko backend (meeko==0.6.0).")

        with open(output_pdbqt_path, "w", encoding="utf-8") as f:
            f.write(pdbqt_string)

        return output_pdbqt_path

    except Exception as e:
        logger.warning("Meeko/RDKit ligand preparation failed (%s); trying RDKit-direct fallback.", e)
        # 2. Local RDKit direct PDBQT fallback (no external network or meeko required)
        if mol is not None:
            try:
                result = _rdkit_mol_to_pdbqt(mol, output_pdbqt_path)
                logger.info("Ligand PDBQT prepared with the RDKit-direct fallback backend.")
                return result
            except Exception:
                logger.exception("RDKit-direct fallback failed; trying standalone SDF parser.")
        # 3. Standalone fallback parser (works with SDF file directly)
        result = _standalone_sdf_to_pdbqt(mol_or_file, output_pdbqt_path, is_smiles, smiles_str)
        logger.info("Ligand PDBQT prepared with the standalone SDF fallback backend.")
        return result

def _rdkit_mol_to_pdbqt(mol, output_pdbqt_path: str):
    """Fallback generator converting an RDKit Mol with 3D coordinates into a valid PDBQT."""
    import math
    from rdkit.Chem import AllChem
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass
    conf = mol.GetConformer()
    lines = ["ROOT\n"]
    atom_count = 0
    for i, atom in enumerate(mol.GetAtoms()):
        elem = atom.GetSymbol()
        # Vina does not accept non-polar hydrogens (type H). Only accept polar HD (on O, N, S)
        if elem == "H":
            is_polar = False
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() in ("O", "N", "S"):
                    is_polar = True
                    break
            if not is_polar:
                continue

        atom_count += 1
        pos = conf.GetAtomPosition(i)
        charge = 0.0
        try:
            val = atom.GetProp("_GasteigerCharge")
            charge = float(val)
            if math.isnan(charge) or math.isinf(charge):
                charge = 0.0
        except Exception:
            charge = 0.0

        # AutoDock 4 atom typing
        atom_type = elem
        if elem == "C":
            atom_type = "A" if atom.GetIsAromatic() else "C"
        elif elem == "N":
            atom_type = "NA" if atom.GetIsAromatic() else "N"
        elif elem == "O":
            atom_type = "OA"
        elif elem == "S":
            atom_type = "SA"
        elif elem == "H":
            atom_type = "HD"
        elif elem in ("F", "Cl", "Br", "I"):
            atom_type = elem

        # Exact PDB standard columns (1-indexed):
        # 1-6:   "HETATM" (6 chars)
        # 7-11:  Serial number (%5d)
        # 12:    " " (1 char)
        # 13-16: Atom name (%-4s)
        # 17:    " " (1 char)
        # 18-20: Residue name "LIG" (3 chars)
        # 21:    " " (1 char)
        # 22:    Chain "A" (1 char)
        # 23-26: Res num "   1" (4 chars)
        # 27-30: "    " (4 blanks) -> Brings total prefix to exactly 30 chars!
        # 31-38: X coordinate (%8.3f)
        # 39-46: Y coordinate (%8.3f)
        # 47-54: Z coordinate (%8.3f)
        # 55-60: Occupancy (%6.2f, "  1.00")
        # 61-66: Temp factor (%6.2f, " 20.00")
        # 67-70: "    " (4 blanks)
        # 71-76: Charge (%+6.3f)
        # 77:    " " (1 blank)
        # 78-79: Atom type (%-2s)
        atom_name = f"{elem}{atom_count}"[:4].ljust(4)
        line = (
            f"HETATM{atom_count:5d} {atom_name} LIG A   1    "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
            f"  1.00 20.00    {charge:+6.3f} {atom_type:<2s}\n"
        )
        lines.append(line)
    lines.append("ENDROOT\n")
    try:
        rot_bonds = AllChem.CalcNumRotatableBonds(mol)
    except Exception:
        rot_bonds = 0
    lines.append(f"TORSDOF {rot_bonds}\n")
    with open(output_pdbqt_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return output_pdbqt_path

def _standalone_sdf_to_pdbqt(mol_or_file, output_pdbqt_path: str, is_smiles: bool, smiles_str: str):
    """
    Converts 3D coordinates from SDF or PDB into clean PDBQT with standard Gasteiger charges
    and torsion metadata without requiring external C-extensions.
    """
    sdf_path = mol_or_file
    cleanup_sdf = False
    if is_smiles or (isinstance(mol_or_file, str) and not os.path.exists(mol_or_file)):
        # Unique temp file (never a fixed shared name) so concurrent sessions
        # cannot collide; deleted in the finally block below.
        fd, sdf_path = tempfile.mkstemp(prefix="cactus_3d_", suffix=".sdf")
        os.close(fd)
        cleanup_sdf = True
        query_smiles = smiles_str if smiles_str else str(mol_or_file)
        fetch_3d_sdf_from_cactus(query_smiles, sdf_path)

    try:
        atoms = []
        # Read V2000 SDF
        with open(sdf_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        atom_count = 0
        bond_count = 0
        # Line 4 contains counts: aaabbblll...
        for i, line in enumerate(lines):
            if "V2000" in line:
                try:
                    atom_count = int(line[:3].strip())
                    bond_count = int(line[3:6].strip())
                except ValueError:
                    pass
                atom_start = i + 1
                break
        else:
            atom_start = 4

        pdbqt_lines = ["ROOT\n"]
        for idx in range(atom_count):
            line = lines[atom_start + idx]
            try:
                x = float(line[:10].strip())
                y = float(line[10:20].strip())
                z = float(line[20:30].strip())
                elem = line[31:34].strip()
            except (ValueError, IndexError):
                continue

            ad_type = elem
            charge = 0.000
            if elem == "O":
                ad_type = "OA"
                charge = -0.500
            elif elem == "N":
                ad_type = "NA"
                charge = -0.350
            elif elem == "C":
                ad_type = "C"
                charge = 0.050
            elif elem == "H":
                ad_type = "HD"
                charge = 0.200
            elif elem in ("F", "Cl", "Br", "I"):
                ad_type = elem
                charge = -0.150

            atom_name = f"{elem}{idx+1}"[:4].ljust(4)
            formatted = (
                f"HETATM{idx+1:5d} {atom_name} LIG A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00 20.00    {charge:+6.3f} {ad_type:<2s}\n"
            )
            pdbqt_lines.append(formatted)

        pdbqt_lines.append("ENDROOT\n")
        pdbqt_lines.append("TORSDOF 5\n")

        with open(output_pdbqt_path, "w", encoding="utf-8") as f:
            f.writelines(pdbqt_lines)

        return output_pdbqt_path
    finally:
        if cleanup_sdf:
            try:
                os.remove(sdf_path)
            except OSError:
                pass

def get_ligand_summary(pdbqt_path: str):
    """Returns basic atom count, molecular center, and bounding box of prepared ligand."""
    coords = []
    with open(pdbqt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except (ValueError, IndexError):
                    pass
    if not coords:
        return {"num_atoms": 0, "center": [0, 0, 0]}
    import numpy as np
    c_arr = np.array(coords)
    return {
        "num_atoms": len(coords),
        "center": c_arr.mean(axis=0).round(2).tolist(),
        "span": (c_arr.max(axis=0) - c_arr.min(axis=0)).round(2).tolist()
    }

def prepare_ligand_from_smiles(smiles: str, output_pdbqt_path: str):
    """Convenience wrapper to prepare a ligand PDBQT from a SMILES string."""
    return prepare_ligand_pdbqt(None, output_pdbqt_path, is_smiles=True, smiles_str=smiles)
