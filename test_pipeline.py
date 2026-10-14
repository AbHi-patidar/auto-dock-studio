"""
End-to-End Pipeline Verification Test for AutoDock Studio
Tests protein cleaning, ligand preparation, Vina docking, and interaction profiling.
"""

import os
import sys

# Setup paths
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_dir)

from utils.protein_prep import fetch_pdb_from_rcsb, clean_protein_pdb, receptor_pdb_to_pdbqt
from utils.ligand_prep import prepare_ligand_pdbqt
from utils.docking_runner import run_docking, create_complex_pdb
from utils.interaction_profiler import profile_interactions, generate_pymol_script

data_dir = os.path.join(curr_dir, "data")
os.makedirs(data_dir, exist_ok=True)

print("--- 1. Testing Protein Preparation ---")
raw_pdb = os.path.join(data_dir, "1hsg_raw.pdb")
if not os.path.exists(raw_pdb):
    fetch_pdb_from_rcsb("1HSG", raw_pdb)
print(f"Downloaded 1HSG PDB: {os.path.getsize(raw_pdb)} bytes")

clean_pdb = os.path.join(data_dir, "1hsg_clean.pdb")
meta = clean_protein_pdb(raw_pdb, clean_pdb)
print(f"Cleaned protein: {meta['num_atoms']} heavy atoms, center: {meta['center']}")
print(f"Detected ligand pockets: {list(meta['detected_ligands'].keys())}")

receptor_pdbqt = os.path.join(data_dir, "1hsg_receptor.pdbqt")
receptor_pdb_to_pdbqt(clean_pdb, receptor_pdbqt)
print(f"Receptor PDBQT generated: {os.path.getsize(receptor_pdbqt)} bytes")

print("\n--- 2. Testing Ligand Preparation ---")
# Indinavir simplified SMILES
smiles = "CC(C)(C)NC(=O)C1CN(CCN1CC(CC2=CC=CC=C2)C(=O)NC(CC3=CC=CC=C3)C(=O)O)CC4=CN=CC=C4"
ligand_pdbqt = os.path.join(data_dir, "indinavir.pdbqt")
prepare_ligand_pdbqt(None, ligand_pdbqt, is_smiles=True, smiles_str=smiles)
print(f"Ligand PDBQT generated: {os.path.getsize(ligand_pdbqt)} bytes")

print("\n--- 3. Testing AutoDock Vina Execution ---")
out_pdbqt = os.path.join(data_dir, "docked_test.pdbqt")
# HIV-1 protease active site pocket coordinates
center = meta["detected_ligands"]["MK1"]["center"] if "MK1" in meta["detected_ligands"] else [16.0, 25.0, 4.0]
print(f"Running docking around binding pocket: {center}")

result = run_docking(
    receptor_pdbqt=receptor_pdbqt,
    ligand_pdbqt=ligand_pdbqt,
    output_pdbqt=out_pdbqt,
    center_x=center[0],
    center_y=center[1],
    center_z=center[2],
    size_x=20.0,
    size_y=20.0,
    size_z=20.0,
    exhaustiveness=1,  # ultra fast for verification
    num_modes=2
)

print(f"Docking finished! Found {len(result['poses'])} pose(s)")
for pose in result["poses"]:
    print(f"  Mode {pose['mode']}: Affinity = {pose['affinity']} kcal/mol, RMSD = {pose['rmsd_lb']}")

print("\n--- 4. Testing Interaction Profiler (PyMOL Replacement) ---")
best_pose = result["poses"][0]
interactions = profile_interactions(clean_pdb, best_pose["pdb_lines"])
print(f"Detected {len(interactions['h_bonds'])} Hydrogen Bonds:")
for hb in interactions["h_bonds"]:
    print(f"  - {hb['residue']} ({hb['protein_atom']} ... {hb['ligand_atom']}) : {hb['distance']} A")

print(f"Detected {len(interactions['hydrophobic'])} Hydrophobic Contacts:")
for hc in interactions["hydrophobic"][:3]:
    print(f"  - {hc['residue']} ({hc['protein_atom']} ... {hc['ligand_atom']}) : {hc['distance']} A")

complex_pdb = os.path.join(data_dir, "complex_test.pdb")
create_complex_pdb(clean_pdb, best_pose["pdb_lines"], complex_pdb)
pml_path = os.path.join(data_dir, "test_session.pml")
generate_pymol_script(complex_pdb, interactions["pocket_residues"], pml_path)
print(f"Generated merged complex: {complex_pdb}")
print(f"Generated PyMOL script: {pml_path}")
print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")
