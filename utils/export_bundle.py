"""
Publication Bundle Export Module for AutoDock Studio
Packages all 3D/2D publication figures, tables, coordinate files,
and manuscript-ready methods into a single ZIP archive.
"""

import os
import zipfile
import io
import string
import pandas as pd

def create_publication_zip(
    data_dir: str,
    target_name: str,
    ligand_name: str,
    docking_poses: list,
    interactions: dict,
    drug_likeness: dict,
    clean_receptor_path: str,
    docked_pdbqt_path: str,
    figure_2d_png: str,
    figure_2d_svg: str,
    pml_script_path: str,
    chosen_pose_idx: int = 0,
    interpretation: dict = None
) -> bytes:
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Coordinate Files
        if os.path.exists(clean_receptor_path):
            zf.write(clean_receptor_path, arcname="Receptor_Clean.pdb")
        if os.path.exists(docked_pdbqt_path):
            zf.write(docked_pdbqt_path, arcname="Docked_Poses.pdbqt")

        # Validate/clamp the requested pose index before indexing docking_poses
        if docking_poses:
            chosen_pose_idx = max(0, min(int(chosen_pose_idx), len(docking_poses) - 1))
            chosen_pose = docking_poses[chosen_pose_idx]
        else:
            chosen_pose = None
        if chosen_pose and os.path.exists(clean_receptor_path):
            complex_path = os.path.join(data_dir, f"complex_pose_{chosen_pose['mode']}.pdb")
            with open(complex_path, "w", encoding="utf-8") as out:
                with open(clean_receptor_path, "r", encoding="utf-8", errors="ignore") as rec:
                    for line in rec:
                        if not line.startswith("END"):
                            out.write(line)
                out.write("TER\n")
                for line in chosen_pose["pdb_lines"]:
                    out.write(line)
                out.write("END\n")
            zf.write(complex_path, arcname=f"Complex_Pose_{chosen_pose['mode']}.pdb")

        # 2. PyMOL Script
        if os.path.exists(pml_script_path):
            zf.write(pml_script_path, arcname="Publication_PyMOL_Session.pml")

        # 3. 2D Figures (PNG 300 DPI + Vector SVG)
        if figure_2d_png and os.path.exists(figure_2d_png):
            zf.write(figure_2d_png, arcname="Figure_2_2D_Interaction_Map_300DPI.png")
        if figure_2d_svg and os.path.exists(figure_2d_svg):
            zf.write(figure_2d_svg, arcname="Figure_2_2D_Interaction_Map.svg")

        # 4. Tables (CSV)
        if docking_poses:
            poses_df = pd.DataFrame([
                {
                    "Mode (Rank)": p["mode"],
                    "Binding Affinity (kcal/mol)": p["affinity"],
                    "RMSD Lower Bound (Å)": p["rmsd_lb"],
                    "RMSD Upper Bound (Å)": p["rmsd_ub"]
                }
                for p in docking_poses
            ])
            zf.writestr("Table_1_Docking_Scores.csv", poses_df.to_csv(index=False))

        interaction_rows = []
        for hb in interactions.get("h_bonds", []):
            interaction_rows.append({
                "Interaction Type": "Hydrogen Bond",
                "Pocket Residue": hb.get("residue", ""),
                "Receptor Atom": hb.get("protein_atom", ""),
                "Ligand Atom": hb.get("ligand_atom", ""),
                "Distance (Å)": hb.get("distance", "")
            })
        for hc in interactions.get("hydrophobic", []):
            interaction_rows.append({
                "Interaction Type": "Hydrophobic Contact",
                "Pocket Residue": hc.get("residue", ""),
                "Receptor Atom": hc.get("protein_atom", ""),
                "Ligand Atom": hc.get("ligand_atom", ""),
                "Distance (Å)": hc.get("distance", "")
            })
        for sb in interactions.get("salt_bridges", []):
            interaction_rows.append({
                "Interaction Type": "Salt Bridge / Ionic",
                "Pocket Residue": sb.get("residue", ""),
                "Receptor Atom": sb.get("protein_atom", ""),
                "Ligand Atom": sb.get("ligand_atom", ""),
                "Distance (Å)": sb.get("distance", "")
            })
        for pi in interactions.get("pi_stacking", []):
            interaction_rows.append({
                "Interaction Type": pi.get("type", "Pi-Stacking"),
                "Pocket Residue": pi.get("residue", ""),
                "Receptor Atom": f"Ring ({pi.get('protein_ring', '')})",
                "Ligand Atom": "Aromatic Centroid",
                "Distance (Å)": pi.get("centroid_distance", "")
            })
        for mc in interactions.get("metal_coordination", []):
            interaction_rows.append({
                "Interaction Type": "Metal Coordination",
                "Pocket Residue": mc.get("residue", ""),
                "Receptor Atom": f"{mc.get('protein_atom', '')} ({mc.get('metal', '')})",
                "Ligand Atom": mc.get("ligand_atom", ""),
                "Distance (Å)": mc.get("distance", "")
            })
        for hx in interactions.get("halogen_bonds", []):
            interaction_rows.append({
                "Interaction Type": "Halogen Bond",
                "Pocket Residue": hx.get("residue", ""),
                "Receptor Atom": f"{hx.get('protein_atom', '')} ({hx.get('acceptor', '')})",
                "Ligand Atom": f"{hx.get('ligand_atom', '')} ({hx.get('halogen', '')})",
                "Distance (Å)": hx.get("distance", "")
            })

        if interaction_rows:
            inter_df = pd.DataFrame(interaction_rows)
            zf.writestr("Table_2_Detailed_Interactions.csv", inter_df.to_csv(index=False))

        if drug_likeness:
            prop_df = pd.DataFrame([
                {"Property": "Molecular Weight (MW)", "Value": f"{drug_likeness.get('mw')} Da", "Criterion": "≤ 500 Da"},
                {"Property": "Calculated LogP", "Value": str(drug_likeness.get('logp')), "Criterion": "≤ 5.0"},
                {"Property": "H-Bond Donors (HBD)", "Value": str(drug_likeness.get('hbd')), "Criterion": "≤ 5"},
                {"Property": "H-Bond Acceptors (HBA)", "Value": str(drug_likeness.get('hba')), "Criterion": "≤ 10"},
                {"Property": "Polar Surface Area (TPSA)", "Value": f"{drug_likeness.get('tpsa')} Å²", "Criterion": "≤ 140 Å²"},
                {"Property": "Rotatable Bonds", "Value": str(drug_likeness.get('rotb')), "Criterion": "≤ 10"},
                {"Property": "Lipinski Rule of 5 Status", "Value": drug_likeness.get('rule_summary'), "Criterion": "≤ 1 violation"}
            ])
            zf.writestr("Table_3_Drug_Likeness_Lipinski.csv", prop_df.to_csv(index=False))

        # 5. Scientific Interpretation & Manuscript Draft
        if interpretation:
            zf.writestr("Manuscript_Results_and_Discussion_Draft.md", interpretation.get("manuscript_draft", ""))

            interp_table = pd.DataFrame([
                {"Metric": "Binding Free Energy (ΔG)", "Value": f"{interpretation.get('affinity_kcal')} kcal/mol"},
                {"Metric": "Estimated Inhibition Constant (Ki)", "Value": interpretation.get('ki_str')},
                {"Metric": "Binding Affinity Category", "Value": interpretation.get('affinity_tier')},
                {"Metric": "Ligand Efficiency (LE)", "Value": f"{interpretation.get('ligand_efficiency')} kcal/mol/atom"},
                {"Metric": "Pose Convergence", "Value": interpretation.get('pose_convergence')},
                {"Metric": "Drug-likeness Assessment", "Value": interpretation.get('druglikeness_summary')}
            ])
            zf.writestr("Table_4_Thermodynamic_and_Inhibition_Metrics.csv", interp_table.to_csv(index=False))

        # 6. Detailed Experimental Methods & Academic Citations (for direct paste into manuscript)
        # string.Template is used instead of str.format() so that user-supplied
        # target/ligand names containing '{...}' cannot raise KeyError.
        methods_template = string.Template("""# Molecular Docking & Computational Methods

## 1. Receptor & Ligand Preparation
The 3D atomic coordinates of the target receptor **${target_name}** were processed by stripping non-catalytic crystallization solvent and buffer salts. Essential catalytic metal ions and structural co-factors were retained with formal charge assignment. Polar hydrogens and AutoDock AD4 atom types were assigned.

The small molecule ligand **${ligand_name}** was generated from canonical SMILES strings using RDKit's ETKDGv3 distance geometry algorithm, followed by energy minimization with the MMFF94 force field. Partial charges and rotatable bonds were assigned via Meeko to construct the PDBQT representation. Physicochemical drug-likeness was evaluated according to Lipinski's Rule of 5 criteria.

## 2. Molecular Docking Protocol
Flexible-ligand rigid-receptor docking simulations were performed using the AutoDock Vina engine. Search space coordinates and dimensions were centered on the binding cavity with comprehensive exhaustiveness. Binding modes were ranked by calculated binding free energy (ΔG in kcal/mol).

Theoretical inhibition constants (Ki) were derived according to the standard thermodynamic relationship:
Ki = exp(ΔG / (R * T))
where R = 1.9872e-3 kcal/(mol·K) and T = 298.15 K. Ligand Efficiency (LE) was evaluated as:
LE = |ΔG| / N_heavy

## 3. Non-Covalent Interaction Profiling
Protein-ligand complexes were profiled for non-covalent contacts:
- **Hydrogen Bonds:** Donor-acceptor distance <= 3.5 Å.
- **Hydrophobic Contacts:** Non-polar carbon-carbon contacts between 3.0 - 4.2 Å.
- **Salt Bridges:** Interatomic distances <= 4.0 Å between oppositely charged functional groups.
- **π-π Stacking:** Centroid-to-centroid distance <= 5.5 Å categorized into parallel and edge-to-face geometry.
- **Metal Coordination:** Coordination distance <= 3.0 Å between catalytic metals and heteroatoms.

## 4. Academic References & Software Citations
1. **AutoDock Vina:** Eberhardt, J., et al. (2021). AutoDock Vina 1.2.0: New Docking Methods, Expanded Force Field, and Python Bindings. J. Chem. Inf. Model., 61(8), 3891–3898.
2. **RDKit:** Landrum, G., et al. RDKit: Open-source cheminformatics. https://www.rdkit.org
3. **Meeko:** Forli Lab, The Scripps Research Institute. Meeko: Preparation of small molecules for AutoDock. https://github.com/forlilab/Meeko
4. **3Dmol.js:** Rego, N., & Koes, D. (2015). 3Dmol.js: molecular visualization without Java. Bioinformatics, 31(8), 1322–1324.
5. **Lipinski's Rule of Five:** Lipinski, C. A., et al. (2001). Experimental and computational approaches to estimate solubility and permeability in drug discovery. Adv. Drug Deliv. Rev., 46(1-3), 3-26.
""")
        zf.writestr("Manuscript_Methods_and_Citations.md", methods_template.substitute(target_name=target_name, ligand_name=ligand_name))

    zip_buffer.seek(0)
    return zip_buffer.getvalue()