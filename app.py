"""
AutoDock Studio: Autonomous Molecular Docking, Biological Interpretation & Publication Platform
A unified, zero-install computational drug discovery platform that automates
protein & ligand preparation, AutoDock Vina execution, 2D/3D interaction profiling,
automated scientific interpretation, and 300 DPI publication figure generation.
"""

import os
import sys
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Add local utils to path
sys.path.append(os.path.dirname(__file__))
from utils.protein_prep import fetch_pdb_from_rcsb, clean_protein_pdb, receptor_pdb_to_pdbqt, compute_blind_docking_box
from utils.ligand_prep import prepare_ligand_pdbqt, get_ligand_summary
try:
    from utils.ligand_prep import prepare_ligand_from_smiles
except ImportError:
    def prepare_ligand_from_smiles(smiles: str, output_pdbqt_path: str):
        return prepare_ligand_pdbqt(None, output_pdbqt_path, is_smiles=True, smiles_str=smiles)
from utils.docking_runner import run_docking, create_complex_pdb, parse_vina_output
from utils.interaction_profiler import profile_interactions, generate_pymol_script
from utils.advanced_scenarios import get_scenario_panel_data, append_advanced_scenario_to_manuscript, SPECIAL_SCENARIO_PROTOCOLS
from utils.interaction_2d import calculate_drug_likeness, generate_2d_interaction_diagram
from utils.interpretation import generate_scientific_interpretation, calculate_inhibition_constant, calculate_ligand_efficiency
from utils.export_bundle import create_publication_zip
from utils.viewer_3d import generate_3dmol_html

# Advanced Academic Discovery Modules
from utils.admet_profiler import profile_extended_admet, format_admet_summary_markdown
from utils.validation import extract_native_ligand_pdb, calculate_heavy_atom_rmsd, evaluate_validation_tier
from utils.batch_docking import parse_batch_input, run_batch_virtual_screening
from utils.energy_decomposition import compute_residue_decomposition, generate_decomposition_plot
from utils.md_generator import generate_openmm_script, generate_gromacs_mdp_files, create_md_package_zip

# Page Configuration
st.set_page_config(
    page_title="AutoDock Studio | Autonomous Molecular Docking & Publication Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Biotechnology CSS Design System (Minimal Academic Light Edition)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Lora:ital,wght@0,500;0,600;0,700;1,400;1,500&display=swap');

    /* Global Typography & Minimal Palette */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background-color: #f8fafc;
        color: #1e293b;
    }

    h1, h2, h3, .academic-heading {
        font-family: 'Lora', Georgia, 'Times New Roman', serif;
        color: #0f172a;
        letter-spacing: -0.2px;
    }

    code, pre, .mono-val {
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
        background-color: #f1f5f9;
        color: #0f172a;
        border-radius: 4px;
        padding: 2px 5px;
    }

    /* Minimal Academic Masthead */
    .academic-masthead {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: 3px solid #1d4ed8;
        border-radius: 8px;
        padding: 22px 28px 18px 28px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .masthead-pretitle {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #64748b;
        margin-bottom: 6px;
    }
    .masthead-title {
        font-family: 'Lora', Georgia, serif;
        color: #0f172a;
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.3px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
    }
    .academic-edition-tag {
        font-family: 'Inter', sans-serif;
        font-size: 11px;
        font-weight: 600;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1d4ed8;
        padding: 3px 10px;
        border-radius: 4px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .masthead-desc {
        color: #475569;
        font-size: 13.5px;
        line-height: 1.6;
        max-width: 980px;
        margin-bottom: 12px;
    }
    .masthead-meta-bar {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        border-top: 1px solid #f1f5f9;
        padding-top: 10px;
    }
    .meta-pill {
        font-size: 11.5px;
        color: #475569;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 3px 10px;
        border-radius: 4px;
    }
    .meta-pill strong {
        color: #0f172a;
    }

    /* Standard Operating Protocol Stepper Bar */
    .stepper-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 22px;
    }
    .step-item {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 10px 14px;
        display: flex;
        align-items: center;
        gap: 10px;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .step-item.active {
        border-color: #2563eb;
        border-left: 3px solid #2563eb;
        background: #eff6ff;
    }
    .step-item.done {
        border-color: #059669;
        border-left: 3px solid #059669;
        background: #f0fdf4;
    }
    .step-num {
        width: 22px;
        height: 22px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: 700;
        background: #f1f5f9;
        color: #64748b;
        font-family: 'JetBrains Mono', monospace;
    }
    .step-item.active .step-num {
        background: #2563eb;
        color: white;
    }
    .step-item.done .step-num {
        background: #059669;
        color: white;
    }
    .step-label {
        font-size: 12px;
        font-weight: 600;
        color: #0f172a;
    }
    .step-status {
        font-size: 10.5px;
        color: #64748b;
    }
    .step-item.done .step-status {
        color: #059669;
    }
    .step-item.active .step-status {
        color: #2563eb;
    }

    /* Scientific Instrument Metric Badges */
    .metric-badge {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 12px 14px;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }
    .metric-val {
        font-family: 'Lora', Georgia, serif;
        font-size: 21px;
        font-weight: 600;
        color: #0f172a;
        letter-spacing: -0.2px;
    }
    .metric-label {
        font-size: 10.5px;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.6px;
        margin-top: 4px;
    }

    /* Academic Drug-likeness Compliance Stamps */
    .druglike-pass {
        background: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 11.5px;
        font-weight: 600;
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
    }
    .druglike-warn {
        background: #fffbeb;
        color: #d97706;
        border: 1px solid #fde68a;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 11.5px;
        font-weight: 600;
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Manuscript Results & Discussion Sections */
    .interp-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .interp-title {
        font-family: 'Lora', Georgia, serif;
        color: #0f172a;
        font-weight: 600;
        font-size: 14.5px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .interp-body {
        color: #334155;
        font-size: 13px;
        line-height: 1.65;
    }

    /* Academic Preset Showcase Card */
    .preset-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 16px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }

    /* Authoritative Academic Action Buttons */
    div.stButton > button:first-child {
        background: #1d4ed8;
        color: #ffffff;
        border: 1px solid #1e40af;
        border-radius: 6px;
        padding: 9px 16px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.1px;
        transition: all 0.15s ease-in-out;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    div.stButton > button:first-child:hover {
        background: #1e40af;
        border-color: #1e3a8a;
    }
    div.stButton > button:first-child:active {
        background: #1e3a8a;
    }

    /* Clean Minimal Academic Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    /* Academic Dataframes & Tables */
    [data-testid="stDataFrame"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Preset Benchmark Demos
PRESETS = {
    "Select or Pick a Benchmark Demo": {
        "pdb": "", "smiles": "", "name": "Custom", "box": [0, 0, 0],
        "desc": "Custom docking simulation. Enter any PDB ID and ligand SMILES below."
    },
    "HIV-1 Protease (1HSG) + Indinavir": {
        "pdb": "1HSG",
        "smiles": "CC(C)(C)NC(=O)C1CN(CCN1CC(CC2=CC=CC=C2)C(=O)NC(CC3=CC=CC=C3)C(=O)O)CC4=CN=CC=C4",
        "name": "Indinavir",
        "box": [13.07, 22.47, 5.56],
        "desc": "FDA-approved HIV-1 protease inhibitor targeting the C2-symmetric active site (Asp25 catalytic dyad)."
    },
    "SARS-CoV-2 Mpro (6LU7) + Nirmatrelvir": {
        "pdb": "6LU7",
        "smiles": "CC1(C2C1C(N(C2)C(=O)C(C(C)(C)C)NC(=O)C(F)(F)F)C(=O)NC(CC3CCNC3=O)C#N)C",
        "name": "Nirmatrelvir",
        "box": [-10.5, 12.4, 68.8],
        "desc": "Main protease (Mpro / 3CLpro) of SARS-CoV-2 with Paxlovid active ingredient."
    },
    "EGFR Kinase Domain (1M17) + Erlotinib": {
        "pdb": "1M17",
        "smiles": "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
        "name": "Erlotinib",
        "box": [22.0, 0.3, 52.8],
        "desc": "Oncology target for non-small cell lung cancer; quinazoline scaffold binding the ATP pocket."
    },
    "Human Serum Albumin (1AO6) + Ibuprofen": {
        "pdb": "1AO6",
        "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "name": "Ibuprofen",
        "box": [30.0, 15.0, 22.0],
        "desc": "Plasma transport protein; model system for NSAID drug-protein interaction."
    }
}

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
os.makedirs(DATA_DIR, exist_ok=True)

# Academic Sidebar
with st.sidebar:
    st.markdown("""
    <div style="padding: 6px 0 14px 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 16px;">
        <div style="font-family: 'Lora', Georgia, serif; font-size: 19px; font-weight: 700; color: #0f172a; letter-spacing: -0.2px;">
            AutoDock Studio
        </div>
        <div style="font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 2px;">
            Structural Biophysics Laboratory
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🔬 Computational Bench")
    st.caption("**Docking Core:** AutoDock Vina v1.2.5 (Scripps Research)")
    st.caption("**Cheminformatics:** RDKit Release 2023.9+ / Meeko v0.5")
    st.caption("**Dielectric Constant (ε):** 78.5 (Solvent continuum)")
    st.caption("**Standard Temperature (T):** 298.15 K (25.0 °C)")

    st.markdown("---")
    st.markdown("#### 📐 Thermodynamic Formulations")
    st.markdown("""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; font-size: 11.5px; color: #1e293b; font-family: 'JetBrains Mono', monospace; line-height: 1.6; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
        <strong style="color:#0f172a;">ΔG = -RT ln(K_i)</strong><br>
        RT @ 298.15K = 0.592 kcal/mol<br>
        K_i = exp(ΔG / RT)<br>
        LE = |ΔG| / N_heavy
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📚 Methodological Citations")
    bibtex_vina = """@article{trott2010autodock,
  title={AutoDock Vina: improving speed & accuracy},
  author={Trott, Oleg and Olson, Arthur J},
  journal={J. Comput. Chem.},
  volume={31},
  pages={455--461},
  year={2010}
}"""
    st.code(bibtex_vina, language="bibtex")
    st.caption("AutoDock Vina empirical free energy scoring function.")

    st.markdown("---")
    st.markdown("#### 🛡️ Standard Operating Protocols")
    st.markdown("""
    <div style="font-size: 12px; color: #475569; line-height: 1.6;">
        • <strong>Receptors:</strong> Strip non-catalytic crystallographic waters and buffer salts.<br>
        • <strong>Cofactors:</strong> Retain catalytic Zn²⁺/Mg²⁺ and structural prosthetic groups.<br>
        • <strong>Ligands:</strong> Generate low-energy conformers via ETKDGv3 and minimize with MMFF94.<br>
        • <strong>Search Box:</strong> Ensure ≥4 Å buffer around anticipated binding pocket envelope.
    </div>
    """, unsafe_allow_html=True)

# Main Academic Masthead
st.markdown("""
<div class="academic-masthead">
    <div class="masthead-pretitle">STRUCTURAL BIOINFORMATICS & COMPUTATIONAL BIOPHYSICS LABORATORY</div>
    <div class="masthead-title">
        AUTODOCK STUDIO <span class="academic-edition-tag">Research Edition v2.0</span>
    </div>
    <div class="masthead-desc">
        A rigorous, peer-reviewed computational biophysics workbench for autonomous macromolecular docking,
        thermodynamic inhibition constant quantification, 2D/3D non-covalent interaction mapping,
        and publication-grade figure generation compliant with ACS, Nature, and Elsevier standards.
    </div>
    <div class="masthead-meta-bar">
        <span class="meta-pill"><strong>Core Engine:</strong> AutoDock Vina v1.2.5</span>
        <span class="meta-pill"><strong>Forcefield:</strong> Vina Empirical Free Energy + MMFF94</span>
        <span class="meta-pill"><strong>Conformational Sampling:</strong> ETKDGv3</span>
        <span class="meta-pill"><strong>Visual Graphics:</strong> 300 DPI Journal Grade</span>
        <span class="meta-pill"><strong>Citation Format:</strong> J. Med. Chem. / ACS</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Session States
if "protein_ready" not in st.session_state:
    st.session_state.protein_ready = False
if "ligand_ready" not in st.session_state:
    st.session_state.ligand_ready = False
if "docking_done" not in st.session_state:
    st.session_state.docking_done = False
if "selected_pose_idx" not in st.session_state:
    st.session_state.selected_pose_idx = 0
if "poses" not in st.session_state:
    st.session_state.poses = []
if "publication_mode" not in st.session_state:
    st.session_state.publication_mode = False
if "active_scenarios" not in st.session_state:
    st.session_state.active_scenarios = []

# Dynamic SOP Stepper Header
s1_class = "done" if st.session_state.protein_ready else "active"
s2_class = "done" if st.session_state.ligand_ready else ("active" if st.session_state.protein_ready else "")
s3_class = "done" if st.session_state.docking_done else ("active" if st.session_state.protein_ready and st.session_state.ligand_ready else "")
s4_class = "done" if st.session_state.docking_done else ""

st.markdown(f"""
<div class="stepper-container">
    <div class="step-item {s1_class}">
        <div class="step-num">I</div>
        <div>
            <div class="step-label">Receptor Topology</div>
            <div class="step-status">{'✓ Cleaned & PDBQT' if st.session_state.protein_ready else 'Awaiting Prep'}</div>
        </div>
    </div>
    <div class="step-item {s2_class}">
        <div class="step-num">II</div>
        <div>
            <div class="step-label">Ligand Parametrization</div>
            <div class="step-status">{'✓ 3D Conformer Minimized' if st.session_state.ligand_ready else 'Awaiting Prep'}</div>
        </div>
    </div>
    <div class="step-item {s3_class}">
        <div class="step-num">III</div>
        <div>
            <div class="step-label">Search Space (Grid)</div>
            <div class="step-status">{'✓ Pocket Centered' if st.session_state.protein_ready else 'Default Box'}</div>
        </div>
    </div>
    <div class="step-item {s4_class}">
        <div class="step-num">IV</div>
        <div>
            <div class="step-label">Docking & Energetics</div>
            <div class="step-status">{'✓ Poses & Report' if st.session_state.docking_done else 'Ready to Run'}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Preset Benchmark Selection Bar
preset_col1, preset_col2 = st.columns([7, 3])
with preset_col1:
    selected_preset_name = st.selectbox("Standard Reference Complexes & Validation Benchmarks", list(PRESETS.keys()))
    preset_data = PRESETS[selected_preset_name]
with preset_col2:
    st.write("")
    st.write("")
    is_preset_active = selected_preset_name != "Select or Pick a Benchmark Demo"
    run_1click = st.button("⚡ Execute Validation Benchmark in 1-Click", disabled=not is_preset_active, use_container_width=True, help="Executes the full pipeline: receptor download, ligand minimization, pocket auto-centering, and Vina docking.")

if is_preset_active:
    st.markdown(f"""
    <div class="preset-card">
        <strong style="color:#0f172a;">{preset_data['name']}</strong> &nbsp;|&nbsp; PDB Accession: <code style="color:#1d4ed8; font-weight:600;">{preset_data['pdb']}</code> &nbsp;|&nbsp; <span style="color:#475569; font-size:12px;">{preset_data['desc']}</span>
    </div>
    """, unsafe_allow_html=True)

# Helper function to run the full benchmark in 1 click
if run_1click and is_preset_active:
    with st.spinner(f"Running automated 1-click docking benchmark for {selected_preset_name}..."):
        try:
            # 1. Protein Prep
            raw_pdb_path = os.path.join(DATA_DIR, f"{preset_data['pdb']}_raw.pdb")
            clean_pdb_path = os.path.join(DATA_DIR, f"{preset_data['pdb']}_clean.pdb")
            receptor_pdbqt_path = os.path.join(DATA_DIR, f"{preset_data['pdb']}_receptor.pdbqt")

            fetch_pdb_from_rcsb(preset_data["pdb"], raw_pdb_path)
            prep_meta = clean_protein_pdb(raw_pdb_path, clean_pdb_path, preserve_metals=True, preserve_cofactors=True)
            receptor_pdb_to_pdbqt(clean_pdb_path, receptor_pdbqt_path)

            st.session_state.target_label = preset_data["pdb"]
            st.session_state.protein_meta = prep_meta
            st.session_state.clean_pdb_path = clean_pdb_path
            st.session_state.receptor_pdbqt_path = receptor_pdbqt_path
            st.session_state.protein_ready = True
            st.session_state.active_scenarios = get_scenario_panel_data(
                clean_pdb_path=clean_pdb_path,
                pdb_id=preset_data["pdb"],
                preserved_hetero=prep_meta.get("preserved_hetero", [])
            )

            # 2. Ligand Prep
            ligand_pdbqt_path = os.path.join(DATA_DIR, f"{preset_data['name']}.pdbqt")
            prepare_ligand_pdbqt(None, ligand_pdbqt_path, is_smiles=True, smiles_str=preset_data["smiles"])
            st.session_state.ligand_label = preset_data["name"]
            st.session_state.active_smiles = preset_data["smiles"]
            st.session_state.ligand_pdbqt_path = ligand_pdbqt_path
            st.session_state.ligand_summary = get_ligand_summary(ligand_pdbqt_path)
            st.session_state.drug_likeness = calculate_drug_likeness(preset_data["smiles"])
            st.session_state.ligand_ready = True

            # 3. Docking
            out_pdbqt = os.path.join(DATA_DIR, f"docked_{preset_data['pdb']}.pdbqt")
            box_c = preset_data["box"]
            dock_result = run_docking(
                receptor_pdbqt=receptor_pdbqt_path,
                ligand_pdbqt=ligand_pdbqt_path,
                output_pdbqt=out_pdbqt,
                center_x=box_c[0],
                center_y=box_c[1],
                center_z=box_c[2],
                size_x=22.0,
                size_y=22.0,
                size_z=22.0,
                exhaustiveness=8,
                num_modes=9
            )

            st.session_state.docked_pdbqt_path = out_pdbqt
            st.session_state.poses = dock_result["poses"]
            st.session_state.docking_done = True
            st.session_state.selected_pose_idx = 0
            st.success(f"Benchmark finished! Found {len(dock_result['poses'])} valid binding poses with top affinity ΔG = {dock_result['poses'][0]['affinity']} kcal/mol.")
            st.rerun()

        except Exception as e:
            st.error(f"1-Click Benchmark failed: {e}")

# Protocol Discovery Mode Selection
studio_mode = st.radio(
    "Select Computational Discovery Protocol",
    [
        "🎯 Macromolecular Docking & In-Depth Interaction Profiler",
        "⚡ High-Throughput Virtual Screening (HTVS & Batch Docking)"
    ],
    horizontal=True
)
st.markdown("---")

if studio_mode == "⚡ High-Throughput Virtual Screening (HTVS & Batch Docking)":
    st.subheader("⚡ High-Throughput Virtual Screening (HTVS) & Batch Docking Bench")
    st.caption("Autonomous screening of compound libraries against a macromolecular target. Calculates binding free energy (ΔG), estimated Ki (nM), Ligand Efficiency (LE), Lipophilic Ligand Efficiency (LLE), and filters Pan-Assay Interference (PAINS).")

    # Target Selection
    htvs_tcol1, htvs_tcol2 = st.columns([6, 4])
    with htvs_tcol1:
        if st.session_state.protein_ready:
            st.success(f"Active Target Receptor: **{st.session_state.get('target_label', 'Receptor')}**")
            current_target = st.session_state.get('target_label', '1HSG')
        else:
            current_target = st.selectbox("Select Target Complex for Screening", ["1HSG (HIV-1 Protease)", "6LU7 (SARS-CoV-2 Mpro)", "1M17 (EGFR Kinase)"])
            st.info("Receptor will be prepared automatically upon screening launch.")
    with htvs_tcol2:
        htvs_exh = st.slider("Screening Exhaustiveness", min_value=1, max_value=16, value=4, help="Exhaustiveness 4 is recommended for rapid screening of multiple compounds.")
        htvs_modes = st.slider("Top Modes per Molecule", min_value=1, max_value=5, value=2)

    st.markdown("#### Library Specification")
    lib_choice = st.radio("Screening Library Source", ["Curated Reference Drug Library", "Upload CSV / SDF File", "Paste SMILES List"], horizontal=True)

    CURATED_LIBRARIES = {
        "HIV-1 Protease Inhibitors (5 FDA Drugs)": [
            {"name": "Indinavir", "smiles": "CC(C)(C)NC(=O)C1CN(CCN1CC(CC2=CC=CC=C2)C(=O)NC(CC3=CC=CC=C3)C(=O)O)CC4=CN=CC=C4"},
            {"name": "Darunavir", "smiles": "CC(C)CN(CC(C(CC1=CC=CC=C1)NC(=O)OC2COC3C2OCCO3)O)S(=O)(=O)C4=CC=C(C=C4)N"},
            {"name": "Saquinavir", "smiles": "CC(C)(C)NC(=O)C1CC2CCCCC2CN1CC(C(CC3=CC=CC=C3)NC(=O)C(CC(=O)N)NC(=O)C4=NC5=CC=CC=C5C=C4)O"},
            {"name": "Ritonavir", "smiles": "CC(C)C1=NC(=CS1)CN(C)C(=O)NC(C(C)C)C(=O)NC(CC2=CC=CC=C2)CC(C(CC3=CC=CC=C3)NC(=O)OCC4=CN=CS4)O"},
            {"name": "Amprenavir", "smiles": "CC(C)CN(CC(C(CC1=CC=CC=C1)NC(=O)OC2CCOC2)O)S(=O)(=O)C3=CC=C(C=C3)N"}
        ],
        "EGFR Kinase Inhibitors (4 FDA Drugs)": [
            {"name": "Erlotinib", "smiles": "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC"},
            {"name": "Gefitinib", "smiles": "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"},
            {"name": "Osimertinib", "smiles": "CN1CCN(CC1)C2=CC(=C(C=C2)NC3=NC=CC(=N3)C4=CN(C5=CC=CC=C54)C)NC(=O)C=C"},
            {"name": "Afatinib", "smiles": "CN(C)CC=CC(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)Cl)F)OC4CCOC4"}
        ]
    }

    compounds_to_screen = []
    if lib_choice == "Curated Reference Drug Library":
        lib_name = st.selectbox("Select Curated Library", list(CURATED_LIBRARIES.keys()))
        compounds_to_screen = CURATED_LIBRARIES[lib_name]
        st.write(f"Loaded **{len(compounds_to_screen)}** compounds for screening:")
        st.dataframe(pd.DataFrame(compounds_to_screen), use_container_width=True)
    elif lib_choice == "Upload CSV / SDF File":
        up_file = st.file_uploader("Upload Compound Library (.csv, .tsv, .sdf)", type=["csv", "tsv", "sdf"])
        if up_file:
            content = up_file.getvalue().decode("utf-8", errors="ignore")
            compounds_to_screen = parse_batch_input(content, up_file.name)
            st.success(f"Parsed {len(compounds_to_screen)} compounds from {up_file.name}")
            st.dataframe(pd.DataFrame(compounds_to_screen).head(10), use_container_width=True)
    else:
        text_smiles = st.text_area("Paste SMILES (format: CompoundName,SMILES or one SMILES per line)", value="Aspirin,CC(=O)Oc1ccccc1C(=O)O\nParacetamol,CC(=O)Nc1ccc(O)cc1\nIbuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O")
        if text_smiles.strip():
            compounds_to_screen = parse_batch_input(text_smiles, "input.txt")
            st.write(f"Parsed {len(compounds_to_screen)} compounds.")

    if st.button("🚀 Launch Autonomous Batch Virtual Screening", use_container_width=True, disabled=not compounds_to_screen):
        prog_bar = st.progress(0.0)
        status_txt = st.empty()

        def batch_callback(current, total, msg):
            prog_bar.progress(float(current) / float(total))
            status_txt.text(msg)

        rec_path = st.session_state.get("receptor_pdbqt_path")
        grid_p = {"center_x": 13.07, "center_y": 22.47, "center_z": 5.56, "size_x": 22.0, "size_y": 22.0, "size_z": 22.0}

        if not rec_path or not os.path.exists(rec_path):
            pdb_key = "1HSG"
            if "6LU7" in str(current_target):
                pdb_key = "6LU7"
                grid_p = {"center_x": -10.5, "center_y": 12.4, "center_z": 68.8, "size_x": 22.0, "size_y": 22.0, "size_z": 22.0}
            elif "1M17" in str(current_target):
                pdb_key = "1M17"
                grid_p = {"center_x": 22.0, "center_y": 0.3, "center_z": 52.8, "size_x": 22.0, "size_y": 22.0, "size_z": 22.0}

            raw_p = os.path.join(DATA_DIR, f"{pdb_key}_raw.pdb")
            clean_p = os.path.join(DATA_DIR, f"{pdb_key}_clean.pdb")
            rec_path = os.path.join(DATA_DIR, f"{pdb_key}_receptor.pdbqt")
            fetch_pdb_from_rcsb(pdb_key, raw_p)
            clean_protein_pdb(raw_p, clean_p)
            receptor_pdb_to_pdbqt(clean_p, rec_path)
            st.session_state.receptor_pdbqt_path = rec_path
            st.session_state.clean_pdb_path = clean_p
            st.session_state.target_label = pdb_key
            st.session_state.protein_ready = True

        with st.spinner("Executing virtual screening across compound library..."):
            screening_df = run_batch_virtual_screening(
                compounds=compounds_to_screen,
                receptor_pdbqt_path=rec_path,
                grid_params=grid_p,
                exhaustiveness=htvs_exh,
                num_modes=htvs_modes,
                progress_callback=batch_callback
            )
            st.session_state.screening_results = screening_df
            status_txt.success("Batch Virtual Screening Completed!")

    if st.session_state.get("screening_results") is not None:
        sdf = st.session_state.screening_results
        st.markdown("### 🏆 Virtual Screening Hit Leaderboard")

        succ = sdf[sdf["Status"] == "Success"]
        if not succ.empty:
            best_hit = succ.iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Top Ranked Hit", best_hit["Compound"])
            with m2:
                st.metric("Best Affinity (ΔG)", f"{best_hit['Affinity_kcal']} kcal/mol")
            with m3:
                st.metric("Best Est. Ki", f"{best_hit['Est_Ki_nM']} nM")
            with m4:
                st.metric("Successful / Total", f"{len(succ)} / {len(sdf)}")

        disp_cols = [c for c in ["Rank", "Compound", "Affinity_kcal", "Est_Ki_nM", "LE", "LLE", "MW", "LogP", "Lipinski_Violations", "PAINS_Alerts", "Status"] if c in sdf.columns]
        st.dataframe(sdf[disp_cols], use_container_width=True)

        csv_bytes = sdf.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Virtual Screening Leaderboard (CSV)",
            csv_bytes,
            file_name="HTVS_Virtual_Screening_Leaderboard.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.stop()

# Main 2-Column Workflow Layout
col_left, col_right = st.columns([4, 6], gap="large")

with col_left:
    st.subheader("Protocol Phase I: Macromolecular Receptor Topology")
    prot_tab1, prot_tab2 = st.tabs(["RCSB PDB ID", "Upload .pdb File"])

    with prot_tab1:
        pdb_id_input = st.text_input("PDB ID", value=preset_data["pdb"], placeholder="e.g. 1HSG, 6LU7, 1M17, 1AO6")

    with prot_tab2:
        uploaded_pdb = st.file_uploader("Upload Receptor PDB", type=["pdb"])

    p_opt1, p_opt2 = st.columns(2)
    with p_opt1:
        preserve_metals = st.checkbox("Retain Metal Ions (Zn, Mg, Ca, Fe)", value=True, help="Preserves catalytic metal ions essential for metalloenzymes.")
    with p_opt2:
        preserve_cofactors = st.checkbox("Retain Co-factors (HEM, NAD, ATP)", value=True, help="Preserves essential structural co-factors and prosthetics.")

    if st.button("🔧 Prepare Receptor (Auto-clean & PDBQT)", use_container_width=True):
        with st.spinner("Downloading PDB, stripping solvent, adding polar hydrogens, computing charges..."):
            try:
                raw_pdb_path = os.path.join(DATA_DIR, "receptor_raw.pdb")
                if uploaded_pdb is not None:
                    with open(raw_pdb_path, "wb") as f:
                        f.write(uploaded_pdb.getbuffer())
                    target_label = uploaded_pdb.name
                elif pdb_id_input.strip():
                    fetch_pdb_from_rcsb(pdb_id_input.strip(), raw_pdb_path)
                    target_label = pdb_id_input.strip().upper()
                else:
                    st.error("Please enter a PDB ID or upload a .pdb file.")
                    st.stop()

                clean_pdb_path = os.path.join(DATA_DIR, "receptor_clean.pdb")
                prep_meta = clean_protein_pdb(
                    raw_pdb_path,
                    clean_pdb_path,
                    preserve_metals=preserve_metals,
                    preserve_cofactors=preserve_cofactors
                )

                receptor_pdbqt_path = os.path.join(DATA_DIR, "receptor.pdbqt")
                receptor_pdb_to_pdbqt(clean_pdb_path, receptor_pdbqt_path)

                st.session_state.target_label = target_label
                st.session_state.protein_meta = prep_meta
                st.session_state.clean_pdb_path = clean_pdb_path
                st.session_state.receptor_pdbqt_path = receptor_pdbqt_path
                st.session_state.protein_ready = True

                # Auto-detect special scenarios
                st.session_state.active_scenarios = get_scenario_panel_data(
                    clean_pdb_path=clean_pdb_path,
                    pdb_id=target_label,
                    preserved_hetero=prep_meta.get("preserved_hetero", [])
                )
                st.success(f"Receptor ready! {prep_meta['num_atoms']} heavy atoms prepared.")

            except Exception as e:
                st.error(f"Error preparing receptor: {e}")

    if st.session_state.protein_ready:
        meta = st.session_state.protein_meta
        st.info(f"Receptor Center of Mass: `{meta['center']}` | Span: `{meta['dimensions']}` Å")
        if meta.get("preserved_hetero"):
            st.caption(f"Preserved Co-factors: {', '.join(meta['preserved_hetero'])}")
        if meta.get("detected_ligands"):
            lig_names = list(meta["detected_ligands"].keys())
            st.success(f"Detected Pocket(s): **{', '.join(lig_names)}** (Coordinates auto-centered)")

    st.markdown("---")

    # Step 2: Ligand
    st.subheader("Protocol Phase II: Ligand Stereochemistry & Conformer Energy")
    lig_tab1, lig_tab2, lig_tab3 = st.tabs(["SMILES String", "Upload File (.sdf, .mol, .pdb)", "✏️ 2D Molecular Sketcher"])

    with lig_tab1:
        smiles_input = st.text_area("SMILES String", value=preset_data["smiles"], placeholder="e.g. CC(C)(C)NC(=O)... or CC(=O)Oc1ccccc1C(=O)O")

    with lig_tab2:
        uploaded_lig = st.file_uploader("Upload Ligand", type=["sdf", "mol", "pdb"])

    with lig_tab3:
        st.markdown("**Interactive 2D Chemical Structure Sketcher (JSME)**")
        st.caption("Draw new chemical scaffolds, modify substituents, or select a template below. Click 'Copy Drawn SMILES' to paste into the SMILES input tab.")
        jsme_html = """
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 12px; color: #334155; margin-bottom: 8px; display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
            <strong style="color: #0f172a;">Sample Scaffolds:</strong>
            <button onclick="loadSketcherTemplate('CC(=O)Oc1ccccc1C(=O)O')" style="background:#f1f5f9; border:1px solid #cbd5e1; border-radius:4px; padding:3px 8px; font-size:11px; cursor:pointer;">Aspirin</button>
            <button onclick="loadSketcherTemplate('CC(=O)Nc1ccc(O)cc1')" style="background:#f1f5f9; border:1px solid #cbd5e1; border-radius:4px; padding:3px 8px; font-size:11px; cursor:pointer;">Paracetamol</button>
            <button onclick="loadSketcherTemplate('CC(C)Cc1ccc(cc1)C(C)C(=O)O')" style="background:#f1f5f9; border:1px solid #cbd5e1; border-radius:4px; padding:3px 8px; font-size:11px; cursor:pointer;">Ibuprofen</button>
            <button onclick="loadSketcherTemplate('COCCOc1cc2c(cc1OCCOC)ncnc2Nc3cccc(c3)C#C')" style="background:#f1f5f9; border:1px solid #cbd5e1; border-radius:4px; padding:3px 8px; font-size:11px; cursor:pointer;">Erlotinib</button>
        </div>
        <script type="text/javascript" src="https://jsme-editor.github.io/dist/jsme/jsme.nocache.js"></script>
        <script>
        function jsmeOnLoad() {
            jsmeApplet = new JSME("jsme_container", "100%", "330px", {
                "options": "query,hydrogens,paste,zoom"
            });
        }
        function copyDrawnSmiles() {
            if (typeof jsmeApplet !== "undefined") {
                var smi = jsmeApplet.smiles();
                navigator.clipboard.writeText(smi);
                var alertEl = document.getElementById("jsme_alert");
                alertEl.innerText = "✓ Copied to clipboard: " + smi;
                alertEl.style.display = "inline-block";
            }
        }
        function loadSketcherTemplate(smi) {
            if (typeof jsmeApplet !== "undefined") {
                jsmeApplet.readGenericMolecularInput(smi);
            }
        }
        </script>
        <div id="jsme_container" style="border: 1px solid #cbd5e1; border-radius: 6px; background:#ffffff; min-height: 330px;"></div>
        <div style="margin-top: 10px; display: flex; align-items: center; gap: 10px;">
            <button onclick="copyDrawnSmiles()" style="background:#1d4ed8; color:#ffffff; border:none; padding:7px 14px; border-radius:5px; font-size:12px; font-weight:600; cursor:pointer;">
                📋 Copy Drawn SMILES to Clipboard
            </button>
            <span id="jsme_alert" style="display:none; font-family:'JetBrains Mono', monospace; font-size:11.5px; color:#059669; font-weight:600;"></span>
        </div>
        """
        components.html(jsme_html, height=430)

    if st.button("🧪 Prepare Ligand & Drug-Likeness", use_container_width=True):
        with st.spinner("Generating 3D conformer (ETKDGv3), minimizing energy (MMFF94)..."):
            try:
                ligand_pdbqt_path = os.path.join(DATA_DIR, "ligand.pdbqt")
                if uploaded_lig is not None:
                    temp_lig_path = os.path.join(DATA_DIR, uploaded_lig.name)
                    with open(temp_lig_path, "wb") as f:
                        f.write(uploaded_lig.getbuffer())
                    prepare_ligand_pdbqt(temp_lig_path, ligand_pdbqt_path)
                    ligand_label = uploaded_lig.name
                    active_smiles = smiles_input.strip()
                elif smiles_input.strip():
                    prepare_ligand_pdbqt(None, ligand_pdbqt_path, is_smiles=True, smiles_str=smiles_input.strip())
                    ligand_label = preset_data["name"] if preset_data["name"] != "Custom" else "Ligand_1"
                    active_smiles = smiles_input.strip()
                else:
                    st.error("Please enter a SMILES string or upload a ligand file.")
                    st.stop()

                st.session_state.ligand_label = ligand_label
                st.session_state.active_smiles = active_smiles
                st.session_state.ligand_pdbqt_path = ligand_pdbqt_path
                st.session_state.ligand_summary = get_ligand_summary(ligand_pdbqt_path)
                st.session_state.drug_likeness = calculate_drug_likeness(active_smiles) if active_smiles else {}
                st.session_state.ligand_ready = True
                st.success("Ligand 3D conformer minimized and converted to PDBQT!")

            except Exception as e:
                st.error(f"Error preparing ligand: {e}")

    # Drug-likeness Scorecard (Lipinski)
    if st.session_state.ligand_ready and st.session_state.get("drug_likeness"):
        dl = st.session_state.drug_likeness
        status_html = f'<span class="druglike-pass">✓ {dl["rule_summary"]}</span>' if dl["is_druglike"] else f'<span class="druglike-warn">⚠️ {dl["rule_summary"]}</span>'
        st.markdown(f"**Lipinski Rule of 5:** {status_html}", unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("MW", f"{dl['mw']} Da")
        with d2:
            st.metric("LogP", dl['logp'])
        with d3:
            st.metric("HBD / HBA", f"{dl['hbd']} / {dl['hba']}")
        with d4:
            st.metric("TPSA", f"{dl['tpsa']} Å²")

    st.markdown("---")

    # Step 3: Docking Grid Box
    st.subheader("Protocol Phase III: Orthosteric Search Space (Grid Box)")

    pocket_mode = st.radio(
        "Search Space Mode",
        ["Binding Pocket (Auto)", "Blind Docking (Entire Protein)", "Manual Coordinates"],
        horizontal=True
    )

    default_center = [0.0, 0.0, 0.0]
    default_size = [22.0, 22.0, 22.0]

    if pocket_mode == "Blind Docking (Entire Protein)" and st.session_state.protein_ready:
        blind_box = compute_blind_docking_box(st.session_state.clean_pdb_path)
        default_center = blind_box["center"]
        default_size = blind_box["size"]
    elif preset_data["box"] != [0, 0, 0]:
        default_center = preset_data["box"]
    elif st.session_state.protein_ready:
        meta = st.session_state.protein_meta
        if meta.get("detected_ligands"):
            first_lig = next(iter(meta["detected_ligands"].values()))
            default_center = first_lig["center"]
        else:
            default_center = meta["center"]

    c1, c2, c3 = st.columns(3)
    with c1:
        center_x = st.number_input("Center X (Å)", value=float(default_center[0]), step=1.0)
    with c2:
        center_y = st.number_input("Center Y (Å)", value=float(default_center[1]), step=1.0)
    with c3:
        center_z = st.number_input("Center Z (Å)", value=float(default_center[2]), step=1.0)

    s1, s2, s3 = st.columns(3)
    with s1:
        size_x = st.number_input("Size X (Å)", value=float(default_size[0]), min_value=10.0, max_value=90.0, step=2.0)
    with s2:
        size_y = st.number_input("Size Y (Å)", value=float(default_size[1]), min_value=10.0, max_value=90.0, step=2.0)
    with s3:
        size_z = st.number_input("Size Z (Å)", value=float(default_size[2]), min_value=10.0, max_value=90.0, step=2.0)

    p1, p2 = st.columns(2)
    with p1:
        exhaustiveness = st.slider("Exhaustiveness", min_value=1, max_value=32, value=8, help="Higher values increase sampling depth and run time.")
    with p2:
        num_modes = st.slider("Max Modes (Poses)", min_value=1, max_value=20, value=9)

    st.markdown("---")

    # Step 4: Run Autonomous Docking
    ready_to_dock = st.session_state.protein_ready and st.session_state.ligand_ready
    if st.button("🚀 Execute Autonomous Molecular Docking", use_container_width=True, disabled=not ready_to_dock):
        with st.spinner("Executing AutoDock Vina engine and computing binding affinities..."):
            try:
                out_pdbqt = os.path.join(DATA_DIR, "docked_output.pdbqt")
                dock_result = run_docking(
                    receptor_pdbqt=st.session_state.receptor_pdbqt_path,
                    ligand_pdbqt=st.session_state.ligand_pdbqt_path,
                    output_pdbqt=out_pdbqt,
                    center_x=center_x,
                    center_y=center_y,
                    center_z=center_z,
                    size_x=size_x,
                    size_y=size_y,
                    size_z=size_z,
                    exhaustiveness=exhaustiveness,
                    num_modes=num_modes
                )

                st.session_state.docked_pdbqt_path = out_pdbqt
                st.session_state.poses = dock_result["poses"]
                st.session_state.docking_done = True
                st.session_state.selected_pose_idx = 0
                st.success(f"Docking complete! Found {len(dock_result['poses'])} valid binding poses.")

            except Exception as e:
                st.error(f"Docking failed: {e}")

with col_right:
    st.subheader("Protocol Phase IV: 3D Biomolecular Studio & Visualization")

    # 3D View Controls & Publication Mode Toggle
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([3, 2, 2, 2])
    with ctrl_col1:
        st.session_state.publication_mode = st.toggle("📄 Publication Figure Mode (White)", value=st.session_state.publication_mode)
    with ctrl_col2:
        zoom_pocket = st.checkbox("Pocket Focus", value=True)
    with ctrl_col3:
        show_surface = st.checkbox("Receptor Surface", value=False)
    with ctrl_col4:
        show_labels = st.checkbox("Residue Labels", value=True)

    st.caption("✨ **Publication-Grade Graphics Active:** 300 DPI supersampling, hardware WebGL MSAA antialiasing, telephoto FOV, and TrueType typography enabled.")

    # Load 3D coordinates
    rec_pdb_str = ""
    if st.session_state.protein_ready and os.path.exists(st.session_state.clean_pdb_path):
        with open(st.session_state.clean_pdb_path, "r", encoding="utf-8", errors="ignore") as f:
            rec_pdb_str = f.read()

    active_lig_str = ""
    interactions_data = {}
    current_affinity = "N/A"
    current_rmsd = "0.0 Å"
    current_ki = "N/A"
    current_le = 0.0

    grid_box_def = {
        "center": [center_x, center_y, center_z],
        "size": [size_x, size_y, size_z]
    }

    if st.session_state.docking_done and st.session_state.poses:
        poses = st.session_state.poses
        pose_options = [f"Pose {p['mode']} (Affinity: {p['affinity']} kcal/mol, RMSD: {p['rmsd_lb']} Å)" for p in poses]
        selected_pose_str = st.selectbox("Select Docked Binding Mode", pose_options, index=st.session_state.selected_pose_idx)
        selected_mode_num = int(selected_pose_str.split()[1])
        st.session_state.selected_pose_idx = selected_mode_num - 1

        chosen_pose = poses[st.session_state.selected_pose_idx]
        current_affinity = f"{chosen_pose['affinity']} kcal/mol"
        current_rmsd = f"{chosen_pose['rmsd_lb']} Å"
        active_lig_str = "".join(chosen_pose["pdb_lines"])

        # Compute real-time non-covalent interactions (replaces PyMOL)
        interactions_data = profile_interactions(st.session_state.clean_pdb_path, chosen_pose["pdb_lines"])

        # Automated Scientific Interpretation Calculation
        heavy_atoms = st.session_state.get("ligand_summary", {}).get("num_atoms", 30)
        interpretation_data = generate_scientific_interpretation(
            target_name=st.session_state.get("target_label", "Receptor"),
            ligand_name=st.session_state.get("ligand_label", "Ligand"),
            best_pose=chosen_pose,
            all_poses=st.session_state.poses,
            interactions=interactions_data,
            drug_likeness=st.session_state.get("drug_likeness", {}),
            num_heavy_atoms=heavy_atoms
        )
        st.session_state.interpretation_data = interpretation_data

        # Enhanced Score Metric Badges with Ki & Ligand Efficiency
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-badge"><div class="metric-val">{current_affinity}</div><div class="metric-label">Binding Free Energy (ΔG)</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-badge"><div class="metric-val" style="color:#f59e0b;">{interpretation_data["ki_str"]}</div><div class="metric-label">Inhibition Constant (Ki)</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-badge"><div class="metric-val" style="color:#10b981;">{interpretation_data["ligand_efficiency"]} <span style="font-size:11px; font-weight:normal; color:#64748b;">kcal/mol/atom</span></div><div class="metric-label">Ligand Efficiency (LE)</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-badge"><div class="metric-val">{len(interactions_data.get("h_bonds", []))}</div><div class="metric-label">Polar H-Bonds Detected</div></div>', unsafe_allow_html=True)

    # Render interactive 3Dmol viewer
    viewer_html = generate_3dmol_html(
        receptor_pdb_content=rec_pdb_str,
        ligand_pdb_content=active_lig_str,
        grid_box=grid_box_def if not st.session_state.docking_done else None,
        interactions=interactions_data,
        publication_mode=st.session_state.publication_mode,
        show_surface=show_surface,
        show_labels=show_labels,
        zoom_pocket=zoom_pocket,
        height=540
    )
    components.html(viewer_html, height=560)

    # Empty State Guidance when docking has not been executed yet
    if not st.session_state.docking_done:
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 3px solid #1d4ed8; border-radius: 6px; padding: 20px 24px; margin-top: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
            <div style="color: #0f172a; font-family: 'Lora', serif; font-weight: 600; font-size: 15px; margin-bottom: 6px;">
                Standard Computational Biophysics Pipeline
            </div>
            <div style="color: #475569; font-size: 13px; line-height: 1.65;">
                1. <strong>Receptor Topology:</strong> Select a benchmark complex above or specify any 4-letter RCSB PDB ID to strip bulk solvent and preserve catalytic metals.<br>
                2. <strong>Ligand Conformer:</strong> Provide a valid SMILES string, sketch a structure in the 2D editor, or upload an SDF file for ETKDGv3 conformer generation.<br>
                3. <strong>Orthosteric Space:</strong> Define the grid box around known co-crystallized coordinates or utilize blind docking spanning the full macromolecule.<br>
                4. <strong>Iterated Local Search:</strong> Execute AutoDock Vina to compute binding modes, thermodynamic Ki, and generate 300 DPI figures.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Analysis & Publication Tabs
    if st.session_state.docking_done and interactions_data:
        st.markdown("### 📜 Peer-Reviewed Manuscript & Analytical Findings")

        tab_interp, tab_2d, tab_decomp, tab_admet, tab_interactions, tab_md, tab_validation, tab_adv, tab_export = st.tabs([
            "§ 1. Results & Discussion",
            "§ 2. 2D Interaction Map (300 DPI)",
            "§ 3. Per-Residue Energy Decomposition (PLIF)",
            "§ 4. Extended ADMET & PAINS Profiler",
            "§ 5. Non-Covalent Interactions Network",
            "§ 6. Molecular Dynamics (MD) Simulation Kit",
            "§ 7. Crystallographic Re-docking & RMSD",
            "§ 8. Target Protocol Advisor",
            "§ 9. Complete Manuscript Submission Kit"
        ])

        # Tab 1: Scientific Interpretation
        with tab_interp:
            interp = st.session_state.get("interpretation_data", {})
            if interp:
                st.markdown(f"""
                <div class="interp-card" style="border-left: 3px solid {interp['tier_color']};">
                    <div class="interp-title">§ 3.1 Thermodynamic Potency Category: {interp['affinity_tier']}</div>
                    <div class="interp-body">
                        The primary docked conformer exhibits a calculated standard free energy of binding of <strong>ΔG = {interp['affinity_kcal']} kcal·mol⁻¹</strong>, 
                        yielding a predicted thermodynamic inhibition constant of <strong>K_i = {interp['ki_str']}</strong> (at T = 298.15 K). 
                        This binding profile {interp['affinity_tier_desc']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="interp-card">
                    <div class="interp-title">§ 3.2 Biophysical Mechanism & Pharmacophore Stabilization</div>
                    <div class="interp-body">
                        • <strong>Electrostatic Anchor:</strong> {interp['hbond_summary']}<br>
                        • <strong>Desolvation & Burial:</strong> {interp['hydro_summary']}<br>
                        • <strong>Cluster Convergence:</strong> {interp['pose_convergence']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="interp-card">
                    <div class="interp-title">§ 3.3 Pharmacokinetic Feasibility & In Vitro Recommendations</div>
                    <div class="interp-body">
                        • <strong>Lipinski Rule of 5:</strong> {interp['druglikeness_summary']}<br>
                        • <strong>Translational Directive:</strong> {'High confidence lead candidate: Recommend proceeding directly to in vitro enzymatic assays and biophysical surface plasmon resonance (SPR/ITC).' if abs(interp['affinity_kcal']) >= 7.0 else 'Early hit candidate: Recommend synthetic scaffold decoration to introduce directional polar contacts.'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Manuscript Draft with optional special scenarios
                active_scens = st.session_state.get("active_scenarios", [])
                enriched_manuscript = append_advanced_scenario_to_manuscript(interp["manuscript_draft"], active_scens)

                st.markdown("#### 📜 Manuscript Section: Results and Discussion")
                st.caption("Formatted according to ACS / Journal of Medicinal Chemistry guidelines. Ready for inclusion in submission text:")
                st.code(enriched_manuscript, language="markdown")

        # Tab 2: 2D Diagram
        fig_2d_png = os.path.join(DATA_DIR, f"Figure_2_2D_pose_{chosen_pose['mode']}.png")
        fig_2d_svg = os.path.join(DATA_DIR, f"Figure_2_2D_pose_{chosen_pose['mode']}.svg")
        try:
            generate_2d_interaction_diagram(
                smiles=st.session_state.active_smiles,
                interactions=interactions_data,
                output_png_path=fig_2d_png,
                output_svg_path=fig_2d_svg
            )
            has_2d_fig = True
        except Exception:
            has_2d_fig = False

        with tab_2d:
            st.write("**Figure 2: 2D Protein-Ligand Interaction Schematic (LigPlot style)**")
            st.caption("🔬 **Quality:** Rendered at 2400 × 1800 px (300 DPI) with TrueType typography, pocket residue callouts, and infinite-resolution vector SVG.")
            if has_2d_fig and os.path.exists(fig_2d_png):
                st.image(fig_2d_png, use_container_width=True)
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    with open(fig_2d_png, "rb") as f:
                        st.download_button("📥 Download 2D Figure (300 DPI PNG)", f.read(), file_name=f"Figure_2D_Pose_{chosen_pose['mode']}.png", mime="image/png", use_container_width=True)
                with col_d2:
                    with open(fig_2d_svg, "r", encoding="utf-8") as f:
                        st.download_button("📥 Download 2D Figure (Vector SVG)", f.read(), file_name=f"Figure_2D_Pose_{chosen_pose['mode']}.svg", mime="image/svg+xml", use_container_width=True)
            else:
                st.info("2D diagram generation requires an active ligand SMILES string.")

        # Tab 3: Per-Residue Energy Decomposition
        with tab_decomp:
            st.markdown("#### § 3. Per-Residue Binding Free Energy Decomposition & PLIF")
            st.caption("Decomposes the overall binding affinity into individual Lennard-Jones (Van der Waals) and Coulombic electrostatic/H-bond contributions for each active-site residue.")
            
            with st.spinner("Computing per-residue Lennard-Jones & Coulombic energy decomposition..."):
                try:
                    with open(st.session_state.clean_pdb_path, "r", encoding="utf-8", errors="ignore") as f:
                        prot_lines = f.readlines()
                    decomp_df = compute_residue_decomposition(prot_lines, chosen_pose["pdb_lines"], cutoff_distance=5.5)
                    
                    if not decomp_df.empty:
                        plot_fig = generate_decomposition_plot(decomp_df, top_n=15)
                        if plot_fig is not None:
                            st.plotly_chart(plot_fig, use_container_width=True)
                        
                        st.markdown("##### Active Site Residue Energetics Breakdown")
                        disp_df = decomp_df[["Residue", "Total_Interaction_kcal", "vdW_Energy_kcal", "Elec_Energy_kcal", "HBonds", "Hydrophobic", "Min_Distance_A"]].copy()
                        disp_df.columns = ["Residue", "Total ΔG (kcal/mol)", "vdW / Steric (kcal/mol)", "Electrostatic (kcal/mol)", "H-Bonds", "Hydrophobic Contacts", "Min Distance (Å)"]
                        st.dataframe(disp_df, use_container_width=True)

                        csv_data = decomp_df.to_csv(index=False)
                        st.download_button("📥 Download Energy Decomposition Data (CSV)", csv_data, file_name="Per_Residue_Energy_Decomposition.csv", mime="text/csv", use_container_width=True)
                    else:
                        st.info("No active site residues identified within 5.5 Å cutoff.")
                except Exception as e:
                    st.error(f"Decomposition calculation failed: {e}")

        # Tab 4: Extended ADMET Profiler
        with tab_admet:
            st.markdown("#### § 4. Extended ADMET, SwissADME Rules & PAINS Promiscuity Analysis")
            st.caption("Comprehensive computational pharmacology profiling: Physicochemical parameters, Lipinski/Veber/Ghose compliance, QED drug-likeness, Synthetic Accessibility (SAScore), and Pan-Assay Interference (PAINS) substructure alerts.")
            
            active_smi = st.session_state.get("active_smiles", "")
            if active_smi:
                admet = profile_extended_admet(active_smi)
                if admet:
                    a1, a2, a3, a4, a5 = st.columns(5)
                    with a1:
                        st.metric("Molecular Weight", f"{admet['mw']} Da")
                    with a2:
                        st.metric("LogP (Lipophilicity)", admet['logp'])
                    with a3:
                        st.metric("TPSA", f"{admet['tpsa']} Å²")
                    with a4:
                        st.metric("Rotatable Bonds", admet['rotb'])
                    with a5:
                        st.metric("Fraction Csp3", admet['fsp3'])

                    b1, b2, b3, b4 = st.columns(4)
                    with b1:
                        qed_val = admet['qed']
                        qed_color = "#059669" if qed_val >= 0.6 else ("#d97706" if qed_val >= 0.4 else "#dc2626")
                        st.markdown(f'<div class="metric-badge"><div class="metric-val" style="color:{qed_color};">{qed_val} / 1.0</div><div class="metric-label">QED Drug-Likeness</div></div>', unsafe_allow_html=True)
                    with b2:
                        sa_val = admet['sa_score']
                        sa_color = "#059669" if sa_val <= 4.0 else ("#d97706" if sa_val <= 6.5 else "#dc2626")
                        st.markdown(f'<div class="metric-badge"><div class="metric-val" style="color:{sa_color};">{sa_val} / 10</div><div class="metric-label">Synthetic Accessibility (SAScore)</div></div>', unsafe_allow_html=True)
                    with b3:
                        st.markdown(f'<div class="metric-badge"><div class="metric-val" style="font-size:16px; color:#2563eb;">{admet["pharmacokinetics"]["gastrointestinal_absorption"]}</div><div class="metric-label">Human GI Absorption (HIA)</div></div>', unsafe_allow_html=True)
                    with b4:
                        st.markdown(f'<div class="metric-badge"><div class="metric-val" style="font-size:16px; color:#7c3aed;">{admet["pharmacokinetics"]["blood_brain_barrier"]}</div><div class="metric-label">Blood-Brain Barrier (BBB)</div></div>', unsafe_allow_html=True)

                    st.markdown("---")
                    st.markdown("##### Filter Compliance & Promiscuity Assessment")
                    f1, f2, f3 = st.columns(3)
                    with f1:
                        st.markdown(f"**Lipinski Rule of 5:**")
                        lip_pass = admet['lipinski']['pass']
                        st.markdown(f'<span class="druglike-pass" style="width:100%; text-align:center;">✓ {admet["lipinski"]["summary"]}</span>' if lip_pass else f'<span class="druglike-warn" style="width:100%; text-align:center;">⚠️ {admet["lipinski"]["summary"]}</span>', unsafe_allow_html=True)
                    with f2:
                        st.markdown(f"**Veber Bioavailability Rule:**")
                        veb_pass = admet['veber']['pass']
                        st.markdown(f'<span class="druglike-pass" style="width:100%; text-align:center;">✓ {admet["veber"]["summary"]}</span>' if veb_pass else f'<span class="druglike-warn" style="width:100%; text-align:center;">⚠️ {admet["veber"]["summary"]}</span>', unsafe_allow_html=True)
                    with f3:
                        st.markdown(f"**Ghose Filter:**")
                        gh_pass = admet['ghose']['pass']
                        st.markdown(f'<span class="druglike-pass" style="width:100%; text-align:center;">✓ {admet["ghose"]["summary"]}</span>' if gh_pass else f'<span class="druglike-warn" style="width:100%; text-align:center;">⚠️ {admet["ghose"]["summary"]}</span>', unsafe_allow_html=True)

                    st.markdown("")
                    st.markdown("##### Pan-Assay Interference Compounds (PAINS) Alert Filter")
                    if admet["pains"]["has_alerts"]:
                        st.error(f"⚠️ **PAINS Alert Triggered:** Substructures detected: **{', '.join(admet['pains']['alerts'])}**. High risk of false-positive biochemical assay interference.")
                    else:
                        st.success("✅ **Clean Chemical Scaffold:** No reactive quinones, rhodanines, catechols, alkyl halides, or promiscuous PAINS alerts detected.")

                    st.markdown("---")
                    admet_md = format_admet_summary_markdown(admet)
                    st.code(admet_md, language="markdown")
            else:
                st.info("Extended ADMET profiling requires an active ligand SMILES string.")

        # Tab 5: Non-Covalent Interactions Network
        with tab_interactions:
            st.markdown("#### § 5. Comprehensive Non-Covalent Interactions Network")
            it_tab1, it_tab2, it_tab3, it_tab4 = st.tabs([
                "Polar Hydrogen Bonds",
                "Hydrophobic Contacts",
                "π-Stacking & Halogen Bonds",
                "Metal Coordination & Salt Bridges"
            ])

            with it_tab1:
                if interactions_data.get("h_bonds"):
                    hb_df = pd.DataFrame(interactions_data["h_bonds"])[["residue", "protein_atom", "ligand_atom", "distance"]]
                    hb_df.columns = ["Pocket Residue", "Protein Atom", "Ligand Atom", "Distance (Å)"]
                    st.dataframe(hb_df, use_container_width=True)
                else:
                    st.info("No polar hydrogen bonds detected within 3.5 Å cutoff.")

            with it_tab2:
                if interactions_data.get("hydrophobic"):
                    hc_df = pd.DataFrame(interactions_data["hydrophobic"])[["residue", "protein_atom", "ligand_atom", "distance"]]
                    hc_df.columns = ["Pocket Residue", "Protein Atom", "Ligand Atom", "Distance (Å)"]
                    st.dataframe(hc_df, use_container_width=True)
                else:
                    st.info("No non-polar hydrophobic contacts detected within 4.2 Å cutoff.")

            with it_tab3:
                pi_data = interactions_data.get("pi_stacking", [])
                if pi_data:
                    st.write("**Pi-Stacking and Aromatic Interactions** (centroid-to-centroid < 5.5 Å)")
                    pi_df = pd.DataFrame(pi_data)[["type", "residue", "protein_ring", "centroid_distance"]]
                    pi_df.columns = ["Interaction Type", "Residue", "Aromatic Ring", "Centroid Distance (Å)"]
                    st.dataframe(pi_df, use_container_width=True)
                else:
                    st.info("No pi-stacking detected.")

                halogen_data = interactions_data.get("halogen_bonds", [])
                if halogen_data:
                    st.write("**Halogen Bonds** (C-X...Acceptor < 3.8 Å)")
                    hx_df = pd.DataFrame(halogen_data)[["residue", "protein_atom", "ligand_atom", "halogen", "acceptor", "distance"]]
                    hx_df.columns = ["Residue", "Protein Atom", "Ligand Atom", "Halogen", "Acceptor", "Distance (Å)"]
                    st.dataframe(hx_df, use_container_width=True)
                else:
                    st.info("No halogen bonds detected.")

            with it_tab4:
                metal_data = interactions_data.get("metal_coordination", [])
                if metal_data:
                    st.success(f"**{len(metal_data)} Metal Coordination Bond(s) Detected**")
                    mc_df = pd.DataFrame(metal_data)[["metal", "residue", "protein_atom", "ligand_atom", "distance"]]
                    mc_df.columns = ["Metal Ion", "Residue", "Protein Atom", "Ligand Atom", "Distance (Å)"]
                    st.dataframe(mc_df, use_container_width=True)
                else:
                    st.info("No direct metal coordination bonds detected.")

                sb_data = interactions_data.get("salt_bridges", [])
                if sb_data:
                    st.write("**Salt Bridges & Ionic Contacts** (< 4.0 Å)")
                    sb_df = pd.DataFrame(sb_data)[["residue", "protein_atom", "ligand_atom", "distance"]]
                    sb_df.columns = ["Residue", "Protein Atom", "Ligand Atom", "Distance (Å)"]
                    st.dataframe(sb_df, use_container_width=True)

        # Tab 6: Molecular Dynamics Setup
        with tab_md:
            st.markdown("#### § 6. Molecular Dynamics (MD) Simulation Input Export")
            st.caption("Generate production-grade molecular dynamics simulation packages for OpenMM (GPU-accelerated Python), GROMACS, and AMBER to investigate binding pose stability, RMSD over time, and MM-PBSA thermodynamics.")
            
            md_col1, md_col2, md_col3 = st.columns(3)
            with md_col1:
                md_time = st.slider("Simulation Length (ns)", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
            with md_col2:
                md_temp = st.number_input("Equilibration Temperature (K)", value=300.0, step=5.0)
            with md_col3:
                md_ions = st.number_input("Ionic Strength (M NaCl)", value=0.15, step=0.05)

            openmm_code = generate_openmm_script(
                protein_pdb_name=st.session_state.get("target_label", "Receptor"),
                ligand_name=st.session_state.get("ligand_label", "Ligand"),
                sim_time_ns=md_time,
                temperature_k=md_temp,
                ionic_strength_molar=md_ions
            )
            gromacs_mdp = generate_gromacs_mdp_files()

            with open(st.session_state.clean_pdb_path, "r", encoding="utf-8", errors="ignore") as f:
                rec_lines = f.read()
            complex_str = rec_lines + "TER\n" + "".join(chosen_pose["pdb_lines"]) + "END\n"

            md_zip_bytes = create_md_package_zip(complex_str, openmm_code, gromacs_mdp)

            st.download_button(
                "📥 Download Complete MD Simulation Bundle (.ZIP)",
                md_zip_bytes,
                file_name=f"MD_Simulation_Bundle_{st.session_state.get('target_label', 'Target')}_{md_time:.0f}ns.zip",
                mime="application/zip",
                use_container_width=True
            )

            with st.expander("📄 View Generated OpenMM GPU Simulation Pipeline (run_openmm.py)"):
                st.code(openmm_code, language="python")

            with st.expander("📄 View GROMACS Production MDP Control File (npt.mdp)"):
                st.code(gromacs_mdp["npt.mdp"], language="ini")

        # Tab 7: Crystallographic Re-docking & RMSD Validation
        with tab_validation:
            st.markdown("#### § 7. Crystallographic Re-docking & RMSD Protocol Validation")
            st.caption("Peer-reviewed validation benchmark: Calculates heavy-atom Root Mean Square Deviation (RMSD) between the docked pose and experimental co-crystallized coordinates (Gold-standard criteria: RMSD <= 2.0 Å).")
            
            raw_path = os.path.join(DATA_DIR, f"{st.session_state.get('target_label', 'receptor')}_raw.pdb")
            if not os.path.exists(raw_path):
                raw_path = os.path.join(DATA_DIR, "receptor_raw.pdb")
                
            detected_ligands = st.session_state.get("protein_meta", {}).get("detected_ligands", {})
            default_lig_id = list(detected_ligands.keys())[0] if detected_ligands else "MK1"
            
            val_col1, val_col2 = st.columns([6, 4])
            with val_col1:
                crystal_lig_name = st.text_input("Native Co-crystallized Ligand Residue Name (PDB 3-letter code)", value=default_lig_id, help="e.g. MK1 for 1HSG, N3 for 6LU7, AQ4 for 1M17")
            with val_col2:
                st.write("")
                st.write("")
                run_rmsd_check = st.button("🔬 Compute Heavy-Atom RMSD", use_container_width=True)

            if run_rmsd_check or st.session_state.get("validation_result"):
                if os.path.exists(raw_path):
                    try:
                        native_pdb = os.path.join(DATA_DIR, f"native_{crystal_lig_name}.pdb")
                        extract_native_ligand_pdb(raw_path, crystal_lig_name, native_pdb)
                        with open(native_pdb, "r", encoding="utf-8", errors="ignore") as f:
                            crystal_lines = f.readlines()
                        
                        rmsd_val = calculate_heavy_atom_rmsd(crystal_lines, chosen_pose["pdb_lines"])
                        tier_info = evaluate_validation_tier(rmsd_val)
                        st.session_state.validation_result = tier_info

                        v1, v2 = st.columns([4, 6])
                        with v1:
                            st.markdown(f'<div class="metric-badge" style="border-top: 3px solid {tier_info["color"]};"><div class="metric-val" style="color:{tier_info["color"]};">{tier_info["rmsd"]} Å</div><div class="metric-label">{tier_info["tier"]}</div></div>', unsafe_allow_html=True)
                        with v2:
                            st.markdown(f"""
                            <div class="interp-card" style="border-left: 3px solid {tier_info['color']};">
                                <strong style="color:#0f172a;">Peer-Review Compliance:</strong><br>
                                <span style="font-size:12.5px; color:#334155;">{tier_info['desc']}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Could not compute RMSD for '{crystal_lig_name}': {e}. Ensure the raw PDB contains HETATM coordinates for this ligand code.")
                else:
                    st.info("Raw crystallographic PDB coordinates are required for RMSD validation.")

        with tab_adv:
            active_scenarios = st.session_state.get("active_scenarios", [])

            # Manual Scenario explorer option
            manual_scen_choice = st.selectbox(
                "Explore Protocol Recommendations by Protein Target Class",
                ["[Auto-Detect Only]"] + [f"{SPECIAL_SCENARIO_PROTOCOLS[k]['icon']} {SPECIAL_SCENARIO_PROTOCOLS[k]['label']}" for k in SPECIAL_SCENARIO_PROTOCOLS.keys()]
            )
            if manual_scen_choice != "[Auto-Detect Only]":
                selected_key = next((k for k in SPECIAL_SCENARIO_PROTOCOLS.keys() if SPECIAL_SCENARIO_PROTOCOLS[k]['label'] in manual_scen_choice), None)
                if selected_key:
                    active_scenarios = get_scenario_panel_data(
                        clean_pdb_path=st.session_state.get("clean_pdb_path"),
                        pdb_id=st.session_state.get("target_label", ""),
                        preserved_hetero=st.session_state.get("protein_meta", {}).get("preserved_hetero", []),
                        user_scenario=selected_key
                    )

            if not st.session_state.protein_ready:
                st.info("Prepare the receptor first to enable automatic advanced scenario detection.")
            elif not active_scenarios:
                st.success("✅ **Standard Globular Target:** No unusual docking constraints detected. AutoDock Vina scoring and Ki/LE metrics are fully applicable.")
            else:
                st.warning(f"⚠️ **{len(active_scenarios)} Special Scenario(s) Identified:** Review the expert guidance below before submitting results.")
                for sc in active_scenarios:
                    with st.expander(f"{sc['icon']} {sc['label']}", expanded=True):
                        st.markdown(f"<div style='background:#1e293b; border-left: 4px solid {sc['color']}; padding: 12px 16px; border-radius: 8px; margin-bottom: 8px;'><span style='color:{sc['color']}; font-weight:700;'>{sc['short']}</span></div>", unsafe_allow_html=True)
                        if sc.get("source") == "auto":
                            st.caption(f"Auto-detected from PDB: {sc.get('auto_reason', '')}")
                        elif sc.get("source") == "knowledge_base":
                            st.caption(f"Knowledge base match: {sc.get('matched_target', '')} — {sc.get('kb_note', '')}")
                        st.markdown("**Protocol Recommendations:**")
                        for i, step in enumerate(sc["protocol"], 1):
                            st.markdown(f"{i}. {step}")

            st.markdown("---")
            st.markdown("#### Protein Class Compatibility Guide")
            compat_df = pd.DataFrame({
                "Target Class": ["Enzyme (globular)", "Receptor Tyrosine Kinase", "GPCR (membrane)", "Nuclear Receptor", "Metalloenzyme (Zn/Mg)", "Serine/Cys Protease (covalent)", "Protein-Protein Interface", "Intrinsically Disordered", "RNA-Binding Protein", "Allosteric Site"],
                "Vina Reliability": ["Excellent", "Very Good", "Good (remove detergents)", "Very Good", "Moderate (metal approx)", "Encounter complex only", "Flat pocket - underestimated", "Qualitative only", "Good for groove binders", "Site-specific box needed"],
                "Ki Interpretation": ["Quantitative", "Quantitative", "Semi-quantitative", "Quantitative", "Semi-quantitative", "Non-covalent only", "Relative ranking", "Qualitative", "Semi-quantitative", "Relative ranking"],
                "Special Action": ["None", "Use ATP-site box", "TM bundle box", "Use LBD structure", "Verify metal preserved", "Check warhead geometry", "Include both chains", "Use induced-fit PDB", "Remove RNA chains", "Set allosteric box"]
            })
            st.dataframe(compat_df, use_container_width=True, hide_index=True)

        with tab_export:
            st.write("#### 📦 One-Click Publication Submission Package")
            st.write("Generates a zipped package containing 300 DPI figures, vector SVGs, data tables, scientific interpretation, PDB complex, PyMOL PML script, and manuscript text.")

            pml_path = os.path.join(DATA_DIR, "session.pml")
            complex_pdb_path = os.path.join(DATA_DIR, f"complex_pose_{chosen_pose['mode']}.pdb")
            create_complex_pdb(st.session_state.clean_pdb_path, chosen_pose["pdb_lines"], complex_pdb_path)
            generate_pymol_script(complex_pdb_path, interactions_data["pocket_residues"], pml_path)

            interp_data = st.session_state.get("interpretation_data", {})
            if interp_data:
                interp_data["manuscript_draft"] = append_advanced_scenario_to_manuscript(interp_data.get("manuscript_draft", ""), st.session_state.get("active_scenarios", []))

            zip_bytes = create_publication_zip(
                data_dir=DATA_DIR,
                target_name=st.session_state.get("target_label", "Receptor"),
                ligand_name=st.session_state.get("ligand_label", "Ligand"),
                docking_poses=st.session_state.poses,
                interactions=interactions_data,
                drug_likeness=st.session_state.get("drug_likeness", {}),
                clean_receptor_path=st.session_state.clean_pdb_path,
                docked_pdbqt_path=st.session_state.docked_pdbqt_path,
                figure_2d_png=fig_2d_png if has_2d_fig else None,
                figure_2d_svg=fig_2d_svg if has_2d_fig else None,
                pml_script_path=pml_path,
                chosen_pose_idx=st.session_state.selected_pose_idx,
                interpretation=interp_data
            )

            st.download_button(
                label="📥 Download Complete Publication Package (.ZIP)",
                data=zip_bytes,
                file_name=f"Publication_Package_{st.session_state.get('target_label', 'Receptor')}_{chosen_pose['mode']}.zip",
                mime="application/zip",
                use_container_width=True
            )

            st.markdown("---")
            st.write("##### Direct Individual Downloads:")
            d_col1, d_col2, d_col3 = st.columns(3)
            with d_col1:
                with open(complex_pdb_path, "r", encoding="utf-8") as f:
                    st.download_button("📄 Merged Complex (.pdb)", f.read(), file_name=f"Complex_Pose_{chosen_pose['mode']}.pdb", mime="chemical/x-pdb", use_container_width=True)
            with d_col2:
                with open(st.session_state.docked_pdbqt_path, "r", encoding="utf-8") as f:
                    st.download_button("🧪 Docked Poses (.pdbqt)", f.read(), file_name="Docked_Poses.pdbqt", mime="text/plain", use_container_width=True)
            with d_col3:
                with open(pml_path, "r", encoding="utf-8") as f:
                    st.download_button("🔮 PyMOL Session Script (.pml)", f.read(), file_name="Publication_Session.pml", mime="text/plain", use_container_width=True)
