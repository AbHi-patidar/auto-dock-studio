# 🧬 AutoDock Studio: Autonomous Molecular Docking & Publication Studio

**AutoDock Studio** is a complete, zero-install computational drug discovery and structural biology platform. It eliminates the manual bottlenecks of **AutoDockTools** / MGLTools and **PyMOL**, delivering **publication-ready figures and submission bundles** directly from your web browser.

---

## 🌟 Key Features

### 1. Autonomous Receptor Preparation
- **PDB Fetcher:** 1-click download from RCSB PDB or custom `.pdb` upload.
- **Automated Solvent & Salt Stripping:** Removes crystallographic waters (`HOH`), crystallization buffers (`SO4`, `PO4`, `PEG`), and cryoprotectants.
- **Catalytic Metal Ion Retention:** Keeps essential catalytic metals (`Zn²⁺`, `Mg²⁺`, `Fe²⁺/Fe³⁺`, `Ca²⁺`, `Mn²⁺`, `Cu²⁺`) and cofactors (`HEM`, `NAD`, `FAD`, `ATP`) with proper formal charges for accurate metalloenzyme docking.
- **Binding Pocket Identification:** Automatically detects co-crystallized small molecules and calculates active site coordinates.
- **PDBQT Generation:** Assigns polar hydrogens, Kollman/Gasteiger charges, and AutoDock atom types.
- **Special Scenario Auto-Detection:** Automatically scans structures for metalloenzyme centers, covalent reactive residues, disulfide bonds, and multi-chain protein-protein interfaces.

### 2. Autonomous Ligand Preparation & Drug-Likeness
- **SMILES to 3D:** Converts SMILES strings to 3D structures using RDKit's ETKDGv3 algorithm (with NIH Cactus 3D resolver fallback).
- **Energy Minimization:** MMFF94 force field optimization.
- **Lipinski's Rule of 5 Scorecard:** Real-time computation of Molecular Weight, LogP, H-Bond Donors, H-Bond Acceptors, TPSA, and Rotatable Bonds with drug-likeness verification badges.
- **PDBQT Conversion:** Assigns partial charges and rotatable bonds via Meeko.

### 3. Smart Search Space (Grid Box)
- **Active Pocket Mode:** Centers the box on the auto-detected co-crystallized ligand pocket.
- **Blind Docking Mode:** Automatically computes the bounding box encompassing the entire protein surface with padding.
- **Manual Mode:** Interactive coordinate sliders $(x, y, z)$ and dimensions $(\Delta x, \Delta y, \Delta z)$.

### 4. Docking Engine (AutoDock Vina 1.2.7)
- Multi-threaded headless execution with configurable exhaustiveness (1 to 32) and number of output binding modes.
- Instant extraction of binding affinities ($\Delta G$ in kcal/mol) and RMSD values.
- Real-time thermodynamic derivation of theoretical inhibition constants ($K_i$) with adaptive unit scaling (pM, nM, $\mu$M, mM) and Ligand Efficiency ($LE = |\Delta G| / N_{\text{heavy}}$).

### 5. Non-Covalent Interaction Profiling (PyMOL & LigPlot+ Replacement)
- **Hydrogen Bonds:** Donor-acceptor distance $\le 3.5\text{ \AA}$ with residue labels and distance callouts.
- **Hydrophobic Contacts:** Non-polar carbon-carbon contacts ($3.0\text{--}4.2\text{ \AA}$).
- **Salt Bridges & Ionic Contacts:** Electrostatic pairs ($\le 4.0\text{ \AA}$) between basic and acidic residues.
- **$\pi$-$\pi$ Stacking:** Centroid-to-centroid distance $\le 5.5\text{ \AA}$ classified into parallel and T-shaped geometry.
- **Catalytic Metal Coordination:** Coordination distance $\le 3.0\text{ \AA}$ between active site metal ions and ligand heteroatoms.
- **Halogen Bonds:** Halogen donor to nitrogen/oxygen/sulfur acceptor ($\le 3.8\text{ \AA}$).

### 6. Crystallographic Re-docking & RMSD Validation Engine
- **Self-Docking Protocol:** Extracts experimental co-crystallized coordinates from raw PDB files and performs validation re-docking.
- **Heavy-Atom RMSD Calculation:** Measures Cartesian deviation between crystal and docked conformations.
- **Peer-Reviewed Journal Standards:** Automatic compliance categorization ($\text{RMSD} \le 2.0\text{ \AA}$ gold-standard for Nature / J. Med. Chem.).

### 7. High-Throughput Virtual Screening (HTVS) & Batch Docking Bench
- **Library Screening:** Multi-compound virtual screening via CSV/TSV, multi-mol SDF, or SMILES lists.
- **Curated Drug Libraries:** Built-in libraries for HIV-1 Protease and EGFR Kinase inhibitors.
- **Cheminformatics Ranking:** Real-time leaderboard scoring binding affinity ($\Delta G$), estimated $K_i$, Ligand Efficiency ($LE$), and Lipophilic Ligand Efficiency ($LLE = \text{p}K_i - \text{LogP}$).

### 8. Extended ADMET, SwissADME Rules & PAINS Filter
- **Multi-Filter Drug-Likeness:** Evaluates Lipinski Rule of 5, Veber oral bioavailability criteria, and Ghose filter.
- **Lead-Likeness & Synthesis:** Computes QED (Quantitative Estimate of Drug-likeness) and Synthetic Accessibility score ($\text{SAScore}$).
- **Pan-Assay Interference (PAINS):** Substructure filtering for promiscuous false positives (quinones, rhodanines, catechols, Michael acceptors).
- **Pharmacokinetics:** In silico estimation of human intestinal absorption (HIA) and blood-brain barrier (BBB) permeability.

### 9. Per-Residue Energy Decomposition & PLIF Bar Charts
- **Thermodynamic Decomposition:** Decomposes overall binding free energy into per-residue Lennard-Jones (Van der Waals/steric) and Coulombic electrostatics / H-bond contributions.
- **Publication Charts:** Interactive Plotly stacked bar charts displaying protein-ligand interaction fingerprints (PLIF).

### 10. Interactive 2D Molecular Sketcher (JSME)
- **Direct In-Browser Drawing:** Embedded JavaScript Molecular Editor (JSME) allowing researchers to draw chemical scaffolds, modify functional groups, and export SMILES directly to docking in 1 click.

### 11. Molecular Dynamics (MD) Simulation Input Packager
- **OpenMM GPU Pipeline:** Standalone GPU-accelerated Python simulation pipeline (`run_openmm.py`) with AMBER14 force field, explicit TIP3P water box, neutralizing counterions, NVT/NPT equilibration, and production MD.
- **GROMACS & AMBER:** Pre-configured MDP parameters (`em.mdp`, `nvt.mdp`, `npt.mdp`, `run_gromacs.sh`) bundled in a 1-click downloadable ZIP package.

### 12. Publication-Ready Figure Studio & Complete Export Package
- **Minimalist Academic Theme:** Clean off-white paper canvas (`#F8FAFC`), crisp white cards, royal blue accents, and publication typography.
- **3D Interactive Viewer (`3Dmol.js`):**
  - **Publication Figure Mode:** Switches to pure white background (`#FFFFFF`), high-contrast ribbons, and clean typography.
  - **Interaction Lines & Badges:** Distinct colors for H-bonds, hydrophobic contacts, $\pi$-stacking, and metal coordination.
  - **300 DPI Snapshot Export:** 1-click button to download the exact 3D camera pose at publication print resolution.
- **2D Interaction Diagram (LigPlot / PoseView Style):**
  - Renders the flat 2D chemical structure with highlighted interacting atoms and residue callouts in **Vector SVG** and **300 DPI PNG**.
- **One-Click Publication Submission Package (.ZIP):**
  - Figures, tables, merged complex PDB, PyMOL PML script, and ACS-formatted manuscript text.

---

## 🚀 Quick Start

### 1. Launch with One Click
Double-click `run.bat` or run:
```bash
python -m streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### 2. Built-in Benchmark Demos (1-Click Execution)
Select from the top dropdown and click **"⚡ Execute Validation Benchmark in 1-Click"**:
- **HIV-1 Protease (`1HSG`) + Indinavir** (FDA-approved protease inhibitor)
- **SARS-CoV-2 Mpro (`6LU7`) + Nirmatrelvir** (Paxlovid active ingredient)
- **EGFR Kinase (`1M17`) + Erlotinib** (Cancer therapeutic)
- **Human Serum Albumin (`1AO6`) + Ibuprofen** (NSAID transport model)
