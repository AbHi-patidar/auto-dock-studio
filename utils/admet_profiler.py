"""
Extended ADMET & Medicinal Chemistry Profiler for AutoDock Studio
Evaluates drug-likeness rules (Lipinski, Veber, Ghose), QED score,
Synthetic Accessibility (SAScore), and Pan-Assay Interference Compounds (PAINS).
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, Crippen
import math

# Canonical PAINS filter SMARTS (Pan-Assay Interference Compounds)
PAINS_SMARTS = {
    "Rhodanine": "O=C1NC(=S)SC1",
    "Ene_Rhodanine": "O=C1NC(=S)SC1=C",
    "Quinone": "O=C1C=CC(=O)C=C1",
    "Catechol": "Oc1ccccc1O",
    "Alkyl_Halide": "[CX4][Cl,Br,I]",
    "Michael_Acceptor": "[CX3]=[CX3][CX3](=[OX1])",
    "Azo_Compound": "[NX2]=[NX2]",
    "Nitro_Aromatic": "c[N+](=O)[O-]",
    "Aniline_Di_Substituted": "c[NX3]([#6])[#6]",
    "Thiourea": "NC(=S)N",
    "Polyene": "C=CC=CC=C"
}

def profile_extended_admet(smiles: str) -> dict:
    """
    Computes a comprehensive SwissADME-grade medicinal chemistry and pharmacokinetic report.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    # Basic Physicochemical Properties
    mw = round(Descriptors.MolWt(mol), 2)
    logp = round(Crippen.MolLogP(mol), 2)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = round(Descriptors.TPSA(mol), 2)
    rotb = Lipinski.NumRotatableBonds(mol)
    mr = round(Crippen.MolMR(mol), 2)
    heavy_atoms = mol.GetNumHeavyAtoms()
    fsp3 = round(Descriptors.FractionCSP3(mol), 2)
    num_rings = Lipinski.RingCount(mol)
    aromatic_rings = Lipinski.NumAromaticRings(mol)

    # 1. Lipinski Rule of 5
    lipinski_violations = 0
    if mw > 500: lipinski_violations += 1
    if logp > 5.0: lipinski_violations += 1
    if hbd > 5: lipinski_violations += 1
    if hba > 10: lipinski_violations += 1

    # 2. Veber Filter (Bioavailability)
    veber_violations = 0
    if rotb > 10: veber_violations += 1
    if tpsa > 140.0: veber_violations += 1
    veber_pass = veber_violations == 0

    # 3. Ghose Filter
    ghose_violations = 0
    if not (160 <= mw <= 480): ghose_violations += 1
    if not (-0.4 <= logp <= 5.6): ghose_violations += 1
    if not (40 <= mr <= 130): ghose_violations += 1
    if not (20 <= mol.GetNumAtoms() <= 70): ghose_violations += 1
    ghose_pass = ghose_violations <= 1

    # 4. QED (Quantitative Estimate of Drug-likeness: 0 to 1)
    qed_score = round(QED.qed(mol), 3)

    # 5. Synthetic Accessibility Score Approximation (1 easy -> 10 difficult)
    # Based on molecular weight, ring complexity, and sp3 fraction
    sa_score = round(1.0 + (mw / 180.0) + (num_rings * 0.45) + (rotb * 0.15) - (fsp3 * 0.8), 2)
    sa_score = max(1.0, min(10.0, sa_score))

    # 6. PAINS Substructure Screening
    pains_alerts = []
    for name, smarts in PAINS_SMARTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            pains_alerts.append(name.replace("_", " "))

    # 7. Pharmacokinetic Model Estimations (Egan / Veber models)
    hia_high = tpsa < 131.6 and logp > -2.0 and logp < 5.8
    bbb_permeant = tpsa < 90.0 and logp > 1.0 and logp < 4.5 and mw < 400

    physico = {
        "mw": mw,
        "logp": logp,
        "hbd": hbd,
        "hba": hba,
        "tpsa": tpsa,
        "rotb": rotb,
        "mr": mr,
        "fsp3": fsp3,
        "heavy_atoms": heavy_atoms,
        "num_rings": num_rings,
        "aromatic_rings": aromatic_rings
    }

    return {
        **physico,
        "physicochemical": physico,
        "qed": qed_score,
        "sa_score": sa_score,
        "lipinski": {
            "violations": lipinski_violations,
            "pass": lipinski_violations <= 1,
            "summary": "Compliant" if lipinski_violations <= 1 else f"{lipinski_violations} Violations"
        },
        "veber": {
            "violations": veber_violations,
            "pass": veber_pass,
            "summary": "Oral Bioavailable" if veber_pass else "Poor Bioavailability"
        },
        "ghose": {
            "violations": ghose_violations,
            "pass": ghose_pass,
            "summary": "Passed" if ghose_pass else "Failed"
        },
        "pains": {
            "alerts": pains_alerts,
            "has_alerts": len(pains_alerts) > 0,
            "summary": "Clean (No PAINS)" if not pains_alerts else f"⚠️ Alert: {', '.join(pains_alerts)}"
        },
        "pharmacokinetics": {
            "gastrointestinal_absorption": "High" if hia_high else "Low",
            "blood_brain_barrier": "Permeant" if bbb_permeant else "Non-permeant"
        }
    }

calculate_comprehensive_admet = profile_extended_admet

def format_admet_summary_markdown(admet: dict) -> str:
    """Formats an ADMET dict into a clean academic report block."""
    if not admet:
        return "No ADMET data available."
    p = admet["physicochemical"]
    return f"""### Extended ADMET & Pharmacokinetic Report
- **Physicochemical:** MW = {p['mw']} Da | LogP = {p['logp']} | TPSA = {p['tpsa']} A^2 | Rotatable Bonds = {p['rotb']}
- **H-Bond Donors / Acceptors:** {p['hbd']} / {p['hba']} | Fraction Csp3: {p['fsp3']}
- **Drug-likeness Scores:** QED = {admet['qed']} | Synthetic Accessibility (SAScore) = {admet['sa_score']}/10
- **Filter Compliance:**
  - **Lipinski (Ro5):** {admet['lipinski']['summary']} ({admet['lipinski']['violations']} violations)
  - **Veber Filter:** {admet['veber']['summary']}
  - **Ghose Filter:** {admet['ghose']['summary']}
- **Promiscuity / PAINS:** {admet['pains']['summary']}
- **Pharmacokinetics:** GI Absorption = {admet['pharmacokinetics']['gastrointestinal_absorption']} | BBB Permeability = {admet['pharmacokinetics']['blood_brain_barrier']}
"""
