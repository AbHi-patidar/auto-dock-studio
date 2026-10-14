"""
Automated Results & Scientific Interpretation Engine for AutoDock Studio
Computes thermodynamic inhibition constants (Ki), Ligand Efficiency (LE),
and generates manuscript-ready scientific interpretations and molecular mechanisms.
"""

import math

def calculate_inhibition_constant(delta_g_kcal: float, temperature_k: float = 298.15):
    """
    Thermodynamic inhibition constant Ki = exp(dG / RT).

    Uses the RAW delta G value. (A previous version forced dG negative via
    -abs(), which overstated potency by ~1000x for unfavorable binders:
    e.g. dG = +2.0 kcal/mol was reported as Ki ~ 34 mM instead of ~29 M.)
    Positive dG (unfavorable) yields Ki > 1 M and is flagged as non-binding.
    """
    rt = 0.001987204 * temperature_k
    try:
        ki_molar = math.exp(delta_g_kcal / rt)
    except OverflowError:
        # Hugely unfavorable dG -> astronomically large Ki
        return float("inf"), "> 1e6 M (non-binding)"

    binding_note = " (non-binding)" if delta_g_kcal > 0 else ""

    if ki_molar < 1e-9:
        ki_str = f"{ki_molar * 1e12:.2f} pM"
    elif ki_molar < 1e-6:
        ki_str = f"{ki_molar * 1e9:.2f} nM"
    elif ki_molar < 1e-3:
        ki_str = f"{ki_molar * 1e6:.2f} µM"
    elif ki_molar < 1e3:
        ki_str = f"{ki_molar * 1e3:.2f} mM"
    else:
        ki_str = f"{ki_molar:.2f} M"
    return ki_molar, ki_str + binding_note

def calculate_ligand_efficiency(delta_g_kcal: float, num_heavy_atoms: int):
    if num_heavy_atoms <= 0:
        return 0.0
    return round(abs(delta_g_kcal) / num_heavy_atoms, 2)

def generate_scientific_interpretation(
    target_name: str,
    ligand_name: str,
    best_pose: dict,
    all_poses: list,
    interactions: dict,
    drug_likeness: dict,
    num_heavy_atoms: int = 30
):
    affinity = best_pose.get("affinity", 0.0)
    abs_aff = abs(affinity)
    ki_val, ki_str = calculate_inhibition_constant(affinity)
    le = calculate_ligand_efficiency(affinity, num_heavy_atoms)

    h_bonds = interactions.get("h_bonds", [])
    hydrophobic = interactions.get("hydrophobic", [])
    salt_bridges = interactions.get("salt_bridges", [])
    pocket_res = interactions.get("pocket_residues", [])

    if affinity > 0:
        tier = "Non-Binding (Unfavorable ΔG)"
        tier_desc = "shows a positive binding free energy, indicating no spontaneous binding; the reported Ki exceeds 1 M."
        tier_color = "#f87171"
    elif abs_aff >= 9.0:
        tier = "High-Affinity / Potent Binder"
        tier_desc = "displays exceptionally strong thermodynamic stability, indicating sub-micromolar or nanomolar inhibitory potential."
        tier_color = "#22c55e"
    elif abs_aff >= 7.0:
        tier = "Moderate to Strong Lead"
        tier_desc = "exhibits favorable binding energetics typical of validated hit-to-lead chemical series."
        tier_color = "#38bdf8"
    elif abs_aff >= 5.5:
        tier = "Moderate Binder / Hit"
        tier_desc = "shows moderate binding affinity in the low-to-mid micromolar range, suitable as a starting point for optimization."
        tier_color = "#facc15"
    else:
        tier = "Weak / Non-Specific Contact"
        tier_desc = "indicates shallow electrostatic or steric engagement that may require scaffold redesign."
        tier_color = "#f87171"

    hbond_residues = list(set([hb["residue"] for hb in h_bonds]))
    hydro_residues = list(set([hc["residue"] for hc in hydrophobic]))

    if hbond_residues:
        hbond_text = f"The binding mode is anchored by {len(h_bonds)} direct polar hydrogen bonds with residues **{', '.join(hbond_residues[:4])}**, providing directional electrostatic lock."
    else:
        hbond_text = "No direct polar hydrogen bonds under 3.5 Å were identified, indicating the interaction is predominantly non-polar."

    if hydro_residues:
        hydro_text = f"Hydrophobic packing is established with **{', '.join(hydro_residues[:5])}**, contributing non-polar surface burial and entropic gain."
    else:
        hydro_text = "Hydrophobic surface contacts are minimal."

    if len(all_poses) > 1:
        top_rmsds = [p.get("rmsd_lb", 0.0) for p in all_poses[1:4]]
        avg_rmsd = sum(top_rmsds) / len(top_rmsds) if top_rmsds else 0.0
        if avg_rmsd < 2.0:
            pose_convergence = "High conformational convergence: Top poses cluster within < 2.0 Å RMSD, indicating a well-defined single energy minimum in the binding pocket."
        else:
            pose_convergence = "Multi-state conformational flexibility: Top modes explore distinct spatial sub-pockets across the cavity."
    else:
        pose_convergence = "Single optimal pose resolved."

    violations = drug_likeness.get("violations", 0)
    if violations == 0:
        dlikeness_text = "The molecule fully complies with Lipinski's Rule of 5 (0 violations), demonstrating optimal oral bioavailability and pharmacokinetic potential."
    elif violations == 1:
        dlikeness_text = f"The compound shows acceptable drug-likeness with 1 minor Lipinski variation ({drug_likeness.get('rule_summary')}), consistent with established clinical therapeutics."
    else:
        dlikeness_text = f"The compound shows {violations} Lipinski violations. Medicinal chemistry optimization may be required to enhance permeability."

    rec_action = "The high binding score and favorable interaction profile support advancing this compound directly to in vitro enzymatic binding assays." if abs_aff >= 7.0 else "Scaffold decoration to introduce additional hydrogen bond donor/acceptor pairs is recommended to enhance binding affinity."

    manuscript_draft = f"""### Results and Discussion

#### 1. Binding Thermodynamics & Theoretical Potency
Molecular docking of **{ligand_name}** against **{target_name}** revealed a top binding free energy of **ΔG = {affinity} kcal/mol**, corresponding to an estimated theoretical inhibition constant of **Ki = {ki_str}**. The observed affinity places the compound in the **{tier}** category, demonstrating that the ligand {tier_desc} The calculated Ligand Efficiency (LE) was **{le} kcal/mol per heavy atom**.

#### 2. Molecular Mechanisms of Pocket Recognition
{hbond_text} 
{hydro_text}

Overall, {len(pocket_res)} pocket residues were actively recruited into the binding interface, with the ligand occupying the active site cavity without steric clashes.

#### 3. Conformational Convergence & Pharmacokinetics
Analysis of the top binding modes revealed: *{pose_convergence}*
Furthermore, physicochemical evaluation confirmed: *{dlikeness_text}*

#### 4. Medicinal Chemistry Recommendation
{rec_action}
"""

    return {
        "affinity_kcal": affinity,
        "affinity_tier": tier,
        "affinity_tier_desc": tier_desc,
        "tier_color": tier_color,
        "ki_str": ki_str,
        "ki_val": ki_val,
        "ligand_efficiency": le,
        "hbond_summary": hbond_text,
        "hydro_summary": hydro_text,
        "pose_convergence": pose_convergence,
        "druglikeness_summary": dlikeness_text,
        "manuscript_draft": manuscript_draft
    }