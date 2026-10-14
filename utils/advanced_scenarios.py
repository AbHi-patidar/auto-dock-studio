"""
Advanced Scenario Advisor for AutoDock Studio
Detects special cases that require modified docking protocols:
- Covalent binding targets (reactive cysteines, serine proteases)
- Metal-coordinated active sites (Zn2+, Mg2+, Fe2+ enzymes)
- Membrane proteins / GPCRs (require lipid bilayer context)
- Allosteric pockets (not the orthosteric site)
- Intrinsically disordered regions (IDRs)
- RNA/DNA binding proteins
- Glycosylated proteins
- Homodimeric / oligomeric interface binders
"""

import math

# Residue signatures for special case detection
COVALENT_WARHEAD_RESIDUES = {"CYS", "SER", "LYS", "THR", "HIS"}
SERINE_PROTEASE_CATALYTIC = {("SER", "HIS", "ASP"), ("SER", "HIS", "GLU")}  # catalytic triad

METAL_IONS = {"ZN", "MG", "FE", "CA", "MN", "CU", "NI", "CO", "CD", "HG", "PT"}

MEMBRANE_PROTEIN_INDICATORS = {
    "TM", "TRANSMEMBRANE", "GPCR", "CHANNEL", "TRANSPORTER",
    "7TM", "SLC", "ABC", "KCNA", "KCNQ", "SCN", "CACNA"
}

RNA_BINDING_DOMAINS = {"RRM", "KH", "DEAD", "TUDOR", "DSRM", "SAM", "PUF"}

GLYCOSYLATION_RESIDUES = {"ASN", "SER", "THR"}  # N-linked: ASN, O-linked: SER/THR

KNOWN_SPECIAL_TARGETS = {
    # Covalent targets
    "EGFR":  {"type": "covalent", "note": "Contains reactive Cys797; covalent inhibitors (e.g., Osimertinib) form irreversible bonds. Standard Vina gives ΔG but not covalent TS energy."},
    "BTK":   {"type": "covalent", "note": "Cys481 is the covalent warhead site (Ibrutinib target). Reversible docking score only."},
    "KRAS":  {"type": "covalent", "note": "G12C mutation (Sotorasib, Adagrasib) requires covalent constraint. Non-covalent docking only gives initial recognition score."},
    "MAO-B": {"type": "covalent", "note": "FAD-dependent; some inhibitors form covalent adducts with FAD prosthetic group."},

    # Metal-coordinated
    "HDAC":  {"type": "metal", "note": "Zn2+ chelation is pharmacophoric. Ensure ZN is preserved during protein prep and set up zinc coordination geometry."},
    "MMP":   {"type": "metal", "note": "Matrix metalloproteinases require Zn2+ for catalysis. Hydroxamate warheads coordinate Zn—check that ZN HETATM was preserved."},
    "CA2":   {"type": "metal", "note": "Carbonic anhydrase active site Zn2+ coordinates the sulfonamide pharmacophore. Verify ZN preservation."},
    "HIV":   {"type": "metal", "note": "HIV integrase has Mg2+-dependent active site. Diketo acids coordinate Mg."},
    "ACE":   {"type": "metal", "note": "ACE is Zn2+ carboxypeptidase. ACE inhibitors (captopril, lisinopril) chelate catalytic Zn2+."},

    # Membrane / GPCR
    "ADRB":  {"type": "membrane", "note": "Beta-adrenergic GPCR. Docking within TM bundle is valid; ensure full TM helix coverage in box."},
    "DRD":   {"type": "membrane", "note": "Dopamine receptor GPCR. Standard Vina docking in orthosteric binding pocket is feasible."},
    "CHRM":  {"type": "membrane", "note": "Muscarinic GPCR. Remove detergent molecules (DM, OG, CYMAL) before prep."},
    "ADORA": {"type": "membrane", "note": "Adenosine receptor GPCR. Allosteric site available (NAM/PAM binding)."},

    # RNA-binding / Nucleic acid
    "RNAP":  {"type": "rna_dna", "note": "RNA polymerase. Rifampicin binds the RNA exit channel. Structural waters may be important."},
    "REV":   {"type": "rna_dna", "note": "HIV Rev protein binds RRE RNA. Ensure RNA is removed or treated properly if co-crystalized."},

    # Allosteric sites
    "AKT":   {"type": "allosteric", "note": "PH domain allosteric site available in addition to ATP orthosteric site. Specify box accordingly."},
    "MEK":   {"type": "allosteric", "note": "MEK1/2 has an allosteric inhibitor binding site adjacent to ATP pocket; known non-competitive inhibitors bind here."},
    "BCR":   {"type": "allosteric", "note": "BCR-ABL myristoyl allosteric pocket (Asciminib site). Different from ATP site."},

    # IDR / flexible
    "P53":   {"type": "idr", "note": "p53 transactivation domain is intrinsically disordered. Pocket forms transiently; consider ensemble docking if available."},
    "MYC":   {"type": "idr", "note": "c-Myc is largely disordered. BET bromodomain or Max-Myc interface are the druggable entry points."},
}

SPECIAL_SCENARIO_PROTOCOLS = {
    "covalent": {
        "label": "⚗️ Covalent Binding Target",
        "color": "#f97316",
        "icon": "⚗️",
        "short": "This target may undergo covalent bond formation with inhibitors.",
        "protocol": [
            "Standard AutoDock Vina scoring reflects **non-covalent recognition only** — the docking score does not capture covalent bond energy.",
            "For true covalent docking, use Schrödinger CovDock, AutoDock-GPU with covalent constraints, or GOLD with tethered docking.",
            "The **pose geometry** (orientation of warhead relative to reactive residue) is still meaningful — a correctly oriented reactive group (e.g., Michael acceptor pointing at Cys) indicates covalent competence.",
            "Check that the ligand warhead (vinyl sulfone, acrylamide, epoxide, etc.) is within 3.5–4.0 Å of the reactive residue in the top pose.",
            "Ki values from this platform are theoretical estimates for the non-covalent encounter complex, NOT for the covalently modified enzyme.",
        ]
    },
    "metal": {
        "label": "🔬 Metal-Coordinated Active Site",
        "color": "#a78bfa",
        "icon": "🔬",
        "short": "Active site contains catalytic metal ion(s) that are pharmacophoric.",
        "protocol": [
            "This platform **preserves catalytic metals** (Zn²⁺, Mg²⁺, Fe²⁺, etc.) during protein preparation — verify 'Preserved Metals' in the prep summary.",
            "AutoDock Vina uses van der Waals + electrostatic scoring which approximates metal coordination but does not explicitly model coordination bonds.",
            "For hydroxamic acid, sulfonamide, or catechol warheads that coordinate metals, inspect the top pose: the metal-chelating atoms should be ≤ 2.5 Å from the metal ion.",
            "If the metal is missing from your docking result (binding score worse than expected), manually set the grid box to center on the metal ion coordinates.",
            "For rigorous metal coordination: use AutoDock4 with metal-aware force field or Glide with custom metal constraints.",
        ]
    },
    "membrane": {
        "label": "🧬 Membrane Protein / GPCR",
        "color": "#34d399",
        "icon": "🧬",
        "short": "Membrane-embedded receptor requiring special box placement.",
        "protocol": [
            "Use PDB structures of **detergent-solubilized** or **lipid nanodisc** membrane proteins (confirm crystallization conditions in RCSB TITLE/REMARK records).",
            "Remove non-essential HETATM crystallography agents: detergents (DM, OG, CYMAL, DDM), lipids (POP, POPC, PE), and PEG molecules.",
            "Set the grid box to enclose the **orthosteric binding pocket** (typically within TM3–TM5–TM6–TM7 for GPCRs) using the co-crystal ligand centroid.",
            "For GPCRs, keep structural waters within the TM bundle (especially water-mediated H-bonds in extracellular loops).",
            "Blind docking over the full protein is valid for discovering novel allosteric sites but will give non-specific contacts at the membrane-facing surface.",
        ]
    },
    "rna_dna": {
        "label": "🧬 RNA/DNA Binding Target",
        "color": "#22d3ee",
        "icon": "🔬",
        "short": "Protein binds nucleic acids; nucleic acid co-factors may be in the structure.",
        "protocol": [
            "If the PDB contains RNA/DNA, remove it unless docking to the RNA-binding groove (where RNA is the competing substrate).",
            "Structural Mg²⁺ or Zn²⁺ ions in RNA-binding sites should be preserved.",
            "Many RNA-binding pockets are large and open — use smaller search box (≤ 20 × 20 × 20 Å) focused on the nucleotide recognition motif.",
            "Competitive inhibitors targeting RNA polymerase rifamycin-binding channels or ribosome peptidyl transferase center are feasible with standard Vina.",
            "Drugs targeting protein–RNA interfaces (e.g., Branaplam for SMN2 exon splicing) are an active area — docking score may underestimate due to entropic surface.",
        ]
    },
    "allosteric": {
        "label": "🔄 Allosteric Binding Site",
        "color": "#fb923c",
        "icon": "🔄",
        "short": "Secondary pocket remote from the orthosteric/catalytic site.",
        "protocol": [
            "Set the grid box to center on the **allosteric cavity**, NOT the ATP/substrate binding site, unless testing competitive inhibitors.",
            "Allosteric sites often exhibit higher flexibility — if available, use **multiple PDB conformations** or an MD-derived ensemble.",
            "Scoring functions may underperform at allosteric sites due to less-defined electrostatic landscapes — treat scores as relative rankings.",
            "Consider using the co-crystal structure of an allosteric inhibitor (if available) to identify the site centroid.",
            "Positive allosteric modulators (PAMs) and negative allosteric modulators (NAMs) can both be explored; Ki interpretation differs (not competitive inhibition).",
        ]
    },
    "idr": {
        "label": "🌀 Intrinsically Disordered Region (IDR)",
        "color": "#f472b6",
        "icon": "🌀",
        "short": "Target lacks stable tertiary structure; pocket is transient.",
        "protocol": [
            "Classic rigid docking has **limited predictive power** for IDR targets — scores should be treated as qualitative.",
            "Prefer cryo-EM/X-ray structures where a co-crystal ligand induces a defined pocket (induced-fit mechanism).",
            "For MYC/MAX and p53 TAD, use **protein–protein interface structures** (e.g., p53:MDM2) as the receptor — the bound conformation is most predictive.",
            "Fragment-based or tethering approaches in wet lab are preferred for IDR targets.",
            "If using this platform, run blind docking over the full structure and look for clusters of poses — convergent binding hotspots indicate transient pockets.",
        ]
    },
    "glycosylated": {
        "label": "🍬 Glycoprotein Surface",
        "color": "#4ade80",
        "icon": "🍬",
        "short": "N- or O-linked glycans may occlude or form part of the binding site.",
        "protocol": [
            "Most X-ray structures model only the first 1–2 glycan residues; the full glycan tree is absent — docking to exposed glycan-proximal sites may be unrealistic.",
            "For drugs targeting glycan-adjacent pockets (e.g., SARS-CoV-2 Spike glycan shield), remove glycans and note this limitation.",
            "NAG (N-acetylglucosamine) residues can be preserved as part of the receptor if they form direct interactions with the ligand.",
            "Cryo-EM structures with full glycan models provide better context for glycan-shielded epitopes.",
        ]
    },
    "homodimer": {
        "label": "⚖️ Protein–Protein Interface (PPI) / Dimer Site",
        "color": "#818cf8",
        "icon": "⚖️",
        "short": "Ligand targets the interface between two protein chains.",
        "protocol": [
            "Include **both chains** of the dimer/oligomer in the receptor PDBQT to correctly represent the interfacial pocket.",
            "Set the grid box to span the protein–protein interface — use the center of mass between the two chains.",
            "PPI pockets are typically flat and large — expect lower docking scores relative to deep enzymatic sites.",
            "Look for 'hot-spot' residues (Trp, Arg, Tyr) that dominate the interface energy — good ligands mimic these residue side chains.",
            "Examples: Bcl-2/Bcl-xL BH3 mimetics (Venetoclax), MDM2-p53 (Nutlins), IL-2 receptor interface (SP4206).",
        ]
    }
}


def detect_special_scenarios_from_pdb(clean_pdb_path: str, preserved_hetero: list = None) -> list:
    """
    Parses a cleaned PDB to auto-detect special docking scenarios:
    - Metal ions present
    - Unusual residues suggesting covalent warheads
    - Multi-chain oligomers
    - Disulfide bridges
    Returns list of scenario dicts.
    """
    detected = []
    chain_ids = set()
    residues = {}
    has_metals = []
    disulfide_cys = 0
    non_standard = []

    try:
        with open(clean_pdb_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                rec = line[:6].strip()
                if rec in ("ATOM", "HETATM"):
                    resname = line[17:20].strip()
                    chain = line[21:22].strip()
                    resid = line[22:26].strip()
                    key = f"{resname}{resid}{chain}"
                    chain_ids.add(chain)

                    if resname in METAL_IONS or rec == "HETATM" and resname in METAL_IONS:
                        has_metals.append(resname)

                    residues[key] = resname

                if line.startswith("SSBOND"):
                    disulfide_cys += 1
    except Exception:
        pass

    unique_chains = len(chain_ids)
    has_metals = list(set(has_metals))

    # Metal coordination
    if has_metals:
        detected.append({
            "scenario": "metal",
            "auto_reason": f"Catalytic metal ion(s) detected in structure: {', '.join(has_metals)}",
            "ions": has_metals
        })

    # Multi-chain (potential PPI site)
    if unique_chains >= 2:
        detected.append({
            "scenario": "homodimer",
            "auto_reason": f"Structure contains {unique_chains} chains (A–{list(chain_ids)[-1]}). If targeting the inter-chain interface, include all chains in the receptor."
        })

    # Preserved hetero cofactors that imply covalent activity
    if preserved_hetero:
        cofactors = [h for h in preserved_hetero if "Cofactor" in h]
        if cofactors:
            detected.append({
                "scenario": "covalent",
                "auto_reason": f"Cofactor(s) detected: {', '.join(cofactors)}. Some cofactor-dependent enzymes form covalent adducts with inhibitors."
            })

    # Disulfide bridges
    if disulfide_cys > 0:
        detected.append({
            "scenario": "covalent",
            "auto_reason": f"{disulfide_cys} disulfide bridge(s) detected (SSBOND records). If ligand is a disulfide-based covalent probe, special modeling applies."
        })

    return detected


def detect_target_by_pdb_id(pdb_id: str) -> dict:
    """
    Matches a PDB ID prefix against known special targets database.
    """
    pdb_id = pdb_id.upper().strip()
    for target_key, info in KNOWN_SPECIAL_TARGETS.items():
        if target_key in pdb_id:
            return {"matched_key": target_key, **info}
    return {}


def get_scenario_panel_data(
    clean_pdb_path: str = None,
    pdb_id: str = "",
    preserved_hetero: list = None,
    user_scenario: str = None
) -> list:
    """
    Returns list of active scenario dicts for UI display.
    Combines auto-detection from PDB + PDB-ID matching + user-selected scenario.
    Each dict: {scenario_key, label, color, icon, short, protocol, source, auto_reason?}
    """
    active = {}

    # 1. Auto-detect from structure
    if clean_pdb_path:
        auto_detected = detect_special_scenarios_from_pdb(clean_pdb_path, preserved_hetero)
        for det in auto_detected:
            sc = det["scenario"]
            if sc not in active and sc in SPECIAL_SCENARIO_PROTOCOLS:
                entry = dict(SPECIAL_SCENARIO_PROTOCOLS[sc])
                entry["scenario_key"] = sc
                entry["source"] = "auto"
                entry["auto_reason"] = det.get("auto_reason", "")
                active[sc] = entry

    # 2. Match known PDB IDs
    if pdb_id:
        kb_match = detect_target_by_pdb_id(pdb_id)
        if kb_match:
            sc = kb_match.get("type", "")
            if sc and sc in SPECIAL_SCENARIO_PROTOCOLS:
                entry = dict(SPECIAL_SCENARIO_PROTOCOLS[sc])
                entry["scenario_key"] = sc
                entry["source"] = "knowledge_base"
                entry["kb_note"] = kb_match.get("note", "")
                entry["matched_target"] = kb_match.get("matched_key", "")
                if sc not in active:
                    active[sc] = entry
                else:
                    # Enrich existing with KB note
                    active[sc]["kb_note"] = kb_match.get("note", "")
                    active[sc]["matched_target"] = kb_match.get("matched_key", "")

    # 3. User-selected manual scenario
    if user_scenario and user_scenario in SPECIAL_SCENARIO_PROTOCOLS:
        if user_scenario not in active:
            entry = dict(SPECIAL_SCENARIO_PROTOCOLS[user_scenario])
            entry["scenario_key"] = user_scenario
            entry["source"] = "user_selected"
            active[user_scenario] = entry

    return list(active.values())


def append_advanced_scenario_to_manuscript(manuscript_text: str, active_scenarios: list) -> str:
    """
    Appends an 'Advanced Considerations' section to the manuscript draft
    when special docking scenarios are detected.
    """
    if not active_scenarios:
        return manuscript_text

    section = "\n#### 5. Special Docking Considerations & Limitations\n"
    for sc in active_scenarios:
        section += f"\n**{sc['label']}**: {sc['short']}\n"
        if sc.get("kb_note"):
            section += f"> ⚠️ Target-specific note: {sc['kb_note']}\n"
        section += "Key protocol adjustments:\n"
        for step in sc["protocol"][:3]:  # top 3 for manuscript brevity
            section += f"- {step}\n"

    return manuscript_text + section
