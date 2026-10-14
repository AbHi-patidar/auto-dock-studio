"""
2D Interaction Diagram and Drug-Likeness Profiler for AutoDock Studio
Generates publication-quality 2D interaction maps (LigPlot / PoseView style)
in vector SVG and 300 DPI PNG, alongside Lipinski Rule of 5 evaluation.
"""

import os
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image, ImageDraw, ImageFont
import io

def calculate_drug_likeness(smiles: str) -> dict:
    """
    Computes physicochemical properties and evaluates Lipinski's Rule of 5.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    mw = round(Descriptors.MolWt(mol), 2)
    logp = round(Descriptors.MolLogP(mol), 2)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = round(Descriptors.TPSA(mol), 2)
    rotb = Lipinski.NumRotatableBonds(mol)

    violations = 0
    if mw > 500:
        violations += 1
    if logp > 5.0:
        violations += 1
    if hbd > 5:
        violations += 1
    if hba > 10:
        violations += 1

    is_druglike = violations <= 1

    return {
        "mw": mw,
        "logp": logp,
        "hbd": hbd,
        "hba": hba,
        "tpsa": tpsa,
        "rotb": rotb,
        "violations": violations,
        "is_druglike": is_druglike,
        "rule_summary": "Passed (Drug-like)" if is_druglike else f"{violations} Violations (Borderline)"
    }

def _get_font(size: int, bold: bool = False):
    """Loads system TrueType font for razor-sharp typography on high-DPI export."""
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()

def _get_system_font_path():
    """Returns the path to a high-quality system TrueType font for RDKit vector rendering."""
    for p in ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf"]:
        if os.path.exists(p):
            return p
    return None

def generate_2d_interaction_diagram(
    smiles: str,
    interactions: dict,
    output_png_path: str,
    output_svg_path: str = None,
    width: int = 2400,
    height: int = 1800
):
    """
    Creates an ultra-high resolution (300 DPI / 4K print grade) 2D interaction diagram
    showing the flat chemical structure with color-coded interacting atoms and residue callouts.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string for 2D diagram generation.")

    # Generate optimal 2D coordinates
    AllChem.Compute2DCoords(mol)

    h_bonds = interactions.get("h_bonds", [])
    hydrophobic = interactions.get("hydrophobic", [])
    salt_bridges = interactions.get("salt_bridges", [])
    pi_stacking = interactions.get("pi_stacking", [])
    metal_coordination = interactions.get("metal_coordination", [])
    halogen_bonds = interactions.get("halogen_bonds", [])

    highlight_atoms = []
    highlight_colors = {}
    atom_annotations = {}

    for hb in h_bonds:
        lig_atom_name = hb.get("ligand_atom", "").strip()
        matched_idx = _find_matching_atom(mol, lig_atom_name, is_polar=True)
        if matched_idx is not None:
            highlight_atoms.append(matched_idx)
            highlight_colors[matched_idx] = (0.92, 0.70, 0.05, 0.45)  # Warm Amber/Gold
            res_label = hb.get("residue", "")
            if res_label:
                atom_annotations[matched_idx] = f"{res_label}"

    for hc in hydrophobic:
        lig_atom_name = hc.get("ligand_atom", "").strip()
        matched_idx = _find_matching_atom(mol, lig_atom_name, is_polar=False)
        if matched_idx is not None:
            if matched_idx not in highlight_colors:
                highlight_atoms.append(matched_idx)
                highlight_colors[matched_idx] = (0.02, 0.71, 0.85, 0.35)  # Cyan
            if matched_idx not in atom_annotations:
                res_label = hc.get("residue", "")
                if res_label:
                    atom_annotations[matched_idx] = f"{res_label}"

    for sb in salt_bridges:
        lig_atom_name = sb.get("ligand_atom", "").strip()
        matched_idx = _find_matching_atom(mol, lig_atom_name, is_polar=True)
        if matched_idx is not None:
            highlight_atoms.append(matched_idx)
            highlight_colors[matched_idx] = (0.85, 0.20, 0.65, 0.45)  # Magenta
            res_label = sb.get("residue", "")
            if res_label:
                atom_annotations[matched_idx] = f"{res_label} (ionic)"

    for mc in metal_coordination:
        lig_atom_name = mc.get("ligand_atom", "").strip()
        matched_idx = _find_matching_atom(mol, lig_atom_name, is_polar=True)
        if matched_idx is not None:
            highlight_atoms.append(matched_idx)
            highlight_colors[matched_idx] = (0.95, 0.45, 0.10, 0.45)  # Orange
            atom_annotations[matched_idx] = f"{mc.get('metal', 'M')} Coord"

    for hx in halogen_bonds:
        lig_atom_name = hx.get("ligand_atom", "").strip()
        matched_idx = _find_matching_atom(mol, lig_atom_name, is_polar=False)
        if matched_idx is not None:
            highlight_atoms.append(matched_idx)
            highlight_colors[matched_idx] = (0.06, 0.72, 0.50, 0.45)  # Emerald green
            res_label = hx.get("residue", "")
            if res_label:
                atom_annotations[matched_idx] = f"{res_label} (X-bond)"

    # Attach residue labels directly as atom annotations
    for idx, note in atom_annotations.items():
        if 0 <= idx < mol.GetNumAtoms():
            mol.GetAtomWithIdx(idx).SetProp("atomNote", note)

    highlight_atoms = list(set(highlight_atoms))
    font_file = _get_system_font_path()

    # 1. Generate Vector SVG (Infinite scalability for vector publishing)
    drawer_svg = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts_svg = drawer_svg.drawOptions()
    opts_svg.clearBackground = True
    opts_svg.bondLineWidth = 4.8
    opts_svg.minFontSize = 26
    opts_svg.maxFontSize = 42
    opts_svg.annotationFontScale = 0.82
    opts_svg.atomHighlightsAreCircles = True
    opts_svg.fillHighlights = True
    opts_svg.highlightRadius = 0.40
    opts_svg.legendFontSize = 42
    if font_file:
        opts_svg.fontFile = font_file

    drawer_svg.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        legend="Figure 2: 2D Protein-Ligand Interaction Map (LigPlot Style)"
    )
    drawer_svg.FinishDrawing()
    svg_text = drawer_svg.GetDrawingText()

    if output_svg_path:
        with open(output_svg_path, "w", encoding="utf-8") as f:
            f.write(svg_text)

    # 2. Generate Ultra-High Resolution 300 DPI PNG
    drawer_png = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts_png = drawer_png.drawOptions()
    opts_png.clearBackground = True
    opts_png.bondLineWidth = 4.8
    opts_png.minFontSize = 26
    opts_png.maxFontSize = 42
    opts_png.annotationFontScale = 0.82
    opts_png.atomHighlightsAreCircles = True
    opts_png.fillHighlights = True
    opts_png.highlightRadius = 0.40
    opts_png.legendFontSize = 42
    if font_file:
        opts_png.fontFile = font_file

    drawer_png.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        legend="Figure 2: 2D Protein-Ligand Interaction Map (LigPlot Style)"
    )
    drawer_png.FinishDrawing()
    png_bytes = drawer_png.GetDrawingText()

    # Post-process with PIL to draw publication interaction callout card
    base_image = Image.open(io.BytesIO(png_bytes))
    final_image = _annotate_publication_legend(
        base_image,
        h_bonds=h_bonds,
        hydrophobic=hydrophobic,
        salt_bridges=salt_bridges,
        pi_stacking=pi_stacking,
        metal_coordination=metal_coordination,
        halogen_bonds=halogen_bonds
    )
    final_image.save(output_png_path, dpi=(300, 300))

    return {
        "svg": svg_text,
        "png_path": output_png_path,
        "svg_path": output_svg_path
    }

def _find_matching_atom(mol, lig_atom_name: str, is_polar: bool = True):
    """Matches atom name (e.g. 'O35', 'N10', 'C1') to RDKit atom index."""
    if not lig_atom_name:
        return None
    elem = lig_atom_name[0]
    idx_str = "".join([c for c in lig_atom_name if c.isdigit()])
    if idx_str:
        idx = int(idx_str) - 1
        if 0 <= idx < mol.GetNumAtoms() and mol.GetAtomWithIdx(idx).GetSymbol() == elem:
            return idx

    for a in mol.GetAtoms():
        if a.GetSymbol() == elem:
            if is_polar and elem in ("N", "O", "S"):
                return a.GetIdx()
            elif not is_polar and elem == "C":
                return a.GetIdx()
    return None

def _annotate_publication_legend(
    image: Image.Image,
    h_bonds: list,
    hydrophobic: list,
    salt_bridges: list,
    pi_stacking: list = None,
    metal_coordination: list = None,
    halogen_bonds: list = None
) -> Image.Image:
    """Adds a clean journal publication legend card with TrueType anti-aliased typography."""
    draw = ImageDraw.Draw(image)
    w, h = image.size

    title_font = _get_font(34, bold=True)
    body_font = _get_font(26, bold=False)

    items = [
        {"name": f"Hydrogen Bond ({len(h_bonds)} contacts)", "color": (234, 179, 8), "outline": (202, 138, 4)},
        {"name": f"Hydrophobic Contact ({len(hydrophobic)} contacts)", "color": (6, 182, 212), "outline": (8, 145, 178)},
    ]
    if salt_bridges:
        items.append({"name": f"Salt Bridge / Ionic ({len(salt_bridges)} contacts)", "color": (217, 70, 239), "outline": (192, 38, 211)})
    if pi_stacking:
        items.append({"name": f"π-Stacking / Aromatic ({len(pi_stacking)} contacts)", "color": (168, 85, 247), "outline": (126, 34, 206)})
    if metal_coordination:
        items.append({"name": f"Metal Coordination ({len(metal_coordination)} contacts)", "color": (249, 115, 22), "outline": (194, 65, 12)})
    if halogen_bonds:
        items.append({"name": f"Halogen Bond ({len(halogen_bonds)} contacts)", "color": (16, 185, 129), "outline": (5, 150, 105)})

    row_h = 48
    card_h = 75 + len(items) * row_h
    card_w = 880
    card_x, card_y = 60, h - card_h - 60

    # Draw card background with clean subtle border
    draw.rounded_rectangle(
        [card_x, card_y, card_x + card_w, card_y + card_h],
        radius=16,
        fill=(255, 255, 255, 252),
        outline=(203, 213, 225),
        width=3
    )
    draw.text((card_x + 30, card_y + 20), "NON-COVALENT INTERACTION LEGEND", font=title_font, fill=(15, 23, 42))

    current_y = card_y + 70
    for item in items:
        # Draw legend dot
        draw.ellipse([card_x + 30, current_y + 4, card_x + 54, current_y + 28], fill=item["color"], outline=item["outline"], width=2)
        draw.text((card_x + 68, current_y), item["name"], font=body_font, fill=(51, 65, 85))
        current_y += row_h

    return image
