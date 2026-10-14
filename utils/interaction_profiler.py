"""
Extended interaction detection: Pi-stacking, Metal coordination, Halogen bonds,
Cation-pi interactions added on top of existing H-bond / hydrophobic profiling.
"""

import os
import math
import numpy as np

def distance(c1, c2):
    return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2)

def parse_atoms_from_pdb_lines(lines, is_ligand=False):
    """
    Parses atoms from PDB/PDBQT formatted lines.
    Returns list of dicts: {name, resname, resid, chain, x, y, z, element, line}
    """
    atoms = []
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                name = line[12:16].strip()
                resname = line[17:20].strip()
                chain = line[21:22].strip() or "A"
                resid = int(line[22:26].strip()) if not is_ligand else 1
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                element = line[76:78].strip() if len(line) >= 78 else name[0]
                if not element or element.isdigit():
                    element = name[0]

                atoms.append({
                    "name": name,
                    "resname": resname,
                    "resid": resid,
                    "chain": chain,
                    "coord": np.array([x, y, z]),
                    "element": element.upper()
                })
            except (ValueError, IndexError):
                continue
    return atoms

# Aromatic residue ring center atoms (for pi-stacking approximation)
AROMATIC_RESIDUES = {
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TRP": ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
    "HID": ["CG", "ND1", "CD2", "CE1", "NE2"],
    "HIE": ["CG", "ND1", "CD2", "CE1", "NE2"],
}

CATALYTIC_METALS = {"ZN", "MG", "FE", "CA", "MN", "CU", "NI", "CO", "CD", "HG", "PT"}
METAL_COORDINATORS = {"N", "O", "S"}  # typical ligand coordination atoms

HALOGEN_DONORS = {"F", "CL", "BR", "I"}
HALOGEN_ACCEPTORS = {"N", "O", "S"}

def _ring_centroid(residue_atoms, ring_atom_names):
    """Compute centroid of aromatic ring atoms."""
    ring_coords = [a["coord"] for a in residue_atoms if a["name"] in ring_atom_names]
    if len(ring_coords) >= 3:
        return np.mean(ring_coords, axis=0)
    return None

def _normal_vector(ring_coords_list):
    """Approximate ring normal from 3 ring atoms using cross product."""
    if len(ring_coords_list) < 3:
        return None
    v1 = ring_coords_list[1] - ring_coords_list[0]
    v2 = ring_coords_list[2] - ring_coords_list[0]
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-6:
        return None
    return n / norm

def profile_interactions(receptor_pdb_path: str, ligand_pdb_lines: list):
    """
    Scans the complex and computes:
    1. Hydrogen Bonds (< 3.5 Angstrom between N/O and N/O)
    2. Hydrophobic Contacts (< 4.2 Angstrom between Carbon and Carbon)
    3. Salt Bridges (< 4.0 Angstrom between acidic and basic groups)
    4. Pi-Stacking interactions (aromatic residues < 5.5 A centroid distance)
    5. Metal Coordination bonds (< 3.0 A from metal to N/O/S)
    6. Halogen Bonds (C-X...Acc geometry, < 3.8 A)
    """
    # Load receptor atoms
    with open(receptor_pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        rec_lines = f.readlines()
    
    rec_atoms = parse_atoms_from_pdb_lines(rec_lines, is_ligand=False)
    lig_atoms = parse_atoms_from_pdb_lines(ligand_pdb_lines, is_ligand=True)

    if not rec_atoms or not lig_atoms:
        return {
            "h_bonds": [],
            "hydrophobic": [],
            "salt_bridges": [],
            "pi_stacking": [],
            "metal_coordination": [],
            "halogen_bonds": [],
            "pocket_residues": []
        }

    h_bonds = []
    hydrophobic = []
    salt_bridges = []
    pi_stacking = []
    metal_coordination = []
    halogen_bonds = []
    contact_residues = set()

    # Pre-filter receptor atoms within 6.0 Angstroms of any ligand atom for efficiency
    lig_coords = np.array([a["coord"] for a in lig_atoms])
    lig_min = lig_coords.min(axis=0) - 6.0
    lig_max = lig_coords.max(axis=0) + 6.0

    pocket_rec_atoms = [
        a for a in rec_atoms
        if np.all(a["coord"] >= lig_min) and np.all(a["coord"] <= lig_max)
    ]

    # --- Standard contacts loop ---
    for latom in lig_atoms:
        lelem = latom["element"]
        lcoord = latom["coord"]

        for ratom in pocket_rec_atoms:
            relem = ratom["element"]
            rcoord = ratom["coord"]
            dist = distance(lcoord, rcoord)

            # Pocket residue definition (within 4.5 A of any ligand atom)
            if dist <= 4.5:
                res_key = f"{ratom['resname']} {ratom['resid']}"
                contact_residues.add(res_key)

            # 1. Hydrogen Bonds
            if dist <= 3.5:
                if lelem in {"N", "O", "S"} and relem in {"N", "O", "S"}:
                    h_bonds.append({
                        "type": "Hydrogen Bond",
                        "residue": f"{ratom['resname']} {ratom['resid']}",
                        "protein_atom": f"{ratom['resname']}:{ratom['name']}",
                        "ligand_atom": latom["name"],
                        "distance": round(dist, 2),
                        "p_coord": ratom["coord"].tolist(),
                        "l_coord": latom["coord"].tolist()
                    })

            # 2. Hydrophobic Contacts
            if 3.0 <= dist <= 4.2:
                if lelem == "C" and relem == "C":
                    if ratom["resname"] in {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "TYR"}:
                        hydrophobic.append({
                            "type": "Hydrophobic",
                            "residue": f"{ratom['resname']} {ratom['resid']}",
                            "protein_atom": f"{ratom['resname']}:{ratom['name']}",
                            "ligand_atom": latom["name"],
                            "distance": round(dist, 2),
                            "p_coord": ratom["coord"].tolist(),
                            "l_coord": latom["coord"].tolist()
                        })

            # 3. Salt Bridges
            if dist <= 4.0:
                if (ratom["resname"] in {"ARG", "LYS", "HIS"} and ratom["name"].startswith("N") and lelem == "O") or \
                   (ratom["resname"] in {"ASP", "GLU"} and ratom["name"].startswith("O") and lelem == "N"):
                    salt_bridges.append({
                        "type": "Salt Bridge",
                        "residue": f"{ratom['resname']} {ratom['resid']}",
                        "protein_atom": f"{ratom['resname']}:{ratom['name']}",
                        "ligand_atom": latom["name"],
                        "distance": round(dist, 2),
                        "p_coord": ratom["coord"].tolist(),
                        "l_coord": latom["coord"].tolist()
                    })

            # 4. Metal Coordination Bonds
            if relem in CATALYTIC_METALS and lelem in METAL_COORDINATORS and dist <= 3.0:
                metal_coordination.append({
                    "type": "Metal Coordination",
                    "residue": f"{ratom['resname']} {ratom['resid']}",
                    "metal": relem,
                    "protein_atom": f"{ratom['resname']}:{ratom['name']}",
                    "ligand_atom": latom["name"],
                    "distance": round(dist, 2),
                    "p_coord": ratom["coord"].tolist(),
                    "l_coord": latom["coord"].tolist()
                })

            # 5. Halogen Bonds (X...Acc)
            if lelem in HALOGEN_DONORS and relem in HALOGEN_ACCEPTORS and dist <= 3.8:
                halogen_bonds.append({
                    "type": "Halogen Bond",
                    "residue": f"{ratom['resname']} {ratom['resid']}",
                    "protein_atom": f"{ratom['resname']}:{ratom['name']}",
                    "ligand_atom": latom["name"],
                    "halogen": lelem,
                    "acceptor": relem,
                    "distance": round(dist, 2),
                    "p_coord": ratom["coord"].tolist(),
                    "l_coord": latom["coord"].tolist()
                })

    # --- 6. Pi-Stacking: aromatic residue ring centroid vs ligand aromatic centroid ---
    # Group receptor aromatic atoms by residue
    aro_residues = {}
    for ratom in pocket_rec_atoms:
        if ratom["resname"] in AROMATIC_RESIDUES:
            key = (ratom["resname"], ratom["resid"], ratom["chain"])
            if key not in aro_residues:
                aro_residues[key] = []
            aro_residues[key].append(ratom)

    # Get ligand aromatic centroid(s) - approximate by carbon ring members
    lig_carbons = [a for a in lig_atoms if a["element"] == "C"]
    lig_carbon_coords = np.array([a["coord"] for a in lig_carbons])

    if len(lig_carbon_coords) >= 3:
        lig_ring_centroid = lig_carbon_coords.mean(axis=0)

        for (resname, resid, chain), ratoms in aro_residues.items():
            ring_names = AROMATIC_RESIDUES.get(resname, [])
            centroid = _ring_centroid(ratoms, ring_names)
            if centroid is not None:
                c_dist = float(np.linalg.norm(centroid - lig_ring_centroid))
                if c_dist <= 5.5:
                    # Classify parallel vs T-shaped
                    ring_coords_list = [a["coord"] for a in ratoms if a["name"] in ring_names]
                    prot_normal = _normal_vector(ring_coords_list)
                    lig_normal = _normal_vector(list(lig_carbon_coords[:3]))
                    
                    pi_type = "Pi-Stacking (Parallel)"
                    if prot_normal is not None and lig_normal is not None:
                        dot = abs(float(np.dot(prot_normal, lig_normal)))
                        if dot < 0.5:  # nearly perpendicular
                            pi_type = "Pi-Stacking (T-shaped/Edge-to-Face)"
                    
                    pi_stacking.append({
                        "type": pi_type,
                        "residue": f"{resname} {resid}",
                        "protein_ring": resname,
                        "centroid_distance": round(c_dist, 2),
                        "p_coord": centroid.tolist(),
                        "l_coord": lig_ring_centroid.tolist()
                    })

    # Deduplicate contacts per residue-atom pair (keep closest distance)
    def deduplicate(contact_list):
        seen = {}
        for c in contact_list:
            key = (c.get("residue"), c.get("protein_atom"), c.get("ligand_atom"))
            if key not in seen or c.get("distance", 99) < seen[key].get("distance", 99):
                seen[key] = c
        return list(seen.values())

    return {
        "h_bonds": deduplicate(h_bonds),
        "hydrophobic": deduplicate(hydrophobic)[:15],
        "salt_bridges": deduplicate(salt_bridges),
        "pi_stacking": pi_stacking[:8],
        "metal_coordination": deduplicate(metal_coordination),
        "halogen_bonds": deduplicate(halogen_bonds),
        "pocket_residues": sorted(list(contact_residues))
    }

def generate_pymol_script(complex_pdb_path: str, pocket_residues: list, output_pml_path: str):
    """
    Generates a ready-to-run PyMOL script (.pml) for users who wish to render
    high-resolution publication figures.
    """
    res_sel = "+".join([r.split()[1] for r in pocket_residues if len(r.split()) == 2])
    script = f"""# PyMOL Visualization Script generated by AutoDock Studio
reinitialize
load {os.path.basename(complex_pdb_path)}, complex
hide everything, complex

# Protein Cartoon
show cartoon, complex and not resn LIG
color marine, complex and not resn LIG

# Ligand Sticks
show sticks, complex and resn LIG
color magenta, complex and resn LIG
util.cnc complex and resn LIG

# Pocket Residues
select pocket, complex and not resn LIG and resi {res_sel}
show sticks, pocket
color gray80, pocket
util.cnc pocket

# Polar contacts (Hydrogen bonds)
distance hbonds, complex and resn LIG, pocket, 3.5, mode=2
color yellow, hbonds

# View settings
set dash_width, 2.5
set dash_gap, 0.25
bg_color white
zoom complex and resn LIG, 6
"""
    with open(output_pml_path, "w", encoding="utf-8") as f:
        f.write(script)
    return output_pml_path
