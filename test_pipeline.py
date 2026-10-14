"""
Pytest test suite for AutoDock Studio (converted from a demo script).

Fast offline unit tests cover the pure utility functions with small synthetic
fixtures (no network, no RDKit/Meeko, no Vina binary required). The original
end-to-end pipeline test is preserved as a skipped-by-default integration test.

Run:
    python -m pytest test_pipeline.py -v            # offline unit tests only
    python -m pytest test_pipeline.py -v -m integration   # slow integration test
"""

import os
import sys
import math

import pytest

# Setup paths
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curr_dir)

from utils.protein_prep import (
    fetch_pdb_from_rcsb,
    clean_protein_pdb,
    receptor_pdb_to_pdbqt,
    compute_blind_docking_box,
)
from utils.docking_runner import parse_vina_output
from utils.validation import calculate_heavy_atom_rmsd
from utils.interpretation import calculate_inhibition_constant


# --------------------------------------------------------------------------
# Synthetic fixtures
# --------------------------------------------------------------------------

def _pdb_atom(serial, name, resname, chain, resseq, x, y, z, element, record="ATOM"):
    """Builds one fixed-width PDB ATOM/HETATM record (standard column layout)."""
    return (
        f"{record:<6s}{serial:5d} {name:<4s} {resname:>3s} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {element:>2s}\n"
    )


@pytest.fixture
def tiny_pdb(tmp_path):
    """A 10-atom synthetic protein + one 3-atom hetero ligand."""
    lines = []
    serial = 1
    for i, (name, res, elem) in enumerate([
        ("N", "ALA", "N"), ("CA", "ALA", "C"), ("C", "ALA", "C"), ("O", "ALA", "O"),
        ("N", "GLY", "N"), ("CA", "GLY", "C"), ("C", "GLY", "C"), ("O", "GLY", "O"),
        ("N", "SER", "N"), ("CA", "SER", "C"),
    ]):
        lines.append(_pdb_atom(serial, name, res, "A", i // 4 + 1, 1.0 + i, 2.0, 3.0, elem))
        serial += 1
    # Co-crystallized ligand (3 atoms, >= 6 lines not required for detection check here)
    for j, (name, elem) in enumerate([("C1", "C"), ("N1", "N"), ("O1", "O")]):
        lines.append(_pdb_atom(serial, name, "LIG", "A", 99, 10.0 + j, 20.0, 30.0, elem, record="HETATM"))
        serial += 1
    lines.append("END\n")
    p = tmp_path / "tiny.pdb"
    p.write_text("".join(lines), encoding="utf-8")
    return str(p)


@pytest.fixture
def garbage_pdb(tmp_path):
    p = tmp_path / "garbage.pdb"
    p.write_text("THIS IS NOT A PDB FILE\nHELLO WORLD\n", encoding="utf-8")
    return str(p)


@pytest.fixture
def vina_pdbqt(tmp_path):
    """Two valid MODELs plus one malformed bare 'MODEL' line (must not crash)."""
    content = (
        "MODEL        1\n"
        "REMARK VINA RESULT:   -8.400      0.000      0.000\n"
        "HETATM    1  C1  LIG A   1       1.000   2.000   3.000  1.00  0.00    +0.100 C\n"
        "ENDMDL\n"
        "MODEL\n"
        "REMARK VINA RESULT:   -7.200      1.000      2.000\n"
        "HETATM    1  C1  LIG A   1       1.500   2.500   3.500  1.00  0.00    +0.100 C\n"
        "ENDMDL\n"
        "MODEL        3\n"
        "REMARK VINA RESULT:   -6.000      2.000      3.000\n"
        "HETATM    1  C1  LIG A   1       2.000   3.000   4.000  1.00  0.00    +0.100 C\n"
        "ENDMDL\n"
    )
    p = tmp_path / "docked.pdbqt"
    p.write_text(content, encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------------
# protein_prep
# --------------------------------------------------------------------------

def test_clean_protein_pdb_basic(tiny_pdb, tmp_path):
    out = str(tmp_path / "clean.pdb")
    meta = clean_protein_pdb(tiny_pdb, out)
    assert meta["num_atoms"] == 10
    assert len(meta["center"]) == 3
    assert len(meta["dimensions"]) == 3
    assert os.path.exists(out)


def test_clean_protein_pdb_empty_raises(garbage_pdb, tmp_path):
    """Empty/garbage input must fail fast instead of producing a [0,0,0] center."""
    with pytest.raises(ValueError, match="No protein ATOM records"):
        clean_protein_pdb(garbage_pdb, str(tmp_path / "clean.pdb"))


def test_receptor_pdb_to_pdbqt_charges(tiny_pdb, tmp_path):
    clean = str(tmp_path / "clean.pdb")
    clean_protein_pdb(tiny_pdb, clean)
    out = str(tmp_path / "receptor.pdbqt")
    receptor_pdb_to_pdbqt(clean, out)
    assert os.path.exists(out)
    with open(out, encoding="utf-8") as f:
        lines = [l for l in f if l.startswith(("ATOM", "HETATM"))]
    assert lines, "PDBQT output contains no atom records"
    # Heuristic charge model: backbone N should carry -0.350
    n_lines = [l for l in lines if l[12:16].strip() == "N"]
    assert n_lines
    assert "-0.350" in n_lines[0]


def test_fetch_pdb_rejects_bad_id(tmp_path):
    """PDB ID validation must fail fast on garbage, without touching the network."""
    with pytest.raises(ValueError, match="Invalid PDB ID"):
        fetch_pdb_from_rcsb("not-an-id", str(tmp_path / "x.pdb"))
    with pytest.raises(ValueError, match="Invalid PDB ID"):
        fetch_pdb_from_rcsb("xyz", str(tmp_path / "x.pdb"))


def test_blind_docking_box_clamp_warns(tmp_path, caplog):
    """A protein spanning >90 A must log a warning when the box is clamped."""
    # Fabricate a huge protein: two atoms 200 A apart
    lines = [
        _pdb_atom(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
        _pdb_atom(2, "CA", "ALA", "A", 2, 200.0, 0.0, 0.0, "C"),
        "END\n",
    ]
    big = tmp_path / "big.pdb"
    big.write_text("".join(lines), encoding="utf-8")
    import logging
    with caplog.at_level(logging.WARNING, logger="autodock_studio"):
        box = compute_blind_docking_box(str(big))
    assert box["size"][0] == 90.0
    assert any("clamped" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# docking_runner (parser only — no Vina binary needed)
# --------------------------------------------------------------------------

def test_parse_vina_output_modes(vina_pdbqt):
    poses = parse_vina_output(vina_pdbqt)
    assert len(poses) == 2  # malformed bare MODEL record is skipped
    assert [p["mode"] for p in poses] == [1, 3]
    assert poses[0]["affinity"] == pytest.approx(-8.4)
    assert len(poses[0]["pdb_lines"]) == 1


def test_parse_vina_output_missing_file(tmp_path):
    assert parse_vina_output(str(tmp_path / "does_not_exist.pdbqt")) == []


# --------------------------------------------------------------------------
# interpretation (Ki math)
# --------------------------------------------------------------------------

def test_ki_math_favorable():
    ki_val, ki_str = calculate_inhibition_constant(-8.0)
    assert ki_val < 1e-3  # sub-millimolar
    assert ki_str.endswith("µM")
    assert "non-binding" not in ki_str


def test_ki_math_unfavorable_flagged():
    """The old sign bug reported +2.0 kcal/mol as ~34 mM; the true Ki is ~29 M."""
    ki_val, ki_str = calculate_inhibition_constant(2.0)
    assert ki_val == pytest.approx(math.exp(2.0 / (0.001987204 * 298.15)), rel=1e-6)
    assert ki_val > 1.0  # molar scale, not millimolar
    assert "non-binding" in ki_str


def test_ki_ordering():
    ki_neg8, _ = calculate_inhibition_constant(-8.0)
    ki_neg2, _ = calculate_inhibition_constant(-2.0)
    ki_pos2, _ = calculate_inhibition_constant(2.0)
    assert ki_neg8 < ki_neg2 < ki_pos2


# --------------------------------------------------------------------------
# validation (RMSD) — synthetic Vina-style lines
# --------------------------------------------------------------------------

def test_rmsd_identical_conformers(vina_pdbqt):
    """Docked pose identical to crystal coordinates must give RMSD ~ 0."""
    poses = parse_vina_output(vina_pdbqt)
    crystal = [
        _pdb_atom(1, "C1", "LIG", "A", 1, 1.0, 2.0, 3.0, "C"),
        "END\n",
    ]
    rmsd = calculate_heavy_atom_rmsd(crystal, poses[0]["pdb_lines"])
    assert isinstance(rmsd, float)
    assert rmsd == pytest.approx(0.0, abs=1e-6)


def test_rmsd_count_mismatch_warns(vina_pdbqt, caplog):
    """Differing atom counts must warn instead of silently truncating."""
    import logging
    poses = parse_vina_output(vina_pdbqt)
    crystal = [
        _pdb_atom(1, "C1", "LIG", "A", 1, 1.0, 2.0, 3.0, "C"),
        _pdb_atom(2, "N1", "LIG", "A", 1, 4.0, 5.0, 6.0, "N"),
        "END\n",
    ]
    with caplog.at_level(logging.WARNING, logger="autodock_studio"):
        rmsd = calculate_heavy_atom_rmsd(crystal, poses[0]["pdb_lines"])
    assert isinstance(rmsd, float)
    assert any("mismatch" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# Integration test (skipped by default — needs network, RDKit/Meeko, Windows Vina)
# --------------------------------------------------------------------------

@pytest.mark.skip(reason="requires network (RCSB/Cactus), RDKit+Meeko, and the Windows Vina binary")
def test_end_to_end_pipeline_integration(tmp_path):
    """Original demo flow, preserved for manual runs on a Windows workstation."""
    from utils.ligand_prep import prepare_ligand_pdbqt
    from utils.docking_runner import run_docking, create_complex_pdb
    from utils.interaction_profiler import profile_interactions, generate_pymol_script

    data_dir = str(tmp_path)
    raw_pdb = os.path.join(data_dir, "1hsg_raw.pdb")
    fetch_pdb_from_rcsb("1HSG", raw_pdb)
    assert os.path.getsize(raw_pdb) > 0

    clean_pdb = os.path.join(data_dir, "1hsg_clean.pdb")
    meta = clean_protein_pdb(raw_pdb, clean_pdb)
    assert meta["num_atoms"] > 0

    receptor_pdbqt = os.path.join(data_dir, "1hsg_receptor.pdbqt")
    receptor_pdb_to_pdbqt(clean_pdb, receptor_pdbqt)
    assert os.path.getsize(receptor_pdbqt) > 0

    smiles = "CC(C)(C)NC(=O)C1CN(CCN1CC(CC2=CC=CC=C2)C(=O)NC(CC3=CC=CC=C3)C(=O)O)CC4=CN=CC=C4"
    ligand_pdbqt = os.path.join(data_dir, "indinavir.pdbqt")
    prepare_ligand_pdbqt(None, ligand_pdbqt, is_smiles=True, smiles_str=smiles)
    assert os.path.getsize(ligand_pdbqt) > 0

    out_pdbqt = os.path.join(data_dir, "docked_test.pdbqt")
    center = meta["detected_ligands"]["MK1"]["center"] if "MK1" in meta["detected_ligands"] else [16.0, 25.0, 4.0]
    result = run_docking(
        receptor_pdbqt=receptor_pdbqt,
        ligand_pdbqt=ligand_pdbqt,
        output_pdbqt=out_pdbqt,
        center_x=center[0], center_y=center[1], center_z=center[2],
        size_x=20.0, size_y=20.0, size_z=20.0,
        exhaustiveness=1, num_modes=2,
    )
    assert len(result["poses"]) > 0, "Vina returned no poses"

    best_pose = result["poses"][0]
    interactions = profile_interactions(clean_pdb, best_pose["pdb_lines"])
    assert isinstance(interactions["h_bonds"], list)

    complex_pdb = os.path.join(data_dir, "complex_test.pdb")
    create_complex_pdb(clean_pdb, best_pose["pdb_lines"], complex_pdb)
    pml_path = os.path.join(data_dir, "test_session.pml")
    generate_pymol_script(complex_pdb, interactions["pocket_residues"], pml_path)
    assert os.path.exists(pml_path)
