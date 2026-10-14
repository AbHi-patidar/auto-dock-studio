"""
Docking Runner Module for AutoDock Studio
Manages execution of AutoDock Vina binary, output parsing, and merged complex generation.
"""

import os
import subprocess
import re

VINA_BIN = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bin", "vina.exe"))

def run_docking(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    output_pdbqt: str,
    center_x: float,
    center_y: float,
    center_z: float,
    size_x: float = 20.0,
    size_y: float = 20.0,
    size_z: float = 20.0,
    exhaustiveness: int = 8,
    num_modes: int = 9,
    energy_range: float = 3.0,
    cpu: int = 0,
    timeout: float = 7200.0
):
    """
    Executes AutoDock Vina via CLI binary and returns stdout log and parsed poses.
    `timeout` (seconds) guards against a hung Vina process blocking forever.
    """
    if not os.path.exists(VINA_BIN):
        raise FileNotFoundError(f"Vina binary not found at {VINA_BIN}")

    cmd = [
        VINA_BIN,
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--out", output_pdbqt,
        "--center_x", f"{center_x:.3f}",
        "--center_y", f"{center_y:.3f}",
        "--center_z", f"{center_z:.3f}",
        "--size_x", f"{size_x:.3f}",
        "--size_y", f"{size_y:.3f}",
        "--size_z", f"{size_z:.3f}",
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
        "--energy_range", str(energy_range)
    ]
    if cpu > 0:
        cmd.extend(["--cpu", str(cpu)])

    try:
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"AutoDock Vina timed out after {timeout:.0f} seconds; "
            "the docking job was terminated. Try a smaller box or lower exhaustiveness."
        ) from e

    if process.returncode != 0:
        raise RuntimeError(f"AutoDock Vina failed with code {process.returncode}:\n{process.stderr}\n{process.stdout}")

    parsed_poses = parse_vina_output(output_pdbqt)
    return {
        "log": process.stdout,
        "poses": parsed_poses
    }

def parse_vina_output(output_pdbqt: str):
    """
    Parses AutoDock Vina multi-model output file into structured poses.
    """
    if not os.path.exists(output_pdbqt):
        return []

    poses = []
    current_pose = None
    lines = []

    with open(output_pdbqt, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("MODEL"):
                parts = line.split()
                if len(parts) < 2:
                    # Malformed MODEL record (no mode number) — skip instead of crashing
                    current_pose = None
                    continue
                try:
                    mode_num = int(parts[1])
                except ValueError:
                    current_pose = None
                    continue
                current_pose = {
                    "mode": mode_num,
                    "affinity": 0.0,
                    "rmsd_lb": 0.0,
                    "rmsd_ub": 0.0,
                    "pdbqt_lines": [],
                    "pdb_lines": []
                }
            elif line.startswith("REMARK VINA RESULT:") and current_pose is not None:
                parts = line.split()
                # Format: REMARK VINA RESULT:   -8.4      0.000      0.000
                try:
                    current_pose["affinity"] = float(parts[3])
                    current_pose["rmsd_lb"] = float(parts[4])
                    current_pose["rmsd_ub"] = float(parts[5])
                except (IndexError, ValueError):
                    pass
            elif line.startswith("ENDMDL") and current_pose is not None:
                poses.append(current_pose)
                current_pose = None
            elif current_pose is not None:
                current_pose["pdbqt_lines"].append(line)
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    # Standard PDB coordinate line
                    # Convert to HETATM for ligand
                    pdb_line = "HETATM" + line[6:54].ljust(48) + "  1.00  0.00           " + line[76:78].strip() + "\n"
                    current_pose["pdb_lines"].append(pdb_line)

    return poses

def create_complex_pdb(receptor_pdb: str, pose_pdb_lines: list, output_complex_pdb: str):
    """
    Merges receptor PDB and ligand pose lines into a single complex PDB file.
    """
    with open(output_complex_pdb, "w", encoding="utf-8") as out:
        with open(receptor_pdb, "r", encoding="utf-8", errors="ignore") as rec:
            for line in rec:
                if not line.startswith("END"):
                    out.write(line)
        out.write("TER\n")
        for line in pose_pdb_lines:
            out.write(line)
        out.write("END\n")
    return output_complex_pdb
