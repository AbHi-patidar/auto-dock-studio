"""
Interactive 3D Molecular Viewer Module for AutoDock Studio
Renders protein-ligand complexes, docking grid boxes, and molecular interactions using 3Dmol.js.
Supports both dark laboratory mode and pure-white 300 DPI Publication Figure Mode with 1-click snapshot export.
"""

import json

def generate_3dmol_html(
    receptor_pdb_content: str = "",
    ligand_pdb_content: str = "",
    grid_box: dict = None,
    interactions: dict = None,
    publication_mode: bool = False,
    show_surface: bool = False,
    show_labels: bool = True,
    zoom_pocket: bool = True,
    width: str = "100%",
    height: int = 580
) -> str:
    """
    Builds an interactive HTML/JavaScript 3Dmol.js container with high-res publication export.
    """
    grid_json = json.dumps(grid_box or {})
    interactions_json = json.dumps(interactions or {})
    rec_escaped = receptor_pdb_content.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    lig_escaped = ligand_pdb_content.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

    bg_color = "#ffffff" if publication_mode else "#0b1120"
    panel_bg = "rgba(255, 255, 255, 0.95)" if publication_mode else "rgba(15, 23, 42, 0.88)"
    panel_border = "#cbd5e1" if publication_mode else "#334155"
    text_color = "#0f172a" if publication_mode else "#f8fafc"
    container_border = "#e2e8f0" if publication_mode else "#334155"

    hbond_color = "#ca8a04" if publication_mode else "#eab308"
    hydro_color = "#0284c7" if publication_mode else "#06b6d4"

    label_bg = "#ffffff" if publication_mode else "#1e293b"
    label_fg = "#0f172a" if publication_mode else "#fde047"

    is_pub_mode_str = "true" if publication_mode else "false"
    show_surface_str = "true" if show_surface else "false"
    show_labels_str = "true" if show_labels else "false"
    zoom_pocket_str = "true" if zoom_pocket else "false"

    html_template = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
  <style>
    body {
      margin: 0;
      padding: 0;
      background: __BG_COLOR__;
      color: __TEXT_COLOR__;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
      overflow: hidden;
    }
    #container {
      position: relative;
      width: __WIDTH__;
      height: __HEIGHT__px;
      border-radius: 6px;
      overflow: hidden;
      box-shadow: 0 4px 12px -2px rgba(0,0,0,0.3);
      border: 1px solid __CONTAINER_BORDER__;
      background: __BG_COLOR__;
    }
    #gldiv {
      width: 100%;
      height: 100%;
    }
    .controls-overlay {
      position: absolute;
      top: 10px;
      left: 10px;
      right: 10px;
      background: __PANEL_BG__;
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      border: 1px solid __PANEL_BORDER__;
      padding: 7px 12px;
      border-radius: 6px;
      font-size: 11.5px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      z-index: 10;
    }
    .legend-group {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: 5px;
      font-weight: 500;
      font-size: 11px;
      letter-spacing: 0.1px;
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 2px;
      display: inline-block;
    }
    .actions-group {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .btn-action {
      border: none;
      border-radius: 4px;
      padding: 5px 11px;
      cursor: pointer;
      font-size: 11px;
      font-weight: 600;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }
    .btn-reset {
      background: #1e293b;
      color: __TEXT_COLOR__;
      border: 1px solid __PANEL_BORDER__;
    }
    .btn-reset:hover {
      background: #334155;
    }
    .btn-snapshot {
      background: #1d4ed8;
      color: white;
      border: 1px solid #2563eb;
    }
    .btn-snapshot:hover {
      background: #1e40af;
      border-color: #3b82f6;
    }
  </style>
</head>
<body>
  <div id="container">
    <div class="controls-overlay">
      <div class="legend-group">
        <div class="legend-item"><span class="dot" style="background:#38bdf8;"></span> Ribbon</div>
        <div class="legend-item"><span class="dot" style="background:#ec4899;"></span> Ligand</div>
        <div class="legend-item"><span class="dot" style="background:__HBOND_COLOR__;"></span> H-Bond</div>
        <div class="legend-item"><span class="dot" style="background:__HYDRO_COLOR__;"></span> Hydrophobic</div>
        <div class="legend-item"><span class="dot" style="background:#a855f7;"></span> π-Stack</div>
        <div class="legend-item"><span class="dot" style="background:#f97316;"></span> Metal / Salt</div>
      </div>
      <div class="actions-group">
        <button class="btn-action btn-reset" onclick="resetView()">↺ Center</button>
        <button class="btn-action btn-snapshot" onclick="downloadPublicationSnapshot()">📸 300 DPI Export</button>
      </div>
    </div>
    <div id="gldiv"></div>
  </div>

  <script>
    let viewer = null;
    const recPdb = `__REC_PDB__`;
    const ligPdb = `__LIG_PDB__`;
    const gridBox = __GRID_BOX__;
    const interactions = __INTERACTIONS__;
    const isPubMode = __IS_PUB_MODE__;
    const showSurface = __SHOW_SURFACE__;
    const showLabels = __SHOW_LABELS__;
    const zoomPocket = __ZOOM_POCKET__;

    window.onload = function() {
      const element = document.getElementById('gldiv');
      const config = { 
        backgroundColor: '__BG_COLOR__',
        antialias: true,
        preserveDrawingBuffer: true
      };
      viewer = $3Dmol.createViewer(element, config);

      if (viewer.setCameraParameters) {
        viewer.setCameraParameters({ fov: 38 });
      }

      // 1. Add Receptor
      if (recPdb && recPdb.trim().length > 0) {
        viewer.addModel(recPdb, "pdb");
        viewer.setStyle({model: 0}, {
          cartoon: { color: isPubMode ? '#0284c7' : 'spectrum', opacity: 0.90 }
        });
        if (showSurface) {
          viewer.addSurface($3Dmol.SurfaceType.MS, {
            opacity: isPubMode ? 0.20 : 0.25,
            color: isPubMode ? '#94a3b8' : '#38bdf8'
          }, {model: 0});
        }
      }

      // 2. Add Ligand
      if (ligPdb && ligPdb.trim().length > 0) {
        viewer.addModel(ligPdb, "pdb");
        viewer.setStyle({model: 1}, {
          stick: { colorscheme: 'magentaCarbon', radius: 0.28 }
        });
      }

      // 3. Highlight Pocket Sidechains (within 4.5 A of ligand)
      if (ligPdb && recPdb && interactions && interactions.pocket_residues) {
        interactions.pocket_residues.forEach(r => {
          const parts = r.split(' ');
          if (parts.length === 2) {
            const resi = parseInt(parts[1]);
            viewer.setStyle({model: 0, resi: resi}, {
              cartoon: { color: isPubMode ? '#0284c7' : 'spectrum', opacity: 0.90 },
              stick: { colorscheme: isPubMode ? 'grayCarbon' : 'default', radius: 0.16 }
            });
          }
        });
      }

      // 4. Render Non-Covalent Interactions
      // 4a. Hydrogen Bonds
      if (interactions && interactions.h_bonds) {
        interactions.h_bonds.forEach(hb => {
          if (hb.p_coord && hb.l_coord) {
            viewer.addCylinder({
              start: { x: hb.p_coord[0], y: hb.p_coord[1], z: hb.p_coord[2] },
              end: { x: hb.l_coord[0], y: hb.l_coord[1], z: hb.l_coord[2] },
              radius: 0.09,
              color: '__HBOND_COLOR__',
              dashed: true,
              fromCap: 1,
              toCap: 1
            });
            if (showLabels) {
              viewer.addLabel(`${hb.residue} (${hb.distance}Å)`, {
                position: { x: hb.p_coord[0], y: hb.p_coord[1], z: hb.p_coord[2] },
                backgroundColor: '__LABEL_BG__',
                backgroundOpacity: 0.85,
                fontColor: '__LABEL_FG__',
                fontSize: 11,
                borderThickness: 1,
                borderColor: '#cbd5e1'
              });
            }
          }
        });
      }

      // 4b. Hydrophobic Contacts
      if (interactions && interactions.hydrophobic) {
        interactions.hydrophobic.forEach(hc => {
          if (hc.p_coord && hc.l_coord) {
            viewer.addCylinder({
              start: { x: hc.p_coord[0], y: hc.p_coord[1], z: hc.p_coord[2] },
              end: { x: hc.l_coord[0], y: hc.l_coord[1], z: hc.l_coord[2] },
              radius: 0.07,
              color: '__HYDRO_COLOR__',
              dashed: true
            });
          }
        });
      }

      // 4c. Salt Bridges
      if (interactions && interactions.salt_bridges) {
        interactions.salt_bridges.forEach(sb => {
          if (sb.p_coord && sb.l_coord) {
            viewer.addCylinder({
              start: { x: sb.p_coord[0], y: sb.p_coord[1], z: sb.p_coord[2] },
              end: { x: sb.l_coord[0], y: sb.l_coord[1], z: sb.l_coord[2] },
              radius: 0.11,
              color: '#d946ef',
              dashed: true
            });
            if (showLabels) {
              viewer.addLabel(`Ionic: ${sb.residue}`, {
                position: { x: sb.p_coord[0], y: sb.p_coord[1], z: sb.p_coord[2] },
                backgroundColor: '__LABEL_BG__',
                backgroundOpacity: 0.85,
                fontColor: '#d946ef',
                fontSize: 10
              });
            }
          }
        });
      }

      // 4d. Pi-Stacking
      if (interactions && interactions.pi_stacking) {
        interactions.pi_stacking.forEach(pi => {
          if (pi.p_coord && pi.l_coord) {
            viewer.addCylinder({
              start: { x: pi.p_coord[0], y: pi.p_coord[1], z: pi.p_coord[2] },
              end: { x: pi.l_coord[0], y: pi.l_coord[1], z: pi.l_coord[2] },
              radius: 0.10,
              color: '#a855f7',
              dashed: true
            });
            if (showLabels) {
              viewer.addLabel(`π-π: ${pi.residue} (${pi.centroid_distance}Å)`, {
                position: { x: pi.p_coord[0], y: pi.p_coord[1], z: pi.p_coord[2] },
                backgroundColor: '__LABEL_BG__',
                backgroundOpacity: 0.85,
                fontColor: '#a855f7',
                fontSize: 10
              });
            }
          }
        });
      }

      // 4e. Metal Coordination
      if (interactions && interactions.metal_coordination) {
        interactions.metal_coordination.forEach(mc => {
          if (mc.p_coord && mc.l_coord) {
            viewer.addCylinder({
              start: { x: mc.p_coord[0], y: mc.p_coord[1], z: mc.p_coord[2] },
              end: { x: mc.l_coord[0], y: mc.l_coord[1], z: mc.l_coord[2] },
              radius: 0.12,
              color: '#f97316',
              dashed: true
            });
            if (showLabels) {
              viewer.addLabel(`Metal Coord: ${mc.metal} (${mc.distance}Å)`, {
                position: { x: mc.p_coord[0], y: mc.p_coord[1], z: mc.p_coord[2] },
                backgroundColor: '__LABEL_BG__',
                backgroundOpacity: 0.85,
                fontColor: '#f97316',
                fontSize: 10
              });
            }
          }
        });
      }

      // 4f. Halogen Bonds
      if (interactions && interactions.halogen_bonds) {
        interactions.halogen_bonds.forEach(hx => {
          if (hx.p_coord && hx.l_coord) {
            viewer.addCylinder({
              start: { x: hx.p_coord[0], y: hx.p_coord[1], z: hx.p_coord[2] },
              end: { x: hx.l_coord[0], y: hx.l_coord[1], z: hx.l_coord[2] },
              radius: 0.08,
              color: '#10b981',
              dashed: true
            });
          }
        });
      }

      // 5. Render Grid Box (Search Space)
      if (gridBox && gridBox.center && gridBox.size) {
        const cx = gridBox.center[0], cy = gridBox.center[1], cz = gridBox.center[2];
        const sx = gridBox.size[0] / 2, sy = gridBox.size[1] / 2, sz = gridBox.size[2] / 2;

        const corners = [
          [cx - sx, cy - sy, cz - sz], [cx + sx, cy - sy, cz - sz],
          [cx + sx, cy + sy, cz - sz], [cx - sx, cy + sy, cz - sz],
          [cx - sx, cy - sy, cz + sz], [cx + sx, cy - sy, cz + sz],
          [cx + sx, cy + sy, cz + sz], [cx - sx, cy + sy, cz + sz]
        ];
        const edges = [
          [0,1],[1,2],[2,3],[3,0],
          [4,5],[5,6],[6,7],[7,4],
          [0,4],[1,5],[2,6],[3,7]
        ];
        edges.forEach(e => {
          viewer.addLine({
            start: { x: corners[e[0]][0], y: corners[e[0]][1], z: corners[e[0]][2] },
            end: { x: corners[e[1]][0], y: corners[e[1]][1], z: corners[e[1]][2] },
            color: isPubMode ? '#16a34a' : '#22c55e',
            wireframe: true
          });
        });
      }

      // Camera Positioning
      if (ligPdb && ligPdb.trim().length > 0 && zoomPocket) {
        viewer.zoomTo({model: 1}, 700);
      } else {
        viewer.zoomTo();
      }
      viewer.render();
    };

    function resetView() {
      if (viewer) {
        if (ligPdb && ligPdb.trim().length > 0 && zoomPocket) {
          viewer.zoomTo({model: 1}, 700);
        } else {
          viewer.zoomTo();
        }
      }
    }

    function downloadPublicationSnapshot() {
      if (!viewer) return;
      const btn = document.querySelector('.btn-snapshot');
      const origText = btn ? btn.innerHTML : '';
      if (btn) btn.innerHTML = '⏳ Rendering 300 DPI...';

      const container = document.getElementById('container');
      const origContainerW = container.style.width;
      const origContainerH = container.style.height;

      // High-resolution publication dimensions (3200 x 2400 px, 300 DPI print-ready)
      const targetW = 3200;
      const targetH = 2400;

      container.style.width = targetW + 'px';
      container.style.height = targetH + 'px';
      if (viewer.resize) viewer.resize();
      viewer.render();

      setTimeout(function() {
        try {
          const uri = viewer.pngURI();
          const link = document.createElement('a');
          link.download = 'Figure_1_3D_Binding_Pose_Publication_300DPI.png';
          link.href = uri;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
        } catch(err) {
          console.error("Snapshot generation failed:", err);
        } finally {
          container.style.width = origContainerW;
          container.style.height = origContainerH;
          if (viewer.resize) viewer.resize();
          viewer.render();
          if (btn) btn.innerHTML = origText;
        }
      }, 180);
    }
  </script>
</body>
</html>"""

    html = (
        html_template
        .replace("__BG_COLOR__", bg_color)
        .replace("__TEXT_COLOR__", text_color)
        .replace("__WIDTH__", str(width))
        .replace("__HEIGHT__", str(height))
        .replace("__CONTAINER_BORDER__", container_border)
        .replace("__PANEL_BG__", panel_bg)
        .replace("__PANEL_BORDER__", panel_border)
        .replace("__HBOND_COLOR__", hbond_color)
        .replace("__HYDRO_COLOR__", hydro_color)
        .replace("__LABEL_BG__", label_bg)
        .replace("__LABEL_FG__", label_fg)
        .replace("__REC_PDB__", rec_escaped)
        .replace("__LIG_PDB__", lig_escaped)
        .replace("__GRID_BOX__", grid_json)
        .replace("__INTERACTIONS__", interactions_json)
        .replace("__IS_PUB_MODE__", is_pub_mode_str)
        .replace("__SHOW_SURFACE__", show_surface_str)
        .replace("__SHOW_LABELS__", show_labels_str)
        .replace("__ZOOM_POCKET__", zoom_pocket_str)
    )
    return html
