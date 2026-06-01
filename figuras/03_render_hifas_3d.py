import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# RUTA DE TU SAMPLE
# ==========================================
sample = "sample_035"
sample_dir = r"sample"
graph_path = os.path.join(sample_dir, "graph")

if not os.path.exists(graph_path):
    graph_path = os.path.join(sample_dir, "graph.json")

# ==========================================
# CARGAR JSON
# ==========================================
with open(graph_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Claves principales:", list(data.keys()))

# ==========================================
# DETECTAR NODOS Y EDGES
# ==========================================
nodes = data["nodes"] if "nodes" in data else []
edges = data["edges"] if "edges" in data else []

print("Total nodos:", len(nodes))
print("Total edges:", len(edges))

# ==========================================
# MAPA ID -> COORDENADAS DE NODOS
# ==========================================
def get_xyz(node):
    if all(k in node for k in ["x", "y", "z"]):
        return np.array([node["x"], node["y"], node["z"]], dtype=float)
    elif "xyz" in node:
        return np.array(node["xyz"], dtype=float)
    elif "coord" in node:
        return np.array(node["coord"], dtype=float)
    elif "position" in node:
        return np.array(node["position"], dtype=float)
    else:
        raise ValueError(f"No pude leer coordenadas del nodo: {node}")

node_coords = {}

for i, node in enumerate(nodes):
    node_id = node["id"] if "id" in node else i
    node_coords[node_id] = get_xyz(node)

# ==========================================
# CONSTRUIR CURVAS REALES DESDE POLYLINE
# ==========================================
polylines = []

for edge in edges:
    # caso ideal: usar la polilínea real
    if "polyline" in edge and len(edge["polyline"]) >= 2:
        pts = np.array(edge["polyline"], dtype=float)
        polylines.append(pts)
    else:
        # fallback: unir parent_node -> child_node con línea recta
        if "parent_node" in edge and "child_node" in edge:
            p = edge["parent_node"]
            c = edge["child_node"]
            if p in node_coords and c in node_coords:
                pts = np.vstack([node_coords[p], node_coords[c]])
                polylines.append(pts)

print("Total trayectorias reales construidas:", len(polylines))

if len(polylines) == 0:
    raise ValueError("No se construyó ninguna trayectoria. Revisa el JSON.")

# ==========================================
# VISUALIZACIÓN 3D
# ==========================================
fig = plt.figure(figsize=(10, 8), facecolor="black")
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("black")

# quitar paneles
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.xaxis.pane.set_edgecolor("black")
ax.yaxis.pane.set_edgecolor("black")
ax.zaxis.pane.set_edgecolor("black")

ax.set_axis_off()

# dibujar polylines reales
for pts in polylines:
    z_mean = np.mean(pts[:, 2])
    color = plt.cm.plasma(z_mean / max(1, np.max(pts[:, 2])))

    ax.plot(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        color=color,
        linewidth=0.8
    )

# ==========================================
# AJUSTE AUTOMÁTICO DE LÍMITES
# ==========================================
all_points = np.vstack(polylines)

mins = all_points.min(axis=0)
maxs = all_points.max(axis=0)

ax.set_xlim(mins[0], maxs[0])
ax.set_ylim(mins[1], maxs[1])
ax.set_zlim(mins[2], maxs[2])

ax.view_init(elev=22, azim=38)
plt.title("Reconstrucción 3D real del "+sample, color="white")
plt.show()