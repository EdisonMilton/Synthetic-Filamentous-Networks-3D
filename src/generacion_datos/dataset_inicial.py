import os
import math
import json
import csv
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Set, Optional
from scipy.ndimage import (
    gaussian_filter,
    label,
    generate_binary_structure,
    convolve,
    binary_opening,
    binary_closing,
    binary_fill_holes,
    distance_transform_edt,
)

import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    label,
    generate_binary_structure,
    convolve,
    binary_opening,
    binary_closing,
    binary_fill_holes,
)

# =========================================================
# DEPENDENCIAS OPCIONALES / REQUERIDAS PARA EVALUACIÓN
# =========================================================
SKIMAGE_OK = False
FRANGI_OK = False
SKELETON_MODE = None

try:
    from skimage.morphology import skeletonize_3d
    SKIMAGE_OK = True
    SKELETON_MODE = "skeletonize_3d"
except Exception:
    SKIMAGE_OK = False
    SKELETON_MODE = None

try:
    from skimage.filters import frangi
    FRANGI_OK = True
except Exception:
    FRANGI_OK = False


# =========================================================
# CONFIGURACIÓN GLOBAL
# =========================================================
DATASET_DIR = r"samples"

MODE = "generate"   # "generate", "evaluate", "all"

NUM_SAMPLES = 50
VOL_SHAPE = (512, 512, 256)   # (x, y, z)

BASE_SEED = 12345
MAX_ATTEMPTS_PER_SAMPLE = 80

# ---------- Estructura global ----------
NUM_TREES_MIN = 3
NUM_TREES_MAX = 5
TREE_SEED_SEPARATION = 75.0

# ---------- Crecimiento ----------
MAX_BRANCH_DEPTH = 3
BRANCH_PROB = 0.035

STEPS_MIN = 180
STEPS_MAX = 280


STEP_SIZE_MIN = 1.2
STEP_SIZE_MAX = 2.8

# ---------- Geometría ----------
FILAMENT_RADIUS_MIN = 2.0
FILAMENT_RADIUS_MAX = 3.2
RADIUS_JITTER = 0.10
MASK_RADIUS_PAD = 1.25
MARGIN = 20

# ---------- Dirección ----------
#DIRECTION_PERSISTENCE = 0.84
#DIRECTION_NOISE = 0.18
#MAX_TURN_ANGLE_DEG = 24.0

DIRECTION_PERSISTENCE = 0.84
DIRECTION_NOISE = 0.18
MAX_TURN_ANGLE_DEG = 24.0

# ---------- Colisiones / separación ----------
SKELETON_CLEARANCE_RADIUS = 1.2
TUBE_CLEARANCE_RADIUS = 1.6
IGNORE_START_DISTANCE = 6.0
ALLOW_TOUCH_PROB = 0.06
MIN_NODE_SEPARATION = 2.5

# ---------- Reglas extra ----------
MIN_BRANCH_ADVANCE_STEPS = 18
MIN_PARENT_SEGMENT_FOR_BRANCH = 16.0
MIN_ENDPOINT_DISTANCE_TO_EXISTING_NODE = 2.5

# ---------- Rasterizado ----------
LINE_SAMPLES_PER_VOXEL = 4.0

# ---------- Volumen sintético ----------
GAUSSIAN_SIGMA_XY = 1.0
GAUSSIAN_SIGMA_Z = 1.35
BACKGROUND_MEAN = 0.055
BACKGROUND_STD = 0.015
NOISE_STD = 0.020
DEPTH_ATTENUATION = 0.12
POISSON_LAMBDA_SCALE = 28.0
FILAMENT_INTENSITY_JITTER = 0.18

# ---------- Validación dura ----------
MIN_MASK_VOXELS = 5000
MAX_MASK_FRACTION = 0.12
MIN_SKELETON_VOXELS = 450
MAX_CONNECTED_COMPONENTS = 12
MIN_EDGES = 18
MIN_ENDPOINTS = 2
MAX_BRANCHPOINTS_FACTOR = 0.55

# ---------- Evaluación ----------
VOXEL_SPACING = (1.0, 1.0, 1.0)

# ---------- Otsu global ----------
OTSU_MIN_OBJECT_SIZE = 64
USE_FILL_HOLES_OTSU = False

# ---------- Segmentación simple ----------
SIMPLE_GAUSSIAN_SIGMA = (1.0, 1.0, 1.2)
SIMPLE_THRESHOLD_OFFSET = 0.00
SIMPLE_MIN_OBJECT_SIZE = 96
USE_FILL_HOLES_SIMPLE = False

# ---------- Pipeline propuesto ----------
PROPOSED_HESSIAN_SCALES = [1.0, 2.0, 3.0]
PROPOSED_GAUSSIAN_PRE = (0.8, 0.8, 1.0)
PROPOSED_VESSELNESS_ALPHA = 0.5
PROPOSED_VESSELNESS_BETA = 0.5
PROPOSED_VESSELNESS_C = 15.0
PROPOSED_MIN_OBJECT_SIZE = 96
PROPOSED_CLOSE_ITERS = 1
USE_FILL_HOLES_PROPOSED = False

# ---------- Skeleton graph simplificado ----------
PRUNE_MIN_BRANCH_LENGTH = 3.0
DECIMALS_SUMMARY = 6


# =========================================================
# DATACLASSES
# =========================================================
@dataclass
class Nodo3D:
    id: int
    xyz: List[float]
    tipo: str       # root | endpoint | branchpoint | internal
    tree_id: int


@dataclass
class Arista3D:
    id: int
    tree_id: int
    parent_node: int
    child_node: int
    depth: int
    radius_mean: float
    length: float
    polyline: List[List[float]]


# =========================================================
# UTILIDADES
# =========================================================
def asegurar_directorio(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def guardar_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def limpiar_archivos_sample(sample_dir: str) -> None:
    if not os.path.isdir(sample_dir):
        return
    for nombre in os.listdir(sample_dir):
        if nombre.endswith(".npy") or nombre.endswith(".json") or nombre.endswith(".csv"):
            ruta = os.path.join(sample_dir, nombre)
            if os.path.isfile(ruta):
                try:
                    os.remove(ruta)
                except Exception:
                    pass


def normalizar(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def angulo_entre_grados(a: np.ndarray, b: np.ndarray) -> float:
    a = normalizar(a)
    b = normalizar(b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(c))


def limitar_giro(prev_dir: np.ndarray, new_dir: np.ndarray, max_angle_deg: float) -> np.ndarray:
    ang = angulo_entre_grados(prev_dir, new_dir)
    if ang <= max_angle_deg:
        return normalizar(new_dir)
    t = max_angle_deg / max(ang, 1e-6)
    mezcla = (1.0 - t) * prev_dir + t * new_dir
    return normalizar(mezcla)


def dentro_volumen(p: np.ndarray, shape: Tuple[int, int, int], margin: int = MARGIN) -> bool:
    x, y, z = p
    return (
        margin <= x < shape[0] - margin and
        margin <= y < shape[1] - margin and
        margin <= z < shape[2] - margin
    )


def distancia_punto_punto(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def clip_idx(idx: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    out = idx.copy()
    out[0] = np.clip(out[0], 0, shape[0] - 1)
    out[1] = np.clip(out[1], 0, shape[1] - 1)
    out[2] = np.clip(out[2], 0, shape[2] - 1)
    return out


def clamp_bbox(x0, x1, y0, y1, z0, z1, shape):
    x0 = max(0, x0)
    y0 = max(0, y0)
    z0 = max(0, z0)
    x1 = min(shape[0], x1)
    y1 = min(shape[1], y1)
    z1 = min(shape[2], z1)
    return x0, x1, y0, y1, z0, z1


def bbox_segmento(p0: np.ndarray, p1: np.ndarray, radius: float, shape, pad=1.25):
    extra = radius + pad
    x0 = int(math.floor(min(p0[0], p1[0]) - extra))
    x1 = int(math.ceil(max(p0[0], p1[0]) + extra)) + 1
    y0 = int(math.floor(min(p0[1], p1[1]) - extra))
    y1 = int(math.ceil(max(p0[1], p1[1]) + extra)) + 1
    z0 = int(math.floor(min(p0[2], p1[2]) - extra))
    z1 = int(math.ceil(max(p0[2], p1[2]) + extra)) + 1
    return clamp_bbox(x0, x1, y0, y1, z0, z1, shape)


def distancia_punto_segmento_local(X, Y, Z, p0, p1):
    vx = float(p1[0] - p0[0])
    vy = float(p1[1] - p0[1])
    vz = float(p1[2] - p0[2])
    seg_len2 = vx * vx + vy * vy + vz * vz

    if seg_len2 < 1e-8:
        dx = X - float(p0[0])
        dy = Y - float(p0[1])
        dz = Z - float(p0[2])
        return np.sqrt(dx * dx + dy * dy + dz * dz)

    wx = X - float(p0[0])
    wy = Y - float(p0[1])
    wz = Z - float(p0[2])

    t = (wx * vx + wy * vy + wz * vz) / seg_len2
    t = np.clip(t, 0.0, 1.0)

    projx = float(p0[0]) + t * vx
    projy = float(p0[1]) + t * vy
    projz = float(p0[2]) + t * vz

    dx = X - projx
    dy = Y - projy
    dz = Z - projz
    return np.sqrt(dx * dx + dy * dy + dz * dz)


# =========================================================
# MORFOLOGÍA
# =========================================================
def remove_small_objects_3d(mask: np.ndarray, min_size: int) -> np.ndarray:
    mask = mask.astype(bool)
    if min_size <= 1:
        return mask

    struct = generate_binary_structure(rank=3, connectivity=3)
    labeled, num = label(mask, structure=struct)
    if num == 0:
        return mask

    counts = np.bincount(labeled.ravel())
    keep = counts >= min_size
    keep[0] = False
    return keep[labeled]


def keep_largest_components(mask: np.ndarray, max_components: int = 12) -> np.ndarray:
    mask = mask.astype(bool)
    struct = generate_binary_structure(rank=3, connectivity=3)
    labeled, num = label(mask, structure=struct)
    if num == 0:
        return mask

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    ids = np.argsort(counts)[::-1]
    ids = ids[counts[ids] > 0][:max_components]
    return np.isin(labeled, ids)


# =========================================================
# LÍNEA 3D SUPER-COVER
# =========================================================
def linea_voxeles_3d(p0: np.ndarray, p1: np.ndarray, shape) -> np.ndarray:
    dist = float(np.linalg.norm(p1 - p0))
    n = max(2, int(math.ceil(dist * LINE_SAMPLES_PER_VOXEL)))

    pts = np.linspace(p0, p1, n, dtype=np.float32)
    idx = np.round(pts).astype(np.int32)

    out = []
    prev = clip_idx(idx[0], shape)
    out.append(prev.copy())

    for cur in idx[1:]:
        cur = clip_idx(cur, shape)
        delta = cur - prev
        pasos = int(np.max(np.abs(delta)))
        if pasos <= 1:
            out.append(cur.copy())
            prev = cur
            continue

        for t in range(1, pasos + 1):
            inter = np.round(prev + delta * (t / pasos)).astype(np.int32)
            inter = clip_idx(inter, shape)
            out.append(inter.copy())

        prev = cur

    out = np.array(out, dtype=np.int32)
    uniq = [out[0]]
    for i in range(1, len(out)):
        if not np.array_equal(out[i], out[i - 1]):
            uniq.append(out[i])
    return np.array(uniq, dtype=np.int32)


def polyline_a_voxeles(polyline: List[np.ndarray], shape) -> np.ndarray:
    if len(polyline) == 0:
        raise ValueError("Polyline vacía")
    if len(polyline) == 1:
        return np.array([clip_idx(np.round(polyline[0]).astype(np.int32), shape)], dtype=np.int32)

    acc = []
    for i in range(len(polyline) - 1):
        seg = linea_voxeles_3d(polyline[i], polyline[i + 1], shape)
        if i > 0:
            seg = seg[1:]
        acc.append(seg)

    out = np.vstack(acc)
    uniq = [out[0]]
    for i in range(1, len(out)):
        if not np.array_equal(out[i], out[i - 1]):
            uniq.append(out[i])
    return np.array(uniq, dtype=np.int32)


# =========================================================
# TOPOLOGÍA BÁSICA
# =========================================================
def contar_componentes(mask: np.ndarray) -> int:
    struct = generate_binary_structure(rank=3, connectivity=3)
    _, num = label(mask.astype(bool), structure=struct)
    return int(num)


def neighbor_count_26(skel: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    return convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)


def contar_endpoints_branchpoints_voxel(skel: np.ndarray) -> Tuple[int, int]:
    n = neighbor_count_26(skel)
    sk = skel.astype(bool)
    endpoints = int(np.sum(sk & (n == 1)))
    branchpoints = int(np.sum(sk & (n >= 3)))
    return endpoints, branchpoints


def skeleton_contenido_en_mask(skel: np.ndarray, mask: np.ndarray) -> bool:
    return bool(np.all(mask[skel.astype(bool)] == 1))


def voxel_path_is_26_connected(path: np.ndarray) -> bool:
    if len(path) <= 1:
        return True
    dif = np.abs(np.diff(path, axis=0))
    return bool(np.all(np.max(dif, axis=1) <= 1))


def graph_degrees(graph: Dict[str, Any]) -> Dict[int, int]:
    deg = {}
    for n in graph["nodes"]:
        deg[n["id"]] = 0
    for e in graph["edges"]:
        deg[e["parent_node"]] += 1
        deg[e["child_node"]] += 1
    return deg


def graph_is_forest(graph: Dict[str, Any]) -> bool:
    nodes = graph["nodes"]
    edges = graph["edges"]

    parent = {n["id"]: n["id"] for n in nodes}
    rank = {n["id"]: 0 for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    for e in edges:
        if not union(e["parent_node"], e["child_node"]):
            return False
    return True


# =========================================================
# RASTERIZADO TUBULAR
# =========================================================
def draw_tube(mask: np.ndarray, p0: np.ndarray, p1: np.ndarray, radius: float) -> None:
    x0, x1, y0, y1, z0, z1 = bbox_segmento(p0, p1, radius, mask.shape, pad=MASK_RADIUS_PAD)

    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    zs = np.arange(z0, z1, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    D = distancia_punto_segmento_local(X, Y, Z, p0, p1)
    local = D <= radius
    mask[x0:x1, y0:y1, z0:z1] |= local


def paint_tube_metadata(
    edge_id_map: np.ndarray,
    radius_map: np.ndarray,
    tree_id_map: np.ndarray,
    intensity_map: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float,
    edge_id: int,
    tree_id: int,
    intensity_scale: float
) -> None:
    x0, x1, y0, y1, z0, z1 = bbox_segmento(p0, p1, radius, edge_id_map.shape, pad=MASK_RADIUS_PAD)

    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    zs = np.arange(z0, z1, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    D = distancia_punto_segmento_local(X, Y, Z, p0, p1)
    local = D <= radius

    loc_edge = edge_id_map[x0:x1, y0:y1, z0:z1]
    loc_rad = radius_map[x0:x1, y0:y1, z0:z1]
    loc_tree = tree_id_map[x0:x1, y0:y1, z0:z1]
    loc_int = intensity_map[x0:x1, y0:y1, z0:z1]

    fill = local & (loc_edge == 0)
    loc_edge[fill] = edge_id + 1
    loc_rad[fill] = radius
    loc_tree[fill] = tree_id + 1
    loc_int[fill] = intensity_scale

    overlap_same_tree = local & (loc_edge != 0) & (loc_tree == (tree_id + 1))
    loc_rad[overlap_same_tree] = np.maximum(loc_rad[overlap_same_tree], radius)
    loc_int[overlap_same_tree] = np.maximum(loc_int[overlap_same_tree], intensity_scale)


# =========================================================
# COLISIONES
# =========================================================
def segment_intersects_existing(
    occ_mask: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    radius_check: float,
    ignore_start_distance: float
) -> bool:
    x0, x1, y0, y1, z0, z1 = bbox_segmento(p0, p1, radius_check, occ_mask.shape, pad=1.0)
    if x1 <= x0 or y1 <= y0 or z1 <= z0:
        return True

    local_occ = occ_mask[x0:x1, y0:y1, z0:z1]
    if not np.any(local_occ):
        return False

    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    zs = np.arange(z0, z1, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    Dseg = distancia_punto_segmento_local(X, Y, Z, p0, p1)

    dx = X - float(p0[0])
    dy = Y - float(p0[1])
    dz = Z - float(p0[2])
    Dstart = np.sqrt(dx * dx + dy * dy + dz * dz)

    collision_zone = (Dseg <= radius_check) & local_occ
    collision_zone &= (Dstart > ignore_start_distance)

    return bool(np.any(collision_zone))


def point_far_from_existing_roots(p: np.ndarray, roots: List[np.ndarray], min_dist: float) -> bool:
    for r in roots:
        if distancia_punto_punto(p, r) < min_dist:
            return False
    return True


# =========================================================
# GENERADOR DE GRAFO
# =========================================================
class GraphBuilder:
    def __init__(self, shape: Tuple[int, int, int], rng: np.random.Generator):
        self.shape = shape
        self.rng = rng

        self.nodes: List[Nodo3D] = []
        self.edges: List[Arista3D] = []

        self.node_id = 0
        self.edge_id = 0

        self.occ_tube = np.zeros(shape, dtype=bool)
        self.occ_skel_clearance = np.zeros(shape, dtype=bool)

        self.node_positions: List[np.ndarray] = []
        self.edge_intensity_scale: Dict[int, float] = {}

    def add_node(self, xyz: np.ndarray, tipo: str, tree_id: int) -> int:
        for p in self.node_positions:
            if distancia_punto_punto(xyz, p) < MIN_NODE_SEPARATION:
                raise ValueError("Nodo demasiado cercano a otro nodo existente")

        nid = self.node_id
        self.node_id += 1

        self.nodes.append(Nodo3D(
            id=nid,
            xyz=[float(xyz[0]), float(xyz[1]), float(xyz[2])],
            tipo=tipo,
            tree_id=tree_id
        ))
        self.node_positions.append(xyz.copy())
        return nid

    def add_edge(
        self,
        tree_id: int,
        parent_node: int,
        child_node: int,
        depth: int,
        radius_mean: float,
        polyline: List[np.ndarray]
    ) -> int:
        eid = self.edge_id
        self.edge_id += 1

        length = 0.0
        for i in range(len(polyline) - 1):
            length += float(np.linalg.norm(polyline[i + 1] - polyline[i]))

        self.edges.append(Arista3D(
            id=eid,
            tree_id=tree_id,
            parent_node=parent_node,
            child_node=child_node,
            depth=depth,
            radius_mean=float(radius_mean),
            length=float(length),
            polyline=[[float(p[0]), float(p[1]), float(p[2])] for p in polyline]
        ))

        self.edge_intensity_scale[eid] = float(self.rng.uniform(
            1.0 - FILAMENT_INTENSITY_JITTER,
            1.0 + FILAMENT_INTENSITY_JITTER
        ))
        return eid

    def marcar_clearance_skeleton(self, p0: np.ndarray, p1: np.ndarray, radius: float) -> None:
        x0, x1, y0, y1, z0, z1 = bbox_segmento(p0, p1, radius, self.shape, pad=1.0)

        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        zs = np.arange(z0, z1, dtype=np.float32)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

        D = distancia_punto_segmento_local(X, Y, Z, p0, p1)
        self.occ_skel_clearance[x0:x1, y0:y1, z0:z1] |= (D <= radius)

    def marcar_tubo(self, p0: np.ndarray, p1: np.ndarray, radius: float) -> None:
        draw_tube(self.occ_tube, p0, p1, radius)

    def puede_aceptar_segmento(self, p0: np.ndarray, p1: np.ndarray, radius_local: float) -> bool:
        ignore_tube = max(IGNORE_START_DISTANCE, radius_local * 2.8 + 1.5)
        ignore_skel = max(IGNORE_START_DISTANCE, radius_local * 2.0 + 1.0)

        col_tube = segment_intersects_existing(
            self.occ_tube,
            p0, p1,
            radius_check=max(TUBE_CLEARANCE_RADIUS, radius_local * 0.75),
            ignore_start_distance=ignore_tube
        )
        if col_tube and self.rng.random() > ALLOW_TOUCH_PROB:
            return False

        col_skel = segment_intersects_existing(
            self.occ_skel_clearance,
            p0, p1,
            radius_check=SKELETON_CLEARANCE_RADIUS,
            ignore_start_distance=ignore_skel
        )
        if col_skel:
            return False

        return True

    def grow_branch(
        self,
        tree_id: int,
        start_xyz: np.ndarray,
        start_dir: np.ndarray,
        radius_base: float,
        depth: int,
        parent_node: int
    ) -> None:
        if depth > MAX_BRANCH_DEPTH:
            return

        steps = int(self.rng.integers(STEPS_MIN, STEPS_MAX + 1))
        step_base = float(self.rng.uniform(STEP_SIZE_MIN, STEP_SIZE_MAX))

        current = start_xyz.copy()
        direction = normalizar(start_dir.copy())

        polyline = [current.copy()]
        edge_radius = max(1.2, radius_base + float(self.rng.normal(0.0, RADIUS_JITTER)))

        for _ in range(steps):
            noise = self.rng.normal(0.0, DIRECTION_NOISE, size=3).astype(np.float32)

            drift = self.rng.normal(0.0, 0.035, size=3).astype(np.float32)

            proposed_dir = normalizar(
                DIRECTION_PERSISTENCE * direction
                + (1.0 - DIRECTION_PERSISTENCE) * noise
                + drift
            )

            proposed_dir = limitar_giro(direction, proposed_dir, MAX_TURN_ANGLE_DEG)

            step = max(0.8, step_base + float(self.rng.normal(0.0, 0.05)))
            nxt = current + proposed_dir * step

            if not dentro_volumen(nxt, self.shape):
                break

            if not self.puede_aceptar_segmento(current, nxt, edge_radius):
                break

            self.marcar_tubo(current, nxt, edge_radius)
            self.marcar_clearance_skeleton(current, nxt, SKELETON_CLEARANCE_RADIUS)

            polyline.append(nxt.copy())

            long_actual = 0.0
            for k in range(len(polyline) - 1):
                long_actual += float(np.linalg.norm(polyline[k + 1] - polyline[k]))

            can_branch = (
                depth < MAX_BRANCH_DEPTH and
                self.rng.random() < BRANCH_PROB and
                len(polyline) >= MIN_BRANCH_ADVANCE_STEPS and
                long_actual >= MIN_PARENT_SEGMENT_FOR_BRANCH
            )

            if can_branch:
                branch_origin = nxt.copy()

                too_close = False
                for pexist in self.node_positions:
                    if distancia_punto_punto(branch_origin, pexist) < MIN_NODE_SEPARATION:
                        too_close = True
                        break

                if not too_close:
                    new_dir = None
                    for _branch_try in range(24):
                        dev = normalizar(self.rng.normal(0.0, 1.0, size=3).astype(np.float32))
                        cand_dir = normalizar(proposed_dir + 0.95 * dev)
                        ang = angulo_entre_grados(proposed_dir, cand_dir)

                        if 30.0 <= ang <= 115.0:
                            new_dir = cand_dir
                            break

                    if new_dir is None:
                        dev = normalizar(self.rng.normal(0.0, 1.0, size=3).astype(np.float32))
                        new_dir = normalizar(proposed_dir + 1.05 * dev)

                    new_radius = max(1.2, edge_radius * float(self.rng.uniform(0.75, 0.95)))

                    try:
                        branch_node = self.add_node(branch_origin, "branchpoint", tree_id)
                        self.add_edge(
                            tree_id=tree_id,
                            parent_node=parent_node,
                            child_node=branch_node,
                            depth=depth,
                            radius_mean=float(edge_radius),
                            polyline=polyline
                        )

                        self.grow_branch(
                            tree_id=tree_id,
                            start_xyz=branch_origin,
                            start_dir=new_dir,
                            radius_base=new_radius,
                            depth=depth + 1,
                            parent_node=branch_node
                        )

                        parent_node = branch_node
                        polyline = [branch_origin.copy()]
                        edge_radius = new_radius

                    except ValueError:
                        pass

            current = nxt
            direction = proposed_dir

        if len(polyline) >= 2:
            endpoint_xyz = polyline[-1].copy()

            too_close = False
            for pexist in self.node_positions:
                if distancia_punto_punto(endpoint_xyz, pexist) < MIN_ENDPOINT_DISTANCE_TO_EXISTING_NODE:
                    too_close = True
                    break

            if not too_close:
                try:
                    end_node = self.add_node(endpoint_xyz, "endpoint", tree_id)
                    self.add_edge(
                        tree_id=tree_id,
                        parent_node=parent_node,
                        child_node=end_node,
                        depth=depth,
                        radius_mean=float(edge_radius),
                        polyline=polyline
                    )
                except ValueError:
                    pass


def generar_punto_inicial(rng: np.random.Generator, shape) -> np.ndarray:
    return np.array([
        rng.integers(MARGIN + 25, shape[0] - MARGIN - 25),
        rng.integers(MARGIN + 25, shape[1] - MARGIN - 25),
        rng.integers(MARGIN + 25, shape[2] - MARGIN - 25),
    ], dtype=np.float32)


def generar_direccion_inicial(rng: np.random.Generator) -> np.ndarray:
    return normalizar(rng.normal(0.0, 1.0, size=3).astype(np.float32))


# =========================================================
# GENERACIÓN DE RED + GT COMPLETO
# =========================================================
def generar_red_filamentosa(shape: Tuple[int, int, int], seed: int):
    rng = np.random.default_rng(seed)
    gb = GraphBuilder(shape, rng)

    num_trees = int(rng.integers(NUM_TREES_MIN, NUM_TREES_MAX + 1))
    root_points = []

    for tree_id in range(num_trees):
        ok = False
        for _ in range(250):
            p = generar_punto_inicial(rng, shape)
            if point_far_from_existing_roots(p, root_points, TREE_SEED_SEPARATION):
                root_points.append(p)
                ok = True
                break
        if not ok:
            p = generar_punto_inicial(rng, shape)
            root_points.append(p)

        root_dir = generar_direccion_inicial(rng)
        root_radius = float(rng.uniform(FILAMENT_RADIUS_MIN, FILAMENT_RADIUS_MAX))
        root_id = gb.add_node(p, "root", tree_id)

        gb.grow_branch(
            tree_id=tree_id,
            start_xyz=p.copy(),
            start_dir=root_dir,
            radius_base=root_radius,
            depth=0,
            parent_node=root_id
        )

    gt_skeleton = np.zeros(shape, dtype=np.uint8)
    gt_mask = gb.occ_tube.astype(np.uint8)

    node_type_map = np.zeros(shape, dtype=np.uint8)
    node_id_map = np.zeros(shape, dtype=np.int32)
    edge_id_map = np.zeros(shape, dtype=np.int32)
    radius_map = np.zeros(shape, dtype=np.float32)
    tree_id_map = np.zeros(shape, dtype=np.uint16)
    intensity_map = np.zeros(shape, dtype=np.float32)

    graph = {
        "num_trees": int(num_trees),
        "nodes": [asdict(n) for n in gb.nodes],
        "edges": [asdict(e) for e in gb.edges]
    }

    deg = graph_degrees(graph)
    node_lookup = {n.id: n for n in gb.nodes}

    for e in gb.edges:
        poly = [np.array(p, dtype=np.float32) for p in e.polyline]
        path = polyline_a_voxeles(poly, shape)

        if len(path) < 2:
            raise ValueError("Edge con path voxelizado degenerado")
        if not voxel_path_is_26_connected(path):
            raise ValueError("Edge con path no 26-conectado")

        gt_skeleton[path[:, 0], path[:, 1], path[:, 2]] = 1

        intensidad = gb.edge_intensity_scale.get(int(e.id), 1.0)

        for i in range(len(poly) - 1):
            p0 = poly[i]
            p1 = poly[i + 1]
            paint_tube_metadata(
                edge_id_map=edge_id_map,
                radius_map=radius_map,
                tree_id_map=tree_id_map,
                intensity_map=intensity_map,
                p0=p0,
                p1=p1,
                radius=float(e.radius_mean),
                edge_id=int(e.id),
                tree_id=int(e.tree_id),
                intensity_scale=float(intensidad)
            )

    for nid, _d in deg.items():
        n = node_lookup[nid]
        xyz = np.round(np.array(n.xyz)).astype(np.int32)
        xyz = clip_idx(xyz, shape)

        gt_skeleton[xyz[0], xyz[1], xyz[2]] = 1
        gt_mask[xyz[0], xyz[1], xyz[2]] = 1
        node_id_map[xyz[0], xyz[1], xyz[2]] = int(nid) + 1
        tree_id_map[xyz[0], xyz[1], xyz[2]] = int(n.tree_id) + 1

        if n.tipo == "root":
            node_type_map[xyz[0], xyz[1], xyz[2]] = 1
        elif n.tipo == "endpoint":
            node_type_map[xyz[0], xyz[1], xyz[2]] = 2
        elif n.tipo == "branchpoint":
            node_type_map[xyz[0], xyz[1], xyz[2]] = 3
        else:
            node_type_map[xyz[0], xyz[1], xyz[2]] = 4

    # Parche de consistencia geométrica:
    # asegura que cualquier voxel etiquetado por metadata quede incluido en gt_mask.
    gt_mask = ((gt_mask > 0) | (edge_id_map > 0) | (node_id_map > 0)).astype(np.uint8)

    if np.any((edge_id_map > 0) & (gt_mask == 0)):
        raise ValueError("edge_id_map marca voxeles fuera de gt_mask")
    if np.any((radius_map > 0) & (gt_mask == 0)):
        raise ValueError("radius_map marca voxeles fuera de gt_mask")
    if np.any((tree_id_map > 0) & ((gt_mask == 0) & (gt_skeleton == 0))):
        raise ValueError("tree_id_map marca voxeles fuera de regiones válidas")

    return {
        "gt_skeleton": gt_skeleton,
        "gt_mask": gt_mask,
        "node_type_map": node_type_map,
        "node_id_map": node_id_map,
        "edge_id_map": edge_id_map,
        "radius_map": radius_map,
        "tree_id_map": tree_id_map,
        "intensity_map": intensity_map,
        "graph": graph
    }


# =========================================================
# VOLUMEN SINTÉTICO
# =========================================================
def robust_normalize(x: np.ndarray, p_low=1.0, p_high=99.5) -> np.ndarray:
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    if hi <= lo:
        return np.clip(x, 0.0, 1.0).astype(np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def generar_volume(gt_mask: np.ndarray, intensity_map: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    mask_f = gt_mask.astype(np.float32)
    intens = np.where(gt_mask > 0, intensity_map, 0.0).astype(np.float32)

    base_signal = mask_f * np.maximum(intens, 0.85)
    base_signal = gaussian_filter(
        base_signal,
        sigma=(GAUSSIAN_SIGMA_XY, GAUSSIAN_SIGMA_XY, GAUSSIAN_SIGMA_Z)
    ).astype(np.float32)

    z = np.linspace(0.0, 1.0, gt_mask.shape[2], dtype=np.float32).reshape(1, 1, -1)
    attenuation = 1.0 - DEPTH_ATTENUATION * z
    base_signal = base_signal * attenuation

    poisson_input = np.clip(base_signal * POISSON_LAMBDA_SCALE, 0.0, None)
    poisson_signal = rng.poisson(poisson_input).astype(np.float32) / max(POISSON_LAMBDA_SCALE, 1e-6)

    background = rng.normal(
        BACKGROUND_MEAN,
        BACKGROUND_STD,
        size=gt_mask.shape
    ).astype(np.float32)

    volume = poisson_signal + background
    volume += rng.normal(0.0, NOISE_STD, size=gt_mask.shape).astype(np.float32)

    volume = robust_normalize(volume, 1.0, 99.5)
    return np.clip(volume, 0.0, 1.0).astype(np.float32)


# =========================================================
# MÉTRICAS Y VALIDACIÓN
# =========================================================
def safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def metrics_segmentation(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = int(np.sum(gt & pred))
    fp = int(np.sum((~gt) & pred))
    fn = int(np.sum(gt & (~pred)))

    dice = safe_div(2.0 * tp, 2.0 * tp + fp + fn)
    iou = safe_div(tp, tp + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def euclidean_length_polyline(poly: List[List[float]], spacing=(1.0, 1.0, 1.0)) -> float:
    if len(poly) < 2:
        return 0.0
    sp = np.array(spacing, dtype=np.float32)
    pts = np.array(poly, dtype=np.float32) * sp
    dif = np.diff(pts, axis=0)
    seg = np.linalg.norm(dif, axis=1)
    return float(np.sum(seg))


def gt_graph_metrics(graph: Dict[str, Any], spacing=(1.0, 1.0, 1.0)) -> Dict[str, float]:
    deg = graph_degrees(graph)

    total_length = 0.0
    tort_list = []
    node_pos = {
        n["id"]: np.array(n["xyz"], dtype=np.float32) * np.array(spacing, dtype=np.float32)
        for n in graph["nodes"]
    }

    for e in graph["edges"]:
        plen = euclidean_length_polyline(e["polyline"], spacing=spacing)
        total_length += plen

        p0 = node_pos[e["parent_node"]]
        p1 = node_pos[e["child_node"]]
        eucl = float(np.linalg.norm(p1 - p0))
        if eucl > 1e-8:
            tort_list.append(plen / eucl)

    endpoints = sum(1 for _, d in deg.items() if d == 1)
    branchpoints = sum(1 for _, d in deg.items() if d >= 3)
    tortuosity = float(np.mean(tort_list)) if tort_list else 1.0

    return {
        "total_length": float(total_length),
        "num_endpoints": int(endpoints),
        "num_branchpoints": int(branchpoints),
        "mean_tortuosity": float(tortuosity),
    }


def voxel_path_length(path: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    if len(path) < 2:
        return 0.0
    sp = np.array(spacing, dtype=np.float32)
    dif = np.diff(path.astype(np.float32) * sp, axis=0)
    seg = np.linalg.norm(dif, axis=1)
    return float(np.sum(seg))


def gt_skeleton_voxel_metrics(graph: Dict[str, Any], shape, spacing=(1.0, 1.0, 1.0)) -> Dict[str, float]:
    total = 0.0
    for e in graph["edges"]:
        poly = [np.array(p, dtype=np.float32) for p in e["polyline"]]
        path = polyline_a_voxeles(poly, shape)
        total += voxel_path_length(path, spacing)
    return {"total_length_voxelized": float(total)}


def validar_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    nodes = graph["nodes"]
    edges = graph["edges"]

    if len(nodes) == 0:
        raise ValueError("El grafo no tiene nodos")
    if len(edges) == 0:
        raise ValueError("El grafo no tiene aristas")
    if len(edges) < MIN_EDGES:
        raise ValueError(f"Muy pocas aristas: {len(edges)}")

    node_ids = set(n["id"] for n in nodes)
    if len(node_ids) != len(nodes):
        raise ValueError("IDs de nodos duplicados")

    edge_ids = set(e["id"] for e in edges)
    if len(edge_ids) != len(edges):
        raise ValueError("IDs de aristas duplicados")

    for e in edges:
        if e["parent_node"] not in node_ids or e["child_node"] not in node_ids:
            raise ValueError("Una arista referencia nodos inexistentes")
        if e["length"] <= 0:
            raise ValueError("Hay una arista con longitud no positiva")
        if e["radius_mean"] <= 0:
            raise ValueError("Hay una arista con radio no positivo")
        if len(e["polyline"]) < 2:
            raise ValueError("Hay una arista con polyline inválida")

    if not graph_is_forest(graph):
        raise ValueError("El grafo contiene ciclos; no es un bosque válido")

    deg = graph_degrees(graph)

    for n in nodes:
        d = deg.get(n["id"], 0)
        if d == 0:
            raise ValueError(f"Nodo aislado detectado en el grafo: {n['id']}")

    endpoints = sum(1 for _, d in deg.items() if d == 1)
    branchpoints = sum(1 for _, d in deg.items() if d >= 3)

    if endpoints < MIN_ENDPOINTS:
        raise ValueError(f"Muy pocos endpoints en el grafo: {endpoints}")

    max_branchpoints = max(10, int(len(edges) * MAX_BRANCHPOINTS_FACTOR))
    if branchpoints > max_branchpoints:
        raise ValueError(f"Demasiados branchpoints en grafo: {branchpoints} > {max_branchpoints}")

    return {
        "graph_nodes": int(len(nodes)),
        "graph_edges": int(len(edges)),
        "graph_endpoints": int(endpoints),
        "graph_branchpoints": int(branchpoints),
        "graph_num_trees": int(graph["num_trees"]),
    }


def validar_sample(
    gt_skeleton: np.ndarray,
    gt_mask: np.ndarray,
    node_type_map: np.ndarray,
    node_id_map: np.ndarray,
    edge_id_map: np.ndarray,
    radius_map: np.ndarray,
    tree_id_map: np.ndarray,
    intensity_map: np.ndarray,
    volume: np.ndarray,
    graph: Dict[str, Any]
) -> Dict[str, Any]:
    shape = gt_skeleton.shape

    if not (
        gt_mask.shape == shape and
        node_type_map.shape == shape and
        node_id_map.shape == shape and
        edge_id_map.shape == shape and
        radius_map.shape == shape and
        tree_id_map.shape == shape and
        intensity_map.shape == shape and
        volume.shape == shape
    ):
        raise ValueError("Las dimensiones no coinciden")

    if gt_skeleton.dtype != np.uint8:
        raise ValueError("gt_skeleton no es uint8")
    if gt_mask.dtype != np.uint8:
        raise ValueError("gt_mask no es uint8")
    if volume.dtype != np.float32:
        raise ValueError("volume no es float32")

    if not np.all(np.isin(np.unique(gt_skeleton), [0, 1])):
        raise ValueError("gt_skeleton no es binario")
    if not np.all(np.isin(np.unique(gt_mask), [0, 1])):
        raise ValueError("gt_mask no es binaria")

    skeleton_voxels = int(gt_skeleton.sum())
    mask_voxels = int(gt_mask.sum())

    if skeleton_voxels < MIN_SKELETON_VOXELS:
        raise ValueError(f"gt_skeleton demasiado pequeño: {skeleton_voxels}")
    if mask_voxels < MIN_MASK_VOXELS:
        raise ValueError(f"gt_mask demasiado pequeña: {mask_voxels}")

    frac = mask_voxels / gt_mask.size
    if frac > MAX_MASK_FRACTION:
        raise ValueError(f"gt_mask demasiado grande: {frac:.4f}")

    vmin = float(volume.min())
    vmax = float(volume.max())
    if not (0.0 <= vmin <= 1.0 and 0.0 <= vmax <= 1.0):
        raise ValueError("volume fuera de [0,1]")

    if not skeleton_contenido_en_mask(gt_skeleton, gt_mask):
        raise ValueError("gt_skeleton no está contenido en gt_mask")

    num_components_mask = contar_componentes(gt_mask)
    num_components_skel = contar_componentes(gt_skeleton)

    if num_components_mask == 0 or num_components_skel == 0:
        raise ValueError("No hay componentes")
    if num_components_mask > MAX_CONNECTED_COMPONENTS:
        raise ValueError(f"Demasiadas componentes en gt_mask: {num_components_mask}")
    if num_components_skel > MAX_CONNECTED_COMPONENTS:
        raise ValueError(f"Demasiadas componentes en gt_skeleton: {num_components_skel}") 
    # En voxelización fina pueden aparecer microcomponentes esqueléticas
    # por discretización, aunque la máscara tubular siga siendo válida.
    # Solo se rechaza si la diferencia es claramente excesiva.
    if num_components_skel > num_components_mask + 2:
        raise ValueError("gt_skeleton tiene muchas más componentes que gt_mask")

    graph_stats = validar_graph(graph)

    voxel_endpoints, voxel_branchpoints = contar_endpoints_branchpoints_voxel(gt_skeleton)

    exact_roots = int(np.sum(node_type_map == 1))
    exact_endpoints = int(np.sum(node_type_map == 2))
    exact_branchpoints = int(np.sum(node_type_map == 3))

    if exact_roots != graph["num_trees"]:
        raise ValueError(f"Número de roots inconsistente: {exact_roots} != {graph['num_trees']}")

    # La voxelización puede fusionar endpoints o branchpoints cercanos por discretización.
    # Por ello no se exige conservación exacta 1:1, sino consistencia razonable.
    min_voxel_endpoints = max(2, int(round(exact_endpoints * 0.35)))
    min_voxel_branchpoints = int(round(exact_branchpoints * 0.25))

    if voxel_endpoints < min_voxel_endpoints:
        raise ValueError(
            f"Voxel endpoints demasiado bajos respecto al grafo exacto: "
            f"voxel={voxel_endpoints}, exact={exact_endpoints}, mínimo_aceptable={min_voxel_endpoints}"
        )

    if voxel_branchpoints < min_voxel_branchpoints:
        raise ValueError(
            f"Voxel branchpoints demasiado bajos respecto al grafo exacto: "
            f"voxel={voxel_branchpoints}, exact={exact_branchpoints}, mínimo_aceptable={min_voxel_branchpoints}"
        )

    if not np.all(gt_skeleton[node_type_map > 0] == 1):
        raise ValueError("node_type_map marca nodos fuera del skeleton")

    if np.any((node_id_map > 0) & (gt_skeleton == 0)):
        raise ValueError("node_id_map marca voxeles fuera del skeleton")

    if np.any((edge_id_map > 0) & (gt_mask == 0)):
        raise ValueError("edge_id_map marca voxeles fuera de gt_mask")

    if np.any((radius_map > 0) & (gt_mask == 0)):
        raise ValueError("radius_map marca voxeles fuera de gt_mask")

    if np.any((tree_id_map > 0) & ((gt_mask == 0) & (gt_skeleton == 0))):
        raise ValueError("tree_id_map marca voxeles inválidos")

    if np.any(radius_map[edge_id_map > 0] <= 0):
        raise ValueError("Hay radios no positivos en voxeles con edge_id")

    gt_geom = gt_graph_metrics(graph, spacing=VOXEL_SPACING)
    gt_vox = gt_skeleton_voxel_metrics(graph, shape, spacing=VOXEL_SPACING)
    discret_err = abs(gt_vox["total_length_voxelized"] - gt_geom["total_length"])

    return {
        "skeleton_voxels": skeleton_voxels,
        "mask_voxels": mask_voxels,
        "mask_fraction": float(frac),
        "num_components_mask": int(num_components_mask),
        "num_components_skeleton": int(num_components_skel),
        "volume_min": float(vmin),
        "volume_max": float(vmax),
        "graph_nodes": int(graph_stats["graph_nodes"]),
        "graph_edges": int(graph_stats["graph_edges"]),
        "graph_endpoints": int(graph_stats["graph_endpoints"]),
        "graph_branchpoints": int(graph_stats["graph_branchpoints"]),
        "graph_num_trees": int(graph_stats["graph_num_trees"]),
        "node_map_roots": int(exact_roots),
        "node_map_endpoints": int(exact_endpoints),
        "node_map_branchpoints": int(exact_branchpoints),
        "voxel_endpoints": int(voxel_endpoints),
        "voxel_branchpoints": int(voxel_branchpoints),
        "gt_total_length_exact": float(gt_geom["total_length"]),
        "gt_total_length_voxelized": float(gt_vox["total_length_voxelized"]),
        "gt_abs_error_discretization_length": float(discret_err),
    }


# =========================================================
# OTSU GLOBAL
# =========================================================
def otsu_threshold(volume: np.ndarray, num_bins: int = 256) -> float:
    v = volume.astype(np.float32).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.5

    hist, bin_edges = np.histogram(v, bins=num_bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)

    prob = hist / np.sum(hist)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(num_bins))
    mu_t = mu[-1]

    sigma_b2 = np.zeros(num_bins, dtype=np.float64)
    denom = omega * (1.0 - omega)
    valid = denom > 1e-12
    sigma_b2[valid] = ((mu_t * omega[valid] - mu[valid]) ** 2) / denom[valid]

    idx = int(np.argmax(sigma_b2))
    thr = 0.5 * (bin_edges[idx] + bin_edges[idx + 1])
    return float(thr)


def segment_otsu_global(volume: np.ndarray) -> np.ndarray:
    thr = otsu_threshold(volume)
    pred = volume >= thr
    pred = remove_small_objects_3d(pred, OTSU_MIN_OBJECT_SIZE)
    pred = keep_largest_components(pred, MAX_CONNECTED_COMPONENTS)

    if USE_FILL_HOLES_OTSU:
        pred = binary_fill_holes(pred)

    return pred.astype(np.uint8)


# =========================================================
# SEGMENTACIÓN SIMPLE
# =========================================================
def segment_simple(volume: np.ndarray) -> np.ndarray:
    sm = gaussian_filter(volume.astype(np.float32), sigma=SIMPLE_GAUSSIAN_SIGMA)
    sm = robust_normalize(sm)

    thr = otsu_threshold(sm) + SIMPLE_THRESHOLD_OFFSET
    pred = sm >= thr

    struct = generate_binary_structure(3, 3)
    pred = binary_opening(pred, structure=struct, iterations=1)
    pred = binary_closing(pred, structure=struct, iterations=1)
    pred = remove_small_objects_3d(pred, SIMPLE_MIN_OBJECT_SIZE)
    pred = keep_largest_components(pred, MAX_CONNECTED_COMPONENTS)

    if USE_FILL_HOLES_SIMPLE:
        pred = binary_fill_holes(pred)

    return pred.astype(np.uint8)


# =========================================================
# PIPELINE PROPUESTO
# =========================================================
def segment_proposed(volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not FRANGI_OK:
        raise RuntimeError("No se encontró scikit-image con frangi. Instala: pip install scikit-image")

    vol = gaussian_filter(volume.astype(np.float32), sigma=PROPOSED_GAUSSIAN_PRE)
    vol = robust_normalize(vol)

    vess = frangi(
        vol,
        sigmas=PROPOSED_HESSIAN_SCALES,
        alpha=PROPOSED_VESSELNESS_ALPHA,
        beta=PROPOSED_VESSELNESS_BETA,
        gamma=PROPOSED_VESSELNESS_C,
        black_ridges=False
    ).astype(np.float32)

    vess = robust_normalize(vess)

    thr = otsu_threshold(vess)
    pred = vess >= thr

    struct = generate_binary_structure(3, 3)
    pred = binary_closing(pred, structure=struct, iterations=PROPOSED_CLOSE_ITERS)
    pred = binary_opening(pred, structure=struct, iterations=1)
    pred = remove_small_objects_3d(pred, PROPOSED_MIN_OBJECT_SIZE)
    pred = keep_largest_components(pred, MAX_CONNECTED_COMPONENTS)

    if USE_FILL_HOLES_PROPOSED:
        pred = binary_fill_holes(pred)

    return pred.astype(np.uint8), vess.astype(np.float32)


# =========================================================
# SKELETONIZACIÓN
# =========================================================
def skeletonize_3d_mask(mask: np.ndarray) -> np.ndarray:
    if not SKIMAGE_OK or SKELETON_MODE != "skeletonize_3d":
        raise RuntimeError(
            "No se encontró skeletonize_3d para volumen 3D. "
            "Instala una versión compatible de scikit-image: pip install scikit-image"
        )

    mask_bool = mask.astype(bool)
    sk = skeletonize_3d(mask_bool)
    return sk.astype(np.uint8)


# =========================================================
# GRAFO DESDE SKELETON PREDICHO (SIMPLIFICADO)
# =========================================================
_NEIGHBOR_OFFSETS_26 = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
]


def voxel_neighbors_26(p: Tuple[int, int, int], shape) -> List[Tuple[int, int, int]]:
    x, y, z = p
    out = []
    for dx, dy, dz in _NEIGHBOR_OFFSETS_26:
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
            out.append((nx, ny, nz))
    return out


def step_length(a: Tuple[int, int, int], b: Tuple[int, int, int], spacing=(1.0, 1.0, 1.0)) -> float:
    d = np.array([
        (b[0] - a[0]) * spacing[0],
        (b[1] - a[1]) * spacing[1],
        (b[2] - a[2]) * spacing[2]
    ], dtype=np.float32)
    return float(np.linalg.norm(d))


def _empty_skeleton_graph_metrics() -> Dict[str, Any]:
    return {
        "nodes": {},
        "edges": [],
        "metrics": {
            "total_length": 0.0,
            "num_endpoints": 0,
            "num_branchpoints": 0,
            "mean_tortuosity": 1.0,
        }
    }


def simplificar_grafo_skeleton(skel: np.ndarray, spacing=(1.0, 1.0, 1.0), prune_min_length=3.0) -> Dict[str, Any]:
    skel = skel.astype(bool)
    coords = np.argwhere(skel)

    if len(coords) == 0:
        return _empty_skeleton_graph_metrics()

    skset = {tuple(map(int, c)) for c in coords}
    adjacency: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]] = {}

    for p in skset:
        neigh = [q for q in voxel_neighbors_26(p, skel.shape) if q in skset]
        neigh.sort()
        adjacency[p] = neigh

    deg_map = {p: len(adjacency[p]) for p in skset}

    visited_comp = set()
    components = []

    for p in skset:
        if p in visited_comp:
            continue
        stack = [p]
        visited_comp.add(p)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adjacency[u]:
                if v not in visited_comp:
                    visited_comp.add(v)
                    stack.append(v)
        components.append(comp)

    all_edges = []
    visited_undirected_edges: Set[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = set()

    def edge_key(a, b):
        return tuple(sorted((a, b)))

    def trace_path(start_node, first_next, topo_nodes_set):
        path = [start_node, first_next]
        visited_undirected_edges.add(edge_key(start_node, first_next))

        prev = start_node
        curr = first_next

        while True:
            if curr in topo_nodes_set and curr != start_node:
                break

            next_candidates = [n for n in adjacency[curr] if n != prev]
            next_candidates.sort()

            if len(next_candidates) == 0:
                break

            nxt = None
            for cand in next_candidates:
                ek = edge_key(curr, cand)
                if ek not in visited_undirected_edges:
                    nxt = cand
                    break

            if nxt is None:
                if curr == start_node:
                    break
                nxt = next_candidates[0]

            ek2 = edge_key(curr, nxt)
            if ek2 in visited_undirected_edges and nxt == prev:
                break

            visited_undirected_edges.add(ek2)
            path.append(nxt)
            prev, curr = curr, nxt

            if curr == start_node:
                break

        length = 0.0
        for i in range(len(path) - 1):
            length += step_length(path[i], path[i + 1], spacing)

        return {
            "start": path[0],
            "end": path[-1],
            "path": path,
            "length": float(length)
        }

    for comp in components:
        comp_set = set(comp)
        topo_nodes = {p for p in comp if deg_map[p] != 2}

        if len(topo_nodes) == 0:
            p0 = min(comp)
            topo_nodes = {p0}

        topo_nodes = set(sorted(topo_nodes))

        for n in sorted(topo_nodes):
            for q in adjacency[n]:
                ek = edge_key(n, q)
                if ek in visited_undirected_edges:
                    continue
                if q not in comp_set:
                    continue
                e = trace_path(n, q, topo_nodes)
                if len(e["path"]) >= 2 and e["length"] > 0:
                    all_edges.append(e)

    if len(all_edges) == 0:
        return _empty_skeleton_graph_metrics()

    def construir_grados(edges):
        grados = {}
        for e in edges:
            grados[e["start"]] = grados.get(e["start"], 0) + 1
            grados[e["end"]] = grados.get(e["end"], 0) + 1
        return grados

    changed = True
    edges_work = all_edges.copy()

    while changed:
        changed = False
        grados = construir_grados(edges_work)
        nuevas = []
        for e in edges_work:
            g0 = grados.get(e["start"], 0)
            g1 = grados.get(e["end"], 0)

            es_espuria = (
                e["length"] < prune_min_length and
                ((g0 == 1 and g1 >= 2) or (g1 == 1 and g0 >= 2))
            )
            if es_espuria:
                changed = True
                continue
            nuevas.append(e)
        edges_work = nuevas

        if len(edges_work) == 0:
            return _empty_skeleton_graph_metrics()

    grados = construir_grados(edges_work)

    total_length = float(sum(e["length"] for e in edges_work))

    torts = []
    for e in edges_work:
        p0 = np.array(e["start"], dtype=np.float32) * np.array(spacing, dtype=np.float32)
        p1 = np.array(e["end"], dtype=np.float32) * np.array(spacing, dtype=np.float32)
        eucl = float(np.linalg.norm(p1 - p0))
        if eucl > 1e-8:
            torts.append(e["length"] / eucl)

    endpoints = sum(1 for _, d in grados.items() if d == 1)
    branchpoints = sum(1 for _, d in grados.items() if d >= 3)
    mean_tortuosity = float(np.mean(torts)) if torts else 1.0

    return {
        "nodes": grados,
        "edges": edges_work,
        "metrics": {
            "total_length": float(total_length),
            "num_endpoints": int(endpoints),
            "num_branchpoints": int(branchpoints),
            "mean_tortuosity": float(mean_tortuosity),
        }
    }


def geometric_error(pred_metrics: Dict[str, float], gt_metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        "abs_error_total_length": abs(pred_metrics["total_length"] - gt_metrics["total_length"]),
        "abs_error_num_endpoints": abs(pred_metrics["num_endpoints"] - gt_metrics["num_endpoints"]),
        "abs_error_num_branchpoints": abs(pred_metrics["num_branchpoints"] - gt_metrics["num_branchpoints"]),
        "abs_error_mean_tortuosity": abs(pred_metrics["mean_tortuosity"] - gt_metrics["mean_tortuosity"]),
    }

#===================
def generar_signed_distance_map(gt_mask: np.ndarray) -> np.ndarray:
    mask = gt_mask.astype(bool)

    dist_outside = distance_transform_edt(~mask).astype(np.float32)
    dist_inside = distance_transform_edt(mask).astype(np.float32)

    signed = dist_outside.copy()
    signed[mask] = -dist_inside[mask]

    return signed.astype(np.float32)
#===================
# =========================================================
# GENERAR SAMPLE
# =========================================================
def generar_sample(idx: int) -> None:
    sample_name = f"sample_{idx:03d}"
    sample_dir = os.path.join(DATASET_DIR, sample_name)
    asegurar_directorio(sample_dir)

    ultimo_error = None

    for intento in range(MAX_ATTEMPTS_PER_SAMPLE):
        seed_tree = BASE_SEED + idx * 1000 + intento
        seed_vol = BASE_SEED + 100000 + idx * 1000 + intento

        try:
            #data = generar_red_filamentosa(VOL_SHAPE, seed_tree)
            #volume = generar_volume(data["gt_mask"], data["intensity_map"], seed_vol)
            data = generar_red_filamentosa(VOL_SHAPE, seed_tree)
            volume = generar_volume(data["gt_mask"], data["intensity_map"], seed_vol)
            signed_distance_map = generar_signed_distance_map(data["gt_mask"])

            stats = validar_sample(
                gt_skeleton=data["gt_skeleton"],
                gt_mask=data["gt_mask"],
                node_type_map=data["node_type_map"],
                node_id_map=data["node_id_map"],
                edge_id_map=data["edge_id_map"],
                radius_map=data["radius_map"],
                tree_id_map=data["tree_id_map"],
                intensity_map=data["intensity_map"],
                volume=volume,
                graph=data["graph"]
            )

            limpiar_archivos_sample(sample_dir)

            np.save(os.path.join(sample_dir, "gt_signed_distance_map.npy"), signed_distance_map)
            np.save(os.path.join(sample_dir, "gt_skeleton.npy"), data["gt_skeleton"])
            np.save(os.path.join(sample_dir, "gt_mask.npy"), data["gt_mask"])
            np.save(os.path.join(sample_dir, "node_type_map.npy"), data["node_type_map"])
            np.save(os.path.join(sample_dir, "node_id_map.npy"), data["node_id_map"])
            np.save(os.path.join(sample_dir, "edge_id_map.npy"), data["edge_id_map"])
            np.save(os.path.join(sample_dir, "radius_map.npy"), data["radius_map"])
            np.save(os.path.join(sample_dir, "tree_id_map.npy"), data["tree_id_map"])
            np.save(os.path.join(sample_dir, "intensity_map.npy"), data["intensity_map"])
            #np.save(os.path.join(sample_dir, "volume.npy"), volume)
            np.save(os.path.join(sample_dir, "volume.npy"), volume)
            np.save(os.path.join(sample_dir, "gt_signed_distance_map.npy"), signed_distance_map)
            
            guardar_json(os.path.join(sample_dir, "graph.json"), data["graph"])

            metadata = {
                "sample": sample_name,
                "shape": list(VOL_SHAPE),
                "seed_tree": int(seed_tree),
                "seed_volume": int(seed_vol),
                "attempt": int(intento + 1),
                "files": [
                    "gt_skeleton.npy",
                    "gt_mask.npy",
                    "node_type_map.npy",
                    "node_id_map.npy",
                    "edge_id_map.npy",
                    "radius_map.npy",
                    "tree_id_map.npy",
                    "intensity_map.npy",
                    "volume.npy",
                    "graph.json"
                ],
                "stats": stats,
                "notes": [
                    "graph.json es la verdad topológica exacta del sample.",
                    "gt_skeleton es una voxelización derivada del grafo exacto, no una esqueletización morfológica.",
                    "gt_mask es la representación volumétrica tubular derivada del grafo exacto.",
                    "intensity_map almacena heterogeneidad intrafilamento utilizada para el volumen sintético.",
                    "Las métricas geométricas del predicho se estiman sobre un grafo simplificado del skeleton."
                ]
            }

            guardar_json(os.path.join(sample_dir, "metadata.json"), metadata)
            print(f"  OK en intento {intento + 1}")
            return

        except Exception as e:
            ultimo_error = str(e)
            print(f"  intento {intento + 1}: {ultimo_error}")
            continue

    raise RuntimeError(
        f"No se pudo generar un sample válido para {sample_name} "
        f"tras {MAX_ATTEMPTS_PER_SAMPLE} intentos. Último error: {ultimo_error}"
    )


# =========================================================
# EVALUACIÓN DE UN SAMPLE
# =========================================================
def evaluar_sample(idx: int) -> Dict[str, Any]:
    sample_name = f"sample_{idx:03d}"
    sample_dir = os.path.join(DATASET_DIR, sample_name)

    gt_mask = np.load(os.path.join(sample_dir, "gt_mask.npy")).astype(np.uint8)
    volume = np.load(os.path.join(sample_dir, "volume.npy")).astype(np.float32)

    with open(os.path.join(sample_dir, "graph.json"), "r", encoding="utf-8") as f:
        graph = json.load(f)

    gt_geom = gt_graph_metrics(graph, spacing=VOXEL_SPACING)

    pred_otsu = segment_otsu_global(volume)
    seg_otsu = metrics_segmentation(gt_mask, pred_otsu)
    np.save(os.path.join(sample_dir, "pred_otsu_mask.npy"), pred_otsu)

    pred_simple = segment_simple(volume)
    seg_simple = metrics_segmentation(gt_mask, pred_simple)
    np.save(os.path.join(sample_dir, "pred_simple_mask.npy"), pred_simple)

    pred_proposed, vesselness = segment_proposed(volume)
    seg_proposed = metrics_segmentation(gt_mask, pred_proposed)
    np.save(os.path.join(sample_dir, "pred_proposed_mask.npy"), pred_proposed)
    np.save(os.path.join(sample_dir, "vesselness.npy"), vesselness)

    pred_otsu_skel = skeletonize_3d_mask(pred_otsu)
    pred_simple_skel = skeletonize_3d_mask(pred_simple)
    pred_proposed_skel = skeletonize_3d_mask(pred_proposed)

    np.save(os.path.join(sample_dir, "pred_otsu_skeleton.npy"), pred_otsu_skel)
    np.save(os.path.join(sample_dir, "pred_simple_skeleton.npy"), pred_simple_skel)
    np.save(os.path.join(sample_dir, "pred_proposed_skeleton.npy"), pred_proposed_skel)

    g_otsu = simplificar_grafo_skeleton(pred_otsu_skel, spacing=VOXEL_SPACING, prune_min_length=PRUNE_MIN_BRANCH_LENGTH)
    g_simple = simplificar_grafo_skeleton(pred_simple_skel, spacing=VOXEL_SPACING, prune_min_length=PRUNE_MIN_BRANCH_LENGTH)
    g_prop = simplificar_grafo_skeleton(pred_proposed_skel, spacing=VOXEL_SPACING, prune_min_length=PRUNE_MIN_BRANCH_LENGTH)

    geom_otsu = g_otsu["metrics"]
    geom_simple = g_simple["metrics"]
    geom_proposed = g_prop["metrics"]

    geom_results = {
        "gt": gt_geom,
        "otsu_global": {
            **geom_otsu,
            **geometric_error(geom_otsu, gt_geom)
        },
        "segmentacion_simple": {
            **geom_simple,
            **geometric_error(geom_simple, gt_geom)
        },
        "pipeline_propuesto": {
            **geom_proposed,
            **geometric_error(geom_proposed, gt_geom)
        }
    }

    results = {
        "sample": sample_name,
        "segmentation_metrics": {
            "otsu_global": seg_otsu,
            "segmentacion_simple": seg_simple,
            "pipeline_propuesto": seg_proposed
        },
        "geometric_metrics": geom_results
    }

    guardar_json(os.path.join(sample_dir, "evaluation.json"), results)
    return results


# =========================================================
# RESUMEN GLOBAL
# =========================================================
def promedio_metricas(lista: List[float]) -> float:
    return float(np.mean(lista)) if lista else 0.0


def std_metricas(lista: List[float]) -> float:
    return float(np.std(lista, ddof=0)) if lista else 0.0


def mediana_metricas(lista: List[float]) -> float:
    return float(np.median(lista)) if lista else 0.0


def redondear_dict(d: Dict[str, Any], decimales: int = DECIMALS_SUMMARY) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = round(v, decimales)
        elif isinstance(v, dict):
            out[k] = redondear_dict(v, decimales)
        else:
            out[k] = v
    return out


def exportar_resumen_csv(results_all: List[Dict[str, Any]]) -> None:
    out_csv = os.path.join(DATASET_DIR, "summary_metrics.csv")

    rows = []
    for r in results_all:
        sample = r["sample"]
        seg = r["segmentation_metrics"]
        geom = r["geometric_metrics"]

        row = {
            "sample": sample,

            "dice_otsu": seg["otsu_global"]["dice"],
            "iou_otsu": seg["otsu_global"]["iou"],
            "precision_otsu": seg["otsu_global"]["precision"],
            "recall_otsu": seg["otsu_global"]["recall"],

            "dice_simple": seg["segmentacion_simple"]["dice"],
            "iou_simple": seg["segmentacion_simple"]["iou"],
            "precision_simple": seg["segmentacion_simple"]["precision"],
            "recall_simple": seg["segmentacion_simple"]["recall"],

            "dice_proposed": seg["pipeline_propuesto"]["dice"],
            "iou_proposed": seg["pipeline_propuesto"]["iou"],
            "precision_proposed": seg["pipeline_propuesto"]["precision"],
            "recall_proposed": seg["pipeline_propuesto"]["recall"],

            "gt_total_length": geom["gt"]["total_length"],
            "gt_num_endpoints": geom["gt"]["num_endpoints"],
            "gt_num_branchpoints": geom["gt"]["num_branchpoints"],
            "gt_mean_tortuosity": geom["gt"]["mean_tortuosity"],

            "prop_total_length": geom["pipeline_propuesto"]["total_length"],
            "prop_num_endpoints": geom["pipeline_propuesto"]["num_endpoints"],
            "prop_num_branchpoints": geom["pipeline_propuesto"]["num_branchpoints"],
            "prop_mean_tortuosity": geom["pipeline_propuesto"]["mean_tortuosity"],

            "abs_error_total_length_proposed": geom["pipeline_propuesto"]["abs_error_total_length"],
            "abs_error_num_endpoints_proposed": geom["pipeline_propuesto"]["abs_error_num_endpoints"],
            "abs_error_num_branchpoints_proposed": geom["pipeline_propuesto"]["abs_error_num_branchpoints"],
            "abs_error_mean_tortuosity_proposed": geom["pipeline_propuesto"]["abs_error_mean_tortuosity"],
        }
        rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def exportar_resumen_global_json(results_all: List[Dict[str, Any]]) -> None:
    out_json = os.path.join(DATASET_DIR, "summary_global.json")

    def collect(method: str, metric: str) -> List[float]:
        vals = []
        for r in results_all:
            vals.append(float(r["segmentation_metrics"][method][metric]))
        return vals

    def collect_geom(method: str, metric: str) -> List[float]:
        vals = []
        for r in results_all:
            vals.append(float(r["geometric_metrics"][method][metric]))
        return vals

    def resumen_estadistico(valores: List[float]) -> Dict[str, float]:
        return {
            "mean": promedio_metricas(valores),
            "std": std_metricas(valores),
            "median": mediana_metricas(valores),
        }

    summary = {
        "num_samples": len(results_all),
        "segmentation_stats": {
            "otsu_global": {
                "dice": resumen_estadistico(collect("otsu_global", "dice")),
                "iou": resumen_estadistico(collect("otsu_global", "iou")),
                "precision": resumen_estadistico(collect("otsu_global", "precision")),
                "recall": resumen_estadistico(collect("otsu_global", "recall")),
            },
            "segmentacion_simple": {
                "dice": resumen_estadistico(collect("segmentacion_simple", "dice")),
                "iou": resumen_estadistico(collect("segmentacion_simple", "iou")),
                "precision": resumen_estadistico(collect("segmentacion_simple", "precision")),
                "recall": resumen_estadistico(collect("segmentacion_simple", "recall")),
            },
            "pipeline_propuesto": {
                "dice": resumen_estadistico(collect("pipeline_propuesto", "dice")),
                "iou": resumen_estadistico(collect("pipeline_propuesto", "iou")),
                "precision": resumen_estadistico(collect("pipeline_propuesto", "precision")),
                "recall": resumen_estadistico(collect("pipeline_propuesto", "recall")),
            }
        },
        "geometric_stats": {
            "otsu_global": {
                "total_length": resumen_estadistico(collect_geom("otsu_global", "total_length")),
                "num_endpoints": resumen_estadistico(collect_geom("otsu_global", "num_endpoints")),
                "num_branchpoints": resumen_estadistico(collect_geom("otsu_global", "num_branchpoints")),
                "mean_tortuosity": resumen_estadistico(collect_geom("otsu_global", "mean_tortuosity")),
            },
            "segmentacion_simple": {
                "total_length": resumen_estadistico(collect_geom("segmentacion_simple", "total_length")),
                "num_endpoints": resumen_estadistico(collect_geom("segmentacion_simple", "num_endpoints")),
                "num_branchpoints": resumen_estadistico(collect_geom("segmentacion_simple", "num_branchpoints")),
                "mean_tortuosity": resumen_estadistico(collect_geom("segmentacion_simple", "mean_tortuosity")),
            },
            "pipeline_propuesto": {
                "total_length": resumen_estadistico(collect_geom("pipeline_propuesto", "total_length")),
                "num_endpoints": resumen_estadistico(collect_geom("pipeline_propuesto", "num_endpoints")),
                "num_branchpoints": resumen_estadistico(collect_geom("pipeline_propuesto", "num_branchpoints")),
                "mean_tortuosity": resumen_estadistico(collect_geom("pipeline_propuesto", "mean_tortuosity")),
                "abs_error_total_length": resumen_estadistico(collect_geom("pipeline_propuesto", "abs_error_total_length")),
                "abs_error_num_endpoints": resumen_estadistico(collect_geom("pipeline_propuesto", "abs_error_num_endpoints")),
                "abs_error_num_branchpoints": resumen_estadistico(collect_geom("pipeline_propuesto", "abs_error_num_branchpoints")),
                "abs_error_mean_tortuosity": resumen_estadistico(collect_geom("pipeline_propuesto", "abs_error_mean_tortuosity")),
            }
        }
    }

    summary = redondear_dict(summary, DECIMALS_SUMMARY)
    guardar_json(out_json, summary)


def guardar_configuracion_dataset() -> None:
    config = {
        "DATASET_DIR": DATASET_DIR,
        "MODE": MODE,
        "NUM_SAMPLES": NUM_SAMPLES,
        "VOL_SHAPE": list(VOL_SHAPE),
        "BASE_SEED": BASE_SEED,
        "MAX_ATTEMPTS_PER_SAMPLE": MAX_ATTEMPTS_PER_SAMPLE,
        "NUM_TREES_MIN": NUM_TREES_MIN,
        "NUM_TREES_MAX": NUM_TREES_MAX,
        "TREE_SEED_SEPARATION": TREE_SEED_SEPARATION,
        "MAX_BRANCH_DEPTH": MAX_BRANCH_DEPTH,
        "BRANCH_PROB": BRANCH_PROB,
        "STEPS_MIN": STEPS_MIN,
        "STEPS_MAX": STEPS_MAX,
        "STEP_SIZE_MIN": STEP_SIZE_MIN,
        "STEP_SIZE_MAX": STEP_SIZE_MAX,
        "FILAMENT_RADIUS_MIN": FILAMENT_RADIUS_MIN,
        "FILAMENT_RADIUS_MAX": FILAMENT_RADIUS_MAX,
        "RADIUS_JITTER": RADIUS_JITTER,
        "MASK_RADIUS_PAD": MASK_RADIUS_PAD,
        "MARGIN": MARGIN,
        "DIRECTION_PERSISTENCE": DIRECTION_PERSISTENCE,
        "DIRECTION_NOISE": DIRECTION_NOISE,
        "MAX_TURN_ANGLE_DEG": MAX_TURN_ANGLE_DEG,
        "SKELETON_CLEARANCE_RADIUS": SKELETON_CLEARANCE_RADIUS,
        "TUBE_CLEARANCE_RADIUS": TUBE_CLEARANCE_RADIUS,
        "IGNORE_START_DISTANCE": IGNORE_START_DISTANCE,
        "ALLOW_TOUCH_PROB": ALLOW_TOUCH_PROB,
        "MIN_NODE_SEPARATION": MIN_NODE_SEPARATION,
        "MIN_BRANCH_ADVANCE_STEPS": MIN_BRANCH_ADVANCE_STEPS,
        "MIN_PARENT_SEGMENT_FOR_BRANCH": MIN_PARENT_SEGMENT_FOR_BRANCH,
        "MIN_ENDPOINT_DISTANCE_TO_EXISTING_NODE": MIN_ENDPOINT_DISTANCE_TO_EXISTING_NODE,
        "LINE_SAMPLES_PER_VOXEL": LINE_SAMPLES_PER_VOXEL,
        "GAUSSIAN_SIGMA_XY": GAUSSIAN_SIGMA_XY,
        "GAUSSIAN_SIGMA_Z": GAUSSIAN_SIGMA_Z,
        "BACKGROUND_MEAN": BACKGROUND_MEAN,
        "BACKGROUND_STD": BACKGROUND_STD,
        "NOISE_STD": NOISE_STD,
        "DEPTH_ATTENUATION": DEPTH_ATTENUATION,
        "POISSON_LAMBDA_SCALE": POISSON_LAMBDA_SCALE,
        "FILAMENT_INTENSITY_JITTER": FILAMENT_INTENSITY_JITTER,
        "MIN_MASK_VOXELS": MIN_MASK_VOXELS,
        "MAX_MASK_FRACTION": MAX_MASK_FRACTION,
        "MIN_SKELETON_VOXELS": MIN_SKELETON_VOXELS,
        "MAX_CONNECTED_COMPONENTS": MAX_CONNECTED_COMPONENTS,
        "MIN_EDGES": MIN_EDGES,
        "MIN_ENDPOINTS": MIN_ENDPOINTS,
        "MAX_BRANCHPOINTS_FACTOR": MAX_BRANCHPOINTS_FACTOR,
        "VOXEL_SPACING": list(VOXEL_SPACING),
        "OTSU_MIN_OBJECT_SIZE": OTSU_MIN_OBJECT_SIZE,
        "USE_FILL_HOLES_OTSU": USE_FILL_HOLES_OTSU,
        "SIMPLE_GAUSSIAN_SIGMA": list(SIMPLE_GAUSSIAN_SIGMA),
        "SIMPLE_THRESHOLD_OFFSET": SIMPLE_THRESHOLD_OFFSET,
        "SIMPLE_MIN_OBJECT_SIZE": SIMPLE_MIN_OBJECT_SIZE,
        "USE_FILL_HOLES_SIMPLE": USE_FILL_HOLES_SIMPLE,
        "PROPOSED_HESSIAN_SCALES": PROPOSED_HESSIAN_SCALES,
        "PROPOSED_GAUSSIAN_PRE": list(PROPOSED_GAUSSIAN_PRE),
        "PROPOSED_VESSELNESS_ALPHA": PROPOSED_VESSELNESS_ALPHA,
        "PROPOSED_VESSELNESS_BETA": PROPOSED_VESSELNESS_BETA,
        "PROPOSED_VESSELNESS_C": PROPOSED_VESSELNESS_C,
        "PROPOSED_MIN_OBJECT_SIZE": PROPOSED_MIN_OBJECT_SIZE,
        "PROPOSED_CLOSE_ITERS": PROPOSED_CLOSE_ITERS,
        "USE_FILL_HOLES_PROPOSED": USE_FILL_HOLES_PROPOSED,
        "PRUNE_MIN_BRANCH_LENGTH": PRUNE_MIN_BRANCH_LENGTH,
        "SKELETON_MODE": SKELETON_MODE,
    }
    guardar_json(os.path.join(DATASET_DIR, "dataset_config.json"), config)


# =========================================================
# MAIN
# =========================================================
def main_generate() -> None:
    asegurar_directorio(DATASET_DIR)
    guardar_configuracion_dataset()

    print("===== GENERANDO DATASET SINTÉTICO 3D PARA PAPER =====")
    print(f"Salida: {DATASET_DIR}")
    print(f"Shape: {VOL_SHAPE}")
    print(f"Samples: {NUM_SAMPLES}")

    for i in range(NUM_SAMPLES):
        print(f"[GENERAR] sample_{i:03d} ...")
        generar_sample(i)

    print("===== FIN GENERACIÓN =====")


def main_evaluate() -> None:
    if not SKIMAGE_OK:
        raise RuntimeError("Para evaluar necesitas scikit-image con skeletonize_3d instalado.")
    if not FRANGI_OK:
        raise RuntimeError("Para evaluar el pipeline propuesto necesitas scikit-image con frangi.")

    print("===== EVALUANDO DATASET =====")
    print(f"Skeletonización 3D disponible: {SKIMAGE_OK}")
    print(f"Frangi disponible: {FRANGI_OK}")

    results_all = []
    for i in range(NUM_SAMPLES):
        print(f"[EVALUAR] sample_{i:03d} ...")
        r = evaluar_sample(i)
        results_all.append(r)

    exportar_resumen_csv(results_all)
    exportar_resumen_global_json(results_all)

    print("===== FIN EVALUACIÓN =====")
    print(f"Resumen CSV: {os.path.join(DATASET_DIR, 'summary_metrics.csv')}")
    print(f"Resumen JSON: {os.path.join(DATASET_DIR, 'summary_global.json')}")


def main() -> None:
    asegurar_directorio(DATASET_DIR)

    if MODE == "generate":
        main_generate()
    elif MODE == "evaluate":
        main_evaluate()
    elif MODE == "all":
        main_generate()
        main_evaluate()
    else:
        raise ValueError("MODE debe ser: 'generate', 'evaluate' o 'all'")


if __name__ == "__main__":
    main()