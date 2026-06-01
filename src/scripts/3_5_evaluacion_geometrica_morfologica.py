# 3_5_evaluacion_geometrica_morfologica.py
import os
import json
import math
import traceback
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, maximum_filter

# =========================================================
# CONFIGURACIÓN
# =========================================================
BASE_DIR = r"samples"

# Orden: (z, y, x)
VOXEL_SIZE = (1.0, 1.0, 1.0)

# Umbral mínimo de radio local para considerar eje central en predicción
# Si tus estructuras son finas, dejar 1.0 suele ser razonable
MIN_RADIUS_FOR_CENTERLINE = 1.0

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def load_binary(path):
    arr = np.load(path)
    return (arr > 0)

def rel_error(pred, gt):
    if gt == 0 or np.isnan(pred) or np.isnan(gt):
        return np.nan
    return abs(pred - gt) / abs(gt)

def get_sample_folders(base_dir):
    return sorted([
        d for d in os.listdir(base_dir)
        if d.startswith("sample_") and os.path.isdir(os.path.join(base_dir, d))
    ])

# =========================================================
# EJE CENTRAL APROXIMADO DE PREDICCIÓN
# =========================================================
def approximate_centerline(mask, voxel_size=(1.0, 1.0, 1.0), min_radius=1.0):
    """
    Extrae un eje central aproximado a partir de máximos locales
    de la transformada de distancia dentro de la máscara.

    NOTA:
    Esto NO es un grafo topológico completo.
    Se usa como aproximación geométrica del eje central.
    """
    dt = distance_transform_edt(mask, sampling=voxel_size)

    # máximos locales 3x3x3
    local_max = (dt == maximum_filter(dt, size=3, mode="constant"))
    centerline = mask & local_max & (dt >= min_radius)

    return centerline, dt

# =========================================================
# LONGITUD APROXIMADA DE UN ESQUELETO / EJE CENTRAL
# =========================================================
def approximate_length(binary_centerline, voxel_size=(1.0, 1.0, 1.0)):
    """
    Estima la longitud sumando conexiones vecinas en 26-conectividad
    y corrigiendo doble conteo.
    """
    coords = np.argwhere(binary_centerline)
    if coords.shape[0] == 0:
        return 0.0

    z_size, y_size, x_size = voxel_size
    coord_set = set(map(tuple, coords.tolist()))
    total = 0.0

    offsets = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                # evitar doble conteo
                if (dz, dy, dx) > (0, 0, 0):
                    offsets.append((dz, dy, dx))

    for z, y, x in coords:
        for dz, dy, dx in offsets:
            nz, ny, nx = z + dz, y + dy, x + dx
            if (nz, ny, nx) in coord_set:
                dist = math.sqrt(
                    (dz * z_size) ** 2 +
                    (dy * y_size) ** 2 +
                    (dx * x_size) ** 2
                )
                total += dist

    return float(total)

# =========================================================
# DIÁMETRO MEDIO / MEDIANO
# =========================================================
def diameter_stats_from_centerline_and_dt(centerline, dt):
    values = dt[centerline]
    if values.size == 0:
        return {
            "diam_mean": np.nan,
            "diam_median": np.nan,
            "diam_std": np.nan,
            "diam_count": 0
        }

    diam = 2.0 * values.astype(float)

    return {
        "diam_mean": float(np.mean(diam)),
        "diam_median": float(np.median(diam)),
        "diam_std": float(np.std(diam)),
        "diam_count": int(diam.size)
    }

def diameter_stats_gt(gt_skeleton, radius_map):
    values = radius_map[gt_skeleton > 0]
    if values.size == 0:
        return {
            "diam_mean": np.nan,
            "diam_median": np.nan,
            "diam_std": np.nan,
            "diam_count": 0
        }

    diam = 2.0 * values.astype(float)

    return {
        "diam_mean": float(np.mean(diam)),
        "diam_median": float(np.median(diam)),
        "diam_std": float(np.std(diam)),
        "diam_count": int(diam.size)
    }

# =========================================================
# LONGITUD GT DESDE graph.json SI EXISTE
# =========================================================
def gt_length_from_graph(graph_path):
    if not os.path.exists(graph_path):
        return np.nan

    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    if "edges" not in graph:
        return np.nan

    total_length = 0.0
    for edge in graph["edges"]:
        if isinstance(edge, dict) and "length" in edge:
            total_length += float(edge["length"])

    return float(total_length)

# =========================================================
# MAIN
# =========================================================
results = []
samples = get_sample_folders(BASE_DIR)

print("========================================")
print("Muestras encontradas:", len(samples))
print("BASE_DIR:", BASE_DIR)
print("VOXEL_SIZE:", VOXEL_SIZE)
print("MIN_RADIUS_FOR_CENTERLINE:", MIN_RADIUS_FOR_CENTERLINE)
print("========================================")

for sample in samples:
    folder = os.path.join(BASE_DIR, sample)
    print(f"\nProcesando {sample}...")

    try:
        pred_mask_path = os.path.join(folder, "mask_clean_frangi.npy")
        gt_mask_path = os.path.join(folder, "gt_mask.npy")
        gt_skeleton_path = os.path.join(folder, "gt_skeleton.npy")
        radius_map_path = os.path.join(folder, "radius_map.npy")
        graph_path = os.path.join(folder, "graph.json")

        required = [pred_mask_path, gt_mask_path, gt_skeleton_path, radius_map_path]
        missing = [p for p in required if not os.path.exists(p)]
        if missing:
            print("  [!] Faltan archivos:", missing)
            continue

        pred_mask = load_binary(pred_mask_path)
        gt_mask = load_binary(gt_mask_path)
        gt_skeleton = load_binary(gt_skeleton_path)
        radius_map = np.load(radius_map_path).astype(np.float32)

        # -------------------------------------------------
        # Volumen
        # -------------------------------------------------
        pred_volume = int(pred_mask.sum())
        gt_volume = int(gt_mask.sum())
        volume_error_rel = rel_error(pred_volume, gt_volume)

        # -------------------------------------------------
        # Predicción: eje central aproximado + DT
        # -------------------------------------------------
        pred_centerline, pred_dt = approximate_centerline(
            pred_mask,
            voxel_size=VOXEL_SIZE,
            min_radius=MIN_RADIUS_FOR_CENTERLINE
        )

        pred_centerline_vox = int(pred_centerline.sum())

        # guardar para inspección visual / trazabilidad
        np.save(os.path.join(folder, "pred_centerline_approx.npy"), pred_centerline.astype(np.uint8))

        # -------------------------------------------------
        # Longitud
        # -------------------------------------------------
        pred_length_approx = approximate_length(pred_centerline, voxel_size=VOXEL_SIZE)

        gt_length_graph = gt_length_from_graph(graph_path)
        gt_length_skeleton = approximate_length(gt_skeleton, voxel_size=VOXEL_SIZE)

        # priorizar longitud GT del grafo si existe
        gt_length_final = gt_length_graph if not np.isnan(gt_length_graph) else gt_length_skeleton

        length_error_rel = rel_error(pred_length_approx, gt_length_final)

        # -------------------------------------------------
        # Diámetro
        # -------------------------------------------------
        pred_diam = diameter_stats_from_centerline_and_dt(pred_centerline, pred_dt)
        gt_diam = diameter_stats_gt(gt_skeleton, radius_map)

        diam_mean_error_rel = rel_error(pred_diam["diam_mean"], gt_diam["diam_mean"])
        diam_median_error_rel = rel_error(pred_diam["diam_median"], gt_diam["diam_median"])

        # -------------------------------------------------
        # Guardar métricas
        # -------------------------------------------------
        metrics = {
            "sample": sample,

            "pred_volume_vox": pred_volume,
            "gt_volume_vox": gt_volume,
            "volume_error_rel": volume_error_rel,

            "pred_centerline_vox": pred_centerline_vox,
            "gt_skeleton_vox": int(gt_skeleton.sum()),

            "pred_length_approx": pred_length_approx,
            "gt_length_graph": gt_length_graph,
            "gt_length_skeleton": gt_length_skeleton,
            "gt_length_final": gt_length_final,
            "length_error_rel": length_error_rel,

            "pred_diam_mean": pred_diam["diam_mean"],
            "pred_diam_median": pred_diam["diam_median"],
            "pred_diam_std": pred_diam["diam_std"],
            "pred_diam_count": pred_diam["diam_count"],

            "gt_diam_mean": gt_diam["diam_mean"],
            "gt_diam_median": gt_diam["diam_median"],
            "gt_diam_std": gt_diam["diam_std"],
            "gt_diam_count": gt_diam["diam_count"],

            "diam_mean_error_rel": diam_mean_error_rel,
            "diam_median_error_rel": diam_median_error_rel,
        }

        with open(os.path.join(folder, "geom_metrics_morfologicos.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        results.append(metrics)

        print(f"  pred_volume_vox   : {pred_volume}")
        print(f"  gt_volume_vox     : {gt_volume}")
        print(f"  pred_length_approx: {pred_length_approx:.4f}")
        print(f"  gt_length_final   : {gt_length_final:.4f}" if not np.isnan(gt_length_final) else "  gt_length_final   : nan")
        print(f"  pred_diam_mean    : {pred_diam['diam_mean']:.4f}" if not np.isnan(pred_diam["diam_mean"]) else "  pred_diam_mean    : nan")
        print(f"  gt_diam_mean      : {gt_diam['diam_mean']:.4f}" if not np.isnan(gt_diam["diam_mean"]) else "  gt_diam_mean      : nan")

    except Exception as e:
        print(f"  [ERROR] en {sample}: {e}")
        traceback.print_exc()

# =========================================================
# RESUMEN GLOBAL
# =========================================================
if len(results) > 0:
    df = pd.DataFrame(results)
    out_csv = os.path.join(BASE_DIR, "tabla_geometria_morfologica.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n========================================")
    print("RESUMEN GLOBAL")
    print("========================================")
    print(df.describe(include="all"))
    print("\nGuardado en:", out_csv)
else:
    print("\nNo se generaron resultados.")