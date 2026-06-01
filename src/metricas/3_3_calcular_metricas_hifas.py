#02_calcular_metricas_hifas.py
import os
import csv
import json
import math
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from scipy.ndimage import distance_transform_edt, convolve, label, generate_binary_structure

# =========================================================
# CONFIGURACIÓN
# =========================================================

DATASET_DIR = r"samples"
DATASET_DIR2 = r"samples"
# Si tus volúmenes están en voxeles isotrópicos, deja 1.0,1.0,1.0
# Si luego usas tamaños físicos reales, por ejemplo (0.5, 0.5, 1.0), cámbialo aquí.
VOXEL_SIZE = (1.0, 1.0, 1.0)  # (x, y, z)

OUTPUT_CSV = os.path.join(DATASET_DIR2, "metricas_hifas.csv")
OUTPUT_JSON = os.path.join(DATASET_DIR2, "metricas_hifas_resumen.json")

# Si existe skeleton.npy lo usa. Si no existe, intenta generarlo desde mask_clean.npy
AUTO_GENERAR_SKELETON = True

# Si el skeleton queda vacío o inválido, el sample se marca con error
STRICT_MODE = False

# =========================================================
# DEPENDENCIAS OPCIONALES
# =========================================================

SKIMAGE_OK = False
SKELETON_MODE = None

try:
    from skimage.morphology import skeletonize_3d
    SKIMAGE_OK = True
    SKELETON_MODE = "skeletonize_3d"
except Exception:
    try:
        from skimage.morphology import skeletonize
        SKIMAGE_OK = True
        SKELETON_MODE = "skeletonize_2d_slice_by_slice"
    except Exception:
        SKIMAGE_OK = False
        SKELETON_MODE = None


# =========================================================
# UTILIDADES BÁSICAS
# =========================================================

def cargar_npy_bool(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.dtype != np.bool_:
        arr = arr > 0
    return arr.astype(bool)


def guardar_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den != 0 else 0.0


def relative_error(est: float, gt: float) -> float:
    return abs(est - gt) / gt if gt > 0 else 0.0


# =========================================================
# MÉTRICAS DE SEGMENTACIÓN
# =========================================================

def metricas_binarias(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()

    dice = safe_div(2 * tp, (2 * tp + fp + fn))
    iou = safe_div(tp, (tp + fp + fn))
    precision = safe_div(tp, (tp + fp))
    recall = safe_div(tp, (tp + fn))

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


# =========================================================
# SKELETON
# =========================================================

def skeleton_from_mask(mask: np.ndarray) -> np.ndarray:
    if not SKIMAGE_OK:
        raise RuntimeError(
            "No se puede generar skeleton automáticamente porque no está instalado scikit-image."
        )

    if SKELETON_MODE == "skeletonize_3d":
        sk = skeletonize_3d(mask.astype(np.uint8)) > 0
        return sk.astype(bool)

    if SKELETON_MODE == "skeletonize_2d_slice_by_slice":
        # Fallback menos ideal, pero funcional si no existe skeletonize_3d
        out = np.zeros_like(mask, dtype=bool)
        for z in range(mask.shape[2]):
            out[:, :, z] = skeletonize(mask[:, :, z].astype(np.uint8)) > 0
        return out.astype(bool)

    raise RuntimeError("No hay método de skeleton disponible.")


# =========================================================
# TOPOLOGÍA DEL SKELETON
# =========================================================

def contar_vecinos_skeleton(skel: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    vecinos = convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    return vecinos


def stats_topologicas_skeleton(skel: np.ndarray) -> Dict[str, int]:
    skel = skel.astype(bool)

    if skel.sum() == 0:
        return {
            "num_skel_voxels": 0,
            "num_tips": 0,
            "num_branchpoints": 0,
            "num_components": 0,
            "num_intermediate": 0,
        }

    vecinos = contar_vecinos_skeleton(skel)

    tips = np.logical_and(skel, vecinos == 1).sum()
    branchpoints = np.logical_and(skel, vecinos >= 3).sum()
    intermediate = np.logical_and(skel, vecinos == 2).sum()

    struct = generate_binary_structure(3, 3)
    labeled, ncomp = label(skel.astype(np.uint8), structure=struct)

    return {
        "num_skel_voxels": int(skel.sum()),
        "num_tips": int(tips),
        "num_branchpoints": int(branchpoints),
        "num_components": int(ncomp),
        "num_intermediate": int(intermediate),
    }


def error_topologico(pred_skel: np.ndarray, gt_skel: np.ndarray) -> Dict[str, float]:
    pred_stats = stats_topologicas_skeleton(pred_skel)
    gt_stats = stats_topologicas_skeleton(gt_skel)

    err_tips = relative_error(pred_stats["num_tips"], gt_stats["num_tips"])
    err_branch = relative_error(pred_stats["num_branchpoints"], gt_stats["num_branchpoints"])
    err_components = relative_error(pred_stats["num_components"], gt_stats["num_components"])

    # Error topológico global promedio
    global_err = (err_tips + err_branch + err_components) / 3.0

    return {
        "pred_num_tips": pred_stats["num_tips"],
        "gt_num_tips": gt_stats["num_tips"],
        "pred_num_branchpoints": pred_stats["num_branchpoints"],
        "gt_num_branchpoints": gt_stats["num_branchpoints"],
        "pred_num_components": pred_stats["num_components"],
        "gt_num_components": gt_stats["num_components"],
        "err_tips": err_tips,
        "err_branchpoints": err_branch,
        "err_components": err_components,
        "error_topologico": global_err,
    }


# =========================================================
# LONGITUD DEL SKELETON
# =========================================================

_NEIGHBOR_OFFSETS = []
_NEIGHBOR_WEIGHTS = []

for dx in [-1, 0, 1]:
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            if dx == 0 and dy == 0 and dz == 0:
                continue
            _NEIGHBOR_OFFSETS.append((dx, dy, dz))


def distancia_offset(dx: int, dy: int, dz: int, voxel_size: Tuple[float, float, float]) -> float:
    sx, sy, sz = voxel_size
    return math.sqrt((dx * sx) ** 2 + (dy * sy) ** 2 + (dz * sz) ** 2)


def longitud_skeleton(skel: np.ndarray, voxel_size: Tuple[float, float, float]) -> float:
    """
    Aproximación de longitud total:
    suma conexiones entre voxeles vecinos del skeleton, dividido entre 2
    para no duplicar aristas.
    """
    skel = skel.astype(bool)
    if skel.sum() == 0:
        return 0.0

    coords = np.argwhere(skel)
    coord_set = set(map(tuple, coords.tolist()))

    total = 0.0
    for x, y, z in coords:
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            nb = (x + dx, y + dy, z + dz)
            if nb in coord_set:
                total += distancia_offset(dx, dy, dz, voxel_size)

    return total / 2.0


# =========================================================
# DIÁMETRO A PARTIR DE DISTANCE TRANSFORM
# =========================================================

def diametro_promedio(mask: np.ndarray, skel: np.ndarray, voxel_size: Tuple[float, float, float]) -> float:
    """
    Calcula diámetro medio como 2 * radio medio medido en el skeleton.
    """
    mask = mask.astype(bool)
    skel = skel.astype(bool)

    if mask.sum() == 0 or skel.sum() == 0:
        return 0.0

    # Distance transform del foreground al background
    dt = distance_transform_edt(mask, sampling=voxel_size)

    radios = dt[skel]
    if radios.size == 0:
        return 0.0

    return float(2.0 * np.mean(radios))


# =========================================================
# PROCESAMIENTO DE SAMPLE
# =========================================================

def resolver_skeleton(sample_dir: str, mask_clean: np.ndarray) -> Optional[np.ndarray]:
    skel_path = os.path.join(sample_dir, "skeleton.npy")
    if os.path.exists(skel_path):
        return cargar_npy_bool(skel_path)

    if AUTO_GENERAR_SKELETON:
        return skeleton_from_mask(mask_clean)

    return None


def procesar_sample(sample_dir: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "sample": os.path.basename(sample_dir),
        "status": "ok",
        "error_msg": "",
    }

    try:
        gt_mask_path = os.path.join(sample_dir, "gt_mask.npy")
        gt_skel_path = os.path.join(sample_dir, "gt_skeleton.npy")
        pred_mask_path = os.path.join(sample_dir, "mask_clean_frangi.npy")

        if not os.path.exists(gt_mask_path):
            raise FileNotFoundError(f"No existe gt_mask.npy en {sample_dir}")
        if not os.path.exists(gt_skel_path):
            raise FileNotFoundError(f"No existe gt_skeleton.npy en {sample_dir}")
        if not os.path.exists(pred_mask_path):
            raise FileNotFoundError(f"No existe mask_clean.npy en {sample_dir}")

        gt_mask = cargar_npy_bool(gt_mask_path)
        gt_skel = cargar_npy_bool(gt_skel_path)
        pred_mask = cargar_npy_bool(pred_mask_path)

        if gt_mask.shape != pred_mask.shape:
            raise ValueError(
                f"gt_mask y mask_clean tienen formas distintas: {gt_mask.shape} vs {pred_mask.shape}"
            )
        if gt_skel.shape != pred_mask.shape:
            raise ValueError(
                f"gt_skeleton y mask_clean tienen formas distintas: {gt_skel.shape} vs {pred_mask.shape}"
            )

        pred_skel = resolver_skeleton(sample_dir, pred_mask)

        if pred_skel is None:
            raise RuntimeError("No se pudo obtener skeleton.npy ni generarlo automáticamente.")

        if pred_skel.shape != pred_mask.shape:
            raise ValueError(
                f"skeleton predicho y máscara tienen formas distintas: {pred_skel.shape} vs {pred_mask.shape}"
            )

        # ------------------------------
        # 1) Métricas voxel
        # ------------------------------
        m_bin = metricas_binarias(pred_mask, gt_mask)

        # ------------------------------
        # 2) Longitud
        # ------------------------------
        pred_length = longitud_skeleton(pred_skel, VOXEL_SIZE)
        gt_length = longitud_skeleton(gt_skel, VOXEL_SIZE)
        err_length = relative_error(pred_length, gt_length)

        # ------------------------------
        # 3) Diámetro
        # ------------------------------
        pred_diam = diametro_promedio(pred_mask, pred_skel, VOXEL_SIZE)
        gt_diam = diametro_promedio(gt_mask, gt_skel, VOXEL_SIZE)
        err_diam = relative_error(pred_diam, gt_diam)

        # ------------------------------
        # 4) Topología
        # ------------------------------
        topo = error_topologico(pred_skel, gt_skel)

        result.update(m_bin)
        result.update({
            "pred_length": pred_length,
            "gt_length": gt_length,
            "error_longitud": err_length,
            "pred_diameter": pred_diam,
            "gt_diameter": gt_diam,
            "error_diametro": err_diam,
        })
        result.update(topo)

    except Exception as e:
        result["status"] = "error"
        result["error_msg"] = str(e)
        if STRICT_MODE:
            raise

    return result


# =========================================================
# MAIN
# =========================================================

def main():
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(f"No existe DATASET_DIR: {DATASET_DIR}")

    sample_dirs = sorted([
        os.path.join(DATASET_DIR, d)
        for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d)) and d.startswith("sample_")
    ])

    if len(sample_dirs) == 0:
        raise RuntimeError("No se encontraron carpetas sample_xxx dentro de DATASET_DIR.")

    resultados: List[Dict[str, Any]] = []

    print("=" * 80)
    print("INICIO DE EVALUACIÓN DE MÉTRICAS")
    print(f"DATASET_DIR: {DATASET_DIR}")
    print(f"VOXEL_SIZE: {VOXEL_SIZE}")
    print(f"SKIMAGE_OK: {SKIMAGE_OK}")
    print(f"SKELETON_MODE: {SKELETON_MODE}")
    print("=" * 80)

    for sample_dir in sample_dirs:
        sample_name = os.path.basename(sample_dir)
        r = procesar_sample(sample_dir)
        resultados.append(r)

        if r["status"] == "ok":
            print(
                f"[OK] {sample_name} | "
                f"Dice={r['dice']:.4f} | IoU={r['iou']:.4f} | "
                f"Prec={r['precision']:.4f} | Recall={r['recall']:.4f} | "
                f"Err_L={r['error_longitud']:.4f} | "
                f"Err_D={r['error_diametro']:.4f} | "
                f"Err_Topo={r['error_topologico']:.4f}"
            )
        else:
            print(f"[ERROR] {sample_name} | {r['error_msg']}")

    # Guardar CSV
    fieldnames = sorted(set(k for r in resultados for k in r.keys()))
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(resultados)

    # Resumen global
    ok_results = [r for r in resultados if r["status"] == "ok"]

    resumen = {
        "num_samples_total": len(resultados),
        "num_samples_ok": len(ok_results),
        "num_samples_error": len(resultados) - len(ok_results),
        "promedios": {},
        "archivos_salida": {
            "csv": OUTPUT_CSV,
            "json": OUTPUT_JSON,
        }
    }

    metricas_resumen = [
        "dice", "iou", "precision", "recall",
        "error_longitud", "error_diametro", "error_topologico"
    ]

    for m in metricas_resumen:
        vals = [float(r[m]) for r in ok_results if m in r]
        resumen["promedios"][m] = float(np.mean(vals)) if len(vals) > 0 else None

    guardar_json(OUTPUT_JSON, resumen)

    print("=" * 80)
    print("PROCESO FINALIZADO")
    print(f"CSV guardado en: {OUTPUT_CSV}")
    print(f"Resumen JSON guardado en: {OUTPUT_JSON}")
    print("Promedios globales:")
    for k, v in resumen["promedios"].items():
        if v is None:
            print(f"  - {k}: None")
        else:
            print(f"  - {k}: {v:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()