#07_evaluar_tips_branchpoints26_tabla2
import os
import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from skimage.morphology import skeletonize

# =========================================================
# CONFIGURACIÓN
# =========================================================
DATASET_DIR = r"samples"
OUTPUT_CSV = os.path.join(DATASET_DIR, "tabla2_topologia_pred_vs_gt_conectividad26.csv")

print("DATASET_DIR =", DATASET_DIR)
print("Existe DATASET_DIR:", os.path.exists(DATASET_DIR))

# =========================================================
# KERNEL 3D PARA CONECTIVIDAD 26
# =========================================================
kernel26 = np.ones((3, 3, 3), dtype=np.uint8)
kernel26[1, 1, 1] = 0

# =========================================================
# FUNCIONES
# =========================================================
def skeleton_from_mask_3d_slice_by_slice(mask):
    mask = (mask > 0)
    skel = np.zeros_like(mask, dtype=np.uint8)

    for z in range(mask.shape[0]):
        skel[z] = skeletonize(mask[z]).astype(np.uint8)

    return skel


def analizar_skeleton_conectividad26(skel: np.ndarray):
    skel = (skel > 0).astype(np.uint8)

    vecinos = convolve(skel, kernel26, mode="constant", cval=0)
    vecinos_skel = vecinos[skel > 0]

    num_tips = int(np.sum(vecinos_skel == 1))
    num_intermediate = int(np.sum(vecinos_skel == 2))
    num_branchpoints = int(np.sum(vecinos_skel >= 3))
    num_isolated = int(np.sum(vecinos_skel == 0))
    num_nodes = int(len(vecinos_skel))

    return {
        "num_nodes": num_nodes,
        "num_tips": num_tips,
        "num_branchpoints": num_branchpoints,
        "num_intermediate_nodes": num_intermediate,
        "num_isolated_nodes": num_isolated
    }


def error_relativo(pred, gt):
    if gt == 0:
        return np.nan
    return abs(pred - gt) / gt


# =========================================================
# LISTAR SAMPLES
# =========================================================
samples = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d)) and d.startswith("sample_")
])

print("\nSamples encontrados:", len(samples))

resultados = []

# =========================================================
# RECORRER DATASET
# =========================================================
for sample in samples:
    sample_dir = os.path.join(DATASET_DIR, sample)

    pred_mask_path = os.path.join(sample_dir, "mask_clean_frangi.npy")
    gt_path = os.path.join(sample_dir, "gt_skeleton.npy")

    if not os.path.exists(pred_mask_path):
        print(f"[WARNING] No se encontró mask_clean_frangi.npy en {sample}")
        continue

    if not os.path.exists(gt_path):
        print(f"[WARNING] No se encontró gt_skeleton.npy en {sample}")
        continue

    try:
        pred_mask = np.load(pred_mask_path)
        gt_skel = np.load(gt_path)

        pred_skel = skeleton_from_mask_3d_slice_by_slice(pred_mask)

        pred = analizar_skeleton_conectividad26(pred_skel)
        gt = analizar_skeleton_conectividad26(gt_skel)

        err_tips = error_relativo(pred["num_tips"], gt["num_tips"])
        err_branch = error_relativo(pred["num_branchpoints"], gt["num_branchpoints"])
        err_nodes = error_relativo(pred["num_nodes"], gt["num_nodes"])

        errores_validos = [e for e in [err_tips, err_branch, err_nodes] if not np.isnan(e)]
        error_topologico = float(np.mean(errores_validos)) if len(errores_validos) > 0 else np.nan

        fila = {
            "sample": sample,
            "pred_num_nodes": pred["num_nodes"],
            "gt_num_nodes": gt["num_nodes"],
            "pred_num_tips": pred["num_tips"],
            "gt_num_tips": gt["num_tips"],
            "pred_num_branchpoints": pred["num_branchpoints"],
            "gt_num_branchpoints": gt["num_branchpoints"],
            "pred_num_intermediate_nodes": pred["num_intermediate_nodes"],
            "gt_num_intermediate_nodes": gt["num_intermediate_nodes"],
            "pred_num_isolated_nodes": pred["num_isolated_nodes"],
            "gt_num_isolated_nodes": gt["num_isolated_nodes"],
            "err_tips": err_tips,
            "err_branchpoints": err_branch,
            "err_nodes": err_nodes,
            "error_topologico": error_topologico
        }

        resultados.append(fila)

        print(
            f"[OK] {sample}: "
            f"tips_pred={pred['num_tips']}, tips_gt={gt['num_tips']}, "
            f"branch_pred={pred['num_branchpoints']}, branch_gt={gt['num_branchpoints']}, "
            f"err_topo={error_topologico:.4f}"
        )

    except Exception as e:
        print(f"[ERROR] {sample}: {e}")

# =========================================================
# GUARDAR RESULTADOS
# =========================================================
df = pd.DataFrame(resultados)

if len(df) > 0:
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\nResumen general:")
    cols_resumen = [
        "pred_num_nodes", "gt_num_nodes",
        "pred_num_tips", "gt_num_tips",
        "pred_num_branchpoints", "gt_num_branchpoints",
        "error_topologico", "err_tips", "err_branchpoints", "err_nodes"
    ]
    print(df[cols_resumen].describe())

    print("\nPromedios:")
    print("Mean pred nodes =", df["pred_num_nodes"].mean())
    print("Mean gt nodes =", df["gt_num_nodes"].mean())
    print("Mean pred tips =", df["pred_num_tips"].mean())
    print("Mean gt tips =", df["gt_num_tips"].mean())
    print("Mean pred branchpoints =", df["pred_num_branchpoints"].mean())
    print("Mean gt branchpoints =", df["gt_num_branchpoints"].mean())
    print("Mean error_topologico =", df["error_topologico"].mean())
    print("Mean err_tips =", df["err_tips"].mean())
    print("Mean err_branchpoints =", df["err_branchpoints"].mean())
    print("Mean err_nodes =", df["err_nodes"].mean())

    print("\nArchivo guardado en:", OUTPUT_CSV)
else:
    print("\nNo se generaron resultados.")