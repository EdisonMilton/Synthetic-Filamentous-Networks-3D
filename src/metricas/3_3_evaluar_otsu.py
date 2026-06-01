#05_evaluar_otsu.py
import os
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure

# =========================
# CONFIG
# =========================
DATASET_DIR = r"resultados"
OUTPUT_CSV = os.path.join(DATASET_DIR, "RESULTADOS", "metricas_hifas_otsu.csv")

# =========================
# MÉTRICAS
# =========================
def calcular_metricas(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return dice, iou, precision, recall


# =========================
# LIMPIEZA MORFOLÓGICA
# =========================
def limpiar(mask):
    struct = generate_binary_structure(3, 1)
    mask = binary_opening(mask, structure=struct)
    mask = binary_closing(mask, structure=struct)
    return mask


# =========================
# LOOP
# =========================
resultados = []

samples = sorted([
    d for d in os.listdir(DATASET_DIR)
    if d.startswith("sample_") and os.path.isdir(os.path.join(DATASET_DIR, d))
])

for s in samples:
    sample_dir = os.path.join(DATASET_DIR, s)

    volume_path = os.path.join(sample_dir, "volume.npy")
    gt_path = os.path.join(sample_dir, "gt_mask.npy")

    if not os.path.exists(volume_path) or not os.path.exists(gt_path):
        print(f"{s} ⚠️ falta volume.npy o gt_mask.npy")
        continue

    try:
        volume = np.load(volume_path)
        gt = np.load(gt_path)

        # =========================
        # OTSU
        # =========================
        thresh = threshold_otsu(volume)
        pred = volume > thresh
        pred = limpiar(pred)

        # =========================
        # MÉTRICAS
        # =========================
        dice, iou, precision, recall = calcular_metricas(gt, pred)

        resultados.append({
            "sample": s,
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall
        })

        print(f"{s} ✔ dice={dice:.4f} iou={iou:.4f}")

    except Exception as e:
        print(f"{s} ❌ error: {e}")

# =========================
# GUARDAR CSV
# =========================
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

df = pd.DataFrame(resultados)
df.to_csv(OUTPUT_CSV, index=False)

print("\n🔥 CSV generado:", OUTPUT_CSV)
print(f"Total de samples evaluados: {len(df)}")