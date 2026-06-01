#estadistica_frangi.py
import pandas as pd
from scipy.stats import wilcoxon
import os

# =========================
# RUTA DEL ARCHIVO
# =========================
ruta = r"resultados"
archivo = os.path.join(ruta, "metricas_hifas_frangi.csv")

# =========================
# CARGAR CSV
# =========================
df = pd.read_csv(archivo)

# =========================
# MÉTRICAS A ANALIZAR
# =========================
metrics = [
    "dice",
    "iou",
    "precision",
    "recall",
    "error_longitud",
    "error_diametro"
]

print("\n===== ESTADÍSTICA DESCRIPTIVA =====\n")

for m in metrics:
    mean = df[m].mean()
    std = df[m].std()
    print(f"{m}: {mean:.4f} ± {std:.4f}")

print("\n===== PRUEBA WILCOXON =====\n")

for m in metrics:
    diferencias = df[m] - df[m].median()
    stat, p = wilcoxon(diferencias)
    print(f"{m}: estadístico={stat:.4f}, p-value={p:.5f}")