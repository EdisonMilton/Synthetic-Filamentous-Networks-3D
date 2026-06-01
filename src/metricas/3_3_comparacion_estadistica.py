#06_comparacion_estadistica.py
import pandas as pd
from scipy.stats import wilcoxon

# =========================
# CARGAR DATOS
# =========================

frangi = pd.read_csv(r"\RESULTADOS\metricas_hifas_frangi.csv")

otsu = pd.read_csv(r"\RESULTADOS\metricas_hifas_otsu.csv")

# =========================
# MÉTRICAS
# =========================
metrics = ["dice", "iou", "precision", "recall"]

print("\n===== COMPARACIÓN FRANGI vs OTSU =====\n")

for m in metrics:

    f = frangi[m]
    o = otsu[m]

    stat, p = wilcoxon(f, o)

    print(f"{m.upper()}:")
    print(f"  Frangi mean = {f.mean():.4f}")
    print(f"  Otsu   mean = {o.mean():.4f}")
    print(f"  p-value = {p:.6f}")

    if p < 0.05:
        print("  ✅ Diferencia significativa")
    else:
        print("  ⚠️ No significativa")

    print("-" * 40)