#3_5_correlacion_geometrica.py
import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr, spearmanr

BASE_DIR = r"samples"
CSV_PATH = os.path.join(BASE_DIR, "3_5_tabla_geometria_morfologica.csv")

df = pd.read_csv(CSV_PATH)

print("\n==============================")
print("ANÁLISIS DE CORRELACIÓN")
print("==============================\n")

# =========================================================
# 1. Error de diámetro vs volumen GT
# =========================================================
x = df["gt_volume_vox"]
y = df["diam_mean_error_rel"]

mask = (~np.isnan(x)) & (~np.isnan(y))

pearson_vol = pearsonr(x[mask], y[mask])
spearman_vol = spearmanr(x[mask], y[mask])

print("Error diámetro vs Volumen GT")
print("Pearson:", pearson_vol)
print("Spearman:", spearman_vol)
print()

# =========================================================
# 2. Error de diámetro vs longitud GT
# =========================================================
x = df["gt_length_final"]
y = df["diam_mean_error_rel"]

mask = (~np.isnan(x)) & (~np.isnan(y))

pearson_len = pearsonr(x[mask], y[mask])
spearman_len = spearmanr(x[mask], y[mask])

print("Error diámetro vs Longitud GT")
print("Pearson:", pearson_len)
print("Spearman:", spearman_len)
print()

# =========================================================
# GUARDAR RESULTADOS
# =========================================================
results = {
    "pearson_vol_r": pearson_vol[0],
    "pearson_vol_p": pearson_vol[1],
    "spearman_vol_r": spearman_vol.correlation,
    "spearman_vol_p": spearman_vol.pvalue,

    "pearson_len_r": pearson_len[0],
    "pearson_len_p": pearson_len[1],
    "spearman_len_r": spearman_len.correlation,
    "spearman_len_p": spearman_len.pvalue,
}

out_path = os.path.join(BASE_DIR, "correlaciones_geometricas.json")

import json
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print("Guardado en:", out_path)