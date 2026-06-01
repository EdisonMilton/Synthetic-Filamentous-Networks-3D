#3_5_tabla_geometria_morfologica.py
import pandas as pd

df = pd.read_csv(r"resultados\3_5_tabla_geometria_morfologica.csv")
print("\n=== DIÁMETRO (MEDIA) ===")
print("mean:", df["diam_mean_error_rel"].mean())
print("std:", df["diam_mean_error_rel"].std())
print("median:", df["diam_mean_error_rel"].median())

print("\n=== DIÁMETRO (MEDIANA) ===")
print("mean:", df["diam_median_error_rel"].mean())
print("std:", df["diam_median_error_rel"].std())
print("median:", df["diam_median_error_rel"].median())