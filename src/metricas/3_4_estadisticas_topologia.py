#3_4_estadisticas
import pandas as pd

# =========================
# RUTA DEL CSV
# =========================
archivo = r"resultados\3_4_tabla2_topologia_pred_vs_gt_conectividad6.csv"

df = pd.read_csv(archivo)

print("\n===== RESULTADOS TOPOLOGÍA =====\n")

# MÉTRICA PRINCIPAL
mean_topo = df["error_topologico"].mean()
std_topo = df["error_topologico"].std()

print(f"error_topologico = {mean_topo:.4f} ± {std_topo:.4f}")

# DESGLOSE (opcional pero recomendado)
print("\n--- Desglose ---")
print("err_tips =", df["err_tips"].mean())
print("err_branchpoints =", df["err_branchpoints"].mean())
print("err_nodes =", df["err_nodes"].mean())