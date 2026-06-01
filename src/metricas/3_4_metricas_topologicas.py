#topologia.py
import os
import json
import pandas as pd

BASE_DIR = r"samples"
UMBRAL = 15

resultados = []

for sample in os.listdir(BASE_DIR):

    sample_path = os.path.join(BASE_DIR, sample)

    if not os.path.isdir(sample_path):
        continue

    graph_path = os.path.join(sample_path, "graph.json")

    if not os.path.exists(graph_path):
        continue

    with open(graph_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    edges_filtradas = []
    nodos_validos = set()

    for e in data["edges"]:
        if e["length"] >= UMBRAL:
            edges_filtradas.append(e)
            nodos_validos.add(e["parent_node"])
            nodos_validos.add(e["child_node"])

    nodos_filtrados = [n for n in data["nodes"] if n["id"] in nodos_validos]

    num_nodes = len(nodos_filtrados)
    num_edges = len(edges_filtradas)
    num_tips = sum(1 for n in nodos_filtrados if n["tipo"] == "endpoint")
    num_branchpoints = sum(1 for n in nodos_filtrados if n["tipo"] == "branchpoint")

    resultados.append({
        "sample": sample,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_tips": num_tips,
        "num_branchpoints": num_branchpoints
    })

df = pd.DataFrame(resultados)

cols = ["num_nodes", "num_edges", "num_tips", "num_branchpoints"]

print("\n===== RESULTADO FINAL =====")
print(df[cols].mean())

print("\n===== DESVIACIÓN ESTÁNDAR =====")
print(df[cols].std())

output_csv = os.path.join(BASE_DIR, "topologia_filtrada.csv")
df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print("\nArchivo guardado en:")
print(output_csv)