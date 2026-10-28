import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RUTA = "data/chronic_kidney_disease.arff"
ELIMINAR = ["rbc", "rbcc", "wbcc"]

# Categorías de texto a números (la etiqueta: ckd = 1, notckd = 0)
BINARIAS = {
    "pc": {"normal": 0, "abnormal": 1},
    "pcc": {"notpresent": 0, "present": 1},
    "ba": {"notpresent": 0, "present": 1},
    "htn": {"no": 0, "yes": 1},
    "dm": {"no": 0, "yes": 1},
    "cad": {"no": 0, "yes": 1},
    "appet": {"good": 0, "poor": 1},
    "pe": {"no": 0, "yes": 1},
    "ane": {"no": 0, "yes": 1},
    "class": {"notckd": 0, "ckd": 1},
}

# 1. Leer el archivo
with open(RUTA, encoding="utf-8", errors="replace") as f:
    lineas = f.readlines()

inicio = next(i for i, l in enumerate(lineas) if l.strip().lower() == "@data")
columnas = [l.split()[1].strip("'\"") for l in lineas[:inicio] if l.lower().startswith("@attribute")]

filas = []
for linea in lineas[inicio + 1:]:
    linea = linea.strip()
    if not linea or linea.startswith("%"):
        continue
    valores = [v.strip() for v in linea.split(",")]
    if len(valores) == len(columnas) + 1 and valores[-1] == "":
        valores = valores[:-1]
    if len(valores) == len(columnas):
        filas.append(valores)

df = pd.DataFrame(filas, columns=columnas)
df = df.replace(["?", ""], np.nan)

# 2. Limpieza
df = df.drop(columns=ELIMINAR)
for col in ["sg", "al", "su"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

for col, mapa in BINARIAS.items():
    valores_raros = df[col].dropna()
    valores_raros = valores_raros[~valores_raros.isin(mapa.keys())].unique()
    if len(valores_raros) > 0:
        print(f"Aviso: valores no reconocidos en {col}: {valores_raros}")
    df[col] = df[col].map(mapa)

df = df.apply(pd.to_numeric, errors="coerce")
print("Dataset limpio:", df.shape)

# 3. Poder de separación de cada variable (AUC de una sola variable)
y = df["class"]
X = df.drop(columns=["class"])

resultados = []
for col in X.columns:
    mascara = X[col].notna() & y.notna()
    if mascara.sum() < 50:
        continue
    auc = roc_auc_score(y[mascara], X[col][mascara])
    separacion = max(auc, 1 - auc)  # no importa la dirección, solo qué tan bien separa
    media_sin = X[col][y == 0].mean()
    media_con = X[col][y == 1].mean()
    resultados.append((col, separacion, media_sin, media_con))

resultados.sort(key=lambda r: r[1], reverse=True)

print("\nVariable   AUC (separación)   Media sin ERC   Media con ERC")
print("-" * 60)
for col, sep, m0, m1 in resultados:
    marca = "  <-- revisar" if sep >= 0.95 else ""
    print(f"{col:<10} {sep:>10.3f}      {m0:>12.2f}    {m1:>12.2f}{marca}")