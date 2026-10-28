import numpy as np
import pandas as pd

RUTA = "data/chronic_kidney_disease.arff"

with open(RUTA, encoding="utf-8", errors="replace") as f:
    lineas = f.readlines()

inicio = next(i for i, l in enumerate(lineas) if l.strip().lower() == "@data")

# Nombres de columnas (sin comillas) y tipo de cada una
columnas = []
es_numerica = []
for l in lineas[:inicio]:
    if l.lower().startswith("@attribute"):
        partes = l.split()
        columnas.append(partes[1].strip("'\""))
        es_numerica.append(partes[2].lower() in ("numeric", "real", "integer"))
print("Columnas encontradas:", len(columnas))

# Leer las filas
filas = []
for num, linea in enumerate(lineas[inicio + 1:], start=inicio + 2):
    linea = linea.strip()
    if not linea or linea.startswith("%"):
        continue
    valores = [v.strip() for v in linea.split(",")]
    # Filas con una coma final: quitamos el valor vacío de más
    if len(valores) == len(columnas) + 1 and valores[-1] == "":
        valores = valores[:-1]
    if len(valores) != len(columnas):
        print(f"Fila descartada, línea {num}: {len(valores)} valores")
        continue
    filas.append(valores)

df = pd.DataFrame(filas, columns=columnas)
df = df.replace(["?", ""], np.nan)

# Convertir a número solo las columnas numéricas
for col, numerica in zip(columnas, es_numerica):
    if numerica:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nDimensiones (filas, columnas):", df.shape)
print("\nPrimeras filas:")
print(df.head())
print("\nTipos de datos:")
print(df.dtypes)
print("\nValores nulos por columna:")
print(df.isna().sum())
print("\nDistribución de la etiqueta (class):")
print(df["class"].value_counts())