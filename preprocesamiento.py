import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RUTA = "data/chronic_kidney_disease.arff"
CARPETA_SALIDA = "data/procesado"
SEMILLA = 42
PORCENTAJE_PRUEBA = 0.15

ELIMINAR = ["rbc", "rbcc", "wbcc"]
NUMERICAS_EN_TEXTO = ["sg", "al", "su"]

# Categorías de texto a números
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


def cargar_arff(ruta):
    with open(ruta, encoding="utf-8", errors="replace") as f:
        lineas = f.readlines()

    inicio = next(i for i, l in enumerate(lineas) if l.strip().lower() == "@data")
    columnas = [
        l.split()[1].strip("'\"")
        for l in lineas[:inicio]
        if l.lower().startswith("@attribute")
    ]

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
    return df.replace(["?", ""], np.nan)


def limpiar(df):
    df = df.drop(columns=ELIMINAR)

    for col in NUMERICAS_EN_TEXTO:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col, mapa in BINARIAS.items():
        no_reconocidos = df[col].dropna()
        no_reconocidos = no_reconocidos[~no_reconocidos.isin(mapa.keys())].unique()
        if len(no_reconocidos) > 0:
            print(f"Aviso: valores no reconocidos en {col}: {no_reconocidos}")
        df[col] = df[col].map(mapa)

    return df.apply(pd.to_numeric, errors="coerce")


if __name__ == "__main__":
    df = limpiar(cargar_arff(RUTA))
    print("Dataset limpio:", df.shape)

    X = df.drop(columns=["class"])
    y = df["class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=PORCENTAJE_PRUEBA,
        stratify=y,
        random_state=SEMILLA,
    )

    entrenamiento = X_train.copy()
    entrenamiento["class"] = y_train
    prueba = X_test.copy()
    prueba["class"] = y_test

    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    entrenamiento.to_csv(f"{CARPETA_SALIDA}/entrenamiento.csv", index=False)
    prueba.to_csv(f"{CARPETA_SALIDA}/prueba.csv", index=False)

    print(f"\nEntrenamiento (85 %, con validación cruzada): {len(X_train)} pacientes")
    print(y_train.value_counts())
    print(f"\nPrueba (15 %, reservada hasta la evaluación final): {len(X_test)} pacientes")
    print(y_test.value_counts())