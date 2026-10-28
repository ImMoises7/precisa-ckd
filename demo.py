import joblib
import pandas as pd

MODELO = "modelos/modelo_xgboost.joblib"
PRUEBA = "data/procesado/prueba.csv"
UMBRAL = 0.5

# Nombres legibles para las variables que se muestran en la ficha
ETIQUETAS = {
    "age": "Edad (años)",
    "bp": "Presión arterial",
    "sg": "Densidad urinaria",
    "al": "Albúmina",
    "sc": "Creatinina sérica",
    "hemo": "Hemoglobina",
    "bgr": "Glucosa",
}


def formato(valor):
    if pd.isna(valor):
        return "no registrado"
    if float(valor).is_integer():
        return str(int(valor))
    return str(valor)


if __name__ == "__main__":
    modelo = joblib.load(MODELO)
    prueba = pd.read_csv(PRUEBA)
    X = prueba.drop(columns=["class"])
    y = prueba["class"]

    # Dos pacientes con ERC y uno sin ERC, tomados del conjunto de prueba
    indices = list(y[y == 1].index[:2]) + list(y[y == 0].index[:1])
    probs = modelo.predict_proba(X.loc[indices])[:, 1]

    print("=" * 60)
    print("DEMO: diagnóstico de ERC con XGBoost")
    print("=" * 60)

    for num, (idx, prob) in enumerate(zip(indices, probs), start=1):
        real_erc = y[idx] == 1
        pred_erc = prob >= UMBRAL
        print(f"\nPaciente {num}")
        print("-" * 60)
        for col, nombre in ETIQUETAS.items():
            print(f"  {nombre:<24}{formato(X.loc[idx, col])}")
        print()
        print(f"  Diagnóstico real:            {'ERC' if real_erc else 'Sin ERC'}")
        print(f"  Probabilidad estimada de ERC: {prob:.1%}")
        print(f"  Predicción del modelo:       {'ERC' if pred_erc else 'Sin ERC'}")
        print(f"  ¿Coincide con el real?:      {'Sí' if pred_erc == real_erc else 'No'}")

    print("\n" + "=" * 60)
    print("Estos pacientes no se usaron para entrenar el modelo.")
    print("=" * 60)