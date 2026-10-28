import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ENTRENAMIENTO = "data/procesado/entrenamiento.csv"
PRUEBA = "data/procesado/prueba.csv"
CARPETA_MODELOS = "modelos"
SEMILLA = 42


def construir_pipeline(clasificador):
    return Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
        ("clasificador", clasificador),
    ])


if __name__ == "__main__":
    tr = pd.read_csv(ENTRENAMIENTO)
    te = pd.read_csv(PRUEBA)

    X_tr, y_tr = tr.drop(columns=["class"]), tr["class"]
    X_te, y_te = te.drop(columns=["class"]), te["class"]

    # Pesos calculados solo con el conjunto de entrenamiento
    peso_positivo = (y_tr == 0).sum() / (y_tr == 1).sum()

    modelos = {
        "Regresión logística": construir_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEMILLA)),
        "Random Forest": construir_pipeline(
            RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=1,
                                   class_weight="balanced", random_state=SEMILLA)),
        "XGBoost": construir_pipeline(
            XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=2,
                          scale_pos_weight=peso_positivo,
                          eval_metric="logloss", random_state=SEMILLA)),
    }

    print(f"Entrenamiento: {len(X_tr)} pacientes | Prueba: {len(X_te)} pacientes")
    print("Evaluación final sobre el conjunto de prueba (una sola vez)\n")

    for nombre, modelo in modelos.items():
        modelo.fit(X_tr, y_tr)
        prob = modelo.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()

        print(nombre)
        print("-" * 60)
        print(f"AUC           {roc_auc_score(y_te, prob):.3f}")
        print(f"Exactitud     {accuracy_score(y_te, pred):.3f}")
        print(f"F1            {f1_score(y_te, pred):.3f}")
        print(f"Sensibilidad  {tp / (tp + fn):.3f}")
        print(f"Especificidad {tn / (tn + fp):.3f}")
        print(f"Matriz de confusión: VN={tn}  FP={fp}  FN={fn}  VP={tp}\n")

    os.makedirs(CARPETA_MODELOS, exist_ok=True)
    joblib.dump(modelos["XGBoost"], f"{CARPETA_MODELOS}/modelo_xgboost.joblib")
    print("Modelo XGBoost guardado en modelos/modelo_xgboost.joblib")