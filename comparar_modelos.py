import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ENTRENAMIENTO = "data/procesado/entrenamiento.csv"
SEMILLA = 42

df = pd.read_csv(ENTRENAMIENTO)
X = df.drop(columns=["class"])
y = df["class"]

def preprocesamiento():
    return [
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ]

def construir_modelos(peso_positivo):
    return {
        "Regresión logística": Pipeline(preprocesamiento() + [
            ("clasificador", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=SEMILLA)),
        ]),
        "Random Forest": Pipeline(preprocesamiento() + [
            ("clasificador", RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=SEMILLA)),
        ]),
        "XGBoost": Pipeline(preprocesamiento() + [
            ("clasificador", XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=3,
                scale_pos_weight=peso_positivo,
                eval_metric="logloss", random_state=SEMILLA)),
        ]),
    }

nombres = ["Regresión logística", "Random Forest", "XGBoost"]
metricas = ["auc", "exactitud", "f1", "sensibilidad", "especificidad"]
resultados = {n: {m: [] for m in metricas} for n in nombres}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

for tr, va in cv.split(X, y):
    X_tr, X_va = X.iloc[tr], X.iloc[va]
    y_tr, y_va = y.iloc[tr], y.iloc[va]

    # Peso de la clase positiva calculado solo con el entrenamiento del pliegue
    peso = (y_tr == 0).sum() / (y_tr == 1).sum()

    for nombre, modelo in construir_modelos(peso).items():
        modelo.fit(X_tr, y_tr)
        prob = modelo.predict_proba(X_va)[:, 1]
        pred = (prob >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_va, pred).ravel()
        r = resultados[nombre]
        r["auc"].append(roc_auc_score(y_va, prob))
        r["exactitud"].append(accuracy_score(y_va, pred))
        r["f1"].append(f1_score(y_va, pred))
        r["sensibilidad"].append(tp / (tp + fn))
        r["especificidad"].append(tn / (tn + fp))

for nombre in nombres:
    print(f"\n{nombre}")
    print("-" * 60)
    for m in metricas:
        v = np.array(resultados[nombre][m])
        print(f"{m:<14} media {v.mean():.3f}   desviación {v.std():.3f}")