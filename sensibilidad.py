import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ENTRENAMIENTO = "data/procesado/entrenamiento.csv"
SEMILLA = 42

df = pd.read_csv(ENTRENAMIENTO)
y = df["class"]

metricas = {
    "auc": "roc_auc",
    "exactitud": "accuracy",
    "f1": "f1",
    "sensibilidad": make_scorer(recall_score, pos_label=1),
    "especificidad": make_scorer(recall_score, pos_label=0),
}

def evaluar(nombre, excluir):
    X = df.drop(columns=["class"] + excluir)
    modelo = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
        ("clasificador", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEMILLA)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    r = cross_validate(modelo, X, y, cv=cv, scoring=metricas)

    print(f"\n{nombre}  ({X.shape[1]} variables)")
    print("-" * 60)
    for m in metricas:
        v = r[f"test_{m}"]
        print(f"{m:<14} media {v.mean():.3f}   desviación {v.std():.3f}")

evaluar("Con todas las variables", excluir=[])
evaluar("Sin hemo y pcv", excluir=["hemo", "pcv"])
evaluar("Sin hemo, pcv y sc (creatinina)", excluir=["hemo", "pcv", "sc"])