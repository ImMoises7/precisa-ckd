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
X = df.drop(columns=["class"])
y = df["class"]

modelo = Pipeline([
    ("imputar", SimpleImputer(strategy="median")),
    ("escalar", StandardScaler()),
    ("clasificador", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=SEMILLA,
    )),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

metricas = {
    "auc": "roc_auc",
    "exactitud": "accuracy",
    "f1": "f1",
    "sensibilidad": make_scorer(recall_score, pos_label=1),
    "especificidad": make_scorer(recall_score, pos_label=0),
}

resultados = cross_validate(modelo, X, y, cv=cv, scoring=metricas)

print("Regresión logística (baseline), validación cruzada de 5 pliegues")
print("-" * 60)
for nombre in metricas:
    valores = resultados[f"test_{nombre}"]
    print(f"{nombre:<14} media {valores.mean():.3f}   desviación {valores.std():.3f}")