import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ENTRENAMIENTO = "data/procesado/entrenamiento.csv"
SEMILLA = 42


def construir_pipeline(clasificador):
    return Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
        ("clasificador", clasificador),
    ])


if __name__ == "__main__":
    df = pd.read_csv(ENTRENAMIENTO)
    X = df.drop(columns=["class"])
    y = df["class"]

    # Peso de la clase positiva (ckd) calculado solo con el entrenamiento
    peso_positivo = (y == 0).sum() / (y == 1).sum()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

    buscadores = {
        "Random Forest": GridSearchCV(
            construir_pipeline(
                RandomForestClassifier(class_weight="balanced", random_state=SEMILLA)
            ),
            param_grid={
                "clasificador__n_estimators": [100, 300],
                "clasificador__max_depth": [3, 5, None],
                "clasificador__min_samples_leaf": [1, 5],
            },
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
        ),
        "XGBoost": GridSearchCV(
            construir_pipeline(
                XGBClassifier(
                    scale_pos_weight=peso_positivo,
                    eval_metric="logloss",
                    random_state=SEMILLA,
                )
            ),
            param_grid={
                "clasificador__n_estimators": [100, 300],
                "clasificador__max_depth": [2, 3, 4],
                "clasificador__learning_rate": [0.05, 0.1],
            },
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
        ),
    }

    for nombre, buscador in buscadores.items():
        print(f"\nOptimizando {nombre}...")
        buscador.fit(X, y)
        print(f"Mejores hiperparámetros: {buscador.best_params_}")
        print(f"AUC medio en validación cruzada: {buscador.best_score_:.3f}")

        resultados = pd.DataFrame(buscador.cv_results_)
        top = resultados.sort_values("rank_test_score").head(5)
        print(top[["params", "mean_test_score", "std_test_score"]].to_string(index=False))