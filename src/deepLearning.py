# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars",
#     "scikit-learn",
#     "pandas",
#     "matplotlib",
#     "seaborn",
#     "pyarrow",
# ]
# ///

import polars as pl
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# =========================
# 1. PRÉPARATION DES DONNÉES
# =========================
def preparer_donnees_nn(fichier="data/processed/autoscout_clean_ml.json"):
    print("📦 Chargement des données...")
    try:
        df = pl.read_json(fichier).to_pandas()
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return None, None, None, None

    # Sécurité
    df = df.dropna(subset=["prix"])

    # Target
    y = df["prix"]

    # Features
    X = df.drop(columns=["prix", "code_postal", "ville"], errors="ignore")

    # Encodage catégoriel (même logique que RF / XGB)
    categorical_cols = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    X = pd.get_dummies(X, columns=categorical_cols)

    # Split identique aux autres modèles
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Alignement strict des colonnes
    X_train, X_test = X_train.align(
        X_test, join="left", axis=1, fill_value=0
    )

    print(f"✅ Données prêtes : {X_train.shape[0]} train / {X_test.shape[0]} test")
    print(f"📊 Nombre de features : {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


# =========================
# 2. GRID SEARCH & ENTRAÎNEMENT
# =========================
def optimiser_et_entrainer_mlp(X_train, X_test, y_train, y_test):
    print("\n🔍 Grid Search – Réseau de Neurones")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            max_iter=2000,
            early_stopping=True,
            random_state=42
        ))
    ])

    param_grid = {
        "mlp__hidden_layer_sizes": [(128, 64)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [0.01],
        "mlp__learning_rate_init": [0.001, 0.01]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1
    )

    print("🚀 Entraînement en cours...")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"\n✅ Meilleurs paramètres : {grid.best_params_}")
    # --- Évaluation Train ---
    y_train_pred = best_model.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)


    # --- Évaluation Test ---
    y_pred = best_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("\n🏆 PERFORMANCE RÉSEAU DE NEURONES")
    print(f"R² Train : {r2_train:.4f}")
    print(f"R² Test  : {r2:.4f}")
    print(f"RMSE     : €{rmse:,.2f}")
    print(f"MAE      : €{mae:,.2f}")


    # Cross-validation (comme RF / XGB)
    cv_scores = cross_val_score(
        best_model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1
    )
    print(f"🔁 CV R² : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Sauvegarde
    os.makedirs("models", exist_ok=True)
    with open("models/mlp_regressor.pkl", "wb") as f:
        pickle.dump(best_model, f)

    return best_model, y_pred


# =========================
# 3. MAIN
# =========================
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preparer_donnees_nn()

    if X_train is not None:
        best_mlp, y_pred = optimiser_et_entrainer_mlp(
            X_train, X_test, y_train, y_test
        )

        # Visualisation
        plt.figure(figsize=(10, 6))
        sns.regplot(
            x=y_test,
            y=y_pred,
            scatter_kws={"alpha": 0.3},
            line_kws={"color": "red"}
        )
        plt.title("Réseau de Neurones – Prix réel vs prix prédit")
        plt.xlabel("Prix réel (€)")
        plt.ylabel("Prix prédit (€)")
        plt.tight_layout()
        plt.show()
