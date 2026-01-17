# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars",
#     "scikit-learn",
#     "xgboost",
#     "pandas",
#     "fastexcel",
#     "pyarrow",
#     "openpyxl",
# ]
# ///

import polars as pl
import pandas as pd
import numpy as np
import os
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# =========================
# CLUSTERING
# =========================
def ajouter_cluster_vehicule(df, n_clusters=5):
    print(f"🔍 Clustering des véhicules ({n_clusters} clusters)")

    features_cluster = [
        "annee",
        "kilometrage",
        "puissance_kw",
        "type_de_vehicule",
        "carburant",
        "boite_de_vitesse",
        "transmission"
    ]
    features_cluster = [c for c in features_cluster if c in df.columns]

    df_cluster = df[features_cluster].copy()
    categorical_cols = df_cluster.select_dtypes(include=["object", "category"]).columns
    df_cluster = pd.get_dummies(df_cluster, columns=categorical_cols)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster_vehicule"] = kmeans.fit_predict(X_scaled)

    print("📊 Répartition des clusters :")
    print(df["cluster_vehicule"].value_counts().sort_index())

    # Sauvegarde clustering
    with open("models/kmeans_cluster.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    with open("models/scaler_cluster.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return df


def analyser_clusters(df):
    resume = (
        df.groupby("cluster_vehicule")
        .agg(
            prix_median=("prix", "median"),
            kilometrage_median=("kilometrage", "median"),
            annee_mediane=("annee", "median"),
            puissance_mediane=("puissance_kw", "median"),
            count=("prix", "count")
        )
        .sort_index()
    )
    print("\n🧠 Profil des clusters :")
    print(resume)
    resume.to_excel("models/analyse_clusters.xlsx")


# =========================
# CHARGEMENT & PRÉPARATION
# =========================
def charger_et_preparer_donnees(fichier="data/processed/autoscout_clean_ml.xlsx"):
    try:
        df = pl.read_excel(fichier).to_pandas()
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        return None, None, None, None

    if "modele_identifie" in df.columns:
        df = df[df["modele_identifie"] == True]

    df = df.dropna(subset=["prix"])

    # 🔹 Clustering
    df = ajouter_cluster_vehicule(df, n_clusters=5)
    analyser_clusters(df)

    y = df["prix"]

    colonnes_a_exclure = [
        "prix",
        "code_postal",
        "pays",
        "le_reste",
        "ville",
        "modele_identifie"
    ]
    X = df.drop(columns=[c for c in colonnes_a_exclure if c in df.columns])

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"📦 Colonnes encodées : {categorical_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = pd.get_dummies(X_train, columns=categorical_cols)
    X_test = pd.get_dummies(X_test, columns=categorical_cols)

    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    assert "cluster_vehicule" in X_train.columns

    return X_train, X_test, y_train, y_test


# =========================
# ÉVALUATION
# =========================
def evaluer_modele(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"RMSE: €{rmse:,.2f} | R²: {r2:.4f} | MAE: €{mae:,.2f}")
    return y_pred, rmse, r2, mae


def cross_validation_model(model, X_train, y_train, cv=5):
    scores = cross_val_score(model, X_train, y_train, scoring="r2", cv=cv, n_jobs=-1)
    print(f"🔁 CV R²: {scores.mean():.4f} ± {scores.std():.4f}")


def enregistrer_erreurs(X_test, y_test, y_pred, fichier):
    df_err = X_test.copy()
    df_err["prix_reel"] = y_test.values
    df_err["prix_predit"] = y_pred
    df_err["erreur_abs"] = np.abs(y_pred - y_test.values)
    df_err["erreur_%"] = 100 * df_err["erreur_abs"] / df_err["prix_reel"]
    df_err.sort_values("erreur_abs", ascending=False).to_excel(fichier, index=False)


# =========================
# MODÈLES
# =========================
def entrainer_random_forest(X_train, X_test, y_train, y_test):
    print("\n🌲 RANDOM FOREST")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred, rmse, r2, mae = evaluer_modele(model, X_test, y_test)
    cross_validation_model(model, X_train, y_train)

    fi = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print(fi.head(10))
    print(fi[fi["feature"] == "cluster_vehicule"])

    with open("models/random_forest_model.pkl", "wb") as f:
        pickle.dump(model, f)

    enregistrer_erreurs(X_test, y_test, y_pred, "models/erreurs_rf.xlsx")
    
    return {
        'model': model,
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'feature_importance': fi
    }


def entrainer_xgboost(X_train, X_test, y_train, y_test):
    print("\n⚡ XGBOOST")
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train)
    y_pred, rmse, r2, mae = evaluer_modele(model, X_test, y_test)
    cross_validation_model(model, X_train, y_train)
    
    fi = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    with open("models/xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)

    enregistrer_erreurs(X_test, y_test, y_pred, "models/erreurs_xgb.xlsx")
    
    return {
        'model': model,
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'feature_importance': fi
    }


# =========================
# MAIN
# =========================
def main():
    os.makedirs("models", exist_ok=True)
    print("🚀 Pipeline ML avec clustering")

    X_train, X_test, y_train, y_test = charger_et_preparer_donnees()
    if X_train is None:
        return

    print(f"📈 Dataset: {X_train.shape[0]} train / {X_test.shape[0]} test")

    results_rf = entrainer_random_forest(X_train, X_test, y_train, y_test)
    results_xgb = entrainer_xgboost(X_train, X_test, y_train, y_test)

    print("\n🏆 Meilleur modèle :", "RF" if results_rf['r2'] > results_xgb['r2'] else "XGB")


if __name__ == "__main__":
    main()
