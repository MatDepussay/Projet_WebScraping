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
#     "matplotlib",
#     "seaborn",
# ]
# ///

import polars as pl
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer # A NE SURTOUT PAS ENLEVER
from sklearn.impute import IterativeImputer, SimpleImputer


# =========================
# 1. CLUSTERING
# =========================
def ajouter_cluster_vehicule(df, n_clusters=5):
    """
    Crée un clustering robuste en utilisant un Pipeline Scikit-Learn.
    Supporte Polars et Pandas.
    """
    print(f"🔍 Clustering des véhicules ({n_clusters} clusters)...")
    
    # Conversion en Pandas pour sklearn (plus stable pour les pipelines complexes)
    data_pd = df.to_pandas() if isinstance(df, pl.DataFrame) else df.copy()

    num_features = ["annee", "kilometrage", "puissance_kw", "cylindree_l"]
    cat_features = ["carburant", "boite_de_vitesse", "transmission", "type_de_vehicule"]

    # 1. Pipeline numérique avec Imputation Itérative (basée sur la régression)
    # L'imputer va apprendre les corrélations entre puissance, année et cylindrée
    num_pipeline = Pipeline([
        ('imputer', IterativeImputer(max_iter=10, random_state=42)),
        ('scaler', StandardScaler())
    ])

    # 2. Pipeline catégoriel (on garde most_frequent car l'imputation itérative est complexe sur le texte)
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_features),
            ('cat', cat_pipeline, cat_features)
        ])

    cluster_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('kmeans', KMeans(n_clusters=n_clusters, random_state=42, n_init=10))
    ])

    clusters = cluster_pipeline.fit_predict(data_pd)
    
    # Ajout du résultat au DataFrame d'origine
    if isinstance(df, pl.DataFrame):
        df = df.with_columns(pl.Series("cluster_vehicule", clusters))
    else:
        df["cluster_vehicule"] = clusters

    # Sauvegarde du pipeline COMPLET (mieux que 2 fichiers séparés)
    Path("models").mkdir(parents=True, exist_ok=True)
    with open("models/cluster_full_pipeline.pkl", "wb") as f:
        pickle.dump(cluster_pipeline, f)
    print(df['cluster_vehicule'].value_counts())
    print("✅ Clustering terminé et pipeline sauvegardé.")
    return df

# =========================
# 2. ANALYSE ET VISUALISATION
# =========================
def analyser_clusters(df):
    """
    Analyse statistique et graphique des clusters générés.
    """
    # Conversion pour l'analyse
    data_pd = df.to_pandas() if isinstance(df, pl.DataFrame) else df

    # 1. Statistiques descriptives
    resume = data_pd.groupby("cluster_vehicule").agg({
        "prix": ["median", "mean"],
        "kilometrage": "median",
        "annee": "median",
        "puissance_kw": "median"
    }).round(0)
    
    print("\n🧠 Profil statistique des clusters :")
    print(resume)

    # 2. Visualisation des distributions (Boxplots)
    cols_a_voir = ["prix", "kilometrage", "annee", "puissance_kw"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, col in enumerate(cols_a_voir):
        if col in data_pd.columns:
            sns.boxplot(x="cluster_vehicule", y=col, data=data_pd, ax=axes[i], hue="cluster_vehicule", palette="Set2", legend=False)
            axes[i].set_title(f"Distribution de {col}")
    
    plt.tight_layout()
    plt.show()

    # 3. Visualisation 2D (PCA)
    visualiser_pca(data_pd)

def visualiser_pca(df_pd):
    """ Projection des clusters en 2D pour vérifier la séparation """
    try:
        # On recharge le pipeline pour transformer les données
        with open("models/cluster_full_pipeline.pkl", "rb") as f:
            pipeline = pickle.load(f)
        
        # Transformation des données par le preprocessor du pipeline
        X_transformed = pipeline.named_steps['preprocessor'].transform(df_pd)
        
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_transformed)
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=df_pd["cluster_vehicule"], palette="deep", alpha=0.5)
        plt.title("Séparation des clusters (Projection PCA)")
        plt.show()
    except Exception as e:
        print(f"Impossible de générer la PCA : {e}")

def trouver_meilleur_k(df, max_k=10):
    from sklearn.cluster import KMeans
    data_pd = df.to_pandas() if isinstance(df, pl.DataFrame) else df
    
    # Prétraitement rapide
    num_features = ["annee", "kilometrage", "puissance_kw", "prix"]
    X = StandardScaler().fit_transform(data_pd[num_features].dropna())
    
    inertias = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
    
    plt.plot(range(1, max_k + 1), inertias, 'go-')
    plt.title("Méthode du Coude (Elbow)")
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Inertie")
    plt.show()

# =========================
# CHARGEMENT & PRÉPARATION
# =========================
def charger_et_preparer_donnees(fichier="data/processed/autoscout_clean_ml.json"):
    try:
        df = pl.read_json(fichier).to_pandas()
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        return None, None, None, None

    if "modele_identifie" in df.columns:
        df = df[df["modele_identifie"]]

    df = df.dropna(subset=["prix"])

    # 🔹 1. Clustering
    df = ajouter_cluster_vehicule(df, n_clusters=5)
    
    # 🔹 2. AJOUT ICI : Conversion en string pour que get_dummies le traite en catégories
    df["cluster_vehicule"] = df["cluster_vehicule"].astype(str) 
    
    analyser_clusters(df)

    # 🔹 3. Préparation des variables
    y = df["prix"]

    colonnes_a_exclure = [
        "prix",
        "code_postal",
        "ville",
    ]
    X = df.drop(columns=[c for c in colonnes_a_exclure if c in df.columns])

    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    print(f"📦 Colonnes encodées : {categorical_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = pd.get_dummies(X_train, columns=categorical_cols)
    X_test = pd.get_dummies(X_test, columns=categorical_cols)

    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # Cherche n'importe quelle colonne qui commence par "cluster_vehicule"
    assert any("cluster_vehicule" in col for col in X_train.columns)

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
def tune_random_forest(X_train, y_train):
    print("\n🔍 Tuning Hyperparamètres : RANDOM FOREST")
    
    param_dist = {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # n_iter=10 teste 10 combinaisons au hasard
    search = RandomizedSearchCV(
        rf, param_distributions=param_dist, 
        n_iter=10, cv=3, scoring='r2', verbose=1, random_state=42, n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    print(f"✅ Meilleurs paramètres RF: {search.best_params_}")
    return search.best_estimator_

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

def tune_xgboost(X_train, y_train):
    print("\n🔍 Tuning Hyperparamètres : XGBOOST")
    
    param_dist = {
        'n_estimators': [200, 500, 1000],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2]
    }
    
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
    
    search = RandomizedSearchCV(
        xgb_model, param_distributions=param_dist, 
        n_iter=10, cv=3, scoring='r2', verbose=1, random_state=42, n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    print(f"✅ Meilleurs paramètres XGB: {search.best_params_}")
    return search.best_estimator_

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
    print("🚀 Démarrage du Pipeline ML")

    fichier = "data/processed/autoscout_clean_ml.json"
    rf_path = "models/best_rf_final.pkl"
    xgb_path = "models/best_xgb_final.pkl"
    
    # 1. Préparation des données
    X_train, X_test, y_train, y_test = charger_et_preparer_donnees(fichier)
    
    if X_train is None:
        return
    
    print(f"💾 Sauvegarde de la liste des {len(X_train.columns)} colonnes...")
    with open("models/model_features.pkl", "wb") as f:
        pickle.dump(X_train.columns.tolist(), f)
        
    print(f"📈 Dataset prêt: {X_train.shape[0]} train / {X_test.shape[0]} test")

    # 2. CHARGEMENT ou TUNING
    if os.path.exists(rf_path) and os.path.exists(xgb_path):
        print("♻️  Modèles optimisés trouvés. Chargement en cours...")
        with open(rf_path, "rb") as f:
            best_rf_model = pickle.load(f)
        with open(xgb_path, "rb") as f:
            best_xgb_model = pickle.load(f)
        # On récupère les colonnes que XGBoost attend
        expected_features = best_xgb_model.get_booster().feature_names
        
        # On force X_test à avoir EXACTEMENT ces colonnes (et dans le bon ordre)
        # Les colonnes manquantes sont remplies par 0, les colonnes en trop sont supprimées
        X_test = X_test.reindex(columns=expected_features, fill_value=0)
        X_train = X_train.reindex(columns=expected_features, fill_value=0) # Sécurité pour la CV
    else:
        print("🚀 Aucun modèle trouvé. Lancement du Tuning...")
        best_rf_model = tune_random_forest(X_train, y_train)
        best_xgb_model = tune_xgboost(X_train, y_train)
        
        # Sauvegarde immédiate
        with open(rf_path, "wb") as f:
            pickle.dump(best_rf_model, f)
        with open(xgb_path, "wb") as f:
            pickle.dump(best_xgb_model, f)
        print("✅ Modèles tunés sauvegardés.")

    # --- ÉTAPE B : ANALYSE DES IMPORTANCES (Hors du else !) ---
    print("\n📊 ANALYSE DES VARIABLES (Modèles optimisés)")
    
    # Importance Random Forest
    fi_rf = pd.DataFrame({
        "feature": X_train.columns,
        "importance": best_rf_model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=fi_rf.head(15), palette="viridis")
    plt.title("Top 15 - Importance RF (Optimisé)")
    plt.tight_layout()
    plt.show()

    # Importance XGBoost
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(best_xgb_model, max_num_features=15, importance_type='weight')
    plt.title("Top 15 - Importance XGBoost (Optimisé)")
    plt.tight_layout()
    plt.show()

    # --- ÉTAPE C : ÉVALUATION FINALE ---
    print("\n🏆 PERFORMANCES DES MODÈLES OPTIMISÉS")
    
    print("\n🌲 RANDOM FOREST (Paramètres tunés) :")
    y_pred_rf, rmse_rf, r2_rf, mae_rf = evaluer_modele(best_rf_model, X_test, y_test)
    
    print("\n⚡ XGBOOST (Paramètres tunés) :")
    y_pred_xgb, rmse_xgb, r2_xgb, mae_xgb = evaluer_modele(best_xgb_model, X_test, y_test)

    # --- ÉTAPE D : SAUVEGARDE DES ERREURS ---
    enregistrer_erreurs(X_test, y_test, y_pred_rf, "models/erreurs_rf_tuned.xlsx")
    enregistrer_erreurs(X_test, y_test, y_pred_xgb, "models/erreurs_xgb_tuned.xlsx")

    print("\n✅ Pipeline terminé.")
    print(f"Meilleur score final : {'XGBoost' if r2_xgb > r2_rf else 'Random Forest'} (R²: {max(r2_rf, r2_xgb):.4f})")

if __name__ == "__main__":
    main()
