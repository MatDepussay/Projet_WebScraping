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
from sklearn.metrics import silhouette_score
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

    # --- Calcul et Intégration ---
    clusters = cluster_pipeline.fit_predict(data_pd)
    
    # On ajoute la colonne à data_pd (essentiel pour ton print final !)
    data_pd["cluster_vehicule"] = clusters
    
    # On met à jour l'objet d'origine (df)
    if isinstance(df, pl.DataFrame):
        df = df.with_columns(pl.Series("cluster_vehicule", clusters))
    else:
        df["cluster_vehicule"] = clusters

    # Sauvegarde du pipeline COMPLET (mieux que 2 fichiers séparés)
    Path("models").mkdir(parents=True, exist_ok=True)
    with open("models/cluster_full_pipeline.pkl", "wb") as f:
        pickle.dump(cluster_pipeline, f)
    
    print("\n📊 Répartition des clusters :")
    print(df['cluster_vehicule'].value_counts())
    
    # --- Analyse du Cluster 3 ---

    cluster_3_cars = data_pd[data_pd["cluster_vehicule"] == 4]
    print("\n🔍 Détail des voitures du Cluster 4 (les 6 premières) :")
    
    # Sécurité : on vérifie si des voitures existent dans ce cluster
    if not cluster_3_cars.empty:
        cols_affichage = ["marque", "modele", "prix", "kilometrage", "annee", "puissance_kw"]
        # On ne garde que les colonnes présentes pour éviter une nouvelle KeyError
        cols_presentes = [c for c in cols_affichage if c in cluster_3_cars.columns]
        print(cluster_3_cars[cols_presentes].head(6))
    else:
        print("Aucun véhicule trouvé dans le cluster 3.")
    
    print("✅ Clustering terminé et pipeline sauvegardé.")
    return df

# =========================
# 2. ANALYSE ET VISUALISATION
# =========================
def analyser_clusters(df):
    """
    Analyse statistique et graphique des clusters générés.
    S'adapte dynamiquement aux colonnes présentes.
    """
    # Conversion pour l'analyse
    data_pd = df.to_pandas() if isinstance(df, pl.DataFrame) else df

    if "cluster_vehicule" not in data_pd.columns:
        print("⚠️ Erreur : Colonne 'cluster_vehicule' manquante pour l'analyse.")
        return
    
    # 1. Statistiques descriptives
    stats_config = {
        "prix": ["median", "mean"],
        "kilometrage": "median",
        "annee": "median",
        "puissance_kw": "median",
        "cylindree_l": "median"
    }
    
    # On ne garde que les clés présentes dans le DataFrame
    agg_dict = {k: v for k, v in stats_config.items() if k in data_pd.columns}

    print("\n🧠 Profil statistique des clusters :")
    if agg_dict:
        resume = data_pd.groupby("cluster_vehicule").agg(agg_dict).round(0)
        print(resume)
    else:
        print("Aucune colonne numérique trouvée pour les statistiques.")

    # 2. Visualisation (Boxplots) dynamique
    cols_a_voir = [c for c in ["prix", "kilometrage", "annee", "puissance_kw"] if c in data_pd.columns]
    
    if cols_a_voir:
        # Calcul du nombre de lignes/colonnes pour les subplots
        n_cols = 2
        n_rows = (len(cols_a_voir) + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(cols_a_voir):
            sns.boxplot(x="cluster_vehicule", y=col, data=data_pd, ax=axes[i], 
                        hue="cluster_vehicule", palette="Set2", legend=False)
            axes[i].set_title(f"Distribution de {col}")
        
        # Supprimer les axes vides si nombre impair de graphiques
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.show()

    # 3. PCA (uniquement si on a assez de colonnes numériques)
    if len(data_pd.select_dtypes(include=[np.number]).columns) >= 3:
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
    data_pd = df.to_pandas() if isinstance(df, pl.DataFrame) else df.copy()
    
    # Liste des features idéales
    cibles = ["annee", "kilometrage", "puissance_kw", "prix", "cylindree_l"]
    # On ne garde que celles qui existent vraiment dans le DF
    num_features = [c for c in cibles if c in data_pd.columns]
    
    if len(num_features) < 2:
        print("⚠️ Pas assez de colonnes numériques pour le clustering.")
        return [], []

    # Prétraitement : on drop les lignes avec des NaN uniquement sur ces colonnes
    data_clean = data_pd[num_features].dropna()
    
    if len(data_clean) < max_k:
        print(f"⚠️ Trop peu de données ({len(data_clean)}) pour max_k={max_k}.")
        max_k = max(2, len(data_clean) - 1)

    X = StandardScaler().fit_transform(data_clean)
    
    # 1. Calcul des inerties (Elbow)
    inertias = []
    ks_elbow = range(1, max_k + 1)
    for k in ks_elbow:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
    
    plt.figure(figsize=(8, 4))
    plt.plot(ks_elbow, inertias, 'go-')
    plt.title("Méthode du Coude (Elbow)")
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Inertie")
    plt.show()
    
    # 2. Calcul des silhouettes
    silhouettes = []
    ks_sil = range(2, max_k + 1)
    for k in ks_sil:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        silhouettes.append(silhouette_score(X, km.labels_))
    
    plt.figure(figsize=(8, 4))
    plt.plot(ks_sil, silhouettes, 'bo-')
    plt.title("Score de Silhouette")
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Score")
    plt.show()

    return inertias, silhouettes

# =========================
# CHARGEMENT & PRÉPARATION
# =========================
def charger_et_preparer_donnees(fichier="data/processed/autoscout_clean_ml.json"):
    try:
        df = pl.read_json(fichier).to_pandas()
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        return None, None, None, None

    if "modele_identifie" in df.columns: #modele_identifié supp dans cleaning
        df = df[df["modele_identifie"]]

    df = df.dropna(subset=["prix"]) #pour être sûr

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
    )#test_size a verif pour que ce soit l'input de l'app (0.2 par defaut)

    X_train = pd.get_dummies(X_train, columns=categorical_cols)
    X_test = pd.get_dummies(X_test, columns=categorical_cols)

    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0) # à vérifier

    # Vérification de sécurité
    if not any("cluster_vehicule" in col for col in X_train.columns):
        raise ValueError("🚨 Erreur critique : la colonne cluster_vehicule a disparu lors de l'encodage !")
    
    # Cherche n'importe quelle colonne qui commence par "cluster_vehicule"
    assert any("cluster_vehicule" in col for col in X_train.columns)

    return X_train, X_test, y_train, y_test


# =========================
# ÉVALUATION
# =========================
def evaluer_modele(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_test = r2_score(y_test, y_pred)
    mae_test = mean_absolute_error(y_test, y_pred)
    print(f"RMSE: €{rmse_test:,.2f} | R²: {r2_test:.4f} | MAE: €{mae_test:,.2f}")
    return y_pred, rmse_test, r2_test, mae_test


def cross_validation_model(model, X_train, y_train, cv=5):
    scores = cross_val_score(model, X_train, y_train, scoring="r2", cv=cv, n_jobs=-1)
    print(f"🔁 CV R²: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


def enregistrer_erreurs(X_test, y_test, y_pred, fichier):
    # 1. On ne garde que les colonnes numériques "réelles" pour que l'Excel soit lisible
    # On exclut les colonnes de type dummies
    cols_lisibles = [c for c in X_test.columns if '_' not in c]
    df_err = X_test[cols_lisibles].copy()

    # 2. Calculs des erreurs
    df_err["prix_reel"] = y_test.values
    df_err["prix_predit"] = np.round(y_pred, 2)
    df_err["erreur_abs"] = np.abs(df_err["prix_predit"] - df_err["prix_reel"])
    df_err["erreur_%"] = (df_err["erreur_abs"] / df_err["prix_reel"] * 100).round(2)

    # 3. Ajout d'un diagnostic métier
    df_err["diagnostic"] = np.where(
        df_err["prix_predit"] > df_err["prix_reel"], 
        "Sur-estimé", 
        "Sous-estimé"
    )
    
    # 4. Tri par les plus grosses erreurs en pourcentage
    df_err = df_err.sort_values("erreur_%", ascending=False)
    
    # 5. Sauvegarde
    if fichier.endswith('.xlsx'):
        df_err.to_excel(fichier, index=False)
    else:
        df_err.to_csv(fichier, index=False)
        
    print(f"💾 Fichier d'erreurs enregistré : {fichier} ({len(df_err)} lignes)")

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
        n_iter=10, cv=5, scoring='r2', verbose=1, random_state=42, n_jobs=-1
    )
    #scoring( 'r2' a passer en input metrique app) voir si il existe d'autre façons de scoring
    search.fit(X_train, y_train)
    print(f"✅ Meilleurs paramètres RF: {search.best_params_}")
    return search.best_estimator_

def entrainer_random_forest(model_tune, X_train, X_test, y_train, y_test): #casser pas les couilles faut prendre ce qu'on tune
    print("\n🌲 RANDOM FOREST")
    model = model_tune
    model.fit(X_train, y_train) #Optionnel
    
    # 1. Score R² sur les données d'entraînement (Train Score)
    r2_train = model.score(X_train, y_train)
    print(f"📊 R² sur données Train : {r2_train:.4f}")
    
    # 2. Évaluation sur test
    y_pred, rmse_test, r2_test, mae_test = evaluer_modele(model, X_test, y_test)
    
    # 3. Cross-Validation (on récupère la moyenne des scores R²)
    print("🔄 Calcul de la Cross-Validation (R²)...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    r2_cv = cv_scores.mean()
    print(f"🎯 R² Moyen (Cross-Validation) : {r2_cv:.4f}")

    # Importance des variables
    fi = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print(fi.head(10))
    print(fi[fi["feature"] == "cluster_vehicule"])

    # Sauvegarde
    with open("models/random_forest_model.pkl", "wb") as f:
        pickle.dump(model, f)

    enregistrer_erreurs(X_test, y_test, y_pred, "models/erreurs_rf.xlsx")
    
    return {
        'model': model,
        'rmse': rmse_test,
        'r2': r2_cv,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'mae': mae_test,
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
        n_iter=10, cv=5, scoring='r2', verbose=1, random_state=42, n_jobs=-1
    ) #scoring comme pour RF passer a l'imput app
    
    search.fit(X_train, y_train)
    print(f"✅ Meilleurs paramètres XGB: {search.best_params_}")
    return search.best_estimator_

def entrainer_xgboost(model_tune,X_train, X_test, y_train, y_test): # prendre les paramétres tune
    print("\n⚡ XGBOOST")
    model = model_tune
    
    model.fit(X_train, y_train) # Optionnel
    
    # 1. Score R² sur les données d'entraînement
    r2_train = model.score(X_train, y_train)
    print(f"📊 R² sur données Train : {r2_train:.4f}")
    
    # 2. Évaluation sur test
    y_pred, rmse_test, r2_test, mae_test = evaluer_modele(model, X_test, y_test)
    
    # 3. Cross-Validation
    print("🔄 Calcul de la Cross-Validation (R²)...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    r2_cv = cv_scores.mean()
    print(f"🎯 R² Moyen (Cross-Validation) : {r2_cv:.4f}")
    
    # Importance des variables
    fi = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print(fi.head(10))
    print(fi[fi["feature"] == "cluster_vehicule"])
    
    # Sauvegarde
    with open("models/xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    enregistrer_erreurs(X_test, y_test, y_pred, "models/erreurs_xgb.xlsx")

    return {
        'model': model,
        'rmse': rmse_test,
        'r2': r2_cv, #a verif prendre le R2 de cross validation et pas celui d'evaluation
        'r2_train': r2_train,
        'r2_test': r2_test,
        'mae': mae_test,
        'feature_importance': fi
    }

# =========================
# MAIN
# =========================
def main():
    # Configuration initiale
    os.makedirs("models", exist_ok=True)
    print("\n" + "="*50)
    print("🚀 DÉMARRAGE DU PIPELINE MACHINE LEARNING")
    print("="*50)

    fichier_input = "data/processed/autoscout_clean_ml.json"
    rf_path = "models/best_rf_final.pkl"
    xgb_path = "models/best_xgb_final.pkl"
    features_path = "models/model_features.pkl"
    
    # 1. PRÉPARATION DES DONNÉES & CLUSTERING
    print("\n📦 1. Chargement et Préparation des données...")
    X_train, X_test, y_train, y_test = charger_et_preparer_donnees(fichier_input)
    
    if X_train is None:
        print("❌ Échec du chargement. Fin du programme.")
        return
    
    # Sauvegarde des colonnes pour l'application Streamlit
    with open(features_path, "wb") as f:
        pickle.dump(X_train.columns.tolist(), f)
        
    print(f"✅ Dataset prêt : {X_train.shape[0]} train / {X_test.shape[0]} test")
    print(f"📊 Nombre de features (colonnes) : {len(X_train.columns)}")
    
    # 2. CHARGEMENT OU TUNING (A revoir car conflits OneHotEncoder et get dummies)
    if os.path.exists(rf_path) and os.path.exists(xgb_path):
        print("\n♻️  2. Modèles optimisés trouvés. Chargement en cours...")
        with open(rf_path, "rb") as f:
            best_rf_model = pickle.load(f)
        with open(xgb_path, "rb") as f:
            best_xgb_model = pickle.load(f)
        
        # Sécurité : Alignement des données sur les modèles chargés
        expected_features = best_rf_model.feature_names_in_ if hasattr(best_rf_model, "feature_names_in_") else X_train.columns.tolist()
        X_train = X_train.reindex(columns=expected_features, fill_value=0)
        X_test = X_test.reindex(columns=expected_features, fill_value=0)
    else:
        print("\n🔍 2. Aucun modèle trouvé. Lancement de l'optimisation (Tuning)...")
        # On utilise tes fonctions de tuning qui font le RandomizedSearchCV
        best_rf_model = tune_random_forest(X_train, y_train)
        best_xgb_model = tune_xgboost(X_train, y_train)
        
        # Sauvegarde des modèles optimisés
        with open(rf_path, "wb") as f:
            pickle.dump(best_rf_model, f)
        with open(xgb_path, "wb") as f:
            pickle.dump(best_xgb_model, f)
        print("✅ Tuning terminé et modèles sauvegardés.")

    # 3. ANALYSE DES VARIABLES (Feature Importance)
    print("\n📊 3. Analyse des variables d'influence...")
    
    # --- GRAPHIQUE 1 : RANDOM FOREST ---
    fi_rf = pd.DataFrame({
        "feature": X_train.columns,
        "importance": best_rf_model.feature_importances_
    }).sort_values("importance", ascending=False).head(15)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x="importance", 
        y="feature", 
        data=fi_rf, 
        hue="feature", 
        palette="magma", 
        legend=False
    )
    plt.title("Top 15 Features - Random Forest")
    plt.tight_layout()
    plt.show()  # Bloque ici jusqu'à ce que tu fermes la fenêtre

    # --- GRAPHIQUE 2 : XGBOOST ---
    xgb_importances = best_xgb_model.get_booster().get_score(importance_type='gain')
    fi_xgb = pd.DataFrame({
        "feature": list(xgb_importances.keys()),
        "importance": list(xgb_importances.values())
    }).sort_values("importance", ascending=False).head(15)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x="importance", 
        y="feature", 
        data=fi_xgb, 
        hue="feature", 
        palette="viridis", 
        legend=False
    )
    plt.title("Top 15 Features - XGBoost (Importance par Gain)")
    plt.xlabel("Gain moyen apporté par la variable")
    plt.ylabel("Variables")
    plt.tight_layout()
    plt.show()

    # 4. ÉVALUATION FINALE & COMPARAISON
    print("\n🏆 4. ÉVALUATION DÉTAILLÉE DES MODÈLES")
    
    # --- RANDOM FOREST ---
    print("\n" + "="*40)
    print("🌲 MODÈLE : RANDOM FOREST")
    print("="*40)
    # Cette fonction affiche déjà le R2 Train, le R2 CV et le R2 Test
    res_rf = entrainer_random_forest(best_rf_model, X_train, X_test, y_train, y_test)
    
    # --- XGBOOST ---
    print("\n" + "="*40)
    print("⚡ MODÈLE : XGBOOST")
    print("="*40)
    res_xgb = entrainer_xgboost(best_xgb_model, X_train, X_test, y_train, y_test)
    
    print("\n" + "="*40)
    
    # 5. RÉSUMÉ COMPARATIF FINAL
    print("\n🏁 RÉSUMÉ DES PERFORMANCES (R²)")
    print(f"{'Modèle':<20} | {'CV (Train)':<12} | {'Test Set':<12}")
    print("-" * 50)
    print(f"{'Random Forest':<20} | {res_rf['r2']:.4f}     | {res_rf['r2_test']:.4f}")
    print(f"{'XGBoost':<20} | {res_xgb['r2']:.4f}     | {res_xgb['r2_test']:.4f}")

    # On récupère les scores pour la conclusion finale
    r2_rf_final = res_rf['r2_test']
    r2_xgb_final = res_xgb['r2_test']
    
    # 5. ENREGISTREMENT DES ERREURS
    enregistrer_erreurs(X_test, y_test, res_rf['model'].predict(X_test), "models/erreurs_rf_tuned.xlsx")
    enregistrer_erreurs(X_test, y_test, res_xgb['model'].predict(X_test), "models/erreurs_xgb_tuned.xlsx")
    
    # Conclusion
    meilleur_modele = "XGBoost" if r2_xgb_final > r2_rf_final else "Random Forest"
    best_score = max(r2_rf_final, r2_xgb_final)
    
    print("\n" + "="*50)
    print("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
    print(f"⭐ MEILLEUR MODÈLE : {meilleur_modele}")
    print(f"🎯 SCORE R² FINAL : {best_score:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
