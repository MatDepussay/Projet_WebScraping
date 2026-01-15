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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import pickle

# =========================
# 1. CHARGEMENT & PRÉPARATION (sans imputation)
# =========================
def charger_et_preparer_donnees(fichier="autoscout_clean_ml.xlsx"):
    try:
        df = pl.read_excel(fichier).to_pandas()
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        return None, None, None, None

    # 0. Filtrer pour ne garder que les modèles identifiés
    if "modele_identifie" in df.columns:
        initial_count = len(df)
        df = df[df["modele_identifie"] == True]
        filtered_count = len(df)
        print(f"✅ Modèles identifiés: {filtered_count}/{initial_count} voitures gardées")
    else:
        print(f"⚠️ Colonne 'modele_identifie' introuvable, toutes les données seront utilisées")

    # 1. Nettoyage
    df = df.dropna(subset=["prix"])

    # 2. Séparation cible
    y = df["prix"]

    # 3. Suppression des colonnes que tu ne veux pas encoder
    # On retire 'modele' et 'modele_identifie' pour éviter l'erreur
    colonnes_a_exclure = ["prix", "cylindree_l", "code_postal", "pays", "ville", "modele_identifie"] 
    # Ajoute d'autres colonnes ici si besoin (ex: "id", "url", "description")
    
    X = df.drop(columns=[c for c in colonnes_a_exclure if c in df.columns])

    # 4. Identification automatique des autres colonnes textuelles
    # (marque, boite_vitesse, etc. seront encodées, mais pas 'modele')
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"📦 Colonnes encodées : {categorical_cols}")

    # 5. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. One-Hot Encoding
    X_train = pd.get_dummies(X_train, columns=categorical_cols)
    X_test = pd.get_dummies(X_test, columns=categorical_cols)

    # 7. Alignement
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    return X_train, X_test, y_train, y_test

# =========================
# 2. FONCTIONS D’ÉVALUATION
# =========================
def evaluer_modele(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"RMSE: €{rmse:,.2f} | R²: {r2:.4f} | MAE: €{mae:,.2f}")
    return y_pred, rmse, r2, mae

def cross_validation_model(model, X_train, y_train, cv=5):
    scores = cross_val_score(model, X_train, y_train, scoring="r2", cv=cv, n_jobs=-1)
    print(f"🔁 Cross-validation R² moyenne: {scores.mean():.4f} | std: {scores.std():.4f}")
    return scores

def enregistrer_erreurs(X_test, y_test, y_pred, fichier="models/tableau_erreurs.xlsx"):
    df_err = X_test.copy()
    df_err["prix_reel"] = y_test.values
    df_err["prix_predit"] = y_pred
    df_err["erreur"] = y_pred - y_test.values
    df_err["erreur_absolue"] = np.abs(df_err["erreur"])
    df_err["erreur_relative_%"] = 100 * df_err["erreur_absolue"] / df_err["prix_reel"]
    df_err.sort_values("erreur_absolue", ascending=False, inplace=True)
    df_err.to_excel(fichier, index=False)
    print(f"📉 Tableau des erreurs sauvegardé dans {fichier}")
    return df_err

# =========================
# 3. RANDOM FOREST
# =========================
def entrainer_random_forest(X_train, X_test, y_train, y_test):
    print("\n🌲 RANDOM FOREST REGRESSOR")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_pred, rmse, r2, mae = evaluer_modele(rf_model, X_test, y_test)
    cross_validation_model(rf_model, X_train, y_train)

    # Feature importance
    fi = pd.DataFrame({
        "feature": X_train.columns,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False)
    print("🎯 Top 5 features:")
    print(fi.head(5).to_string(index=False))

    # Sauvegarde modèle
    with open("models/random_forest_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    print("✅ Random Forest sauvegardé")

    enregistrer_erreurs(X_test, y_test, y_pred, "models/tableau_erreurs_rf.xlsx")

    return {
        "model": rf_model, "rmse": rmse, "r2": r2, "mae": mae, "feature_importance": fi}

# =========================
# 4. XGBOOST
# =========================
def entrainer_xgboost(X_train, X_test, y_train, y_test):
    print("\n⚡ XGBOOST REGRESSOR")
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred, rmse, r2, mae = evaluer_modele(xgb_model, X_test, y_test)
    cross_validation_model(xgb_model, X_train, y_train)

    fi = pd.DataFrame({
        "feature": X_train.columns,
        "importance": xgb_model.feature_importances_
    }).sort_values("importance", ascending=False)
    print("🎯 Top 5 features:")
    print(fi.head(5).to_string(index=False))

    # Sauvegarde modèle
    with open("models/xgboost_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    print("✅ XGBoost sauvegardé")

    enregistrer_erreurs(X_test, y_test, y_pred, "models/tableau_erreurs_xgb.xlsx")

    return {"model": xgb_model, "rmse": rmse, "r2": r2, "mae": mae, "feature_importance": fi}

# =========================
# 5. COMPARAISON
# =========================
def comparer_modeles(results_rf, results_xgb):
    comparaison = pd.DataFrame({
        "Métrique": ["RMSE", "R²", "MAE"],
        "Random Forest": [f"€{results_rf['rmse']:,.2f}", f"{results_rf['r2']:.4f}", f"€{results_rf['mae']:,.2f}"],
        "XGBoost": [f"€{results_xgb['rmse']:,.2f}", f"{results_xgb['r2']:.4f}", f"€{results_xgb['mae']:,.2f}"]
    })
    print("\n📊 Comparaison des modèles")
    print(comparaison.to_string(index=False))
    comparaison.to_excel("models/comparaison_modeles.xlsx", index=False)
    meilleur = "Random Forest" if results_rf["r2"] > results_xgb["r2"] else "XGBoost"
    print(f"\n🏆 Meilleur modèle (R² test): {meilleur}")
    return comparaison

# =========================
# 6. MAIN
# =========================
def main():
    os.makedirs("models", exist_ok=True)
    print("🚀 Démarrage du pipeline ML...")

    X_train, X_test, y_train, y_test = charger_et_preparer_donnees()
    if X_train is None:
        return

    print(f"\n📈 Dataset: {X_train.shape[0]} train / {X_test.shape[0]} test")
    results_rf = entrainer_random_forest(X_train, X_test, y_train, y_test)
    results_xgb = entrainer_xgboost(X_train, X_test, y_train, y_test)
    comparer_modeles(results_rf, results_xgb)
    print("\n✅ Pipeline ML terminé! Modèles et fichiers d'erreurs sauvegardés dans 'models/'")

if __name__ == "__main__":
    main()
