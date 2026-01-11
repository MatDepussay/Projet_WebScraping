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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import pickle
import json

# =========================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# =========================
def charger_et_preparer_donnees():
    """Charge les données nettoyées et prépare les features pour ML"""
    try:
        df = pl.read_excel("autoscout_clean_ml.xlsx")
    except FileNotFoundError:
        print("❌ Fichier autoscout_clean_ml.xlsx non trouvé. Exécutez d'abord cleaning.py")
        return None, None, None, None

    # Conversion en pandas pour faciliter le preprocessing
    df_pd = df.to_pandas()
    
    # Séparation target et features
    X = df_pd.drop("prix", axis=1)
    y = df_pd["prix"]

    # Identification des colonnes numériques et catégoriques
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()

    print(f"📊 Features numériques : {numeric_cols}")
    print(f"📊 Features catégoriques : {categorical_cols}")

    # Encodage des variables catégoriques
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Remplissage des NaN éventuels
    X = X.fillna(X.median(numeric_only=True))

    return X, y, numeric_cols, label_encoders

# =========================
# 2. RANDOM FOREST
# =========================
def entrainer_random_forest(X_train, X_test, y_train, y_test):
    """Entraîne un modèle Random Forest"""
    print("\n" + "="*50)
    print("🌲 RANDOM FOREST REGRESSOR")
    print("="*50)
    
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Prédictions
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    
    # Métriques
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    print(f"✅ Entraînement terminé")
    print(f"   RMSE Train: €{rmse_train:,.2f}")
    print(f"   RMSE Test:  €{rmse_test:,.2f}")
    print(f"   R² Train:   {r2_train:.4f}")
    print(f"   R² Test:    {r2_test:.4f}")
    print(f"   MAE Test:   €{mae_test:,.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"\n🎯 Top 5 Features:")
    print(feature_importance.head(5).to_string(index=False))
    
    # Sauvegarder le modèle
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    return {
        'model': rf_model,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'mae_test': mae_test,
        'feature_importance': feature_importance
    }

# =========================
# 3. XGBOOST
# =========================
def entrainer_xgboost(X_train, X_test, y_train, y_test):
    """Entraîne un modèle XGBoost"""
    print("\n" + "="*50)
    print("⚡ XGBOOST REGRESSOR")
    print("="*50)
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=1
    )
    
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Prédictions
    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)
    
    # Métriques
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    print(f"✅ Entraînement terminé")
    print(f"   RMSE Train: €{rmse_train:,.2f}")
    print(f"   RMSE Test:  €{rmse_test:,.2f}")
    print(f"   R² Train:   {r2_train:.4f}")
    print(f"   R² Test:    {r2_test:.4f}")
    print(f"   MAE Test:   €{mae_test:,.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"\n🎯 Top 5 Features:")
    print(feature_importance.head(5).to_string(index=False))
    
    # Sauvegarder le modèle
    with open('models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    
    return {
        'model': xgb_model,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'mae_test': mae_test,
        'feature_importance': feature_importance
    }

# =========================
# 4. COMPARAISON DES MODÈLES
# =========================
def comparer_modeles(results_rf, results_xgb):
    """Compare les performances des deux modèles"""
    print("\n" + "="*50)
    print("📊 COMPARAISON DES MODÈLES")
    print("="*50)
    
    comparison = pd.DataFrame({
        'Métrique': ['RMSE Train', 'RMSE Test', 'R² Train', 'R² Test', 'MAE Test'],
        'Random Forest': [
            f"€{results_rf['rmse_train']:,.2f}",
            f"€{results_rf['rmse_test']:,.2f}",
            f"{results_rf['r2_train']:.4f}",
            f"{results_rf['r2_test']:.4f}",
            f"€{results_rf['mae_test']:,.2f}"
        ],
        'XGBoost': [
            f"€{results_xgb['rmse_train']:,.2f}",
            f"€{results_xgb['rmse_test']:,.2f}",
            f"{results_xgb['r2_train']:.4f}",
            f"{results_xgb['r2_test']:.4f}",
            f"€{results_xgb['mae_test']:,.2f}"
        ]
    })
    
    print(comparison.to_string(index=False))
    
    # Sauvegarder la comparaison
    comparison.to_excel('models/comparaison_modeles.xlsx', index=False)
    
    # Déterminer le meilleur modèle
    best_model = "Random Forest" if results_rf['r2_test'] > results_xgb['r2_test'] else "XGBoost"
    print(f"\n🏆 Meilleur modèle (R² test): {best_model}")
    
    return comparison

# =========================
# 5. MAIN
# =========================
def main():
    import os
    
    # Créer le dossier models s'il n'existe pas
    os.makedirs('models', exist_ok=True)
    
    print("🚀 Démarrage du pipeline ML...")
    
    # Chargement et préparation
    X, y, numeric_cols, label_encoders = charger_et_preparer_donnees()
    if X is None:
        return
    
    print(f"\n📈 Dataset: {X.shape[0]} lignes, {X.shape[1]} features")
    print(f"   Prix moyen: €{y.mean():,.2f}")
    print(f"   Prix min: €{y.min():,.2f} | Prix max: €{y.max():,.2f}")
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n📊 Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    
    # Entraînement des modèles
    results_rf = entrainer_random_forest(X_train, X_test, y_train, y_test)
    results_xgb = entrainer_xgboost(X_train, X_test, y_train, y_test)
    
    # Comparaison
    comparer_modeles(results_rf, results_xgb)
    
    print("\n✅ Pipeline ML terminé!")
    print("💾 Modèles sauvegardés dans le dossier 'models/'")

if __name__ == "__main__":
    main()

