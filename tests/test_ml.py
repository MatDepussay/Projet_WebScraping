import pytest
import pandas as pd
import numpy as np
import polars as pl
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

def test_ajouter_cluster_vehicule():
    from src.MachineLearning import ajouter_cluster_vehicule
    # Création d'un mini-dataframe de test
    df_test = pd.DataFrame({
        "annee": [2020, 2015, 2010, 2022],
        "kilometrage": [10000, 50000, 150000, 500],
        "puissance_kw": [110, 80, 60, 200],
        "cylindree_l": [2.0, 1.5, 1.2, 3.0],
        "carburant": ["Essence", "Diesel", "Essence", "Electrique"],
        "boite_de_vitesse": ["Manuelle", "Manuelle", "Manuelle", "Automatique"],
        "transmission": ["Avant", "Avant", "Avant", "Intégrale"],
        "type_de_vehicule": ["Berline", "SUV", "Citadine", "Coupé"]
    })
    
    # Exécution
    result_df = ajouter_cluster_vehicule(df_test, n_clusters=2)
    
    # Vérifications
    assert "cluster_vehicule" in result_df.columns
    assert result_df["cluster_vehicule"].nunique() <= 2
    assert result_df["cluster_vehicule"].isnull().sum() == 0

# On crée un "fixture" : un jeu de données réutilisable pour les tests
@pytest.fixture
def sample_data():
    data = {
        "cluster_vehicule": [0, 0, 1, 1],
        "prix": [10000, 12000, 50000, 55000],
        "kilometrage": [100000, 90000, 10000, 5000],
        "annee": [2015, 2016, 2022, 2023],
        "puissance_kw": [80, 85, 200, 210]
    }
    return pd.DataFrame(data)

def test_analyser_clusters_pandas(sample_data, monkeypatch):
    """Vérifie que la fonction s'exécute sans erreur avec Pandas"""
    from src.MachineLearning import analyser_clusters
    # monkeypatch permet d'empêcher l'ouverture des fenêtres de graphiques pendant le test
    monkeypatch.setattr(plt, 'show', lambda: None)
    
    try:
        analyser_clusters(sample_data)
    except Exception as e:
        pytest.fail(f"La fonction a levé une exception avec Pandas : {e}")

def test_analyser_clusters_polars(sample_data, monkeypatch):
    """Vérifie la compatibilité avec Polars"""
    from src.MachineLearning import analyser_clusters
    monkeypatch.setattr(plt, 'show', lambda: None)
    df_pl = pl.from_pandas(sample_data)
    
    try:
        analyser_clusters(df_pl)
    except Exception as e:
        pytest.fail(f"La fonction a levé une exception avec Polars : {e}")

def test_analyser_clusters_missing_column(sample_data, monkeypatch):
    """Vérifie que la fonction gère l'absence d'une colonne optionnelle"""
    from src.MachineLearning import analyser_clusters
    monkeypatch.setattr(plt, 'show', lambda: None)
    # On supprime 'puissance_kw' qui est dans cols_a_voir
    df_incomplete = sample_data.drop(columns=["puissance_kw"])
    
    try:
        analyser_clusters(df_incomplete)
    except Exception as e:
        pytest.fail(f"La fonction a planté alors qu'une colonne était manquante : {e}")


def test_trouver_meilleur_k_logic(sample_data, monkeypatch):
    """Vérifie que la fonction calcule les bonnes longueurs de scores"""
    from src.MachineLearning import trouver_meilleur_k
    # On empêche l'affichage des graphiques
    monkeypatch.setattr(plt, 'show', lambda: None)
    
    # On définit un max_k petit pour aller vite
    max_k = 3
    
    # On ne peut pas facilement tester les variables locales 'inertias' 
    # sauf si la fonction les retournait. 
    # Mais on vérifie au moins qu'elle s'exécute sans erreur.
    try:
        trouver_meilleur_k(sample_data, max_k=max_k)
    except Exception as e:
        pytest.fail(f"trouver_meilleur_k a planté : {e}")

def test_trouver_meilleur_k_min_data(monkeypatch):
    """Vérifie que la fonction gère un dataset très petit"""
    from src.MachineLearning import trouver_meilleur_k
    monkeypatch.setattr(plt, 'show', lambda: None)
    
    # Dataset minimal avec 4 lignes (nécessaire pour k=3)
    df_mini = pd.DataFrame({
        "annee": [2010, 2011, 2012, 2013],
        "kilometrage": [10, 20, 30, 40],
        "puissance_kw": [50, 60, 70, 80],
        "prix": [1000, 2000, 3000, 4000],
        "cylindree_l": [1.0, 1.2, 1.4, 1.6]
    })
    
    try:
        trouver_meilleur_k(df_mini, max_k=3)
    except Exception as e:
        pytest.fail(f"La fonction a planté sur un petit dataset : {e}")

def test_colonnes_identiques():
    from src.MachineLearning import charger_et_preparer_donnees
    X_train, X_test, _, _ = charger_et_preparer_donnees()
    assert list(X_train.columns) == list(X_test.columns)

def test_evaluer_modele():
    from sklearn.dummy import DummyRegressor
    from src.MachineLearning import evaluer_modele
    X = np.array([[1], [2], [3]])
    y_true = np.array([10, 20, 30])
    
    # On crée un modèle simple qui prédit toujours la moyenne
    model = DummyRegressor(strategy="mean")
    model.fit(X, y_true)
    
    y_pred, rmse, r2, mae = evaluer_modele(model, X, y_true)
    
    assert isinstance(rmse, float)
    assert rmse >= 0
    assert -1 <= r2 <= 1  # Le R2 est dans cet intervalle

def test_cross_validation_logic():
    """Vérifie que la CV retourne le bon nombre de folds et des scores valides"""
    from sklearn.linear_model import LinearRegression
    from src.MachineLearning import cross_validation_model
    # Création d'un dataset synthétique
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    model = LinearRegression()
    cv_folds = 5
    
    scores = cross_validation_model(model, X, y, cv=cv_folds)
    
    # 1. Vérifier le nombre de scores (un par fold)
    assert len(scores) == cv_folds
    
    # 2. Vérifier que les scores sont des nombres (float)
    assert all(isinstance(s, float) for s in scores)
    
    # 3. Le R2 ne peut pas dépasser 1.0
    assert np.all(scores <= 1.0)

def test_cross_validation_empty_data():
    """Vérifie que la fonction lève une erreur si les données sont vides"""
    from sklearn.linear_model import LinearRegression
    from src.MachineLearning import cross_validation_model
    model = LinearRegression()
    X = np.array([]).reshape(0, 5)
    y = np.array([])
    
    with pytest.raises(ValueError):
        cross_validation_model(model, X, y, cv=5)

def test_enregistrer_erreurs_calculs(tmp_path):
    """Vérifie que les colonnes d'erreurs et le diagnostic sont mathématiquement corrects"""
    from src.MachineLearning import enregistrer_erreurs
    # 1. Préparation des données de test
    X_test = pd.DataFrame({
        "kilometrage": [100, 200],
        "annee": [2020, 2010],
        "marque_Audi": [1, 0] # Colonne dummy avec '_' à exclure
    })
    y_test = pd.Series([20000, 10000])
    y_pred = np.array([22000, 9000]) # Erreurs : +2000 (sur-estimé) et -1000 (sous-estimé)
    
    # Chemin temporaire pour le fichier
    test_file = tmp_path / "test_erreurs.csv"
    
    # 2. Exécution
    enregistrer_erreurs(X_test, y_test, y_pred, str(test_file))
    
    # 3. Rechargement pour vérification
    df_verif = pd.read_csv(test_file)
    
    # Vérification des colonnes conservées (pas de '_')
    assert "marque_Audi" not in df_verif.columns
    assert "kilometrage" in df_verif.columns
    
    # Vérification des calculs
    # Ligne 0 : Prix 20000, Pred 22000 -> Erreur abs 2000, Erreur % 10.0, Sur-estimé
    row_sur = df_verif[df_verif["prix_reel"] == 20000].iloc[0]
    assert row_sur["erreur_abs"] == 2000
    assert row_sur["erreur_%"] == 10.0
    assert row_sur["diagnostic"] == "Sur-estimé"
    
    # Ligne 1 : Prix 10000, Pred 9000 -> Erreur abs 1000, Erreur % 10.0, Sous-estimé
    row_sous = df_verif[df_verif["prix_reel"] == 10000].iloc[0]
    assert row_sous["diagnostic"] == "Sous-estimé"

def test_enregistrer_erreurs_format_excel(tmp_path):
    """Vérifie que le format Excel est bien géré"""
    from src.MachineLearning import enregistrer_erreurs
    X_test = pd.DataFrame({"km": [100]})
    y_test = pd.Series([10000])
    y_pred = np.array([10000])
    
    excel_file = tmp_path / "erreurs.xlsx"
    
    # Si openpyxl n'est pas installé, le test échouera (ce qui est une bonne info)
    try:
        enregistrer_erreurs(X_test, y_test, y_pred, str(excel_file))
        assert os.path.exists(excel_file)
    except ModuleNotFoundError:
        pytest.fail("Le moteur openpyxl est requis pour enregistrer en .xlsx")

def test_tune_random_forest_logic(sample_data):
    """Vérifie que la fonction de tuning appelle bien RandomizedSearchCV"""
    from sklearn.ensemble import RandomForestRegressor
    from unittest.mock import patch
    from src.MachineLearning import tune_random_forest
    # 1. Préparation des mini-données
    X = sample_data.drop(columns=["prix", "cluster_vehicule"])
    X = pd.get_dummies(X)
    y = sample_data["prix"]

    # 2. On "mock" RandomizedSearchCV pour éviter le vrai calcul
    with patch('src.MachineLearning.RandomizedSearchCV') as mock_search:
        # On simule un objet de recherche qui a déjà "trouvé" un modèle
        mock_instance = mock_search.return_value
        mock_instance.best_estimator_ = RandomForestRegressor(n_estimators=10)
        mock_instance.best_params_ = {'n_estimators': 10}

        # 3. Exécution
        result = tune_random_forest(X, y)

        # 4. Vérifications
        # A-t-on bien appelé la recherche ?
        assert mock_search.called
        # L'objet retourné est-il bien notre faux best_estimator_ ?
        assert result.n_estimators == 10

def test_entrainer_random_forest_full(sample_data, tmp_path):
    """Vérifie le cycle complet d'entraînement et le dictionnaire de retour"""
    from unittest.mock import MagicMock, patch
    from src.MachineLearning import entrainer_random_forest
    # 1. Setup des données
    X = sample_data.drop(columns=["prix", "cluster_vehicule"])
    X = pd.get_dummies(X)
    y = sample_data["prix"]
    
    # On mock le modèle
    mock_model = MagicMock()
    mock_model.score.return_value = 0.95
    mock_model.feature_importances_ = np.random.rand(len(X.columns))
    
    # On ajoute 'pickle' à la liste des patchs pour empêcher l'erreur de sauvegarde
    with patch('src.MachineLearning.evaluer_modele') as mock_eval, \
         patch('src.MachineLearning.cross_val_score') as mock_cv, \
         patch('src.MachineLearning.enregistrer_erreurs') as mock_err, \
         patch('src.MachineLearning.pickle.dump') as mock_pickle_dump, \
         patch('builtins.open', MagicMock()):
         
        mock_eval.return_value = (np.array([10000]), 500.0, 0.90, 400.0)
        mock_cv.return_value = np.array([0.88, 0.89, 0.90])
        
        # 2. Exécution
        result = entrainer_random_forest(mock_model, X, X, y, y)
        
        # 3. Vérifications
        assert result['r2_test'] == 0.90
        assert result['r2_train'] == 0.95
        assert isinstance(result['feature_importance'], pd.DataFrame)
        assert mock_pickle_dump.called # On vérifie que pickle a bien été "appelé"
        assert mock_err.called

def test_tune_xgboost_logic(sample_data):
    """Vérifie le tuning XGBoost sans calcul réel"""
    from unittest.mock import MagicMock, patch
    import xgboost as xgb
    from src.MachineLearning import tune_xgboost
    X = pd.get_dummies(sample_data.drop(columns=["prix", "cluster_vehicule"]))
    y = sample_data["prix"]

    with patch('src.MachineLearning.RandomizedSearchCV') as mock_search:
        mock_instance = mock_search.return_value
        # On simule un objet XGBRegressor factice
        mock_instance.best_estimator_ = MagicMock(spec=xgb.XGBRegressor)
        mock_instance.best_params_ = {'n_estimators': 100}

        result = tune_xgboost(X, y)

        assert mock_search.called
        assert result is mock_instance.best_estimator_

def test_entrainer_xgboost_full(sample_data):
    """Vérifie le cycle complet XGBoost"""
    from unittest.mock import MagicMock, patch
    from src.MachineLearning import entrainer_xgboost
    X = pd.get_dummies(sample_data.drop(columns=["prix", "cluster_vehicule"]))
    y = sample_data["prix"]
    
    # Mock du modèle XGBoost
    mock_model = MagicMock()
    mock_model.score.return_value = 0.92
    mock_model.feature_importances_ = np.random.rand(len(X.columns))
    
    with patch('src.MachineLearning.evaluer_modele') as mock_eval, \
         patch('src.MachineLearning.cross_val_score') as mock_cv, \
         patch('src.MachineLearning.enregistrer_erreurs') as mock_err, \
         patch('src.MachineLearning.pickle.dump') as mock_pickle_dump, \
         patch('builtins.open', MagicMock()):
         
        mock_eval.return_value = (np.array([12000]), 600.0, 0.88, 450.0)
        mock_cv.return_value = np.array([0.85, 0.86, 0.87])
        
        result = entrainer_xgboost(mock_model, X, X, y, y)
        
        assert result['r2'] == 0.86 # Moyenne de mock_cv
        assert result['r2_train'] == 0.92
        assert mock_pickle_dump.called
        assert mock_err.called

def test_main_pipeline_flow(monkeypatch, tmp_path):
    """Vérifie que le main s'exécute sans erreur en simulant les dépendances"""
    from unittest.mock import MagicMock, patch
    from src.MachineLearning import main
    # 1. On détourne la création de dossier
    monkeypatch.setattr("os.makedirs", lambda x, exist_ok: None)
    
    # 2. On mock TOUTES les fonctions lourdes et le pickling
    with patch('src.MachineLearning.charger_et_preparer_donnees') as mock_load, \
         patch('src.MachineLearning.tune_random_forest') as mock_tune_rf, \
         patch('src.MachineLearning.tune_xgboost') as mock_tune_xgb, \
         patch('src.MachineLearning.entrainer_random_forest') as mock_ent_rf, \
         patch('src.MachineLearning.entrainer_xgboost') as mock_ent_xgb, \
         patch('src.MachineLearning.enregistrer_erreurs'), \
         patch('src.MachineLearning.plt.show'), \
         patch('src.MachineLearning.pickle.dump'), \
         patch('os.path.exists', return_value=False), \
         patch('builtins.open', MagicMock()):

        # --- CONFIGURATION DES MOCKS ---
        
        # On utilise un vrai Index Pandas pour que .tolist() fonctionne !
        fake_columns = pd.Index(["col1", "col2"])
        
        mock_df = MagicMock()
        mock_df.columns = fake_columns
        mock_df.shape = (10, 2)
        mock_df.reindex.return_value = mock_df
        
        mock_load.return_value = (mock_df, mock_df, MagicMock(), MagicMock())
        
        # Simulation des modèles avec des importances cohérentes (taille 2)
        mock_rf_model = MagicMock()
        mock_rf_model.feature_importances_ = [0.6, 0.4]
        
        mock_xgb_model = MagicMock()
        mock_xgb_model.get_booster().get_score.return_value = {'col1': 0.5, 'col2': 0.3}
        mock_xgb_model.feature_importances_ = [0.7, 0.3]

        res_mock_rf = {'r2': 0.85, 'r2_test': 0.87, 'model': mock_rf_model}
        res_mock_xgb = {'r2': 0.86, 'r2_test': 0.88, 'model': mock_xgb_model}
        
        mock_ent_rf.return_value = res_mock_rf
        mock_ent_xgb.return_value = res_mock_xgb

        mock_tune_rf.return_value = mock_rf_model
        mock_tune_xgb.return_value = mock_xgb_model

        # 3. Exécution du main
        try:
            main()
        except Exception as e:
            import traceback
            traceback.print_exc() # Pour voir l'erreur exacte si ça replante
            pytest.fail(f"Le main() a planté : {e}")