import polars as pl
import pytest
import json
import re
from pathlib import Path
from unittest.mock import patch, mock_open

# --- Tests pour nettoyer_numeriques ---

def test_nettoyer_numeriques_format():
    """Vérifie que les symboles (€, km) et espaces sont bien supprimés."""
    from src.cleaning import nettoyer_numeriques
    data = {
        "prix": ["15 000 €", "12000", "9.500"],
        "kilometrage": ["100 000 km", "50k", "10"]
    }
    df = pl.DataFrame(data)
    
    result = nettoyer_numeriques(df)
    
    # Vérification des types et des valeurs
    assert result["prix"].dtype == pl.Int64
    assert result["prix"][0] == 15000
    # Note : "9.500" avec ton regex devient 9500, est-ce le comportement voulu ?

def test_nettoyer_numeriques_filtrage_prix():
    """Vérifie que les prix <= 100 ou nulls sont supprimés."""
    from src.cleaning import nettoyer_numeriques
    data = {
        "prix": ["50", "500", None, "abc"],
        "kilometrage": ["10", "10", "10", "10"]
    }
    df = pl.DataFrame(data)
    
    result = nettoyer_numeriques(df)
    
    # Seul le prix "500" devrait rester
    assert result.height == 1
    assert result["prix"][0] == 500

# --- Tests pour charger_donnees (Mocking) ---

def test_charger_donnees_error_handling():
    """Vérifie que la fonction retourne un DataFrame vide en cas d'erreur."""
    from src.cleaning import charger_donnees
    # On simule une erreur de lecture sans avoir besoin du fichier JSON
    with patch("polars.read_json") as mocked_read:
        mocked_read.side_effect = Exception("Fichier introuvable")
        
        df = charger_donnees()
        
        assert isinstance(df, pl.DataFrame)
        assert df.is_empty()

def test_normalisation_marques():
    from src.cleaning import reparer_marque_modele
    # Arrange
    data = {
        "marque": ["VW Golf", "Mercedes Benz Classe A", "Citroen C3", "Alfa-Romeo Giulia"]
    }
    df = pl.DataFrame(data)

    # Act
    result = reparer_marque_modele(df)

    # Assert
    expected_marques = ["Volkswagen", "Mercedes-Benz", "Citroën", "Alfa Romeo"]
    assert result["marque"].to_list() == expected_marques

def test_nettoyage_modele_technique():
    # Teste si les motorisations et cylindrées sont bien supprimées
    from src.cleaning import reparer_marque_modele
    data = {
        "marque": ["Audi A3 2.0 TDI 150CH", "Ford Focus 1.6 TDCi", "Hyundai Tucson CRDi"]
    }
    df = pl.DataFrame(data)

    result = reparer_marque_modele(df)

    # Assert : On vérifie que le modèle est "propre"
    # Note : Ta fonction fait un split/slice(0,2), donc "A3" est attendu
    assert result.filter(pl.col("marque") == "Audi")["modele"][0] == "A3"
    assert result.filter(pl.col("marque") == "Ford")["modele"][0] == "Focus"

def test_marques_inconnues_et_nulles():
    # Arrange : "Zorglub" n'est pas dans ton dictionnaire
    from src.cleaning import reparer_marque_modele
    data = {
        "marque": ["Zorglub X1", None, ""]
    }
    df = pl.DataFrame(data)

    # Act
    result = reparer_marque_modele(df)

    # Assert : Le filtre final .filter(pl.col("marque").is_not_null()) 
    # devrait supprimer ces lignes car l'extraction regex échouera.
    assert result.height == 0

def test_extraire_specs_techniques():
    from src.cleaning import extraire_specs_et_lieu
    data = {
        "puissance": ["110 kW (150 CH)", "80kW", "NC"],
        "modele": ["Golf 1.6 TDI", "Clio 1,2 16V", "Tesla Model 3"],
        "localisation": ["75001 Paris", "75001 Paris", "75001 Paris"] # Dummy
    }
    df = pl.DataFrame(data)
    result = extraire_specs_et_lieu(df)

    # Vérification Puissance
    assert result["puissance_kw"].to_list() == [110, 80, None]
    
    # Vérification Cylindrée (gestion du . et de la ,)
    assert result["cylindree_l"][0] == pytest.approx(1.6)
    assert result["cylindree_l"][1] == pytest.approx(1.2)
    assert result["cylindree_l"][2] is None

@pytest.mark.parametrize("loc, pays_attendu, ville_attendue, cp_attendu", [
    ("75008 Paris", "France", "Paris", "75008"),
    ("1000 Bruxelles", "Belgique", "Bruxelles", "1000"),
    ("1011 AB Amsterdam", "Pays-Bas", "Amsterdam", "1011 AB"),
    ("80331 München", "Allemagne", "München", "80331"),
    ("20121 Milano - MI", "Italie", "Milano", "20121"),
    ("Cassano d'Adda - MI", "Italie", "Cassano d'Adda", None), # Le cas critique
])
def test_localisation_pays(loc, pays_attendu, ville_attendue, cp_attendu):
    from src.cleaning import extraire_specs_et_lieu
    # Création d'un mini-df pour le test
    df = pl.DataFrame({
        "puissance": ["100 kW"], 
        "modele": ["A3"], 
        "localisation": [loc]
    })
    
    result = extraire_specs_et_lieu(df)

    assert result["pays"][0] == pays_attendu
    assert result["ville"][0] == ville_attendue
    assert result["code_postal"][0] == cp_attendu

def test_extraire_specs_et_lieu_final():
    from src.cleaning import extraire_specs_et_lieu
    data = {
        "puissance": ["110 kW", "80kW"],
        "modele": ["Golf 1.6 TDI", "Clio 1,2"],
        "localisation": ["75008 Paris", "80331 München"]
    }
    df = pl.DataFrame(data)
    result = extraire_specs_et_lieu(df)

    # 1. Test de la précision Float32
    assert result["cylindree_l"][0] == pytest.approx(1.6)
    
    # 2. Test de la discrimination France/Allemagne
    assert result.filter(pl.col("ville") == "Paris")["pays"][0] == "France"
    assert result.filter(pl.col("ville") == "München")["pays"][0] == "Allemagne"

def test_italie_special_case():
    from src.cleaning import extraire_specs_et_lieu
    # Le fameux Cassano d'Adda
    df = pl.DataFrame({
        "puissance": ["100 kW"], "modele": ["A3"], 
        "localisation": ["Cassano d'Adda - MI"]
    })
    result = extraire_specs_et_lieu(df)
    assert result["pays"][0] == "Italie"
    assert result["ville"][0] == "Cassano d'Adda"


def test_cylindree_marques_allemandes():
    from src.cleaning import corriger_cylindree
    data = {
        "marque": ["BMW", "Mercedes-Benz", "BMW"],
        "modele": ["320", "C 200", "i3"],
        "modele_identifie": [True, True, False],
        "carburant": ["Essence", "Diesel", "Électrique"],
        "cylindree_l": [None, None, None],
        "le_reste": ["", "", ""]
    }
    df = pl.DataFrame(data)
    result = corriger_cylindree(df)

    # Correction BMW : on utilise .item() pour récupérer la valeur scalaire
    valeur_bmw = result.filter(pl.col("modele") == "320")["cylindree_l"].item()
    assert valeur_bmw == pytest.approx(2.0)

    # Correction Mercedes : 
    # .item() est très pratique quand on sait qu'il n'y a qu'une ligne après filtrage
    valeur_mercedes = result.filter(pl.col("marque") == "Mercedes-Benz")["cylindree_l"].item()
    assert valeur_mercedes == pytest.approx(2.0)

    # Bonus : Vérifier que l'électrique (i3) est bien resté à None (ou 0.0 selon ta règle finale)
    valeur_i3 = result.filter(pl.col("modele") == "i3")["cylindree_l"].item()
    # Dans ta fonction, "électrique" force à 0.0 même si c'est None au départ
    assert valeur_i3 == 0.0

def test_cylindree_bornes_et_electrique():
    from src.cleaning import corriger_cylindree
    data = {
        "marque": ["Peugeot", "Ferrari", "Tesla"],
        "modele": ["208", "812", "Model 3"],
        "modele_identifie": [True, True, True],
        "carburant": ["Essence", "Essence", "Électrique"],
        "cylindree_l": [0.2, 12.0, 1.6], # 0.2 est trop petit, 12.0 trop gros
        "le_reste": ["", "", ""]
    }
    df = pl.DataFrame(data)
    result = corriger_cylindree(df)

    # 0.2 -> devient None car < 0.6
    assert result["cylindree_l"][0] is None
    # 12.0 -> devient None car > 8.5
    assert result["cylindree_l"][1] is None
    # Électrique -> devient 0.0 même si 1.6 était présent
    assert result["cylindree_l"][2] == 0.0

def test_nettoyer_transmission_ok():
    from src.cleaning import nettoyer_transmission
    data = {
        "transmission": ["Avant", "Arrière", "4x4", "arrière"]
    }
    df = pl.DataFrame(data)
    result = nettoyer_transmission(df)

    # Tout doit être en minuscule
    expected = ["avant", "arrière", "4x4", "arrière"]
    assert result["transmission"].to_list() == expected

def test_nettoyer_transmission_ko():
    from src.cleaning import nettoyer_transmission
    data = {
        "transmission": ["Automatique", "Intégrale", "Propulsion", None]
    }
    df = pl.DataFrame(data)
    result = nettoyer_transmission(df)

    # On vérifie que les synonymes sont bien convertis et les inconnus mis à None
    expected = [None, "4x4", "arrière", None]
    assert result["transmission"].to_list() == expected

def test_charger_modeles_json_success():
    from src.cleaning import charger_modeles_json
    # Données fictives simulant le contenu du fichier JSON
    mock_data = [
        {"Make": " Audi ", "Models": [" A3 ", "A4"]},
        {"Make": "BMW", "Models": ["Série 1", "X5"]}
    ]
    json_content = json.dumps(mock_data)

    # On simule l'existence du fichier et son contenu
    with patch("src.cleaning.Path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=json_content)):
        
        result = charger_modeles_json()

        # Vérification de la normalisation (minuscules et strip)
        assert "audi" in result
        assert set(result["audi"]) == {"A3", "A4"}
        assert "bmw" in result
        assert set(result["bmw"]) == {"Série 1", "X5"}

def test_charger_modeles_json_not_found():
    from src.cleaning import charger_modeles_json
    # On simule le fait que le fichier n'existe pas
    with patch("src.cleaning.Path.exists", return_value=False):
        result = charger_modeles_json()
        assert result == {}

def test_charger_modeles_json_fusion_doublons():
    from src.cleaning import charger_modeles_json
    # Deux entrées pour la même marque 'Audi'
    mock_data = [
        {"Make": "Audi", "Models": ["A3", "A4"]},
        {"Make": "AUDI ", "Models": ["A4", "A5"]} # A4 est en double
    ]
    json_content = json.dumps(mock_data)

    with patch("src.cleaning.Path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=json_content)):
        
        result = charger_modeles_json()

        # On vérifie que les listes ont fusionné sans doublons
        assert len(result["audi"]) == 3
        assert set(result["audi"]) == {"A3", "A4", "A5"}





@pytest.fixture
def ref_dict():
    return {
        "abarth": ["595", "595 Competizione", "695"],
        "audi": ["A3 Sportback", "A3", "Q3"],
        "bmw": ["Série 1", "X5"]
    }

def test_identifier_modele_priorite_long(ref_dict):
    from src.cleaning import identifier_modele
    # Doit identifier "595 Competizione" et non juste "595"
    m, reste, success = identifier_modele("Abarth", "595 Competizione Jaune", ref_dict)
    assert success is True
    assert m == "595 Competizione"
    assert reste == "Jaune"

def test_identifier_modele_tolerance_format(ref_dict):
    from src.cleaning import identifier_modele
    # Test de la recherche "loose" (espaces/tirets)
    m, reste, success = identifier_modele("Audi", "A-3 Sportback TDI", ref_dict)
    assert success is True
    assert m == "A3 Sportback"
    assert reste == "TDI"

def test_identifier_modele_inconnu(ref_dict):
    from src.cleaning import identifier_modele
    # Marque connue mais modèle absent du référentiel
    m, reste, success = identifier_modele("Audi", "Modèle Inexistant", ref_dict)
    assert success is False
    assert m == "Modèle Inexistant"

def test_raffiner_modele_integration():
    from src.cleaning import raffiner_modele_csv
    # 1. On prépare des données qui simulent le retour de reparer_marque_modele
    data = {
        "marque": ["Audi", "Abarth"],
        "modele": ["A3 Sportback TDI", "595 Competizione"]
    }
    df = pl.DataFrame(data)

    # 2. On mocke le JSON pour que le test ne dépende pas d'un fichier externe
    mock_dict = {
        "audi": ["A3", "A3 Sportback"],
        "abarth": ["595", "595 Competizione"]
    }

    with patch("src.cleaning.charger_modeles_json", return_value=mock_dict):
        result = raffiner_modele_csv(df)

    # 3. Vérifications
    # Pour Audi A3 Sportback TDI -> modele="A3 Sportback", reste="TDI"
    audi_row = result.filter(pl.col("marque") == "Audi")
    assert audi_row["modele"][0] == "A3 Sportback"
    assert audi_row["le_reste"][0] == "TDI"
    assert audi_row["modele_identifie"][0] is True

def test_main_pipeline_integration():
    # 1. Création d'un mini-dataset "sale"
    raw_data = pl.DataFrame({
        "prix": ["15 000 €"],
        "kilometrage": ["10 000 km"],
        "marque": ["Audi"],
        "modele": ["A3 Sportback TDI"],
        "puissance": ["110 kW"],
        "localisation": ["75001 Paris"],
        "transmission": ["Avant"],
        "carburant": ["Diesel"],
        "annee": [2022],
        "lien_fiche": ["http://autoscout.fr/123"]
    })

    # 2. Mock des entrées/sorties pour ne pas toucher au disque
    with patch("src.cleaning.charger_donnees", return_value=raw_data), \
         patch("src.cleaning.charger_modeles_json", return_value={"audi": ["A3", "A3 Sportback"]}), \
         patch("polars.DataFrame.write_json") as mock_write:
        
        from src.cleaning import main
        main()

        # 3. Vérification : write_json doit avoir été appelé
        assert mock_write.called
        # Tu peux même vérifier le chemin d'export
        args, _ = mock_write.call_args
        assert "autoscout_clean_ml.json" in args[0]