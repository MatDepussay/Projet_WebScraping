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
#     "plotly",
#     "numpy",
#     "streamlit",
#     "pydantic",
#     "selenium",
#     "beautifulsoup4",
#     "lxml",
# ]
# ///
import json
import re
import time
import io
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pickle
from sklearn.model_selection import train_test_split
import streamlit as st
from bs4 import BeautifulSoup as BS
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.safari.service import Service
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import polars as pl

import numpy as np

# Import des fonctions de nettoyage depuis cleaning.py
from cleaning import (
    nettoyer_numeriques,
    reparer_marque_modele,
    raffiner_modele_csv,
    extraire_specs_et_lieu,
    nettoyer_transmission,
    corriger_cylindree,
    traiter_valeurs_aberrantes,
    preparer_ml
)

# Import des fonctions ML depuis MachineLearning.py
from MachineLearning import (
    tune_random_forest,
    tune_xgboost,
    entrainer_random_forest,
    entrainer_xgboost,
    ajouter_cluster_vehicule,
    #comparer_modeles
)


# --- Modèles Pydantic ---
class Voiture(BaseModel):
    annonce_disponible: int = 1
    lien_fiche: str | None = None
    prix: str | None = None
    marque: str | None = None
    localisation: str | None = None
    kilometrage: int | None = None
    annee: int | None = None
    carburant: str | None = None
    boite_de_vitesse: str | None = None
    transmission: str | None = None
    puissance: str | None = None
    type_de_vehicule: str | None = None
    sieges: int | None = None
    portes: int | None = None
    annonce_disponible: int = 1


# --- 1. Récupérer la page listage avec Selenium ---
def recupere_page_listage(url: str) -> str:
    dir_path = Path(".").resolve()

    if "matde" in str(dir_path):
        options = ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(service=ChromeService(), options=options)
    else:
        driver = webdriver.Safari(service=Service())

    try:
        driver.get(url)

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "article"))
        )

        last_height = driver.execute_script("return document.body.scrollHeight")
        scroll_attempts = 0
        max_scrolls = 5

        while scroll_attempts < max_scrolls:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            scroll_attempts += 1

        return driver.page_source

    finally:
        driver.quit()


# --- 2. Extraire les URLs de la page listage ---
def extraire_urls_annonces(html_content: str) -> list[str]:
    soupe = BS(html_content, "lxml")
    articles = soupe.find_all("article")

    urls = []
    for art in articles:
        lien_tag = art.find("a", href=re.compile(r"^/offres/"))
        if lien_tag and lien_tag.get("href"):
            urls.append("https://www.autoscout24.fr" + lien_tag["href"])
    return urls


# --- 3. Récupérer une page d'annonce ---
def recupere_page_annonce(driver, url: str) -> str | None:
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(1)
        return driver.page_source
    except Exception:
        return None


# --- 4. Extraire les détails d'une page d'annonce ---
def extraire_details_annonce(html_content: str | None, url: str) -> dict:
    voiture = {"lien_fiche": url, "annonce_disponible": 1}
    if not html_content:
        return voiture

    soupe = BS(html_content, "lxml")

    # JSON-LD (robuste)
    json_ld_scripts = soupe.find_all("script", type="application/ld+json")
    json_data = None

    for script in json_ld_scripts:
        try:
            parsed = json.loads(script.string)
        except Exception:
            continue

        candidates = parsed if isinstance(parsed, list) else [parsed]
        for cand in candidates:
            if isinstance(cand, dict) and ("offers" in cand or cand.get("@type") == "Product"):
                json_data = cand
                break
        if json_data:
            break

    if json_data:
        try:
            if "offers" in json_data and "price" in json_data["offers"]:
                price = json_data["offers"]["price"]
                voiture["prix"] = f"€ {price:,.0f}".replace(",", " ")

            # Localisation depuis offeredBy/seller/availableAtOrFrom
            if "offers" in json_data and "offeredBy" in json_data["offers"]:
                offered_by = json_data["offers"]["offeredBy"]
                address = offered_by.get("address", {}) if isinstance(offered_by, dict) else {}
                city = address.get("addressLocality", "")
                postal = address.get("postalCode", "")
                if city or postal:
                    voiture["localisation"] = f"{postal} {city}".strip()

            if not voiture.get("localisation") and "offers" in json_data:
                offer = json_data["offers"]
                seller = offer.get("seller") or offer.get("offeredBy")
                if seller and isinstance(seller, dict):
                    address = seller.get("address", {})
                    city = address.get("addressLocality", "")
                    postal = address.get("postalCode", "")
                    if city or postal:
                        voiture["localisation"] = f"{postal} {city}".strip()

            if not voiture.get("localisation") and "availableAtOrFrom" in json_data:
                addr = json_data["availableAtOrFrom"].get("address", {})
                city = addr.get("addressLocality", "")
                postal = addr.get("postalCode", "")
                if city or postal:
                    voiture["localisation"] = f"{postal} {city}".strip()

            if "itemOffered" in json_data and "mileageFromOdometer" in json_data["itemOffered"]:
                km_data = json_data["itemOffered"]["mileageFromOdometer"]
                if isinstance(km_data, dict) and "value" in km_data:
                    voiture["kilometrage"] = int(km_data["value"])

            if "itemOffered" in json_data and "productionDate" in json_data["itemOffered"]:
                prod_date = json_data["itemOffered"]["productionDate"]
                if prod_date:
                    m = re.search(r"(\d{4})", str(prod_date))
                    if m:
                        voiture["annee"] = int(m.group(1))

            if "itemOffered" in json_data and "driveWheelConfiguration" in json_data["itemOffered"]:
                voiture["transmission"] = json_data["itemOffered"]["driveWheelConfiguration"]

            if "itemOffered" in json_data and "vehicleTransmission" in json_data["itemOffered"]:
                transmission_text = json_data["itemOffered"]["vehicleTransmission"]
                if isinstance(transmission_text, str):
                    if "automatique" in transmission_text.lower():
                        voiture["boite_de_vitesse"] = "Automatique"
                    elif "manuelle" in transmission_text.lower():
                        voiture["boite_de_vitesse"] = "Manuelle"
                    else:
                        voiture["boite_de_vitesse"] = transmission_text

            if "itemOffered" in json_data and "numberOfDoors" in json_data["itemOffered"]:
                try:
                    voiture["portes"] = int(json_data["itemOffered"]["numberOfDoors"])
                except Exception:
                    pass

            if "itemOffered" in json_data and "bodyType" in json_data["itemOffered"]:
                voiture["type_de_vehicule"] = json_data["itemOffered"]["bodyType"]

            if "itemOffered" in json_data and "vehicleEngine" in json_data["itemOffered"]:
                engines = json_data["itemOffered"].get("vehicleEngine")
                if engines:
                    engine = engines[0] if isinstance(engines, list) else engines
                    powers = engine.get("enginePower", []) if isinstance(engine, dict) else []
                    kw = None
                    hp = None
                    for power in powers:
                        if power.get("unitCode") == "KWT":
                            kw = power.get("value")
                        elif power.get("unitCode") == "BHP":
                            hp = power.get("value")
                    if kw and hp:
                        voiture["puissance"] = f"{kw} kW ({hp} CH)"
        except Exception:
            pass

    # HTML fallback
    titre_tag = soupe.find("h1")
    if titre_tag:
        voiture["marque"] = titre_tag.get_text(strip=True)

    if not voiture.get("prix"):
        prix_tag = soupe.find("span", class_=re.compile(r"PriceInfo_price"))
        if prix_tag:
            voiture["prix"] = prix_tag.get_text(strip=True)

    if not voiture.get("localisation"):
        location_links = soupe.find_all("a", href=re.compile(r"dealer.*location", re.I))
        for link in location_links:
            text = link.get_text(strip=True)
            if text and len(text) > 2:
                voiture["localisation"] = text
                break

    all_text = soupe.get_text()

    if not voiture.get("kilometrage"):
        km_match = re.search(r"([\d\s]+)\s*km", all_text, re.IGNORECASE)
        if km_match:
            km_str = km_match.group(1).replace(" ", "").replace("\xa0", "").replace("\u202f", "")
            try:
                voiture["kilometrage"] = int(km_str)
            except ValueError:
                pass

    if not voiture.get("annee"):
        annee_match = re.search(r"(\d{2})?/?(\d{4})", all_text)
        if annee_match:
            try:
                year = int(annee_match.group(2))
                if 1950 <= year <= 2030:
                    voiture["annee"] = year
            except ValueError:
                pass

    if not voiture.get("carburant"):
        carburants = ["Essence", "Diesel", "Électrique", "Hybride", "Gaz", "Hybrid"]
        lower = all_text.lower()
        for carb in carburants:
            if carb.lower() in lower:
                voiture["carburant"] = carb
                break

    if not voiture.get("boite_de_vitesse"):
        lower = all_text.lower()
        if "manuelle" in lower:
            voiture["boite_de_vitesse"] = "Manuelle"
        elif "automatique" in lower:
            voiture["boite_de_vitesse"] = "Automatique"

    if not voiture.get("transmission"):
        upper = all_text.upper()
        lower = all_text.lower()
        if "4X4" in upper:
            voiture["transmission"] = "4x4"
        elif "avant" in lower:
            voiture["transmission"] = "Avant"
        elif "arrière" in lower:
            voiture["transmission"] = "Arrière"

    if not voiture.get("puissance"):
        m = re.search(r"([\d\s]+)\s*kW\s*\(?\s*([\d\s]+)\s*CH\)?", all_text, re.IGNORECASE)
        if m:
            voiture["puissance"] = f"{m.group(1).strip()} kW ({m.group(2).strip()} CH)"

    data_grids = soupe.find_all("dl", class_=re.compile(r"DataGrid"))
    for grid in data_grids:
        dt_tags = grid.find_all("dt")
        dd_tags = grid.find_all("dd")

        for dt, dd in zip(dt_tags, dd_tags):
            label = dt.get_text(strip=True).lower()
            value = dd.get_text(strip=True)

            if "porte" in label and not voiture.get("portes"):
                m = re.search(r"(\d+)", value)
                if m:
                    voiture["portes"] = int(m.group(1))

            if ("siège" in label or "place" in label) and not voiture.get("sieges"):
                m = re.search(r"(\d+)", value)
                if m:
                    voiture["sieges"] = int(m.group(1))

            if "carburant" in label and not voiture.get("carburant"):
                voiture["carburant"] = value

            if "transmission" in label and not voiture.get("transmission"):
                voiture["transmission"] = value

            if "boîte" in label and not voiture.get("boite_de_vitesse"):
                voiture["boite_de_vitesse"] = value

            if "carrosserie" in label and not voiture.get("type_de_vehicule"):
                voiture["type_de_vehicule"] = value

    return voiture


def sauvegarder_json(voitures: list[Voiture], filename: str = "data/raw/annonces_autoscout24.json") -> Path:
    data = [v.model_dump() for v in voitures]
    filepath = Path(filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return filepath


def charger_voitures_depuis_json(uploaded_file) -> list[Voiture]:
    try:
        raw = uploaded_file.read()
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        data = json.loads(text)
    except Exception as exc:
        st.error(f"❌ Erreur lors de la lecture du fichier: {exc}")
        return []

    voitures: list[Voiture] = []
    for item in data:
        try:
            voitures.append(Voiture(**item))
        except Exception:
            pass
    return voitures


@st.cache_data(ttl=300, show_spinner="Chargement des données...")
def charger_voitures_depuis_fichier(filename: str = "data/raw/annonces_autoscout24.json") -> list[Voiture]:
    """Charge les voitures depuis un fichier JSON local (avec cache)"""
    filepath = Path(filename)
    if not filepath.exists():
        return []
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    
    voitures: list[Voiture] = []
    for item in data:
        try:
            voitures.append(Voiture(**item))
        except Exception:
            pass
    return voitures


@st.cache_data(ttl=600, show_spinner="Nettoyage des données en cours...")
def appliquer_cleaning(filename: str = "data/raw/annonces_autoscout24.json") -> pl.DataFrame:
    """Applique tout le pipeline de cleaning et retourne un DataFrame propre (avec cache)"""
    
    df = pl.read_json(filename)
    df = nettoyer_numeriques(df)
    df = reparer_marque_modele(df)
    df = raffiner_modele_csv(df)
    
    # Filtrer pour ne garder que les voitures avec modèle identifié
    if "modele_identifie" in df.columns:
        initial_count = df.height
        df = df.filter(pl.col("modele_identifie"))
        filtered_count = df.height
        print(f"✅ Filtrage modèles identifiés: {filtered_count}/{initial_count} voitures gardées")
    
    df = extraire_specs_et_lieu(df)
    df = nettoyer_transmission(df)
    df = corriger_cylindree(df)
    df = traiter_valeurs_aberrantes(df)
    df = preparer_ml(df)
    
    # Supprimer les lignes où il manque une info capitale (comme dans cleaning.py main)
    colonnes_essentielles = ["prix", "ville", "annee", "kilometrage", "marque", "modele"]
    colonnes_existantes = [col for col in colonnes_essentielles if col in df.columns]
    if colonnes_existantes:
        df = df.drop_nulls(colonnes_existantes)
    
    # Supprimer les colonnes inutiles pour le ML (comme dans cleaning.py main)
    # On conserve lien_fiche pour l'affichage des cartes.
    colonnes_a_supprimer = ["puissance", "modele_identifie", "le_reste"]
    colonnes_a_supprimer_existantes = [col for col in colonnes_a_supprimer if col in df.columns]
    if colonnes_a_supprimer_existantes:
        df = df.drop(colonnes_a_supprimer_existantes)
    
    print(f"✅ Nettoyage complet: {df.height} lignes après tous les filtrages")
    
    return df


def sauvegarder_donnees_nettoyees(df: pl.DataFrame, filename: str = "data/processed/voitures_nettoyees.json"):
    """Sauvegarde les données nettoyées en JSON"""
    df.write_json(filename)
    return Path(filename)


def fusionner_et_proteger_annonces(
    nouvelles_voitures: list[Voiture], 
    filename: str = "data/raw/annonces_autoscout24.json"
) -> tuple[Path, int]:
    """
    Fusionne les nouvelles annonces avec les existantes.
    Protège le fichier en s'assurant que le nombre d'annonces augmente seulement.
    Retourne le chemin du fichier et le nombre net d'annonces ajoutées.
    """
    existantes = charger_voitures_depuis_fichier(filename)
    
    # Créer un dictionnaire des URLs existantes pour éviter les doublons
    urls_existantes = {v.lien_fiche for v in existantes if v.lien_fiche}
    
    # Ajouter seulement les nouvelles voitures (par URL)
    voitures_a_ajouter = [v for v in nouvelles_voitures if v.lien_fiche not in urls_existantes]
    
    # Fusionner: existantes + nouvelles
    voitures_finales = existantes + voitures_a_ajouter
    
    # Sauvegarder
    filepath = Path(filename)
    data = [v.model_dump() for v in voitures_finales]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    return filepath, len(voitures_a_ajouter)



def run_scraping(nb_pages: int) -> list[Voiture] | None:
    url_base = (
        "https://www.autoscout24.fr/lst?atype=C&cy=D%2CA%2CB%2CE%2CF%2CI%2CL%2CNL"
        "&damaged_listing=exclude&desc=0&powertype=kw&search_id=k9p7elkop"
        "&sort=standard&source=listpage_pagination&ustate=N%2CU"
    )

    # --- PHASE 1: Récupération des pages de listage ---
    st.subheader("📋 Phase 1: Récupération des pages de listage")
    progress_listing = st.progress(0)
    status_listing = st.empty()
    
    urls_annonces_toutes: dict[str, bool] = {}
    
    for page in range(1, nb_pages + 1):
        status_listing.info(f"⏳ Récupération page {page}/{nb_pages}...")
        url_page = f"{url_base}&page={page}"
        html_listage = recupere_page_listage(url_page)
        urls_page = extraire_urls_annonces(html_listage)
        for url in urls_page:
            urls_annonces_toutes[url] = True
        
        progress_listing.progress(page / nb_pages)
    
    urls_annonces = list(urls_annonces_toutes.keys())
    status_listing.success(f"✅ {len(urls_annonces)} annonces uniques trouvées")
    
    if not urls_annonces:
        st.error("❌ Aucune annonce trouvée")
        return None

    # --- PHASE 2: Scraping des détails des annonces ---
    st.subheader("📖 Phase 2: Extraction des détails")
    progress_details = st.progress(0)
    status_details = st.empty()
    metric_placeholder = st.empty()
    
    dir_path = Path(".").resolve()
    if "matde" in str(dir_path):
        options = ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(service=ChromeService(), options=options)
    else:
        driver = webdriver.Safari(service=Service())

    try:
        liste_voitures: list[Voiture] = []
        total = len(urls_annonces)
        metric_placeholder = st.empty()
        
        for idx, url in enumerate(urls_annonces, 1):
            html_annonce = recupere_page_annonce(driver, url)
            details = extraire_details_annonce(html_annonce, url)
            try:
                liste_voitures.append(Voiture(**details))
            except Exception:
                pass
            
            # Mise à jour de la barre de progression
            progress = idx / total
            progress_details.progress(progress)
            
            # Mise à jour du statut et métrique unique
            status_details.info(f"🔄 {idx}/{total} annonces traitées ({int(progress * 100)}%)")
            
            # Mise à jour de la métrique unique (remplace l'ancienne à chaque itération)
            with metric_placeholder.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Annonces traitées", idx)
                with col2:
                    st.metric("Valides", len(liste_voitures))
                with col3:
                    st.metric("Échouées", idx - len(liste_voitures))
            
            time.sleep(1)

        if not liste_voitures:
            st.error("❌ Aucune annonce valide trouvée")
            return None

        return liste_voitures

    finally:
        driver.quit()


@st.cache_data(ttl=300)
def charger_donnees_nettoyees():
    """Charge et nettoie les données (avec cache)"""
    if not Path("data/raw/annonces_autoscout24.json").exists():
        return None
    return appliquer_cleaning("data/raw/annonces_autoscout24.json")

def afficher_selection_voitures():
    """Interface pour visualiser, filtrer et sélectionner les voitures nettoyées"""
    st.header("🔍 Sélection et visualisation des voitures")
    
    # Charger le modèle ML depuis la session state (chargé dans l'onglet ML)
    model = st.session_state.get("model_ml", None)
    model_source = st.session_state.get("model_source", "")
    
    if model_source:
        st.caption(f"📦 Modèle actif: {model_source}")
    else:
        st.info("ℹ️ Aucun modèle ML chargé. Allez dans l'onglet 'Régression ML' pour charger ou entraîner un modèle.")
    
    # Style CSS pour mettre les sliders en gris
    st.markdown("""
        <style>
        /* Style pour les sliders */
        div[data-baseweb="slider"] > div > div {
            background-color: #e0e0e0 !important;
        }
        div[data-baseweb="slider"] > div > div > div {
            background-color: #808080 !important;
        }
        div[data-baseweb="slider"] > div > div > div > div {
            background-color: #606060 !important;
        }
        /* Curseurs des sliders */
        div[data-baseweb="slider"] [role="slider"] {
            background-color: #505050 !important;
        }
        /* Chiffres/valeurs des sliders en noir */
        div[data-baseweb="slider"] [data-testid="stTickBar"] > div {
            color: #000000 !important;
        }
        div[data-baseweb="slider"] div[data-testid="stThumbValue"] {
            color: #000000 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Charger les données avec cache
    voitures_df = charger_donnees_nettoyees()
    if voitures_df is None:
        st.warning("⚠️ Aucune donnée disponible. Veuillez d'abord scraper les annonces.")
        return
    
    if voitures_df.height == 0:
        st.warning("⚠️ Aucune donnée valide après nettoyage.")
        return
    
    # Ajouter le clustering si un modèle est chargé (et si le clustering n'existe pas déjà)
    if model is not None and "cluster_vehicule" not in voitures_df.columns:
        try:
            st.info("🔍 Ajout du clustering pour les prédictions...")
            voitures_df = ajouter_cluster_vehicule(voitures_df, n_clusters=5)
            st.success("✅ Clustering ajouté")
        except Exception as e:
            st.warning(f"⚠️ Impossible d'ajouter le clustering: {e}")
    
    st.info(f"📊 Total: {voitures_df.height} voitures disponibles après nettoyage")
    
    # Calculer les valeurs min/max réelles pour les sliders
    prix_max_reel = int(voitures_df["prix"].max() or 200000)
    km_max_reel = int(voitures_df["kilometrage"].max() or 500000)
    puissance_max_reel = int(voitures_df["puissance_kw"].max() or 500)
    
    # --- Filtres ---
    st.subheader("⚙️ Filtres")
    
    # Première ligne de filtres
    col_filter1, col_filter2, col_filter3, col_filter4 = st.columns(4)
    
    with col_filter1:
        marques_disponibles = sorted(voitures_df["marque"].drop_nulls().unique().to_list())
        marque_selectionnee = st.multiselect("Marque", marques_disponibles, key="filter_marque")
     
    with col_filter2:
        # Filtrer les modèles selon la marque sélectionnée
        if marque_selectionnee:
            df_marques = voitures_df.filter(pl.col("marque").is_in(marque_selectionnee))
        else:
            df_marques = voitures_df
        modeles_disponibles = sorted(df_marques["modele"].drop_nulls().unique().to_list())
        modele_selectionne = st.multiselect("Modèle", modeles_disponibles, key="filter_modele")
    
    with col_filter3:
        st.write("**Prix (€)**")
        prix_min = st.number_input(
            "Prix minimum",
            min_value=0,
            max_value=int(prix_max_reel),
            value=0,
            step=500,
            key="filter_prix_min"
        )
        prix_max = st.number_input(
            "Prix maximum",
            min_value=0,
            max_value=int(prix_max_reel),
            value=int(prix_max_reel),
            step=500,
            key="filter_prix_max"
        )
    
    with col_filter4:
        carburants_disponibles = sorted(voitures_df["carburant"].drop_nulls().unique().to_list())
        carburant_selectionne = st.multiselect("Carburant", carburants_disponibles, key="filter_carburant")
    
    # Deuxième ligne de filtres
    col_filter5, col_filter6, col_filter7, col_filter8 = st.columns(4)
    
    with col_filter5:
        km_range = st.slider(
            "Kilométrage",
            min_value=0,
            max_value=km_max_reel,
            value=(0, km_max_reel),
            step=5000,
            key="filter_km_range"
        )
        km_min, km_max = km_range
    
    with col_filter6:
        portes_disponibles = sorted(voitures_df["portes"].drop_nulls().unique().to_list())
        portes_selectionnees = st.multiselect("Portes", portes_disponibles, key="filter_portes")
    
    with col_filter7:
        puissance_range = st.slider(
            "Puissance (kW)",
            min_value=0,
            max_value=puissance_max_reel,
            value=(0, puissance_max_reel),
            step=5,
            key="filter_puissance_range"
        )
        puissance_min, puissance_max = puissance_range
    
    with col_filter8:
        villes_disponibles = sorted(voitures_df["ville"].drop_nulls().unique().to_list())
        ville_selectionnee = st.multiselect("Ville", villes_disponibles, key="filter_ville")
    
    # Troisième ligne de filtres (Pays + Catégorie de prix)
    col_filter9, col_filter10, col_filter_vide = st.columns([1, 1, 2])
    
    with col_filter9:
        if "pays" in voitures_df.columns:
            pays_disponibles = sorted(voitures_df["pays"].drop_nulls().unique().to_list())
            pays_selectionne = st.multiselect("Pays", pays_disponibles, key="filter_pays")
        else:
            pays_selectionne = []
            
    with col_filter10:
        categorie_prix_selectionnee = None
        if model is not None:
            categories_disponibles = ["✅ Bonne Affaire", "⚠️ Normal", "❌ Arnaque"]
            categorie_prix_selectionnee = st.multiselect(
                "Catégorie de prix",
                categories_disponibles,
                default=categories_disponibles,
                key="filter_categorie_prix_header"
            )
    voitures_filtrees = voitures_df
    
    if marque_selectionnee:
        voitures_filtrees = voitures_filtrees.filter(pl.col("marque").is_in(marque_selectionnee))
    
    if modele_selectionne:
        voitures_filtrees = voitures_filtrees.filter(pl.col("modele").is_in(modele_selectionne))
    
    if carburant_selectionne:
        voitures_filtrees = voitures_filtrees.filter(pl.col("carburant").is_in(carburant_selectionne))
    
    if ville_selectionnee:
        voitures_filtrees = voitures_filtrees.filter(pl.col("ville").is_in(ville_selectionnee))
    
    if pays_selectionne:
        voitures_filtrees = voitures_filtrees.filter(pl.col("pays").is_in(pays_selectionne))
    
    if portes_selectionnees:
        voitures_filtrees = voitures_filtrees.filter(pl.col("portes").is_in(portes_selectionnees))
    
    voitures_filtrees = voitures_filtrees.filter(
        (pl.col("prix") >= prix_min) & (pl.col("prix") <= prix_max)
    )
    
    voitures_filtrees = voitures_filtrees.filter(
        (pl.col("kilometrage") >= km_min) & (pl.col("kilometrage") <= km_max)
    )
    
    voitures_filtrees = voitures_filtrees.filter(
        (pl.col("puissance_kw") >= puissance_min) & (pl.col("puissance_kw") <= puissance_max)
    )
    
    # Filtrer pour ne garder que les annonces disponibles
    if "annonce_disponible" in voitures_filtrees.columns:
        voitures_filtrees = voitures_filtrees.filter(pl.col("annonce_disponible") == 1)
    
    # Appliquer le filtre de catégorie de prix si sélectionné et modèle disponible
    # (Ce filtre sera appliqué après le calcul des prédictions)
    
    st.success(f"✅ {voitures_filtrees.height} voiture(s) correspondent aux critères")
    
    # --- Affichage des voitures ---
    
    if voitures_filtrees.height == 0:
        st.warning("Aucune voiture ne correspond aux critères de filtre.")
    else:
        # Préparer les données avec prédictions (pour Tableau ET Carte)
        colonnes_affichage = [
            "lien_fiche",
            "marque",
            "modele",
            "prix",
            "kilometrage",
            "annee",
            "carburant",
            "boite_de_vitesse",
            "ville",
            "code_postal",
            "puissance_kw",
        ]
        colonnes_valides = [col for col in colonnes_affichage if col in voitures_filtrees.columns]
        
        data_affichage = voitures_filtrees.select(colonnes_valides).to_dicts()
        
        # Ajouter les prédictions si le modèle est chargé
        predictions_disponibles = False
        if model is not None:
            try:
                # 1. Préparer les données pour la prédiction (Pandas)
                df_for_pred = voitures_filtrees.to_pandas()
                
                # 2. Nettoyage des colonnes comme lors de l'entraînement
                # On enlève le prix (la cible) et les colonnes textuelles inutiles
                cols_exclure = ["prix", "code_postal", "ville", "modele_identifie", "annonce_disponible"]
                X_base = df_for_pred.drop(columns=[c for c in cols_exclure if c in df_for_pred.columns])
                
                # 3. Forcer le type string pour le cluster avant l'encodage
                if "cluster_vehicule" in X_base.columns:
                    X_base["cluster_vehicule"] = X_base["cluster_vehicule"].astype(str)
                
                # 4. Encodage One-Hot (génère les colonnes marque_Audi, carburant_Diesel, etc.)
                X_encoded = pd.get_dummies(X_base)
                
                # 5. Alignement des features sur le schéma du modèle
                features_path = "models/model_features.pkl"
                expected_cols = None
                if hasattr(model, "get_booster"):
                    try:
                        expected_cols = model.get_booster().feature_names
                    except Exception:
                        expected_cols = None
                if not expected_cols and hasattr(model, "feature_names_in_"):
                    expected_cols = list(model.feature_names_in_)
                if not expected_cols and os.path.exists(features_path):
                    with open(features_path, "rb") as f:
                        expected_cols = pickle.load(f)

                if expected_cols:
                    X_final = X_encoded.reindex(columns=expected_cols, fill_value=0)
                    X_final = X_final[expected_cols]
                else:
                    X_final = X_encoded
                
                # 6. Faire les prédictions
                predictions = model.predict(X_final)
                predictions_disponibles = True
                
                # 7. Ajouter les prédictions et catégories aux données d'affichage
                for idx, v in enumerate(data_affichage):
                    prix_reel = v.get('prix', 0)
                    prix_predit = float(predictions[idx])
                    difference_pct = ((prix_reel - prix_predit) / prix_predit * 100) if prix_predit != 0 else 0
                    
                    v["prix_predit"] = prix_predit
                    v["difference_pct"] = difference_pct
                    
                    if difference_pct < -5:
                        v["categorie_prix"] = "✅ Bonne Affaire"
                    elif difference_pct > 5:
                        v["categorie_prix"] = "❌ Arnaque"
                    else:
                        v["categorie_prix"] = "⚠️ Normal"

            except Exception as e:
                st.error(f"❌ Erreur lors des prédictions: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        # Appliquer le filtre de catégorie si sélectionné
        if predictions_disponibles and categorie_prix_selectionnee:
            data_affichage = [row for row in data_affichage if row.get("categorie_prix") in categorie_prix_selectionnee]
            st.info(f"📊 {len(data_affichage)} voiture(s) après filtrage par catégorie")
        
        # Options d'affichage
        affichage_type = st.radio("Affichage", ("Tableau", "Carte"), horizontal=True, key="affichage_type")
        
        if affichage_type == "Tableau":
            
            # Formater les données pour l'affichage tableau (convertir tous les types pour éviter les erreurs Arrow)
            data_tableau = []
            for v in data_affichage:
                row = {
                    "Marque": str(v.get("marque") or "N/A"),
                    "Modèle": str(v.get("modele") or "N/A"),
                    "Prix réel (€)": f"{v.get('prix'):,.0f}".replace(",", " ") if v.get("prix") else "N/A",
                }
                
                # Ajouter prix prédit et catégorie si disponible
                if predictions_disponibles and "prix_predit" in v:
                    prix_predit = v.get('prix_predit', 0)
                    
                    row["Prix prédit (€)"] = f"{prix_predit:,.0f}".replace(",", " ")
                    row["Catégorie"] = str(v.get("categorie_prix", "⚠️ Normal"))
                
                row.update({
                    "Km": f"{v.get('kilometrage'):,}".replace(",", " ") if v.get("kilometrage") else "N/A",
                    "Année": str(v.get("annee")) if v.get("annee") else "N/A",
                    "Carburant": str(v.get("carburant") or "N/A"),
                    "Boîte": str(v.get("boite_de_vitesse") or "N/A"),
                    "Puissance (kW)": str(v.get("puissance_kw")) if v.get("puissance_kw") else "N/A",
                    "Ville": str(v.get("ville") or "N/A"),
                })
                data_tableau.append(row)
            
            # Afficher le tableau
            st.dataframe(data_tableau, width='stretch', height=500)
            
            # Export en JSON
            if st.button("📥 Exporter les résultats en JSON", key="export_json"):
                export_json = voitures_filtrees.write_json()
                st.download_button(
                    label="Télécharger JSON",
                    data=export_json,
                    file_name="selection_voitures.json",
                    mime="application/json"
                )
        
        else:  # Affichage en carte
            # Trier les données par différence de prix (croissant)
            if predictions_disponibles:
                data_affichage_sorted = sorted(
                    data_affichage,
                    key=lambda x: x.get((prix_predit-prix_reel), 0)
                )
            else:
                data_affichage_sorted = sorted(data_affichage, key=lambda x: x.get('marque', ''))
            
            for idx, voiture in enumerate(data_affichage_sorted, 1):
                # Construire l'en-tête avec la catégorie si disponible
                header = f"🚗 {voiture.get('marque')} {voiture.get('modele')} - {voiture.get('prix', 'N/A')}€"
                if predictions_disponibles and "categorie_prix" in voiture:
                    header = f"{voiture.get('categorie_prix')} {voiture.get('marque')} {voiture.get('modele')} - {voiture.get('prix', 'N/A')}€"
                
                with st.expander(header, expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Marque:** {voiture.get('marque') or 'N/A'}")
                        st.write(f"**Modèle:** {voiture.get('modele') or 'N/A'}")
                        st.write(f"**Cylindrée:** {voiture.get('cylindree_l') or 'N/A'} L")
                    
                    with col2:
                        st.write(f"**Prix:** {voiture.get('prix')}€" if voiture.get('prix') else "**Prix:** N/A")
                        km_formatted = f"{voiture.get('kilometrage'):,}".replace(',', ' ') if voiture.get('kilometrage') else None
                        st.write(f"**Km:** {km_formatted} km" if km_formatted else "**Km:** N/A")
                        st.write(f"**Année:** {voiture.get('annee') or 'N/A'}")
                    
                    with col3:
                        st.write(f"**Carburant:** {voiture.get('carburant') or 'N/A'}")
                        st.write(f"**Boîte:** {voiture.get('boite_de_vitesse') or 'N/A'}")
                        st.write(f"**Puissance:** {voiture.get('puissance_kw')} kW" if voiture.get('puissance_kw') else "**Puissance:** N/A")
                    # Afficher les infos de prédiction si disponibles
                    if predictions_disponibles and "prix_predit" in voiture:
                        st.divider()
                        col_pred1, col_pred2, col_pred3 = st.columns(3)
                        with col_pred1:
                            st.metric("Prix prédit", f"€{voiture.get('prix_predit'):,.0f}".replace(",", " "))
                        with col_pred2:
                            diff_pct = voiture.get('difference_pct', 0)
                            st.metric("Différence", f"{diff_pct:+.1f}%")
                        with col_pred3:
                            st.metric("Catégorie", voiture.get('categorie_prix', 'N/A'))
                    
                    
                    st.write(f"**Localisation:** {voiture.get('code_postal')} {voiture.get('ville')}" if voiture.get('ville') else "**Localisation:** N/A")
                    
                    if voiture.get('lien_fiche'):
                        st.write(f"**[🔗 Lien de l'annonce]({voiture.get('lien_fiche')})**")


def afficher_resultats_modele(model, X_test, y_test, feature_importance=None):
    """Affiche les graphiques et résultats d'un modèle ML (réutilisable)"""
    
    # Prédictions
    y_pred_test = model.predict(X_test)
    
    # Calculer les métriques
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    
    # Affichage des métriques
    st.subheader("📊 Résultats")
    
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    
    with col_metric1:
        st.metric("RMSE", f"€{rmse:,.0f}")
    with col_metric2:
        st.metric("R² Score", f"{r2:.4f}")
    with col_metric3:
        st.metric("MAE", f"€{mae:,.0f}")
    
    # Feature Importance (si disponible)
    if feature_importance is not None:
        st.subheader("🎯 Importance des features")
        
        col_fi1, col_fi2 = st.columns(2)
        
        with col_fi1:
            st.dataframe(feature_importance.head(10), width='stretch')
        
        with col_fi2:
            fig_fi = px.bar(
                feature_importance.head(10),
                x='importance',
                y='feature',
                orientation='h',
                title='Top 10 Features',
                labels={'importance': 'Importance', 'feature': 'Feature'}
            )
            fig_fi.update_layout(height=400)
            st.plotly_chart(fig_fi, width='stretch')
    
    # Visualisation: Prix réel vs Prix prédit
    st.subheader("💰 Prix réel vs Prix prédit")
    
    # Créer un DataFrame pour les prédictions
    results_df = pd.DataFrame({
        'Prix réel': y_test.values,
        'Prix prédit': y_pred_test,
        'Erreur (€)': y_test.values - y_pred_test,
        'Erreur (%)': ((y_test.values - y_pred_test) / y_test.values * 100)
    }).reset_index(drop=True)
    
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        # Graphique de dispersion
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=y_test.values,
            y=y_pred_test,
            mode='markers',
            marker=dict(
                size=6,
                color=results_df['Erreur (%)'].abs(),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Erreur (%)")
            ),
            text=[f"Réel: €{r:,.0f}<br>Prédit: €{p:,.0f}<br>Erreur: {e:+.1f}%" 
                  for r, p, e in zip(y_test.values, y_pred_test, results_df['Erreur (%)'])],
            hoverinfo='text',
            name='Prédictions'
        ))
        
        # Ajouter la ligne de perfection
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfection',
            line=dict(dash='dash', color='red')
        ))
        
        fig_scatter.update_layout(
            title='Prédictions vs Réalité',
            xaxis_title='Prix réel (€)',
            yaxis_title='Prix prédit (€)',
            height=500,
            hovermode='closest'
        )
        st.plotly_chart(fig_scatter, width='stretch')
    
    with col_viz2:
        # Graphique des erreurs
        fig_error = go.Figure()
        
        # Séparer les erreurs positives et négatives
        erreurs_pos = results_df[results_df['Erreur (%)'] > 0]['Erreur (%)']
        erreurs_neg = results_df[results_df['Erreur (%)'] <= 0]['Erreur (%)']
        
        # Histogramme pour erreurs négatives (sur-estimation)
        fig_error.add_trace(go.Histogram(
            x=erreurs_neg,
            nbinsx=30,
            name='Sur-estimation (prix prédit > réel)',
            marker_color='rgba(255, 100, 100, 0.7)',
            opacity=0.7
        ))
        
        # Histogramme pour erreurs positives (sous-estimation)
        fig_error.add_trace(go.Histogram(
            x=erreurs_pos,
            nbinsx=30,
            name='Sous-estimation (prix prédit < réel)',
            marker_color='rgba(100, 150, 255, 0.7)',
            opacity=0.7
        ))
        
        # Ajouter une ligne verticale à 0
        fig_error.add_vline(
            x=0, 
            line_dash="dash", 
            line_color="black",
            annotation_text="Erreur = 0",
            annotation_position="top"
        )
        
        fig_error.update_layout(
            title='Distribution des erreurs (%)',
            xaxis_title='Erreur (%)',
            yaxis_title='Nombre de prédictions',
            height=500,
            barmode='overlay',
            showlegend=True
        )
        st.plotly_chart(fig_error, width='stretch')
    
    # Tableau détaillé
    st.subheader("📋 Détails des prédictions")
    st.dataframe(results_df, width='stretch')
    
    # Statistiques des erreurs
    st.subheader("📈 Statistiques des erreurs")
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.metric("Erreur moyenne (%)", f"{results_df['Erreur (%)'].mean():+.2f}%")
    with col_stat2:
        st.metric("Écart-type erreur (%)", f"{results_df['Erreur (%)'].std():.2f}%")
    with col_stat3:
        st.metric("Min erreur (%)", f"{results_df['Erreur (%)'].min():+.2f}%")
    with col_stat4:
        st.metric("Max erreur (%)", f"{results_df['Erreur (%)'].max():+.2f}%")


def afficher_regression_ml():
    """Interface Streamlit pour entraîner les modèles ML"""

    
    st.header("📊 Régression - Prédiction des Prix")
    
    # Charger le modèle ML si disponible
    st.subheader("📦 Charger ou gérer un modèle ML")
    col_upload, col_path = st.columns(2)
    with col_upload:
        uploaded_model = st.file_uploader("Importer un modèle .pkl", type=["pkl"], key="upload_model_regression")
        if uploaded_model:
            try:
                model_loaded = pickle.load(io.BytesIO(uploaded_model.read()))
                st.session_state.model_ml = model_loaded
                st.session_state.model_source = "Modèle importé (uploader)"
                st.success("✅ Modèle chargé depuis l'upload")
            except Exception as e:
                st.error(f"❌ Échec du chargement du modèle uploadé: {e}")
    with col_path:
        default_model_path = "models/random_forest_model.pkl"
        custom_model_path = st.text_input("Chemin vers un modèle local", value=default_model_path, key="custom_model_path_regression")
        if st.button("Charger ce fichier", key="load_custom_model_regression"):
            path_obj = Path(custom_model_path)
            if path_obj.exists():
                try:
                    with open(path_obj, "rb") as f:
                        model_loaded = pickle.load(f)
                    st.session_state.model_ml = model_loaded
                    st.session_state.model_source = f"Modèle local: {path_obj}"
                    st.success(f"✅ Modèle chargé depuis {path_obj}")
                except Exception as e:
                    st.error(f"❌ Échec du chargement: {e}")
            else:
                st.warning("⚠️ Fichier introuvable")
    
    # Chercher le modèle Random Forest ou XGBoost par défaut si aucun modèle n'a été fourni
    if "model_ml" not in st.session_state:
        model_path_rf = Path("models/random_forest_model.pkl")
        model_path_xgb = Path("models/xgboost_model.pkl")
        model_path_old = Path("models/random_forest.pkl")
        
        if model_path_rf.exists():
            try:
                with open(model_path_rf, "rb") as f:
                    st.session_state.model_ml = pickle.load(f)
                st.session_state.model_source = str(model_path_rf)
                st.info("✅ Modèle ML (Random Forest) chargé par défaut")
            except Exception as e:
                st.warning(f"⚠️ Impossible de charger le modèle Random Forest: {e}")
        elif model_path_xgb.exists():
            try:
                with open(model_path_xgb, "rb") as f:
                    st.session_state.model_ml = pickle.load(f)
                st.session_state.model_source = str(model_path_xgb)
                st.info("✅ Modèle ML (XGBoost) chargé par défaut")
            except Exception as e:
                st.warning(f"⚠️ Impossible de charger le modèle XGBoost: {e}")
        elif model_path_old.exists():
            try:
                with open(model_path_old, "rb") as f:
                    st.session_state.model_ml = pickle.load(f)
                st.session_state.model_source = str(model_path_old)
                st.info("✅ Modèle ML chargé par défaut")
            except Exception as e:
                st.warning(f"⚠️ Impossible de charger le modèle: {e}")
    
    if "model_source" in st.session_state:
        st.caption(f"Modèle actif: {st.session_state.model_source}")
    
    st.divider()
    
    # Vérifier que le fichier JSON des annonces existe
    if not Path("data/raw/annonces_autoscout24.json").exists():
        st.warning("⚠️ Fichier annonces_autoscout24.json introuvable. Lancez un scraping ou chargez un JSON.")
        return
    
    # Utiliser le cache pour charger les données nettoyées
    df_clean_pl = charger_donnees_nettoyees()
    if df_clean_pl is None:
        st.error("❌ Erreur lors du chargement des données.")
        return
    
    df_pd = df_clean_pl.to_pandas()
    
    if "prix" not in df_pd.columns:
        st.error("❌ La colonne 'prix' est manquante après nettoyage. Vérifiez `preparer_ml` dans cleaning.py.")
        return
    
    st.info(f"📈 Données nettoyées: {len(df_pd)} lignes, {len(df_pd.columns)} colonnes")
    
    # --- Paramètres de configuration ---
    st.subheader("⚙️ Configuration du modèle")
    
    col_config1, col_config2, col_config3 , = st.columns(3)
    
    with col_config1:
        model_type = st.selectbox(
            "Modèle",
            ["Random Forest", "XGBoost"],
            key="model_selection"
        )
    
    with col_config2:
        test_split = st.slider(
            "Test Split (%)",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            key="test_split_slider"
        ) / 100
    
    with col_config3:
        st.selectbox(
            "Métrique principale",
            ["R² Score", "RMSE", "MAE"],
            key="metric_selection"
        )
    # Random state fixé à 42
    random_state = 42
        # Bouton pour lancer l'entraînement
    if st.button("🚀 Lancer l'entraînement", width='stretch'):
        with st.spinner("⏳ Préparation des données..."):
            try:
                # Ajouter le clustering (comme dans MachineLearning.py)
                st.info("🔍 Ajout du clustering des véhicules...")
                df_clean_pl_clustered = ajouter_cluster_vehicule(df_clean_pl, n_clusters=5)
                df_pd = df_clean_pl_clustered.to_pandas()
                
                # Construire X/y depuis les données nettoyées (via cleaning.py)
                y = df_pd["prix"]
                cols_exclure = ["prix", "code_postal", "ville", "modele_identifie"]
                X = df_pd.drop(columns=[c for c in cols_exclure if c in df_pd.columns])
                
                cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
                X_encoded = pd.get_dummies(X, columns=cat_cols)
                
                # S'assurer que les features sont numériques pour sklearn
                # 3. ALIGNEMENT (Si tu évalues un modèle existant)
                if st.session_state.get("model_ml") is not None:
                    model = st.session_state.model_ml
                    # Récupérer les colonnes attendues par le modèle
                    if hasattr(model, "feature_names_in_"): # Pour Random Forest
                        expected_cols = model.feature_names_in_
                    else: # Pour XGBoost
                        expected_cols = model.get_booster().feature_names
                    
                    # Forcer l'alignement des colonnes
                    X_final = X_encoded.reindex(columns=expected_cols, fill_value=0)
                  
                    if not hasattr(model, "feature_names_in_"):
                        X_final = X_final[expected_cols]
                else:
                    X_final = X_encoded  
                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_final, y, test_size=test_split, random_state=random_state
                )
                
                st.success(f"✅ Données préparées: Train={len(X_train)}, Test={len(X_test)}")
                
                # Tuning puis entraînement du modèle
                if model_type == "Random Forest":
                    with st.spinner("🔍 Tuning des hyperparamètres Random Forest..."):
                        model_tune = tune_random_forest(X_train, y_train)
                    st.success("✅ Tuning terminé")
                    
                    with st.spinner("⏳ Entraînement final Random Forest..."):
                        results = entrainer_random_forest(model_tune, X_train, X_test, y_train, y_test)
                else:
                    with st.spinner("🔍 Tuning des hyperparamètres XGBoost..."):
                        model_tune = tune_xgboost(X_train, y_train)
                    st.success("✅ Tuning terminé")
                    
                    with st.spinner("⏳ Entraînement final XGBoost..."):
                        results = entrainer_xgboost(model_tune, X_train, X_test, y_train, y_test)
                
                model = results['model']
                
                # Sauvegarder le modèle en session state
                st.session_state.model_ml = model
                st.session_state.model_source = f"Modèle entraîné ({model_type})"
                
                # Sauvegarde et export du modèle entraîné
                st.subheader("💾 Sauvegarder le modèle entraîné")
                models_dir = Path("models")
                models_dir.mkdir(parents=True, exist_ok=True)

                default_filename = models_dir / f"{model_type.lower().replace(' ', '_')}_model.pkl"
                filename_input = st.text_input(
                    "Nom de fichier",
                    value=str(default_filename),
                    help="Chemin local où enregistrer le modèle picklé"
                )

                col_save_local, col_download = st.columns(2)
                with col_save_local:
                    if st.button("💾 Enregistrer sur le disque", width='stretch'):
                        try:
                            target_path = Path(filename_input)
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(target_path, "wb") as f:
                                pickle.dump(model, f)
                            st.success(f"Modèle sauvegardé: {target_path}")
                        except Exception as e:
                            st.error(f"Erreur lors de la sauvegarde: {e}")

                with col_download:
                    try:
                        model_bytes = pickle.dumps(model)
                        st.download_button(
                            label="⬇️ Télécharger le modèle (.pkl)",
                            data=model_bytes,
                            file_name=Path(filename_input).name,
                            mime="application/octet-stream",
                            width='stretch'
                        )
                    except Exception as e:
                        st.error(f"Erreur lors de la préparation du téléchargement: {e}")
                
                # Afficher les résultats avec la fonction réutilisable
                afficher_resultats_modele(model, X_test, y_test, results['feature_importance'])
                
            except Exception as e:
                st.error(f"❌ Erreur: {e}")
    # Évaluer le modèle chargé si disponible
    if st.session_state.get("model_ml") is not None:
        st.divider()
        st.subheader("📊 Évaluation du modèle chargé")
        
        if st.button("📈 Évaluer ce modèle sur les données actuelles", width='stretch'):
            try:
                # 1. Clustering et Préparation (comme avant)
                df_clean_pl_clustered = ajouter_cluster_vehicule(df_clean_pl, n_clusters=5)
                df_pd_eval = df_clean_pl_clustered.to_pandas()
                y_all = df_pd_eval["prix"]
                X_all = df_pd_eval.drop(columns=["prix", "code_postal", "ville", "modele_identifie"], errors="ignore")
                X_all["cluster_vehicule"] = X_all["cluster_vehicule"].astype(str)
                X_encoded = pd.get_dummies(X_all)

                # 2. CHARGEMENT / DÉDUCTION DU SCHÉMA DE FEATURES
                features_path = Path("models/model_features.pkl")
                model = st.session_state.model_ml
                expected_cols = None
                if hasattr(model, "get_booster"):
                    try:
                        expected_cols = model.get_booster().feature_names
                    except Exception:
                        expected_cols = None
                if not expected_cols and hasattr(model, "feature_names_in_"):
                    expected_cols = list(model.feature_names_in_)
                if not expected_cols and features_path.exists():
                    with open(features_path, "rb") as f:
                        expected_cols = pickle.load(f)

                # 3. ALIGNEMENT STRICT
                if expected_cols:
                    X_final = X_encoded.reindex(columns=expected_cols, fill_value=0)
                    X_final = X_final[expected_cols]
                else:
                    X_final = X_encoded

                # 4. Split et Prédictions
                _, X_test, _, y_test = train_test_split(X_final, y_all, test_size=0.2, random_state=42)
                afficher_resultats_modele(model, X_test, y_test, feature_importance=None)

            except Exception as e:
                st.error(f"❌ Erreur : {e}")
                import traceback
                st.code(traceback.format_exc())
        
        st.divider()




# --- STREAMLIT UI ---
st.set_page_config(page_title="AutoScout24 Scraper", layout="wide")
st.title("🚗 Scraping AutoScout24")

# Bouton pour vider le cache dans la sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    if st.button("🗑️ Vider le cache", help="Vide le cache des données pour forcer le rechargement"):
        st.cache_data.clear()
        st.success("✅ Cache vidé !")
        st.rerun()
    
    st.divider()
    st.caption("💡 Le cache accélère l'application en mémorisant les données nettoyées pendant 5-10 minutes.")

# Créer les onglets
tab1, tab2, tab3 = st.tabs(["📥 Scraper", "📊 Régression ML", "🔍 Sélectionner"])

with tab1:
    st.write("Gérez vos données d'annonces AutoScout24")

    mode = st.radio(
        "Mode de fonctionnement",
        ("🔄 Redémarrer de zéro", "📂 Charger un JSON", "➕ Ajouter des données"),
        horizontal=True,
    )

    col1, col2 = st.columns([3, 1])

    if mode == "🔄 Redémarrer de zéro":
        with col1:
            nb_pages = st.number_input("Nombre de pages à scraper", min_value=1, max_value=50, value=5, step=1)
        with col2:
            st.write("")
            st.write("")
            if st.button("🚀 Lancer le scrapping", width='stretch'):
                liste_voitures = run_scraping(int(nb_pages))
                if liste_voitures:
                    # --- PHASE 3: Sauvegarde JSON ---
                    st.subheader("💾 Phase 3: Sauvegarde du fichier JSON")
                    filepath = sauvegarder_json(liste_voitures, "data/raw/annonces_autoscout24.json")
                    st.success(f"✅ {len(liste_voitures)} annonces sauvegardées")
                    
                    # --- PHASE 4: Nettoyage des données ---
                    st.subheader("🧹 Phase 4: Nettoyage des données")
                    try:
                        with st.spinner("Nettoyage en cours..."):
                            df_nettoyees = appliquer_cleaning("data/raw/annonces_autoscout24.json")
                        sauvegarder_donnees_nettoyees(df_nettoyees, "data/processed/voitures_nettoyees.json")
                        st.success(f"✅ {df_nettoyees.height} annonces nettoyées et sauvegardées")
                        st.balloons()
                        
                        # Affichage des stats
                        col_success_1, col_success_2, col_success_3 = st.columns(3)
                        with col_success_1:
                            st.metric("Brutes", len(liste_voitures))
                        with col_success_2:
                            st.metric("Nettoyées", df_nettoyees.height)
                        with col_success_3:
                            st.metric("Ratio", f"{(df_nettoyees.height / len(liste_voitures) * 100):.1f}%")
                    except Exception as e:
                        st.error(f"❌ Erreur lors du nettoyage: {e}")

    elif mode == "📂 Charger un JSON":
        with col1:
            uploaded_file = st.file_uploader("Fichier JSON existant", type=["json"])
        with col2:
            st.write("")
            st.write("")
            if st.button("📂 Charger le JSON", width='stretch'):
                if not uploaded_file:
                    st.warning("⚠️ Sélectionnez un fichier JSON pour continuer")
                else:
                    voitures = charger_voitures_depuis_json(uploaded_file)
                    if not voitures:
                        st.error("❌ Aucun enregistrement valide dans le fichier")
                    else:
                        st.success(f"✅ {len(voitures)} annonces chargées depuis le JSON")
                        col_success_1, col_success_2 = st.columns(2)
                        with col_success_1:
                            st.metric("Fichier chargé", uploaded_file.name)
                        with col_success_2:
                            st.metric("Nombre d'annonces", len(voitures))

    else:  # Mode "➕ Ajouter des données"
        with col1:
            nb_pages = st.number_input("Nombre de pages à scraper", min_value=1, max_value=50, value=5, step=1)
        with col2:
            st.write("")
            st.write("")
            if st.button("➕ Ajouter des données", width='stretch'):
                # Vérifier que le fichier existant existe
                if not Path("data/raw/annonces_autoscout24.json").exists():
                    st.error("❌ Le fichier annonces_autoscout24.json n'existe pas. Utilisez 'Redémarrer de zéro' d'abord.")
                else:
                    # Afficher le nombre d'annonces actuellement
                    voitures_actuelles = charger_voitures_depuis_fichier("data/raw/annonces_autoscout24.json")
                    st.info(f"📊 Fichier actuel contient {len(voitures_actuelles)} annonces")
                    
                    # Scraper les nouvelles données
                    liste_voitures = run_scraping(int(nb_pages))
                    if liste_voitures:
                        # --- PHASE 3: Fusion et sauvegarde protégée ---
                        st.subheader("💾 Phase 3: Fusion et sauvegarde protégée")
                        filepath, ajoutees = fusionner_et_proteger_annonces(liste_voitures, "data/raw/annonces_autoscout24.json")
                        
                        # Recharger pour vérifier
                        voitures_finales = charger_voitures_depuis_fichier("data/raw/annonces_autoscout24.json")
                        st.success(f"✅ {ajoutees} nouvelles annonces ajoutées")
                        
                        # --- PHASE 4: Nettoyage des données ---
                        st.subheader("🧹 Phase 4: Nettoyage des données mises à jour")
                        try:
                            with st.spinner("Nettoyage en cours..."):
                                df_nettoyees = appliquer_cleaning("data/raw/annonces_autoscout24.json")
                            sauvegarder_donnees_nettoyees(df_nettoyees, "data/processed/voitures_nettoyees.json")
                            st.success(f"✅ {df_nettoyees.height} annonces nettoyées au total")
                            st.balloons()
                            
                            # Affichage des stats
                            col_stats_1, col_stats_2, col_stats_3 = st.columns(3)
                            with col_stats_1:
                                st.metric("Avant", len(voitures_actuelles))
                            with col_stats_2:
                                st.metric("Ajoutées", ajoutees)
                            with col_stats_3:
                                st.metric("Total (brutes)", len(voitures_finales))
                            
                            col_clean_1, col_clean_2 = st.columns(2)
                            with col_clean_1:
                                st.metric("Nettoyées", df_nettoyees.height)
                            with col_clean_2:
                                st.metric("Ratio qualité", f"{(df_nettoyees.height / len(voitures_finales) * 100):.1f}%")
                        except Exception as e:
                            st.error(f"❌ Erreur lors du nettoyage: {e}")
    
    # Bouton pour forcer le nettoyage des données existantes
    st.divider()
    col_clean_force_1, col_clean_force_2 = st.columns([3, 1])
    with col_clean_force_1:
        st.write("**🧹 Forcer le nettoyage des données existantes**")
        st.caption("Nettoie le fichier annonces_autoscout24.json et régénère voitures_nettoyees.json")
    with col_clean_force_2:
        if st.button("🧹 Nettoyer", width='stretch', type="secondary"):
            if not Path("data/raw/annonces_autoscout24.json").exists():
                st.error("❌ Fichier annonces_autoscout24.json introuvable")
            else:
                try:
                    with st.spinner("Nettoyage en cours..."):
                        df_nettoyees = appliquer_cleaning("data/raw/annonces_autoscout24.json")
                    sauvegarder_donnees_nettoyees(df_nettoyees, "data/processed/voitures_nettoyees.json")
                    st.success(f"✅ {df_nettoyees.height} annonces nettoyées avec succès")
                    
                    col_force_1, col_force_2 = st.columns(2)
                    with col_force_1:
                        voitures_brutes = len(charger_voitures_depuis_fichier("annonces_autoscout24.json"))
                        st.metric("Annonces brutes", voitures_brutes)
                    with col_force_2:
                        st.metric("Annonces nettoyées", df_nettoyees.height)
                except Exception as e:
                    st.error(f"❌ Erreur lors du nettoyage: {e}")

with tab2:
    afficher_regression_ml()

with tab3:
    afficher_selection_voitures()
