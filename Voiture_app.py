import json
import re
import time
from pathlib import Path

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

# Import des fonctions de nettoyage depuis cleaning.py
from cleaning import (
    eliminer_doublons,
    nettoyer_prix,
    extraire_puissance_kw,
    separer_marque_modele,
    separer_specs,
    normaliser_localisation,
    normaliser_types
)


# --- Modèles Pydantic ---
class Voiture(BaseModel):
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
    voiture = {"lien_fiche": url}
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


def sauvegarder_json(voitures: list[Voiture], filename: str = "annonces_autoscout24.json") -> Path:
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


def charger_voitures_depuis_fichier(filename: str = "annonces_autoscout24.json") -> list[Voiture]:
    """Charge les voitures depuis un fichier JSON local"""
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


def appliquer_cleaning(filename: str = "annonces_autoscout24.json") -> pl.DataFrame:
    """Applique tout le pipeline de cleaning et retourne un DataFrame propre"""
    df = pl.read_json(filename)
    df = eliminer_doublons(df)
    df = nettoyer_prix(df)
    df = extraire_puissance_kw(df)
    df = separer_marque_modele(df)
    df = separer_specs(df)
    df = normaliser_types(df)
    df = normaliser_localisation(df)
    
    # Retirer les lignes avec prix, km ou année manquants
    df = df.drop_nulls(["prix", "kilometrage", "annee"])
    
    return df


def sauvegarder_donnees_nettoyees(df: pl.DataFrame, filename: str = "voitures_nettoyees.json"):
    """Sauvegarde les données nettoyées en JSON"""
    df.write_json(filename)
    return Path(filename)


def fusionner_et_proteger_annonces(
    nouvelles_voitures: list[Voiture], 
    filename: str = "annonces_autoscout24.json"
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


def afficher_selection_voitures():
    """Interface pour visualiser, filtrer et sélectionner les voitures nettoyées"""
    st.header("🔍 Sélection et visualisation des voitures")
    
    # Vérifier que les fichiers existent
    if not Path("annonces_autoscout24.json").exists():
        st.warning("⚠️ Aucune donnée disponible. Veuillez d'abord scraper les annonces.")
        return
    
    # Charger et nettoyer les données
    try:
        voitures_df = appliquer_cleaning("annonces_autoscout24.json")
    except Exception as e:
        st.error(f"❌ Erreur lors du nettoyage des données: {e}")
        return
    
    if voitures_df.height == 0:
        st.warning("⚠️ Aucune donnée valide après nettoyage.")
        return
    
    st.info(f"📊 Total: {voitures_df.height} voitures disponibles après nettoyage")
    
    # --- Filtres ---
    st.subheader("⚙️ Filtres")
    
    col_filter1, col_filter2, col_filter3, col_filter4 = st.columns(4)
    
    with col_filter1:
        marques_disponibles = sorted(voitures_df["marque"].drop_nulls().unique().to_list())
        marque_selectionnee = st.multiselect("Marque", marques_disponibles, key="filter_marque")
    
    with col_filter2:
        prix_min = st.number_input("Prix min (€)", value=0, step=500, key="filter_prix_min")
        prix_max = st.number_input("Prix max (€)", value=100000, step=500, key="filter_prix_max")
    
    with col_filter3:
        carburants_disponibles = sorted(voitures_df["carburant"].drop_nulls().unique().to_list())
        carburant_selectionne = st.multiselect("Carburant", carburants_disponibles, key="filter_carburant")
    
    with col_filter4:
        villes_disponibles = sorted(voitures_df["ville"].drop_nulls().unique().to_list())
        ville_selectionnee = st.multiselect("Ville", villes_disponibles, key="filter_ville")
    
    # Appliquer les filtres
    voitures_filtrees = voitures_df
    
    if marque_selectionnee:
        voitures_filtrees = voitures_filtrees.filter(pl.col("marque").is_in(marque_selectionnee))
    
    if carburant_selectionne:
        voitures_filtrees = voitures_filtrees.filter(pl.col("carburant").is_in(carburant_selectionne))
    
    if ville_selectionnee:
        voitures_filtrees = voitures_filtrees.filter(pl.col("ville").is_in(ville_selectionnee))
    
    voitures_filtrees = voitures_filtrees.filter(
        (pl.col("prix") >= prix_min) & (pl.col("prix") <= prix_max)
    )
    
    st.success(f"✅ {voitures_filtrees.height} voiture(s) correspondent aux critères")
    
    # --- Affichage des voitures ---
    st.subheader("📋 Résultats")
    
    if voitures_filtrees.height == 0:
        st.warning("Aucune voiture ne correspond aux critères de filtre.")
    else:
        # Options d'affichage
        affichage_type = st.radio("Affichage", ("Tableau", "Carte"), horizontal=True, key="affichage_type")
        
        if affichage_type == "Tableau":
            # Préparer les données pour le tableau
            data_affichage = voitures_filtrees.select([
                "marque", "modele", "moteur", "prix", "kilometrage", "annee",
                "carburant", "boite_de_vitesse", "ville", "code_postal", "puissance_kw"
            ]).to_dicts()
            
            # Formater les données pour l'affichage
            data_tableau = []
            for v in data_affichage:
                data_tableau.append({
                    "Marque": v.get("marque") or "N/A",
                    "Modèle": v.get("modele") or "N/A",
                    "Moteur": v.get("moteur") or "N/A",
                    "Prix (€)": f"{v.get('prix'):,}".replace(",", " ") if v.get("prix") else "N/A",
                    "Km": f"{v.get('kilometrage'):,}".replace(",", " ") if v.get("kilometrage") else "N/A",
                    "Année": v.get("annee") or "N/A",
                    "Carburant": v.get("carburant") or "N/A",
                    "Boîte": v.get("boite_de_vitesse") or "N/A",
                    "Puissance (kW)": v.get("puissance_kw") or "N/A",
                    "Ville": v.get("ville") or "N/A",
                })
            
            st.dataframe(data_tableau, use_container_width=True, height=500)
            
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
            voitures_list = voitures_filtrees.to_dicts()
            for idx, voiture in enumerate(voitures_list, 1):
                with st.expander(
                    f"🚗 {voiture.get('marque')} {voiture.get('modele')} - {voiture.get('prix', 'N/A')}€",
                    expanded=False
                ):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Marque:** {voiture.get('marque') or 'N/A'}")
                        st.write(f"**Modèle:** {voiture.get('modele') or 'N/A'}")
                        st.write(f"**Moteur:** {voiture.get('moteur') or 'N/A'}")
                        st.write(f"**Pack:** {voiture.get('pack') or 'N/A'}")
                    
                    with col2:
                        st.write(f"**Prix:** {voiture.get('prix')}€" if voiture.get('prix') else "**Prix:** N/A")
                        km_formatted = f"{voiture.get('kilometrage'):,}".replace(',', ' ') if voiture.get('kilometrage') else None
                        st.write(f"**Km:** {km_formatted} km" if km_formatted else "**Km:** N/A")
                        st.write(f"**Année:** {voiture.get('annee') or 'N/A'}")
                        st.write(f"**Puissance:** {voiture.get('puissance_kw')} kW" if voiture.get('puissance_kw') else "**Puissance:** N/A")
                    
                    with col3:
                        st.write(f"**Carburant:** {voiture.get('carburant') or 'N/A'}")
                        st.write(f"**Boîte:** {voiture.get('boite_de_vitesse') or 'N/A'}")
                        st.write(f"**Transmission:** {voiture.get('transmission') or 'N/A'}")
                        st.write(f"**Type:** {voiture.get('type_de_vehicule') or 'N/A'}")
                    
                    st.write(f"**Localisation:** {voiture.get('code_postal')} {voiture.get('ville')}" if voiture.get('ville') else "**Localisation:** N/A")
                    st.write(f"**Options:** {voiture.get('options') or 'N/A'}")
                    st.write(f"**Sièges:** {voiture.get('sieges') or 'N/A'} | **Portes:** {voiture.get('portes') or 'N/A'}")
                    if voiture.get('lien_fiche'):
                        st.write(f"**[🔗 Lien de l'annonce]({voiture.get('lien_fiche')})**")


# --- STREAMLIT UI ---
st.set_page_config(page_title="AutoScout24 Scraper", layout="wide")
st.title("🚗 Scraping AutoScout24")

# Créer les onglets
tab1, tab2 = st.tabs(["📥 Scraper", "🔍 Sélectionner"])

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
            if st.button("🚀 Lancer le scrapping", use_container_width=True):
                liste_voitures = run_scraping(int(nb_pages))
                if liste_voitures:
                    # --- PHASE 3: Sauvegarde JSON ---
                    st.subheader("💾 Phase 3: Sauvegarde du fichier JSON")
                    filepath = sauvegarder_json(liste_voitures, "annonces_autoscout24.json")
                    st.success(f"✅ {len(liste_voitures)} annonces sauvegardées")
                    
                    # --- PHASE 4: Nettoyage des données ---
                    st.subheader("🧹 Phase 4: Nettoyage des données")
                    try:
                        with st.spinner("Nettoyage en cours..."):
                            df_nettoyees = appliquer_cleaning("annonces_autoscout24.json")
                        sauvegarder_donnees_nettoyees(df_nettoyees, "voitures_nettoyees.json")
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
                        
                        # Aperçu des données nettoyées
                        st.subheader("📊 Aperçu des données nettoyées")
                        preview_data = df_nettoyees.select([
                            "marque", "modele", "moteur", "prix", "kilometrage", "annee"
                        ]).head(5).to_dicts()
                        st.json(preview_data)
                    except Exception as e:
                        st.error(f"❌ Erreur lors du nettoyage: {e}")

    elif mode == "📂 Charger un JSON":
        with col1:
            uploaded_file = st.file_uploader("Fichier JSON existant", type=["json"])
        with col2:
            st.write("")
            st.write("")
            if st.button("📂 Charger le JSON", use_container_width=True):
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

                        st.subheader("📊 Aperçu des données")
                        preview_data = [v.model_dump() for v in voitures[:5]]
                        st.json(preview_data)

    else:  # Mode "➕ Ajouter des données"
        with col1:
            nb_pages = st.number_input("Nombre de pages à scraper", min_value=1, max_value=50, value=5, step=1)
        with col2:
            st.write("")
            st.write("")
            if st.button("➕ Ajouter des données", use_container_width=True):
                # Vérifier que le fichier existant existe
                if not Path("annonces_autoscout24.json").exists():
                    st.error("❌ Le fichier annonces_autoscout24.json n'existe pas. Utilisez 'Redémarrer de zéro' d'abord.")
                else:
                    # Afficher le nombre d'annonces actuellement
                    voitures_actuelles = charger_voitures_depuis_fichier("annonces_autoscout24.json")
                    st.info(f"📊 Fichier actuel contient {len(voitures_actuelles)} annonces")
                    
                    # Scraper les nouvelles données
                    liste_voitures = run_scraping(int(nb_pages))
                    if liste_voitures:
                        # --- PHASE 3: Fusion et sauvegarde protégée ---
                        st.subheader("💾 Phase 3: Fusion et sauvegarde protégée")
                        filepath, ajoutees = fusionner_et_proteger_annonces(liste_voitures, "annonces_autoscout24.json")
                        
                        # Recharger pour vérifier
                        voitures_finales = charger_voitures_depuis_fichier("annonces_autoscout24.json")
                        st.success(f"✅ {ajoutees} nouvelles annonces ajoutées")
                        
                        # --- PHASE 4: Nettoyage des données ---
                        st.subheader("🧹 Phase 4: Nettoyage des données mises à jour")
                        try:
                            with st.spinner("Nettoyage en cours..."):
                                df_nettoyees = appliquer_cleaning("annonces_autoscout24.json")
                            sauvegarder_donnees_nettoyees(df_nettoyees, "voitures_nettoyees.json")
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
                            
                            # Affichage d'un aperçu des données nettoyées
                            st.subheader("📊 Aperçu des données nettoyées finales")
                            preview_data = df_nettoyees.select([
                                "marque", "modele", "moteur", "prix", "kilometrage", "annee"
                            ]).head(5).to_dicts()
                            st.json(preview_data)
                        except Exception as e:
                            st.error(f"❌ Erreur lors du nettoyage: {e}")

with tab2:
    afficher_selection_voitures()
