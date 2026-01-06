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


def run_scraping(nb_pages: int) -> Path | None:
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
    metrics = st.columns(3)
    
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
            
            # Mise à jour du statut
            status_details.info(f"🔄 {idx}/{total} annonces traitées ({int(progress * 100)}%)")
            
            # Mise à jour des métriques
            with metrics[0]:
                st.metric("Annonces traitées", idx)
            with metrics[1]:
                st.metric("Valides", len(liste_voitures))
            with metrics[2]:
                st.metric("Échouées", idx - len(liste_voitures))
            
            time.sleep(1)

        if not liste_voitures:
            st.error("❌ Aucune annonce valide trouvée")
            return None

        # --- PHASE 3: Sauvegarde JSON ---
        st.subheader("💾 Phase 3: Sauvegarde du fichier JSON")
        status_details.info("💾 Sauvegarde en cours...")
        filepath = sauvegarder_json(liste_voitures)
        
        # Notification finale avec succès
        st.success(f"✅ Succès! {len(liste_voitures)} annonces sauvegardées")
        st.balloons()
        
        # Affichage du chemin du fichier et des stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fichier généré", str(filepath.name))
        with col2:
            st.metric("Nombre d'annonces", len(liste_voitures))
        
        # Affichage d'un aperçu des données
        st.subheader("📊 Aperçu des données")
        preview_data = [v.model_dump() for v in liste_voitures[:5]]
        st.json(preview_data)
        
        return filepath

    finally:
        driver.quit()


# --- STREAMLIT UI ---
st.set_page_config(page_title="AutoScout24 Scraper", layout="wide")
st.title("🚗 Scraping AutoScout24")
st.write("Sélectionnez le nombre de pages à scraper et lancez le scraping")

mode = st.radio(
    "Source des données",
    ("Scraper AutoScout24", "Charger un JSON"),
    horizontal=True,
)

col1, col2 = st.columns([3, 1])

if mode == "Scraper AutoScout24":
    with col1:
        nb_pages = st.number_input("Nombre de pages à scraper", min_value=1, max_value=50, value=5, step=1)
    with col2:
        st.write("")
        st.write("")
        if st.button("🚀 Lancer le scrapping", use_container_width=True):
            run_scraping(int(nb_pages))
else:
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
