# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "bs4",
#     "pydantic",
#     "requests",
#     "selenium",
#     "lxml",
#     "fake-useragent",
# ]
# ///

import json
import re
import os
import sys
from bs4 import BeautifulSoup as BS
from pydantic import BaseModel
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.safari.service import Service
from selenium.webdriver.safari.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from fake_useragent import UserAgent
from selenium.common.exceptions import TimeoutException

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
    """Ouvre la page listage AutoScout24, scrolle, et retourne le HTML."""
    print(f"🌐 Ouverture de la page listage : {url}")
    
    # Détecter l'environnement: CI/CD (GitHub Actions) ou OS local
    is_ci = os.getenv('CI') == 'true' or os.getenv('GITHUB_ACTIONS') == 'true'
    is_linux = sys.platform.startswith('linux')
    is_windows = sys.platform.startswith('win')
    
    # Utiliser Chrome si: CI/CD, Linux, Windows, ou chemin contient "matde"
    if is_ci or is_linux or is_windows or "matde" in str(Path(".").resolve()):
        ua = UserAgent()
        
        options = ChromeOptions()
        options.add_argument(f"user-agent={ua.random}")  # User-Agent aléatoire
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(service=ChromeService(), options=options)
    else:
        # macOS local sans "matde" = Safari
        driver = webdriver.Safari(service=Service())
    
    try:
        driver.get(url)

        print("⏳ Attente du chargement des annonces...")
        try:
            # Attend plus longtemps pour des pages lentes, mais ne plante pas si timeout
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "article"))
            )
        except TimeoutException:
            print("⏱️ Délai dépassé pour le chargement. On continue avec le HTML disponible.")

        # Scroll pour charger plus d'annonces (même si l'attente a expiré)
        print("📜 Scroll de la page listage...")
        try:
            last_height = driver.execute_script("return document.body.scrollHeight")
        except Exception:
            last_height = 0

        scroll_attempts = 0
        max_scrolls = 5

        while scroll_attempts < max_scrolls:
            try:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                new_height = driver.execute_script("return document.body.scrollHeight")
            except Exception:
                break

            if new_height == last_height:
                break
            last_height = new_height
            scroll_attempts += 1

        html_content = driver.page_source or ""
        print(f"✅ Page listage récupérée ({len(html_content)} caractères)")
        return html_content

    except Exception as e:
        print(f"❌ Erreur lors de l'ouverture de la page listage: {e}")
        return ""
    finally:
        driver.quit()

# --- 2. Extraire les URLs de la page listage ---
def extraire_urls_annonces(html_content: str) -> list[str]:
    """Extrait les URLs de toutes les annonces de la page listage."""
    soupe = BS(html_content, "lxml")
    articles = soupe.find_all("article")
    
    urls = []
    for art in articles:
        # Chercher le lien href qui commence par /offres/
        lien_tag = art.find("a", href=re.compile(r"^/offres/"))
        if lien_tag and lien_tag.get("href"):
            url_complete = "https://www.autoscout24.fr" + lien_tag["href"]
            urls.append(url_complete)
    
    print(f"🔗 {len(urls)} URLs d'annonces extraites")
    return urls

# --- 3. Récupérer une page d'annonce ---
def recupere_page_annonce(driver, url: str) -> str:
    """Récupère le HTML d'une page d'annonce."""
    try:
        driver.get(url)
        
        # Attendre le chargement
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        time.sleep(1)  # Petit délai pour s'assurer que tout est chargé
        return driver.page_source
    except Exception as e:
        print(f"❌ Erreur lors de la récupération de {url}: {e}")
        return None

# --- 4. Extraire les détails d'une page d'annonce ---
def extraire_details_annonce(html_content: str, url: str) -> dict:
    """Extrait tous les détails d'une page d'annonce."""
    voiture = {"lien_fiche": url}
    
    if not html_content:
        return voiture
    
    soupe = BS(html_content, "lxml")
    
    # ===== EXTRACTION PRIORITAIRE: JSON-LD Schema =====
    # AutoScout24 inclut les données structurées en JSON-LD
    json_ld_scripts = soupe.find_all("script", type="application/ld+json")
    json_data = None

    # Cherche le bloc JSON-LD qui contient l'offre (le premier <script> est souvent le fil d'Ariane)
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
            # Prix
            if "offers" in json_data and "price" in json_data["offers"]:
                price = json_data["offers"]["price"]
                currency = json_data["offers"].get("priceCurrency", "EUR")
                voiture["prix"] = f"€ {price:,.0f}".replace(",", " ")
            
            # Location (depuis offeredBy)
            if "offers" in json_data and "offeredBy" in json_data["offers"]:
                offered_by = json_data["offers"]["offeredBy"]
                if "address" in offered_by:
                    address = offered_by["address"]
                    city = address.get("addressLocality", "")
                    postal = address.get("postalCode", "")
                    if city or postal:
                        voiture["localisation"] = f"{postal} {city}".strip()

            # Location alternative: certains schémas utilisent seller ou availableAtOrFrom
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
                address = json_data["availableAtOrFrom"].get("address", {})
                city = address.get("addressLocality", "")
                postal = address.get("postalCode", "")
                if city or postal:
                    voiture["localisation"] = f"{postal} {city}".strip()
            
            # Kilomètrage
            if "itemOffered" in json_data and "mileageFromOdometer" in json_data["itemOffered"]:
                km_data = json_data["itemOffered"]["mileageFromOdometer"]
                if "value" in km_data:
                    voiture["kilometrage"] = int(km_data["value"])
            
            # Année
            if "itemOffered" in json_data and "productionDate" in json_data["itemOffered"]:
                prod_date = json_data["itemOffered"]["productionDate"]
                if prod_date:
                    year_match = re.search(r'(\d{4})', prod_date)
                    if year_match:
                        voiture["annee"] = int(year_match.group(1))
            
            # Transmission
            if "itemOffered" in json_data and "driveWheelConfiguration" in json_data["itemOffered"]:
                voiture["transmission"] = json_data["itemOffered"]["driveWheelConfiguration"]
            
            # Boîte de vitesse
            if "itemOffered" in json_data and "vehicleTransmission" in json_data["itemOffered"]:
                transmission_text = json_data["itemOffered"]["vehicleTransmission"]
                if "automatique" in transmission_text.lower():
                    voiture["boite_de_vitesse"] = "Automatique"
                elif "manuelle" in transmission_text.lower():
                    voiture["boite_de_vitesse"] = "Manuelle"
                else:
                    voiture["boite_de_vitesse"] = transmission_text
            
            # Nombre de portes
            if "itemOffered" in json_data and "numberOfDoors" in json_data["itemOffered"]:
                voiture["portes"] = int(json_data["itemOffered"]["numberOfDoors"])
            
            # Type de carrosserie
            if "itemOffered" in json_data and "bodyType" in json_data["itemOffered"]:
                voiture["type_de_vehicule"] = json_data["itemOffered"]["bodyType"]
            
            # Puissance
            if "itemOffered" in json_data and "vehicleEngine" in json_data["itemOffered"]:
                engines = json_data["itemOffered"]["vehicleEngine"]
                if engines and len(engines) > 0:
                    engine = engines[0]
                    if "enginePower" in engine:
                        powers = engine["enginePower"]
                        kw = None
                        hp = None
                        for power in powers:
                            if power.get("unitCode") == "KWT":
                                kw = power.get("value")
                            elif power.get("unitCode") == "BHP":
                                hp = power.get("value")
                        if kw and hp:
                            voiture["puissance"] = f"{kw} kW ({hp} CH)"
        except Exception as e:
            print(f"⚠️  Erreur lors du parsing JSON-LD: {e}")
    
    # ===== EXTRACTION HTML =====
    
    # Marque/Modèle - chercher dans le titre principal (L'ANCIENNE MÉTHODE QUI MARCHAIT)
    titre_tag = soupe.find("h1")
    if titre_tag:
        voiture["marque"] = titre_tag.get_text(strip=True)
    
    # Prix (si pas trouvé dans JSON)
    if not voiture.get("prix"):
        prix_tag = soupe.find("span", class_=re.compile(r"PriceInfo_price"))
        if prix_tag:
            voiture["prix"] = prix_tag.get_text(strip=True)
    
    # Localisation (si pas trouvée dans JSON)
    if not voiture.get("localisation"):
        # Chercher près de l'icône de localisation
        location_links = soupe.find_all("a", href=re.compile(r"dealer.*location", re.I))
        for link in location_links:
            text = link.get_text(strip=True)
            if text and len(text) > 2:
                voiture["localisation"] = text
                break
    
    # Tous les détails dans les spans et divs
    all_text = soupe.get_text()
    
    # Kilomètrage (si pas trouvé dans JSON)
    if not voiture.get("kilometrage"):
        km_match = re.search(r'([\d\s]+)\s*km', all_text, re.IGNORECASE)
        if km_match:
            km_str = km_match.group(1).replace(" ", "").replace("\xa0", "").replace("\u202f", "")
            try:
                voiture["kilometrage"] = int(km_str)
            except ValueError:
                pass
    
    # Année (si pas trouvée dans JSON)
    if not voiture.get("annee"):
        annee_match = re.search(r'(\d{2})?/?(\d{4})', all_text)
        if annee_match:
            year_str = annee_match.group(2)
            try:
                year = int(year_str)
                if 1950 <= year <= 2030:
                    voiture["annee"] = year
            except ValueError:
                pass
    
    # Carburant
    if not voiture.get("carburant"):
        carburants = ["Essence", "Diesel", "Électrique", "Hybride", "Gaz", "Hybrid"]
        for carb in carburants:
            if carb.lower() in all_text.lower():
                voiture["carburant"] = carb
                break
    
    # Boîte de vitesse (si pas trouvée dans JSON)
    if not voiture.get("boite_de_vitesse"):
        if "manuelle" in all_text.lower():
            voiture["boite_de_vitesse"] = "Manuelle"
        elif "automatique" in all_text.lower():
            voiture["boite_de_vitesse"] = "Automatique"
    
    # Transmission (si pas trouvée dans JSON)
    if not voiture.get("transmission"):
        if "4x4" in all_text.upper():
            voiture["transmission"] = "4x4"
        elif "avant" in all_text.lower():
            voiture["transmission"] = "Avant"
        elif "arrière" in all_text.lower():
            voiture["transmission"] = "Arrière"
    
    # Puissance (si pas trouvée dans JSON)
    if not voiture.get("puissance"):
        puissance_match = re.search(r'([\d\s]+)\s*kW\s*\(?\s*([\d\s]+)\s*CH\)?', all_text, re.IGNORECASE)
        if puissance_match:
            voiture["puissance"] = f"{puissance_match.group(1).strip()} kW ({puissance_match.group(2).strip()} CH)"
    
    # Chercher les détails dans les DataGrid (tables de caractéristiques)
    data_grids = soupe.find_all("dl", class_=re.compile(r"DataGrid"))
    for grid in data_grids:
        dt_tags = grid.find_all("dt")
        dd_tags = grid.find_all("dd")
        
        for dt, dd in zip(dt_tags, dd_tags):
            label = dt.get_text(strip=True).lower()
            value = dd.get_text(strip=True)
            
            # Portes (si pas trouvé dans JSON)
            if "porte" in label and not voiture.get("portes"):
                portes_match = re.search(r'(\d+)', value)
                if portes_match:
                    voiture["portes"] = int(portes_match.group(1))
            
            # Sièges / Places
            if ("siège" in label or "place" in label) and not voiture.get("sieges"):
                sieges_match = re.search(r'(\d+)', value)
                if sieges_match:
                    voiture["sieges"] = int(sieges_match.group(1))
            
            # Carburant (si pas trouvé)
            if "carburant" in label and not voiture.get("carburant"):
                voiture["carburant"] = value
            
            # Transmission (si pas trouvée)
            if "transmission" in label and not voiture.get("transmission"):
                voiture["transmission"] = value
            
            # Boîte de vitesse (si pas trouvée)
            if "boîte" in label and not voiture.get("boite_de_vitesse"):
                voiture["boite_de_vitesse"] = value
            
            # Type de véhicule / Carrosserie (si pas trouvé)
            if "carrosserie" in label and not voiture.get("type_de_vehicule"):
                voiture["type_de_vehicule"] = value
    
    return voiture

# --- 5. Charger les voitures depuis un fichier JSON ---
def charger_voitures_depuis_fichier(filename: str = "data/raw/annonces_autoscout24.json") -> list[Voiture]:
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

# --- 6. Fusionner et protéger les annonces ---
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
    
    print(f"💾 {len(voitures_finales)} annonces totales ({len(existantes)} existantes + {len(voitures_a_ajouter)} nouvelles) dans {filepath}")
    return filepath, len(voitures_a_ajouter)

# --- 7. Main ---
def main():
    # URL de base (sans page, on va l'ajouter)
    url_base = "https://www.autoscout24.fr/lst?atype=C&cy=D%2CA%2CB%2CE%2CF%2CI%2CL%2CNL&damaged_listing=exclude&desc=0&powertype=kw&search_id=k9p7elkop&sort=standard&source=listpage_pagination&ustate=N%2CU"
    
    print("🚀 Démarrage du scraping AutoScout24")
    print("=" * 60)
    
    # Étape 1 : Récupérer les URLs de plusieurs pages (1 à 5)
    print("\n📋 Récupération des pages de listage...")
    urls_annonces_toutes = {}  # Utiliser un dict pour éviter les doublons
    
    for page in range(1, 51):  # Pages 1 à 50
        url_page = f"{url_base}&page={page}"
        print(f"\n--- Page {page}/50 ---")
        
        html_listage = recupere_page_listage(url_page)
        urls_page = extraire_urls_annonces(html_listage)
        
        # Ajouter au dictionnaire (les doublons seront automatiquement ignorés)
        for url in urls_page:
            urls_annonces_toutes[url] = True
        
        print(f"📊 Total URLs uniques jusqu'ici : {len(urls_annonces_toutes)}")
    
    # Convertir en liste
    urls_annonces = list(urls_annonces_toutes.keys())
    
    print(f"\n✅ Total final : {len(urls_annonces)} annonces uniques à scraper")
    print("=" * 60)
    
    if not urls_annonces:
        print("⚠️  Aucune annonce trouvée")
        return
    
    # Étape 2 : Créer un driver pour consulter chaque annonce
    print(f"\n📖 Consultation de {len(urls_annonces)} annonces...")
    print("=" * 60)
    
    # Détecter l'environnement: CI/CD (GitHub Actions) ou OS local
    is_ci = os.getenv('CI') == 'true' or os.getenv('GITHUB_ACTIONS') == 'true'
    is_linux = sys.platform.startswith('linux')
    is_windows = sys.platform.startswith('win')
    
    # Utiliser Chrome si: CI/CD, Linux, Windows, ou chemin contient "matde"
    if is_ci or is_linux or is_windows or "matde" in str(Path(".").resolve()):
        options = ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(service=ChromeService(), options=options)
    else:
        # macOS local sans "matde" = Safari
        driver = webdriver.Safari(service=Service())
    
    try:
        liste_voitures = []
        
        for idx, url in enumerate(urls_annonces, 1):
            print(f"\n[{idx}/{len(urls_annonces)}] Traitement de {url}")
            
            # Récupérer la page d'annonce
            html_annonce = recupere_page_annonce(driver, url)
            
            if html_annonce:
                # Extraire les détails
                details = extraire_details_annonce(html_annonce, url)
                
                try:
                    voiture = Voiture(**details)
                    liste_voitures.append(voiture)
                    
                    # Afficher un résumé
                    info_parts = []
                    if voiture.marque:
                        info_parts.append(voiture.marque)
                    if voiture.prix:
                        info_parts.append(voiture.prix)
                    if voiture.portes:
                        info_parts.append(f"{voiture.portes} portes")
                    if voiture.localisation:
                        info_parts.append(voiture.localisation)
                    
                    print(f"✅ Annonce ajoutée: {' - '.join(info_parts)}")
                except Exception as e:
                    print(f"⚠️  Erreur validation: {e}")
            
            # Petit délai entre les requêtes pour ne pas surcharger le serveur
            time.sleep(1)
        
        # Étape 3 : Fusionner avec les données existantes
        print("\n" + "=" * 60)
        if liste_voitures:
            filepath, nb_nouvelles = fusionner_et_proteger_annonces(liste_voitures)
            print(f"✨ Scraping terminé : {nb_nouvelles} nouvelles annonces ajoutées")
            if nb_nouvelles == 0:
                print("ℹ️  Toutes les annonces scrapées existaient déjà")
        else:
            print("⚠️  Aucune annonce valide trouvée")
    
    finally:
        driver.quit()
        print("=" * 60)

if __name__ == "__main__":
    main()