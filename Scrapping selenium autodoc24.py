# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "bs4",
#     "pydantic",
#     "requests",
#     "selenium",
#     "lxml",
# ]
# ///

import json
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
import re

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
        
        # Attendre que les articles se chargent
        print("⏳ Attente du chargement des annonces...")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "article"))
        )
        
        # Scroll pour charger plus d'annonces
        print("📜 Scroll de la page listage...")
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
        
        html_content = driver.page_source
        print(f"✅ Page listage récupérée ({len(html_content)} caractères)")
        return html_content
    
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
    
    # Prix
    prix_tag = soupe.find(class_=re.compile(r"CurrentPrice_price"))
    if prix_tag:
        voiture["prix"] = prix_tag.get_text(strip=True)
    
    # Marque/Modèle - chercher dans le titre principal
    titre_tag = soupe.find("h1")
    if titre_tag:
        voiture["marque"] = titre_tag.get_text(strip=True)
    
    # Localisation
    localisation_tag = soupe.find("span", attrs={"data-testid": "dealer-address"})
    if localisation_tag:
        voiture["localisation"] = localisation_tag.get_text(strip=True)
    
    # Tous les détails dans les spans et divs
    all_text = soupe.get_text()
    
    # Kilomètrage
    km_match = re.search(r'([\d\s]+)\s*km', all_text, re.IGNORECASE)
    if km_match:
        km_str = km_match.group(1).replace(" ", "").replace("\xa0", "")
        try:
            voiture["kilometrage"] = int(km_str)
        except ValueError:
            pass
    
    # Année
    annee_match = re.search(r'(\d{2})?/?(\d{4})', all_text)
    if annee_match:
        year_str = annee_match.group(2)
        try:
            voiture["annee"] = int(year_str)
        except ValueError:
            pass
    
    # Carburant
    carburants = ["essence", "diesel", "électrique", "hybride", "gaz"]
    for carb in carburants:
        if carb.lower() in all_text.lower():
            voiture["carburant"] = carb.capitalize()
            break
    
    # Boîte de vitesse
    if "manuelle" in all_text.lower():
        voiture["boite_de_vitesse"] = "Manuelle"
    elif "automatique" in all_text.lower():
        voiture["boite_de_vitesse"] = "Automatique"
    
    # Transmission
    if "4x4" in all_text.upper():
        voiture["transmission"] = "4x4"
    elif "avant" in all_text.lower():
        voiture["transmission"] = "Avant"
    elif "arrière" in all_text.lower():
        voiture["transmission"] = "Arrière"
    
    # Puissance
    puissance_match = re.search(r'([\d\s]+)\s*kW\s*\(?\s*([\d\s]+)\s*CH\)?', all_text, re.IGNORECASE)
    if puissance_match:
        voiture["puissance"] = f"{puissance_match.group(1).strip()} kW ({puissance_match.group(2).strip()} CH)"
    
    return voiture

# --- 5. Sauvegarder en JSON ---
def sauvegarder_json(voitures: list[Voiture], filename: str = "annonces_autoscout24.json"):
    """Sauvegarde la liste des voitures dans un fichier JSON."""
    data = [v.model_dump() for v in voitures]
    
    filepath = Path(filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"💾 {len(voitures)} annonces sauvegardées dans {filepath}")
    return filepath

# --- 6. Main ---
def main():
    url_listage = "https://www.autoscout24.fr/lst?atype=C&cy=D%2CA%2CB%2CE%2CF%2CI%2CL%2CNL&damaged_listing=exclude&desc=0&powertype=kw&search_id=k9p7elkop&sort=standard&source=listpage_pagination&ustate=N%2CU"
    
    print("🚀 Démarrage du scraping AutoScout24")
    print("=" * 60)
    
    # Étape 1 : Récupérer la page listage
    html_listage = recupere_page_listage(url_listage)
    
    # Étape 2 : Extraire les URLs
    urls_annonces = extraire_urls_annonces(html_listage)
    
    if not urls_annonces:
        print("⚠️  Aucune annonce trouvée")
        return
    
    # Étape 3 : Créer un driver pour consulter chaque annonce
    print(f"\n📖 Consultation de {len(urls_annonces)} annonces...")
    print("=" * 60)
    
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
                    print(f"✅ Annonce ajoutée: {voiture.marque} - {voiture.prix}")
                except Exception as e:
                    print(f"⚠️  Erreur validation: {e}")
            
            # Petit délai entre les requêtes pour ne pas surcharger le serveur
            time.sleep(1)
        
        # Étape 4 : Sauvegarder
        print("\n" + "=" * 60)
        if liste_voitures:
            sauvegarder_json(liste_voitures)
            print(f"✨ Scraping terminé : {len(liste_voitures)} annonces")
        else:
            print("⚠️  Aucune annonce valide trouvée")
    
    finally:
        driver.quit()
        print("=" * 60)

if __name__ == "__main__":
    main()