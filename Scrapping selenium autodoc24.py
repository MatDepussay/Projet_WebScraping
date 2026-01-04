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
from datetime import datetime
import re

# --- Modèles Pydantic ---
class PageData(BaseModel):
    url: str
    date_recup: datetime
    html_content: str

class Voiture(BaseModel):
    lien_fiche: str | None = None
    prix: str | None = None
    marque: str | None = None
    localisation: str | None = None
    kilometrage: int | None = None
    annee: int | None = None
    carburant: str | None = None
    boite_de_vitesse: str | None = None
    transmission: str | None = None  # Avant / Arrière / 4x4
    puissance: str | None = None
    type_de_vehicule: str | None = None  # Occasion / Neuf
    sieges: int | None = None
    portes: int | None = None

# --- 1. Ouvrir la page avec Selenium ---
def recupere_page_selenium(url: str) -> str:
    """Ouvre la page AutoScout24, scrolle pour charger les annonces, et retourne le HTML complet."""
    print(f"🌐 Ouverture de : {url}")
    dir_path = Path(".").resolve()

    if "matde" in str(dir_path):
        options = ChromeOptions()
        options.add_argument("--headless")  # Mode sans interface graphique (optionnel)
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
        print("📜 Scroll de la page...")
        last_height = driver.execute_script("return document.body.scrollHeight")
        scroll_attempts = 0
        max_scrolls = 3  # Limiter à 3 scrolls pour ne pas trop attendre
        
        while scroll_attempts < max_scrolls:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            scroll_attempts += 1
        
        html_content = driver.page_source
        print(f"✅ Page récupérée ({len(html_content)} caractères)")
        return html_content
    
    finally:
        driver.quit()

# --- 2. Parser le HTML et extraire les annonces ---
def extraire_annonces(html_content: str) -> list[Voiture]:
    """Parse le HTML et extrait toutes les annonces de voitures."""
    soupe = BS(html_content, "lxml")
    articles = soupe.find_all("article")
    
    print(f"🔍 {len(articles)} annonces trouvées")
    
    liste_voitures = []
    
    for art in articles:
        voiture = {}
        
        # Lien vers la fiche
        lien_tag = art.find("a", href=True)
        if lien_tag and lien_tag.get("href", "").startswith("/offres/"):
            voiture["lien_fiche"] = "https://www.autoscout24.fr" + lien_tag["href"]
        
        # Prix
        prix_tag = art.find(class_=lambda x: x and "CurrentPrice_price" in x)
        if prix_tag:
            voiture["prix"] = prix_tag.get_text(strip=True)
        
        # Marque / modèle
        marque_tag = art.find(class_=lambda x: x and "ListItemTitle_title" in x)
        if marque_tag:
            voiture["marque"] = marque_tag.get_text(strip=True)
        
        # Localisation
        localisation_tag = art.find("span", attrs={"data-testid": "dealer-address"})
        if localisation_tag:
            voiture["localisation"] = localisation_tag.get_text(strip=True)
        
        # Kilomètrage - chercher dans les détails du véhicule
        details = art.find_all(class_=lambda x: x and "ListItemPill" in x)
        for detail in details:
            text = detail.get_text(strip=True)
            # Kilomètrage (ex: "137 000 km")
            if "km" in text.lower():
                km_match = re.search(r'([\d\s]+)\s*km', text, re.IGNORECASE)
                if km_match:
                    km_str = km_match.group(1).replace(" ", "").replace("\xa0", "")
                    try:
                        voiture["kilometrage"] = int(km_str)
                    except:
                        pass
            
            # Année (ex: "10/2017")
            if "/" in text and len(text) <= 10:
                annee_match = re.search(r'/(\d{4})', text)
                if annee_match:
                    try:
                        voiture["annee"] = int(annee_match.group(1))
                    except:
                        pass
            
            # Carburant
            carburants = ["essence", "diesel", "électrique", "hybride", "gaz"]
            for carb in carburants:
                if carb.lower() in text.lower():
                    voiture["carburant"] = text
                    break
            
            # Boîte de vitesse
            if "manuelle" in text.lower() or "automatique" in text.lower():
                voiture["boite_de_vitesse"] = text
            
            # Puissance (ex: "110 kW (150 CH)")
            if "kw" in text.lower() or "ch" in text.lower():
                voiture["puissance"] = text
        
        # Ajouter uniquement si on a au moins un lien ou une marque
        if voiture.get("lien_fiche") or voiture.get("marque"):
            try:
                liste_voitures.append(Voiture(**voiture))
            except Exception as e:
                print(f"⚠️  Erreur validation Pydantic : {e}")
                continue
    
    return liste_voitures

# --- 3. Sauvegarder en JSON ---
def sauvegarder_json(voitures: list[Voiture], filename: str = "annonces_autoscout24.json"):
    """Sauvegarde la liste des voitures dans un fichier JSON."""
    data = [v.model_dump() for v in voitures]
    
    filepath = Path(filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"💾 {len(voitures)} annonces sauvegardées dans {filepath}")
    return filepath

# --- 4. Main ---
def main():
    url = "https://www.autoscout24.fr/lst?atype=C&cy=D%2CA%2CB%2CE%2CF%2CI%2CL%2CNL&damaged_listing=exclude&desc=0&powertype=kw&search_id=k9p7elkop&sort=standard&source=listpage_pagination&ustate=N%2CU"
    
    print("🚀 Démarrage du scraping AutoScout24")
    print("=" * 50)
    
    # Étape 1 : Récupérer le HTML
    html_content = recupere_page_selenium(url)
    
    # Étape 2 : Parser et extraire les annonces
    voitures = extraire_annonces(html_content)
    
    # Étape 3 : Sauvegarder
    if voitures:
        sauvegarder_json(voitures)
        print(f"\n✨ Scraping terminé avec succès : {len(voitures)} annonces")
    else:
        print("\n⚠️  Aucune annonce trouvée")
    
    print("=" * 50)

if __name__ == "__main__":
    main()