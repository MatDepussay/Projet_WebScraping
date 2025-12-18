# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "bs4",
#     "lxml",
#     "pydantic",
# ]
# ///

import re
from bs4 import BeautifulSoup as BS
from pathlib import Path
import json

# Charger le fichier HTML
repertoire = Path(".").resolve()
dossier = list(repertoire.glob("*"))[18]
fichier = list(dossier.glob("*"))[7]

code_html = fichier.read_text(encoding="utf-8")
soupe = BS(code_html, features="html.parser")

# Trouver tous les articles (ou containers des annonces)
articles = soupe.find_all("article")  # ou remplacer par le tag spécifique, parfois <div class="listing-item">

liste_annonces = []

liste_annonces = []

for art in articles:
    annonce = {}

    # Lien vers la fiche
    lien_tag = art.find("a", href=True)
    if lien_tag:
        annonce["lien_fiche"] = lien_tag["href"]

    # Prix
    prix_tag = art.find(class_=lambda x: x and "Price" in x)
    if prix_tag:
        texte_prix = prix_tag.get_text(strip=True)
        # Extraire uniquement le montant (ex: "€ 4 480")
        match = re.match(r"(€\s*[\d\s]+)", texte_prix)
        if match:
            annonce["prix"] = match.group(1).strip()
        else:
            annonce["prix"] = texte_prix

    # Marque / modèle
    marque_tag = art.find(class_=lambda x: x and "MakeModel" in x)
    if marque_tag:
        annonce["marque"] = marque_tag.get_text(strip=True)

    # Localisation
    localisation_tag = art.find(class_=lambda x: x and "Location" in x)
    if localisation_tag:
        annonce["localisation"] = localisation_tag.get_text(strip=True)

    # Ajouter si on a au moins un lien
    if "lien_fiche" in annonce:
        liste_annonces.append(annonce)

# Export JSON
with open("liste_annonces.json", "w", encoding="utf-8") as f:
    json.dump(liste_annonces, f, ensure_ascii=False, indent=2)

# Affichage propre
print(json.dumps(liste_annonces, ensure_ascii=False, indent=2))


def main() -> None:
    print("Hello from scraping2point0.py!")


if __name__ == "__main__":
    main()
