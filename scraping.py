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
fichier = list(dossier.glob("*"))[3]

code_html = fichier.read_text(encoding="utf-8")
soupe = BS(code_html, features="html.parser")


def extraire_dl(dl_tag):
    data = {}
    if not dl_tag:
        return data
    dts = dl_tag.find_all("dt")
    dds = dl_tag.find_all("dd")
    for dt, dd in zip(dts, dds):
        label = dt.get_text(strip=True)
        value = dd.get_text(strip=True)
        
        value = value.replace('\u202f', '').replace('\xa0', '').strip()

        if "Kilométrage" in label:
            km_match = re.search(r'\d+', value.replace(' ', ''))
            if km_match:
                km_number = int(km_match.group(0))
                value = f"{km_number:,}".replace(',', ' ') + " km"

        data[label] = value
    return data

# Extraire les sections
def extraire_section(soupe, section_id):
    section = soupe.find("section", id=section_id)
    if not section:
        return {}
    dl_tag = section.find("dl")
    return extraire_dl(dl_tag)

caracteristiques = extraire_section(soupe, "basic-details-section")
historique = extraire_section(soupe, "listing-history-section")
technique = extraire_section(soupe, "technical-details-section")
environnement = extraire_section(soupe, "environment-details-section")
esthetique = extraire_section(soupe, "color-section")

# Regrouper toutes les sections
data = {
    "donnees_base": caracteristiques,
    "historique": historique,
    "technique": technique,
    "environnement": environnement,
    "esthetique": esthetique
}

# Sauvegarder en JSON
with open("caracteristiques.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# Affichage rapide
print(json.dumps(data, ensure_ascii=False, indent=2))
