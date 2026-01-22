import json
import requests
import time

SAVE_EVERY = 100

with open("data/raw/annonces_autoscout24.json", "r", encoding="utf-8") as f:
    data = json.load(f)

session = requests.Session()

for i, annonce in enumerate(data, 1):
    if "annonce_disponible" in annonce:
        continue  # déjà traitée

    url = annonce.get("lien_fiche")
    if not url:
        annonce["annonce_disponible"] = 0
        continue

    try:
        response = session.get(url, timeout=10)
        indispo = "Ce véhicule n'est malheureusement plus disponible." in response.text
        annonce["annonce_disponible"] = 0 if indispo else 1
    except requests.RequestException:
        annonce["annonce_disponible"] = 0

    if i % SAVE_EVERY == 0:
        with open("data/raw/annonces_autoscout24.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Sauvegarde intermédiaire à {i}")

    time.sleep(0.2)

# sauvegarde finale
with open("data/raw/annonces_autoscout24.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
