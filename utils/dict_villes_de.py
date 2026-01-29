import requests
import json
import time
from pathlib import Path

def completer_json_allemagne():
    json_path = Path("data/references/ville_de.json")
    villes_set = set()

    # 1. On charge l'existant pour ne rien perdre
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            villes_set = set(json.load(f))
        print(f"📂 Fichier existant chargé : {len(villes_set)} villes.")
    else:
        print("⚠️ Aucun fichier existant trouvé. Un nouveau sera créé.")

    # 2. On ne cible QUE les lettres qui ont échoué
    lettres_manquantes = "HLRY"
    sparql_url = "https://query.wikidata.org/sparql"
    headers = {'User-Agent': 'CarDataBot/1.2 (contact: admin@example.com)'}

    for lettre in lettres_manquantes:
        success = False
        retries = 3
        while not success and retries > 0:
            try:
                print(f"🔤 Récupération de la lettre {lettre}...", end=" ", flush=True)
                query = f"""
                SELECT DISTINCT ?nameDE ?nameFR WHERE {{
                    ?city wdt:P17 wd:Q183 .
                    ?city wdt:P31/wdt:P279* wd:Q486972 . 
                    ?city rdfs:label ?nameDE .
                    FILTER(LANG(?nameDE) = "de")
                    FILTER(STRSTARTS(STR(?nameDE), "{lettre}"))
                    OPTIONAL {{ ?city rdfs:label ?nameFR . FILTER(LANG(?nameFR) = "fr") }}
                }}
                """
                response = requests.get(sparql_url, params={"query": query, "format": "json"}, headers=headers, timeout=180)
                response.raise_for_status()
                
                bindings = response.json().get('results', {}).get('bindings', [])
                for b in bindings:
                    villes_set.add(b['nameDE']['value'])
                    if 'nameFR' in b:
                        villes_set.add(b['nameFR']['value'])
                
                print(f"✅ {len(bindings)} ajoutés.")
                success = True
                # Petit repos pour éviter le 429 (Too Many Requests)
                time.sleep(2) 
                
            except Exception as e:
                retries -= 1
                print(f"⚠️ Erreur ({e}). Retries: {retries}")
                time.sleep(10)

    # 3. Sauvegarde finale fusionnée
    villes_liste = sorted(list(villes_set))
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(villes_liste, f, ensure_ascii=False, indent=4)
    
    print(f"\n✨ Fusion terminée ! Le fichier contient maintenant {len(villes_liste)} lieux.")

if __name__ == "__main__":
    completer_json_allemagne()