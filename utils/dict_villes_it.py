import requests
import json
from pathlib import Path

def generer_json_villes_italie():
    sparql_url = "https://query.wikidata.org/sparql"
    
    # Requête corrigée (Syntaxe SPARQL propre)
    query = """
    SELECT DISTINCT ?nameIT ?nameFR WHERE {
        ?city wdt:P17 wd:Q38 . # Pays : Italie
        
        { 
            ?city wdt:P31/wdt:P279* wd:Q747066 . # Commune
        }
        UNION
        { 
            ?city wdt:P31/wdt:P279* wd:Q486972 . # Localité / Village
        }
        
        ?city rdfs:label ?nameIT .
        FILTER(LANG(?nameIT) = "it")
        
        OPTIONAL {
            ?city rdfs:label ?nameFR .
            FILTER(LANG(?nameFR) = "fr")
        }
    }
    """

    params = {
        "query": query,
        "format": "json"
    }
    
    # Définition du chemin de sortie
    json_path = Path("data/references/ville_it.json")
    json_path.parent.mkdir(parents=True, exist_ok=True) # Crée le dossier s'il n'existe pas

    try:
        print("📡 Connexion à Wikidata (cela peut prendre 30-60 secondes)...")
        # Augmentation du timeout à 60 car la liste est très longue
        response = requests.get(sparql_url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        villes_set = set()
        results = data.get('results', {}).get('bindings', [])
        
        for binding in results:
            if 'nameIT' in binding:
                villes_set.add(binding['nameIT']['value'])
            if 'nameFR' in binding:
                villes_set.add(binding['nameFR']['value'])

        if not villes_set:
            print("⚠️ Wikidata a renvoyé 0 résultat. La requête a peut-être échoué silencieusement.")
            return

        villes_liste = sorted(list(villes_set))

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(villes_liste, f, ensure_ascii=False, indent=4)
        
        print(f"✅ Succès ! Fichier '{json_path}' créé avec {len(villes_liste)} noms de lieux.")

    except requests.exceptions.HTTPError as e:
        print(f"❌ Erreur HTTP : {e}")
    except Exception as e:
        print(f"❌ Erreur inattendue : {e}")

if __name__ == "__main__":
    generer_json_villes_italie()