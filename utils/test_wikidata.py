import requests

# Test direct de la requête Wikidata pour Volkswagen
sparql_url = "https://query.wikidata.org/sparql"

# Requête améliorée avec Q20165 pour les générations
query = """
SELECT DISTINCT ?modelLabel WHERE {
  ?model wdt:P176 wd:Q246 .
  
  # Capturer tous les types de véhicules
  {
    # Modèles automobiles (Q3231690)
    ?model wdt:P31 wd:Q3231690 .
  }
  UNION
  {
    # Générations automobiles (Q20165) - CRUCIAL pour Golf I, II, etc.
    ?model wdt:P31 wd:Q20165 .
  }
  UNION
  {
    # Véhicules motorisés en général
    ?model wdt:P31/wdt:P279* wd:Q1420 .
  }
  UNION
  {
    # Modèles de concept car
    ?model wdt:P31 wd:Q850270 .
  }
  
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en,fr,de" . }
}
ORDER BY ?modelLabel
LIMIT 1000
"""

params = {
    "query": query,
    "format": "json"
}

print("🔍 Interrogation Wikidata pour Volkswagen...")
response = requests.get(sparql_url, params=params, timeout=30)
data = response.json()

modeles = []
if "results" in data and "bindings" in data["results"]:
    for binding in data["results"]["bindings"]:
        if "modelLabel" in binding:
            model_name = binding["modelLabel"]["value"]
            modeles.append(model_name)

print(f"\n✅ {len(modeles)} modèles trouvés")
print("\nModèles contenant 'Golf':")
golf_models = [m for m in modeles if 'Golf' in m or 'golf' in m]
for g in sorted(golf_models):
    print(f"  - {g}")
