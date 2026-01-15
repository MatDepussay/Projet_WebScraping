import json

# Charger le fichier fusionné
with open('data/references/vehicle_models_merged.json', encoding='utf-8') as f:
    data = json.load(f)

# Trouver Volkswagen
vw = [x for x in data if x['Make'] == 'Volkswagen'][0]

# Filtrer les Golf
golf = [m for m in vw['Models'] if 'Golf' in m]

print(f"Modèles Golf trouvés: {len(golf)}")
print("\nPremiers 30 modèles Golf:")
for g in sorted(golf)[:30]:
    print(f"  - {g}")
