# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars",
#     "xlsxwriter",
# ]
# ///

import polars as pl
from pathlib import Path

# =========================
# 0. CHARGEMENT DES MODÈLES DE VOITURES
# =========================
def charger_modeles_csv() -> dict:
    """
    Charge le CSV des modèles de voitures et retourne un dictionnaire:
    {marque: [liste des modèles]}
    """
    csv_path = Path("Cars Datasets 2025.csv")
    if not csv_path.exists():
        return {}
    
    try:
        df_cars = pl.read_csv(csv_path, encoding="latin1", ignore_errors=True)
        # Créer un dictionnaire: marque -> liste de modèles uniques
        modeles_par_marque = {}
        
        for row in df_cars.iter_rows(named=True):
            company = row["Company Names"].strip().lower() if row["Company Names"] else None
            car_name = row["Cars Names"].strip() if row["Cars Names"] else None
            
            if company and car_name:
                if company not in modeles_par_marque:
                    modeles_par_marque[company] = []
                # Ajouter le modèle s'il n'existe pas
                if car_name not in modeles_par_marque[company]:
                    modeles_par_marque[company].append(car_name)
        
        return modeles_par_marque
    except Exception as e:
        print(f"⚠️ Erreur lors du chargement du CSV: {e}")
        return {}


def identifier_modele(marque: str, modele_complet: str, modeles_dict: dict) -> tuple:
    """
    Identifie le modèle exact et retourne (modèle_identifié, le_reste)
    
    Exemple:
    - identifier_modele("BMW", "M5 CS", {...}) -> ("M5", "CS")
    - identifier_modele("AUDI", "RS7 SPORTBACK", {...}) -> ("RS7", "SPORTBACK")
    """
    if not modele_complet or not marque:
        return (modele_complet, "")
    
    marque_lower = marque.strip().lower()
    modele_clean = modele_complet.strip()
    
    # Récupérer les modèles pour cette marque
    modeles_marque = modeles_dict.get(marque_lower, [])
    
    if not modeles_marque:
        return (modele_clean, "")
    
    # Chercher le meilleur match (le plus long, pour éviter les faux positifs)
    meilleur_match = None
    meilleur_longueur = 0
    
    for modele_ref in modeles_marque:
        modele_ref_lower = modele_ref.strip().lower()
        if modele_clean.lower().startswith(modele_ref_lower):
            if len(modele_ref) > meilleur_longueur:
                meilleur_match = modele_ref
                meilleur_longueur = len(modele_ref)
    
    # Si match trouvé, extraire le reste
    if meilleur_match:
        le_reste = modele_clean[len(meilleur_match):].strip()
        return (meilleur_match, le_reste)
    
    return (modele_clean, "")

# =========================
# 3. MARQUE / MODÈLE
# =========================
def reparer_marque_modele(df: pl.DataFrame) -> pl.DataFrame:
    marques_ref = {
        "Alfa Romeo": ["Alfa Romeo", "Alfa", "Alfa-Romeo"],
        "Aston Martin": ["Aston Martin", "Aston"],
        "Audi": ["Audi"],
        "Bentley": ["Bentley"],
        "BMW": ["BMW", "Bmw"],
        "Bugatti": ["Bugatti"],
        "Cadillac": ["Cadillac"],
        "Chevrolet": ["Chevrolet", "Chevy"],
        "Chrysler": ["Chrysler"],
        "Citroën": ["Citroen", "Citroën"],
        "Cupra": ["Cupra"],
        "Dacia": ["Dacia"],
        "Daihatsu": ["Daihatsu"],
        "Dodge": ["Dodge"],
        "DS": ["DS", "DS Automobiles"],
        "Ferrari": ["Ferrari"],
        "Fiat": ["Fiat"],
        "Ford": ["Ford"],
        "Genesis": ["Genesis"],
        "Honda": ["Honda"],
        "Hummer": ["Hummer"],
        "Hyundai": ["Hyundai"],
        "Infiniti": ["Infiniti"],
        "Isuzu": ["Isuzu"],
        "Jaguar": ["Jaguar"],
        "Jeep": ["Jeep"],
        "Kia": ["Kia"],
        "Lada": ["Lada"],
        "Lamborghini": ["Lamborghini"],
        "Lancia": ["Lancia"],
        "Land Rover": ["Land Rover", "Landrover", "Land-Rover"],
        "Lexus": ["Lexus"],
        "Lotus": ["Lotus"],
        "Maserati": ["Maserati"],
        "Mazda": ["Mazda"],
        "McLaren": ["McLaren", "Mclaren"],
        "Mercedes-Benz": ["Mercedes-Benz", "Mercedes", "Mercedes Benz"],
        "MG": ["MG", "Mg"],
        "Mini": ["Mini", "MINI"],
        "Mitsubishi": ["Mitsubishi"],
        "Nissan": ["Nissan"],
        "Opel": ["Opel"],
        "Peugeot": ["Peugeot"],
        "Polestar": ["Polestar"],
        "Porsche": ["Porsche"],
        "Renault": ["Renault"],
        "Rolls-Royce": ["Rolls-Royce", "Rolls Royce", "Rollsroyce"],
        "Saab": ["Saab"],
        "Seat": ["Seat", "SEAT"],
        "Skoda": ["Skoda", "Škoda"],
        "Smart": ["Smart", "SMART"],
        "SsangYong": ["SsangYong", "Ssangyong"],
        "Subaru": ["Subaru"],
        "Suzuki": ["Suzuki"],
        "Tesla": ["Tesla"],
        "Toyota": ["Toyota"],
        "Volkswagen": ["Volkswagen", "VW"],
        "Volvo": ["Volvo"],
        "Omoda": ["Omoda"],
        "BYD": ["BYD"],
        "Lynk & Co": ["Lynk & Co", "Lynk&Co", "Lynk"],
        "Aiways": ["Aiways"],
        "Nio": ["Nio", "NIO"],
        "Rivian": ["Rivian"],
        "Lucid": ["Lucid"],
        "Fisker": ["Fisker"]}
    
    # Extraire TOUTES les variantes
    all_variants = []
    for variants in marques_ref.values():
        all_variants.extend(variants)
    
    # Créer mapping inverse: variante → marque canonique
    variant_to_canonical = {}
    for canonical, variants in marques_ref.items():
        for variant in variants:
            variant_to_canonical[variant.lower()] = canonical
    
    regex_str = f"(?i)({'|'.join(all_variants)})"

    df_temp = df.with_columns([
        pl.col("marque").str.extract(regex_str, 1).alias("marque_extracted")
    ])

    return df_temp.with_columns([
        pl.col("marque_extracted")
        .str.to_lowercase()
        .replace(variant_to_canonical)
        # On ne met plus .fill_null("Inconnu") ici
        .alias("marque_clean"),
        
        pl.col("marque")
        .str.replace(regex_str, "")
        .str.replace_all(r"(?i)(TDI|TDCi|TGDI|TSI|CRDi|crdi|HDI|VTi|1\.2|1\.4|1\.6|2\.0|kW|CH|CV)", "")
        .str.strip_chars()
        .str.split(" ").list.slice(0, 2).list.join(" ")
        .alias("modele_clean")
    ]).drop([
        "marque", "marque_extracted"
    ]).rename({
        "marque_clean": "marque", 
        "modele_clean": "modele"
    }).filter(
        pl.col("marque").is_not_null() # Supprime les lignes où la marque n'a pas été trouvée
    )

def raffiner_modele_csv(df: pl.DataFrame) -> pl.DataFrame:
    """
    Identifie le modèle exact en cherchant dans le CSV et ajoute une colonne 'le_reste'
    """
    modeles_dict = charger_modeles_csv()
    
    if not modeles_dict:
        # Si CSV non trouvé, ajouter une colonne vide
        return df.with_columns(pl.lit("").alias("le_reste"))
    
    # Créer une fonction Python pour appliquer sur chaque ligne
    def identifier_et_splitter(marque, modele):
        modele_id, le_reste = identifier_modele(marque, modele, modeles_dict)
        return (modele_id, le_reste)
    
    # Appliquer sur le DataFrame
    result = df.with_columns([
        pl.struct(["marque", "modele"])
          .map_elements(lambda x: identifier_et_splitter(x["marque"], x["modele"]))
          .alias("modele_split")
    ])
    
    # Extraire les résultats
    return result.with_columns([
        pl.col("modele_split").list.get(0).fill_null("").alias("modele"),
        pl.col("modele_split").list.get(1).fill_null("").alias("le_reste")
    ]).drop("modele_split")

# =========================
# 1. CHARGEMENT
# =========================
def charger_donnees() -> pl.DataFrame:
    try:
        return pl.read_json("annonces_autoscout24.json")
    except Exception as e:
        print(f"Erreur de lecture : {e}")
        return pl.DataFrame()

# =========================
# 2. NETTOYAGE NUMÉRIQUE
# =========================
def nettoyer_numeriques(df: pl.DataFrame) -> pl.DataFrame:
    for col in ["prix", "kilometrage"]:
        df = df.with_columns(
            pl.col(col).cast(pl.Utf8).str.replace_all(r"[^\d]", "")
            .cast(pl.Int64, strict=False)
        )
    return df.filter(pl.col("prix").is_not_null())

# =========================
# 4. SPECS & LOCALISATION
# =========================
def extraire_specs_et_lieu(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.col("puissance").str.extract(r"(\d+)\s*kW", 1).cast(pl.Int64, strict=False).alias("puissance_kw"),
        pl.col("modele").str.extract(r"(\d+[.,]\d+)", 1).str.replace(",", ".").cast(pl.Float32, strict=False).alias("cylindree_l"),
        pl.col("localisation").str.extract(r"^(\d{4,5})", 1).alias("code_postal"),
        pl.col("localisation").str.replace(r"^\d{4,5}\s*", "").str.split("-").list.get(0).str.strip_chars().alias("ville"),
        pl.lit("Europe").alias("pays")
    ]).drop("localisation")

def corriger_cylindree(df: pl.DataFrame) -> pl.DataFrame:
    """
    - Véhicule électrique -> 0.0
    - Cylindrée hors bornes réalistes (<0.6 ou >8.5) -> null
    - Autres -> garder la valeur existante
    """
    return df.with_columns(
        pl.when(pl.col("carburant").str.to_lowercase() == "électrique")
          .then(0.0)
        .when((pl.col("cylindree_l") < 0.6) | (pl.col("cylindree_l") > 8.5))
          .then(None)
        .otherwise(pl.col("cylindree_l"))
        .alias("cylindree_l")
    )


def nettoyer_transmission(df: pl.DataFrame) -> pl.DataFrame:
    """
    Conserve uniquement les valeurs de transmission :
    - avant
    - arriere
    - 4x4
    Supprime les valeurs liées à la boîte de vitesses.
    """

    return df.with_columns(
        pl.when(
            pl.col("transmission")
            .str.to_lowercase()
            .is_in(["avant", "arriere", "arrière", "4x4"])
        )
        .then(pl.col("transmission").str.to_lowercase())
        .otherwise(None)
        .alias("transmission")
    )

# =========================
# 5. GESTION DES VALEURS ABERRANTES
# =========================
def traiter_valeurs_aberrantes(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remplace les valeurs manquantes/aberrantes par des moyennes intelligentes :
    - Année/Km : moyenne par marque+modèle, sinon moyenne générale
    - Puissance : laissée intacte (null si aberrante)
    """
    
    # Calculer les moyennes par marque+modèle pour année et km
    moyennes_marque_modele = df.group_by(["marque", "modele"]).agg([
        pl.col("annee")
          .filter((pl.col("annee") >= 1980) & (pl.col("annee") <= 2026))
          .mean()
          .alias("annee_moyenne_mm"),
        pl.col("kilometrage")
          .filter((pl.col("kilometrage") >= 0) & (pl.col("kilometrage") <= 500_000))
          .mean()
          .alias("km_moyenne_mm")
    ])
    
    # Calculer les moyennes générales (fallback)
    annee_moyenne_generale = df.select(
        pl.col("annee")
          .filter((pl.col("annee") >= 1980) & (pl.col("annee") <= 2026))
          .mean()
    ).item()
    
    km_moyenne_generale = df.select(
        pl.col("kilometrage")
          .filter((pl.col("kilometrage") >= 0) & (pl.col("kilometrage") <= 500_000))
          .mean()
    ).item()
    
    # Joindre les moyennes au DataFrame principal
    df = df.join(moyennes_marque_modele, on=["marque", "modele"], how="left")
    
    # Remplacer les valeurs aberrantes/nulles
    return df.with_columns([
        # Puissance : null si aberrante ou null (pas de remplacement)
        pl.when((pl.col("puissance_kw").is_null()) | 
                (pl.col("puissance_kw") < 5) | 
                (pl.col("puissance_kw") > 1000))
          .then(None)
          .otherwise(pl.col("puissance_kw"))
          .alias("puissance_kw"),

        # Année : moyenne marque+modèle, sinon moyenne générale
        pl.when((pl.col("annee").is_null()) | 
                (pl.col("annee") < 1980) | 
                (pl.col("annee") > 2026))
          .then(
              pl.when(pl.col("annee_moyenne_mm").is_not_null())
                .then(pl.col("annee_moyenne_mm").round(0).cast(pl.Int64))
                .otherwise(pl.lit(int(annee_moyenne_generale) if annee_moyenne_generale else 2010))
          )
          .otherwise(pl.col("annee"))
          .alias("annee"),

        # Kilométrage : moyenne marque+modèle, sinon moyenne générale
        pl.when((pl.col("kilometrage").is_null()) | 
                (pl.col("kilometrage") < 0) | 
                (pl.col("kilometrage") > 500_000))
          .then(
              pl.when(pl.col("km_moyenne_mm").is_not_null())
                .then(pl.col("km_moyenne_mm").round(0).cast(pl.Int64))
                .otherwise(pl.lit(int(km_moyenne_generale) if km_moyenne_generale else 100000))
          )
          .otherwise(pl.col("kilometrage"))
          .alias("kilometrage"),
    ]).drop(["annee_moyenne_mm", "km_moyenne_mm"])


# =========================
# 7. PRÉPARATION ML (SIMPLIFIÉE)
# =========================
def preparer_ml(df: pl.DataFrame) -> pl.DataFrame:
    """
    On ne remplit plus les médianes ici. 
    On se contente de s'assurer que les types sont corrects.
    """
    return df


# =========================
# MAIN
# =========================
def main():
    df = charger_donnees()
    if df.is_empty(): 
        return

    df = (
        df.pipe(nettoyer_numeriques)
          .pipe(reparer_marque_modele)
          .pipe(raffiner_modele_csv)
          .pipe(extraire_specs_et_lieu)
          .pipe(corriger_cylindree)
          .pipe(nettoyer_transmission)
          .pipe(traiter_valeurs_aberrantes)
          .pipe(preparer_ml)
          .drop(["lien_fiche", "puissance"])
          .drop_nulls(subset=["marque", "modele", "prix", "kilometrage", "annee", "cylindree_l", "transmission"])
    )

    print(df.select(pl.col("cylindree_l").null_count()))
    print(df.select(pl.col("carburant").value_counts()).unnest("carburant"))

    print(df.select(pl.col("transmission").null_count()))
    print(df.select(pl.col("transmission").value_counts()).unnest("transmission"))


    df.write_excel("autoscout_clean_ml.xlsx")
    print(f"✅ Export réussi : {df.shape[0]} lignes.")
    print(df.head(5))


if __name__ == "__main__":
    main()