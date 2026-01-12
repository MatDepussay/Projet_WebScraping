# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars",
#     "xlsxwriter",
# ]
# ///

import polars as pl

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
        .fill_null("Inconnu")
        .alias("marque_clean"),
        
        pl.col("marque")
        .str.replace(regex_str, "")
        .str.replace_all(r"(?i)(TDI|HDI|VTi|1\.2|1\.4|1\.6|2\.0|kW|CH|CV)", "")
        .str.strip_chars()
        .str.split(" ").list.slice(0, 2).list.join(" ")
        .alias("modele_clean")
    ]).with_columns([
        pl.col("modele_clean").fill_null("Inconnu")
    ]).drop(["marque", "marque_extracted"]).rename({"marque_clean": "marque", "modele_clean": "modele"})

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
    Supprime les lignes inexploitables et corrige les valeurs aberrantes
    à l'aide de bornes métier.
    """

    # Suppression des lignes sans année ou kilométrage
    df = df.filter(
        pl.col("annee").is_not_null() &
        pl.col("kilometrage").is_not_null()
    )

    med_puissance = (
        df.select(pl.col("puissance_kw").drop_nulls().median()).item()
        or 100
    )

    return df.with_columns([
        pl.when((pl.col("puissance_kw") < 5) | (pl.col("puissance_kw") > 1000))
          .then(med_puissance)
          .otherwise(pl.col("puissance_kw"))
          .alias("puissance_kw"),

        pl.when((pl.col("annee") < 1980) | (pl.col("annee") > 2026))
          .then(None)
          .otherwise(pl.col("annee"))
          .alias("annee"),

        pl.when((pl.col("kilometrage") < 0) | (pl.col("kilometrage") > 500_000))
          .then(None)
          .otherwise(pl.col("kilometrage"))
          .alias("kilometrage"),
    ]).drop_nulls(["annee", "kilometrage"])


# =========================
# 7. PRÉPARATION ML (SIMPLIFIÉE)
# =========================
def preparer_ml(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.col("puissance_kw").fill_null(df.select(pl.col("puissance_kw").median()).item()),
        pl.col("annee").fill_null(df.select(pl.col("annee").median()).item()),
        pl.col("cylindree_l").fill_null(0.0),
        pl.col("kilometrage").fill_null(df.select(pl.col("kilometrage").median()).item()),
    ])


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
          .pipe(extraire_specs_et_lieu)
          .pipe(nettoyer_transmission)
          .pipe(traiter_valeurs_aberrantes)
          .pipe(preparer_ml)
          .drop(["lien_fiche", "puissance"])
    )

    df.write_excel("autoscout_clean_ml.xlsx")
    print(f"✅ Export réussi : {df.shape[0]} lignes.")
    print(df.head(5))


if __name__ == "__main__":
    main()