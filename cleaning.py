# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars",
#     "xlsxwriter",
# ]
# ///

import polars as pl
import re

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
    
    regex_str = f"(?i)({'|'.join(marques_ref)})"

    return df.with_columns([
        pl.col("marque").str.extract(regex_str, 1).str.to_titlecase()
        .replace("Mercedes", "Mercedes-Benz").alias("marque_clean"),
        
        pl.col("marque")
        .str.replace(regex_str, "")
        .str.replace_all(r"(?i)(TDI|HDI|VTi|1\.2|1\.4|1\.6|2\.0|kW|CH|CV)", "")
        .str.strip_chars()
        .str.split(" ").list.slice(0, 2).list.join(" ")
        .alias("modele_clean")
    ]).with_columns([
        pl.col("marque_clean").fill_null("Inconnu"),
        pl.col("modele_clean").fill_null("Inconnu")
    ]).drop("marque").rename({"marque_clean": "marque", "modele_clean": "modele"})

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

# =========================
# 5. PRÉPARATION ML (FIX DÉFINITIF)
# =========================
def preparer_ml(df: pl.DataFrame) -> pl.DataFrame:
    # On calcule les médianes avant pour éviter l'erreur de "strategy"
    med_p = df.select(pl.col("puissance_kw").median()).to_series()[0]
    med_a = df.select(pl.col("annee").median()).to_series()[0]

    # Un seul with_columns pour tout traiter sans doublons
    return df.with_columns([
        # Remplissage des Nulls ET correction des aberrations en une fois
        pl.when((pl.col("puissance_kw").is_null()) | (pl.col("puissance_kw") < 5) | (pl.col("puissance_kw") > 1000))
          .then(med_p)
          .otherwise(pl.col("puissance_kw"))
          .alias("puissance_kw"),
          
        pl.when((pl.col("annee").is_null()) | (pl.col("annee") < 1980) | (pl.col("annee") > 2026))
          .then(med_a)
          .otherwise(pl.col("annee"))
          .alias("annee"),
          
        pl.col("cylindree_l").fill_null(0.0)
    ])

# =========================
# MAIN
# =========================
def main():
    df = charger_donnees()
    if df.is_empty(): return

    df = (
        df.pipe(nettoyer_numeriques)
        .pipe(reparer_marque_modele)
        .pipe(extraire_specs_et_lieu)
        .pipe(preparer_ml)
    )

    df.write_excel("autoscout_clean_ml.xlsx")
    print(f"✅ Export réussi : {df.shape[0]} lignes.")
    print(df.head(5))

if __name__ == "__main__":
    main()