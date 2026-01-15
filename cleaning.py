# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars",
#     "xlsxwriter",
# ]
# ///

import polars as pl
from pathlib import Path
import csv
import re

# =========================
# 1. CHARGEMENT DES DONNÉES
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
# 3. RÉPARATION MARQUE/MODÈLE
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
    
    all_variants = []
    for variants in marques_ref.values():
        all_variants.extend(variants)
    
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
        pl.col("marque").is_not_null()
    )

# =========================
# 4. IDENTIFICATION MODÈLE (CSV)
# =========================
def charger_modeles_csv() -> dict:
    csv_path = Path("Cars Datasets 2025.csv")
    if not csv_path.exists():
        print(f"❌ Fichier introuvable: {csv_path}")
        return {}
    
    encodages = ["cp1252", "windows-1252", "latin-1", "iso-8859-1", "utf-8-sig", "utf-8"]
    
    for enc in encodages:
        try:
            rows = []
            with open(csv_path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
            
            modeles_par_marque = {}
            for row in rows:
                company = row.get("Company Names", "").strip().lower() if row.get("Company Names") else None
                car_name = row.get("Cars Names", "").strip() if row.get("Cars Names") else None
                
                if company and car_name:
                    if company not in modeles_par_marque:
                        modeles_par_marque[company] = []
                    if car_name not in modeles_par_marque[company]:
                        modeles_par_marque[company].append(car_name)
            
            print(f"✅ CSV chargé avec succès ({enc}): {len(modeles_par_marque)} marques")
            return modeles_par_marque
            
        except (UnicodeDecodeError, LookupError):
            continue
        except Exception as e:
            print(f"⚠️ Erreur avec {enc}: {e}")
            continue
    
    print(f"⚠️ Impossible de charger le CSV")
    return {}


def identifier_modele(marque: str, modele_complet: str, modeles_dict: dict) -> tuple:
    if not modele_complet or not marque:
        return (modele_complet or "", "", False)

    marque_lower = marque.strip().lower()
    s_orig = modele_complet.strip()
    if not s_orig:
        return ("", "", False)

    refs = modeles_dict.get(marque_lower, [])
    if not refs:
        return (s_orig, "", False)

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    s_norm = norm(s_orig)

    candidats = []
    for ref in refs:
        rn = norm(ref)
        if rn:
            candidats.append((ref, rn))

    candidats.sort(key=lambda x: len(x[1]), reverse=True)

    best_ref = None
    best_ref_norm = None
    for ref, rn in candidats:
        if rn in s_norm:
            best_ref = ref
            best_ref_norm = rn
            break

    if not best_ref:
        return (s_orig, "", False)

    chars = [re.escape(ch) for ch in best_ref_norm]
    loose_pattern = r"(?i)" + r"[^a-z0-9]*".join(chars)

    m = re.search(loose_pattern, s_orig)
    if not m:
        return (best_ref, "", True)

    start, end = m.span()
    avant = s_orig[:start].strip()
    apres = s_orig[end:].strip()
    le_reste = f"{avant} {apres}".strip()

    return (best_ref, le_reste, True)


def raffiner_modele_csv(df: pl.DataFrame) -> pl.DataFrame:
    modeles_dict = charger_modeles_csv()
    
    if not modeles_dict:
        return df.with_columns([
            pl.lit("").alias("le_reste"),
            pl.lit(False).alias("modele_identifie")
        ])
    
    def identifier_et_splitter(row):
        marque = row["marque"]
        modele = row["modele"]
        modele_id, le_reste, identifie = identifier_modele(marque, modele, modeles_dict)
        return {
            "modele": modele_id,
            "le_reste": le_reste,
            "modele_identifie": identifie
        }
    
    result = df.with_columns([
        pl.struct(["marque", "modele"])
          .map_elements(identifier_et_splitter, return_dtype=pl.Struct([
              pl.Field("modele", pl.Utf8),
              pl.Field("le_reste", pl.Utf8),
              pl.Field("modele_identifie", pl.Boolean)
          ]))
          .alias("modele_split")
    ])
    
    return result.with_columns([
        pl.col("modele_split").struct.field("modele").alias("modele"),
        pl.col("modele_split").struct.field("le_reste").alias("le_reste"),
        pl.col("modele_split").struct.field("modele_identifie").alias("modele_identifie")
    ]).drop("modele_split")

# =========================
# 5. EXTRACTION SPECS & LOCALISATION
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
# 6. NETTOYAGE TRANSMISSION
# =========================
def nettoyer_transmission(df: pl.DataFrame) -> pl.DataFrame:
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
# 7. GESTION VALEURS ABERRANTES
# =========================
def traiter_valeurs_aberrantes(df: pl.DataFrame) -> pl.DataFrame:
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
    
    df = df.join(moyennes_marque_modele, on=["marque", "modele"], how="left")
    
    return df.with_columns([
        pl.when((pl.col("puissance_kw").is_null()) | 
                (pl.col("puissance_kw") < 5) | 
                (pl.col("puissance_kw") > 1000))
          .then(None)
          .otherwise(pl.col("puissance_kw"))
          .alias("puissance_kw"),

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
# 8. PRÉPARATION ML
# =========================
def preparer_ml(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.col("cylindree_l").fill_null(0.0),
    ])

# =========================
# 9. MAIN
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

