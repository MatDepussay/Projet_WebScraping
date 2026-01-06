# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars",
# ]
# ///

import polars as pl
import re

# =========================
# 1. CHARGEMENT
# =========================
def charger_donnees() -> pl.DataFrame:
    return pl.read_json("annonces_autoscout24.json")


# =========================
# 2. DÉDOUBLONNAGE
# =========================
def eliminer_doublons(df: pl.DataFrame) -> pl.DataFrame:
    return df.unique(subset=["lien_fiche"])


# =========================
# 3. NETTOYAGE DU PRIX
# =========================
def nettoyer_prix(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col("prix")
            .str.replace_all(r"[^\d]", "")
            .cast(pl.Int64, strict=False)
            .alias("prix")
        )
        .filter(pl.col("prix").is_between(1000, 200_000))
    )


# =========================
# 4. PUISSANCE (kW)
# =========================
def extraire_puissance_kw(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("puissance")
        .str.extract(r"(\d+)\s*kW", 1)
        .cast(pl.Int64, strict=False)
        .alias("puissance_kw")
    )


# =========================
# 5. SÉPARATION MARQUE / MODELE
# =========================
def separer_marque_modele(df: pl.DataFrame) -> pl.DataFrame:
    marques_2mots = [
        "Alfa Romeo", "Mercedes Benz", "Aston Martin",
        "Land Rover", "Rolls Royce", "Mini Cooper"
    ]

    marques, modeles = [], []
    for s in df["marque"].to_list():
        if not s:
            marques.append("")
            modeles.append("")
            continue
        s = s.strip()
        for m in marques_2mots:
            if s.startswith(m):
                marques.append(m)
                modeles.append(s[len(m):].strip())
                break
        else:
            parts = s.split(" ", 1)
            marques.append(parts[0])
            modeles.append(parts[1].strip() if len(parts) > 1 else "")
    
    df = df.with_columns([
        pl.Series("marque", marques),
        pl.Series("modele", modeles)
    ])
    return df


# =========================
# 6. SÉPARATION MODELE / MOTEUR / PACK / OPTIONS
# =========================
def separer_specs(df: pl.DataFrame) -> pl.DataFrame:
    """
    Pour chaque ligne, on extrait :
    - modele : le nom du modèle pur
    - moteur : moteur (ex: 2.2 Turbodiesel, 1.4i, 1.6 T-GDi)
    - pack : reste général
    - options : mots-clés connus dans la specs
    """
    options_connues = ["Navi", "LED", "Tempomat", "PDC", "Caméra", "Clim", "BOSE"]

    modeles_clean, moteurs, packs, options_list = [], [], [], []

    for s in df["modele"].to_list():
        s = s.strip() if s else ""
        # EXTRACTION DU MODELE = premier mot ou mot+chiffre
        modele_match = re.match(r"^([A-Za-z0-9\-]+)", s)
        modele = modele_match.group(1) if modele_match else ""
        reste = s[len(modele):].strip() if modele else s

        # EXTRACTION DU MOTEUR = motif classique moteurs
        moteur_match = re.search(r"(\d\.\d\s*(?:[iI]|T-[A-Za-z]+|CV|bitdi|ecoblue|dci|T-GDi|turbodiesel)?)", reste)
        moteur = moteur_match.group(1) if moteur_match else ""

        # SUPPRIMER moteur du reste
        reste_sans_moteur = reste.replace(moteur, "").strip() if moteur else reste

        # EXTRACTION OPTIONS
        opts = [o for o in options_connues if re.search(rf"\b{o}\b", reste_sans_moteur, re.IGNORECASE)]

        # PACK = reste après suppression des options
        pack = reste_sans_moteur
        for o in opts:
            pack = re.sub(rf"\b{o}\b", "", pack, flags=re.IGNORECASE)
        pack = re.sub(r"\s{2,}", " ", pack).strip()

        # AJOUT AUX LISTES
        modeles_clean.append(modele)
        moteurs.append(moteur)
        packs.append(pack if pack else None)
        options_list.append(", ".join(opts) if opts else None)

    df = df.with_columns([
        pl.Series("modele", modeles_clean),
        pl.Series("moteur", moteurs),
        pl.Series("pack", packs),
        pl.Series("options", options_list)
    ])

    return df

# =========================
# LOCALISATION
# =========================
def normaliser_localisation(df: pl.DataFrame) -> pl.DataFrame:
    """
    Sépare la colonne 'localisation' en :
    - 'code_postal' : les chiffres initiaux (4 ou 5 chiffres)
    - 'ville'       : tout ce qui suit le code postal jusqu'à un tiret ou fin de chaîne
    - 'pays'        : vide pour l'instant
    """
    # Extraire le code postal (4 ou 5 chiffres au début)
    df = df.with_columns(
        pl.col("localisation")
        .str.extract(r"^(\d{4,5})", 1)
        .alias("code_postal")
    )

    # Extraire la ville
    df = df.with_columns(
        pl.col("localisation")
        .str.replace(r"^\d{4,5}\s*", "", literal=False)  # enlever code postal + espace
        .str.extract(r"^([^-]+)", 1)                     # tout jusqu'au premier tiret
        .str.strip_chars()                               # supprimer espaces début/fin
        .alias("ville")
    )

    # pays = vide pour l'instant
    df = df.with_columns([
        pl.lit(None).cast(pl.Utf8).alias("pays")
    ])

    return df


# =========================
# 7. CAST DES COLONNES NUMÉRIQUES
# =========================
def normaliser_types(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.col("kilometrage").cast(pl.Int64, strict=False),
        pl.col("annee").cast(pl.Int64, strict=False),
        pl.col("sieges").cast(pl.Int64, strict=False),
        pl.col("portes").cast(pl.Int64, strict=False),
    ])


# =========================
# 8. DATAFRAME FINAL ML
# =========================
def preparer_dataframe_ml(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.select([
            "prix",
            "kilometrage",
            "annee",
            "puissance_kw",
            "sieges",
            "portes",
            "marque",
            "modele",
            "moteur",
            "pack",
            "options",
            "carburant",
            "boite_de_vitesse",
            "transmission",
            "type_de_vehicule",
            "localisation",
            "code_postal",  
            "ville",       
            "pays",        
        ])
        .drop_nulls(["prix", "kilometrage", "annee"])
    )


# =========================
# 9. MAIN
# =========================
def main():
    df = charger_donnees()
    df = eliminer_doublons(df)
    df = nettoyer_prix(df)
    df = extraire_puissance_kw(df)
    df = separer_marque_modele(df)
    df = separer_specs(df)
    df = normaliser_types(df)
    df = normaliser_localisation(df)
    df.select(["localisation", "code_postal", "ville", "pays"]).head(10)

    df = preparer_dataframe_ml(df)

    # Affichage des 10 premières lignes
    print(df.head(10))

    # Statistiques rapides pour les colonnes qualitatives
    qualitatives = [
        "marque", "modele", "moteur", "pack", "options",
        "carburant", "boite_de_vitesse", "transmission",
        "type_de_vehicule", "localisation","code_postal",  
        "ville","pays"
    ]

    for col in qualitatives:
        if col not in df.columns:
            continue
        print(f"\n📊 {col.upper()}")
        vc = (
            df.select(pl.col(col).value_counts())
            .unnest(col)
            .sort("count", descending=True)
            .head(10)
        )
        print(vc)
    df_ml = df.drop(["localisation"])
    categorical_cols = ["marque", "modele", "moteur", "pack", "options",
                    "carburant", "boite_de_vitesse", "transmission",
                    "type_de_vehicule", "ville", "pays"]

    for col in categorical_cols:
        df_ml = df_ml.with_columns(
            pl.col(col).cast(pl.Categorical)
        )
    print(df_ml.null_count())

if __name__ == "__main__":
    main()
