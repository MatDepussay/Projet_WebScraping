# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars",
#     "xlsxwriter",
# ]
# ///

import polars as pl
from pathlib import Path
import json
import re

# =========================
# 1. CHARGEMENT DES DONNÉES
# =========================
def charger_donnees() -> pl.DataFrame:
    """
    Charge les données d'annonces automobiles depuis le fichier JSON brut récupéré lors du scraping.

    Cette fonction lit le fichier JSON contenant les annonces issues
    d'AutoScout24 et retourne les données sous forme de DataFrame Polars.
    En cas d'erreur lors de la lecture (fichier manquant, format invalide, etc.),
    un DataFrame vide est retourné et un message d'erreur est affiché.

    Returns
    -------
    pl.DataFrame
        DataFrame Polars contenant les annonces automobiles si la lecture
        réussit, sinon un DataFrame vide.
    """
    try:
        return pl.read_json("data/raw/annonces_autoscout24.json")
    except Exception as e:
        print(f"Erreur de lecture : {e}")
        return pl.DataFrame()

# =========================
# 2. NETTOYAGE NUMÉRIQUE
# =========================
def nettoyer_numeriques(df: pl.DataFrame) -> pl.DataFrame:
    """
    Nettoie et standardise les colonnes numériques issues des données brutes.

    Cette fonction traite les colonnes 'prix' et 'kilometrage' en supprimant
    tous les caractères non numériques, puis en les convertissant en entiers.
    Elle applique ensuite un filtre pour exclure les annonces dont le prix est
    nul, manquant ou manifestement incohérent (inférieur ou égal à 100).

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame Polars contenant les données brutes des annonces automobiles.

    Returns
    -------
    pl.DataFrame
        DataFrame Polars nettoyé, avec des colonnes numériques typées et
        des annonces à prix non cohérent exclues.
    """
    for col in ["prix", "kilometrage"]:
        df = df.with_columns(
            pl.col(col).cast(pl.Utf8).str.replace_all(r"[^\d]", "")
            .cast(pl.Int64, strict=False)
        )
    return df.filter((pl.col("prix").is_not_null()) & (pl.col("prix") > 100))

# =========================
# 3. RÉPARATION MARQUE/MODÈLE
# =========================
def reparer_marque_modele(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalise et fiabilise les informations de marque et de modèle des annonces.

    Cette fonction corrige les incohérences de saisie liées aux marques
    automobiles (variantes d'écriture, casse, abréviations) en s'appuyant
    sur un référentiel de marques standardisées. Elle extrait la marque
    depuis le champ textuel d'origine, puis nettoie le modèle en supprimant
    les informations techniques parasites (motorisation, puissance, cylindrée).

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame Polars contenant au minimum une colonne 'marque' issue
        des données brutes d'annonces automobiles.

    Returns
    -------
    pl.DataFrame
        DataFrame Polars avec deux colonnes normalisées :
        - 'marque' : marque canonique standardisée,
        - 'modele' : modèle nettoyé et homogénéisé.
        Les annonces dont la marque n'a pas pu être identifiée sont exclues.
    """
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

    # 1. On identifie les colonnes à supprimer qui existent réellement dans le DF
    cols_to_drop = ["marque", "marque_extracted"]
    if "modele" in df.columns:
        cols_to_drop.append("modele")

    # 2. On applique le drop et le rename
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
        .str.strip_chars()
        .alias("modele_clean")
    ]).drop(cols_to_drop) \
      .rename({
        "marque_clean": "marque", 
        "modele_clean": "modele"
    }).filter(
        pl.col("marque").is_not_null()
    )

# =========================
# 4. IDENTIFICATION MODÈLE (JSON)
# =========================
def charger_modeles_json() -> dict:
    """
    Charge un référentiel de modèles de véhicules depuis un fichier JSON.

    Le fichier JSON est structuré sous la forme d'une liste d'objets,
    chacun contenant une marque ("Make") et la liste des modèles associés
    ("Models"). Les marques sont normalisées en minuscules afin de
    faciliter les correspondances avec les données d'annonces nettoyées.

    En cas d'absence du fichier ou d'erreur de lecture, la fonction retourne
    un dictionnaire vide.

    Returns
    -------
    dict[str, list[str]]
        Dictionnaire associant chaque marque (en minuscules) à la liste
        de ses modèles officiels.
    """
    json_path = Path("data/references/vehicle_models_merged.json")
    if not json_path.exists():
        print(f"❌ Fichier introuvable: {json_path}")
        return {}
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        modeles_par_marque = {}
        for item in data:
            make = item.get("Make", "").strip().lower()
            models = item.get("Models", [])
            
            if make and models:
                existing_models = modeles_par_marque.get(make, [])
                modeles_par_marque[make] = list(set(existing_models + [m.strip() for m in models if m.strip()]))
        
        print(f"✅ JSON chargé avec succès: {len(modeles_par_marque)} marques")
        return modeles_par_marque
        
    except json.JSONDecodeError as e:
        print(f"❌ Erreur JSON: {e}")
        return {}
    except Exception as e:
        print(f"⚠️ Erreur lors du chargement: {e}")
        return {}


def identifier_modele(marque: str, modele_complet: str, modeles_dict: dict) -> tuple:
    """
    Identifie le modèle canonique d'un véhicule à partir d'une chaîne descriptive.

    Cette fonction compare une description textuelle complète du véhicule
    (incluant éventuellement des variantes, finitions ou informations techniques)
    à un référentiel de modèles associé à une marque donnée. L'identification
    repose sur une normalisation des chaînes et une recherche tolérante aux
    variations de format (espaces, tirets, caractères spéciaux).

    Lorsque plusieurs correspondances sont possibles, le modèle le plus long
    est privilégié afin d'éviter les faux positifs (ex. "595 Competizione"
    avant "595").

    Parameters
    ----------
    marque : str
        Marque du véhicule, préalablement nettoyée et normalisée.
    modele_complet : str
        Libellé complet du modèle tel qu'extrait de l'annonce.
    modeles_dict : dict
        Dictionnaire de référence associant chaque marque à la liste
        de ses modèles officiels.

    Returns
    -------
    tuple
        Tuple de la forme (modele_identifie, reste_libelle, succes) où :
        - modele_identifie : modèle reconnu selon le référentiel,
        - reste_libelle : partie résiduelle du libellé non utilisée
          (finition, motorisation, options),
        - succes : booléen indiquant si une correspondance fiable a été trouvée.
    """
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

    # Chercher TOUS les matches et prendre le PLUS LONG
    matches_valides = []
    for ref, rn in candidats:
        if rn in s_norm:
            matches_valides.append((ref, rn))
    
    if not matches_valides:
        return (s_orig, "", False)
    
    # Le premier match_valides est le plus long (grâce au tri)
    best_ref, best_ref_norm = matches_valides[0]

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
    modeles_dict = charger_modeles_json()
    
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
    """
    Extrait les caractéristiques techniques et géographiques des colonnes textuelles.

    Transforme la puissance en kW, tente une première extraction de la cylindrée,
    et décompose la localisation en code postal et ville.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame contenant les colonnes 'puissance', 'modele' et 'localisation'.

    Returns
    -------
    pl.DataFrame
        DataFrame enrichi des colonnes 'puissance_kw', 'cylindree_l', 
        'code_postal', 'ville' et 'pays'.
    """
    # Liste étendue incluant Mainz, Potsdam, Quickborn et les villes secondaires
    villes_allemagne = (
    r"(?i)(Berlin|Hamburg|München|Munich|Köln|Cologne|Frankfurt|Stuttgart|Düsseldorf|Dortmund|Essen|Leipzig|"
    r"Bremen|Dresden|Hannover|Nürnberg|Nuremberg|Duisburg|Bochum|Wuppertal|Bielefeld|Bonn|Münster|Karlsruhe|"
    r"Mannheim|Augsburg|Wiesbaden|Gelsenkirchen|Mönchengladbach|Braunschweig|Chemnitz|Kiel|Aachen|Halle|"
    r"Magdeburg|Freiburg|Krefeld|Mainz|Lübeck|Erfurt|Oberhausen|Rostock|Kassel|Hagen|Saarbrücken|Potsdam|"
    r"Ludwigshafen|Oldenburg|Osnabrück|Leverkusen|Heidelberg|Solingen|Darmstadt|Herne|Neuss|Regensburg|"
    r"Paderborn|Ingolstadt|Offenbach|Fürth|Würzburg|Ulm|Heilbronn|Pforzheim|Wolfsburg|Göttingen|Bottrop|"
    r"Reutlingen|Koblenz|Bremerhaven|Erlangen|Bergisch|Jena|Remscheid|Trier|Recklinghausen|Zwickau|"
    r"Heidelberg|Salzgitter|Siegen|Gütersloh|Cottbus|Gera|Hildesheim|Hanau|Witten|Iserlohn|Ludwigsburg|"
    r"Esslingen|Tübingen|Ratingen|Flensburg|Lünen|Villingen|Giessen|Marl|Konstanz|Worms|Velbert|Minden|"
    r"Neumünster|Norderstedt|Delmenhorst|Wilhelmshaven|Viersen|Gladbeck|Rheine|Detmold|Troisdorf|Castrop|"
    r"Arnsberg|Marburg|Lüdenscheid|Bamberg|Bayreuth|Brandenburg|Bocholt|Aschaffenburg|Landshut|Kempten|"
    r"Fulda|Aalen|Lippstadt|Herford|Kerpen|Plauen|Neu-Ulm|Weimar|Düren|Quickborn|Pinneberg|Elmshorn)"
)
    regex_suffixes_allemands = r"(?i).*(berg|burg|furt|ingen|shafen|stadt|dorf|bach|wald|bruck|haven|stein|itz|ow|au)$"
    
    # Charger ton fichier JSON
    json_it = Path("data/references/it.json")
    with open(json_it, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extraire les noms de villes et créer une regex géante
    # On récupère "city" (ex: Milan) mais il serait sage d'ajouter aussi une version 
    # italienne si ton JSON ne l'a pas (ex: Milano)
    noms_villes = set()
    for item in data:
        noms_villes.add(item['city'])
        # Optionnel : si ton JSON est en anglais, on peut ajouter manuellement 
        # des variantes ou gérer les accents
        
    # Transformer en pattern Regex : (Rome|Milan|Naples|...)
    regex_villes_it_json = r"(?i)\b(" + "|".join(sorted(noms_villes, key=len, reverse=True)) + r")\b"
    
    sigle_it = r"(?i)\b(AG|AL|AN|AO|AP|AQ|AR|AT|AV|BA|BG|BI|BL|BN|BO|BR|BS|BT|BZ|CA|CB|CE|CH|CI|CL|CN|CO|CR|CS|CT|CZ|EN|FC|FE|FG|FI|FM|FR|GE|GO|GR|IM|IS|KR|LC|LE|LI|LO|LT|LU|MB|MC|ME|MI|MN|MO|MS|MT|NA|NO|NU|OG|OR|OT|PA|PC|PD|PE|PG|PI|PN|PO|PR|PT|PU|PV|PZ|RA|RC|RE|RI|RN|RM|RO|SA|SI|SO|SP|SR|SS|SU|SV|TA|TE|TN|TO|TP|TR|TS|TV|UD|VA|VB|VC|VE|VI|VR|VS|VT|VV)\b"
        
    return df.with_columns([
        # 1. Puissance
        pl.col("puissance").str.extract(r"(\d+)\s*kW", 1).cast(pl.Int64, strict=False).alias("puissance_kw"),

        # 2. Initialisation cylindrée
        pl.col("modele").str.extract(r"(\d+[.,]\d+)", 1).str.replace(",", ".").cast(pl.Float32, strict=False).alias("cylindree_l"),

        # 3. Extraction du Code Postal
        pl.col("localisation").str.extract(r"^(\d{4,5}(?:\s*[A-Z]{2})?)", 1).alias("code_postal"),

        # 4. Identification du PAYS
        pl.when(
            # On checke la liste issue du JSON
            pl.col("localisation").str.contains(regex_villes_it_json) | 
            # On checke la province à la fin ou entre tirets
            pl.col("localisation").str.contains(sigle_it + r"$") |
            pl.col("localisation").str.contains(r"[\s\-\(]" + sigle_it + r"[\s\)]")
        ).then(pl.lit("Italie"))
        
        .when(pl.col("localisation").str.contains(r"^\d{4}\s*[A-Z]{2}"))
          .then(pl.lit("Pays-Bas"))
          
        .when(pl.col("localisation").str.contains(r"^\d{4}\s"))
          .then(pl.lit("Belgique"))
          
        # Bloc Allemagne
        .when(
            pl.col("localisation").str.contains(villes_allemagne) |
            pl.col("localisation").str.contains(regex_suffixes_allemands) | # Attrape Senftenberg, Coburg, etc.
            pl.col("localisation").str.contains(r"[ßüöäÜÖÄ]") |
            pl.col("localisation").str.contains(r"^[D|d]-\d{5}")
        ).then(pl.lit("Allemagne"))

        .when(pl.col("localisation").str.contains(r"^\d{5}\s"))
          .then(pl.lit("France"))
          
        .otherwise(None)
        .alias("pays"),

        # 5. Extraction de la Ville
        pl.col("localisation")
          .str.replace(r"^\d{4,5}(?:\s*[A-Z]{2})?\s*", "") # Vire le CP
          .str.replace(r"[\s\-\(]+[A-Z]{2}\)?$", "")      # Vire la province italienne propre (ex: - MI)
          .str.strip_chars()
          .alias("ville")
    ]).drop("localisation")

def corriger_cylindree(df: pl.DataFrame) -> pl.DataFrame:
    """
    Affiner l'extraction de la cylindrée selon des règles métier spécifiques.

    Applique une logique conditionnelle :
    - Positionne à 0.0 pour les véhicules électriques.
    - Utilise les codes modèles BMW (ex: 320 -> 2.0L) et Mercedes.
    - Recherche des motifs décimaux dans le reste du libellé.
    - Invalide les valeurs aberrantes (inférieures à 0.6L ou supérieures à 8.5L).

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame avec les colonnes 'marque', 'modele' et 'carburant'.

    Returns
    -------
    pl.DataFrame
        DataFrame avec la colonne 'cylindree_l' corrigée et nettoyée.
    """
    # Extraire cylindrée BMW uniquement pour les modèles identifiés comme XXX (3 chiffres)
    df = df.with_columns([
        pl.when(
            (pl.col("marque").str.to_lowercase() == "bmw") &
            (pl.col("modele_identifie")) &
            (pl.col("modele").str.contains(r"^\d{3}$"))
        )
        .then(
            pl.col("modele").str.slice(-2).cast(pl.Float32, strict=False)
        )
        .otherwise(None)
        .alias("cylindree_bmw")
    ])
    
    # Extraire cylindrée Mercedes uniquement pour les modèles identifiés comme XXX (3 chiffres)
    df = df.with_columns([
        pl.when(
            pl.col("marque").str.to_lowercase() == "mercedes-benz"
        )
        .then(
            # Extraire le nombre dans le modèle (ex: "C 200" -> 200)
            pl.col("modele")
            .str.extract(r"(\d+)", 1)
            .cast(pl.Float32) / 100  # diviser par 100 pour obtenir 2.0 à partir de 200
        )
        .otherwise(None)
        .alias("cylindree_mercedes")
    ])

    df = df.with_columns([
        pl.col("le_reste").str.extract(r"(\d+\.\d+)", 1).cast(pl.Float32).alias("cylindree_extraite")
    ])
    
    # Appliquer les règles finales (calcul de la cylindrée)
    df = df.with_columns(
        pl.when(pl.col("carburant").str.to_lowercase() == "électrique")
          .then(0.0)

        .when(
            pl.col("cylindree_l").is_null() &
            pl.col("cylindree_bmw").is_not_null()
        )
          .then(pl.col("cylindree_bmw") / 10)
        
        .when(
            pl.col("cylindree_l").is_null() &
            pl.col("cylindree_mercedes").is_not_null()
        )
          .then(pl.col("cylindree_mercedes"))
        
        .when(
            pl.col("cylindree_l").is_null() &
            pl.col("cylindree_extraite").is_not_null()
        )
          .then(pl.col("cylindree_extraite"))

        .otherwise(pl.col("cylindree_l"))
        .alias("cylindree_l")
    )
    
    # Validation des seuils (appliquée APRÈS le calcul)
    df = df.with_columns(
        pl.when(
            (pl.col("cylindree_l") < 0.6) |
            (pl.col("cylindree_l") > 8.5)
        )
          .then(None)
          .otherwise(pl.col("cylindree_l"))
          .alias("cylindree_l")
    ).drop("cylindree_bmw", "cylindree_mercedes", "cylindree_extraite")

    return df

# =========================
# 6. NETTOYAGE TRANSMISSION
# =========================
def nettoyer_transmission(df: pl.DataFrame) -> pl.DataFrame:
    """
    Standardise les types de transmission.

    Ne conserve que les valeurs connues ('avant', 'arriere', '4x4') et
    remplace les autres entrées (souvent des erreurs de saisie) par null.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame avec la colonne 'transmission'.

    Returns
    -------
    pl.DataFrame
        DataFrame avec les types de transmission normalisés.
    """
    # Mapping des synonymes pour ne rien perdre
    mapping = {
        "propulsion": "arrière",
        "arriere": "arrière",
        "intégrale": "4x4",
        "integrale": "4x4",
        "awd": "4x4",
        "quattro": "4x4"
    }
    return df.with_columns(
        pl.col("transmission")
        .str.to_lowercase()
        .str.strip_chars()
        .replace(mapping) # Remplace les synonymes
        .cast(pl.Utf8)
    ).with_columns(
        # On ne garde que les valeurs finales souhaitées
        pl.when(pl.col("transmission").is_in(["avant", "arrière", "4x4"]))
        .then(pl.col("transmission"))
        .otherwise(None)
        .alias("transmission")
    )

# =========================
# 7. GESTION VALEURS ABERRANTES
# =========================
def traiter_valeurs_aberrantes(df: pl.DataFrame) -> pl.DataFrame:
    # On définit les seuils de cohérence métier
    return df.with_columns([
        # Année : si hors plage, on met à null
        pl.when((pl.col("annee") >= 1980) & (pl.col("annee") <= 2027))
          .then(pl.col("annee"))
          .otherwise(None)
          .alias("annee"),

        # Kilométrage : si négatif ou trop élevé, on met à null
        pl.when((pl.col("kilometrage") >= 0) & (pl.col("kilometrage") <= 500_000))
          .then(pl.col("kilometrage"))
          .otherwise(None)
          .alias("kilometrage"),

        # Puissance : si hors limites réalistes, on met à null
        pl.when((pl.col("puissance_kw") >= 5) & (pl.col("puissance_kw") <= 1000))
          .then(pl.col("puissance_kw"))
          .otherwise(None)
          .alias("puissance_kw"),
          
        # Prix : (Déjà filtré dans nettoyer_numeriques, mais on peut sécuriser ici)
        pl.when(pl.col("prix") > 100)
          .then(pl.col("prix"))
          .otherwise(None)
          .alias("prix")
    ])

# =========================
# 8. PRÉPARATION ML
# =========================
def preparer_ml(df: pl.DataFrame) -> pl.DataFrame:
    return df


# =========================
# 9. MAIN
# =========================
def main():
    """
    Point d'entrée du script : exécute le pipeline complet de nettoyage.

    Ordonne les étapes de chargement, nettoyage numérique, réparation des labels,
    imputation des valeurs et export final vers un fichier Excel.
    """
    df = charger_donnees()
    if df.is_empty(): 
        return
    initial_count = df.height
    print(f"📊 Lignes initiales : {initial_count}")

    # 1. Pipeline de transformation
    df = (
        df.pipe(nettoyer_numeriques)
          .pipe(reparer_marque_modele)
          .pipe(raffiner_modele_csv)
          .filter(pl.col("modele_identifie"))
          .pipe(extraire_specs_et_lieu)
          .pipe(nettoyer_transmission)
          .pipe(corriger_cylindree)   
          .pipe(traiter_valeurs_aberrantes)
    )
    
    # 2. Affichage des NA AVANT suppression
    null_counts = df.null_count()
    print(f"❓ Valeurs manquantes avant filtrage final : {null_counts}")
    
    # 3. Nettoyage des colonnes techniques
    df = df.drop(["lien_fiche", "puissance", "modele_identifie", "le_reste"])
    
    # 4. Filtrage des colonnes capitales
    cols_critiques = ["prix", "ville", "annee", "kilometrage", "marque", "modele"]
    df = df.drop_nulls(cols_critiques)
    
    # 5. Affichage des NA APRÈS suppression
    final_count = df.height
    print(f"\n✅ Valeurs manquantes après filtrage (sur {final_count} lignes) :")
    final_nulls = df.null_count()
    for col in df.columns:
        count = final_nulls[col][0]
        if count > 0:
            print(f"  - {col}: {count} ({ (count/final_count)*100:.1f}%)")
            
    print(f"📊 Lignes filtrées : {initial_count - final_count} ({((initial_count - final_count)/initial_count)*100:.1f}%)")

    # Export en JSON (format 'records' pour avoir une liste d'objets [{}, {}])
    df.write_json("data/processed/autoscout_clean_ml.json")
    print(f"✅ Export réussi : {df.shape[0]} lignes.")
    print(df.head(5))


if __name__ == "__main__":
    main()

