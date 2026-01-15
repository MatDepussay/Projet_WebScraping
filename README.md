# 🚗 Projet Web Scraping - Prédiction de Prix AutoScout24

## 📋 Table des matières

- [Objectif](#-objectif)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Structure du projet](#-structure-du-projet)
- [Pipeline de données](#-pipeline-de-données)

---

## 🎯 Objectif

Développer une application complète de **scraping et analyse de données automobiles** pour estimer si le prix de vente d'une annonce AutoScout24 est cohérent avec le marché.

**Fonctionnalités principales:**
- ✅ Scraping automatisé des annonces AutoScout24
- ✅ Nettoyage et normalisation des données
- ✅ Identification intelligente des modèles de voitures
- ✅ Entraînement de modèles ML (Random Forest & XGBoost)
- ✅ Interface Streamlit pour visualiser et analyser les données
- ✅ Automatisation via GitHub Actions (scraping quotidien)

---

## 🏗️ Architecture

### Composants principaux

```
Projet_WebScraping/
├── Voiture_app.py              # 🎨 Interface Streamlit (3 onglets)
├── cleaning.py                 # 🧹 Pipeline de nettoyage (9 étapes)
├── MachineLearning.py          # 🤖 Entraînement des modèles
├── Scrapping_selenium_autodoc24.py  # 🕷️ Scraper autonome (GitHub Actions)
├── hello.py                    # 🔄 Fusion des sources de données
├── vehicle_models_merged.json   # 📚 Référence des modèles de voitures
└── pyproject.toml              # ⚙️ Configuration du projet
```

---

## 📥 Installation

### Prérequis
- Python 3.12+
- UV (package manager)

### Setup
```bash
# Cloner le projet
git clone <repo>
cd Projet_WebScraping

# Installer les dépendances
uv sync

# Optionnel: Fusionner les sources de données
uv run hello.py
```

---

## 🚀 Utilisation

### 1️⃣ Application Streamlit (Interface utilisateur)
```bash
uv run streamlit run Voiture_app.py
```

**Trois onglets disponibles:**

| Onglet | Fonction |
|--------|----------|
| 📥 **Scraper** | Scrape les annonces, charge JSON, ou fusionne nouvelles données |
| 📊 **Régression ML** | Entraîne Random Forest & XGBoost, affiche métriques |
| 🔍 **Sélectionner** | Visualise, filtre et prédit le prix des voitures |

### 2️⃣ Nettoyage des données
```bash
uv run cleaning.py
```
Génère `autoscout_clean_ml.xlsx` avec données nettoyées.

### 3️⃣ Entraînement ML
```bash
uv run MachineLearning.py
```
Entraîne les 2 modèles et sauvegarde les résultats dans `models/`.

### 4️⃣ Scraping autonome
```bash
uv run Scrapping_selenium_autodoc24.py
```
Scrape AutoScout24 et fusionne les nouvelles annonces (utilisé par GitHub Actions).

---

## 📂 Structure du projet

### `Voiture_app.py` - Interface Streamlit
Combine scraping, nettoyage et ML dans une seule application:
- **Onglet 1:** Scraping avec déduplication par URL
- **Onglet 2:** Entraînement et comparaison des modèles ML
- **Onglet 3:** Filtrage avancé et prédictions de prix

### `cleaning.py` - Pipeline de nettoyage (9 étapes)

| Étape | Fonction | Détails |
|-------|----------|---------|
| 1 | `charger_donnees()` | Charge `annonces_autoscout24.json` |
| 2 | `nettoyer_numeriques()` | Extrait prix/kilometrage, supprime nulls |
| 3 | `reparer_marque_modele()` | Normalise les marques et modèles |
| 4 | `raffiner_modele_csv()` | Identifie modèles via JSON (3 niveaux: strict/substring/fuzzy) |
| 5 | `extraire_specs_et_lieu()` | Extrait puissance, cylindrée, localisation |
| 6 | `nettoyer_transmission()` | Filtre transmission (avant/arrière/4x4) |
| 7 | `traiter_valeurs_aberrantes()` | Imputations intelligentes par marque+modèle |
| 8 | `preparer_ml()` | Prépare les colonnes numériques |
| 9 | `main()` | Exécute le pipeline complet |

### `MachineLearning.py` - Modèles prédictifs
Entraîne 2 modèles en parallèle:

**Random Forest:**
- 200 arbres, profondeur max 15
- Ensemble randomisé pour robustesse

**XGBoost:**
- 200 estimateurs, learning rate 0.1
- Optimisation gradient boostée

**Métriques:**
- RMSE (erreur quadratique moyenne)
- R² (coefficient de détermination)
- MAE (erreur absolue moyenne)
- Cross-validation 5-fold

### `hello.py` - Fusion des données
Fusionne 3 sources en un seul JSON:
- `vehicle models.json` (Wikipedia scraping)
- `EUROPEAN CARS DATASET.xlsx`
- `autoscout24-germany-dataset.csv`

**Résultat:** `vehicle_models_merged.json` (référence maître)

---

## 🔄 Pipeline de données

```
annonces_autoscout24.json
        ↓
    cleaning.py
   (9 étapes)
        ↓
   autoscout_clean_ml.xlsx
        ↓
   MachineLearning.py
        ↓
   Prédictions de prix
```

### Étapes clés du nettoyage

1. **Extraction numérique:** `"12 500 €"` → `12500`
2. **Normalisation marques:** `"Alfa" → "Alfa Romeo"`
3. **Identification modèles:** 
   - Niveau 1 (strict): `"M5 CS"` → `"M5"` ✓
   - Niveau 2 (substring): `"C31.1 Seduction"` → `"C3"` ✓
   - Niveau 3 (fuzzy): `"AygoAygo"` → `"Aygo"` ✓
4. **Imputations intelligentes:** Remplace valeurs aberrantes par moyennes (marque+modèle)
5. **One-Hot Encoding:** Variables catégorielles → numériques

---

## 📊 Résultats attendus

### Machine Learning
- **Métriques:** RMSE < €10k, R² > 0.75
- **Features importantes:** Marque, modèle, année, kilométrage
- **Prédictions:** Estimations de prix par catégorie

### Données
- **Avant nettoyage:** ~6500 annonces
- **Après filtrage:** ~2600 annonces (modèles identifiés)
- **Couverture:** 100+ marques automobiles

---

## ⚙️ Configuration (GitHub Actions)

**Workflow quotidien:** 3 AM UTC (chaque jour)
- Lance le scraper autonome
- Fusionne les nouvelles annonces
- Exécute le nettoyage
- Entraîne les modèles ML

---

## 📝 Notes techniques

### Encodages gérés
- Excel: Lectures multi-encodages (cp1252, utf-8, latin-1, etc.)
- JSON: UTF-8 standard
- CSV: Détection automatique

### Polars vs Pandas
- **Polars:** Scraping + nettoyage (performance)
- **Pandas:** ML (sklearn compatibility)

### Dépendances principales
```
polars          # DataFrames haute performance
scikit-learn    # ML classique
xgboost         # Gradient boosting
selenium        # Web scraping
streamlit       # Interface web
pydantic        # Validation
```

---

## 🔗 Fichiers clés

| Fichier | Taille | Rôle |
|---------|--------|------|
| `annonces_autoscout24.json` | ~10 MB | Données brutes scrappées |
| `autoscout_clean_ml.xlsx` | ~5 MB | Données nettoyées |
| `vehicle_models_merged.json` | ~500 KB | Référence de modèles |
| `models/random_forest_model.pkl` | ~50 MB | Modèle Random Forest |
| `models/xgboost_model.pkl` | ~20 MB | Modèle XGBoost |

---

## 📧 Contact

Projet réalisé dans le cadre du Master 2 - Janvier 2026