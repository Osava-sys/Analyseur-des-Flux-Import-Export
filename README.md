# 🇧🇫 Analyseur Import/Export - Burkina Faso

[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-green.svg)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Solution d'Intelligence Artificielle pour l'analyse commerciale et l'identification des opportunités de substitution aux importations**

---

## 🎯 Objectif

Transformer les données commerciales du Burkina Faso (2014-2025) en **insights stratégiques actionnables** pour :
- Identifier les secteurs à fort potentiel de substitution
- Réduire la dépendance aux importations
- Guider les investissements stratégiques
- Améliorer la balance commerciale nationale

---

## ✨ Fonctionnalités

### 📊 7 Modules Principaux

| Module | Description |
|--------|-------------|
| **🏠 Accueil** | Dashboard avec KPIs, graphiques d'évolution, top secteurs |
| **⚡ Temps Réel** | Monitoring live des indicateurs économiques |
| **📈 Analyse** | Analyse sectorielle détaillée avec diagnostic |
| **🎯 Recommandations** | Système de scoring IA pour priorisation |
| **🧪 Simulateur** | Multi-scénarios, sensibilité, projections temporelles |
| **📊 Performance ML** | Métriques des modèles XGBoost |
| **🤖 Assistant IA** | Chatbot RAG avec expertise économique |

### 🚀 Points Forts

- ✅ **185+ secteurs** analysés
- ✅ **Modèles XGBoost** (Régression + Classification)
- ✅ **Système RAG** avec Groq LLM
- ✅ **Thème Dark/Light** responsive
- ✅ **Export CSV/JSON** des données
- ✅ **Simulations avancées** multi-scénarios

---

## 🏗️ Architecture

```
hackathon-24h/
├── app.py                 # Application Streamlit principale
├── api.py                 # API REST pour prédictions
├── rag_system.py          # Système RAG + LLM
├── config.yaml            # Configuration
├── requirements.txt       # Dépendances Python
│
├── data/
│   ├── raw/               # Données brutes (CSV)
│   └── processed/         # Données nettoyées
│
├── models/
│   ├── xgb_regression_substitution.pkl
│   ├── xgb_classification_opportunity.pkl
│   ├── scaler.pkl
│   └── *.json             # Métadonnées et configs
│
├── documents/             # PDFs pour RAG
├── notebooks/             # Jupyter notebooks d'exploration
└── rag_index/             # Index FAISS
```

---

## 🚀 Installation

### Prérequis
- Python 3.10+
- pip ou conda

### Étapes

```bash
# 1. Cloner le repository
git clone https://github.com/votre-repo/hackathon-24h.git
cd hackathon-24h

# 2. Créer l'environnement virtuel
python -m venv venv_hackathon
source venv_hackathon/bin/activate  # Linux/Mac
# ou
.\venv_hackathon\Scripts\Activate.ps1  # Windows PowerShell

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

### Accès
Ouvrir http://localhost:8501 dans votre navigateur.

---

## 📈 Impact Économique Potentiel

| Indicateur | Estimation |
|------------|------------|
| **Économies sur imports** | 250-500 Mds FCFA/an |
| **Création d'emplois** | 150 000 - 450 000 |
| **Amélioration PIB** | +2% à +5% |
| **Secteurs prioritaires** | 30+ identifiés |

---

## 🛠️ Technologies

- **Frontend**: Streamlit, Plotly, Custom CSS
- **ML**: XGBoost, Scikit-learn
- **IA**: Groq LLM, FAISS, Sentence-Transformers
- **Data**: Pandas, NumPy

---

## 📖 Documentation

Voir le [Rapport Complet](Rapport_Analyseur_ImportExport_BurkinaFaso.md) pour :
- Description détaillée des fonctionnalités
- Valeur ajoutée par acteur
- Cas d'usage concrets
- Recommandations stratégiques

---

## 👥 Équipe

Projet développé dans le cadre du **Hackathon 24H - Décembre 2025**

---

## 📄 Licence

MIT License - Voir [LICENSE](LICENSE) pour plus de détails.

---

**Développé avec ❤️ pour le Burkina Faso**