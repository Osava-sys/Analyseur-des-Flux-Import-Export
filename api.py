"""
🚀 API FastAPI pour la Substitution aux Importations
=====================================================
API REST pour prédire le potentiel de substitution aux importations
et classifier les opportunités par secteur économique.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
MODELS_PATH = Path("models")
DATA_PATH = Path("data/processed")

# ============================================================
# Chargement des modèles et données
# ============================================================
def load_models():
    """Charge les modèles et le scaler"""
    with open(MODELS_PATH / "xgb_regression_substitution.pkl", 'rb') as f:
        model_reg = pickle.load(f)
    
    with open(MODELS_PATH / "xgb_classification_opportunity.pkl", 'rb') as f:
        model_clf = pickle.load(f)
    
    with open(MODELS_PATH / "scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    with open(MODELS_PATH / "model_metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    with open(MODELS_PATH / "api_config.json", 'r', encoding='utf-8') as f:
        api_config = json.load(f)
    
    return model_reg, model_clf, scaler, metadata, api_config

# Charger les modèles au démarrage
model_reg, model_clf, scaler, metadata, api_config = load_models()

# Charger les données de référence
df_data = pd.read_csv(DATA_PATH / "dataset_ml_complete.csv")
recommendations_df = pd.read_csv(MODELS_PATH / "recommendations_report.csv")

# ============================================================
# Modèles Pydantic
# ============================================================
class SectorInput(BaseModel):
    """Données d'entrée pour un secteur"""
    production_fcfa: float = Field(..., description="Production en milliards FCFA")
    production_tonnes: float = Field(..., description="Production en tonnes")
    imports_fcfa: float = Field(..., description="Importations en milliards FCFA")
    exports_fcfa: float = Field(..., description="Exportations en milliards FCFA")
    consommation_fcfa: float = Field(0, description="Consommation en milliards FCFA")
    year: int = Field(2023, description="Année de référence")

class PredictionResponse(BaseModel):
    """Réponse de prédiction"""
    substitution_score: float
    opportunity_class: str
    opportunity_level: int
    confidence: str
    recommendations: List[str]

class SectorAnalysis(BaseModel):
    """Analyse d'un secteur"""
    secteur: str
    production_mds_fcfa: float
    imports_mds_fcfa: float
    exports_mds_fcfa: float
    balance_commerciale: float
    ratio_production_imports: float
    croissance_production_pct: float
    score_substitution: float
    classification: str

class ModelInfo(BaseModel):
    """Informations sur les modèles"""
    version: str
    date_training: str
    regression_r2: float
    classification_accuracy: float
    features_count: int

# ============================================================
# Application FastAPI
# ============================================================
app = FastAPI(
    title="🌍 API Substitution aux Importations",
    description="""
    API pour identifier les opportunités de substitution aux importations
    basée sur des modèles XGBoost entraînés sur les données économiques.
    
    ## Fonctionnalités
    - 📊 Prédiction du score de substitution
    - 🎯 Classification des opportunités (Faible/Moyenne/Haute)
    - 📈 Analyse sectorielle complète
    - 🏆 Recommandations stratégiques
    """,
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Endpoints
# ============================================================

@app.get("/", tags=["Info"])
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "🌍 API Substitution aux Importations",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "sectors": "/sectors",
            "recommendations": "/recommendations"
        }
    }

@app.get("/health", tags=["Info"])
async def health_check():
    """Vérification de l'état de l'API"""
    return {
        "status": "healthy",
        "models_loaded": True,
        "regression_model": "XGBoost Regressor",
        "classification_model": "XGBoost Classifier"
    }

@app.get("/model-info", response_model=ModelInfo, tags=["Info"])
async def get_model_info():
    """Informations sur les modèles entraînés"""
    return ModelInfo(
        version=api_config['model_info']['version'],
        date_training=api_config['model_info']['date_training'],
        regression_r2=api_config['model_info']['regression_r2'],
        classification_accuracy=api_config['model_info']['classification_accuracy'],
        features_count=len(api_config['feature_columns'])
    )

@app.get("/sectors", tags=["Secteurs"])
async def get_sectors():
    """Liste des secteurs disponibles"""
    sectors = df_data['LIBELLES'].unique().tolist()
    return {
        "count": len(sectors),
        "sectors": sectors
    }

@app.get("/sectors/{sector_name}", tags=["Secteurs"])
async def get_sector_details(sector_name: str):
    """Détails d'un secteur spécifique"""
    # Recherche flexible
    sector_data = df_data[df_data['LIBELLES'].str.contains(sector_name, case=False, na=False)]
    
    if sector_data.empty:
        raise HTTPException(status_code=404, detail=f"Secteur '{sector_name}' non trouvé")
    
    # Données récentes (2021-2023)
    recent = sector_data[sector_data['year'] >= 2021]
    
    return {
        "secteur": sector_data['LIBELLES'].iloc[0],
        "annees_disponibles": sector_data['year'].tolist(),
        "production_moyenne_mds_fcfa": float(recent['production_fcfa'].mean()),
        "imports_moyenne_mds_fcfa": float(recent['imports_fcfa'].mean()),
        "exports_moyenne_mds_fcfa": float(recent['exports_fcfa'].mean()),
        "croissance_moyenne_pct": float(recent['production_fcfa_growth'].mean()),
        "historique": sector_data[['year', 'production_fcfa', 'imports_fcfa', 'exports_fcfa']].to_dict('records')
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prédictions"])
async def predict_substitution(input_data: SectorInput):
    """
    Prédit le score de substitution et la classe d'opportunité
    pour un ensemble de données sectorielles.
    """
    try:
        # Créer un DataFrame avec les features nécessaires
        # On utilise des valeurs par défaut pour les features manquantes
        features = {col: 0 for col in api_config['feature_columns']}
        
        # Remplir avec les données d'entrée
        features['production_fcfa'] = input_data.production_fcfa
        features['production_tonnes'] = input_data.production_tonnes
        features['imports_fcfa'] = input_data.imports_fcfa
        features['exports_fcfa'] = input_data.exports_fcfa
        features['consommation_fcfa'] = input_data.consommation_fcfa
        features['year'] = input_data.year
        
        # Calculer les features dérivées
        features['balance_commerciale_fcfa'] = input_data.exports_fcfa - input_data.imports_fcfa
        features['taux_couverture'] = (input_data.exports_fcfa / input_data.imports_fcfa * 100) if input_data.imports_fcfa > 0 else 0
        features['ratio_prod_conso_fcfa'] = (input_data.production_fcfa / abs(input_data.consommation_fcfa)) if input_data.consommation_fcfa != 0 else 0
        features['prix_unitaire_production'] = (input_data.production_fcfa / input_data.production_tonnes) if input_data.production_tonnes > 0 else 0
        
        # Créer le vecteur de features
        X = np.array([[features[col] for col in api_config['feature_columns']]])
        
        # Normaliser
        X_scaled = scaler.transform(X)
        
        # Prédictions
        score = float(model_reg.predict(X_scaled)[0])
        class_pred = int(model_clf.predict(X_scaled)[0])
        
        # Mapper la classe
        class_labels = {0: "Faible", 1: "Moyenne", 2: "Haute"}
        
        # Générer des recommandations
        recommendations = []
        if class_pred == 2:
            recommendations.append("✅ Secteur à fort potentiel - Priorité d'investissement")
            recommendations.append("📈 Augmenter les capacités de production")
        elif class_pred == 1:
            recommendations.append("⚠️ Potentiel modéré - Analyser les opportunités")
            recommendations.append("🔧 Améliorer la compétitivité")
        else:
            recommendations.append("📊 Faible potentiel actuel - Étude approfondie nécessaire")
            recommendations.append("🎯 Identifier les niches de marché")
        
        if input_data.imports_fcfa > input_data.production_fcfa:
            recommendations.append("🚨 Déficit de production - Opportunité de substitution")
        
        return PredictionResponse(
            substitution_score=round(score, 2),
            opportunity_class=class_labels[class_pred],
            opportunity_level=class_pred,
            confidence="Haute" if score > 10 else "Moyenne" if score > 1 else "Faible",
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")

@app.get("/recommendations", tags=["Recommandations"])
async def get_recommendations(top_n: int = 10):
    """Obtenir les recommandations sectorielles"""
    top_sectors = recommendations_df.head(top_n)
    
    return {
        "count": len(top_sectors),
        "recommendations": top_sectors.to_dict('records')
    }

@app.get("/recommendations/priority", tags=["Recommandations"])
async def get_priority_sectors():
    """Secteurs prioritaires pour l'action"""
    # Secteurs à fort potentiel
    high_potential = recommendations_df[
        recommendations_df['classification'].str.contains('Fort', na=False)
    ].head(5)
    
    # Secteurs déficitaires à développer
    deficit_sectors = recommendations_df[
        recommendations_df['balance_commerciale'] < 0
    ].nsmallest(5, 'balance_commerciale')
    
    return {
        "secteurs_champions": high_potential.to_dict('records'),
        "secteurs_prioritaires": deficit_sectors.to_dict('records')
    }

@app.get("/statistics", tags=["Statistiques"])
async def get_statistics():
    """Statistiques globales"""
    recent_data = df_data[df_data['year'] >= 2021]
    
    return {
        "total_secteurs": int(df_data['LIBELLES'].nunique()),
        "periode_analyse": f"{df_data['year'].min()} - {df_data['year'].max()}",
        "production_totale_mds_fcfa": float(recent_data.groupby('year')['production_fcfa'].sum().mean()),
        "imports_totaux_mds_fcfa": float(recent_data.groupby('year')['imports_fcfa'].sum().mean()),
        "exports_totaux_mds_fcfa": float(recent_data.groupby('year')['exports_fcfa'].sum().mean()),
        "balance_commerciale_mds_fcfa": float(
            recent_data.groupby('year')['exports_fcfa'].sum().mean() - 
            recent_data.groupby('year')['imports_fcfa'].sum().mean()
        )
    }

# ============================================================
# Lancement
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
