"""
Système RAG (Retrieval Augmented Generation) pour l'Analyseur Import/Export
============================================================================
Utilise all-MiniLM-L6-v2 pour les embeddings et FAISS pour le stockage vectoriel.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import hashlib
from datetime import datetime

# PDF Processing
from PyPDF2 import PdfReader

# Embeddings
from sentence_transformers import SentenceTransformer

# Vector Store
import faiss

# LLM
from groq import Groq

# ============================================================
# Configuration
# ============================================================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CHUNK_SIZE = 500  # caractères par chunk
CHUNK_OVERLAP = 100  # chevauchement entre chunks
TOP_K_RETRIEVAL = 5  # nombre de documents à récupérer

# Chemins
BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RAG_INDEX_DIR = BASE_DIR / "rag_index"

# ============================================================
# Data Classes
# ============================================================
@dataclass
class DocumentChunk:
    """Représente un chunk de document avec métadonnées."""
    id: str
    text: str
    source: str
    source_type: str  # 'pdf', 'xgboost_data', 'recommendation'
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        # Convertir les valeurs numpy en types Python natifs
        def convert_value(v):
            if hasattr(v, 'item'):  # numpy types
                return v.item()
            elif isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            elif isinstance(v, (list, tuple)):
                return [convert_value(val) for val in v]
            return v
        
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "source_type": self.source_type,
            "metadata": convert_value(self.metadata)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DocumentChunk':
        return cls(
            id=data["id"],
            text=data["text"],
            source=data["source"],
            source_type=data["source_type"],
            metadata=data.get("metadata", {})
        )


# ============================================================
# PDF Processor
# ============================================================
class PDFProcessor:
    """Processeur de documents PDF avec chunking et métadonnées."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extrait le texte d'un fichier PDF."""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            return text.strip()
        except Exception as e:
            print(f"Erreur lors de la lecture de {pdf_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Nettoie le texte extrait."""
        # Supprimer les espaces multiples
        import re
        text = re.sub(r'\s+', ' ', text)
        # Supprimer les caractères spéciaux problématiques
        text = re.sub(r'[^\w\s\.,;:!?\-\(\)\[\]\'\"€%°/]', '', text)
        return text.strip()
    
    def create_chunks(self, text: str, source: str) -> List[DocumentChunk]:
        """Découpe le texte en chunks avec chevauchement."""
        chunks = []
        text = self.clean_text(text)
        
        if not text:
            return chunks
        
        # Découpage par phrases pour un meilleur contexte
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        
        current_chunk = ""
        chunk_idx = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunk_id = self._generate_chunk_id(source, chunk_idx)
                    chunks.append(DocumentChunk(
                        id=chunk_id,
                        text=current_chunk.strip(),
                        source=source,
                        source_type="pdf",
                        metadata={
                            "chunk_index": chunk_idx,
                            "char_count": len(current_chunk),
                            "extraction_date": datetime.now().isoformat()
                        }
                    ))
                    chunk_idx += 1
                    
                    # Garder le chevauchement
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                    current_chunk = overlap_text + sentence + ". "
                else:
                    current_chunk = sentence + ". "
        
        # Dernier chunk
        if current_chunk.strip():
            chunk_id = self._generate_chunk_id(source, chunk_idx)
            chunks.append(DocumentChunk(
                id=chunk_id,
                text=current_chunk.strip(),
                source=source,
                source_type="pdf",
                metadata={
                    "chunk_index": chunk_idx,
                    "char_count": len(current_chunk),
                    "extraction_date": datetime.now().isoformat()
                }
            ))
        
        return chunks
    
    def _generate_chunk_id(self, source: str, idx: int) -> str:
        """Génère un ID unique pour un chunk."""
        content = f"{source}_{idx}_{datetime.now().timestamp()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def process_all_pdfs(self, documents_dir: Path) -> List[DocumentChunk]:
        """Traite tous les PDFs d'un répertoire."""
        all_chunks = []
        pdf_files = list(documents_dir.glob("*.pdf"))
        
        print(f"[PDF] Traitement de {len(pdf_files)} fichiers PDF...")
        
        for pdf_path in pdf_files:
            print(f"  [DOC] Traitement: {pdf_path.name}")
            text = self.extract_text_from_pdf(pdf_path)
            if text:
                chunks = self.create_chunks(text, pdf_path.name)
                all_chunks.extend(chunks)
                print(f"     -> {len(chunks)} chunks crees")
        
        print(f"[OK] Total: {len(all_chunks)} chunks extraits des PDFs")
        return all_chunks


# ============================================================
# XGBoost Data Processor
# ============================================================
class XGBoostDataProcessor:
    """Convertit les données XGBoost en texte structuré pour le RAG."""
    
    def __init__(self, data_dir: Path = DATA_DIR, models_dir: Path = MODELS_DIR):
        self.data_dir = data_dir
        self.models_dir = models_dir
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Charge les données et recommandations."""
        df = pd.read_csv(self.data_dir / "dataset_ml_complete.csv")
        recommendations = pd.read_csv(self.models_dir / "recommendations_report.csv")
        
        with open(self.models_dir / "model_metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return df, recommendations, metadata
    
    def create_sector_descriptions(self, df: pd.DataFrame, recommendations: pd.DataFrame) -> List[DocumentChunk]:
        """Crée des descriptions textuelles détaillées par secteur."""
        chunks = []
        
        # Grouper par secteur
        sectors = df.groupby('LIBELLES')
        
        for sector_name, sector_data in sectors:
            # Données récentes (2021-2023)
            recent = sector_data[sector_data['year'] >= 2021]
            
            if recent.empty:
                continue
            
            # Calculs agrégés
            avg_production = recent['production_fcfa'].mean()
            avg_imports = recent['imports_fcfa'].mean()
            avg_exports = recent['exports_fcfa'].mean()
            avg_growth = recent['production_fcfa_growth'].mean()
            balance = avg_exports - avg_imports
            
            # Récupérer la recommandation si disponible
            reco = recommendations[recommendations['secteur'].str.contains(sector_name[:30], na=False)]
            
            score = reco['score_substitution'].values[0] if not reco.empty else 0
            classification = reco['classification'].values[0] if not reco.empty else "Non classifié"
            
            # Créer la description textuelle
            description = f"""
            SECTEUR: {sector_name}
            
            INDICATEURS ECONOMIQUES (Moyenne 2021-2023):
            - Production: {avg_production:.2f} milliards FCFA
            - Importations: {avg_imports:.2f} milliards FCFA
            - Exportations: {avg_exports:.2f} milliards FCFA
            - Balance commerciale: {balance:.2f} milliards FCFA ({'excédentaire' if balance > 0 else 'déficitaire'})
            - Croissance production: {avg_growth:.1f}%
            
            ANALYSE SUBSTITUTION AUX IMPORTATIONS:
            - Score de substitution: {score:.1f}/100
            - Classification: {classification}
            - Potentiel: {'ÉLEVÉ' if score > 70 else 'MODÉRÉ' if score > 40 else 'FAIBLE'}
            
            RECOMMANDATIONS:
            """
            
            if score > 70:
                description += """
            - Secteur prioritaire pour l'investissement
            - Fort potentiel de substitution aux importations
            - Recommandation: Augmenter les capacités de production
            - Action: Soutenir les producteurs locaux avec des incitations fiscales
            """
            elif score > 40:
                description += """
            - Potentiel modéré de substitution
            - Recommandation: Analyser les opportunités spécifiques
            - Action: Identifier les niches de marché à fort potentiel
            """
            else:
                description += """
            - Potentiel limité actuellement
            - Recommandation: Étude approfondie nécessaire
            - Action: Améliorer la compétitivité et la qualité
            """
            
            # Ajouter l'évolution historique
            historical = sector_data.sort_values('year')
            if len(historical) > 1:
                first_year = historical.iloc[0]
                last_year = historical.iloc[-1]
                prod_evolution = ((last_year['production_fcfa'] - first_year['production_fcfa']) / first_year['production_fcfa'] * 100) if first_year['production_fcfa'] > 0 else 0
                
                description += f"""
            ÉVOLUTION HISTORIQUE ({int(first_year['year'])} - {int(last_year['year'])}):
            - Évolution production: {prod_evolution:+.1f}%
            - Tendance: {'Croissante' if prod_evolution > 0 else 'Décroissante'}
            """
            
            chunk_id = hashlib.md5(f"sector_{sector_name}".encode()).hexdigest()[:16]
            chunks.append(DocumentChunk(
                id=chunk_id,
                text=description.strip(),
                source=f"XGBoost_Data_{sector_name[:30]}",
                source_type="xgboost_data",
                metadata={
                    "sector": sector_name,
                    "score": score,
                    "classification": classification,
                    "avg_production": avg_production,
                    "avg_imports": avg_imports,
                    "balance": balance
                }
            ))
        
        return chunks
    
    def create_global_summary(self, df: pd.DataFrame, recommendations: pd.DataFrame, metadata: Dict) -> List[DocumentChunk]:
        """Crée un résumé global des données économiques."""
        chunks = []
        
        # Résumé global
        recent_data = df[df['year'] >= 2021]
        yearly_stats = recent_data.groupby('year').agg({
            'production_fcfa': 'sum',
            'imports_fcfa': 'sum',
            'exports_fcfa': 'sum'
        })
        
        total_prod = yearly_stats['production_fcfa'].mean()
        total_imports = yearly_stats['imports_fcfa'].mean()
        total_exports = yearly_stats['exports_fcfa'].mean()
        balance = total_exports - total_imports
        
        # Secteurs prioritaires
        high_potential = recommendations[recommendations['classification'].str.contains('Fort', na=False)]
        
        global_summary = f"""
        RÉSUMÉ ÉCONOMIQUE GLOBAL - BURKINA FASO
        ========================================
        
        PÉRIODE D'ANALYSE: 2014-2025
        DONNÉES ACTUALISÉES: Décembre 2023
        
        INDICATEURS MACROÉCONOMIQUES (Moyenne 2021-2023):
        - Production totale: {total_prod:,.0f} milliards FCFA/an
        - Importations totales: {total_imports:,.0f} milliards FCFA/an
        - Exportations totales: {total_exports:,.0f} milliards FCFA/an
        - Balance commerciale: {balance:,.0f} milliards FCFA ({'EXCÉDENT' if balance > 0 else 'DÉFICIT'})
        
        NOMBRE DE SECTEURS ANALYSÉS: {len(recommendations)}
        
        CLASSIFICATION DES OPPORTUNITÉS:
        - Secteurs à fort potentiel: {len(high_potential)}
        - Secteurs à potentiel modéré: {len(recommendations[recommendations['classification'].str.contains('modéré', case=False, na=False)])}
        
        TOP 5 SECTEURS À FORT POTENTIEL DE SUBSTITUTION:
        """
        
        for idx, row in recommendations.head(5).iterrows():
            global_summary += f"""
        {idx+1}. {row['secteur'][:50]}
           - Score: {row['score_substitution']:.1f}/100
           - Production: {row['production_mds_fcfa']:.1f} Mds FCFA
           - Imports: {row['imports_mds_fcfa']:.1f} Mds FCFA
           - Classification: {row['classification']}
        """
        
        # Nouveau format de métadonnées (version 2.0)
        r2_score = metadata.get('metrics', {}).get('regression', {}).get('r2', 0)
        accuracy = metadata.get('metrics', {}).get('classification', {}).get('accuracy', 0)
        date_training = metadata.get('date_creation', 'N/A')
        periode = metadata.get('periode_donnees', '2014-2025')
        
        global_summary += f"""
        
        PERFORMANCE DES MODÈLES ML:
        - Modèle de régression (XGBoost): R² = {r2_score:.4f}
        - Modèle de classification (XGBoost): Accuracy = {accuracy:.4f}
        - Date d'entraînement: {date_training}
        - Période des données: {periode}
        
        MÉTHODOLOGIE:
        Le score de substitution est calculé à partir de multiples indicateurs économiques
        incluant la production, les importations, les exportations, la croissance et
        le taux de couverture. Les modèles XGBoost ont été entraînés sur les données
        historiques 2014-2025, incluant les nouvelles données des rapports DGD T2-2025.
        """
        
        chunk_id = hashlib.md5("global_summary".encode()).hexdigest()[:16]
        chunks.append(DocumentChunk(
            id=chunk_id,
            text=global_summary.strip(),
            source="XGBoost_Global_Summary",
            source_type="xgboost_data",
            metadata={
                "type": "global_summary",
                "total_production": total_prod,
                "total_imports": total_imports,
                "balance": balance,
                "sectors_count": len(recommendations)
            }
        ))
        
        return chunks

    def create_yearly_commerce_data(self) -> List[DocumentChunk]:
        """Crée des chunks pour les données commerciales historiques par année."""
        chunks = []

        # Charger commerce.csv
        commerce_path = self.data_dir / "commerce.csv"
        if not commerce_path.exists():
            print(f"[WARNING] Fichier commerce.csv non trouve: {commerce_path}")
            return chunks

        commerce_df = pd.read_csv(commerce_path)

        # Grouper par année
        years = commerce_df['year'].unique()

        for year in sorted(years):
            year_data = commerce_df[commerce_df['year'] == year]

            # Données totales de l'année
            total_row = year_data[year_data['LIBELLES'] == 'TOTAL']
            if total_row.empty:
                continue

            total_imports = total_row['imports_fcfa'].values[0]
            total_exports = total_row['exports_fcfa'].values[0]
            total_imports_tonnes = total_row['imports_tonnes'].values[0]
            total_exports_tonnes = total_row['exports_tonnes'].values[0]
            balance = total_exports - total_imports
            balance_tonnes = total_exports_tonnes - total_imports_tonnes
            taux_couverture = (total_exports / total_imports * 100) if total_imports > 0 else 0

            # Créer le texte descriptif pour l'année avec mots-clés bilingues pour meilleure retrieval
            year_int = int(year)
            description = f"""
BURKINA FASO TRADE DATA YEAR {year_int} - DONNÉES COMMERCIALES ANNÉE {year_int}
================================================================================
Keywords: trade balance {year_int}, balance commerciale {year_int}, imports exports {year_int},
commerce extérieur {year_int}, Burkina Faso {year_int}, statistiques commerciales {year_int}

BALANCE COMMERCIALE / TRADE BALANCE {year_int}:
- Importations totales / Total imports {year_int}: {total_imports:,.1f} milliards FCFA
- Exportations totales / Total exports {year_int}: {total_exports:,.1f} milliards FCFA
- Balance commerciale / Trade balance {year_int}: {balance:,.1f} milliards FCFA ({'EXCÉDENT/SURPLUS' if balance > 0 else 'DÉFICIT/DEFICIT'})
- Taux de couverture / Coverage rate {year_int}: {taux_couverture:.1f}%

VOLUMES ÉCHANGÉS EN {year_int} / TRADE VOLUMES {year_int}:
- Importations / Imports: {total_imports_tonnes:,.1f} milliers de tonnes
- Exportations / Exports: {total_exports_tonnes:,.1f} milliers de tonnes
- Balance en volume: {balance_tonnes:,.1f} milliers de tonnes

DÉTAIL PAR SECTEUR ({year_int}) / SECTOR DETAILS:
"""
            # Ajouter les principaux secteurs (hors TOTAL)
            sectors = year_data[year_data['LIBELLES'] != 'TOTAL'].sort_values('imports_fcfa', ascending=False)

            for _, row in sectors.head(10).iterrows():
                sector_name = row['LIBELLES']
                imp = row['imports_fcfa']
                exp = row['exports_fcfa']
                sect_balance = exp - imp
                description += f"""
- {sector_name[:60]}:
  Imports: {imp:.1f} Mds FCFA | Exports: {exp:.1f} Mds FCFA | Balance: {sect_balance:+.1f} Mds FCFA
"""

            chunk_id = hashlib.md5(f"commerce_year_{year}".encode()).hexdigest()[:16]
            chunks.append(DocumentChunk(
                id=chunk_id,
                text=description.strip(),
                source=f"Commerce_Data_{int(year)}",
                source_type="commerce_csv",
                metadata={
                    "year": int(year),
                    "total_imports": float(total_imports),
                    "total_exports": float(total_exports),
                    "balance": float(balance),
                    "taux_couverture": float(taux_couverture),
                    "data_type": "annual_commerce"
                }
            ))

        print(f"[OK] {len(chunks)} chunks crees pour les donnees commerciales historiques (annees {int(min(years))}-{int(max(years))})")
        return chunks

    def create_yearly_sector_details(self) -> List[DocumentChunk]:
        """Crée des chunks détaillés par secteur et par année."""
        chunks = []

        # Charger dataset_ml_complete.csv pour plus de détails
        df_path = self.data_dir / "dataset_ml_complete.csv"
        if not df_path.exists():
            return chunks

        df = pd.read_csv(df_path)

        # Grouper par année
        for year in sorted(df['year'].unique()):
            year_data = df[df['year'] == year]

            # Agrégations pour cette année
            total_production = year_data['production_fcfa'].sum()
            total_imports = year_data['imports_fcfa'].sum()
            total_exports = year_data['exports_fcfa'].sum()
            balance = total_exports - total_imports

            year_int = int(year)
            description = f"""
ECONOMIC INDICATORS BURKINA FASO {year_int} - INDICATEURS ÉCONOMIQUES {year_int}
================================================================================
Keywords: economic data {year_int}, données économiques {year_int}, production {year_int},
trade balance {year_int}, balance commerciale {year_int}, GDP {year_int}, PIB {year_int},
imports exports {year_int}, Burkina Faso statistics {year_int}

SYNTHÈSE ANNUELLE / ANNUAL SUMMARY {year_int}:
- Production nationale totale / Total national production {year_int}: {total_production:,.1f} milliards FCFA
- Importations totales / Total imports {year_int}: {total_imports:,.1f} milliards FCFA
- Exportations totales / Total exports {year_int}: {total_exports:,.1f} milliards FCFA
- Balance commerciale / Trade balance {year_int}: {balance:,.1f} milliards FCFA ({'excédentaire/surplus' if balance > 0 else 'déficitaire/deficit'})

SECTEURS ÉCONOMIQUES EN {year_int} / ECONOMIC SECTORS:
"""
            # Top 5 secteurs par production
            top_production = year_data.nlargest(5, 'production_fcfa')
            description += f"\nTop 5 secteurs par production en {int(year)}:\n"
            for _, row in top_production.iterrows():
                description += f"  - {row['LIBELLES'][:50]}: {row['production_fcfa']:.1f} Mds FCFA\n"

            # Top 5 secteurs importateurs
            top_imports = year_data.nlargest(5, 'imports_fcfa')
            description += f"\nTop 5 secteurs importateurs en {int(year)}:\n"
            for _, row in top_imports.iterrows():
                description += f"  - {row['LIBELLES'][:50]}: {row['imports_fcfa']:.1f} Mds FCFA\n"

            chunk_id = hashlib.md5(f"yearly_detail_{year}".encode()).hexdigest()[:16]
            chunks.append(DocumentChunk(
                id=chunk_id,
                text=description.strip(),
                source=f"Economic_Data_{int(year)}",
                source_type="economic_data",
                metadata={
                    "year": int(year),
                    "total_production": float(total_production),
                    "total_imports": float(total_imports),
                    "total_exports": float(total_exports),
                    "balance": float(balance),
                    "data_type": "yearly_economic"
                }
            ))

        print(f"[OK] {len(chunks)} chunks crees pour les indicateurs economiques annuels")
        return chunks

    def create_methodology_documentation(self) -> List[DocumentChunk]:
        """Crée des chunks documentant la méthodologie de calcul des scores."""
        chunks = []

        # Documentation du score de substitution
        methodology_doc = """
MÉTHODOLOGIE DE CALCUL DU SCORE DE SUBSTITUTION AUX IMPORTATIONS
=================================================================
Keywords: score substitution, calcul score, méthodologie, comment calculer, formule score,
substitution imports, potentiel substitution, methodology, calculation, how to calculate

LE SCORE DE SUBSTITUTION - DÉFINITION:
Le score de substitution est une note sur 100 points qui évalue le potentiel d'un secteur
à remplacer les importations par de la production locale. Plus le score est élevé, plus
le secteur présente un fort potentiel de substitution aux importations.

COMPOSANTES DU SCORE (Total: 100 points maximum):

1. RATIO PRODUCTION/IMPORTS (0 à 40 points):
   - Ratio > 2 (production double des imports): 40 points
   - Ratio entre 1 et 2: 25 points
   - Ratio entre 0.5 et 1: 15 points
   - Ratio < 0.5: 5 points
   Interprétation: Un ratio élevé signifie que la production locale couvre déjà
   une grande partie de la demande intérieure.

2. CROISSANCE DE LA PRODUCTION (0 à 20 points):
   - Croissance > 20%: 20 points
   - Croissance entre 10% et 20%: 15 points
   - Croissance entre 0% et 10%: 10 points
   - Croissance négative: 0 points
   Interprétation: Une croissance forte indique une dynamique positive du secteur.

3. TAUX D'AUTOSUFFISANCE (0 à 20 points):
   - Taux > 80%: 20 points
   - Taux entre 60% et 80%: 15 points
   - Taux entre 40% et 60%: 10 points
   - Taux < 40%: 5 points
   Interprétation: Mesure la capacité du pays à satisfaire sa demande intérieure.

4. INTENSITÉ EXPORT (0 à 20 points):
   - Intensité > 50%: 20 points
   - Intensité entre 20% et 50%: 15 points
   - Intensité entre 5% et 20%: 10 points
   - Intensité < 5%: 5 points
   Interprétation: Un secteur exportateur a démontré sa compétitivité internationale.

CLASSIFICATION DES SECTEURS:
- Score > 70: "Fort potentiel - Secteur dominant" (Priorité HAUTE)
- Score 40-70: "Potentiel modéré - Équilibré" (Priorité MOYENNE)
- Score < 40: "Faible potentiel - Dépendant imports" (Priorité BASSE)

EXEMPLE DE CALCUL:
Pour un secteur avec:
- Ratio production/imports = 2.5 → 40 points
- Croissance production = 15% → 15 points
- Taux autosuffisance = 75% → 15 points
- Intensité export = 30% → 15 points
Score total = 40 + 15 + 15 + 15 = 85/100 → Fort potentiel
"""

        chunk_id = hashlib.md5("methodology_substitution_score".encode()).hexdigest()[:16]
        chunks.append(DocumentChunk(
            id=chunk_id,
            text=methodology_doc.strip(),
            source="Methodology_Substitution_Score",
            source_type="methodology",
            metadata={"type": "methodology", "topic": "score_substitution"}
        ))

        return chunks

    def create_best_sectors_chunk(self, recommendations: pd.DataFrame) -> List[DocumentChunk]:
        """Crée des chunks pour les meilleurs secteurs de substitution."""
        chunks = []

        # Top 10 secteurs pour la substitution
        top_sectors = recommendations.nlargest(10, 'score_substitution')

        best_sectors_doc = """
MEILLEURS SECTEURS POUR LA SUBSTITUTION AUX IMPORTATIONS AU BURKINA FASO
=========================================================================
Keywords: meilleurs secteurs, best sectors, top secteurs, substitution, opportunités,
secteurs prioritaires, investir, investment opportunities, où investir, potentiel élevé

CLASSEMENT DES 10 MEILLEURS SECTEURS (par score de substitution):
"""

        for rank, (_, row) in enumerate(top_sectors.iterrows(), 1):
            sector = row['secteur']
            score = row['score_substitution']
            prod = row['production_mds_fcfa']
            imp = row['imports_mds_fcfa']
            classification = row['classification']

            best_sectors_doc += f"""
{rank}. {sector}
   - Score de substitution: {score}/100
   - Classification: {classification}
   - Production actuelle: {prod:.1f} milliards FCFA
   - Importations: {imp:.1f} milliards FCFA
   - Potentiel: {'EXCELLENT' if score > 80 else 'TRÈS BON' if score > 60 else 'BON'}
"""

        best_sectors_doc += """

RECOMMANDATIONS GÉNÉRALES:
- Les secteurs avec un score > 70 sont prioritaires pour l'investissement
- Privilégier les secteurs où la production locale est déjà significative
- Les secteurs en forte croissance offrent les meilleures perspectives
"""

        chunk_id = hashlib.md5("best_sectors_substitution".encode()).hexdigest()[:16]
        chunks.append(DocumentChunk(
            id=chunk_id,
            text=best_sectors_doc.strip(),
            source="Best_Sectors_Substitution",
            source_type="recommendation",
            metadata={"type": "best_sectors", "count": 10}
        ))

        return chunks

    def create_simulation_guide(self) -> List[DocumentChunk]:
        """Crée un guide pour la simulation d'investissement."""
        chunks = []

        simulation_guide = """
GUIDE DE SIMULATION D'INVESTISSEMENT - BURKINA FASO
====================================================
Keywords: simulation investissement, investment simulation, simuler, comment simuler,
calculer retour, ROI, retour sur investissement, investir FCFA, business plan

COMMENT SIMULER UN INVESTISSEMENT DANS UN SECTEUR:

1. PARAMÈTRES DE SIMULATION:
   - Montant de l'investissement (en FCFA)
   - Secteur cible
   - Durée de l'investissement (années)

2. FACTEURS PRIS EN COMPTE:
   a) Score de substitution du secteur
   b) Croissance historique du secteur
   c) Volume actuel des importations (potentiel de marché)
   d) Taux de couverture actuel

3. ESTIMATION DU POTENTIEL DE MARCHÉ:
   Le potentiel de marché = Importations actuelles × (1 - Taux de couverture/100)
   Exemple: Si imports = 100 Mds FCFA et taux couverture = 60%
   Potentiel = 100 × (1 - 0.6) = 40 Mds FCFA de marché à conquérir

4. CALCUL DU RETOUR ESTIMÉ:
   - Secteurs score > 70: ROI potentiel 15-25% par an
   - Secteurs score 40-70: ROI potentiel 8-15% par an
   - Secteurs score < 40: ROI potentiel 3-8% par an

5. EXEMPLE DE SIMULATION:
   Investissement: 500 millions FCFA dans les Matières textiles (score: 100)
   - Marché potentiel: 68.4 Mds FCFA d'imports
   - Part de marché visée (1%): 684 millions FCFA/an
   - ROI estimé: 20-25% avec score maximal
   - Retour annuel estimé: 100-125 millions FCFA

6. FACTEURS DE RISQUE À CONSIDÉRER:
   - Volatilité du secteur
   - Dépendance aux matières premières importées
   - Concurrence locale et internationale
   - Infrastructure disponible

7. RECOMMANDATIONS PAR NIVEAU D'INVESTISSEMENT:
   - Petit investissement (< 50 millions FCFA): Secteurs textiles, agroalimentaire
   - Investissement moyen (50-500 millions): Produits végétaux, industries alimentaires
   - Gros investissement (> 500 millions): Métaux, matériel de transport
"""

        chunk_id = hashlib.md5("simulation_investment_guide".encode()).hexdigest()[:16]
        chunks.append(DocumentChunk(
            id=chunk_id,
            text=simulation_guide.strip(),
            source="Investment_Simulation_Guide",
            source_type="guide",
            metadata={"type": "guide", "topic": "investment_simulation"}
        ))

        return chunks

    def create_historical_progression(self, df: pd.DataFrame) -> List[DocumentChunk]:
        """Crée des chunks sur la progression historique des secteurs."""
        chunks = []

        # Calculer la progression par secteur depuis 2014
        sectors_progress = []
        for sector in df['LIBELLES'].unique():
            sector_data = df[df['LIBELLES'] == sector].sort_values('year')
            if len(sector_data) < 2:
                continue

            first_year = sector_data.iloc[0]
            last_year = sector_data.iloc[-1]

            if first_year['production_fcfa'] > 0:
                prod_growth = ((last_year['production_fcfa'] - first_year['production_fcfa']) /
                              first_year['production_fcfa'] * 100)
            else:
                prod_growth = 0

            if first_year['imports_fcfa'] > 0:
                import_growth = ((last_year['imports_fcfa'] - first_year['imports_fcfa']) /
                                first_year['imports_fcfa'] * 100)
            else:
                import_growth = 0

            sectors_progress.append({
                'sector': sector,
                'prod_growth': prod_growth,
                'import_growth': import_growth,
                'first_year': int(first_year['year']),
                'last_year': int(last_year['year']),
                'current_prod': last_year['production_fcfa']
            })

        # Trier par croissance de production
        sectors_progress.sort(key=lambda x: x['prod_growth'], reverse=True)

        progression_doc = """
PROGRESSION DES SECTEURS ÉCONOMIQUES AU BURKINA FASO (2014-2025)
=================================================================
Keywords: progression secteurs, évolution, croissance, growth, depuis 2014,
quels secteurs progressé, historical growth, évolution historique, tendances

SECTEURS AYANT LE PLUS PROGRESSÉ EN PRODUCTION:
"""

        for rank, sp in enumerate(sectors_progress[:10], 1):
            progression_doc += f"""
{rank}. {sp['sector'][:55]}
   - Croissance production ({sp['first_year']}-{sp['last_year']}): {sp['prod_growth']:+.1f}%
   - Évolution imports: {sp['import_growth']:+.1f}%
   - Production actuelle: {sp['current_prod']:.1f} Mds FCFA
   - Dynamique: {'FORTE CROISSANCE' if sp['prod_growth'] > 50 else 'CROISSANCE MODÉRÉE' if sp['prod_growth'] > 0 else 'EN DÉCLIN'}
"""

        # Secteurs dont les imports ont le plus diminué (substitution réussie)
        sectors_progress.sort(key=lambda x: x['import_growth'])
        progression_doc += """

SECTEURS OÙ LA SUBSTITUTION AUX IMPORTS A LE MIEUX FONCTIONNÉ:
(Secteurs dont les importations ont diminué)
"""

        for sp in sectors_progress[:5]:
            if sp['import_growth'] < 0:
                progression_doc += f"""
- {sp['sector'][:55]}
  Réduction des imports: {sp['import_growth']:.1f}% | Croissance production: {sp['prod_growth']:+.1f}%
"""

        chunk_id = hashlib.md5("historical_progression".encode()).hexdigest()[:16]
        chunks.append(DocumentChunk(
            id=chunk_id,
            text=progression_doc.strip(),
            source="Historical_Progression",
            source_type="analysis",
            metadata={"type": "historical", "period": "2014-2025"}
        ))

        return chunks

    def create_entrepreneur_recommendations(self, recommendations: pd.DataFrame) -> List[DocumentChunk]:
        """Crée des recommandations pour différents profils d'entrepreneurs."""
        chunks = []

        entrepreneur_doc = """
RECOMMANDATIONS POUR ENTREPRENEURS AU BURKINA FASO
===================================================
Keywords: entrepreneur, débutant, opportunités, recommandations, conseils,
petit budget, investir, créer entreprise, business opportunities, PME, startup

OPPORTUNITÉS POUR ENTREPRENEURS DÉBUTANTS (Budget < 50 millions FCFA):

1. SECTEUR TEXTILE ET HABILLEMENT
   - Score: Fort potentiel (100/100)
   - Investissement initial: 10-30 millions FCFA
   - Activités: Confection, couture, transformation coton local
   - Avantages: Matière première locale, forte demande, savoir-faire existant
   - Conseil: Commencer par des niches (uniformes, vêtements traditionnels)

2. PRODUITS AGROALIMENTAIRES
   - Score: Fort potentiel (90/100)
   - Investissement initial: 15-40 millions FCFA
   - Activités: Transformation céréales, fruits, légumes locaux
   - Avantages: Abondance de matières premières, marché local important
   - Conseil: Se spécialiser dans un produit (jus, farines, conserves)

3. CUIRS ET PEAUX
   - Score: Potentiel modéré
   - Investissement initial: 20-45 millions FCFA
   - Activités: Maroquinerie, chaussures artisanales
   - Avantages: Matière première disponible (élevage), tradition artisanale
   - Conseil: Viser l'export artisanal et le tourisme

4. COSMÉTIQUES NATURELS
   - Score: Bon potentiel
   - Investissement initial: 5-25 millions FCFA
   - Activités: Savons, huiles (karité, baobab), soins naturels
   - Avantages: Ressources locales uniques, tendance mondiale bio
   - Conseil: Certification bio pour l'export

5. ARTISANAT ET DÉCORATION
   - Score: Potentiel modéré
   - Investissement initial: 5-20 millions FCFA
   - Activités: Bronze, tissage, poterie, vannerie
   - Avantages: Savoir-faire unique, valeur culturelle
   - Conseil: E-commerce et marchés touristiques

CONSEILS GÉNÉRAUX POUR RÉUSSIR:
- Commencer petit et réinvestir les bénéfices
- Se former sur la gestion et la comptabilité
- Privilégier les circuits courts (fournisseurs locaux)
- S'associer avec des producteurs locaux
- Viser la qualité plutôt que le volume
- Explorer les possibilités d'export sous-régional (UEMOA)

RESSOURCES ET ACCOMPAGNEMENT:
- Maison de l'Entreprise du Burkina Faso
- Fonds Burkinabè de Développement Économique et Social (FBDES)
- Chambres de Commerce et d'Industrie
"""

        chunk_id = hashlib.md5("entrepreneur_recommendations".encode()).hexdigest()[:16]
        chunks.append(DocumentChunk(
            id=chunk_id,
            text=entrepreneur_doc.strip(),
            source="Entrepreneur_Recommendations",
            source_type="guide",
            metadata={"type": "recommendations", "target": "entrepreneurs"}
        ))

        return chunks

    def create_sector_potential_queries(self, recommendations: pd.DataFrame) -> List[DocumentChunk]:
        """Crée des chunks pour répondre aux questions sur le potentiel des secteurs."""
        chunks = []

        # Créer un chunk par secteur majeur
        for _, row in recommendations.head(15).iterrows():
            sector = row['secteur']
            score = row['score_substitution']
            classification = row['classification']
            prod = row['production_mds_fcfa']
            imp = row['imports_mds_fcfa']
            exp = row['exports_mds_fcfa']
            growth = row.get('croissance_production_pct', 0)
            taux_couv = row.get('taux_couverture', 0)

            # Calculer le potentiel de marché
            potential = imp * (1 - min(taux_couv / 100, 1)) if taux_couv < 100 else 0

            sector_doc = f"""
POTENTIEL DU SECTEUR: {sector.upper()}
{'=' * (20 + len(sector))}
Keywords: potentiel {sector[:30]}, secteur {sector[:30]}, {sector[:30]} Burkina,
analyse {sector[:30]}, investir {sector[:30]}, sector potential, industry analysis

ÉVALUATION DU POTENTIEL:
- Score de substitution: {score}/100
- Classification: {classification}
- Priorité d'investissement: {'HAUTE' if score > 70 else 'MOYENNE' if score > 40 else 'BASSE'}

INDICATEURS CLÉS:
- Production actuelle: {prod:.1f} milliards FCFA
- Importations: {imp:.1f} milliards FCFA
- Exportations: {exp:.1f} milliards FCFA
- Balance commerciale: {exp - imp:+.1f} milliards FCFA
- Taux de couverture: {taux_couv:.1f}%
- Croissance production: {growth:+.1f}%

POTENTIEL DE MARCHÉ:
- Marché capturable (imports substituables): {potential:.1f} milliards FCFA
- Part de marché locale: {(prod / (prod + imp) * 100) if (prod + imp) > 0 else 0:.1f}%

ANALYSE:
"""

            if score > 70:
                sector_doc += f"""
Ce secteur présente un FORT POTENTIEL de substitution aux importations.
La production locale couvre déjà une part significative de la demande.
RECOMMANDATION: Investir massivement pour augmenter les capacités de production.
"""
            elif score > 40:
                sector_doc += f"""
Ce secteur présente un POTENTIEL MODÉRÉ de substitution.
Des opportunités existent mais nécessitent une analyse approfondie des niches.
RECOMMANDATION: Analyser les segments spécifiques à fort potentiel.
"""
            else:
                sector_doc += f"""
Ce secteur a un POTENTIEL LIMITÉ pour la substitution aux importations.
Le pays reste fortement dépendant des imports dans ce domaine.
RECOMMANDATION: Prudence requise. Étudier les causes de la dépendance.
"""

            chunk_id = hashlib.md5(f"sector_potential_{sector[:30]}".encode()).hexdigest()[:16]
            chunks.append(DocumentChunk(
                id=chunk_id,
                text=sector_doc.strip(),
                source=f"Sector_Potential_{sector[:25]}",
                source_type="sector_analysis",
                metadata={"sector": sector, "score": score, "type": "potential_analysis"}
            ))

        print(f"[OK] {len(chunks)} chunks crees pour l'analyse du potentiel des secteurs")
        return chunks

    def process_all_data(self) -> List[DocumentChunk]:
        """Traite toutes les données XGBoost et historiques."""
        print("[XGBOOST] Traitement des donnees XGBoost et historiques...")

        df, recommendations, metadata = self.load_data()

        # Créer les chunks de base
        sector_chunks = self.create_sector_descriptions(df, recommendations)
        summary_chunks = self.create_global_summary(df, recommendations, metadata)

        # Ajouter les données historiques par année
        yearly_commerce_chunks = self.create_yearly_commerce_data()
        yearly_detail_chunks = self.create_yearly_sector_details()

        # Ajouter documentation et guides
        methodology_chunks = self.create_methodology_documentation()
        best_sectors_chunks = self.create_best_sectors_chunk(recommendations)
        simulation_chunks = self.create_simulation_guide()
        progression_chunks = self.create_historical_progression(df)
        entrepreneur_chunks = self.create_entrepreneur_recommendations(recommendations)
        potential_chunks = self.create_sector_potential_queries(recommendations)

        all_chunks = (sector_chunks + summary_chunks + yearly_commerce_chunks +
                     yearly_detail_chunks + methodology_chunks + best_sectors_chunks +
                     simulation_chunks + progression_chunks + entrepreneur_chunks +
                     potential_chunks)

        print(f"[OK] {len(all_chunks)} chunks crees a partir des donnees XGBoost et historiques")
        return all_chunks


# ============================================================
# Embedding Manager
# ============================================================
class EmbeddingManager:
    """Gestionnaire d'embeddings utilisant all-MiniLM-L6-v2."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"[LOADING] Chargement du modele d'embedding: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = EMBEDDING_DIM
        print(f"[OK] Modele charge (dimension: {self.embedding_dim})")
    
    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """
        Encode une liste de textes en vecteurs d'embeddings.
        
        Args:
            texts: Liste de textes à encoder
            batch_size: Taille des batches pour l'encodage
            normalize: Normaliser les vecteurs pour FAISS (similarité cosinus)
        
        Returns:
            np.ndarray de forme (n_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True
        )
        
        if normalize:
            # Normalisation L2 pour la similarité cosinus avec FAISS IndexFlatIP
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-10)
        
        return embeddings.astype('float32')
    
    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode un seul texte."""
        return self.encode([text], normalize=normalize)[0]


# ============================================================
# FAISS Vector Store
# ============================================================
class FAISSVectorStore:
    """Stockage vectoriel utilisant FAISS avec IndexFlatIP (similarité cosinus)."""
    
    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        # IndexFlatIP pour la similarité cosinus (avec vecteurs normalisés)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks: List[DocumentChunk] = []
        self.id_to_idx: Dict[str, int] = {}
    
    def add_documents(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """
        Ajoute des documents à l'index.
        
        Args:
            chunks: Liste de DocumentChunk
            embeddings: Embeddings correspondants (normalisés)
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Le nombre de chunks et d'embeddings doit être identique")
        
        start_idx = len(self.chunks)
        
        # Ajouter à l'index FAISS
        self.index.add(embeddings)
        
        # Stocker les métadonnées
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.id_to_idx[chunk.id] = start_idx + i
        
        print(f"[OK] {len(chunks)} documents ajoutes a l'index (total: {self.index.ntotal})")
    
    def search(self, query_embedding: np.ndarray, top_k: int = TOP_K_RETRIEVAL) -> List[Tuple[DocumentChunk, float]]:
        """
        Recherche les documents les plus similaires.
        
        Args:
            query_embedding: Embedding de la requête (normalisé)
            top_k: Nombre de résultats à retourner
        
        Returns:
            Liste de tuples (chunk, score)
        """
        if self.index.ntotal == 0:
            return []
        
        # Reshape pour FAISS
        query = query_embedding.reshape(1, -1).astype('float32')
        
        # Recherche
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def save(self, save_dir: Path):
        """Sauvegarde l'index et les métadonnées."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder l'index FAISS
        faiss.write_index(self.index, str(save_dir / "faiss_index.bin"))
        
        # Sauvegarder les chunks
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        with open(save_dir / "chunks.json", 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        # Sauvegarder le mapping
        with open(save_dir / "id_mapping.json", 'w', encoding='utf-8') as f:
            json.dump(self.id_to_idx, f)
        
        print(f"[OK] Index sauvegarde dans {save_dir}")
    
    def load(self, save_dir: Path) -> bool:
        """Charge l'index et les métadonnées."""
        try:
            index_path = save_dir / "faiss_index.bin"
            chunks_path = save_dir / "chunks.json"
            mapping_path = save_dir / "id_mapping.json"
            
            if not all(p.exists() for p in [index_path, chunks_path, mapping_path]):
                return False
            
            # Charger l'index FAISS
            self.index = faiss.read_index(str(index_path))
            
            # Charger les chunks
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            self.chunks = [DocumentChunk.from_dict(d) for d in chunks_data]
            
            # Charger le mapping
            with open(mapping_path, 'r', encoding='utf-8') as f:
                self.id_to_idx = json.load(f)
            
            print(f"[OK] Index charge: {self.index.ntotal} documents")
            return True
        except Exception as e:
            print(f"[ERREUR] Erreur lors du chargement: {e}")
            return False


# ============================================================
# RAG System
# ============================================================
class RAGSystem:
    """Système RAG complet combinant PDF, données XGBoost et Groq LLM."""
    
    def __init__(self, groq_api_key: str, auto_load: bool = True):
        self.groq_client = Groq(api_key=groq_api_key)
        
        # Composants
        self.pdf_processor = PDFProcessor()
        self.xgboost_processor = XGBoostDataProcessor()
        self.embedding_manager = None  # Chargé à la demande
        self.vector_store = FAISSVectorStore()
        
        # État
        self.is_initialized = False
        
        # Essayer de charger l'index existant
        if auto_load:
            self._try_load_index()
    
    def _try_load_index(self) -> bool:
        """Essaie de charger un index existant."""
        if RAG_INDEX_DIR.exists():
            if self.vector_store.load(RAG_INDEX_DIR):
                self.is_initialized = True
                print("[OK] Index RAG existant charge avec succes")
                return True
        return False
    
    def _ensure_embedding_manager(self):
        """S'assure que le gestionnaire d'embeddings est chargé."""
        if self.embedding_manager is None:
            self.embedding_manager = EmbeddingManager()
    
    def build_knowledge_base(self, force_rebuild: bool = False):
        """
        Construit la base de connaissances à partir des PDFs et données XGBoost.
        
        Args:
            force_rebuild: Forcer la reconstruction même si un index existe
        """
        if self.is_initialized and not force_rebuild:
            print("[INFO] Index deja initialise. Utilisez force_rebuild=True pour reconstruire.")
            return
        
        print("\n" + "="*60)
        print("[BUILD] CONSTRUCTION DE LA BASE DE CONNAISSANCES RAG")
        print("="*60 + "\n")
        
        # Initialiser le gestionnaire d'embeddings
        self._ensure_embedding_manager()
        
        # Réinitialiser le vector store
        self.vector_store = FAISSVectorStore()
        
        all_chunks = []
        
        # 1. Traiter les PDFs
        if DOCUMENTS_DIR.exists():
            pdf_chunks = self.pdf_processor.process_all_pdfs(DOCUMENTS_DIR)
            all_chunks.extend(pdf_chunks)
        
        # 2. Traiter les données XGBoost
        try:
            xgboost_chunks = self.xgboost_processor.process_all_data()
            all_chunks.extend(xgboost_chunks)
        except Exception as e:
            print(f"[WARN] Erreur lors du traitement XGBoost: {e}")
        
        if not all_chunks:
            print("[ERREUR] Aucun document a indexer")
            return
        
        print(f"\n[TOTAL] Total chunks a indexer: {len(all_chunks)}")
        
        # 3. Créer les embeddings
        print("[EMBED] Creation des embeddings...")
        texts = [chunk.text for chunk in all_chunks]
        embeddings = self.embedding_manager.encode(texts)
        
        # 4. Ajouter à l'index
        self.vector_store.add_documents(all_chunks, embeddings)
        
        # 5. Sauvegarder
        self.vector_store.save(RAG_INDEX_DIR)
        
        self.is_initialized = True
        print("\n[OK] Base de connaissances RAG construite avec succes!")
        print(f"   - Documents indexes: {self.vector_store.index.ntotal}")
        print(f"   - Index sauvegarde dans: {RAG_INDEX_DIR}")
    
    def retrieve_context(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Tuple[DocumentChunk, float]]:
        """
        Récupère le contexte pertinent pour une requête avec détection d'année.

        Args:
            query: Question de l'utilisateur
            top_k: Nombre de documents à récupérer

        Returns:
            Liste de tuples (chunk, score de similarité)
        """
        import re

        if not self.is_initialized:
            print("[WARN] Index non initialise. Appeler build_knowledge_base() d'abord.")
            return []

        self._ensure_embedding_manager()

        # Détecter les années dans la query (2014-2025)
        years_in_query = re.findall(r'\b(201[4-9]|202[0-5])\b', query)

        # Si une année est demandée, chercher plus de documents pour trouver les chunks historiques
        search_top_k = max(top_k * 8, 50) if years_in_query else top_k * 2

        # Encoder la requête
        query_embedding = self.embedding_manager.encode_single(query)

        # Rechercher plus de documents
        all_results = self.vector_store.search(query_embedding, search_top_k)

        # Si des années sont demandées, réordonner les résultats
        if years_in_query:
            boosted_results = []
            other_results = []

            for chunk, score in all_results:
                # Vérifier si le chunk contient l'année demandée
                chunk_has_year = False
                for year in years_in_query:
                    if year in chunk.text or year in chunk.source:
                        chunk_has_year = True
                        break

                if chunk_has_year:
                    # Booster le score des chunks avec l'année
                    boosted_score = min(score + 0.15, 1.0)
                    boosted_results.append((chunk, boosted_score))
                else:
                    other_results.append((chunk, score))

            # Trier les résultats boostés par score
            boosted_results.sort(key=lambda x: x[1], reverse=True)
            other_results.sort(key=lambda x: x[1], reverse=True)

            # Combiner: d'abord les chunks avec l'année, puis les autres
            results = boosted_results + other_results
        else:
            results = all_results

        # Retourner le top_k final
        return results[:top_k]
    
    def format_context(self, retrieved_docs: List[Tuple[DocumentChunk, float]]) -> str:
        """Formate le contexte récupéré pour le LLM sans mentionner les sources."""
        if not retrieved_docs:
            return "Aucune information pertinente trouvée."
        
        context_parts = []
        for i, (chunk, score) in enumerate(retrieved_docs, 1):
            # Ne pas inclure les noms de documents/sources dans le contexte
            context_parts.append(f"""
--- Information {i} ---
{chunk.text}
""")
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, top_k: int = TOP_K_RETRIEVAL, 
                         temperature: float = 0.7, max_tokens: int = 1024) -> Dict[str, Any]:
        """
        Génère une réponse en utilisant le RAG.
        
        Args:
            query: Question de l'utilisateur
            top_k: Nombre de documents à récupérer
            temperature: Température pour la génération
            max_tokens: Nombre maximum de tokens
        
        Returns:
            Dictionnaire avec la réponse et les métadonnées
        """
        # 1. Récupérer le contexte
        retrieved_docs = self.retrieve_context(query, top_k)
        context = self.format_context(retrieved_docs)
        
        # 2. Construire le prompt - STRICTEMENT sans sources
        system_prompt = """Tu es un expert economiste specialise dans l'analyse du commerce international du Burkina Faso.

INTERDICTIONS ABSOLUES (A RESPECTER IMPERATIVEMENT):
- INTERDIT de dire "selon", "d'apres", "source", "document", "rapport", "page", "fichier", "PDF"
- INTERDIT de mettre des crochets [Source: ...] ou toute reference
- INTERDIT de citer des noms de fichiers comme "SCE_2T2025" ou similaires
- INTERDIT de mentionner "la note trimestrielle", "le bulletin", "le rapport"
- REPONDS comme si TU SAVAIS ces informations naturellement

FORMAT DE REPONSE:
- Donne les chiffres directement sans indiquer d'ou ils viennent
- Structure ta reponse avec des titres et listes si necessaire
- Sois precis avec les donnees numeriques
- Ne mentionne JAMAIS d'ou viennent les informations

DONNEES:
""" + context

        user_prompt = f"""{query}

RAPPEL: Reponds directement avec les faits, JAMAIS de mention de source, document, rapport ou fichier."""

        # 3. Appeler Groq LLM
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,  # Temperature tres basse pour eviter les hallucinations
                max_tokens=800,   # Limite raisonnable
                top_p=0.85        # Nucleus sampling pour coherence
            )
            
            answer = response.choices[0].message.content
            
            return {
                "success": True,
                "answer": answer,
                "sources": [
                    {
                        "source": chunk.source,
                        "source_type": chunk.source_type,
                        "score": score,
                        "excerpt": chunk.text[:200] + "..."
                    }
                    for chunk, score in retrieved_docs
                ],
                "query": query,
                "model": "llama-3.3-70b-versatile"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du système RAG."""
        if not self.is_initialized:
            return {"initialized": False}
        
        # Compter par type de source
        source_types = {}
        sources = {}
        
        for chunk in self.vector_store.chunks:
            source_types[chunk.source_type] = source_types.get(chunk.source_type, 0) + 1
            sources[chunk.source] = sources.get(chunk.source, 0) + 1
        
        return {
            "initialized": True,
            "total_documents": self.vector_store.index.ntotal,
            "embedding_dim": EMBEDDING_DIM,
            "embedding_model": EMBEDDING_MODEL,
            "source_types": source_types,
            "sources": sources,
            "index_path": str(RAG_INDEX_DIR)
        }


# ============================================================
# Fonctions utilitaires
# ============================================================
def initialize_rag_system(groq_api_key: str, force_rebuild: bool = False) -> RAGSystem:
    """
    Initialise le système RAG.
    
    Args:
        groq_api_key: Clé API Groq
        force_rebuild: Forcer la reconstruction de l'index
    
    Returns:
        Instance de RAGSystem initialisée
    """
    rag = RAGSystem(groq_api_key)
    
    if not rag.is_initialized or force_rebuild:
        rag.build_knowledge_base(force_rebuild=force_rebuild)
    
    return rag


# ============================================================
# Test et démonstration
# ============================================================
if __name__ == "__main__":
    import sys
    import io
    
    # Fix encoding for Windows
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    # Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
    
    print("="*60)
    print("[RAG] TEST DU SYSTEME RAG")
    print("="*60)
    
    # Initialiser
    rag = initialize_rag_system(GROQ_API_KEY, force_rebuild=True)
    
    # Afficher les stats
    stats = rag.get_stats()
    print("\n[STATS] Statistiques du systeme:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    # Test de requête
    test_queries = [
        "Quels sont les secteurs a fort potentiel de substitution aux importations?",
        "Quelle est la balance commerciale du Burkina Faso?",
        "Quelles sont les recommandations pour le secteur textile?"
    ]
    
    print("\n" + "="*60)
    print("[TEST] TESTS DE REQUETES")
    print("="*60)
    
    for query in test_queries:
        print(f"\n[QUESTION] {query}")
        print("-" * 50)
        
        result = rag.generate_response(query)
        
        if result["success"]:
            print(f"[REPONSE]\n{result['answer']}")
            print(f"\n[SOURCES] Sources utilisees:")
            for source in result["sources"]:
                print(f"  - {source['source']} ({source['source_type']}, score: {source['score']:.3f})")
        else:
            print(f"[ERREUR] {result['error']}")
        
        print()

