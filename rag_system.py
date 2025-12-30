"""
Système RAG (Retrieval Augmented Generation) Amélioré pour l'Analyseur Import/Export
=====================================================================================
Version 2.0 avec:
- Chunking sémantique intelligent
- Hybrid Search (BM25 + dense embeddings)
- Reranking avec cross-encoder
- Multi-query retrieval
- Query expansion
- Cache intelligent
- Metadata filtering
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
from collections import defaultdict
import re
import time

# PDF Processing
from PyPDF2 import PdfReader

# Embeddings
from sentence_transformers import SentenceTransformer, CrossEncoder

# Vector Store
import faiss

# BM25 for hybrid search
from rank_bm25 import BM25Okapi

# LLM
from groq import Groq

# ============================================================
# Configuration
# ============================================================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_DIM = 384
CHUNK_SIZE = 500  # caractères par chunk
CHUNK_OVERLAP = 100  # chevauchement entre chunks
TOP_K_RETRIEVAL = 5  # nombre de documents à récupérer
TOP_K_INITIAL = 20  # nombre initial pour le reranking
CACHE_TTL = 3600  # durée de vie du cache en secondes
HYBRID_ALPHA = 0.7  # poids du dense retrieval vs BM25 (0.7 = 70% dense, 30% BM25)

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
# Cache Manager
# ============================================================
class QueryCache:
    """Cache intelligent pour les requêtes RAG avec TTL."""

    def __init__(self, ttl: int = CACHE_TTL, max_size: int = 100):
        self.ttl = ttl
        self.max_size = max_size
        self.cache: Dict[str, Tuple[Any, float]] = {}

    def _hash_query(self, query: str, params: Dict = None) -> str:
        """Génère un hash unique pour la requête."""
        content = query.lower().strip()
        if params:
            content += json.dumps(params, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, query: str, params: Dict = None) -> Optional[Any]:
        """Récupère une valeur du cache si elle existe et n'est pas expirée."""
        key = self._hash_query(query, params)
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None

    def set(self, query: str, value: Any, params: Dict = None):
        """Stocke une valeur dans le cache."""
        if len(self.cache) >= self.max_size:
            # Supprimer les entrées les plus anciennes
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

        key = self._hash_query(query, params)
        self.cache[key] = (value, time.time())

    def clear(self):
        """Vide le cache."""
        self.cache.clear()


# ============================================================
# PDF Processor Amélioré
# ============================================================
class PDFProcessor:
    """Processeur de documents PDF avec chunking sémantique intelligent."""

    # Patterns pour détecter les sections importantes
    SECTION_PATTERNS = [
        r'^#{1,3}\s+',  # Titres markdown
        r'^[IVX]+\.\s+',  # Numérotation romaine
        r'^\d+\.\d*\s+',  # Numérotation décimale
        r'^[A-Z][A-Z\s]{5,}$',  # Titres en majuscules
        r'^(CHAPITRE|SECTION|PARTIE|ANNEXE)\s+',  # Mots-clés de section
        r'^(Introduction|Conclusion|Résumé|Synthèse|Recommandations)',  # Sections communes
    ]

    # Mots-clés économiques pour enrichir les métadonnées
    ECONOMIC_KEYWORDS = {
        'commerce': ['import', 'export', 'balance', 'commerc', 'échange'],
        'production': ['production', 'produire', 'fabriqu', 'industri'],
        'agriculture': ['agricol', 'culture', 'récolte', 'céréale', 'coton', 'élevage'],
        'finance': ['fcfa', 'milliard', 'million', 'investis', 'financ', 'budget'],
        'substitution': ['substitut', 'remplac', 'local', 'nationale'],
        'secteur': ['secteur', 'filière', 'branche', 'activité'],
    }

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.section_patterns = [re.compile(p, re.MULTILINE | re.IGNORECASE)
                                  for p in self.SECTION_PATTERNS]

    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extrait le texte d'un fichier PDF avec métadonnées."""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            page_texts = []

            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    page_texts.append({
                        'page': page_num + 1,
                        'text': page_text,
                        'char_count': len(page_text)
                    })
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"

            metadata = {
                'total_pages': len(reader.pages),
                'total_chars': len(text),
                'pages_info': page_texts
            }

            return text.strip(), metadata
        except Exception as e:
            print(f"Erreur lors de la lecture de {pdf_path}: {e}")
            return "", {}

    def clean_text(self, text: str) -> str:
        """Nettoie le texte extrait."""
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text)
        # Supprimer les caractères spéciaux problématiques mais garder les accents
        text = re.sub(r'[^\w\s\.,;:!?\-\(\)\[\]\'\"€%°/àâäéèêëïîôùûüçÀÂÄÉÈÊËÏÎÔÙÛÜÇ]', '', text)
        return text.strip()

    def _detect_section_type(self, text: str) -> Optional[str]:
        """Détecte si le texte commence une nouvelle section."""
        for pattern in self.section_patterns:
            if pattern.search(text[:100]):
                return "section_header"
        return None

    def _extract_keywords(self, text: str) -> List[str]:
        """Extrait les mots-clés économiques du texte."""
        text_lower = text.lower()
        found_keywords = []
        for category, keywords in self.ECONOMIC_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append(category)
                    break
        return list(set(found_keywords))

    def _extract_numbers(self, text: str) -> Dict[str, List[float]]:
        """Extrait les valeurs numériques significatives du texte."""
        numbers = {
            'milliards': [],
            'millions': [],
            'percentages': [],
            'years': []
        }

        # Milliards FCFA
        for match in re.finditer(r'([\d\s,\.]+)\s*(?:milliards?|mds?)\s*(?:de\s*)?(?:fcfa)?', text, re.I):
            try:
                val = float(match.group(1).replace(' ', '').replace(',', '.'))
                numbers['milliards'].append(val)
            except:
                pass

        # Millions FCFA
        for match in re.finditer(r'([\d\s,\.]+)\s*millions?\s*(?:de\s*)?(?:fcfa)?', text, re.I):
            try:
                val = float(match.group(1).replace(' ', '').replace(',', '.'))
                numbers['millions'].append(val)
            except:
                pass

        # Pourcentages
        for match in re.finditer(r'([\d,\.]+)\s*%', text):
            try:
                val = float(match.group(1).replace(',', '.'))
                numbers['percentages'].append(val)
            except:
                pass

        # Années (2000-2030)
        for match in re.finditer(r'\b(20[0-3]\d)\b', text):
            numbers['years'].append(int(match.group(1)))

        return {k: v for k, v in numbers.items() if v}

    def create_chunks_semantic(self, text: str, source: str, pdf_metadata: Dict = None) -> List[DocumentChunk]:
        """Découpe le texte en chunks sémantiques intelligents."""
        chunks = []
        text = self.clean_text(text)

        if not text:
            return chunks

        # Découpage par paragraphes d'abord, puis par phrases
        paragraphs = re.split(r'\n\s*\n|\. {2,}', text)

        current_chunk = ""
        current_sentences = []
        chunk_idx = 0

        for para in paragraphs:
            sentences = re.split(r'(?<=[.!?])\s+', para)

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Vérifier si c'est un début de section
                is_section = self._detect_section_type(sentence)

                # Si c'est une nouvelle section et qu'on a du contenu, créer un chunk
                if is_section and current_chunk and len(current_chunk) > 100:
                    chunk = self._create_chunk(
                        current_chunk, source, chunk_idx,
                        current_sentences, pdf_metadata
                    )
                    chunks.append(chunk)
                    chunk_idx += 1

                    # Overlap: garder les dernières phrases
                    overlap_sentences = current_sentences[-2:] if len(current_sentences) > 2 else []
                    current_chunk = " ".join(overlap_sentences) + " " if overlap_sentences else ""
                    current_sentences = overlap_sentences.copy()

                # Ajouter la phrase au chunk courant
                if len(current_chunk) + len(sentence) <= self.chunk_size:
                    current_chunk += sentence + " "
                    current_sentences.append(sentence)
                else:
                    if current_chunk:
                        chunk = self._create_chunk(
                            current_chunk, source, chunk_idx,
                            current_sentences, pdf_metadata
                        )
                        chunks.append(chunk)
                        chunk_idx += 1

                        # Overlap sémantique
                        overlap_sentences = current_sentences[-2:] if len(current_sentences) > 2 else []
                        current_chunk = " ".join(overlap_sentences) + " " + sentence + " "
                        current_sentences = overlap_sentences + [sentence]
                    else:
                        current_chunk = sentence + " "
                        current_sentences = [sentence]

        # Dernier chunk
        if current_chunk.strip() and len(current_chunk.strip()) > 50:
            chunk = self._create_chunk(
                current_chunk, source, chunk_idx,
                current_sentences, pdf_metadata
            )
            chunks.append(chunk)

        return chunks

    def _create_chunk(self, text: str, source: str, idx: int,
                      sentences: List[str], pdf_metadata: Dict = None) -> DocumentChunk:
        """Crée un DocumentChunk avec métadonnées enrichies."""
        text = text.strip()
        chunk_id = self._generate_chunk_id(source, idx)

        # Extraire les métadonnées du contenu
        keywords = self._extract_keywords(text)
        numbers = self._extract_numbers(text)

        metadata = {
            "chunk_index": idx,
            "char_count": len(text),
            "sentence_count": len(sentences),
            "extraction_date": datetime.now().isoformat(),
            "keywords": keywords,
            "numbers": numbers,
        }

        if pdf_metadata:
            metadata["pdf_pages"] = pdf_metadata.get('total_pages', 0)

        return DocumentChunk(
            id=chunk_id,
            text=text,
            source=source,
            source_type="pdf",
            metadata=metadata
        )

    def create_chunks(self, text: str, source: str) -> List[DocumentChunk]:
        """Méthode de compatibilité - utilise le chunking sémantique."""
        return self.create_chunks_semantic(text, source)

    def _generate_chunk_id(self, source: str, idx: int) -> str:
        """Génère un ID unique pour un chunk."""
        content = f"{source}_{idx}_{datetime.now().timestamp()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def process_all_pdfs(self, documents_dir: Path) -> List[DocumentChunk]:
        """Traite tous les PDFs d'un répertoire avec chunking sémantique."""
        all_chunks = []
        pdf_files = list(documents_dir.glob("*.pdf"))

        print(f"[PDF] Traitement de {len(pdf_files)} fichiers PDF...")

        for pdf_path in pdf_files:
            print(f"  [DOC] Traitement: {pdf_path.name}")
            text, pdf_metadata = self.extract_text_from_pdf(pdf_path)
            if text:
                chunks = self.create_chunks_semantic(text, pdf_path.name, pdf_metadata)
                all_chunks.extend(chunks)
                print(f"     -> {len(chunks)} chunks crees (semantique)")

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
    
    def process_all_data(self) -> List[DocumentChunk]:
        """Traite toutes les données XGBoost."""
        print("[XGBOOST] Traitement des donnees XGBoost...")
        
        df, recommendations, metadata = self.load_data()
        
        # Créer les chunks
        sector_chunks = self.create_sector_descriptions(df, recommendations)
        summary_chunks = self.create_global_summary(df, recommendations, metadata)
        
        all_chunks = sector_chunks + summary_chunks
        
        print(f"[OK] {len(all_chunks)} chunks crees a partir des donnees XGBoost")
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
# BM25 Index pour Hybrid Search
# ============================================================
class BM25Index:
    """Index BM25 pour la recherche lexicale sparse."""

    def __init__(self):
        self.bm25 = None
        self.tokenized_corpus = []
        self.chunk_ids = []

    def _tokenize(self, text: str) -> List[str]:
        """Tokenise un texte pour BM25."""
        # Tokenisation simple mais efficace pour le français
        text = text.lower()
        # Supprimer la ponctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        # Tokeniser et filtrer les mots courts
        tokens = [t for t in text.split() if len(t) > 2]
        return tokens

    def build(self, chunks: List[DocumentChunk]):
        """Construit l'index BM25 à partir des chunks."""
        self.tokenized_corpus = [self._tokenize(chunk.text) for chunk in chunks]
        self.chunk_ids = [chunk.id for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"[BM25] Index construit avec {len(chunks)} documents")

    def search(self, query: str, top_k: int = TOP_K_INITIAL) -> List[Tuple[str, float]]:
        """Recherche BM25."""
        if self.bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Récupérer les top_k meilleurs scores
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.chunk_ids[idx], float(scores[idx])))

        return results

    def save(self, save_dir: Path):
        """Sauvegarde l'index BM25."""
        bm25_data = {
            'tokenized_corpus': self.tokenized_corpus,
            'chunk_ids': self.chunk_ids
        }
        with open(save_dir / "bm25_index.json", 'w', encoding='utf-8') as f:
            json.dump(bm25_data, f, ensure_ascii=False)

    def load(self, save_dir: Path) -> bool:
        """Charge l'index BM25."""
        try:
            bm25_path = save_dir / "bm25_index.json"
            if not bm25_path.exists():
                return False

            with open(bm25_path, 'r', encoding='utf-8') as f:
                bm25_data = json.load(f)

            self.tokenized_corpus = bm25_data['tokenized_corpus']
            self.chunk_ids = bm25_data['chunk_ids']
            self.bm25 = BM25Okapi(self.tokenized_corpus)

            print(f"[BM25] Index charge: {len(self.chunk_ids)} documents")
            return True
        except Exception as e:
            print(f"[ERREUR] Erreur chargement BM25: {e}")
            return False


# ============================================================
# FAISS Vector Store Amélioré avec Hybrid Search
# ============================================================
class FAISSVectorStore:
    """Stockage vectoriel hybride FAISS + BM25 avec reranking."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        # IndexFlatIP pour la similarité cosinus (avec vecteurs normalisés)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks: List[DocumentChunk] = []
        self.id_to_idx: Dict[str, int] = {}

        # BM25 pour hybrid search
        self.bm25_index = BM25Index()

        # Index inversé par métadonnées
        self.keyword_index: Dict[str, List[str]] = defaultdict(list)  # keyword -> chunk_ids
        self.source_index: Dict[str, List[str]] = defaultdict(list)   # source -> chunk_ids

    def add_documents(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """
        Ajoute des documents à l'index avec indexation hybride.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Le nombre de chunks et d'embeddings doit être identique")

        start_idx = len(self.chunks)

        # Ajouter à l'index FAISS
        self.index.add(embeddings)

        # Stocker les métadonnées et construire les index
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.id_to_idx[chunk.id] = start_idx + i

            # Index par mots-clés
            keywords = chunk.metadata.get('keywords', [])
            for kw in keywords:
                self.keyword_index[kw].append(chunk.id)

            # Index par source
            self.source_index[chunk.source].append(chunk.id)

        # Construire l'index BM25
        self.bm25_index.build(self.chunks)

        print(f"[OK] {len(chunks)} documents ajoutes a l'index hybride (total: {self.index.ntotal})")

    def search_dense(self, query_embedding: np.ndarray, top_k: int = TOP_K_INITIAL) -> List[Tuple[str, float]]:
        """Recherche dense avec FAISS."""
        if self.index.ntotal == 0:
            return []

        query = query_embedding.reshape(1, -1).astype('float32')
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx].id, float(score)))

        return results

    def search_hybrid(self, query_embedding: np.ndarray, query_text: str,
                      top_k: int = TOP_K_RETRIEVAL, alpha: float = HYBRID_ALPHA) -> List[Tuple[DocumentChunk, float]]:
        """
        Recherche hybride combinant dense (FAISS) et sparse (BM25).

        Args:
            query_embedding: Embedding de la requête
            query_text: Texte de la requête pour BM25
            top_k: Nombre de résultats finaux
            alpha: Poids du dense (1-alpha pour BM25)
        """
        # Recherche dense
        dense_results = self.search_dense(query_embedding, TOP_K_INITIAL)

        # Recherche BM25
        bm25_results = self.bm25_index.search(query_text, TOP_K_INITIAL)

        # Normaliser les scores
        dense_scores = {chunk_id: score for chunk_id, score in dense_results}
        bm25_scores = {chunk_id: score for chunk_id, score in bm25_results}

        # Normalisation min-max
        if dense_scores:
            dense_max = max(dense_scores.values())
            dense_min = min(dense_scores.values())
            dense_range = dense_max - dense_min if dense_max != dense_min else 1
            dense_scores = {k: (v - dense_min) / dense_range for k, v in dense_scores.items()}

        if bm25_scores:
            bm25_max = max(bm25_scores.values())
            bm25_min = min(bm25_scores.values())
            bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1
            bm25_scores = {k: (v - bm25_min) / bm25_range for k, v in bm25_scores.items()}

        # Combiner les scores
        all_chunk_ids = set(dense_scores.keys()) | set(bm25_scores.keys())
        combined_scores = {}

        for chunk_id in all_chunk_ids:
            dense_score = dense_scores.get(chunk_id, 0)
            bm25_score = bm25_scores.get(chunk_id, 0)
            combined_scores[chunk_id] = alpha * dense_score + (1 - alpha) * bm25_score

        # Trier par score combiné
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

        # Récupérer les chunks
        results = []
        for chunk_id in sorted_ids[:top_k]:
            idx = self.id_to_idx.get(chunk_id)
            if idx is not None and idx < len(self.chunks):
                results.append((self.chunks[idx], combined_scores[chunk_id]))

        return results

    def search(self, query_embedding: np.ndarray, top_k: int = TOP_K_RETRIEVAL) -> List[Tuple[DocumentChunk, float]]:
        """Recherche standard (compatible avec l'ancienne API)."""
        if self.index.ntotal == 0:
            return []

        query = query_embedding.reshape(1, -1).astype('float32')
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results

    def filter_by_keywords(self, keywords: List[str]) -> List[DocumentChunk]:
        """Filtre les chunks par mots-clés."""
        matching_ids = set()
        for kw in keywords:
            matching_ids.update(self.keyword_index.get(kw, []))

        return [self.chunks[self.id_to_idx[cid]] for cid in matching_ids if cid in self.id_to_idx]

    def filter_by_source(self, sources: List[str]) -> List[DocumentChunk]:
        """Filtre les chunks par source."""
        matching_ids = set()
        for src in sources:
            matching_ids.update(self.source_index.get(src, []))

        return [self.chunks[self.id_to_idx[cid]] for cid in matching_ids if cid in self.id_to_idx]

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

        # Sauvegarder les index additionnels
        with open(save_dir / "keyword_index.json", 'w', encoding='utf-8') as f:
            json.dump(dict(self.keyword_index), f, ensure_ascii=False)

        with open(save_dir / "source_index.json", 'w', encoding='utf-8') as f:
            json.dump(dict(self.source_index), f, ensure_ascii=False)

        # Sauvegarder BM25
        self.bm25_index.save(save_dir)

        print(f"[OK] Index hybride sauvegarde dans {save_dir}")

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

            # Charger les index additionnels (si présents)
            keyword_path = save_dir / "keyword_index.json"
            if keyword_path.exists():
                with open(keyword_path, 'r', encoding='utf-8') as f:
                    self.keyword_index = defaultdict(list, json.load(f))

            source_path = save_dir / "source_index.json"
            if source_path.exists():
                with open(source_path, 'r', encoding='utf-8') as f:
                    self.source_index = defaultdict(list, json.load(f))

            # Charger BM25
            self.bm25_index.load(save_dir)

            print(f"[OK] Index hybride charge: {self.index.ntotal} documents")
            return True
        except Exception as e:
            print(f"[ERREUR] Erreur lors du chargement: {e}")
            return False


# ============================================================
# Query Expansion et Multi-Query
# ============================================================
class QueryExpander:
    """Expansion et reformulation de requêtes pour améliorer le recall."""

    # Synonymes économiques français
    SYNONYMS = {
        'import': ['importation', 'achat', 'entrée'],
        'export': ['exportation', 'vente', 'sortie'],
        'production': ['fabrication', 'manufacture', 'industrie'],
        'commerce': ['échange', 'transaction', 'négoce'],
        'substitution': ['remplacement', 'alternative', 'local'],
        'secteur': ['domaine', 'filière', 'branche'],
        'agriculture': ['agricole', 'céréales', 'culture'],
        'balance': ['solde', 'différence', 'déficit', 'excédent'],
        'croissance': ['augmentation', 'hausse', 'progression'],
        'baisse': ['diminution', 'réduction', 'déclin'],
        'burkina': ['burkina faso', 'burkinabè'],
        'fcfa': ['francs cfa', 'monnaie', 'devise'],
    }

    # Templates pour générer des sous-requêtes
    QUERY_TEMPLATES = {
        'secteur': [
            "statistiques {sector} Burkina Faso",
            "production {sector}",
            "importations {sector}",
            "exportations {sector}",
        ],
        'temporel': [
            "{query} 2021",
            "{query} 2022",
            "{query} 2023",
            "{query} évolution",
            "{query} tendance",
        ],
        'economique': [
            "{query} valeur FCFA",
            "{query} volume tonnes",
            "{query} croissance",
        ]
    }

    def expand_query(self, query: str) -> List[str]:
        """Expanse une requête avec des synonymes."""
        expanded = [query]
        query_lower = query.lower()

        for term, synonyms in self.SYNONYMS.items():
            if term in query_lower:
                for syn in synonyms[:2]:  # Limiter à 2 synonymes
                    expanded.append(query_lower.replace(term, syn))

        return list(set(expanded))[:5]  # Max 5 variations

    def generate_sub_queries(self, query: str) -> List[str]:
        """Génère des sous-requêtes pour le multi-query retrieval."""
        sub_queries = [query]

        # Détecter le type de requête
        query_lower = query.lower()

        # Requêtes sur les secteurs
        sectors = ['textile', 'agro', 'minier', 'énergie', 'agriculture',
                   'coton', 'or', 'céréales', 'élevage', 'industrie']
        detected_sector = None
        for sector in sectors:
            if sector in query_lower:
                detected_sector = sector
                break

        if detected_sector:
            for template in self.QUERY_TEMPLATES['secteur'][:2]:
                sub_queries.append(template.format(sector=detected_sector))

        # Ajouter des variations temporelles pour les requêtes de données
        if any(word in query_lower for word in ['évolution', 'tendance', 'historique', 'données']):
            for template in self.QUERY_TEMPLATES['temporel'][:2]:
                sub_queries.append(template.format(query=query))

        # Requêtes économiques
        if any(word in query_lower for word in ['valeur', 'montant', 'chiffre', 'statistique']):
            for template in self.QUERY_TEMPLATES['economique'][:2]:
                sub_queries.append(template.format(query=query))

        return list(set(sub_queries))[:6]  # Max 6 sous-requêtes


# ============================================================
# Reranker avec Cross-Encoder
# ============================================================
class Reranker:
    """Reranking des résultats avec un cross-encoder."""

    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model = None
        self.model_name = model_name
        self._loaded = False

    def _ensure_loaded(self):
        """Charge le modèle à la demande."""
        if not self._loaded:
            print(f"[RERANKER] Chargement du modele: {self.model_name}...")
            try:
                self.model = CrossEncoder(self.model_name)
                self._loaded = True
                print("[RERANKER] Modele charge avec succes")
            except Exception as e:
                print(f"[WARN] Impossible de charger le reranker: {e}")
                self._loaded = False

    def rerank(self, query: str, documents: List[Tuple[DocumentChunk, float]],
               top_k: int = TOP_K_RETRIEVAL) -> List[Tuple[DocumentChunk, float]]:
        """
        Rerank les documents avec le cross-encoder.

        Args:
            query: Requête utilisateur
            documents: Liste de tuples (chunk, score initial)
            top_k: Nombre de résultats à retourner
        """
        if not documents:
            return []

        self._ensure_loaded()

        if not self._loaded or self.model is None:
            # Fallback: retourner les documents triés par score initial
            return sorted(documents, key=lambda x: x[1], reverse=True)[:top_k]

        # Préparer les paires (query, document)
        pairs = [(query, chunk.text) for chunk, _ in documents]

        # Calculer les scores de pertinence
        try:
            scores = self.model.predict(pairs)

            # Combiner avec les résultats
            reranked = list(zip([d[0] for d in documents], scores))

            # Trier par score du reranker
            reranked.sort(key=lambda x: x[1], reverse=True)

            return reranked[:top_k]
        except Exception as e:
            print(f"[WARN] Erreur lors du reranking: {e}")
            return sorted(documents, key=lambda x: x[1], reverse=True)[:top_k]


# ============================================================
# RAG System Amélioré
# ============================================================
class RAGSystem:
    """Système RAG amélioré avec hybrid search, reranking et multi-query."""

    def __init__(self, groq_api_key: str, auto_load: bool = True,
                 enable_reranking: bool = True, enable_cache: bool = True):
        self.groq_client = Groq(api_key=groq_api_key)

        # Composants de base
        self.pdf_processor = PDFProcessor()
        self.xgboost_processor = XGBoostDataProcessor()
        self.embedding_manager = None  # Chargé à la demande
        self.vector_store = FAISSVectorStore()

        # Composants avancés
        self.query_expander = QueryExpander()
        self.reranker = Reranker() if enable_reranking else None
        self.cache = QueryCache() if enable_cache else None

        # Configuration
        self.enable_reranking = enable_reranking
        self.enable_cache = enable_cache
        self.enable_hybrid_search = True
        self.enable_multi_query = True

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
    
    def retrieve_context(self, query: str, top_k: int = TOP_K_RETRIEVAL,
                          use_hybrid: bool = None, use_reranking: bool = None,
                          use_multi_query: bool = None) -> List[Tuple[DocumentChunk, float]]:
        """
        Récupère le contexte pertinent pour une requête avec retrieval avancé.

        Args:
            query: Question de l'utilisateur
            top_k: Nombre de documents à récupérer
            use_hybrid: Utiliser la recherche hybride (défaut: self.enable_hybrid_search)
            use_reranking: Utiliser le reranking (défaut: self.enable_reranking)
            use_multi_query: Utiliser le multi-query (défaut: self.enable_multi_query)

        Returns:
            Liste de tuples (chunk, score de similarité)
        """
        if not self.is_initialized:
            print("[WARN] Index non initialise. Appeler build_knowledge_base() d'abord.")
            return []

        # Paramètres par défaut
        use_hybrid = use_hybrid if use_hybrid is not None else self.enable_hybrid_search
        use_reranking = use_reranking if use_reranking is not None else self.enable_reranking
        use_multi_query = use_multi_query if use_multi_query is not None else self.enable_multi_query

        # Vérifier le cache
        if self.cache:
            cache_params = {
                'top_k': top_k,
                'hybrid': use_hybrid,
                'reranking': use_reranking,
                'multi_query': use_multi_query
            }
            cached_result = self.cache.get(query, cache_params)
            if cached_result is not None:
                return cached_result

        self._ensure_embedding_manager()

        # Multi-query retrieval
        if use_multi_query:
            results = self._multi_query_retrieve(query, top_k * 2, use_hybrid)
        else:
            results = self._single_query_retrieve(query, top_k * 2, use_hybrid)

        # Reranking
        if use_reranking and self.reranker and len(results) > 0:
            results = self.reranker.rerank(query, results, top_k)
        else:
            results = results[:top_k]

        # Mettre en cache
        if self.cache:
            self.cache.set(query, results, cache_params)

        return results

    def _single_query_retrieve(self, query: str, top_k: int,
                                use_hybrid: bool) -> List[Tuple[DocumentChunk, float]]:
        """Retrieval avec une seule requête."""
        query_embedding = self.embedding_manager.encode_single(query)

        if use_hybrid:
            return self.vector_store.search_hybrid(query_embedding, query, top_k)
        else:
            return self.vector_store.search(query_embedding, top_k)

    def _multi_query_retrieve(self, query: str, top_k: int,
                               use_hybrid: bool) -> List[Tuple[DocumentChunk, float]]:
        """Multi-query retrieval avec fusion des résultats."""
        # Générer les sous-requêtes
        sub_queries = self.query_expander.generate_sub_queries(query)

        # Collecter les résultats de toutes les requêtes
        all_results: Dict[str, Tuple[DocumentChunk, float, int]] = {}  # id -> (chunk, max_score, count)

        for i, sub_query in enumerate(sub_queries):
            query_embedding = self.embedding_manager.encode_single(sub_query)

            if use_hybrid:
                results = self.vector_store.search_hybrid(query_embedding, sub_query, top_k // 2)
            else:
                results = self.vector_store.search(query_embedding, top_k // 2)

            for chunk, score in results:
                if chunk.id in all_results:
                    # Reciprocal Rank Fusion (RRF)
                    existing = all_results[chunk.id]
                    new_score = max(existing[1], score)
                    new_count = existing[2] + 1
                    all_results[chunk.id] = (chunk, new_score, new_count)
                else:
                    all_results[chunk.id] = (chunk, score, 1)

        # Calculer le score final avec boost pour les résultats trouvés par plusieurs requêtes
        final_results = []
        for chunk_id, (chunk, score, count) in all_results.items():
            # Boost basé sur le nombre de requêtes qui ont trouvé ce document
            boosted_score = score * (1 + 0.1 * (count - 1))
            final_results.append((chunk, boosted_score))

        # Trier par score
        final_results.sort(key=lambda x: x[1], reverse=True)

        return final_results[:top_k]

    def retrieve_with_filters(self, query: str, top_k: int = TOP_K_RETRIEVAL,
                               keywords: List[str] = None,
                               sources: List[str] = None) -> List[Tuple[DocumentChunk, float]]:
        """
        Récupère le contexte avec filtrage par métadonnées.

        Args:
            query: Question de l'utilisateur
            top_k: Nombre de documents à récupérer
            keywords: Filtrer par mots-clés économiques
            sources: Filtrer par sources (noms de fichiers)
        """
        # D'abord, appliquer les filtres pour obtenir les candidats
        if keywords:
            candidates = set(c.id for c in self.vector_store.filter_by_keywords(keywords))
        else:
            candidates = None

        if sources:
            source_chunks = set(c.id for c in self.vector_store.filter_by_source(sources))
            if candidates is not None:
                candidates = candidates & source_chunks
            else:
                candidates = source_chunks

        # Ensuite, faire le retrieval normal
        results = self.retrieve_context(query, top_k * 2)

        # Filtrer les résultats si des filtres sont actifs
        if candidates is not None:
            results = [(chunk, score) for chunk, score in results if chunk.id in candidates]

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
        system_prompt = """Tu es un expert économiste spécialisé dans l'analyse du commerce international du Burkina Faso.

RÈGLES OBLIGATOIRES:
1. Réponds UNIQUEMENT en français correct avec une orthographe impeccable
2. Utilise tous les accents appropriés (é, è, ê, à, ù, ç, etc.)
3. Ne mentionne JAMAIS de sources, documents, rapports ou fichiers
4. Réponds comme si tu connaissais ces informations naturellement

FORMAT DE RÉPONSE:
- Donne les chiffres directement et précisément
- Structure ta réponse avec des titres et listes si nécessaire
- Sois précis avec les données numériques (milliards FCFA, pourcentages, etc.)
- Utilise un français professionnel et soigné

DONNÉES DISPONIBLES:
""" + context

        user_prompt = f"""{query}

IMPORTANT: Réponds en français correct avec une orthographe parfaite. Ne mentionne aucune source."""

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
            import re
            
            error_str = str(e)
            error_type = "unknown"
            wait_time = None
            error_message = "Une erreur est survenue lors de la génération de la réponse."
            
            # Détecter les erreurs de rate limit (429) - vérifier d'abord le code d'erreur
            is_rate_limit = (
                "429" in error_str or 
                "rate_limit" in error_str.lower() or 
                "Rate limit" in error_str or
                "rate_limit_exceeded" in error_str.lower()
            )
            
            if is_rate_limit:
                error_type = "rate_limit"
                
                # Extraire directement les informations depuis le message d'erreur complet
                # Format typique: "Limit 100000, Used 98830, Requested 2531. Please try again in 19m35.904s"
                
                # 1. Extraire le temps d'attente (format: "19m35.904s")
                wait_match = re.search(r'(\d+)m(\d+\.?\d*)s', error_str)
                if wait_match:
                    minutes = int(wait_match.group(1))
                    seconds = float(wait_match.group(2))
                    wait_time = minutes * 60 + seconds
                
                # 2. Extraire les informations de limite (format: "Limit 100000, Used 98830")
                limit_match = re.search(r'Limit\s+(\d+),\s+Used\s+(\d+)', error_str)
                if limit_match:
                    limit = int(limit_match.group(1))
                    used = int(limit_match.group(2))
                    # Formater avec des séparateurs de milliers
                    limit_formatted = f"{limit:,}".replace(",", " ")
                    used_formatted = f"{used:,}".replace(",", " ")
                    error_message = f"Limite quotidienne de tokens atteinte ({used_formatted}/{limit_formatted} tokens utilisés)."
                else:
                    # Si on ne trouve pas les détails, message générique mais clair
                    error_message = "Limite quotidienne de tokens atteinte pour l'API Groq."
            
            return {
                "success": False,
                "error": error_message,
                "error_type": error_type,
                "error_details": error_str,
                "wait_time": wait_time,
                "query": query
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du système RAG amélioré."""
        if not self.is_initialized:
            return {"initialized": False}

        # Compter par type de source
        source_types = {}
        sources = {}
        keywords_count = {}

        for chunk in self.vector_store.chunks:
            source_types[chunk.source_type] = source_types.get(chunk.source_type, 0) + 1
            sources[chunk.source] = sources.get(chunk.source, 0) + 1

            # Compter les mots-clés
            for kw in chunk.metadata.get('keywords', []):
                keywords_count[kw] = keywords_count.get(kw, 0) + 1

        # Statistiques du cache
        cache_stats = None
        if self.cache:
            cache_stats = {
                "size": len(self.cache.cache),
                "max_size": self.cache.max_size,
                "ttl": self.cache.ttl
            }

        return {
            "initialized": True,
            "version": "2.0",
            "total_documents": self.vector_store.index.ntotal,
            "embedding_dim": EMBEDDING_DIM,
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model": RERANKER_MODEL if self.enable_reranking else None,
            "source_types": source_types,
            "sources": sources,
            "keywords_distribution": keywords_count,
            "index_path": str(RAG_INDEX_DIR),
            "features": {
                "hybrid_search": self.enable_hybrid_search,
                "reranking": self.enable_reranking,
                "cache": self.enable_cache,
                "multi_query": self.enable_multi_query,
                "bm25_enabled": self.vector_store.bm25_index.bm25 is not None
            },
            "cache_stats": cache_stats,
            "config": {
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "top_k_retrieval": TOP_K_RETRIEVAL,
                "top_k_initial": TOP_K_INITIAL,
                "hybrid_alpha": HYBRID_ALPHA
            }
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
    
    # Configuration - utiliser variable d'environnement
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
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

