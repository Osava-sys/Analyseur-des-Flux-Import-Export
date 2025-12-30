# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered import/export trade analyzer for Burkina Faso (2014-2025). Identifies import substitution opportunities using XGBoost ML models and provides strategic recommendations via a RAG-enhanced chatbot.

## Commands

```bash
# Run the main Streamlit application
streamlit run app.py

# Run the REST API (FastAPI)
uvicorn api:app --reload

# Rebuild the RAG index after adding new PDFs to documents/
python rebuild_rag_index.py

# Update data and retrain ML models from PDF documents
python update_data_and_retrain.py

# Install dependencies
pip install -r requirements.txt
```

## Architecture

### Core Components

- **app.py**: Main Streamlit app with 7 modules (Dashboard, Real-time, Analysis, Recommendations, Simulator, ML Performance, AI Assistant)
- **api.py**: FastAPI REST endpoints for predictions using XGBoost models
- **rag_system.py**: RAG system using sentence-transformers (all-MiniLM-L6-v2) + FAISS for vector search + Groq LLM

### ML Pipeline

Models are stored in `models/` as pickle files:
- `xgb_regression_substitution.pkl`: Predicts substitution potential score
- `xgb_classification_opportunity.pkl`: Classifies opportunity level
- `scaler.pkl`: Feature scaler for normalization

Data flows: `data/raw/` → processing → `data/processed/dataset_ml_complete.csv` → training → `models/`

### RAG System

PDF documents in `documents/` are chunked and indexed in `rag_index/` (FAISS). The system combines:
1. Document retrieval via semantic search
2. XGBoost model data as context
3. Groq LLM for response generation

### Key Configuration

- Groq API key can be set via `GROQ_API_KEY` env var (fallback key is hardcoded)
- RAG config in `rag_system.py`: `CHUNK_SIZE=500`, `CHUNK_OVERLAP=100`, `TOP_K_RETRIEVAL=5`
- Embedding model: `all-MiniLM-L6-v2` (384 dimensions)

## Data Structure

Trade data uses Roman numeral sector codes (I-XXI) representing economic sectors. Key features: `production_fcfa`, `imports_fcfa`, `exports_fcfa`, `production_tonnes`, `imports_tonnes`, `exports_tonnes`.

## Language

Application interface is in French. Code comments and variable names mix French and English.
