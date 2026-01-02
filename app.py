"""
Analyseur de Flux Import/Export - Burkina Faso
====================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import json
from pathlib import Path
import base64
import os

# LLM et PDF
from groq import Groq
from PyPDF2 import PdfReader

# Système RAG
try:
    from rag_system import RAGSystem, initialize_rag_system
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("⚠️ Système RAG non disponible - Installation requise: pip install sentence-transformers faiss-cpu")

# ============================================================
# Configuration Groq LLM
# ============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# Dossier des documents
DOCUMENTS_DIR = Path(__file__).parent / "documents"
DOCUMENTS_DIR.mkdir(exist_ok=True)

# ============================================================
# Initialisation du système RAG
# ============================================================
@st.cache_resource
def get_rag_system():
    """Initialise le système RAG (mise en cache pour performance)."""
    if not RAG_AVAILABLE:
        return None
    try:
        rag = initialize_rag_system(GROQ_API_KEY, force_rebuild=False)
        return rag
    except Exception as e:
        print(f"Erreur initialisation RAG: {e}")
        return None

# ============================================================
# Fonctions pour les documents PDF (fallback si RAG non disponible)
# ============================================================
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Erreur lecture PDF: {str(e)}"

def get_all_documents_context():
    """Extrait le contexte des documents sans mentionner les noms de fichiers."""
    documents_text = ""
    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        return ""
    for i, pdf_file in enumerate(pdf_files[:3], 1):
        text = extract_text_from_pdf(pdf_file)
        if text and not text.startswith("Erreur"):
            text = ' '.join(text.split())
            # Ne pas inclure le nom du fichier, juste le contenu
            documents_text += f"\n{text[:800]}..."
    return documents_text

# ============================================================
# Configuration de la page
# ============================================================
st.set_page_config(
    page_title="Analyseur Import/Export - Burkina Faso",
    page_icon="BF",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Initialisation du thème
# ============================================================
if 'theme' not in st.session_state:
    st.session_state.theme = "dark"

if 'page' not in st.session_state:
    st.session_state.page = "Accueil"

# ============================================================
# Définition des thèmes 
# ============================================================
def get_theme_colors(theme):
    if theme == "dark":
        return {
            "bg_primary": "#1a1a2e",
            "bg_secondary": "#16213e",
            "bg_card": "#1f2937",
            "bg_sidebar": "#111827",
            "text_primary": "#f3f4f6",
            "text_secondary": "#9ca3af",
            "text_muted": "#6b7280",
            "border": "#374151",
            "accent_blue": "#4285f4",
            "accent_orange": "#ea8600",
            "accent_green": "#34a853",
            "accent_red": "#ea4335",
            "accent_yellow": "#fbbc04",
            "chart_bg": "rgba(0,0,0,0)",
            "hover": "#374151",
        }
    else:
        return {
            "bg_primary": "#f9fafb",
            "bg_secondary": "#f3f4f6",
            "bg_card": "#ffffff",
            "bg_sidebar": "#ffffff",
            "text_primary": "#111827",
            "text_secondary": "#374151",
            "text_muted": "#6b7280",
            "border": "#e5e7eb",
            "accent_blue": "#1a73e8",
            "accent_orange": "#ea8600",
            "accent_green": "#059669",
            "accent_red": "#dc2626",
            "accent_yellow": "#d97706",
            "chart_bg": "rgba(255,255,255,0)",
            "hover": "#e5e7eb",
        }

colors = get_theme_colors(st.session_state.theme)
is_dark = st.session_state.theme == "dark"
plot_template = "plotly_dark" if is_dark else "plotly_white"

# ============================================================
# CSS Google Analytics Style
# ============================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto:wght@300;400;500;700&display=swap');
    
    * {{
        font-family: 'Roboto', 'Google Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    .stApp {{
        background: {colors['bg_primary']};
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background: {colors['bg_sidebar']};
        border-right: 1px solid {colors['border']};
    }}
    
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 0;
    }}
    
    /* Main content */
    .main .block-container {{
        padding: 1.5rem 2rem;
        max-width: 100%;
    }}
    
    /* Header */
    .ga-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 0;
        border-bottom: 1px solid {colors['border']};
        margin-bottom: 24px;
    }}
    
    .ga-header-left {{
        display: flex;
        align-items: center;
        gap: 16px;
    }}
    
    .ga-logo {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    
    .ga-logo-icon {{
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, {colors['accent_orange']} 0%, {colors['accent_yellow']} 100%);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        color: white;
    }}
    
    .ga-title {{
        font-size: 22px;
        font-weight: 400;
        color: {colors['text_primary']};
        letter-spacing: -0.5px;
    }}
    
    .ga-subtitle {{
        font-size: 12px;
        color: {colors['text_muted']};
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    /* Metric Cards - Google Analytics Style */
    .metrics-container {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }}
    
    .metric-card {{
        background: {colors['bg_card']};
        border: 1px solid {colors['border']};
        border-radius: 8px;
        padding: 20px;
        transition: box-shadow 0.2s ease;
    }}
    
    .metric-card:hover {{
        box-shadow: 0 1px 3px rgba(60,64,67,0.15), 0 4px 8px rgba(60,64,67,0.15);
    }}
    
    .metric-label {{
        font-size: 12px;
        font-weight: 500;
        color: {colors['text_secondary']};
        text-transform: uppercase;
        letter-spacing: 0.3px;
        margin-bottom: 8px;
    }}
    
    .metric-value {{
        font-size: 28px;
        font-weight: 400;
        color: {colors['text_primary']};
        letter-spacing: -0.5px;
        margin-bottom: 8px;
    }}
    
    .metric-delta {{
        font-size: 12px;
        font-weight: 500;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }}
    
    .metric-delta.positive {{
        color: {colors['accent_green']};
    }}
    
    .metric-delta.negative {{
        color: {colors['accent_red']};
    }}
    
    /* Cards */
    .ga-card {{
        background: {colors['bg_card']};
        border: 1px solid {colors['border']};
        border-radius: 8px;
        margin-bottom: 16px;
        overflow: hidden;
    }}
    
    .ga-card-header {{
        padding: 16px 20px;
        border-bottom: 1px solid {colors['border']};
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    
    .ga-card-title {{
        font-size: 14px;
        font-weight: 500;
        color: {colors['text_primary']};
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    
    .ga-card-body {{
        padding: 20px;
    }}
    
    /* Navigation Sidebar */
    .nav-section {{
        padding: 8px 0;
    }}
    
    .nav-section-title {{
        font-size: 11px;
        font-weight: 500;
        color: {colors['text_muted']};
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 8px 16px;
        margin-bottom: 4px;
    }}
    
    .nav-item {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 16px;
        color: {colors['text_secondary']};
        font-size: 14px;
        cursor: pointer;
        border-radius: 0 20px 20px 0;
        margin-right: 8px;
        transition: all 0.2s ease;
    }}
    
    .nav-item:hover {{
        background: {colors['hover']};
        color: {colors['text_primary']};
    }}
    
    .nav-item.active {{
        background: #e8f0fe;
        color: {colors['accent_blue']};
        font-weight: 500;
    }}
    
    .nav-icon {{
        font-size: 18px;
        width: 24px;
        text-align: center;
    }}
    
    /* Real-time card */
    .realtime-card {{
        background: {colors['accent_blue']};
        border-radius: 8px;
        padding: 20px;
        color: white;
        margin-bottom: 24px;
    }}
    
    .realtime-title {{
        font-size: 12px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.9;
        margin-bottom: 8px;
    }}
    
    .realtime-value {{
        font-size: 48px;
        font-weight: 400;
        letter-spacing: -2px;
    }}
    
    .realtime-label {{
        font-size: 14px;
        opacity: 0.9;
        margin-top: 4px;
    }}
    
    /* Table styling */
    .ga-table {{
        width: 100%;
        border-collapse: collapse;
    }}
    
    .ga-table th {{
        text-align: left;
        padding: 12px 16px;
        font-size: 12px;
        font-weight: 500;
        color: {colors['text_secondary']};
        text-transform: uppercase;
        letter-spacing: 0.3px;
        border-bottom: 1px solid {colors['border']};
    }}
    
    .ga-table td {{
        padding: 12px 16px;
        font-size: 14px;
        color: {colors['text_primary']};
        border-bottom: 1px solid {colors['border']};
    }}
    
    .ga-table tr:hover {{
        background: {colors['hover']};
    }}
    
    /* Progress bar */
    .progress-bar-container {{
        width: 100%;
        height: 4px;
        background: {colors['border']};
        border-radius: 2px;
        overflow: hidden;
    }}
    
    .progress-bar-fill {{
        height: 100%;
        border-radius: 2px;
        transition: width 0.3s ease;
    }}
    
    /* Badges */
    .badge {{
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 500;
    }}
    
    .badge-green {{
        background: rgba(30, 142, 62, 0.1);
        color: {colors['accent_green']};
    }}
    
    .badge-yellow {{
        background: rgba(249, 171, 0, 0.1);
        color: {colors['accent_yellow']};
    }}
    
    .badge-red {{
        background: rgba(217, 48, 37, 0.1);
        color: {colors['accent_red']};
    }}
    
    .badge-blue {{
        background: rgba(26, 115, 232, 0.1);
        color: {colors['accent_blue']};
    }}
    
    /* Buttons */
    .stButton > button {{
        background: {colors['bg_card']};
        color: {colors['accent_blue']};
        border: 1px solid {colors['border']};
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: 500;
        font-size: 14px;
        transition: all 0.2s ease;
    }}
    
    .stButton > button:hover {{
        background: {colors['hover']};
        border-color: {colors['accent_blue']};
    }}
    
    /* Sidebar button override */
    [data-testid="stSidebar"] .stButton > button {{
        width: 100%;
        justify-content: flex-start;
        background: transparent;
        border: none;
        color: {colors['text_secondary']};
        padding: 10px 16px;
        border-radius: 0 20px 20px 0;
        margin-right: 8px;
    }}
    
    [data-testid="stSidebar"] .stButton > button:hover {{
        background: {colors['hover']};
        color: {colors['text_primary']};
    }}
    
    /* Theme toggle */
    .theme-toggle {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        background: {colors['bg_secondary']};
        border: 1px solid {colors['border']};
        border-radius: 20px;
        cursor: pointer;
        font-size: 13px;
        color: {colors['text_secondary']};
    }}
    
    /* Sector list item */
    .sector-item {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 16px;
        border-bottom: 1px solid {colors['border']};
        transition: background 0.2s ease;
    }}
    
    .sector-item:hover {{
        background: {colors['hover']};
    }}
    
    .sector-info {{
        flex: 1;
    }}
    
    .sector-name {{
        font-size: 14px;
        font-weight: 500;
        color: {colors['text_primary']};
        margin-bottom: 4px;
    }}
    
    .sector-meta {{
        font-size: 12px;
        color: {colors['text_muted']};
    }}
    
    .sector-score {{
        font-size: 24px;
        font-weight: 500;
        color: {colors['accent_blue']};
    }}
    
    /* Streamlit element overrides */
    .stSelectbox label {{
        color: {colors['text_secondary']} !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
    }}
    
    .stTextInput label {{
        color: {colors['text_secondary']} !important;
    }}
    
    div[data-testid="stMetricValue"] {{
        color: {colors['text_primary']};
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {colors['bg_secondary']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {colors['border']};
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {colors['text_muted']};
    }}
    
    /* Chart container */
    .chart-container {{
        background: {colors['bg_card']};
        border: 1px solid {colors['border']};
        border-radius: 8px;
        padding: 16px;
    }}
    
    /* Styles pour les métriques Streamlit en mode clair */
    {f"""
    /* ========== MÉTRIQUES STREAMLIT ========== */
    [data-testid='stMetric'] label {{
        color: #374151 !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }}
    [data-testid='stMetric'] [data-testid='stMetricValue'] {{
        color: #111827 !important;
        font-size: 32px !important;
        font-weight: 600 !important;
    }}
    [data-testid='stMetric'] [data-testid='stMetricDelta'] {{
        color: #374151 !important;
        font-weight: 500 !important;
    }}
    [data-testid='stMetric'] [data-testid='stMetricDelta'] svg {{
        fill: #374151 !important;
    }}
    
    /* ========== TEXTES GLOBAUX MODE CLAIR ========== */
    .stApp {{
        color: #1f2937 !important;
    }}
    
    .stApp p, .stApp span, .stApp div {{
        color: #1f2937;
    }}
    
    .stMarkdown {{
        color: #1f2937 !important;
    }}
    
    .stMarkdown p {{
        color: #1f2937 !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
    }}
    
    /* ========== TITRES ========== */
    h1, h2, h3, h4, h5, h6,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
        color: #111827 !important;
        font-weight: 600 !important;
    }}
    
    /* ========== LABELS ET WIDGETS ========== */
    .stSelectbox label, .stTextInput label, .stSlider label, 
    .stMultiSelect label, .stNumberInput label, .stTextArea label,
    [data-testid="stWidgetLabel"] {{
        color: #374151 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }}
    
    .stSelectbox > div > div, .stMultiSelect > div > div {{
        color: #1f2937 !important;
        background-color: #ffffff !important;
    }}
    
    .stSelectbox [data-baseweb="select"] {{
        background-color: #ffffff !important;
    }}
    
    .stSelectbox [data-baseweb="select"] > div {{
        color: #1f2937 !important;
        background-color: #ffffff !important;
        border-color: #d1d5db !important;
    }}
    
    /* ========== INPUTS ========== */
    .stTextInput input, .stNumberInput input, .stTextArea textarea {{
        background-color: #ffffff !important;
        color: #1f2937 !important;
        border: 1.5px solid #d1d5db !important;
        border-radius: 8px !important;
        font-size: 15px !important;
    }}
    
    .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {{
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25) !important;
    }}
    
    .stTextInput input::placeholder, .stTextArea textarea::placeholder {{
        color: #9ca3af !important;
    }}
    
    /* ========== SLIDERS ========== */
    .stSlider > div > div > div {{
        color: #1f2937 !important;
    }}
    
    .stSlider [data-testid="stTickBarMin"], 
    .stSlider [data-testid="stTickBarMax"],
    .stSlider [data-baseweb="slider"] div {{
        color: #374151 !important;
    }}
    
    /* ========== SIDEBAR MODE CLAIR ========== */
    [data-testid="stSidebar"] {{
        background: #f9fafb !important;
    }}
    
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] div, 
    [data-testid="stSidebar"] label {{
        color: #374151 !important;
    }}
    
    [data-testid="stSidebar"] .stButton button {{
        color: #374151 !important;
        background: transparent !important;
        font-weight: 500 !important;
    }}
    
    [data-testid="stSidebar"] .stButton button:hover {{
        background: #e5e7eb !important;
        color: #111827 !important;
    }}
    
    /* ========== BOUTONS ========== */
    .stButton > button {{
        background: #ffffff !important;
        color: #1a73e8 !important;
        font-weight: 600 !important;
    }}
    
    .stButton > button:hover {{
        background: #f3f4f6 !important;
        border-color: #1a73e8 !important;
    }}
    
    .stButton > button[kind="primary"] {{
        background: #1a73e8 !important;
        color: #ffffff !important;
        border: none !important;
    }}
    
    .stButton > button[kind="primary"]:hover {{
        background: #1557b0 !important;
    }}
    
    /* ========== DATAFRAMES ET TABLEAUX ========== */
    .stDataFrame, [data-testid="stDataFrame"] {{
        color: #1f2937 !important;
    }}
    
    .stDataFrame th {{
        background-color: #f3f4f6 !important;
        color: #111827 !important;
        font-weight: 600 !important;
    }}
    
    .stDataFrame td {{
        color: #374151 !important;
    }}
    
    [data-testid="stDataFrameResizable"] {{
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
    }}
    
    /* ========== TABS ========== */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: #f3f4f6 !important;
        border-radius: 8px !important;
        padding: 4px !important;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        color: #6b7280 !important;
        font-weight: 500 !important;
        background: transparent !important;
    }}
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: #1a73e8 !important;
        background: #ffffff !important;
        border-radius: 6px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        color: #111827 !important;
    }}
    
    /* ========== ALERTS ET MESSAGES ========== */
    .stSuccess, .stInfo, .stWarning, .stError {{
        color: #1f2937 !important;
    }}
    
    .stSuccess p, .stInfo p, .stWarning p, .stError p {{
        color: #1f2937 !important;
        font-weight: 500 !important;
    }}
    
    /* ========== EXPANDERS ========== */
    .streamlit-expanderHeader {{
        color: #1f2937 !important;
        font-weight: 600 !important;
        background-color: #f9fafb !important;
    }}
    
    .streamlit-expanderContent {{
        color: #374151 !important;
    }}
    
    /* ========== DOWNLOAD BUTTONS ========== */
    .stDownloadButton button {{
        background: #f3f4f6 !important;
        color: #1f2937 !important;
        border: 1.5px solid #d1d5db !important;
        font-weight: 600 !important;
    }}
    
    .stDownloadButton button:hover {{
        background: #e5e7eb !important;
        border-color: #9ca3af !important;
    }}
    
    /* ========== SPINNER ========== */
    .stSpinner > div {{
        color: #374151 !important;
    }}
    
    /* ========== NUMBER INPUT SPECIFIQUE ========== */
    [data-testid="stNumberInput"] label {{
        color: #374151 !important;
        font-weight: 600 !important;
    }}
    
    [data-testid="stNumberInput"] input {{
        color: #1f2937 !important;
        background: #ffffff !important;
    }}
    
    /* ========== FORM ========== */
    [data-testid="stForm"] {{
        border: 1px solid #e5e7eb !important;
        background: #fafafa !important;
    }}
    
    /* ========== CARDS PERSONNALISÉS ========== */
    .ga-card {{
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
    }}
    
    .ga-card-header {{
        background: #fafafa !important;
        border-bottom: 1px solid #e5e7eb !important;
    }}
    
    .ga-card-title {{
        color: #111827 !important;
        font-weight: 600 !important;
    }}
    
    .ga-card-body {{
        color: #374151 !important;
    }}
    
    /* ========== METRIC CARDS ========== */
    .metric-card {{
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
    }}
    
    .metric-label {{
        color: #6b7280 !important;
        font-weight: 600 !important;
    }}
    
    .metric-value {{
        color: #111827 !important;
        font-weight: 600 !important;
    }}
    
    .metric-delta {{
        font-weight: 600 !important;
    }}
    
    .metric-delta.positive {{
        color: #059669 !important;
    }}
    
    .metric-delta.negative {{
        color: #dc2626 !important;
    }}
    
    /* ========== BADGES MODE CLAIR ========== */
    .badge {{
        font-weight: 600 !important;
    }}
    
    .badge-green {{
        background: rgba(5, 150, 105, 0.15) !important;
        color: #047857 !important;
    }}
    
    .badge-yellow {{
        background: rgba(217, 119, 6, 0.15) !important;
        color: #b45309 !important;
    }}
    
    .badge-red {{
        background: rgba(220, 38, 38, 0.15) !important;
        color: #b91c1c !important;
    }}
    
    .badge-blue {{
        background: rgba(26, 115, 232, 0.15) !important;
        color: #1a73e8 !important;
    }}
    
    /* ========== FORM SUBMIT BUTTON (Bouton Envoyer) ========== */
    [data-testid="stForm"] .stButton > button,
    [data-testid="stFormSubmitButton"] > button,
    .stFormSubmitButton > button {{
        background: #1a73e8 !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
    }}
    
    [data-testid="stForm"] .stButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover,
    .stFormSubmitButton > button:hover {{
        background: #1557b0 !important;
        color: #ffffff !important;
    }}
    
    /* ========== CURSEUR (CARET) VISIBLE EN MODE CLAIR ========== */
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {{
        caret-color: #1a73e8 !important;
    }}
    
    input, textarea {{
        caret-color: #1a73e8 !important;
    }}
    
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea {{
        caret-color: #1a73e8 !important;
    }}
    
    /* ========== CHAT INPUT SPECIFIQUE ========== */
    [data-testid="stChatInput"] input,
    [data-testid="stChatInputContainer"] input {{
        caret-color: #1a73e8 !important;
        color: #1f2937 !important;
        background-color: #ffffff !important;
    }}
    
    /* ========== TABLEAU TOOLBAR (Export CSV, Search, etc.) ========== */
    [data-testid="stDataFrameToolbar"],
    [data-testid="stDataFrame"] > div > div > div:first-child {{
        background-color: #f3f4f6 !important;
        border-bottom: 1px solid #e5e7eb !important;
    }}
    
    [data-testid="stDataFrameToolbar"] button,
    [data-testid="stDataFrame"] button {{
        color: #374151 !important;
        background: #ffffff !important;
        border: 1px solid #d1d5db !important;
    }}
    
    [data-testid="stDataFrameToolbar"] button:hover,
    [data-testid="stDataFrame"] button:hover {{
        background: #e5e7eb !important;
        color: #111827 !important;
    }}
    
    [data-testid="stDataFrameToolbar"] svg,
    [data-testid="stDataFrame"] svg {{
        fill: #374151 !important;
        color: #374151 !important;
    }}
    
    /* Fix pour les icônes de la toolbar */
    [data-testid="stDataFrame"] [data-testid="stElementToolbar"] {{
        background: #f9fafb !important;
    }}
    
    [data-testid="stDataFrame"] [data-testid="stElementToolbar"] button {{
        color: #374151 !important;
    }}
    
    [data-testid="stDataFrame"] [data-testid="stElementToolbar"] button svg {{
        stroke: #374151 !important;
    }}
    
    /* Element toolbar générique */
    [data-testid="stElementToolbar"] {{
        background: #f9fafb !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
    }}
    
    [data-testid="stElementToolbar"] button {{
        color: #374151 !important;
    }}
    
    [data-testid="stElementToolbar"] button:hover {{
        background: #e5e7eb !important;
    }}
    
    [data-testid="stElementToolbar"] svg {{
        stroke: #374151 !important;
        fill: none !important;
    }}
    
    """ if not is_dark else ""}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Configuration des graphiques pour meilleure lisibilité
# ============================================================
def get_chart_layout_config():
    """Retourne la configuration de layout pour les graphiques Plotly."""
    # Couleurs spécifiques pour les graphiques en mode clair
    if is_dark:
        text_color = "#f3f4f6"
        grid_color = "rgba(255,255,255,0.1)"
        tick_color = "#9ca3af"
        title_color = "#f3f4f6"
    else:
        text_color = "#1f2937"
        grid_color = "rgba(0,0,0,0.1)"
        tick_color = "#374151"
        title_color = "#111827"
    
    return {
        "text_color": text_color,
        "grid_color": grid_color,
        "tick_color": tick_color,
        "title_color": title_color
    }

def apply_chart_styling(fig, height=350, show_legend=True, legend_position="top"):
    """Applique un style cohérent aux graphiques Plotly."""
    chart_config = get_chart_layout_config()
    
    legend_config = {}
    if show_legend:
        if legend_position == "top":
            legend_config = dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                font=dict(size=12, color=chart_config['text_color']),
                bgcolor="rgba(0,0,0,0)"
            )
        else:
            legend_config = dict(
                font=dict(size=12, color=chart_config['text_color']),
                bgcolor="rgba(0,0,0,0)"
            )
    
    fig.update_layout(
        height=height,
        template=plot_template,
        paper_bgcolor=colors['chart_bg'],
        plot_bgcolor=colors['chart_bg'],
        font=dict(color=chart_config['text_color'], size=12),
        title_font=dict(color=chart_config['title_color'], size=14),
        legend=legend_config if show_legend else dict(visible=False),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(
            showgrid=True,
            gridcolor=chart_config['grid_color'],
            tickfont=dict(color=chart_config['tick_color'], size=11),
            title_font=dict(color=chart_config['title_color'], size=12),
            linecolor=chart_config['grid_color']
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=chart_config['grid_color'],
            tickfont=dict(color=chart_config['tick_color'], size=11),
            title_font=dict(color=chart_config['title_color'], size=12),
            linecolor=chart_config['grid_color']
        ),
        hoverlabel=dict(
            bgcolor="white" if not is_dark else "#1f2937",
            font_size=12,
            font_color="#1f2937" if not is_dark else "#f3f4f6"
        )
    )
    return fig

# ============================================================
# Chargement des données (cache avec ttl pour actualisation)
# ============================================================
@st.cache_data(ttl=60)
def load_data():
    DATA_PATH = Path("data/processed")
    MODELS_PATH = Path("models")
    df = pd.read_csv(DATA_PATH / "dataset_ml_complete.csv")
    recommendations = pd.read_csv(MODELS_PATH / "recommendations_report.csv")
    with open(MODELS_PATH / "model_metadata.json", 'r', encoding='utf-8') as f:
        model_metadata = json.load(f)
    # Charger evaluation_report si existe, sinon utiliser un dict vide
    eval_report_path = MODELS_PATH / "evaluation_report.json"
    if eval_report_path.exists():
        with open(eval_report_path, 'r', encoding='utf-8') as f:
            evaluation_report = json.load(f)
    else:
        evaluation_report = {}
    return df, recommendations, model_metadata, evaluation_report

@st.cache_resource
def load_models():
    MODELS_PATH = Path("models")
    with open(MODELS_PATH / "xgb_regression_substitution.pkl", 'rb') as f:
        model_reg = pickle.load(f)
    with open(MODELS_PATH / "xgb_classification_opportunity.pkl", 'rb') as f:
        model_clf = pickle.load(f)
    with open(MODELS_PATH / "scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    with open(MODELS_PATH / "api_config.json", 'r', encoding='utf-8') as f:
        api_config = json.load(f)
    return model_reg, model_clf, scaler, api_config

try:
    df, recommendations, model_metadata, evaluation_report = load_data()
    model_reg, model_clf, scaler, api_config = load_models()
    data_loaded = True
except Exception as e:
    st.error(f"Erreur de chargement des données: {e}")
    data_loaded = False

# ============================================================
# SIDEBAR - Navigation Google Analytics Style
# ============================================================
with st.sidebar:
    # Logo et titre
    st.markdown(f"""
    <div style="padding: 20px 16px; border-bottom: 1px solid {colors['border']};">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="width: 32px; height: 32px; background: linear-gradient(135deg, {colors['accent_orange']} 0%, {colors['accent_yellow']} 100%); border-radius: 4px; display: flex; align-items: center; justify-content: center;">
                <span style="font-size: 14px; font-weight: bold; color: white;">BF</span>
            </div>
            <div>
                <div style="font-size: 16px; font-weight: 500; color: {colors['text_primary']};">Analytics</div>
                <div style="font-size: 11px; color: {colors['text_muted']};">Burkina Faso</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    
    # Navigation
    pages = {
        "Accueil": "",
        "Temps réel": "",
        "Analyse": "",
        "Recommandations": "",
        "Simulateur": "",
        "Performance ML": "",
        "Assistant IA": ""
    }
    
    st.markdown(f"<div class='nav-section-title'>RAPPORTS</div>", unsafe_allow_html=True)
    
    for page_name, icon in pages.items():
        if st.button(f"{page_name}", key=f"nav_{page_name}", use_container_width=True):
            st.session_state.page = page_name
    
    st.markdown(f"<div style='height: 24px; border-top: 1px solid {colors['border']}; margin-top: 16px;'></div>", unsafe_allow_html=True)
    
    # Theme toggle
    st.markdown(f"<div class='nav-section-title'>PARAMETRES</div>", unsafe_allow_html=True)
    
    theme_label = "Mode sombre" if not is_dark else "Mode clair"
    if st.button(theme_label, key="theme_toggle", use_container_width=True):
        st.session_state.theme = "dark" if not is_dark else "light"
        st.rerun()

page = st.session_state.page

# ============================================================
# Header
# ============================================================
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
        <span style="font-size: 28px; font-weight: 400; color: {colors['text_primary']};">{page}</span>
    </div>
    <div style="font-size: 13px; color: {colors['text_muted']};">Analyse des flux commerciaux - Burkina Faso - 2014-2025</div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="text-align: right; font-size: 13px; color: {colors['text_muted']};">
        Derniere mise a jour: Decembre 2025
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"<div style='height: 1px; background: {colors['border']}; margin: 16px 0 24px 0;'></div>", unsafe_allow_html=True)

# ============================================================
# Page: Accueil
# ============================================================
if page == "Accueil":
    if data_loaded:
        recent_data = df[df['year'] >= 2021]
        total_prod = recent_data.groupby('year')['production_fcfa'].sum().mean()
        total_imports = recent_data.groupby('year')['imports_fcfa'].sum().mean()
        total_exports = recent_data.groupby('year')['exports_fcfa'].sum().mean()
        balance = total_exports - total_imports
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Production</div>
                <div class="metric-value">{total_prod:,.0f} Mds</div>
                <div class="metric-delta positive">+5.2% vs periode precedente</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Importations</div>
                <div class="metric-value">{total_imports:,.0f} Mds</div>
                <div class="metric-delta negative">+2.1% vs periode precedente</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Exportations</div>
                <div class="metric-value">{total_exports:,.0f} Mds</div>
                <div class="metric-delta positive">+8.3% vs periode precedente</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            delta_class = "positive" if balance > 0 else "negative"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Balance commerciale</div>
                <div class="metric-value">{balance:,.0f} Mds</div>
                <div class="metric-delta {delta_class}">{"Excedent" if balance > 0 else "Deficit"}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        
        # Graphiques
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="ga-card">
                <div class="ga-card-header">
                    <span class="ga-card-title">Evolution des flux commerciaux</span>
                    <span style="font-size: 12px; color: {colors['text_muted']};">2014 - 2025</span>
                </div>
                <div class="ga-card-body">
            """, unsafe_allow_html=True)
            
            yearly_data = df.groupby('year').agg({
                'production_fcfa': 'sum',
                'imports_fcfa': 'sum',
                'exports_fcfa': 'sum'
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=yearly_data['year'], 
                y=yearly_data['production_fcfa'],
                mode='lines+markers', 
                name='Production',
                line=dict(width=2, color=colors['accent_blue']),
                marker=dict(size=6)
            ))
            fig.add_trace(go.Scatter(
                x=yearly_data['year'], 
                y=yearly_data['imports_fcfa'],
                mode='lines+markers', 
                name='Imports',
                line=dict(width=2, color=colors['accent_red'])
            ))
            fig.add_trace(go.Scatter(
                x=yearly_data['year'], 
                y=yearly_data['exports_fcfa'],
                mode='lines+markers', 
                name='Exports',
                line=dict(width=2, color=colors['accent_green'])
            ))
            
            fig.update_layout(
                height=350,
                template=plot_template,
                paper_bgcolor=colors['chart_bg'],
                plot_bgcolor=colors['chart_bg'],
                font=dict(color=colors['text_primary'], size=12),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                    font=dict(size=12, color=colors['text_primary']),
                    bgcolor="rgba(0,0,0,0)" if is_dark else "rgba(255,255,255,0.8)"
                ),
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(
                    showgrid=True,
                    gridcolor=colors['border'] if is_dark else "rgba(0,0,0,0.08)",
                    title="",
                    tickfont=dict(color=colors['text_primary'], size=11),
                    linecolor=colors['border']
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor=colors['border'] if is_dark else "rgba(0,0,0,0.08)",
                    title="Milliards FCFA",
                    tickfont=dict(color=colors['text_primary'], size=11),
                    title_font=dict(color=colors['text_primary'], size=12),
                    linecolor=colors['border']
                ),
                hovermode='x unified',
                hoverlabel=dict(
                    bgcolor="#1f2937" if is_dark else "#ffffff",
                    font_size=12,
                    font_color="#f3f4f6" if is_dark else "#1f2937",
                    bordercolor=colors['border']
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        with col2:
            # Real-time style card
            st.markdown(f"""
            <div class="realtime-card">
                <div class="realtime-title">Secteurs analyses</div>
                <div class="realtime-value">{len(recommendations)}</div>
                <div class="realtime-label">Opportunites identifiees</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Distribution des opportunités
            st.markdown(f"""
            <div class="ga-card">
                <div class="ga-card-header">
                    <span class="ga-card-title">Repartition</span>
                </div>
                <div class="ga-card-body">
            """, unsafe_allow_html=True)
            
            opp_counts = recommendations['classification'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=opp_counts.index,
                values=opp_counts.values,
                hole=0.65,
                marker=dict(colors=[colors['accent_green'], colors['accent_yellow'], colors['accent_red']]),
                textinfo='percent',
                textposition='outside',
                textfont=dict(size=11)
            )])
            fig.update_layout(
                height=200,
                showlegend=False,
                template=plot_template,
                paper_bgcolor=colors['chart_bg'],
                margin=dict(l=20, r=20, t=20, b=20),
                font=dict(color=colors['text_primary'], size=12)
            )
            # Améliorer la couleur du texte pour le pie chart
            fig.update_traces(
                textfont=dict(color=colors['text_primary'], size=12),
                outsidetextfont=dict(color=colors['text_primary'], size=11)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Top secteurs
        st.markdown(f"""
        <div class="ga-card">
            <div class="ga-card-header">
                <span class="ga-card-title">Top secteurs a fort potentiel</span>
                <span class="badge badge-blue">{len(recommendations[recommendations['classification'].str.contains('Fort', na=False)])} secteurs prioritaires</span>
            </div>
        """, unsafe_allow_html=True)
        
        for idx, row in recommendations.head(5).iterrows():
            score = row['score_substitution']
            badge_class = "green" if score > 70 else "yellow" if score > 40 else "red"
            
            st.markdown(f"""
            <div class="sector-item">
                <div class="sector-info">
                    <div class="sector-name">{row['secteur'][:60]}...</div>
                    <div class="sector-meta">
                        Production: {row['production_mds_fcfa']:.1f} Mds | 
                        Imports: {row['imports_mds_fcfa']:.1f} Mds | 
                        Croissance: {row['croissance_production_pct']:.1f}%
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 16px;">
                    <span class="badge badge-{badge_class}">{row['classification']}</span>
                    <div class="sector-score">{score:.0f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Page: Temps réel
# ============================================================
elif page == "Temps réel":
    if data_loaded:
        import datetime
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        current_date = datetime.datetime.now().strftime("%d/%m/%Y")
        
        # En-tête avec indicateur temps réel
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
            <div style="width: 12px; height: 12px; background: #22c55e; border-radius: 50%; animation: pulse 2s infinite;"></div>
            <span style="color: {colors['text_primary']}; font-weight: 500;">Données en direct - Dernière mise à jour: {current_date} à {current_time}</span>
        </div>
        <style>
            @keyframes pulse {{
                0% {{ opacity: 1; box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }}
                70% {{ opacity: 0.7; box-shadow: 0 0 0 10px rgba(34, 197, 94, 0); }}
                100% {{ opacity: 1; box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }}
            }}
        </style>
        """, unsafe_allow_html=True)
        
        # Métriques principales en temps réel
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculs dynamiques
        latest_year = df['year'].max()
        prev_year = latest_year - 1
        
        current_data = df[df['year'] == latest_year]
        prev_data = df[df['year'] == prev_year]
        
        total_prod = current_data['production_fcfa'].sum()
        total_prod_prev = prev_data['production_fcfa'].sum()
        prod_change = ((total_prod - total_prod_prev) / total_prod_prev * 100) if total_prod_prev > 0 else 0
        
        total_imports = current_data['imports_fcfa'].sum()
        total_imports_prev = prev_data['imports_fcfa'].sum()
        imports_change = ((total_imports - total_imports_prev) / total_imports_prev * 100) if total_imports_prev > 0 else 0
        
        total_exports = current_data['exports_fcfa'].sum()
        total_exports_prev = prev_data['exports_fcfa'].sum()
        exports_change = ((total_exports - total_exports_prev) / total_exports_prev * 100) if total_exports_prev > 0 else 0
        
        balance = total_exports - total_imports
        balance_prev = total_exports_prev - total_imports_prev
        
        with col1:
            delta_class = "positive" if prod_change >= 0 else "negative"
            delta_icon = "↑" if prod_change >= 0 else "↓"
            st.markdown(f"""
            <div class="metric-card" style="padding: 20px;">
                <div class="metric-label"> Production {latest_year}</div>
                <div class="metric-value">{total_prod/1e9:.1f} Mds</div>
                <div class="metric-delta {delta_class}">{delta_icon} {abs(prod_change):.1f}% vs {prev_year}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            delta_class = "negative" if imports_change >= 0 else "positive"
            delta_icon = "↑" if imports_change >= 0 else "↓"
            st.markdown(f"""
            <div class="metric-card" style="padding: 20px;">
                <div class="metric-label"> Importations {latest_year}</div>
                <div class="metric-value">{total_imports/1e9:.1f} Mds</div>
                <div class="metric-delta {delta_class}">{delta_icon} {abs(imports_change):.1f}% vs {prev_year}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            delta_class = "positive" if exports_change >= 0 else "negative"
            delta_icon = "↑" if exports_change >= 0 else "↓"
            st.markdown(f"""
            <div class="metric-card" style="padding: 20px;">
                <div class="metric-label"> Exportations {latest_year}</div>
                <div class="metric-value">{total_exports/1e9:.1f} Mds</div>
                <div class="metric-delta {delta_class}">{delta_icon} {abs(exports_change):.1f}% vs {prev_year}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            balance_class = "positive" if balance >= 0 else "negative"
            balance_icon = "" if balance >= 0 else ""
            st.markdown(f"""
            <div class="metric-card" style="padding: 20px;">
                <div class="metric-label">{balance_icon} Balance Commerciale</div>
                <div class="metric-value" style="color: {'#059669' if balance >= 0 else '#dc2626'};">{balance/1e9:.1f} Mds</div>
                <div class="metric-delta {'positive' if balance >= 0 else 'negative'}">{'Excédent' if balance >= 0 else 'Déficit'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Graphiques en temps réel
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="ga-card">
                <div class="ga-card-header">
                    <span class="ga-card-title"> Evolution des flux (2014-{latest_year})</span>
                </div>
            """, unsafe_allow_html=True)
            
            yearly_data = df.groupby('year').agg({
                'production_fcfa': 'sum',
                'imports_fcfa': 'sum',
                'exports_fcfa': 'sum'
            }).reset_index()
            yearly_data['balance'] = yearly_data['exports_fcfa'] - yearly_data['imports_fcfa']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=yearly_data['year'], y=yearly_data['production_fcfa']/1e9, 
                                     name='Production', mode='lines+markers', line=dict(color=colors['accent_green'], width=3)))
            fig.add_trace(go.Scatter(x=yearly_data['year'], y=yearly_data['imports_fcfa']/1e9, 
                                     name='Importations', mode='lines+markers', line=dict(color=colors['accent_red'], width=3)))
            fig.add_trace(go.Scatter(x=yearly_data['year'], y=yearly_data['exports_fcfa']/1e9, 
                                     name='Exportations', mode='lines+markers', line=dict(color=colors['accent_blue'], width=3)))
            fig.update_layout(
                height=350, template=plot_template,
                paper_bgcolor=colors['chart_bg'], plot_bgcolor=colors['chart_bg'],
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                           bgcolor="rgba(0,0,0,0)" if is_dark else "rgba(255,255,255,0.9)",
                           font=dict(color=colors['text_primary'])),
                xaxis_title="Année", yaxis_title="Milliards FCFA",
                font=dict(color=colors['text_primary'], size=12),
                xaxis=dict(gridcolor=colors['border'] if is_dark else "rgba(0,0,0,0.1)",
                          tickfont=dict(color=colors['text_primary'], size=11),
                          title_font=dict(color=colors['text_primary'], size=12)),
                yaxis=dict(gridcolor=colors['border'] if is_dark else "rgba(0,0,0,0.1)",
                          tickfont=dict(color=colors['text_primary'], size=11),
                          title_font=dict(color=colors['text_primary'], size=12)),
                hoverlabel=dict(bgcolor="#1f2937" if is_dark else "#ffffff",
                               font_color="#f3f4f6" if is_dark else "#1f2937",
                               bordercolor=colors['border'])
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="ga-card">
                <div class="ga-card-header">
                    <span class="ga-card-title"> Top 10 Secteurs - Potentiel de Substitution</span>
                </div>
            """, unsafe_allow_html=True)
            
            sector_potential = current_data.groupby('LIBELLES').agg({
                'production_fcfa': 'sum',
                'imports_fcfa': 'sum'
            }).reset_index()
            sector_potential['gap'] = sector_potential['imports_fcfa'] - sector_potential['production_fcfa']
            sector_potential = sector_potential.nlargest(10, 'gap')
            sector_potential['LIBELLES'] = sector_potential['LIBELLES'].apply(lambda x: x[:25] + '...' if len(x) > 25 else x)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(y=sector_potential['LIBELLES'], x=sector_potential['production_fcfa']/1e9, 
                                 name='Production', orientation='h', marker_color=colors['accent_green']))
            fig.add_trace(go.Bar(y=sector_potential['LIBELLES'], x=sector_potential['imports_fcfa']/1e9, 
                                 name='Importations', orientation='h', marker_color=colors['accent_red']))
            fig.update_layout(
                height=350, template=plot_template, barmode='group',
                paper_bgcolor=colors['chart_bg'], plot_bgcolor=colors['chart_bg'],
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                           bgcolor="rgba(0,0,0,0)" if is_dark else "rgba(255,255,255,0.9)",
                           font=dict(color=colors['text_primary'])),
                xaxis_title="Milliards FCFA",
                font=dict(color=colors['text_primary'], size=12),
                xaxis=dict(gridcolor=colors['border'] if is_dark else "rgba(0,0,0,0.1)",
                          tickfont=dict(color=colors['text_primary'], size=11),
                          title_font=dict(color=colors['text_primary'], size=12)),
                yaxis=dict(tickfont=dict(color=colors['text_primary'], size=11)),
                hoverlabel=dict(bgcolor="#1f2937" if is_dark else "#ffffff",
                               font_color="#f3f4f6" if is_dark else "#1f2937",
                               bordercolor=colors['border'])
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Deuxième rangée de graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="ga-card">
                <div class="ga-card-header">
                    <span class="ga-card-title"> Top 10 Secteurs par Production</span>
                </div>
            """, unsafe_allow_html=True)
            
            top_prod = current_data.groupby('LIBELLES')['production_fcfa'].sum().nlargest(10).reset_index()
            top_prod['LIBELLES'] = top_prod['LIBELLES'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)
            
            fig = px.bar(
                top_prod, x='production_fcfa', y='LIBELLES', orientation='h',
                color='production_fcfa', color_continuous_scale='Greens'
            )
            fig.update_layout(
                height=350, template=plot_template,
                paper_bgcolor=colors['chart_bg'], plot_bgcolor=colors['chart_bg'],
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Production (FCFA)", yaxis_title="",
                showlegend=False, coloraxis_showscale=False,
                font=dict(color=colors['text_primary'], size=12),
                xaxis=dict(gridcolor=colors['border'] if is_dark else "rgba(0,0,0,0.1)",
                          tickfont=dict(color=colors['text_primary'], size=11),
                          title_font=dict(color=colors['text_primary'], size=12)),
                yaxis=dict(tickfont=dict(color=colors['text_primary'], size=11)),
                hoverlabel=dict(bgcolor="#1f2937" if is_dark else "#ffffff",
                               font_color="#f3f4f6" if is_dark else "#1f2937",
                               bordercolor=colors['border'])
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="ga-card">
                <div class="ga-card-header">
                    <span class="ga-card-title"> Top 10 Secteurs par Importations</span>
                </div>
            """, unsafe_allow_html=True)
            
            top_imports = current_data.groupby('LIBELLES')['imports_fcfa'].sum().nlargest(10).reset_index()
            top_imports['LIBELLES'] = top_imports['LIBELLES'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)
            
            fig = px.bar(
                top_imports, x='imports_fcfa', y='LIBELLES', orientation='h',
                color='imports_fcfa', color_continuous_scale='Reds'
            )
            fig.update_layout(
                height=350, template=plot_template,
                paper_bgcolor=colors['chart_bg'], plot_bgcolor=colors['chart_bg'],
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Importations (FCFA)", yaxis_title="",
                showlegend=False, coloraxis_showscale=False,
                font=dict(color=colors['text_primary'], size=12),
                xaxis=dict(gridcolor=colors['border'] if is_dark else "rgba(0,0,0,0.1)",
                          tickfont=dict(color=colors['text_primary'], size=11),
                          title_font=dict(color=colors['text_primary'], size=12)),
                yaxis=dict(tickfont=dict(color=colors['text_primary'], size=11)),
                hoverlabel=dict(bgcolor="#1f2937" if is_dark else "#ffffff",
                               font_color="#f3f4f6" if is_dark else "#1f2937",
                               bordercolor=colors['border'])
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Tableau récapitulatif avec actualisation
        st.markdown(f"""
        <div class="ga-card">
            <div class="ga-card-header">
                <span class="ga-card-title"> Données détaillées - Année {latest_year}</span>
                <span class="badge badge-green">Live</span>
            </div>
            <div class="ga-card-body">
        """, unsafe_allow_html=True)
        
        summary_data = current_data.groupby('LIBELLES').agg({
            'production_fcfa': 'sum',
            'imports_fcfa': 'sum',
            'exports_fcfa': 'sum'
        }).reset_index()
        summary_data['balance'] = summary_data['exports_fcfa'] - summary_data['imports_fcfa']
        summary_data['taux_couverture'] = (summary_data['exports_fcfa'] / summary_data['imports_fcfa'] * 100).fillna(0)
        summary_data = summary_data.sort_values('imports_fcfa', ascending=False).head(20)
        
        st.dataframe(
            summary_data.round(2),
            use_container_width=True,
            hide_index=True,
            column_config={
                'LIBELLES': 'Secteur',
                'production_fcfa': st.column_config.NumberColumn('Production', format="%.0f"),
                'imports_fcfa': st.column_config.NumberColumn('Importations', format="%.0f"),
                'exports_fcfa': st.column_config.NumberColumn('Exportations', format="%.0f"),
                'balance': st.column_config.NumberColumn('Balance', format="%.0f"),
                'taux_couverture': st.column_config.NumberColumn('Taux couv. %', format="%.1f")
            }
        )
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Bouton de rafraîchissement
        if st.button(" Rafraîchir les données", key="refresh_realtime"):
            st.cache_data.clear()
            st.rerun()

# ============================================================
# Page: Analyse Sectorielle Améliorée
# ============================================================
elif page == "Analyse":
    if data_loaded:
        # En-tête avec statistiques globales
        st.markdown(f"""
        <div class="ga-card">
            <div class="ga-card-header">
                <span class="ga-card-title"> Analyse Sectorielle Détaillée</span>
            </div>
            <div class="ga-card-body">
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_sector = st.selectbox(" Sélectionner un secteur", df['LIBELLES'].unique().tolist(), key="analysis_sector")
        with col2:
            year_range = st.select_slider(
                " Période d'analyse",
                options=sorted(df['year'].unique()),
                value=(df['year'].min(), df['year'].max())
            )
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        if selected_sector:
            sector_data = df[(df['LIBELLES'] == selected_sector) & 
                            (df['year'] >= year_range[0]) & 
                            (df['year'] <= year_range[1])].sort_values('year')
            
            if len(sector_data) > 0:
                latest = sector_data[sector_data['year'] == sector_data['year'].max()].iloc[0]
                first = sector_data[sector_data['year'] == sector_data['year'].min()].iloc[0]
                
                # Calcul des variations
                prod_change = ((latest['production_fcfa'] - first['production_fcfa']) / first['production_fcfa'] * 100) if first['production_fcfa'] > 0 else 0
                imp_change = ((latest['imports_fcfa'] - first['imports_fcfa']) / first['imports_fcfa'] * 100) if first['imports_fcfa'] > 0 else 0
                exp_change = ((latest['exports_fcfa'] - first['exports_fcfa']) / first['exports_fcfa'] * 100) if first['exports_fcfa'] > 0 else 0
                
                # Métriques du secteur
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    delta_class = "positive" if prod_change >= 0 else "negative"
                    delta_icon = "↑" if prod_change >= 0 else "↓"
                    st.markdown(f"""
                    <div class="metric-card" style="padding: 20px;">
                        <div class="metric-label"> Production</div>
                        <div class="metric-value">{latest['production_fcfa']:.1f} Mds</div>
                        <div class="metric-delta {delta_class}">{delta_icon} {abs(prod_change):.1f}% sur la période</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    delta_class = "negative" if imp_change >= 0 else "positive"
                    delta_icon = "↑" if imp_change >= 0 else "↓"
                    st.markdown(f"""
                    <div class="metric-card" style="padding: 20px;">
                        <div class="metric-label"> Importations</div>
                        <div class="metric-value">{latest['imports_fcfa']:.1f} Mds</div>
                        <div class="metric-delta {delta_class}">{delta_icon} {abs(imp_change):.1f}% sur la période</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    delta_class = "positive" if exp_change >= 0 else "negative"
                    delta_icon = "↑" if exp_change >= 0 else "↓"
                    st.markdown(f"""
                    <div class="metric-card" style="padding: 20px;">
                        <div class="metric-label"> Exportations</div>
                        <div class="metric-value">{latest['exports_fcfa']:.1f} Mds</div>
                        <div class="metric-delta {delta_class}">{delta_icon} {abs(exp_change):.1f}% sur la période</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    balance = latest['exports_fcfa'] - latest['imports_fcfa']
                    balance_class = "positive" if balance >= 0 else "negative"
                    st.markdown(f"""
                    <div class="metric-card" style="padding: 20px;">
                        <div class="metric-label">⚖️ Balance Commerciale</div>
                        <div class="metric-value" style="color: {'#059669' if balance >= 0 else '#dc2626'};">{balance:.1f} Mds</div>
                        <div class="metric-delta {balance_class}">{'Excédentaire' if balance >= 0 else 'Déficitaire'}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Onglets d'analyse
                analysis_tabs = st.tabs([" Évolution", " Comparaison", " Diagnostic", " Données"])
                
                with analysis_tabs[0]:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="ga-card">
                            <div class="ga-card-header">
                                <span class="ga-card-title"> Évolution des flux commerciaux</span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=sector_data['year'], y=sector_data['production_fcfa'],
                            mode='lines+markers', name='Production',
                            line=dict(width=3, color=colors['accent_green']),
                            fill='tozeroy', fillcolor=f"rgba(5, 150, 105, 0.1)"
                        ))
                        fig.add_trace(go.Scatter(
                            x=sector_data['year'], y=sector_data['imports_fcfa'],
                            mode='lines+markers', name='Importations',
                            line=dict(width=3, color=colors['accent_red'])
                        ))
                        fig.add_trace(go.Scatter(
                            x=sector_data['year'], y=sector_data['exports_fcfa'],
                            mode='lines+markers', name='Exportations',
                            line=dict(width=3, color=colors['accent_blue'])
                        ))
                        fig.update_layout(
                            height=350, template=plot_template,
                            paper_bgcolor=colors['chart_bg'], plot_bgcolor=colors['chart_bg'],
                            font=dict(color=colors['text_primary'], size=12),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                       bgcolor="rgba(0,0,0,0)" if is_dark else "rgba(255,255,255,0.9)",
                                       font=dict(color=colors['text_primary'])),
                            margin=dict(l=0, r=0, t=30, b=0), hovermode='x unified',
                            xaxis=dict(gridcolor=colors['border'] if is_dark else "rgba(0,0,0,0.1)",
                                      tickfont=dict(color=colors['text_primary'], size=11)),
                            yaxis=dict(gridcolor=colors['border'] if is_dark else "rgba(0,0,0,0.1)",
                                      tickfont=dict(color=colors['text_primary'], size=11)),
                            hoverlabel=dict(bgcolor="#1f2937" if is_dark else "#ffffff",
                                           font_color="#f3f4f6" if is_dark else "#1f2937")
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="ga-card">
                            <div class="ga-card-header">
                                <span class="ga-card-title"> Balance commerciale annuelle</span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        sector_data_copy = sector_data.copy()
                        sector_data_copy['balance'] = sector_data_copy['exports_fcfa'] - sector_data_copy['imports_fcfa']
                        bar_colors = [colors['accent_green'] if x >= 0 else colors['accent_red'] for x in sector_data_copy['balance']]
                        
                        fig = go.Figure(go.Bar(
                            x=sector_data_copy['year'], y=sector_data_copy['balance'],
                            marker_color=bar_colors, text=sector_data_copy['balance'].round(1),
                            textposition='outside', textfont=dict(color=colors['text_primary'])
                        ))
                        fig.add_hline(y=0, line_dash="dash", line_color=colors['text_muted'])
                        fig.update_layout(
                            height=350, template=plot_template,
                            paper_bgcolor=colors['chart_bg'], plot_bgcolor=colors['chart_bg'],
                            font=dict(color=colors['text_primary'], size=12),
                            margin=dict(l=0, r=0, t=10, b=0),
                            yaxis_title="Milliards FCFA",
                            xaxis=dict(gridcolor=colors['border'] if is_dark else "rgba(0,0,0,0.1)",
                                      tickfont=dict(color=colors['text_primary'], size=11)),
                            yaxis=dict(gridcolor=colors['border'] if is_dark else "rgba(0,0,0,0.1)",
                                      tickfont=dict(color=colors['text_primary'], size=11),
                                      title_font=dict(color=colors['text_primary'], size=12)),
                            hoverlabel=dict(bgcolor="#1f2937" if is_dark else "#ffffff",
                                           font_color="#f3f4f6" if is_dark else "#1f2937")
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                
                with analysis_tabs[1]:
                    st.markdown(f"""
                    <div class="ga-card">
                        <div class="ga-card-header">
                            <span class="ga-card-title"> Comparaison avec d'autres secteurs</span>
                        </div>
                        <div class="ga-card-body">
                    """, unsafe_allow_html=True)
                    
                    # Comparaison avec les secteurs similaires
                    all_sectors = df[df['year'] == latest['year']].groupby('LIBELLES').agg({
                        'production_fcfa': 'sum', 'imports_fcfa': 'sum', 'exports_fcfa': 'sum'
                    }).reset_index()
                    all_sectors['balance'] = all_sectors['exports_fcfa'] - all_sectors['imports_fcfa']
                    all_sectors = all_sectors.sort_values('production_fcfa', ascending=False).head(15)
                    all_sectors['is_selected'] = all_sectors['LIBELLES'] == selected_sector
                    all_sectors['LIBELLES'] = all_sectors['LIBELLES'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)
                    
                    colors_bar = [colors['accent_blue'] if x else colors['text_muted'] for x in all_sectors['is_selected']]
                    
                    fig = go.Figure(go.Bar(
                        x=all_sectors['LIBELLES'], y=all_sectors['production_fcfa'],
                        marker_color=colors_bar, text=all_sectors['production_fcfa'].round(0),
                        textposition='outside'
                    ))
                    fig.update_layout(
                        height=400, template=plot_template,
                        paper_bgcolor=colors['chart_bg'], plot_bgcolor=colors['chart_bg'],
                        font=dict(color=colors['text_primary'], size=12),
                        margin=dict(l=0, r=0, t=10, b=100),
                        xaxis_tickangle=-45, yaxis_title="Production (Mds FCFA)",
                        xaxis=dict(gridcolor=colors['border'] if is_dark else "rgba(0,0,0,0.1)",
                                  tickfont=dict(color=colors['text_primary'], size=10)),
                        yaxis=dict(gridcolor=colors['border'] if is_dark else "rgba(0,0,0,0.1)",
                                  tickfont=dict(color=colors['text_primary'], size=11),
                                  title_font=dict(color=colors['text_primary'], size=12)),
                        hoverlabel=dict(bgcolor="#1f2937" if is_dark else "#ffffff",
                                       font_color="#f3f4f6" if is_dark else "#1f2937")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("</div></div>", unsafe_allow_html=True)
                
                with analysis_tabs[2]:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="ga-card">
                            <div class="ga-card-header">
                                <span class="ga-card-title"> Diagnostic du secteur</span>
                            </div>
                            <div class="ga-card-body">
                        """, unsafe_allow_html=True)
                        
                        # Générer le diagnostic
                        diagnostics = []
                        
                        if latest['imports_fcfa'] > latest['production_fcfa']:
                            gap_pct = (latest['imports_fcfa'] - latest['production_fcfa']) / latest['imports_fcfa'] * 100
                            diagnostics.append(("🔴", "Dépendance aux importations", f"Les importations dépassent la production de {gap_pct:.0f}%"))
                        else:
                            diagnostics.append(("🟢", "Bonne autonomie", "La production couvre les besoins d'importation"))
                        
                        if prod_change > 10:
                            diagnostics.append(("🟢", "Croissance forte", f"Production en hausse de {prod_change:.1f}% sur la période"))
                        elif prod_change > 0:
                            diagnostics.append(("🟡", "Croissance modérée", f"Production en hausse de {prod_change:.1f}%"))
                        else:
                            diagnostics.append(("🔴", "Déclin", f"Production en baisse de {abs(prod_change):.1f}%"))
                        
                        if balance >= 0:
                            diagnostics.append(("🟢", "Balance positive", f"Excédent commercial de {balance:.1f} Mds FCFA"))
                        else:
                            diagnostics.append(("🔴", "Balance négative", f"Déficit commercial de {abs(balance):.1f} Mds FCFA"))
                        
                        if latest['exports_fcfa'] > 0 and latest['imports_fcfa'] > 0:
                            taux_couv = latest['exports_fcfa'] / latest['imports_fcfa'] * 100
                            if taux_couv > 100:
                                diagnostics.append(("🟢", "Taux de couverture excellent", f"Les exports couvrent {taux_couv:.0f}% des imports"))
                            elif taux_couv > 50:
                                diagnostics.append(("🟡", "Taux de couverture moyen", f"Les exports couvrent {taux_couv:.0f}% des imports"))
                            else:
                                diagnostics.append(("🔴", "Taux de couverture faible", f"Les exports ne couvrent que {taux_couv:.0f}% des imports"))
                        
                        for icon, title, desc in diagnostics:
                            st.markdown(f"""
                            <div style="padding: 12px; margin: 8px 0; background: {colors['bg_secondary']}; border-radius: 8px; border-left: 4px solid {'#059669' if icon == '🟢' else '#d97706' if icon == '🟡' else '#dc2626'};">
                                <strong style="color: {colors['text_primary']};">{icon} {title}</strong><br>
                                <span style="font-size: 13px; color: {colors['text_secondary']};">{desc}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("</div></div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="ga-card">
                            <div class="ga-card-header">
                                <span class="ga-card-title"> Recommandations</span>
                            </div>
                            <div class="ga-card-body">
                        """, unsafe_allow_html=True)
                        
                        recommendations_list = []
                        
                        if latest['imports_fcfa'] > latest['production_fcfa']:
                            recommendations_list.append(("Investir dans la production locale", "Améliorer les capacités de production pour réduire la dépendance"))
                        
                        if exp_change < 0:
                            recommendations_list.append(("Développer les exportations", "Explorer de nouveaux marchés pour relancer les exports"))
                        
                        if balance < 0:
                            recommendations_list.append(("Rééquilibrer la balance", "Stratégie de substitution aux importations"))
                        
                        recommendations_list.append(("Moderniser l'appareil productif", "Investir dans les technologies pour améliorer la compétitivité"))
                        
                        for i, (title, desc) in enumerate(recommendations_list, 1):
                            st.markdown(f"""
                            <div style="padding: 12px; margin: 8px 0; background: {colors['bg_secondary']}; border-radius: 8px;">
                                <strong style="color: {colors['accent_blue']};">{i}. {title}</strong><br>
                                <span style="font-size: 13px; color: {colors['text_secondary']};">{desc}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("</div></div>", unsafe_allow_html=True)
                
                with analysis_tabs[3]:
                    st.markdown(f"""
                    <div class="ga-card">
                        <div class="ga-card-header">
                            <span class="ga-card-title"> Données historiques</span>
                        </div>
                        <div class="ga-card-body">
                    """, unsafe_allow_html=True)
                    
                    display_data = sector_data[['year', 'production_fcfa', 'imports_fcfa', 'exports_fcfa']].copy()
                    display_data['balance'] = display_data['exports_fcfa'] - display_data['imports_fcfa']
                    display_data['taux_couverture'] = (display_data['exports_fcfa'] / display_data['imports_fcfa'] * 100).fillna(0)
                    
                    st.dataframe(
                        display_data.round(2),
                        use_container_width=True, hide_index=True,
                        column_config={
                            'year': 'Année',
                            'production_fcfa': st.column_config.NumberColumn('Production', format="%.1f"),
                            'imports_fcfa': st.column_config.NumberColumn('Importations', format="%.1f"),
                            'exports_fcfa': st.column_config.NumberColumn('Exportations', format="%.1f"),
                            'balance': st.column_config.NumberColumn('Balance', format="%.1f"),
                            'taux_couverture': st.column_config.NumberColumn('Taux couv. %', format="%.1f")
                        }
                    )
                    
                    csv = display_data.to_csv(index=False).encode('utf-8')
                    st.download_button(f"📥 Télécharger les données de {selected_sector[:30]}", csv, f"analyse_{selected_sector[:20]}.csv", "text/csv")
                    
                    st.markdown("</div></div>", unsafe_allow_html=True)

# ============================================================
# Page: Recommandations Améliorée
# ============================================================
elif page == "Recommandations":
    if data_loaded:
        # En-tête avec statistiques
        total_sectors = len(recommendations)
        high_priority = len(recommendations[recommendations['classification'] == 'Haute Priorité']) if 'Haute Priorité' in recommendations['classification'].values else len(recommendations[recommendations['score_substitution'] >= 70])
        avg_score = recommendations['score_substitution'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="padding: 20px;">
                <div class="metric-label"> Secteurs analysés</div>
                <div class="metric-value">{total_sectors}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="padding: 20px;">
                <div class="metric-label"> Haute priorité</div>
                <div class="metric-value" style="color: #059669;">{high_priority}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="padding: 20px;">
                <div class="metric-label"> Score moyen</div>
                <div class="metric-value">{avg_score:.1f}/100</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            potential_total = recommendations['imports_mds_fcfa'].sum() - recommendations['production_mds_fcfa'].sum()
            st.markdown(f"""
            <div class="metric-card" style="padding: 20px;">
                <div class="metric-label"> Potentiel substitution</div>
                <div class="metric-value">{max(0, potential_total):.0f} Mds</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Filtres améliorés
        st.markdown(f"""
        <div class="ga-card">
            <div class="ga-card-header">
                <span class="ga-card-title"> Filtres et Paramètres</span>
            </div>
            <div class="ga-card-body">
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_class = st.multiselect(
                "Classification", 
                recommendations['classification'].unique(), 
                default=recommendations['classification'].unique()
            )
        
        with col2:
            score_range = st.slider("Score de substitution", 0, 100, (0, 100))
        
        with col3:
            top_n = st.slider("Nombre de secteurs", 5, min(50, len(recommendations)), 15)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Appliquer les filtres
        filtered_reco = recommendations[
            (recommendations['classification'].isin(filter_class)) &
            (recommendations['score_substitution'] >= score_range[0]) &
            (recommendations['score_substitution'] <= score_range[1])
        ].head(top_n)
        
        # Onglets de visualisation
        reco_tabs = st.tabs([" Cartographie", " Top Secteurs", " Analyses", " Tableau"])
        
        with reco_tabs[0]:
            st.markdown(f"""
            <div class="ga-card">
                <div class="ga-card-header">
                    <span class="ga-card-title"> Cartographie des Opportunités</span>
                </div>
                <div class="ga-card-body">
            """, unsafe_allow_html=True)
            
            fig = px.scatter(
                filtered_reco, 
                x='imports_mds_fcfa', y='production_mds_fcfa',
                size='score_substitution', color='score_substitution',
                hover_name='secteur', 
                color_continuous_scale='RdYlGn',
                size_max=50,
                labels={
                    'imports_mds_fcfa': 'Importations (Mds FCFA)', 
                    'production_mds_fcfa': 'Production (Mds FCFA)', 
                    'score_substitution': 'Score'
                }
            )
            
            max_val = max(filtered_reco['imports_mds_fcfa'].max(), filtered_reco['production_mds_fcfa'].max()) * 1.1
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val], 
                mode='lines', line=dict(color=colors['text_muted'], dash='dash', width=2), 
                name='Parité Production=Imports', showlegend=True
            ))
            
            # Zone de substitution
            fig.add_annotation(
                x=max_val*0.8, y=max_val*0.3,
                text="Zone de substitution\n(Imports > Production)",
                showarrow=False, font=dict(size=12, color=colors['accent_red'])
            )
            
            fig.update_layout(
                height=500, template=plot_template,
                paper_bgcolor=colors['chart_bg'], plot_bgcolor=colors['chart_bg'],
                margin=dict(l=0, r=0, t=10, b=0),
                font=dict(color=colors['text_primary'], size=12),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                           bgcolor="rgba(0,0,0,0)" if is_dark else "rgba(255,255,255,0.9)",
                           font=dict(color=colors['text_primary'])),
                xaxis=dict(gridcolor=colors['border'] if is_dark else "rgba(0,0,0,0.1)",
                          tickfont=dict(color=colors['text_primary'], size=11),
                          title_font=dict(color=colors['text_primary'], size=12)),
                yaxis=dict(gridcolor=colors['border'] if is_dark else "rgba(0,0,0,0.1)",
                          tickfont=dict(color=colors['text_primary'], size=11),
                          title_font=dict(color=colors['text_primary'], size=12)),
                hoverlabel=dict(bgcolor="#1f2937" if is_dark else "#ffffff",
                               font_color="#f3f4f6" if is_dark else "#1f2937",
                               bordercolor=colors['border'])
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        with reco_tabs[1]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="ga-card">
                    <div class="ga-card-header">
                        <span class="ga-card-title"> Top Secteurs par Score</span>
                    </div>
                """, unsafe_allow_html=True)
                
                top_score = filtered_reco.nlargest(10, 'score_substitution')
                top_score['secteur_short'] = top_score['secteur'].apply(lambda x: x[:25] + '...' if len(x) > 25 else x)
                
                fig = go.Figure(go.Bar(
                    x=top_score['score_substitution'],
                    y=top_score['secteur_short'],
                    orientation='h',
                    marker=dict(
                        color=top_score['score_substitution'],
                        colorscale='Greens'
                    ),
                    text=top_score['score_substitution'].round(1),
                    textposition='outside'
                ))
                fig.update_layout(
                    height=400, template=plot_template,
                    paper_bgcolor=colors['chart_bg'], plot_bgcolor=colors['chart_bg'],
                    font=dict(color=colors['text_primary'], size=12),
                    margin=dict(l=0, r=50, t=10, b=0),
                    xaxis_title="Score de substitution",
                    xaxis=dict(gridcolor=colors['border'] if is_dark else "rgba(0,0,0,0.1)",
                              tickfont=dict(color=colors['text_primary'], size=11),
                              title_font=dict(color=colors['text_primary'], size=12)),
                    yaxis=dict(tickfont=dict(color=colors['text_primary'], size=11)),
                    hoverlabel=dict(bgcolor="#1f2937" if is_dark else "#ffffff",
                                   font_color="#f3f4f6" if is_dark else "#1f2937")
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="ga-card">
                    <div class="ga-card-header">
                        <span class="ga-card-title"> Top Secteurs par Potentiel</span>
                    </div>
                """, unsafe_allow_html=True)
                
                filtered_reco_copy = filtered_reco.copy()
                filtered_reco_copy['gap'] = filtered_reco_copy['imports_mds_fcfa'] - filtered_reco_copy['production_mds_fcfa']
                top_gap = filtered_reco_copy[filtered_reco_copy['gap'] > 0].nlargest(10, 'gap')
                top_gap['secteur_short'] = top_gap['secteur'].apply(lambda x: x[:25] + '...' if len(x) > 25 else x)
                
                if len(top_gap) > 0:
                    fig = go.Figure(go.Bar(
                        x=top_gap['gap'],
                        y=top_gap['secteur_short'],
                        orientation='h',
                        marker=dict(color=colors['accent_blue']),
                        text=top_gap['gap'].round(1),
                        textposition='outside'
                    ))
                    fig.update_layout(
                        height=400, template=plot_template,
                        paper_bgcolor=colors['chart_bg'], plot_bgcolor=colors['chart_bg'],
                        font=dict(color=colors['text_primary'], size=12),
                        margin=dict(l=0, r=50, t=10, b=0),
                        xaxis_title="Écart Import-Production (Mds FCFA)",
                        xaxis=dict(gridcolor=colors['border'] if is_dark else "rgba(0,0,0,0.1)",
                                  tickfont=dict(color=colors['text_primary'], size=11),
                                  title_font=dict(color=colors['text_primary'], size=12)),
                        yaxis=dict(tickfont=dict(color=colors['text_primary'], size=11)),
                        hoverlabel=dict(bgcolor="#1f2937" if is_dark else "#ffffff",
                                       font_color="#f3f4f6" if is_dark else "#1f2937")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Aucun secteur avec écart positif dans la sélection")
                st.markdown("</div>", unsafe_allow_html=True)
        
        with reco_tabs[2]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="ga-card">
                    <div class="ga-card-header">
                        <span class="ga-card-title"> Distribution des Scores</span>
                    </div>
                """, unsafe_allow_html=True)
                
                fig = px.histogram(
                    filtered_reco, x='score_substitution', nbins=20,
                    color_discrete_sequence=[colors['accent_blue']]
                )
                fig.add_vline(x=filtered_reco['score_substitution'].mean(), line_dash="dash", 
                             line_color=colors['accent_orange'], annotation_text="Moyenne")
                fig.update_layout(
                    height=300, template=plot_template,
                    paper_bgcolor=colors['chart_bg'], plot_bgcolor=colors['chart_bg'],
                    font=dict(color=colors['text_primary'], size=12),
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="Score", yaxis_title="Nombre de secteurs",
                    xaxis=dict(gridcolor=colors['border'] if is_dark else "rgba(0,0,0,0.1)",
                              tickfont=dict(color=colors['text_primary'], size=11),
                              title_font=dict(color=colors['text_primary'], size=12)),
                    yaxis=dict(gridcolor=colors['border'] if is_dark else "rgba(0,0,0,0.1)",
                              tickfont=dict(color=colors['text_primary'], size=11),
                              title_font=dict(color=colors['text_primary'], size=12)),
                    hoverlabel=dict(bgcolor="#1f2937" if is_dark else "#ffffff",
                                   font_color="#f3f4f6" if is_dark else "#1f2937")
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="ga-card">
                    <div class="ga-card-header">
                        <span class="ga-card-title"> Répartition par Classification</span>
                    </div>
                """, unsafe_allow_html=True)
                
                class_counts = filtered_reco['classification'].value_counts()
                colors_pie = [colors['accent_green'], colors['accent_yellow'], colors['accent_red']]
                
                fig = go.Figure(go.Pie(
                    labels=class_counts.index,
                    values=class_counts.values,
                    hole=0.4,
                    marker_colors=colors_pie[:len(class_counts)]
                ))
                fig.update_layout(
                    height=300, template=plot_template,
                    paper_bgcolor=colors['chart_bg'],
                    font=dict(color=colors['text_primary'], size=12),
                    margin=dict(l=0, r=0, t=10, b=0),
                    showlegend=True,
                    legend=dict(font=dict(color=colors['text_primary'], size=11),
                               bgcolor="rgba(0,0,0,0)" if is_dark else "rgba(255,255,255,0.9)")
                )
                fig.update_traces(
                    textfont=dict(color=colors['text_primary'] if is_dark else "#ffffff", size=12),
                    outsidetextfont=dict(color=colors['text_primary'], size=11)
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        
        with reco_tabs[3]:
            st.markdown(f"""
            <div class="ga-card">
                <div class="ga-card-header">
                    <span class="ga-card-title"> Tableau Détaillé des Recommandations</span>
                </div>
                <div class="ga-card-body">
            """, unsafe_allow_html=True)
            
            display_cols = ['secteur', 'score_substitution', 'production_mds_fcfa', 'imports_mds_fcfa', 'ratio_production_imports', 'croissance_production_pct', 'classification']
            st.dataframe(
                filtered_reco[display_cols].round(2), 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "secteur": "Secteur",
                    "score_substitution": st.column_config.ProgressColumn("Score", format="%.0f", min_value=0, max_value=100),
                    "production_mds_fcfa": st.column_config.NumberColumn("Production (Mds)", format="%.2f"),
                    "imports_mds_fcfa": st.column_config.NumberColumn("Imports (Mds)", format="%.2f"),
                    "ratio_production_imports": st.column_config.NumberColumn("Ratio P/I", format="%.2f"),
                    "croissance_production_pct": st.column_config.NumberColumn("Croissance %", format="%.1f%%"),
                    "classification": "Classification"
                }
            )
            
            st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Boutons d'export
            col1, col2 = st.columns(2)
            with col1:
                csv = filtered_reco.to_csv(index=False).encode('utf-8')
                st.download_button(" Télécharger CSV", csv, "recommandations.csv", "text/csv")
            with col2:
                import json
                json_data = filtered_reco.to_json(orient='records', force_ascii=False)
                st.download_button(" Télécharger JSON", json_data, "recommandations.json", "application/json")
# ============================================================
# Page: Simulateur Avancé
# ============================================================
elif page == "Simulateur":
    if data_loaded:
        # Initialiser les sessions states pour le simulateur
        if 'scenarios' not in st.session_state:
            st.session_state.scenarios = []
        if 'simulation_results' not in st.session_state:
            st.session_state.simulation_results = None
        
        # Fonction de prédiction
        def predict_score(production, production_tonnes, imports, exports, consommation, year=2023):
            features = {col: 0 for col in api_config['feature_columns']}
            features['production_fcfa'] = production
            features['production_tonnes'] = production_tonnes
            features['imports_fcfa'] = imports
            features['exports_fcfa'] = exports
            features['consommation_fcfa'] = consommation
            features['year'] = year
            features['balance_commerciale_fcfa'] = exports - imports
            features['taux_couverture'] = (exports / imports * 100) if imports > 0 else 0
            features['prix_unitaire_production'] = (production / production_tonnes) if production_tonnes > 0 else 0
            
            X = np.array([[features[col] for col in api_config['feature_columns']]])
            X_scaled = scaler.transform(X)
            
            score = float(model_reg.predict(X_scaled)[0])
            class_pred = int(model_clf.predict(X_scaled)[0])
            
            return score, class_pred, features
        
        # Onglets du simulateur
        sim_tabs = st.tabs([" Simulation Simple", " Multi-Scénarios", " Analyse Sensibilité", " Simulation Temporelle", " Export & Historique"])
        
        # ===========================================
        # ONGLET 1: SIMULATION SIMPLE
        # ===========================================
        with sim_tabs[0]:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"""
                <div class="ga-card">
                    <div class="ga-card-header">
                        <span class="ga-card-title"> Paramètres du secteur</span>
                    </div>
                    <div class="ga-card-body">
                """, unsafe_allow_html=True)
                
                # Sélection d'un secteur existant pour pré-remplir
                secteurs_list = ["-- Nouveau secteur --"] + list(df['LIBELLES'].unique())[:50]
                selected_sector = st.selectbox("Charger un secteur existant", secteurs_list, key="sector_select")
                
                if selected_sector != "-- Nouveau secteur --":
                    sector_data = df[df['LIBELLES'] == selected_sector].iloc[-1]
                    default_prod = float(sector_data.get('production_fcfa', 100))
                    default_prod_t = float(sector_data.get('production_tonnes', 500))
                    default_imp = float(sector_data.get('imports_fcfa', 80))
                    default_exp = float(sector_data.get('exports_fcfa', 50))
                    default_cons = float(sector_data.get('consommation_fcfa', 120))
                else:
                    default_prod, default_prod_t, default_imp, default_exp, default_cons = 100.0, 500.0, 80.0, 50.0, 120.0
                
                production = st.number_input("Production (Mds FCFA)", min_value=0.0, value=default_prod, step=10.0, key="sim_prod")
                production_tonnes = st.number_input("Production (Tonnes)", min_value=0.0, value=default_prod_t, step=50.0, key="sim_prod_t")
                imports = st.number_input("Importations (Mds FCFA)", min_value=0.0, value=default_imp, step=10.0, key="sim_imp")
                exports = st.number_input("Exportations (Mds FCFA)", min_value=0.0, value=default_exp, step=10.0, key="sim_exp")
                consommation = st.number_input("Consommation (Mds FCFA)", min_value=0.0, value=default_cons, step=10.0, key="sim_cons")
                
                predict_btn = st.button(" Analyser le potentiel", use_container_width=True, type="primary", key="predict_simple")
                
                st.markdown("</div></div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="ga-card">
                    <div class="ga-card-header">
                        <span class="ga-card-title"> Résultats de l'analyse</span>
                    </div>
                    <div class="ga-card-body" style="min-height: 40px;">
                """, unsafe_allow_html=True)
                
                if predict_btn:
                    try:
                        score, class_pred, features = predict_score(production, production_tonnes, imports, exports, consommation)
                        st.session_state.simulation_results = {
                            'score': score, 'class': class_pred, 'features': features,
                            'production': production, 'imports': imports, 'exports': exports
                        }
                        
                        class_info = {0: ("Faible", "red"), 1: ("Moyenne", "yellow"), 2: ("Haute", "green")}
                        label, badge_color = class_info[class_pred]
                        
                        # Gauge chart pour le score
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Score de Substitution", 'font': {'size': 16, 'color': colors['text_primary']}},
                            delta={'reference': 50, 'increasing': {'color': colors['accent_green']}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickcolor': colors['text_primary']},
                                'bar': {'color': colors['accent_blue']},
                                'bgcolor': colors['bg_secondary'],
                                'borderwidth': 2,
                                'bordercolor': colors['border'],
                                'steps': [
                                    {'range': [0, 33], 'color': 'rgba(234, 67, 53, 0.3)'},
                                    {'range': [33, 66], 'color': 'rgba(251, 188, 4, 0.3)'},
                                    {'range': [66, 100], 'color': 'rgba(52, 168, 83, 0.3)'}
                                ],
                                'threshold': {'line': {'color': colors['accent_orange'], 'width': 4}, 'thickness': 0.75, 'value': score}
                            }
                        ))
                        fig_gauge.update_layout(
                            height=250, template=plot_template,
                            paper_bgcolor=colors['chart_bg'], font={'color': colors['text_primary']},
                            margin=dict(l=20, r=20, t=50, b=20)
                        )
                        st.plotly_chart(fig_gauge, use_container_width=True)
                        
                        st.markdown(f"""
                        <div style="text-align: center; margin: 10px 0;">
                            <span class="badge badge-{badge_color}" style="font-size: 16px; padding: 10px 24px;">
                                Opportunité {label}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Recommandations personnalisées
                        st.markdown(f"<h4 style='color: {colors['text_primary']}; margin-top: 20px;'>💡 Recommandations</h4>", unsafe_allow_html=True)
                        
                        recommendations_list = []
                        if imports > production:
                            recommendations_list.append(("🔴", "Déficit de production", f"Augmenter la production de {((imports-production)/production*100):.0f}% pour couvrir les imports"))
                        if exports < imports * 0.5:
                            recommendations_list.append(("🟡", "Faibles exportations", "Développer les capacités d'export pour améliorer la balance commerciale"))
                        if score > 60:
                            recommendations_list.append(("🟢", "Fort potentiel", "Secteur prioritaire pour l'investissement en substitution"))
                        if production_tonnes > 0 and production / production_tonnes < 0.5:
                            recommendations_list.append(("🔵", "Valorisation faible", "Améliorer la transformation pour augmenter la valeur ajoutée"))
                        
                        for icon, title, desc in recommendations_list:
                            st.markdown(f"""
                            <div style="padding: 10px; margin: 5px 0; background: {colors['bg_secondary']}; border-radius: 8px;">
                                <strong>{icon} {title}</strong><br>
                                <span style="font-size: 13px; color: {colors['text_muted']};">{desc}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Erreur: {e}")
                else:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 60px; color: {colors['text_muted']};">
                        <p style="font-size: 48px;"></p>
                        <p>Entrez les données et cliquez sur "Analyser" pour obtenir la prédiction</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div></div>", unsafe_allow_html=True)
        
        # ===========================================
        # ONGLET 2: MULTI-SCÉNARIOS
        # ===========================================
        with sim_tabs[1]:
            st.markdown(f"""
            <div class="ga-card">
                <div class="ga-card-header">
                    <span class="ga-card-title"> Comparaison Multi-Scénarios</span>
                </div>
                <div class="ga-card-body">
            """, unsafe_allow_html=True)
            
            st.markdown("Créez et comparez plusieurs scénarios pour évaluer différentes stratégies d'investissement.")
            
            col_add, col_clear = st.columns([3, 1])
            with col_add:
                scenario_name = st.text_input("Nom du scénario", value=f"Scénario {len(st.session_state.scenarios)+1}", key="scenario_name")
            with col_clear:
                if st.button(" Effacer tout", key="clear_scenarios"):
                    st.session_state.scenarios = []
                    st.rerun()
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                sc_prod = st.number_input("Production (Mds)", min_value=0.0, value=100.0, step=10.0, key="sc_prod")
            with col2:
                sc_prod_t = st.number_input("Prod. Tonnes", min_value=0.0, value=500.0, step=50.0, key="sc_prod_t")
            with col3:
                sc_imp = st.number_input("Imports (Mds)", min_value=0.0, value=80.0, step=10.0, key="sc_imp")
            with col4:
                sc_exp = st.number_input("Exports (Mds)", min_value=0.0, value=50.0, step=10.0, key="sc_exp")
            with col5:
                sc_cons = st.number_input("Conso. (Mds)", min_value=0.0, value=120.0, step=10.0, key="sc_cons")
            
            if st.button("➕ Ajouter ce scénario", key="add_scenario", type="primary"):
                score, class_pred, _ = predict_score(sc_prod, sc_prod_t, sc_imp, sc_exp, sc_cons)
                st.session_state.scenarios.append({
                    'name': scenario_name, 'production': sc_prod, 'production_t': sc_prod_t,
                    'imports': sc_imp, 'exports': sc_exp, 'consommation': sc_cons,
                    'score': score, 'class': class_pred
                })
                st.success(f"Scénario '{scenario_name}' ajouté avec un score de {score:.1f}")
                st.rerun()
            
            st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Afficher les scénarios
            if st.session_state.scenarios:
                st.markdown(f"""
                <div class="ga-card">
                    <div class="ga-card-header">
                        <span class="ga-card-title"> Scénarios enregistrés ({len(st.session_state.scenarios)})</span>
                    </div>
                    <div class="ga-card-body">
                """, unsafe_allow_html=True)
                
                scenarios_df = pd.DataFrame(st.session_state.scenarios)
                class_labels = {0: "Faible", 1: "Moyenne", 2: "Haute"}
                scenarios_df['priorite'] = scenarios_df['class'].map(class_labels)
                
                st.dataframe(
                    scenarios_df[['name', 'production', 'imports', 'exports', 'score', 'priorite']].round(2),
                    use_container_width=True, hide_index=True,
                    column_config={
                        'name': 'Scénario', 'production': 'Production (Mds)',
                        'imports': 'Imports (Mds)', 'exports': 'Exports (Mds)',
                        'score': st.column_config.NumberColumn('Score', format="%.1f"),
                        'priorite': 'Priorité'
                    }
                )
                
                # Graphique comparatif
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Bar(
                    name='Score', x=scenarios_df['name'], y=scenarios_df['score'],
                    marker_color=colors['accent_blue'], text=scenarios_df['score'].round(1), textposition='outside'
                ))
                fig_compare.update_layout(
                    title="Comparaison des scores de substitution", height=350, template=plot_template,
                    paper_bgcolor=colors['chart_bg'], plot_bgcolor=colors['chart_bg'],
                    font=dict(color=colors['text_primary']), yaxis=dict(range=[0, 110])
                )
                st.plotly_chart(fig_compare, use_container_width=True)
                
                # Radar chart comparatif
                if len(st.session_state.scenarios) >= 2:
                    categories = ['Production', 'Imports', 'Exports', 'Score']
                    fig_radar = go.Figure()
                    for sc in st.session_state.scenarios[:5]:
                        values = [sc['production']/100, sc['imports']/100, sc['exports']/100, sc['score']/100]
                        fig_radar.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], name=sc['name'], fill='toself'))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 2])),
                        showlegend=True, height=400, template=plot_template,
                        paper_bgcolor=colors['chart_bg'], font=dict(color=colors['text_primary'])
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                st.markdown("</div></div>", unsafe_allow_html=True)
        
        # ===========================================
        # ONGLET 3: ANALYSE DE SENSIBILITÉ
        # ===========================================
        with sim_tabs[2]:
            st.markdown(f"""
            <div class="ga-card">
                <div class="ga-card-header">
                    <span class="ga-card-title"> Analyse de Sensibilité</span>
                </div>
                <div class="ga-card-body">
            """, unsafe_allow_html=True)
            
            st.markdown("Analysez comment le score de substitution varie en modifiant un paramètre à la fois.")
            
            col1, col2 = st.columns(2)
            with col1:
                base_prod = st.number_input("Production base (Mds)", min_value=0.0, value=100.0, step=10.0, key="sens_prod")
                base_imp = st.number_input("Imports base (Mds)", min_value=0.0, value=80.0, step=10.0, key="sens_imp")
                base_exp = st.number_input("Exports base (Mds)", min_value=0.0, value=50.0, step=10.0, key="sens_exp")
            with col2:
                base_prod_t = st.number_input("Production base (T)", min_value=0.0, value=500.0, step=50.0, key="sens_prod_t")
                base_cons = st.number_input("Consommation base (Mds)", min_value=0.0, value=120.0, step=10.0, key="sens_cons")
                variation_range = st.slider("Plage de variation (%)", min_value=10, max_value=100, value=50, step=10, key="var_range")
            
            if st.button("🔬 Lancer l'analyse de sensibilité", type="primary", key="run_sensitivity"):
                with st.spinner("Calcul en cours..."):
                    params = {
                        'Production': (base_prod, 'production'),
                        'Importations': (base_imp, 'imports'),
                        'Exportations': (base_exp, 'exports'),
                        'Consommation': (base_cons, 'consommation')
                    }
                    
                    sensitivity_data = []
                    variations = np.linspace(-variation_range, variation_range, 21)
                    
                    for param_name, (base_val, param_key) in params.items():
                        for var in variations:
                            multiplier = 1 + var/100
                            test_vals = {
                                'production': base_prod, 'production_t': base_prod_t,
                                'imports': base_imp, 'exports': base_exp, 'consommation': base_cons
                            }
                            if param_key == 'production':
                                test_vals['production'] = base_prod * multiplier
                            elif param_key == 'imports':
                                test_vals['imports'] = base_imp * multiplier
                            elif param_key == 'exports':
                                test_vals['exports'] = base_exp * multiplier
                            elif param_key == 'consommation':
                                test_vals['consommation'] = base_cons * multiplier
                            
                            score, _, _ = predict_score(
                                test_vals['production'], test_vals['production_t'],
                                test_vals['imports'], test_vals['exports'], test_vals['consommation']
                            )
                            sensitivity_data.append({
                                'Paramètre': param_name, 'Variation (%)': var, 'Score': score
                            })
                    
                    sens_df = pd.DataFrame(sensitivity_data)
                    
                    # Graphique de sensibilité
                    fig_sens = px.line(
                        sens_df, x='Variation (%)', y='Score', color='Paramètre',
                        title="Impact de chaque paramètre sur le score", markers=True
                    )
                    fig_sens.add_hline(y=50, line_dash="dash", line_color=colors['accent_yellow'], annotation_text="Seuil moyen")
                    fig_sens.update_layout(
                        height=450, template=plot_template,
                        paper_bgcolor=colors['chart_bg'], plot_bgcolor=colors['chart_bg'],
                        font=dict(color=colors['text_primary']), legend=dict(orientation="h", yanchor="bottom", y=1.02)
                    )
                    st.plotly_chart(fig_sens, use_container_width=True)
                    
                    # Tableau récapitulatif
                    st.markdown(f"<h4 style='color: {colors['text_primary']};'> Élasticités des paramètres</h4>", unsafe_allow_html=True)
                    elasticities = []
                    for param_name in params.keys():
                        param_data = sens_df[sens_df['Paramètre'] == param_name]
                        score_min = param_data['Score'].min()
                        score_max = param_data['Score'].max()
                        elasticity = (score_max - score_min) / (2 * variation_range) * 100
                        elasticities.append({'Paramètre': param_name, 'Score Min': score_min, 'Score Max': score_max, 'Élasticité': elasticity})
                    
                    elast_df = pd.DataFrame(elasticities).sort_values('Élasticité', ascending=False)
                    st.dataframe(elast_df.round(2), use_container_width=True, hide_index=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # ===========================================
        # ONGLET 4: SIMULATION TEMPORELLE
        # ===========================================
        with sim_tabs[3]:
            st.markdown(f"""
            <div class="ga-card">
                <div class="ga-card-header">
                    <span class="ga-card-title"> Simulation Temporelle</span>
                </div>
                <div class="ga-card-body">
            """, unsafe_allow_html=True)
            
            st.markdown("Projetez l'évolution du score de substitution sur plusieurs années avec des hypothèses de croissance.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<h5 style='color: {colors['text_primary']};'>Valeurs initiales (2024)</h5>", unsafe_allow_html=True)
                init_prod = st.number_input("Production initiale (Mds)", min_value=0.0, value=100.0, step=10.0, key="temp_prod")
                init_prod_t = st.number_input("Production initiale (T)", min_value=0.0, value=500.0, step=50.0, key="temp_prod_t")
                init_imp = st.number_input("Imports initiaux (Mds)", min_value=0.0, value=80.0, step=10.0, key="temp_imp")
                init_exp = st.number_input("Exports initiaux (Mds)", min_value=0.0, value=50.0, step=10.0, key="temp_exp")
                init_cons = st.number_input("Consommation initiale (Mds)", min_value=0.0, value=120.0, step=10.0, key="temp_cons")
            with col2:
                st.markdown(f"<h5 style='color: {colors['text_primary']};'>Taux de croissance annuels (%)</h5>", unsafe_allow_html=True)
                growth_prod = st.slider("Croissance production", min_value=-10.0, max_value=20.0, value=5.0, step=0.5, key="gr_prod")
                growth_imp = st.slider("Croissance imports", min_value=-10.0, max_value=20.0, value=2.0, step=0.5, key="gr_imp")
                growth_exp = st.slider("Croissance exports", min_value=-10.0, max_value=20.0, value=8.0, step=0.5, key="gr_exp")
                growth_cons = st.slider("Croissance consommation", min_value=-10.0, max_value=20.0, value=3.0, step=0.5, key="gr_cons")
                years_projection = st.slider("Années de projection", min_value=1, max_value=15, value=5, key="years_proj")
            
            if st.button(" Lancer la simulation temporelle", type="primary", key="run_temporal"):
                with st.spinner("Projection en cours..."):
                    temporal_data = []
                    prod, prod_t, imp, exp, cons = init_prod, init_prod_t, init_imp, init_exp, init_cons
                    
                    for year in range(2024, 2024 + years_projection + 1):
                        score, class_pred, _ = predict_score(prod, prod_t, imp, exp, cons, year)
                        temporal_data.append({
                            'Année': year, 'Production': prod, 'Imports': imp, 'Exports': exp,
                            'Consommation': cons, 'Score': score, 'Priorité': class_pred,
                            'Balance': exp - imp
                        })
                        prod *= (1 + growth_prod/100)
                        prod_t *= (1 + growth_prod/100)
                        imp *= (1 + growth_imp/100)
                        exp *= (1 + growth_exp/100)
                        cons *= (1 + growth_cons/100)
                    
                    temp_df = pd.DataFrame(temporal_data)
                    
                    # Graphique d'évolution du score
                    fig_temp = go.Figure()
                    fig_temp.add_trace(go.Scatter(
                        x=temp_df['Année'], y=temp_df['Score'], mode='lines+markers',
                        name='Score de substitution', line=dict(color=colors['accent_blue'], width=3),
                        marker=dict(size=10)
                    ))
                    fig_temp.add_hline(y=66, line_dash="dash", line_color=colors['accent_green'], annotation_text="Seuil haute priorité")
                    fig_temp.add_hline(y=33, line_dash="dash", line_color=colors['accent_red'], annotation_text="Seuil faible")
                    fig_temp.update_layout(
                        title="Évolution du score de substitution", height=350, template=plot_template,
                        paper_bgcolor=colors['chart_bg'], plot_bgcolor=colors['chart_bg'],
                        font=dict(color=colors['text_primary']), yaxis=dict(range=[0, 100])
                    )
                    st.plotly_chart(fig_temp, use_container_width=True)
                    
                    # Graphique des flux
                    fig_flux = go.Figure()
                    fig_flux.add_trace(go.Scatter(x=temp_df['Année'], y=temp_df['Production'], name='Production', line=dict(color=colors['accent_green'])))
                    fig_flux.add_trace(go.Scatter(x=temp_df['Année'], y=temp_df['Imports'], name='Imports', line=dict(color=colors['accent_red'])))
                    fig_flux.add_trace(go.Scatter(x=temp_df['Année'], y=temp_df['Exports'], name='Exports', line=dict(color=colors['accent_blue'])))
                    fig_flux.add_trace(go.Bar(x=temp_df['Année'], y=temp_df['Balance'], name='Balance commerciale', marker_color=colors['accent_yellow'], opacity=0.5))
                    fig_flux.update_layout(
                        title="Évolution des flux commerciaux", height=350, template=plot_template,
                        paper_bgcolor=colors['chart_bg'], plot_bgcolor=colors['chart_bg'],
                        font=dict(color=colors['text_primary']), barmode='overlay'
                    )
                    st.plotly_chart(fig_flux, use_container_width=True)
                    
                    # Tableau récapitulatif
                    st.markdown(f"<h4 style='color: {colors['text_primary']};'> Données projetées</h4>", unsafe_allow_html=True)
                    class_labels = {0: "🔴 Faible", 1: "🟡 Moyenne", 2: "🟢 Haute"}
                    temp_df['Priorité Label'] = temp_df['Priorité'].map(class_labels)
                    st.dataframe(
                        temp_df[['Année', 'Production', 'Imports', 'Exports', 'Balance', 'Score', 'Priorité Label']].round(2),
                        use_container_width=True, hide_index=True
                    )
                    
                    # Stocker pour export
                    st.session_state['temporal_results'] = temp_df
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # ===========================================
        # ONGLET 5: EXPORT & HISTORIQUE
        # ===========================================
        with sim_tabs[4]:
            st.markdown(f"""
            <div class="ga-card">
                <div class="ga-card-header">
                    <span class="ga-card-title"> Export des résultats</span>
                </div>
                <div class="ga-card-body">
            """, unsafe_allow_html=True)
            
            st.markdown("Exportez vos simulations en différents formats.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"<h5 style='color: {colors['text_primary']};'> Scénarios</h5>", unsafe_allow_html=True)
                if st.session_state.scenarios:
                    scenarios_df = pd.DataFrame(st.session_state.scenarios)
                    csv_scenarios = scenarios_df.to_csv(index=False).encode('utf-8')
                    st.download_button(" Télécharger scénarios (CSV)", csv_scenarios, "scenarios_simulation.csv", "text/csv")
                    
                    # Export JSON
                    json_scenarios = json.dumps(st.session_state.scenarios, indent=2, ensure_ascii=False)
                    st.download_button(" Télécharger scénarios (JSON)", json_scenarios, "scenarios_simulation.json", "application/json")
                else:
                    st.info("Aucun scénario enregistré. Créez des scénarios dans l'onglet Multi-Scénarios.")
            
            with col2:
                st.markdown(f"<h5 style='color: {colors['text_primary']};'> Projections temporelles</h5>", unsafe_allow_html=True)
                if 'temporal_results' in st.session_state:
                    temp_df = st.session_state['temporal_results']
                    csv_temporal = temp_df.to_csv(index=False).encode('utf-8')
                    st.download_button(" Télécharger projections (CSV)", csv_temporal, "projections_temporelles.csv", "text/csv")
                else:
                    st.info("Aucune projection temporelle. Lancez une simulation dans l'onglet Simulation Temporelle.")
            
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Données historiques de référence
            st.markdown(f"<h5 style='color: {colors['text_primary']};'> Données historiques de référence</h5>", unsafe_allow_html=True)
            st.markdown("Utilisez les données réelles pour calibrer vos simulations.")
            
            # Statistiques par secteur
            sector_stats = df.groupby('LIBELLES').agg({
                'production_fcfa': ['mean', 'std', 'min', 'max'],
                'imports_fcfa': ['mean', 'std'],
                'exports_fcfa': ['mean', 'std']
            }).round(2)
            sector_stats.columns = ['Prod. Moy', 'Prod. Std', 'Prod. Min', 'Prod. Max', 'Imp. Moy', 'Imp. Std', 'Exp. Moy', 'Exp. Std']
            sector_stats = sector_stats.reset_index().head(20)
            
            st.dataframe(sector_stats, use_container_width=True, hide_index=True)
            
            csv_ref = sector_stats.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Télécharger données de référence (CSV)", csv_ref, "donnees_reference.csv", "text/csv")
            
            st.markdown("</div></div>", unsafe_allow_html=True)

# ============================================================
# Page: Performance ML
# ============================================================
elif page == "Performance ML":
    if data_loaded:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="ga-card">
                <div class="ga-card-header">
                    <span class="ga-card-title">Modele de Regression</span>
                    <span class="badge badge-blue">XGBoost</span>
                </div>
                <div class="ga-card-body">
            """, unsafe_allow_html=True)
            
            # Nouveau format de métadonnées (version 2.0)
            try:
                # Accès direct à la structure metrics.regression
                if isinstance(model_metadata, dict) and 'metrics' in model_metadata:
                    metrics_reg = model_metadata['metrics'].get('regression', {})
                    if isinstance(metrics_reg, dict):
                        r2 = float(metrics_reg.get('r2', 0))
                        rmse = float(metrics_reg.get('rmse', 0))
                    else:
                        r2 = 0.0
                        rmse = 0.0
                elif isinstance(model_metadata, dict) and 'model_performance' in model_metadata:
                    # Ancien format (evaluation_report)
                    perf = model_metadata.get('model_performance', {})
                    reg_perf = perf.get('regression', {}).get('test', {})
                    r2 = float(reg_perf.get('R²', reg_perf.get('r2', 0)))
                    rmse = float(reg_perf.get('RMSE', reg_perf.get('rmse', 0)))
                else:
                    r2 = 0.0
                    rmse = 0.0
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                r2 = 0.0
                rmse = 0.0
            
            date_creation = model_metadata.get('date_creation', 'N/A')
            periode = model_metadata.get('periode_donnees', '2014-2025')
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("R2 Score", f"{r2:.4f}")
            col_b.metric("RMSE", f"{rmse:.4f}")
            col_c.metric("Periode", periode)
            
            st.markdown(f"""
            <div style="margin-top: 16px; padding: 12px; background: {colors['bg_secondary']}; border-radius: 8px;">
                <div style="font-size: 12px; color: {colors['text_muted']};">
                    Date d'entrainement: {date_creation}<br>
                    Algorithme: XGBoost Regressor
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="ga-card">
                <div class="ga-card-header">
                    <span class="ga-card-title">Modele de Classification</span>
                    <span class="badge badge-green">XGBoost</span>
                </div>
                <div class="ga-card-body">
            """, unsafe_allow_html=True)
            
            # Nouveau format de métadonnées (version 2.0)
            try:
                # Accès direct à la structure metrics.classification
                if isinstance(model_metadata, dict) and 'metrics' in model_metadata:
                    metrics_clf = model_metadata['metrics'].get('classification', {})
                    if isinstance(metrics_clf, dict):
                        accuracy = float(metrics_clf.get('accuracy', 0))
                    else:
                        accuracy = 0.0
                elif isinstance(model_metadata, dict) and 'model_performance' in model_metadata:
                    # Ancien format (evaluation_report)
                    perf = model_metadata.get('model_performance', {})
                    clf_perf = perf.get('classification', {})
                    accuracy = float(clf_perf.get('accuracy', 0))
                else:
                    accuracy = 0.0
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                accuracy = 0.0
            
            col_a, col_b = st.columns(2)
            col_a.metric("Accuracy", f"{accuracy:.4f}")
            col_b.metric("Classes", "3")
            
            st.markdown(f"""
            <div style="margin-top: 16px; padding: 12px; background: {colors['bg_secondary']}; border-radius: 8px;">
                <div style="font-size: 12px; color: {colors['text_muted']};">
                    Classes: Faible / Moyenne / Haute<br>
                    Algorithme: XGBoost Classifier
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Feature importance
        st.markdown(f"""
        <div class="ga-card">
            <div class="ga-card-header">
                <span class="ga-card-title">Importance des features</span>
            </div>
            <div class="ga-card-body">
        """, unsafe_allow_html=True)
        
        importance_reg = model_metadata.get('feature_importance_regression', {})
        if importance_reg:
            importance_df = pd.DataFrame([{'feature': k, 'importance': v} for k, v in importance_reg.items()])
            importance_df = importance_df.sort_values('importance', ascending=True).tail(15)
            
            fig = px.bar(
                importance_df, 
                x='importance', 
                y='feature', 
                orientation='h',
                color='importance',
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                height=400,
                showlegend=False,
                template=plot_template,
                paper_bgcolor=colors['chart_bg'],
                plot_bgcolor=colors['chart_bg'],
                margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_showscale=False,
                font=dict(color=colors['text_primary'], size=12),
                xaxis=dict(gridcolor=colors['border'] if is_dark else "rgba(0,0,0,0.1)",
                          tickfont=dict(color=colors['text_primary'], size=11), 
                          title_font=dict(color=colors['text_primary'], size=12)),
                yaxis=dict(tickfont=dict(color=colors['text_primary'], size=11)),
                hoverlabel=dict(bgcolor="#1f2937" if is_dark else "#ffffff",
                               font_color="#f3f4f6" if is_dark else "#1f2937")
            )
            st.plotly_chart(fig, use_container_width=True)
    
        st.markdown("</div></div>", unsafe_allow_html=True)

# ============================================================
# Page: Assistant IA
# ============================================================
elif page == "Assistant IA":
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    if 'rag_sources' not in st.session_state:
        st.session_state.rag_sources = []
    
    if data_loaded:
        # Charger le système RAG
        rag_system = get_rag_system() if RAG_AVAILABLE else None
        rag_active = rag_system is not None and rag_system.is_initialized
        
        def get_data_context():
            """Fallback si RAG non disponible."""
            top_production = df.groupby('LIBELLES')['production_fcfa'].sum().nlargest(5)
            top_imports = df.groupby('LIBELLES')['imports_fcfa'].sum().nlargest(5)
            
            reco_text = ""
            for _, row in recommendations.head(10).iterrows():
                reco_text += f"- {row['secteur']}: Score {row['score_substitution']:.1f}/100, {row['classification']}\n"
            
            docs_context = get_all_documents_context()
            
            context = f"""Tu es un économiste expert du Burkina Faso, spécialisé en politique commerciale, développement économique et substitution aux importations. Tu maîtrises les enjeux macro-économiques, les stratégies sectorielles, lanalyse des politiques publiques et les données statistiques du pays. 

REGLE CRITIQUE: NE MENTIONNE JAMAIS les sources, documents, rapports ou fichiers. Ne dis jamais "selon le document", "d'apres le rapport". Reponds naturellement comme si tu connaissais ces informations.

DONNEES (2014-2025):
- Production moyenne: {df.groupby('year')['production_fcfa'].sum().mean():.0f} Mds FCFA/an
- Imports moyens: {df.groupby('year')['imports_fcfa'].sum().mean():.0f} Mds FCFA/an
- Exports moyens: {df.groupby('year')['exports_fcfa'].sum().mean():.0f} Mds FCFA/an

TOP PRODUCTION: {', '.join([f"{s[:25]}" for s in top_production.head(3).index])}
TOP IMPORTS: {', '.join([f"{s[:25]}" for s in top_imports.head(3).index])}

SUBSTITUTION (top 5): {'; '.join([f"{r['secteur'][:20]}:Score{r['score_substitution']:.0f}" for _, r in recommendations.head(5).iterrows()])}

{f"INFORMATIONS SUPPLEMENTAIRES: {docs_context[:1000]}" if docs_context else ""}"""
            return context
        
        # Header avec statut RAG
        rag_status_badge = "badge-green" if rag_active else "badge-yellow"
        rag_status_text = "RAG Actif" if rag_active else "Mode Simple"
        
        st.markdown(f"""
        <div class="ga-card">
            <div class="ga-card-header">
                <span class="ga-card-title">Assistant IA - Expert Commerce</span>
                <div>
                    <span class="badge {rag_status_badge}">{rag_status_text}</span>
                </div>
            </div>
            <div class="ga-card-body">
        """, unsafe_allow_html=True)
        
        # Info RAG
        if rag_active:
            rag_stats = rag_system.get_stats()
            st.markdown(f"""
            <div style="padding: 12px; background: rgba(30, 142, 62, 0.1); border-radius: 8px; margin-bottom: 16px; font-size: 13px;">
                <strong>Systeme RAG:</strong> {rag_stats['total_documents']} documents indexes | 
                Modele: all-MiniLM-L6-v2 | 
                Sources: {', '.join(rag_stats.get('source_types', {}).keys())}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="padding: 12px; background: rgba(249, 171, 0, 0.1); border-radius: 8px; margin-bottom: 16px; font-size: 13px;">
                <strong>Mode Simple:</strong> Le systeme RAG n'est pas active. 
                Executez <code>python rag_system.py</code> pour indexer les documents.
            </div>
            """, unsafe_allow_html=True)
        
        # Messages
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_messages):
                if message["role"] == "user":
                    st.markdown(f"""
                    <div style="display: flex; justify-content: flex-end; margin-bottom: 12px;">
                        <div style="background: {colors['accent_blue']}; color: white; padding: 12px 16px; border-radius: 18px 18px 4px 18px; max-width: 70%;">
                            {message["content"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="display: flex; justify-content: flex-start; margin-bottom: 12px;">
                        <div style="background: {colors['bg_secondary']}; border: 1px solid {colors['border']}; color: {colors['text_primary']}; padding: 12px 16px; border-radius: 18px 18px 18px 4px; max-width: 70%;">
                            {message["content"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Sources non affichées automatiquement (disponibles uniquement si l'utilisateur demande)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Input - Utiliser un formulaire pour permettre l'envoi avec Entrée
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            with col1:
                user_input = st.text_input("Message...", key="chat_input", placeholder="Posez votre question sur les flux commerciaux...", label_visibility="collapsed")
            with col2:
                send_button = st.form_submit_button("Envoyer", use_container_width=True)
        
        if send_button and user_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            
            with st.spinner("Recherche et analyse en cours..." if rag_active else "Reflexion..."):
                try:
                    if rag_active:
                        # Utiliser le système RAG
                        result = rag_system.generate_response(user_input, top_k=15)
                        
                        if result["success"]:
                            assistant_response = result["answer"]
                            sources = result.get("sources", [])
                        else:
                            assistant_response = f"Erreur RAG: {result.get('error', 'Erreur inconnue')}"
                            sources = []
                    else:
                        # Fallback: mode simple
                        context = get_data_context()
                        messages = [
                            {"role": "system", "content": context},
                            {"role": "user", "content": user_input}
                        ]
                        
                        response = groq_client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=messages,
                            max_tokens=512,
                            temperature=0.7
                        )
                        assistant_response = response.choices[0].message.content
                        sources = []
                        
                except Exception as e:
                    assistant_response = f"Erreur: {str(e)}"
                    sources = []
            
            st.session_state.chat_messages.append({
                "role": "assistant", 
                "content": assistant_response,
                "sources": sources
            })
            st.rerun()
        
        # Bouton Effacer
        if st.button("Effacer la conversation"):
            st.session_state.chat_messages = []
            st.rerun()
        
        # Suggestions
        st.markdown(f"""
        <div style="margin-top: 16px; padding: 16px; background: {colors['bg_secondary']}; border-radius: 8px;">
            <div style="font-size: 12px; font-weight: 500; color: {colors['text_secondary']}; margin-bottom: 8px;">SUGGESTIONS</div>
            <div style="font-size: 13px; color: {colors['text_muted']};">
                - Quelles sont les opportunites de substitution aux importations?<br>
                - Quels secteurs ont le plus grand potentiel de croissance?<br>
                - Quelle est la balance commerciale du Burkina Faso?<br>
                - Quelles sont les donnees du rapport trimestriel DGD T2-2025?<br>
                - Resume les principales statistiques du commerce exterieur.<br>
                - Comment ameliorer la politique commerciale nationale?<br>
                - Quels sont les secteurs a prioriser pour l'investissement?<br>
                -Quelle était  la balance commercial du Burkina Faso en 2016 et 2025 ? et pourquoi ?
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# Footer
# ============================================================
st.markdown(f"""
<div style="margin-top: 48px; padding: 24px 0; border-top: 1px solid {colors['border']}; text-align: center;">
    <div style="font-size: 12px; color: {colors['text_muted']};">
        Analyseur Import/Export Burkina Faso - Propulse par IA
    </div>
</div>
""", unsafe_allow_html=True)
