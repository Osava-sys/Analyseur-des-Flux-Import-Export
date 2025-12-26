"""
Analyseur de Flux Import/Export - Burkina Faso
====================================================
Dashboard interactif inspiré de Google Analytics
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

# ============================================================
# Configuration Groq LLM
# ============================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Dossier des documents
DOCUMENTS_DIR = Path(__file__).parent / "documents"
DOCUMENTS_DIR.mkdir(exist_ok=True)

# ============================================================
# Fonctions pour les documents PDF
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
    documents_text = ""
    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        return ""
    for pdf_file in pdf_files[:3]:
        text = extract_text_from_pdf(pdf_file)
        if text and not text.startswith("Erreur"):
            text = ' '.join(text.split())
            documents_text += f"\n[{pdf_file.name}]: {text[:800]}..."
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
    st.session_state.theme = "light"

if 'page' not in st.session_state:
    st.session_state.page = "Accueil"

# ============================================================
# Définition des thèmes (Google Analytics Style)
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
            "bg_primary": "#ffffff",
            "bg_secondary": "#f8f9fa",
            "bg_card": "#ffffff",
            "bg_sidebar": "#ffffff",
            "text_primary": "#202124",
            "text_secondary": "#5f6368",
            "text_muted": "#80868b",
            "border": "#e8eaed",
            "accent_blue": "#1a73e8",
            "accent_orange": "#ea8600",
            "accent_green": "#1e8e3e",
            "accent_red": "#d93025",
            "accent_yellow": "#f9ab00",
            "chart_bg": "rgba(0,0,0,0)",
            "hover": "#f1f3f4",
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
</style>
""", unsafe_allow_html=True)

# ============================================================
# Chargement des données
# ============================================================
@st.cache_data
def load_data():
    DATA_PATH = Path("data/processed")
    MODELS_PATH = Path("models")
    df = pd.read_csv(DATA_PATH / "dataset_ml_complete.csv")
    recommendations = pd.read_csv(MODELS_PATH / "recommendations_report.csv")
    with open(MODELS_PATH / "model_metadata.json", 'r', encoding='utf-8') as f:
        model_metadata = json.load(f)
    with open(MODELS_PATH / "evaluation_report.json", 'r', encoding='utf-8') as f:
        evaluation_report = json.load(f)
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
    <div style="font-size: 13px; color: {colors['text_muted']};">Analyse des flux commerciaux - Burkina Faso - 2014-2023</div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="text-align: right; font-size: 13px; color: {colors['text_muted']};">
        Derniere mise a jour: Decembre 2023
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
                    <span style="font-size: 12px; color: {colors['text_muted']};">2014 - 2023</span>
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
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                    font=dict(size=12)
                ),
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis=dict(
                    showgrid=True,
                    gridcolor=colors['border'],
                    title=""
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor=colors['border'],
                    title="Milliards FCFA"
                ),
                hovermode='x unified'
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
                margin=dict(l=20, r=20, t=20, b=20)
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
        st.markdown(f"""
        <div class="realtime-card" style="max-width: 300px;">
            <div class="realtime-title">Donnees actives en temps reel</div>
            <div class="realtime-value">{len(df)}</div>
            <div class="realtime-label">Enregistrements dans la base</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="ga-card">
                <div class="ga-card-header">
                    <span class="ga-card-title">Secteurs par production</span>
                </div>
            """, unsafe_allow_html=True)
            
            top_prod = df.groupby('LIBELLES')['production_fcfa'].sum().nlargest(10).reset_index()
            fig = px.bar(
                top_prod, 
                x='production_fcfa', 
                y='LIBELLES',
                orientation='h',
                color_discrete_sequence=[colors['accent_blue']]
            )
            fig.update_layout(
                height=400,
                template=plot_template,
                paper_bgcolor=colors['chart_bg'],
                plot_bgcolor=colors['chart_bg'],
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Production (FCFA)",
                yaxis_title="",
                showlegend=False
            )
            fig.update_yaxes(ticktext=[t[:30]+"..." if len(t) > 30 else t for t in top_prod['LIBELLES']])
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="ga-card">
                <div class="ga-card-header">
                    <span class="ga-card-title">Secteurs par importations</span>
                </div>
            """, unsafe_allow_html=True)
            
            top_imports = df.groupby('LIBELLES')['imports_fcfa'].sum().nlargest(10).reset_index()
            fig = px.bar(
                top_imports, 
                x='imports_fcfa', 
                y='LIBELLES',
                orientation='h',
                color_discrete_sequence=[colors['accent_red']]
            )
            fig.update_layout(
                height=400,
                template=plot_template,
                paper_bgcolor=colors['chart_bg'],
                plot_bgcolor=colors['chart_bg'],
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Importations (FCFA)",
                yaxis_title="",
                showlegend=False
            )
            fig.update_yaxes(ticktext=[t[:30]+"..." if len(t) > 30 else t for t in top_imports['LIBELLES']])
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Page: Analyse Sectorielle
# ============================================================
elif page == "Analyse":
    if data_loaded:
        st.markdown(f"""
        <div class="ga-card">
            <div class="ga-card-header">
                <span class="ga-card-title">Analyse sectorielle detaillee</span>
            </div>
            <div class="ga-card-body">
        """, unsafe_allow_html=True)
        
        selected_sector = st.selectbox("Selectionner un secteur", df['LIBELLES'].unique().tolist())
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        if selected_sector:
            sector_data = df[df['LIBELLES'] == selected_sector].sort_values('year')
            latest = sector_data[sector_data['year'] == sector_data['year'].max()].iloc[0]
            
            # Métriques du secteur
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Production</div>
                    <div class="metric-value">{latest['production_fcfa']:.1f} Mds</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Importations</div>
                    <div class="metric-value">{latest['imports_fcfa']:.1f} Mds</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Exportations</div>
                    <div class="metric-value">{latest['exports_fcfa']:.1f} Mds</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                growth = latest.get('production_fcfa_growth', 0)
                delta_class = "positive" if growth > 0 else "negative"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Croissance</div>
                    <div class="metric-value">{growth:+.1f}%</div>
                    <div class="metric-delta {delta_class}">vs annee precedente</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="ga-card">
                    <div class="ga-card-header">
                        <span class="ga-card-title">Evolution historique</span>
                    </div>
                    <div class="ga-card-body">
                """, unsafe_allow_html=True)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sector_data['year'], 
                    y=sector_data['production_fcfa'],
                    mode='lines+markers', 
                    name='Production',
                    line=dict(width=2, color=colors['accent_blue']),
                    fill='tozeroy',
                    fillcolor=f"rgba(26, 115, 232, 0.1)"
                ))
                fig.add_trace(go.Scatter(
                    x=sector_data['year'], 
                    y=sector_data['imports_fcfa'],
                    mode='lines+markers', 
                    name='Imports',
                    line=dict(width=2, color=colors['accent_red'])
                ))
                fig.add_trace(go.Scatter(
                    x=sector_data['year'], 
                    y=sector_data['exports_fcfa'],
                    mode='lines+markers', 
                    name='Exports',
                    line=dict(width=2, color=colors['accent_green'])
                ))
                
                fig.update_layout(
                    height=300,
                    template=plot_template,
                    paper_bgcolor=colors['chart_bg'],
                    plot_bgcolor=colors['chart_bg'],
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    margin=dict(l=0, r=0, t=30, b=0),
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("</div></div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="ga-card">
                    <div class="ga-card-header">
                        <span class="ga-card-title">Balance commerciale</span>
                    </div>
                    <div class="ga-card-body">
                """, unsafe_allow_html=True)
                
                sector_data = sector_data.copy()
                sector_data['balance'] = sector_data['exports_fcfa'] - sector_data['imports_fcfa']
                bar_colors = [colors['accent_green'] if x >= 0 else colors['accent_red'] for x in sector_data['balance']]
                
                fig = go.Figure(go.Bar(
                    x=sector_data['year'], 
                    y=sector_data['balance'], 
                    marker_color=bar_colors
                ))
                fig.update_layout(
                    height=300,
                    template=plot_template,
                    paper_bgcolor=colors['chart_bg'],
                    plot_bgcolor=colors['chart_bg'],
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis_title="Milliards FCFA"
                )
                st.plotly_chart(fig, use_container_width=True)
    
                st.markdown("</div></div>", unsafe_allow_html=True)

# ============================================================
# Page: Recommandations
# ============================================================
elif page == "Recommandations":
    if data_loaded:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            filter_class = st.multiselect(
                "Classification", 
                recommendations['classification'].unique(), 
                default=recommendations['classification'].unique()
            )
        
        with col2:
            top_n = st.slider("Nombre de secteurs", 5, 22, 10)
        
        filtered_reco = recommendations[recommendations['classification'].isin(filter_class)].head(top_n)
        
        st.markdown(f"""
        <div class="ga-card">
            <div class="ga-card-header">
                <span class="ga-card-title">Cartographie des opportunites</span>
            </div>
            <div class="ga-card-body">
        """, unsafe_allow_html=True)
        
        fig = px.scatter(
            filtered_reco, 
            x='imports_mds_fcfa', 
            y='production_mds_fcfa',
            size='score_substitution', 
            color='croissance_production_pct',
            hover_name='secteur', 
            color_continuous_scale='RdYlGn',
            labels={
                'imports_mds_fcfa': 'Importations (Mds FCFA)', 
                'production_mds_fcfa': 'Production (Mds FCFA)', 
                'croissance_production_pct': 'Croissance (%)'
            }
        )
        
        max_val = max(filtered_reco['imports_mds_fcfa'].max(), filtered_reco['production_mds_fcfa'].max())
        fig.add_trace(go.Scatter(
            x=[0, max_val], 
            y=[0, max_val], 
            mode='lines', 
            line=dict(color=colors['text_muted'], dash='dash', width=1), 
            name='Parite',
            showlegend=False
        ))
        
        fig.update_layout(
            height=450,
            template=plot_template,
            paper_bgcolor=colors['chart_bg'],
            plot_bgcolor=colors['chart_bg'],
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Tableau
        st.markdown(f"""
        <div class="ga-card">
            <div class="ga-card-header">
                <span class="ga-card-title">Tableau des recommandations</span>
            </div>
            <div class="ga-card-body">
        """, unsafe_allow_html=True)
        
        display_cols = ['secteur', 'production_mds_fcfa', 'imports_mds_fcfa', 'ratio_production_imports', 'croissance_production_pct', 'classification']
        st.dataframe(
            filtered_reco[display_cols].round(2), 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "secteur": "Secteur",
                "production_mds_fcfa": st.column_config.NumberColumn("Production (Mds)", format="%.2f"),
                "imports_mds_fcfa": st.column_config.NumberColumn("Imports (Mds)", format="%.2f"),
                "ratio_production_imports": st.column_config.NumberColumn("Ratio P/I", format="%.2f"),
                "croissance_production_pct": st.column_config.NumberColumn("Croissance %", format="%.1f%%"),
                "classification": "Classification"
            }
        )
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        csv = filtered_reco.to_csv(index=False).encode('utf-8')
        st.download_button("Telecharger CSV", csv, "recommandations.csv", "text/csv")

# ============================================================
# Page: Simulateur
# ============================================================
elif page == "Simulateur":
    if data_loaded:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="ga-card">
                <div class="ga-card-header">
                    <span class="ga-card-title">Donnees du secteur</span>
                </div>
                <div class="ga-card-body">
            """, unsafe_allow_html=True)
            
            production = st.number_input("Production (Mds FCFA)", min_value=0.0, value=100.0, step=10.0)
            production_tonnes = st.number_input("Production (Tonnes)", min_value=0.0, value=500.0, step=50.0)
            imports = st.number_input("Importations (Mds FCFA)", min_value=0.0, value=80.0, step=10.0)
            exports = st.number_input("Exportations (Mds FCFA)", min_value=0.0, value=50.0, step=10.0)
            consommation = st.number_input("Consommation (Mds FCFA)", min_value=0.0, value=120.0, step=10.0)
            
            predict_btn = st.button("Analyser le potentiel", use_container_width=True, type="primary")
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="ga-card">
                <div class="ga-card-header">
                    <span class="ga-card-title">Resultats de l'analyse</span>
                </div>
                <div class="ga-card-body" style="min-height: 300px;">
            """, unsafe_allow_html=True)
            
            if predict_btn:
                try:
                    features = {col: 0 for col in api_config['feature_columns']}
                    features['production_fcfa'] = production
                    features['production_tonnes'] = production_tonnes
                    features['imports_fcfa'] = imports
                    features['exports_fcfa'] = exports
                    features['consommation_fcfa'] = consommation
                    features['year'] = 2023
                    features['balance_commerciale_fcfa'] = exports - imports
                    features['taux_couverture'] = (exports / imports * 100) if imports > 0 else 0
                    features['prix_unitaire_production'] = (production / production_tonnes) if production_tonnes > 0 else 0
                    
                    X = np.array([[features[col] for col in api_config['feature_columns']]])
                    X_scaled = scaler.transform(X)
                    
                    score = float(model_reg.predict(X_scaled)[0])
                    class_pred = int(model_clf.predict(X_scaled)[0])
                    
                    class_info = {
                        0: ("Faible", colors['accent_red']),
                        1: ("Moyenne", colors['accent_yellow']),
                        2: ("Haute", colors['accent_green'])
                    }
                    label, color = class_info[class_pred]
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px;">
                        <div style="font-size: 36px; font-weight: 400; color: {colors['text_primary']}; margin-bottom: 8px;">{score:.1f}/100</div>
                        <div class="badge badge-{'green' if class_pred == 2 else 'yellow' if class_pred == 1 else 'red'}" style="font-size: 14px; padding: 8px 20px;">
                            Opportunite {label}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if class_pred == 2:
                        st.success("Secteur a fort potentiel - Priorite d'investissement recommandee")
                    elif class_pred == 1:
                        st.warning("Potentiel modere - Analyser les opportunites specifiques")
                    else:
                        st.info("Faible potentiel actuel - Identifier les niches de marche")
                    
                    if imports > production:
                        st.error("Deficit de production - Forte opportunite de substitution!")
                        
                except Exception as e:
                    st.error(f"Erreur: {e}")
            else:
                st.markdown(f"""
                <div style="text-align: center; padding: 40px; color: {colors['text_muted']};">
                    <p>Entrez les donnees et cliquez sur "Analyser" pour obtenir la prediction</p>
                </div>
                """, unsafe_allow_html=True)
            
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
            
            metrics = model_metadata['regression_model']['metrics']
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("R2 Score", f"{metrics['r2']:.4f}")
            col_b.metric("RMSE", f"{metrics['rmse']:.4f}")
            col_c.metric("MAE", f"{metrics['mae']:.4f}")
            
            st.markdown(f"""
            <div style="margin-top: 16px; padding: 12px; background: {colors['bg_secondary']}; border-radius: 8px;">
                <div style="font-size: 12px; color: {colors['text_muted']};">
                    Date d'entrainement: {model_metadata['date_training']}<br>
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
            
            metrics = model_metadata['classification_model']['metrics']
            
            col_a, col_b = st.columns(2)
            col_a.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            col_b.metric("F1-Score", f"{metrics['f1_macro']:.4f}")
            
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
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
        st.markdown("</div></div>", unsafe_allow_html=True)

# ============================================================
# Page: Assistant IA
# ============================================================
elif page == "Assistant IA":
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    if data_loaded:
        def get_data_context():
            top_production = df.groupby('LIBELLES')['production_fcfa'].sum().nlargest(5)
            top_imports = df.groupby('LIBELLES')['imports_fcfa'].sum().nlargest(5)
            
            reco_text = ""
            for _, row in recommendations.head(10).iterrows():
                reco_text += f"- {row['secteur']}: Score {row['score_substitution']:.1f}/100, {row['classification']}\n"
            
            docs_context = get_all_documents_context()
            
            context = f"""Tu es un economiste expert du Burkina Faso, specialise en politique commerciale et substitution aux importations. Reponds en francais, sois concis et cite les chiffres.

DONNEES (2014-2023):
- Production moyenne: {df.groupby('year')['production_fcfa'].sum().mean():.0f} Mds FCFA/an
- Imports moyens: {df.groupby('year')['imports_fcfa'].sum().mean():.0f} Mds FCFA/an
- Exports moyens: {df.groupby('year')['exports_fcfa'].sum().mean():.0f} Mds FCFA/an

TOP PRODUCTION: {', '.join([f"{s[:25]}" for s in top_production.head(3).index])}
TOP IMPORTS: {', '.join([f"{s[:25]}" for s in top_imports.head(3).index])}

SUBSTITUTION (top 5): {'; '.join([f"{r['secteur'][:20]}:Score{r['score_substitution']:.0f}" for _, r in recommendations.head(5).iterrows()])}

{f"DOCS: {docs_context[:1000]}" if docs_context else ""}"""
            return context
        
        st.markdown(f"""
        <div class="ga-card">
            <div class="ga-card-header">
                <span class="ga-card-title">Assistant IA - Expert Commerce</span>
                <span class="badge badge-green">Actif</span>
            </div>
            <div class="ga-card-body">
        """, unsafe_allow_html=True)
        
        # Messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_messages:
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
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Input
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input("Message...", key="chat_input", placeholder="Posez votre question sur les flux commerciaux...", label_visibility="collapsed")
        with col2:
            send_button = st.button("Envoyer", use_container_width=True)
        
        if send_button and user_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            
            with st.spinner("Reflexion..."):
                try:
                    if not groq_client:
                        assistant_response = "Erreur: Cle API Groq non configuree. Definissez la variable d'environnement GROQ_API_KEY."
                    else:
                        context = get_data_context()
                        messages = [
                            {"role": "system", "content": context},
                            {"role": "user", "content": user_input}
                        ]
                        
                        response = groq_client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=messages,
                            max_tokens=512,
                            temperature=0.7
                        )
                        assistant_response = response.choices[0].message.content
                except Exception as e:
                    assistant_response = f"Erreur: {str(e)}"
            
            st.session_state.chat_messages.append({"role": "assistant", "content": assistant_response})
            st.rerun()
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Effacer"):
                st.session_state.chat_messages = []
                st.rerun()
        
        # Suggestions
        st.markdown(f"""
        <div style="margin-top: 16px; padding: 16px; background: {colors['bg_secondary']}; border-radius: 8px;">
            <div style="font-size: 12px; font-weight: 500; color: {colors['text_secondary']}; margin-bottom: 8px;">SUGGESTIONS</div>
            <div style="font-size: 13px; color: {colors['text_muted']};">
                - Quelles sont les opportunites de substitution?<br>
                - Quels secteurs ont le plus grand potentiel?<br>
                - Quelle est la balance commerciale?
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# Footer
# ============================================================
st.markdown(f"""
<div style="margin-top: 48px; padding: 24px 0; border-top: 1px solid {colors['border']}; text-align: center;">
    <div style="font-size: 12px; color: {colors['text_muted']};">
        Analyseur Import/Export Burkina Faso - Hackathon 2025 - Propulse par XGBoost & Groq LLM
    </div>
</div>
""", unsafe_allow_html=True)
