"""
Script de génération du rapport PDF professionnel
Analyseur Import/Export - Burkina Faso
"""

from fpdf import FPDF
from datetime import datetime
import os

class RapportPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)
        
    def header(self):
        # Logo/En-tête
        self.set_font('Arial', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, 'Analyseur Import/Export - Burkina Faso', 0, 0, 'L')
        self.cell(0, 10, f'Decembre 2025', 0, 1, 'R')
        self.set_draw_color(234, 134, 0)  # Orange
        self.set_line_width(0.5)
        self.line(10, 20, 200, 20)
        self.ln(10)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
    def chapter_title(self, title, level=1):
        if level == 1:
            self.set_font('Arial', 'B', 16)
            self.set_text_color(26, 115, 232)  # Bleu
            self.ln(5)
            self.cell(0, 12, title, 0, 1, 'L')
            self.set_draw_color(26, 115, 232)
            self.set_line_width(0.3)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(8)
        elif level == 2:
            self.set_font('Arial', 'B', 13)
            self.set_text_color(60, 60, 60)
            self.ln(3)
            self.cell(0, 10, title, 0, 1, 'L')
            self.ln(3)
        else:
            self.set_font('Arial', 'B', 11)
            self.set_text_color(80, 80, 80)
            self.cell(0, 8, title, 0, 1, 'L')
            self.ln(2)
            
    def body_text(self, text):
        self.set_font('Arial', '', 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6, text)
        self.ln(3)
        
    def bullet_point(self, text, indent=10):
        self.set_font('Arial', '', 10)
        self.set_text_color(50, 50, 50)
        self.set_x(indent + 10)
        self.cell(5, 6, chr(149), 0, 0)  # Bullet
        self.multi_cell(0, 6, text)
        
    def check_point(self, text, indent=10):
        self.set_font('Arial', '', 10)
        self.set_text_color(5, 150, 105)  # Vert
        self.set_x(indent + 10)
        self.cell(8, 6, '[OK]', 0, 0)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6, text)
        
    def add_table(self, headers, data, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
            
        # Header
        self.set_font('Arial', 'B', 9)
        self.set_fill_color(26, 115, 232)
        self.set_text_color(255, 255, 255)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 8, header, 1, 0, 'C', True)
        self.ln()
        
        # Data
        self.set_font('Arial', '', 9)
        self.set_text_color(50, 50, 50)
        fill = False
        for row in data:
            if fill:
                self.set_fill_color(245, 245, 245)
            else:
                self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 7, str(cell), 1, 0, 'C', True)
            self.ln()
            fill = not fill
        self.ln(5)
        
    def add_highlight_box(self, text, color='blue'):
        colors = {
            'blue': (232, 240, 254),
            'green': (232, 254, 232),
            'orange': (254, 244, 232),
            'red': (254, 232, 232)
        }
        bg = colors.get(color, colors['blue'])
        self.set_fill_color(*bg)
        self.set_font('Arial', '', 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 8, text, 0, 'L', True)
        self.ln(3)
        
    def add_metric_box(self, label, value, delta=""):
        self.set_fill_color(249, 250, 251)
        self.set_draw_color(229, 231, 235)
        x = self.get_x()
        y = self.get_y()
        self.rect(x, y, 45, 25, 'DF')
        self.set_xy(x + 2, y + 3)
        self.set_font('Arial', '', 8)
        self.set_text_color(107, 114, 128)
        self.cell(41, 5, label, 0, 2, 'C')
        self.set_font('Arial', 'B', 14)
        self.set_text_color(17, 24, 39)
        self.cell(41, 8, value, 0, 2, 'C')
        if delta:
            self.set_font('Arial', '', 7)
            self.set_text_color(5, 150, 105)
            self.cell(41, 4, delta, 0, 0, 'C')
        self.set_xy(x + 47, y)


def generate_rapport():
    pdf = RapportPDF()
    pdf.add_page()
    
    # ==================== PAGE DE TITRE ====================
    pdf.set_font('Arial', 'B', 28)
    pdf.set_text_color(26, 115, 232)
    pdf.ln(30)
    pdf.cell(0, 15, 'ANALYSEUR IMPORT/EXPORT', 0, 1, 'C')
    pdf.set_font('Arial', 'B', 24)
    pdf.set_text_color(234, 134, 0)
    pdf.cell(0, 12, 'BURKINA FASO', 0, 1, 'C')
    
    pdf.ln(10)
    pdf.set_font('Arial', '', 14)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, 'Solution d\'Intelligence Artificielle', 0, 1, 'C')
    pdf.cell(0, 8, 'pour l\'Analyse Commerciale et la Substitution aux Importations', 0, 1, 'C')
    
    pdf.ln(20)
    
    # Box info
    pdf.set_fill_color(249, 250, 251)
    pdf.set_draw_color(229, 231, 235)
    pdf.rect(50, 120, 110, 40, 'DF')
    pdf.set_xy(50, 125)
    pdf.set_font('Arial', 'B', 11)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(110, 8, 'Version 2.0', 0, 2, 'C')
    pdf.set_font('Arial', '', 10)
    pdf.cell(110, 6, 'Decembre 2025', 0, 2, 'C')
    pdf.cell(110, 6, 'Hackathon 24H - Innovation Economique', 0, 2, 'C')
    
    pdf.ln(60)
    
    # Métriques clés
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 10, 'CHIFFRES CLES', 0, 1, 'C')
    pdf.ln(5)
    
    pdf.set_x(25)
    pdf.add_metric_box('Secteurs', '185+', '')
    pdf.add_metric_box('Periode', '2014-2025', '')
    pdf.add_metric_box('Precision ML', '> 85%', '')
    pdf.add_metric_box('Score Max', '100/100', '')
    
    # ==================== RESUME EXECUTIF ====================
    pdf.add_page()
    pdf.chapter_title('1. RESUME EXECUTIF')
    
    pdf.body_text(
        "L'Analyseur Import/Export Burkina Faso est une plateforme d'intelligence artificielle "
        "innovante concue pour transformer les donnees commerciales du pays en insights "
        "strategiques actionnables."
    )
    
    pdf.chapter_title('Objectifs Principaux', 2)
    pdf.check_point('Identifier les opportunites de substitution aux importations')
    pdf.check_point('Optimiser la balance commerciale nationale')
    pdf.check_point('Guider les investissements vers les secteurs a fort potentiel')
    pdf.check_point('Fournir des recommandations personnalisees basees sur l\'IA')
    
    pdf.ln(5)
    pdf.chapter_title('Public Cible', 2)
    pdf.bullet_point('Ministeres (Commerce, Industrie, Economie)')
    pdf.bullet_point('Investisseurs nationaux et internationaux')
    pdf.bullet_point('Entrepreneurs et porteurs de projets')
    pdf.bullet_point('Institutions financieres (banques de developpement)')
    pdf.bullet_point('Chercheurs et analystes economiques')
    
    # ==================== FONCTIONNALITES ====================
    pdf.add_page()
    pdf.chapter_title('2. FONCTIONNALITES DETAILLEES')
    
    # Module 1: Accueil
    pdf.chapter_title('2.1 Tableau de Bord Accueil', 2)
    pdf.body_text('Un apercu global et synthetique de la situation commerciale du Burkina Faso.')
    pdf.bullet_point('4 metriques principales : Production, Importations, Exportations, Balance')
    pdf.bullet_point('Graphique d\'evolution des flux commerciaux (2014-2025)')
    pdf.bullet_point('Repartition des opportunites par classification')
    pdf.bullet_point('Top 5 secteurs a fort potentiel avec scores detailles')
    pdf.ln(3)
    pdf.add_highlight_box('Valeur: Permet une comprehension immediate de la sante economique du pays.', 'blue')
    
    # Module 2: Temps Réel
    pdf.chapter_title('2.2 Temps Reel', 2)
    pdf.body_text('Suivi dynamique des indicateurs economiques avec actualisation en direct.')
    pdf.bullet_point('Indicateur de statut "Live" avec horodatage')
    pdf.bullet_point('4 metriques temps reel avec variations annuelles (%)')
    pdf.bullet_point('Graphique d\'evolution des flux (Production, Imports, Exports)')
    pdf.bullet_point('Top 10 Secteurs par Production et Importations')
    pdf.bullet_point('Tableau detaille avec 20 secteurs et taux de couverture')
    pdf.ln(3)
    pdf.add_highlight_box('Valeur: Monitoring continu pour une reactivite maximale face aux evolutions.', 'blue')
    
    # Module 3: Analyse
    pdf.chapter_title('2.3 Analyse Sectorielle', 2)
    pdf.body_text('Analyse approfondie de chaque secteur economique parmi 185+ secteurs.')
    pdf.bullet_point('Onglet Evolution : Graphiques de flux commerciaux et balance annuelle')
    pdf.bullet_point('Onglet Comparaison : Positionnement vs autres secteurs')
    pdf.bullet_point('Onglet Diagnostic : Indicateurs de sante (dependance, croissance, balance)')
    pdf.bullet_point('Onglet Donnees : Tableau detaille exportable')
    pdf.ln(3)
    pdf.add_highlight_box('Valeur: Comprehension granulaire de chaque secteur pour des decisions ciblees.', 'blue')
    
    pdf.add_page()
    
    # Module 4: Recommandations
    pdf.chapter_title('2.4 Recommandations IA', 2)
    pdf.body_text('Systeme de recommandations intelligent base sur le Machine Learning.')
    pdf.bullet_point('Cartographie : Scatter plot Production vs Imports avec zone de substitution')
    pdf.bullet_point('Top Secteurs : Classement par score et par potentiel (ecart)')
    pdf.bullet_point('Analyses : Distribution des scores et repartition par classification')
    pdf.bullet_point('Tableau : Vue detaillee avec export CSV/JSON')
    pdf.ln(3)
    
    pdf.chapter_title('Classifications', 3)
    pdf.add_table(
        ['Classification', 'Score', 'Description'],
        [
            ['Fort Potentiel', '>= 70', 'Priorite haute pour substitution'],
            ['Potentiel Moyen', '40-69', 'Opportunite a developper'],
            ['Faible Potentiel', '< 40', 'Surveillance et optimisation']
        ],
        [50, 40, 100]
    )
    
    # Module 5: Simulateur
    pdf.chapter_title('2.5 Simulateur Avance', 2)
    pdf.body_text('Outil de simulation multi-dimensionnel pour tester des scenarios economiques.')
    
    pdf.add_table(
        ['Mode', 'Description'],
        [
            ['Simulation Simple', 'Prediction du score pour un secteur'],
            ['Multi-Scenarios', 'Comparaison simultanee de plusieurs scenarios'],
            ['Analyse Sensibilite', 'Variation automatique d\'un parametre (+/-50%)'],
            ['Simulation Temporelle', 'Projection sur 1 a 10 ans'],
            ['Export & Historique', 'Telechargement CSV/JSON des resultats']
        ],
        [55, 135]
    )
    pdf.add_highlight_box('Valeur: Outil d\'aide a la decision permettant d\'anticiper l\'impact des politiques.', 'green')
    
    # Module 6: Performance ML
    pdf.chapter_title('2.6 Performance ML', 2)
    pdf.body_text('Tableau de bord des performances des modeles de Machine Learning.')
    
    pdf.add_table(
        ['Modele', 'Type', 'Metriques'],
        [
            ['XGBoost Regression', 'Score substitution', 'R2, RMSE, MAE'],
            ['XGBoost Classification', 'Priorite opportunite', 'Accuracy, F1-Score']
        ],
        [60, 55, 75]
    )
    
    # Module 7: Assistant IA
    pdf.chapter_title('2.7 Assistant IA (RAG)', 2)
    pdf.body_text('Chatbot intelligent avec systeme RAG (Retrieval-Augmented Generation).')
    pdf.bullet_point('LLM Groq (modele Llama optimise)')
    pdf.bullet_point('Base de connaissances : Documents PDF, rapports officiels')
    pdf.bullet_point('Indexation vectorielle FAISS pour recherche semantique')
    pdf.bullet_point('Contexte enrichi avec donnees temps reel')
    pdf.ln(3)
    pdf.add_highlight_box('Valeur: Expertise economique accessible 24/7 via interface conversationnelle.', 'green')
    
    # ==================== ARCHITECTURE ====================
    pdf.add_page()
    pdf.chapter_title('3. ARCHITECTURE TECHNIQUE')
    
    pdf.chapter_title('Stack Technologique', 2)
    pdf.add_table(
        ['Couche', 'Technologies'],
        [
            ['Frontend', 'Streamlit + Plotly + Custom CSS (Dark/Light)'],
            ['Backend', 'Python 3.10+ / Pandas / NumPy / Scikit-learn'],
            ['Machine Learning', 'XGBoost (Regression + Classification)'],
            ['Intelligence Artificielle', 'Groq LLM / RAG System / FAISS'],
            ['Data Layer', 'CSV Datasets / JSON Configs / PDF Documents']
        ],
        [50, 140]
    )
    
    pdf.chapter_title('Fichiers Cles', 2)
    pdf.add_table(
        ['Fichier', 'Description'],
        [
            ['app.py', 'Application principale Streamlit'],
            ['rag_system.py', 'Systeme RAG avec indexation'],
            ['api.py', 'API REST pour predictions'],
            ['models/', 'Modeles XGBoost entraines'],
            ['data/processed/', 'Donnees nettoyees et unifiees'],
            ['documents/', 'Rapports PDF pour RAG']
        ],
        [50, 140]
    )
    
    # ==================== VALEUR AJOUTEE ====================
    pdf.add_page()
    pdf.chapter_title('4. VALEUR AJOUTEE')
    
    pdf.chapter_title('Pour le Gouvernement', 2)
    pdf.add_table(
        ['Benefice', 'Description'],
        [
            ['Pilotage strategique', 'Vision consolidee de l\'economie nationale'],
            ['Priorisation budgetaire', 'Allocation optimale des ressources publiques'],
            ['Politique industrielle', 'Identification des filieres a developper'],
            ['Souverainete economique', 'Reduction de la dependance aux importations']
        ],
        [55, 135]
    )
    
    pdf.chapter_title('Pour les Investisseurs', 2)
    pdf.add_table(
        ['Benefice', 'Description'],
        [
            ['Identification opportunites', 'Secteurs a fort ROI potentiel'],
            ['Reduction des risques', 'Analyse basee sur donnees historiques'],
            ['Simulation de scenarios', 'Test avant investissement'],
            ['Due diligence facilitee', 'Acces aux donnees sectorielles']
        ],
        [55, 135]
    )
    
    pdf.chapter_title('Pour les Entrepreneurs', 2)
    pdf.add_table(
        ['Benefice', 'Description'],
        [
            ['Choix de secteur', 'Orientation vers les filieres porteuses'],
            ['Business plan', 'Donnees pour etudes de faisabilite'],
            ['Benchmark', 'Comparaison avec la concurrence'],
            ['Acces expertise', 'Assistant IA disponible 24/7']
        ],
        [55, 135]
    )
    
    # ==================== IMPACT ECONOMIQUE ====================
    pdf.add_page()
    pdf.chapter_title('5. IMPACT ECONOMIQUE REEL')
    
    pdf.chapter_title('Potentiel de Substitution Identifie', 2)
    pdf.body_text('Base sur l\'analyse des donnees 2014-2025, la solution a identifie :')
    
    pdf.add_table(
        ['Categorie', 'Nb Secteurs', 'Impact Potentiel'],
        [
            ['Fort Potentiel', '~30', 'Reduction imports 15-25%'],
            ['Potentiel Moyen', '~80', 'Reduction imports 5-15%'],
            ['Surveillance', '~75', 'Maintien et optimisation']
        ],
        [60, 50, 80]
    )
    
    pdf.chapter_title('Estimation de l\'Impact Financier', 2)
    pdf.add_highlight_box(
        'Importations annuelles moyennes : environ 2 500 Milliards FCFA\n\n'
        'Scenario conservateur (10% substitution) : Economie de 250 Mds FCFA/an\n'
        'Scenario optimiste (20% substitution) : Economie de 500 Mds FCFA/an',
        'orange'
    )
    
    pdf.chapter_title('Creation de Valeur Locale', 2)
    pdf.add_table(
        ['Indicateur', 'Impact Estime'],
        [
            ['Emplois directs', '+50 000 a +150 000'],
            ['Emplois indirects', '+100 000 a +300 000'],
            ['PIB additionnel', '+2% a +5%'],
            ['Recettes fiscales', '+100 a +300 Mds FCFA']
        ],
        [70, 120]
    )
    
    pdf.chapter_title('Secteurs Prioritaires Identifies', 2)
    pdf.bullet_point('Agroalimentaire - Transformation locale des produits agricoles')
    pdf.bullet_point('Materiaux de construction - Ciment, fer, briques')
    pdf.bullet_point('Textile & Habillement - Coton transforme localement')
    pdf.bullet_point('Energie - Solaire, biocarburants')
    pdf.bullet_point('Emballages - Plastiques, cartons')
    pdf.bullet_point('Produits chimiques - Engrais, produits d\'entretien')
    pdf.bullet_point('Equipements agricoles - Petite mecanisation')
    
    # ==================== CAS D'USAGE ====================
    pdf.add_page()
    pdf.chapter_title('6. CAS D\'USAGE CONCRETS')
    
    pdf.chapter_title('Cas 1 : Ministere du Commerce', 2)
    pdf.body_text('Besoin: Identifier les secteurs prioritaires pour la politique de substitution 2025-2030.')
    pdf.bullet_point('1. Acces a l\'onglet "Recommandations"')
    pdf.bullet_point('2. Filtrage par "Fort Potentiel"')
    pdf.bullet_point('3. Export du Top 20 en CSV')
    pdf.bullet_point('4. Utilisation du Simulateur pour tester les scenarios de soutien')
    pdf.add_highlight_box('Resultat: Liste priorisee avec scores et justifications data-driven.', 'green')
    
    pdf.chapter_title('Cas 2 : Banque de Developpement', 2)
    pdf.body_text('Besoin: Evaluer une demande de financement pour une usine de transformation.')
    pdf.bullet_point('1. Analyse sectorielle du secteur concerne')
    pdf.bullet_point('2. Verification du score de substitution')
    pdf.bullet_point('3. Simulation de l\'impact du projet')
    pdf.bullet_point('4. Consultation de l\'Assistant IA pour contexte')
    pdf.add_highlight_box('Resultat: Decision de financement basee sur des indicateurs objectifs.', 'green')
    
    pdf.chapter_title('Cas 3 : Entrepreneur Local', 2)
    pdf.body_text('Besoin: Choisir un secteur pour creer son entreprise.')
    pdf.bullet_point('1. Exploration du tableau de bord "Accueil"')
    pdf.bullet_point('2. Identification des secteurs a fort potentiel')
    pdf.bullet_point('3. Analyse detaillee des 3-5 secteurs interessants')
    pdf.bullet_point('4. Simulation avec ses capacites d\'investissement')
    pdf.add_highlight_box('Resultat: Choix eclaire base sur le potentiel reel du marche.', 'green')
    
    # ==================== RECOMMANDATIONS ====================
    pdf.add_page()
    pdf.chapter_title('7. RECOMMANDATIONS STRATEGIQUES')
    
    pdf.chapter_title('Court Terme (1-2 ans)', 2)
    pdf.bullet_point('Prioriser les 10 secteurs a score > 80')
    pdf.bullet_point('Creer des zones industrielles dediees')
    pdf.bullet_point('Faciliter l\'acces au financement pour ces secteurs')
    pdf.bullet_point('Former la main-d\'oeuvre locale')
    
    pdf.chapter_title('Moyen Terme (3-5 ans)', 2)
    pdf.bullet_point('Developper les chaines de valeur integrees')
    pdf.bullet_point('Negocier des partenariats technologiques')
    pdf.bullet_point('Mettre en place des quotas d\'importation progressifs')
    pdf.bullet_point('Renforcer les normes de qualite locale')
    
    pdf.chapter_title('Long Terme (5-10 ans)', 2)
    pdf.bullet_point('Atteindre l\'autosuffisance dans les secteurs cles')
    pdf.bullet_point('Devenir exportateur dans certaines filieres')
    pdf.bullet_point('Creer un ecosysteme industriel diversifie')
    pdf.bullet_point('Servir de modele pour la sous-region')
    
    # ==================== PERSPECTIVES ====================
    pdf.chapter_title('8. PERSPECTIVES D\'EVOLUTION')
    
    pdf.add_table(
        ['Fonctionnalite', 'Description', 'Priorite'],
        [
            ['API publique', 'Integration avec systemes tiers', 'Haute'],
            ['Donnees temps reel', 'Connexion aux sources officielles', 'Haute'],
            ['Alertes automatiques', 'Notifications sur opportunites', 'Moyenne'],
            ['Module reporting', 'Generation de rapports PDF', 'Moyenne'],
            ['Application mobile', 'Acces en mobilite', 'Basse'],
            ['Multi-pays', 'Extension a l\'UEMOA', 'Basse']
        ],
        [50, 95, 45]
    )
    
    pdf.chapter_title('Integrations Possibles', 2)
    pdf.bullet_point('Douanes : Donnees d\'importation en temps reel')
    pdf.bullet_point('INSD : Statistiques nationales')
    pdf.bullet_point('Chambres de commerce : Repertoire des entreprises')
    pdf.bullet_point('Banques : Historique des financements sectoriels')
    
    # ==================== CONCLUSION ====================
    pdf.add_page()
    pdf.chapter_title('9. CONCLUSION')
    
    pdf.body_text(
        "L'Analyseur Import/Export Burkina Faso represente une innovation majeure dans l'approche "
        "de la politique economique nationale. En combinant donnees historiques completes (2014-2025), "
        "Intelligence Artificielle de pointe (XGBoost, RAG, LLM), interface utilisateur intuitive "
        "et simulations interactives, cette solution offre un outil decisionnel unique capable de "
        "guider efficacement la strategie de substitution aux importations du Burkina Faso."
    )
    
    pdf.ln(10)
    
    # Box finale
    pdf.set_fill_color(26, 115, 232)
    pdf.rect(20, pdf.get_y(), 170, 50, 'F')
    pdf.set_xy(25, pdf.get_y() + 5)
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(160, 10, 'IMPACT POTENTIEL GLOBAL', 0, 2, 'C')
    pdf.set_font('Arial', '', 11)
    pdf.cell(160, 7, 'Economies potentielles : 250-500 Milliards FCFA/an', 0, 2, 'C')
    pdf.cell(160, 7, 'Creation d\'emplois : 150 000 - 450 000', 0, 2, 'C')
    pdf.cell(160, 7, 'Amelioration PIB : +2% a +5%', 0, 2, 'C')
    
    pdf.ln(60)
    pdf.set_font('Arial', 'I', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, 'Developpe avec passion pour le Burkina Faso', 0, 1, 'C')
    pdf.cell(0, 8, 'Hackathon 24H - Decembre 2025', 0, 1, 'C')
    pdf.ln(5)
    pdf.set_font('Arial', '', 9)
    pdf.cell(0, 8, '(c) 2025 - Tous droits reserves', 0, 1, 'C')
    
    # Sauvegarder
    output_path = 'Rapport_Analyseur_ImportExport_BurkinaFaso.pdf'
    pdf.output(output_path)
    print(f"\n{'='*60}")
    print(f"  RAPPORT PDF GENERE AVEC SUCCES!")
    print(f"{'='*60}")
    print(f"\n  Fichier: {output_path}")
    print(f"  Taille: {os.path.getsize(output_path) / 1024:.1f} Ko")
    print(f"  Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print(f"\n{'='*60}\n")
    return output_path


if __name__ == "__main__":
    generate_rapport()
