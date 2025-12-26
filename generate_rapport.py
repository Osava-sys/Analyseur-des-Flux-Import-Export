"""
Générateur de Rapport PDF - Analyseur Import/Export Burkina Faso
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from datetime import datetime
import os

# Couleurs Burkina Faso
ROUGE_BF = colors.HexColor("#CE1126")
VERT_BF = colors.HexColor("#009E49")
JAUNE_BF = colors.HexColor("#FCD116")
NOIR = colors.HexColor("#1A1A1A")
GRIS = colors.HexColor("#666666")

def create_header_footer(canvas, doc):
    """Ajouter en-tête et pied de page"""
    canvas.saveState()
    
    # En-tête avec bande tricolore
    canvas.setFillColor(ROUGE_BF)
    canvas.rect(0, A4[1] - 15*mm, A4[0], 5*mm, fill=1, stroke=0)
    canvas.setFillColor(VERT_BF)
    canvas.rect(0, A4[1] - 20*mm, A4[0], 5*mm, fill=1, stroke=0)
    canvas.setFillColor(JAUNE_BF)
    canvas.rect(0, A4[1] - 25*mm, A4[0], 5*mm, fill=1, stroke=0)
    
    # Pied de page
    canvas.setFillColor(GRIS)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(2*cm, 1.5*cm, f"© 2025 Hackathon IA - Burkina Faso")
    canvas.drawRightString(A4[0] - 2*cm, 1.5*cm, f"Page {doc.page}")
    
    # Ligne de séparation
    canvas.setStrokeColor(VERT_BF)
    canvas.setLineWidth(1)
    canvas.line(2*cm, 2*cm, A4[0] - 2*cm, 2*cm)
    
    canvas.restoreState()

def generate_report():
    """Générer le rapport PDF"""
    
    filename = "Rapport_Analyseur_ImportExport_BurkinaFaso.pdf"
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        topMargin=3.5*cm,
        bottomMargin=2.5*cm,
        leftMargin=2*cm,
        rightMargin=2*cm
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Style titre principal
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=ROUGE_BF,
        alignment=TA_CENTER,
        spaceAfter=20,
        fontName='Helvetica-Bold'
    )
    
    # Style sous-titre
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=VERT_BF,
        alignment=TA_CENTER,
        spaceAfter=30,
        fontName='Helvetica'
    )
    
    # Style section
    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=ROUGE_BF,
        spaceBefore=20,
        spaceAfter=10,
        fontName='Helvetica-Bold',
        borderColor=VERT_BF,
        borderWidth=1,
        borderPadding=5
    )
    
    # Style sous-section
    subsection_style = ParagraphStyle(
        'Subsection',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=VERT_BF,
        spaceBefore=15,
        spaceAfter=8,
        fontName='Helvetica-Bold'
    )
    
    # Style texte normal
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        textColor=NOIR,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
        leading=14
    )
    
    # Style bullet
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontSize=10,
        textColor=NOIR,
        leftIndent=20,
        spaceAfter=5,
        bulletIndent=10
    )
    
    # Contenu
    story = []
    
    # === PAGE DE TITRE ===
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph("🇧🇫", ParagraphStyle('Flag', fontSize=60, alignment=TA_CENTER)))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("ANALYSEUR IMPORT/EXPORT", title_style))
    story.append(Paragraph("BURKINA FASO", title_style))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Plateforme d'Intelligence Artificielle pour l'Identification<br/>des Opportunités de Substitution aux Importations", subtitle_style))
    story.append(Spacer(1, 2*cm))
    
    # Info box
    info_data = [
        ["Projet", "Hackathon IA 24h"],
        ["Date", datetime.now().strftime("%d/%m/%Y")],
        ["Technologies", "Python, Streamlit, XGBoost, Groq LLM"],
        ["Données", "2014-2023 (10 ans)"],
    ]
    info_table = Table(info_data, colWidths=[4*cm, 8*cm])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), VERT_BF),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
        ('BACKGROUND', (1, 0), (1, -1), colors.HexColor("#F5F5F5")),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, VERT_BF),
    ]))
    story.append(info_table)
    
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph("<i>« La Patrie ou la Mort, Nous Vaincrons »</i>", 
                          ParagraphStyle('Motto', fontSize=12, alignment=TA_CENTER, textColor=ROUGE_BF, fontName='Helvetica-Oblique')))
    
    story.append(PageBreak())
    
    # === SOMMAIRE ===
    story.append(Paragraph("SOMMAIRE", section_style))
    story.append(Spacer(1, 0.5*cm))
    
    sommaire = [
        "1. Résumé Exécutif",
        "2. Fonctionnalités de l'Application",
        "3. Valeur Ajoutée",
        "4. Impact Économique Potentiel",
        "5. Bénéficiaires",
        "6. Innovations Techniques",
        "7. Conclusion et Recommandations"
    ]
    for item in sommaire:
        story.append(Paragraph(f"• {item}", bullet_style))
    
    story.append(PageBreak())
    
    # === 1. RÉSUMÉ EXÉCUTIF ===
    story.append(Paragraph("1. RÉSUMÉ EXÉCUTIF", section_style))
    story.append(Paragraph(
        """Cette plateforme innovante utilise l'Intelligence Artificielle pour analyser les flux commerciaux 
        du Burkina Faso et identifier les secteurs prioritaires pour la substitution aux importations. 
        Basée sur 10 ans de données (2014-2023), elle combine des modèles de Machine Learning (XGBoost) 
        avec un assistant conversationnel (Groq LLM) pour démocratiser l'accès à l'intelligence économique.""",
        body_style
    ))
    story.append(Spacer(1, 0.5*cm))
    
    # KPIs clés
    kpi_data = [
        ["Indicateur", "Valeur", "Signification"],
        ["Précision Régression (R²)", "83.2%", "Qualité des prédictions de score"],
        ["Précision Classification", "98.4%", "Fiabilité de la priorisation"],
        ["Secteurs analysés", "50+", "Couverture économique complète"],
        ["Documents intégrés", "10 PDFs", "Base documentaire officielle"],
    ]
    kpi_table = Table(kpi_data, colWidths=[5*cm, 3*cm, 7*cm])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ROUGE_BF),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, GRIS),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9F9F9")]),
    ]))
    story.append(kpi_table)
    
    story.append(PageBreak())
    
    # === 2. FONCTIONNALITÉS ===
    story.append(Paragraph("2. FONCTIONNALITÉS DE L'APPLICATION", section_style))
    
    story.append(Paragraph("2.1 Page d'Accueil", subsection_style))
    story.append(Paragraph("• KPIs en temps réel : Production, Imports, Exports, Balance commerciale", bullet_style))
    story.append(Paragraph("• Évolution temporelle (2014-2023) avec graphiques interactifs Plotly", bullet_style))
    story.append(Paragraph("• Répartition sectorielle avec visualisations dynamiques", bullet_style))
    
    story.append(Paragraph("2.2 Analyse Détaillée", subsection_style))
    story.append(Paragraph("• Comparaisons multi-sectorielles des flux commerciaux", bullet_style))
    story.append(Paragraph("• Tendances historiques sur 10 ans de données", bullet_style))
    story.append(Paragraph("• Identification automatique des secteurs déficitaires", bullet_style))
    
    story.append(Paragraph("2.3 Recommandations IA", subsection_style))
    story.append(Paragraph("• Score de substitution (0-100) calculé par Machine Learning XGBoost", bullet_style))
    story.append(Paragraph("• Classification automatique : Haute / Moyenne / Faible priorité", bullet_style))
    story.append(Paragraph("• Top 10 des opportunités de substitution aux importations", bullet_style))
    story.append(Paragraph("• Filtrage interactif par secteur et niveau de priorité", bullet_style))
    
    story.append(Paragraph("2.4 Simulateur de Prédiction", subsection_style))
    story.append(Paragraph("• Modèle de régression pour prédire le score de substitution", bullet_style))
    story.append(Paragraph("• Modèle de classification pour déterminer la priorité", bullet_style))
    story.append(Paragraph("• Interface interactive : paramètres → prédiction instantanée", bullet_style))
    
    story.append(Paragraph("2.5 Assistant IA Conversationnel", subsection_style))
    story.append(Paragraph("• Chatbot intelligent propulsé par Groq Llama 3.1", bullet_style))
    story.append(Paragraph("• Connaissance intégrée des données économiques réelles", bullet_style))
    story.append(Paragraph("• 10 documents PDF officiels intégrés (PNDES-II, SNI, rapports...)", bullet_style))
    story.append(Paragraph("• Réponses naturelles et contextualisées en français", bullet_style))
    
    story.append(PageBreak())
    
    # === 3. VALEUR AJOUTÉE ===
    story.append(Paragraph("3. VALEUR AJOUTÉE", section_style))
    
    comparison_data = [
        ["Aspect", "Avant", "Avec l'Application"],
        ["Analyse des données", "Manuelle, longue, coûteuse", "Automatisée, instantanée, gratuite"],
        ["Identification opportunités", "Intuitive, subjective", "Data-driven, score objectif 0-100"],
        ["Priorisation", "Pas de critères clairs", "Classification ML automatique"],
        ["Accessibilité", "Experts économistes uniquement", "Interface intuitive pour tous"],
        ["Documentation", "Dispersée, difficile d'accès", "Centralisée + Assistant IA"],
        ["Temps de décision", "Semaines / Mois", "Minutes"],
    ]
    comparison_table = Table(comparison_data, colWidths=[4*cm, 5.5*cm, 5.5*cm])
    comparison_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), VERT_BF),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, 1), (0, -1), colors.HexColor("#E8F5E9")),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, VERT_BF),
    ]))
    story.append(comparison_table)
    
    story.append(PageBreak())
    
    # === 4. IMPACT ÉCONOMIQUE ===
    story.append(Paragraph("4. IMPACT ÉCONOMIQUE POTENTIEL", section_style))
    
    story.append(Paragraph("4.1 Réduction du Déficit Commercial", subsection_style))
    story.append(Paragraph(
        """Le Burkina Faso présente un déficit commercial structurel de plusieurs centaines de milliards 
        de FCFA sur la période analysée. Cette plateforme permet d'identifier précisément les secteurs 
        où la substitution aux importations est la plus réaliste et impactante.""",
        body_style
    ))
    
    story.append(Paragraph("4.2 Estimation des Économies Potentielles", subsection_style))
    
    impact_data = [
        ["Scénario", "Taux de substitution", "Économie estimée/an"],
        ["Conservateur", "5% des imports prioritaires", "25-50 Mds FCFA"],
        ["Modéré", "10% des imports prioritaires", "50-100 Mds FCFA"],
        ["Ambitieux", "20% des imports prioritaires", "100-200 Mds FCFA"],
    ]
    impact_table = Table(impact_data, colWidths=[4*cm, 5.5*cm, 5.5*cm])
    impact_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), JAUNE_BF),
        ('TEXTCOLOR', (0, 0), (-1, 0), NOIR),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, NOIR),
    ]))
    story.append(impact_table)
    
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("4.3 Effets Induits", subsection_style))
    story.append(Paragraph("• <b>Création d'emplois</b> : Développement des industries locales", bullet_style))
    story.append(Paragraph("• <b>Transfert de compétences</b> : Formation de la main-d'œuvre locale", bullet_style))
    story.append(Paragraph("• <b>Souveraineté économique</b> : Réduction de la dépendance extérieure", bullet_style))
    story.append(Paragraph("• <b>Balance des paiements</b> : Préservation des réserves de change", bullet_style))
    
    story.append(Paragraph("4.4 Alignement Stratégique", subsection_style))
    story.append(Paragraph("• <b>PNDES-II (2021-2025)</b> : Soutient les objectifs de développement national", bullet_style))
    story.append(Paragraph("• <b>SNI (2019-2023)</b> : Conforme à la Stratégie Nationale d'Industrialisation", bullet_style))
    story.append(Paragraph("• <b>Agenda AES</b> : Renforce l'intégration économique régionale", bullet_style))
    
    story.append(PageBreak())
    
    # === 5. BÉNÉFICIAIRES ===
    story.append(Paragraph("5. BÉNÉFICIAIRES", section_style))
    
    beneficiaires_data = [
        ["Acteur", "Bénéfice Principal"],
        ["Ministère de l'Économie", "Outil d'aide à la décision basé sur l'IA et les données"],
        ["Ministère de l'Industrie", "Identification des filières industrielles prioritaires"],
        ["Investisseurs nationaux", "Connaissance des secteurs porteurs et rentables"],
        ["Investisseurs étrangers", "Données fiables pour orienter les IDE"],
        ["Industriels locaux", "Opportunités de marché clairement identifiées"],
        ["Banques et institutions financières", "Justification chiffrée pour financer les projets"],
        ["Bailleurs de fonds", "Base de données pour cibler l'aide au développement"],
        ["Chercheurs et universitaires", "Données consolidées 2014-2023 pour la recherche"],
        ["Société civile", "Transparence sur les flux commerciaux nationaux"],
    ]
    beneficiaires_table = Table(beneficiaires_data, colWidths=[5*cm, 10*cm])
    beneficiaires_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ROUGE_BF),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, GRIS),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#FFF8E1")]),
    ]))
    story.append(beneficiaires_table)
    
    story.append(PageBreak())
    
    # === 6. INNOVATIONS TECHNIQUES ===
    story.append(Paragraph("6. INNOVATIONS TECHNIQUES", section_style))
    
    story.append(Paragraph("6.1 Stack Technologique", subsection_style))
    
    tech_data = [
        ["Composant", "Technologie", "Rôle"],
        ["Backend", "Python 3.14", "Traitement des données et ML"],
        ["Frontend", "Streamlit", "Interface utilisateur interactive"],
        ["ML Régression", "XGBoost", "Prédiction du score de substitution"],
        ["ML Classification", "XGBoost", "Catégorisation des priorités"],
        ["LLM", "Groq Llama 3.1", "Assistant conversationnel IA"],
        ["Visualisation", "Plotly", "Graphiques interactifs"],
        ["PDF Processing", "PyPDF2", "Extraction des documents"],
        ["API", "FastAPI", "Services REST"],
    ]
    tech_table = Table(tech_data, colWidths=[3.5*cm, 4*cm, 7.5*cm])
    tech_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), VERT_BF),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, GRIS),
    ]))
    story.append(tech_table)
    
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("6.2 Points Forts Techniques", subsection_style))
    story.append(Paragraph("• <b>Machine Learning appliqué</b> à l'économie burkinabè (première du genre)", bullet_style))
    story.append(Paragraph("• <b>LLM intégré</b> pour démocratiser l'accès aux données complexes", bullet_style))
    story.append(Paragraph("• <b>Extraction PDF automatique</b> des documents officiels gouvernementaux", bullet_style))
    story.append(Paragraph("• <b>Interface responsive</b> accessible sur desktop et mobile", bullet_style))
    story.append(Paragraph("• <b>Design patriotique</b> aux couleurs nationales 🟥🟩🟨", bullet_style))
    story.append(Paragraph("• <b>Open source</b> et facilement déployable", bullet_style))
    
    story.append(PageBreak())
    
    # === 7. CONCLUSION ===
    story.append(Paragraph("7. CONCLUSION ET RECOMMANDATIONS", section_style))
    
    story.append(Paragraph(
        """Cette solution représente une avancée significative dans l'utilisation de l'Intelligence 
        Artificielle au service du développement économique du Burkina Faso. En transformant des données 
        brutes en intelligence actionnable, elle permet de passer d'une approche réactive à une approche 
        proactive dans la gestion des flux commerciaux.""",
        body_style
    ))
    
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Recommandations pour la mise en œuvre :", subsection_style))
    story.append(Paragraph("1. <b>Déploiement institutionnel</b> : Intégrer la plateforme au Ministère de l'Économie", bullet_style))
    story.append(Paragraph("2. <b>Mise à jour des données</b> : Automatiser l'import des nouvelles statistiques", bullet_style))
    story.append(Paragraph("3. <b>Formation</b> : Capaciter les agents à l'utilisation de l'outil", bullet_style))
    story.append(Paragraph("4. <b>Extension régionale</b> : Adapter pour les pays de l'AES (Mali, Niger)", bullet_style))
    story.append(Paragraph("5. <b>Partenariats</b> : Collaborer avec les universités pour la recherche", bullet_style))
    
    story.append(Spacer(1, 1*cm))
    
    # Encadré final
    conclusion_data = [[
        Paragraph(
            """<b>« Cette solution transforme des données brutes en intelligence actionnable 
            pour la souveraineté économique du Burkina Faso. »</b>""",
            ParagraphStyle('Conclusion', fontSize=11, alignment=TA_CENTER, textColor=colors.white)
        )
    ]]
    conclusion_table = Table(conclusion_data, colWidths=[15*cm])
    conclusion_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), ROUGE_BF),
        ('PADDING', (0, 0), (-1, -1), 20),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(conclusion_table)
    
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(
        "<b>🇧🇫 La Patrie ou la Mort, Nous Vaincrons 🇧🇫</b>",
        ParagraphStyle('FinalMotto', fontSize=14, alignment=TA_CENTER, textColor=VERT_BF)
    ))
    
    # Générer le PDF
    doc.build(story, onFirstPage=create_header_footer, onLaterPages=create_header_footer)
    
    print(f"\n✅ Rapport généré avec succès : {filename}")
    print(f"📁 Emplacement : {os.path.abspath(filename)}")
    return filename

if __name__ == "__main__":
    generate_report()
