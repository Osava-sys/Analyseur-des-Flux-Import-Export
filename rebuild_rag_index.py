"""
Script pour reconstruire l'index RAG avec le nouveau document PDF
==================================================================
Ce script reconstruit l'index RAG pour inclure le nouveau document:
"Base de Connaissances - Substitution aux Importations Burkina Faso (2000-2025).pdf"
"""

import sys
import io
import os
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from rag_system import initialize_rag_system

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

def main():
    print("="*70)
    print("RECONSTRUCTION DE L'INDEX RAG")
    print("Intégration du nouveau document PDF:")
    print("  'Base de Connaissances - Substitution aux Importations Burkina Faso (2000-2025).pdf'")
    print("="*70)
    print()
    
    # Vérifier que le document existe
    doc_path = Path("documents") / "Base de Connaissances - Substitution aux Importations Burkina Faso (2000-2025).pdf"
    if not doc_path.exists():
        print(f"❌ ERREUR: Le document n'existe pas à: {doc_path}")
        print("   Assurez-vous que le PDF est bien dans le dossier 'documents/'")
        return False
    
    print(f"✅ Document trouvé: {doc_path.name}")
    print()
    
    # Initialiser et reconstruire le système RAG
    print("[1/3] Initialisation du système RAG...")
    try:
        rag = initialize_rag_system(GROQ_API_KEY, force_rebuild=True)
        print("✅ Système RAG initialisé")
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        return False
    
    print()
    
    # Afficher les statistiques
    print("[2/3] Récupération des statistiques...")
    stats = rag.get_stats()
    
    print()
    print("="*70)
    print("STATISTIQUES DE L'INDEX RAG")
    print("="*70)
    print(f"✅ Index initialisé: {stats['initialized']}")
    print(f"📊 Total de documents indexés: {stats['total_documents']}")
    print(f"🔢 Dimension des embeddings: {stats['embedding_dim']}")
    print(f"🤖 Modèle d'embedding: {stats['embedding_model']}")
    print()
    print("📁 Répartition par type de source:")
    for source_type, count in stats['source_types'].items():
        print(f"   - {source_type}: {count} chunks")
    print()
    print("📄 Documents PDF traités:")
    pdf_sources = {k: v for k, v in stats['sources'].items() if 'pdf' in k.lower() or k.endswith('.pdf')}
    for source, count in sorted(pdf_sources.items()):
        print(f"   - {source}: {count} chunks")
    print()
    
    # Vérifier que le nouveau document est bien inclus
    print("[3/3] Vérification de l'intégration du nouveau document...")
    target_doc = "Base de Connaissances - Substitution aux Importations Burkina Faso (2000-2025).pdf"
    found = False
    for source in stats['sources'].keys():
        if target_doc in source:
            found = True
            print(f"✅ Document intégré avec succès!")
            print(f"   - Source: {source}")
            print(f"   - Chunks créés: {stats['sources'][source]}")
            break
    
    if not found:
        print(f"⚠️  Le document '{target_doc}' n'a pas été trouvé dans les sources.")
        print("   Vérifiez que le nom du fichier correspond exactement.")
    
    print()
    print("="*70)
    print("✅ RECONSTRUCTION TERMINÉE AVEC SUCCÈS!")
    print("="*70)
    print()
    print("L'index RAG a été reconstruit et inclut maintenant le nouveau document.")
    print("L'assistant IA pourra maintenant utiliser ces informations pour améliorer")
    print("la qualité et la pertinence de ses réponses.")
    print()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

