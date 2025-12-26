Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Démarrage de l'application Streamlit" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Activer l'environnement virtuel
& ".\venv_hackathon\Scripts\Activate.ps1"

# Démarrer Streamlit
Write-Host "Démarrage de l'application..." -ForegroundColor Green
Write-Host "URL: http://localhost:8501" -ForegroundColor Yellow
Write-Host ""

streamlit run app.py --server.port 8501

