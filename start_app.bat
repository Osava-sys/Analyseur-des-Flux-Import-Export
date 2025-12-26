@echo off
echo ============================================
echo  Demarrage de l'application Streamlit
echo ============================================
echo.

REM Activer l'environnement virtuel
call venv_hackathon\Scripts\activate.bat

REM Demarrer Streamlit
echo Demarrage de l'application...
echo URL: http://localhost:8501
echo.
streamlit run app.py --server.port 8501

pause

