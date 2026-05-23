@echo off
echo Starting Global Financial News Sentiment Dashboard...
echo.
cd /d "%~dp0"
.venv\Scripts\streamlit.exe run app.py --server.port 8501 --server.headless false
pause
