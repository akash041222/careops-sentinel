@echo off
cd /d "%~dp0"
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
pip install -q -r requirements.txt
if exist frontend\.needs-ui-build (
  where npm >nul 2>nul
  if errorlevel 1 (
    echo Node.js and npm are required to build the updated alarm UI.
    echo Install Node.js, then run this file again.
    pause
    exit /b 1
  )
  if not exist frontend\node_modules call npm --prefix frontend install
  if errorlevel 1 exit /b 1
  call npm --prefix frontend run build
  if errorlevel 1 exit /b 1
  del frontend\.needs-ui-build
)
if not exist frontend\dist\index.html (cd frontend && call npm install && call npm run build && cd ..)
echo Open http://localhost:8000   (login: EMP-1001 / CareOps@123)
uvicorn app.main:app --host 0.0.0.0 --port 8000
