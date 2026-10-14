@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
py -3.13 -m venv .venv
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -m pip install -r requirements-shap.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -c "import joblib; b=joblib.load('artifacts/studentlife_v2/model.joblib'); print('Included StudentLife model loaded successfully.')"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" verify_privacy.py
if errorlevel 1 goto failed
echo Setup complete. Run start_windows.cmd to open CampusPulse.
exit /b 0
:failed
echo Setup stopped. Read the error above before starting the app.
exit /b 1
