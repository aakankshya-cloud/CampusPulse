@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto run
echo Run setup_windows.cmd first.
exit /b 1
:run
echo Open http://127.0.0.1:8000 after the server starts. Press Ctrl+C to stop.
".venv\Scripts\python.exe" server.py --model-dir artifacts/studentlife_v2
exit /b %errorlevel%
