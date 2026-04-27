@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo Traffic Violation System - Setup
echo ==========================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH.
    echo Install Python 3.10 first, then run this file again.
    pause
    exit /b 1
)

echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
)

echo Installing project requirements...
call ".\venv\Scripts\python.exe" -m pip install --upgrade pip
call ".\venv\Scripts\python.exe" -m pip install -r requirements.txt
call ".\venv\Scripts\python.exe" -m pip install pyinstaller python-pptx protobuf==3.20.3
if errorlevel 1 (
    echo Failed while installing packages.
    pause
    exit /b 1
)

echo.
echo Setup completed successfully.
echo Use run_app.bat to start the application.
pause
exit /b 0
