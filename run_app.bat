@echo off
setlocal
cd /d "%~dp0"

if not exist ".\venv\Scripts\python.exe" (
    echo Virtual environment not found.
    echo Run setup_other_pc.bat first.
    pause
    exit /b 1
)

call ".\venv\Scripts\python.exe" main.py
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    pause
    exit /b 1
)

exit /b 0
