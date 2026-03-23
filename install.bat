@echo off
title CSAI412 - Phishing Detection - Setup
echo.
echo  ========================================
echo   CSAI412 - Phishing Website Detection
echo   Installing...
echo  ========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python is not installed!
    echo  Please download Python from https://python.org/downloads
    echo  IMPORTANT: Check "Add Python to PATH" during install
    pause
    exit /b 1
)

:: Install dependencies
echo  Installing Python packages...
pip install -r requirements.txt
pip install jupyter notebook

:: Create desktop shortcut
echo  Creating desktop shortcut...
set SCRIPT_DIR=%~dp0
set SHORTCUT=%USERPROFILE%\Desktop\Phishing Detection.lnk

powershell -Command "$ws = New-Object -ComObject WScript.Shell; $sc = $ws.CreateShortcut('%SHORTCUT%'); $sc.TargetPath = '%SCRIPT_DIR%run.bat'; $sc.WorkingDirectory = '%SCRIPT_DIR%'; $sc.Description = 'CSAI412 Phishing Website Detection'; $sc.Save()"

echo.
echo  ========================================
echo   Setup complete!
echo   A shortcut "Phishing Detection" has
echo   been created on your Desktop.
echo   Double-click it to run the app.
echo  ========================================
echo.
pause
