@echo off
cd /d "%~dp0"
setlocal

set "ENV=%CD%\installer_files\Environments\captioner"
set "PY=%ENV%\python.exe"

set PYTHONPATH=
set PYTHONHOME=
set PYTHONSTARTUP=
set PYTHONUSERBASE=
set PIP_CONFIG_FILE=
set CONDA_PREFIX=
set CONDA_DEFAULT_ENV=
set VIRTUAL_ENV=

if not exist "%PY%" (
    echo Environment not found. Run install.bat first.
    echo.
    pause
    exit /b 1
)

set "PATH=%ENV%;%ENV%\Scripts;%ENV%\Library\bin;%PATH%"

dir /b "models\*.safetensors" >nul 2>&1
if errorlevel 1 (
    echo No model found in models\
    echo Put the ASID-Captioner-7B files there, then run this again.
    echo.
    pause
    exit /b 1
)

"%PY%" app.py

echo.
pause
