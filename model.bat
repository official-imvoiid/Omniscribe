@echo off
cd /d "%~dp0"
setlocal

set "ENV=%CD%\installer_files\Environments\captioner"
set "PY=%ENV%\python.exe"
set "REPO=AudioVisual-Caption/ASID-Captioner-7B"
set "DEST=%CD%\models"

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

echo Downloading %REPO%
echo Into         %DEST%
echo.
echo About 18 GB. Interrupted downloads resume - just run this again.
echo.

if not exist "%DEST%" mkdir "%DEST%"

REM The Scripts shim carries an absolute path from install time, so it breaks
REM if the folder was moved. Fall back to the module form when that happens.
"%ENV%\Scripts\hf.exe" download %REPO% --local-dir "%DEST%"
if errorlevel 1 (
    echo.
    echo   hf.exe did not run - going through Python instead.
    echo.
    "%PY%" -m huggingface_hub.commands.huggingface_cli download %REPO% --local-dir "%DEST%"
    if errorlevel 1 goto error
)

dir /b "%DEST%\*.safetensors" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Download finished but no .safetensors are in %DEST%
    echo They may have landed in a subfolder - the weights must sit flat.
    echo.
    pause
    exit /b 1
)

echo.
echo Model ready. Run start.bat
echo.
pause
exit /b 0

:error
echo.
echo Download failed. Run model.bat again to resume where it stopped.
echo.
pause
exit /b 1
