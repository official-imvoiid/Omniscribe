@echo off
cd /d "%~dp0"
setlocal

set "ROOT=%CD%\installer_files"
set "CONDA_ROOT=%ROOT%\Miniconda"
set "CONDA=%CONDA_ROOT%\_conda.exe"
set "ENVS=%ROOT%\Environments"
set "ENV=%ENVS%\captioner"
set "PY=%ENV%\python.exe"
set "CONDARC=%CONDA_ROOT%\.condarc"
set "CUDA=cu130"
set "MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-py312_26.7.1-1-Windows-x86_64.exe"

set PYTHONPATH=
set PYTHONHOME=
set PYTHONSTARTUP=
set PYTHONUSERBASE=
set PIP_CONFIG_FILE=
set VIRTUAL_ENV=
set CONDA_PREFIX=
set CONDA_DEFAULT_ENV=

echo ASID Captioner - install
echo Folder: %CD%
echo.

if not exist "%ROOT%" mkdir "%ROOT%"
if not exist "%ENVS%" mkdir "%ENVS%"

REM ---- 1. Miniconda --------------------------------------------------------
if exist "%CONDA%" (
    echo [1/4] Miniconda already installed.
    goto conda_ok
)
echo [1/4] Downloading Miniconda...
curl -L --retry 20 --retry-delay 3 --retry-all-errors -C - "%MINICONDA_URL%" -o "%ROOT%\miniconda_installer.exe"
if not exist "%ROOT%\miniconda_installer.exe" goto error
echo       Installing...
start /wait "" "%ROOT%\miniconda_installer.exe" /InstallationType=JustMe /NoShortcuts=1 /AddToPath=0 /RegisterPython=0 /NoRegistry=1 /S /D=%CONDA_ROOT%
if not exist "%CONDA%" goto error
del "%ROOT%\miniconda_installer.exe" >nul 2>&1

:conda_ok

REM ---- 2. .condarc, rewritten every run so a moved folder is fixed ---------
echo [2/4] Writing .condarc for this folder.
echo envs_dirs: > "%CONDARC%"
echo   - %ENVS% >> "%CONDARC%"
echo pkgs_dirs: >> "%CONDARC%"
echo   - %CONDA_ROOT%\pkgs >> "%CONDARC%"
echo channels: >> "%CONDARC%"
echo   - conda-forge >> "%CONDARC%"

REM ---- 3. environment ------------------------------------------------------
if exist "%PY%" (
    echo [3/4] Environment already exists.
    goto env_ok
)
echo [3/4] Creating environment: Python 3.12 + ffmpeg
"%CONDA%" create -p "%ENV%" -c conda-forge python=3.12 ffmpeg -y
if errorlevel 1 goto error
if not exist "%PY%" goto error

:env_ok

set "PATH=%ENV%;%ENV%\Scripts;%ENV%\Library\bin;%PATH%"

REM ---- 4. packages ---------------------------------------------------------
echo [4/4] Installing packages.
echo.
"%PY%" -m pip install --upgrade pip --quiet

echo       torch %CUDA%
"%PY%" -m pip install --retries 30 --timeout 180 torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/%CUDA%
if errorlevel 1 goto error

echo.
echo       remaining packages
"%PY%" -m pip install --retries 30 --timeout 180 -r requirements.txt
if errorlevel 1 goto error

echo.
"%PY%" -c "import torch;print('torch',torch.__version__,'  cuda',torch.cuda.is_available())"
echo.
echo Install complete.
echo Next: model.bat to download a model, then start.bat
echo.
pause
exit /b 0

:error
echo.
echo A step failed. Run install.bat again to resume.
echo If nvidia-smi reports CUDA below 13.0, change CUDA=cu130 to cu126 at the top.
echo.
pause
exit /b 1
