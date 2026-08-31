@echo off
setlocal enabledelayedexpansion

rem Get the current directory
set "current_dir=%cd%"

rem Create or overwrite the .condarc file with the actual paths
echo envs_dirs: > "%current_dir%\installer_files\Miniconda\.condarc"
echo   - %current_dir%\installer_files\Environments >> "%current_dir%\installer_files\Miniconda\.condarc"
echo pkgs_dirs: >> "%current_dir%\installer_files\Miniconda\.condarc"
echo   - %current_dir%\installer_files\Miniconda\pkgs >> "%current_dir%\installer_files\Miniconda\.condarc"

echo .condarc file has been updated with the following paths:
echo Environment directory: %current_dir%\installer_files\Environments
echo Packages directory: %current_dir%\installer_files\Miniconda\pkgs

pause