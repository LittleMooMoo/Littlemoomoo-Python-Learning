@echo off
setlocal
cd /d "%~dp0"
set "PORTFOLIO_PYTHON=%~dp0..\.venv\Scripts\python.exe"
if exist "%PORTFOLIO_PYTHON%" goto run
set "PORTFOLIO_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%PORTFOLIO_PYTHON%" goto run
set "PORTFOLIO_PYTHON=python"
:run
"%PORTFOLIO_PYTHON%" "%~dp0tools\editor.py"
if errorlevel 1 (
  echo.
  echo The editor could not start. Confirm Python 3.10 or later is installed.
  pause
)
endlocal
