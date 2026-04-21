@echo off
setlocal
cd /d "%~dp0"

set "APP_SCRIPT=%~1"
set "REPO_STREAMLIT_EXE=%~dp0.venv\Scripts\streamlit.exe"
set "STREAMLIT_EXE="
set "PYTHON_EXE="
set "LAUNCH_MODE="

if "%APP_SCRIPT%"=="" (
  echo Missing app script argument.
  exit /b 1
)

if exist "%REPO_STREAMLIT_EXE%" (
  set "STREAMLIT_EXE=%REPO_STREAMLIT_EXE%"
  set "LAUNCH_MODE=repo .venv streamlit"
) else (
  for /f "delims=" %%I in ('where streamlit 2^>nul') do (
    if not defined STREAMLIT_EXE (
      set "STREAMLIT_EXE=%%I"
      set "LAUNCH_MODE=PATH streamlit"
    )
  )
)

if not defined STREAMLIT_EXE (
  for /f "delims=" %%I in ('where python 2^>nul') do (
    if not defined PYTHON_EXE (
      set "PYTHON_EXE=%%I"
    )
  )
  if defined PYTHON_EXE (
    "%PYTHON_EXE%" -m streamlit --version >nul 2>nul
    if not errorlevel 1 set "LAUNCH_MODE=python -m streamlit"
  )
)

if not defined STREAMLIT_EXE if not defined LAUNCH_MODE (
  echo Streamlit executable not found for local launcher.
  echo Tried:
  echo   1. "%REPO_STREAMLIT_EXE%"
  echo   2. streamlit on PATH
  echo   3. python -m streamlit
  echo.
  echo Suggested setup from repo root:
  echo   python -m venv .venv
  echo   .venv\Scripts\python -m pip install -r requirements.txt
  pause
  exit /b 1
)

for /f %%I in ('powershell -NoProfile -Command "$ports = (8501..8520) + (8531..8545); $used = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty LocalPort -Unique); $free = $ports | Where-Object { $used -notcontains $_ } | Select-Object -First 1; if (-not $free) { $free = 8610 }; Write-Output $free"') do set "APP_PORT=%%I"

if not defined APP_PORT set "APP_PORT=8610"
set "APP_URL=http://127.0.0.1:%APP_PORT%"

if defined STREAMLIT_EXE (
  start "" "%STREAMLIT_EXE%" run "%~dp0%APP_SCRIPT%" --server.port %APP_PORT%
) else (
  start "" "%PYTHON_EXE%" -m streamlit run "%~dp0%APP_SCRIPT%" --server.port %APP_PORT%
)

for /l %%I in (1,1,20) do (
  powershell -NoProfile -Command "try { $resp = Invoke-WebRequest -Uri '%APP_URL%/_stcore/health' -UseBasicParsing -TimeoutSec 2; if ($resp.StatusCode -eq 200) { exit 0 } } catch { exit 1 }"
  if not errorlevel 1 goto ready
  timeout /t 1 /nobreak >nul
)

:ready
echo Launched %APP_SCRIPT% at %APP_URL% using %LAUNCH_MODE%
if /I not "%FIELD_ANALYSIS_OPEN_BROWSER%"=="0" start "" "%APP_URL%"
exit /b 0
