@echo off
setlocal
cd /d "%~dp0"

set "APP_SCRIPT=%~1"
set "STREAMLIT_EXE=%~dp0..\.venv\Scripts\streamlit.exe"

if "%APP_SCRIPT%"=="" (
  echo Missing app script argument.
  exit /b 1
)

if not exist "%STREAMLIT_EXE%" (
  echo Streamlit executable not found: "%STREAMLIT_EXE%"
  pause
  exit /b 1
)

for /f %%I in ('powershell -NoProfile -Command "$ports = (8501..8520) + (8531..8545); $used = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty LocalPort -Unique); $free = $ports | Where-Object { $used -notcontains $_ } | Select-Object -First 1; if (-not $free) { $free = 8610 }; Write-Output $free"') do set "APP_PORT=%%I"

if not defined APP_PORT set "APP_PORT=8610"
set "APP_URL=http://127.0.0.1:%APP_PORT%"

start "" "%STREAMLIT_EXE%" run "%~dp0%APP_SCRIPT%" --server.port %APP_PORT%

for /l %%I in (1,1,20) do (
  powershell -NoProfile -Command "try { $resp = Invoke-WebRequest -Uri '%APP_URL%/_stcore/health' -UseBasicParsing -TimeoutSec 2; if ($resp.StatusCode -eq 200) { exit 0 } } catch { exit 1 }"
  if not errorlevel 1 goto ready
  timeout /t 1 /nobreak >nul
)

:ready
echo Launched %APP_SCRIPT% at %APP_URL%
if /I not "%FIELD_ANALYSIS_OPEN_BROWSER%"=="0" start "" "%APP_URL%"
exit /b 0
