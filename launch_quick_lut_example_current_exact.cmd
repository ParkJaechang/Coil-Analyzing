@echo off
setlocal
cd /d "%~dp0"

set "FIELD_ANALYSIS_STARTUP_PRESET=continuous_current_exact"
call "%~dp0launch_streamlit_with_free_port.cmd" "app_field_analysis_quick.py"
