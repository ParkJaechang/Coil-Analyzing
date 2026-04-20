@echo off
setlocal
cd /d "%~dp0"

call "%~dp0launch_streamlit_with_free_port.cmd" "app_field_analysis_quick.py"
