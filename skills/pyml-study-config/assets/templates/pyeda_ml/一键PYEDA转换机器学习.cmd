@echo off
setlocal
set "PYML_PYTHON=%PYML_PYTHON%"
if not defined PYML_PYTHON set "PYML_PYTHON=python"
"%PYML_PYTHON%" -X utf8 "%~dp0pyeda_ml_pipeline.py" %*
exit /b %errorlevel%
