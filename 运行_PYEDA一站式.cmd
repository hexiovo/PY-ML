@echo off
setlocal
if not defined PYML_PYTHON if exist "%~dp0.venv\Scripts\python.exe" set "PYML_PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined PYML_PYTHON set "PYML_PYTHON=python"
"%PYML_PYTHON%" -X utf8 "%~dp0skills\pyml-study-config\assets\templates\pyeda_ml\pyeda_ml_pipeline.py" %*
exit /b %errorlevel%
