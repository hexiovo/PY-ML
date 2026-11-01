@echo off
setlocal EnableExtensions
if defined PYML_PYTHON goto custom
where py >nul 2>nul
if not errorlevel 1 goto pylauncher
where python >nul 2>nul
if not errorlevel 1 goto pythonpath
echo Python is required. Install Python or set PYML_PYTHON.
pause
exit /b 1
:custom
"%PYML_PYTHON%" -X utf8 "%~dp0launch.py" %*
goto finished
:pylauncher
py -3 -X utf8 "%~dp0launch.py" %*
goto finished
:pythonpath
python -X utf8 "%~dp0launch.py" %*
:finished
set "PYML_TEMPLATE_EXIT=%ERRORLEVEL%"
if not "%PYML_TEMPLATE_EXIT%"=="0" pause
exit /b %PYML_TEMPLATE_EXIT%
