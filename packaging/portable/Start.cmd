@echo off
setlocal
if not exist "%~dp0PYML-Workbench.exe" (
  echo PYML-Workbench.exe is missing. Extract the complete ZIP before starting.
  pause
  exit /b 1
)
start "" /D "%~dp0" "%~dp0PYML-Workbench.exe"
