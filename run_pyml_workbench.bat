@echo off
setlocal
pushd "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo PY-ML virtual environment was not found at .venv\Scripts\python.exe.
  echo Run the approved project setup command first; this launcher does not install packages.
  pause
  popd
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -X utf8 -m pyml_workbench
set "exit_code=%errorlevel%"
if not "%exit_code%"=="0" pause
popd
exit /b %exit_code%
