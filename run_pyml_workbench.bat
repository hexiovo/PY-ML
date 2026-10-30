@echo off
setlocal
pushd "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo PY-ML virtual environment was not found at .venv\Scripts\python.exe.
  echo Install the dependencies as described in README.md, then run this launcher again.
  pause
  popd
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -X utf8 -m pyml_workbench
set "exit_code=%errorlevel%"
if not "%exit_code%"=="0" pause
popd
exit /b %exit_code%
