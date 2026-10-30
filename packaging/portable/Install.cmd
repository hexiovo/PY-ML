@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
set "install_result=%errorlevel%"
pause
exit /b %install_result%
