@echo off
call "%~dp0bootstrap.cmd" convert
exit /b %ERRORLEVEL%
