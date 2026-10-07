@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Setup.ps1" %*
set "setup_result=%errorlevel%"
if not "%setup_result%"=="0" pause
exit /b %setup_result%
