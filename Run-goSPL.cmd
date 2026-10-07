@echo off
setlocal
powershell.exe -NoLogo -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0Run-goSPL.ps1" %*
set "GOSPL_LAUNCH_EXIT=%errorlevel%"
if not "%GOSPL_LAUNCH_EXIT%"=="0" echo goSPL launcher exited with code %GOSPL_LAUNCH_EXIT%.
if not "%GOSPL_LAUNCH_EXIT%"=="0" if "%~1"=="" pause
exit /b %GOSPL_LAUNCH_EXIT%
