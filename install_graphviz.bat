@echo off
setlocal

echo ===============================
echo   Installing Portable Graphviz
echo ===============================

set GV_DIR=%~dp0graphviz_portable
set GV_ZIP=%GV_DIR%\graphviz.zip

mkdir "%GV_DIR%" 2>nul

echo.
echo Downloading Graphviz portable ZIP...
curl -L "https://graphviz.gitlab.io/_pages/Download/windows/graphviz-2.38.zip" -o "%GV_ZIP%"

if not exist "%GV_ZIP%" (
    echo ERROR: Download failed.
    pause
    exit /b 1
)

echo.
echo Extracting ZIP...
powershell -Command "Expand-Archive -Path '%GV_ZIP%' -DestinationPath '%GV_DIR%' -Force"

echo.
echo Adding Graphviz to PATH for current session...
set PATH=%GV_DIR%\release\bin;%PATH%

echo.
echo ==============================================
echo   Portable Graphviz Installed Successfully!
echo ==============================================
echo Test with:
echo     dot -V
echo.

pause
endlocal