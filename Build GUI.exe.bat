@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "UV_EXE="
where uv >nul 2>&1 && set "UV_EXE=uv"
if not defined UV_EXE if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
if not defined UV_EXE if exist "%USERPROFILE%\.cargo\bin\uv.exe" set "UV_EXE=%USERPROFILE%\.cargo\bin\uv.exe"
if not defined UV_EXE if exist "%LOCALAPPDATA%\Programs\uv\uv.exe" set "UV_EXE=%LOCALAPPDATA%\Programs\uv\uv.exe"

if not defined UV_EXE (
    echo uv was not found on PATH.
    echo Install uv from https://docs.astral.sh/uv/ then open a new terminal and try again.
    pause
    exit /b 1
)

echo Building single-file transcription GUI executable...
echo This can take several minutes and produce a multi-GB file.
"%UV_EXE%" run --extra gui python scripts\build_gui_exe.py %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo Build failed ^(code %EXIT_CODE%^).
    pause
    exit /b %EXIT_CODE%
)

echo.
echo Built dist\whisper_timestamped_gui.exe
pause
exit /b 0
