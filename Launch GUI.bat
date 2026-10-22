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

echo Launching transcription GUI...
"%UV_EXE%" run --extra gui whisper_timestamped_gui %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo GUI exited with an error ^(code %EXIT_CODE%^).
    pause
)
exit /b %EXIT_CODE%
