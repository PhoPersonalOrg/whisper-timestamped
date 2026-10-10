@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo uv was not found on PATH.
    echo Install uv, then open a new terminal and try again.
    pause
    exit /b 1
)

uv run python scripts\gui
if errorlevel 1 (
    echo.
    echo GUI exited with an error.
    pause
)
