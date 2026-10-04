@echo off
title Restart Local LLM Router Server
echo ================================================================
echo   Restarting Local LLM Router Server...
echo   Scanning .\Models for any new GGUF models
echo ================================================================
echo.

:: Terminate any existing llama-server instances
taskkill /F /IM llama-server.exe >nul 2>&1
timeout /t 1 /nobreak >nul

:: Launch the server
start "Local LLM Server" cmd /k ".\start-server.bat"
timeout /t 2 /nobreak >nul

echo [OK] Server reloaded! Any newly added models are now live in the Web UI.
echo Reopening Web UI in browser...
start http://127.0.0.1:8080
