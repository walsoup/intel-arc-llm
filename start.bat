@echo off
title Launch Local AI Studio
echo ================================================================
echo   Launching Local LLM Router Server & Web UI
echo   Engine: Prism Vulkan (Classic Web UI)
echo   URL: http://127.0.0.1:8080
echo ================================================================
echo.
start "Local LLM Server" cmd /k ".\start-server.bat"
echo Waiting for server to initialize...
timeout /t 2 /nobreak >nul
start http://127.0.0.1:8080
echo.
echo [OK] Server running and Web UI opened in your browser!
