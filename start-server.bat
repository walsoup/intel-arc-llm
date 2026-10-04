@echo off
title Local LLM Router Server (Multi-Model • Vulkan)
echo ================================================================
echo   Local LLM Server (Router Mode - Multi-Model)
echo   Engine: Prism Vulkan (Intel Arc iGPU)
echo   Models Directory: .\Models
echo   Web UI: http://127.0.0.1:8080
echo   API Endpoint: http://127.0.0.1:8080/v1
echo ================================================================
echo.
.\bin-prism-vulkan\llama-server.exe --models-dir .\Models --models-max 1 -ngl 99 -t 6 -c 65536 -np 1 -fa on --host 127.0.0.1 --port 8080
pause
