@echo off
title Hazel bot + GPT-SoVITS TTS
echo ============================================
echo   Hazel bot + GPT-SoVITS TTS  launcher
echo ============================================
echo.
echo [1/2] starting GPT-SoVITS api  (port 9880, HZM-SPEAK e5+e15)...
start "GPT-SoVITS api" cmd /k "cd /d D:\Huizeman-AI-Voice-Cloning\GPT-SoVITS_V4\GPT-SoVITS_V4_250424 && env\python.exe api_v2.py"
timeout /t 2 >nul
echo [2/2] starting Hazel bot...
REM if you run bot via project venv, change python -> .venv\Scripts\python.exe
start "Hazel bot" cmd /k "cd /d D:\my_qq_bot\my_qq_bot && python bot.py"
echo.
echo Ready when:  GPT-SoVITS shows 'Uvicorn running ... 9880',  bot window no error.
echo (Open NapCat / QQ yourself first)
echo Keep these windows open.
pause
