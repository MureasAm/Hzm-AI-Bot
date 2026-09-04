@echo off
chcp 65001 >nul
title 灰泽满 bot + TTS
echo ============================================
echo   灰泽满 bot + GPT-SoVITS TTS · 启动
echo ============================================
echo.

echo [1/2] 启动 GPT-SoVITS 语音服务（端口 9880，模型 HZM-SPEAK e5+e15）...
REM 看到 "Uvicorn running on http://127.0.0.1:9880" 才算就绪
start "GPT-SoVITS api" cmd /k "cd /d D:\Huizeman-AI-Voice-Cloning\GPT-SoVITS_V4\GPT-SoVITS_V4_250424 && env\python.exe api_v2.py"

timeout /t 2 >nul

echo [2/2] 启动灰泽满 bot...
REM 若你平时用项目 .venv 跑，把 python 换成 .venv\Scripts\python.exe
start "灰泽满 bot" cmd /k "cd /d D:\my_qq_bot\my_qq_bot && python bot.py"

echo.
echo 就绪判断：
echo   · GPT-SoVITS 窗口出现 Uvicorn running ... 9880
echo   · bot 窗口无报错、有连接日志
echo （NapCat / QQ 请自己先开好并登录，本脚本不再代开）
echo 窗口别关，关了对应程序就停了。
pause
