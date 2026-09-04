@echo off
chcp 65001 >nul
title 灰泽满 · 一键启动
echo ============================================
echo    灰泽满 QQ bot · 一键启动
echo ============================================
echo.

REM ① QQ 客户端（NapCat，机器人本体登录 QQ 的地方）。
REM    NapCatQQ-Desktop 若没记住登录，会弹窗让你扫码/选号，手动登录一次即可。
start "NapCatQQ" "D:\NapCatQQ\NapCatQQ-Desktop.exe"

echo [2/3] 启动 GPT-SoVITS 语音服务（端口 9880，模型 HZM-SPEAK e5+e15）...
REM ② GPT-SoVITS 合成服务。看到 "Uvicorn running on http://127.0.0.1:9880" 才算就绪。
start "GPT-SoVITS api" cmd /k "cd /d D:\Huizeman-AI-Voice-Cloning\GPT-SoVITS_V4\GPT-SoVITS_V4_250424 && env\python.exe api_v2.py"

timeout /t 3 >nul

echo [3/3] 启动灰泽满 bot...
REM ③ bot 本体（NoneBot）。若你平时用系统 python 跑，把 .venv\Scripts\python.exe 换成 python。
start "灰泽满 bot" cmd /k "cd /d D:\my_qq_bot\my_qq_bot && .venv\Scripts\python.exe bot.py"

echo.
echo 三个窗口都已打开。判断就绪：
echo   · NapCatQQ 已登录机器人号
echo   · GPT-SoVITS 窗口出现 Uvicorn running ... 9880
echo   · bot 窗口无报错、有连接日志
echo 窗口别关（关了对应程序就停了）。
pause
