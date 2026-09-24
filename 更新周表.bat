@echo off
cd /d "%~dp0"
setlocal
title Hazel schedule updater

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo ============================================
echo   灰泽满周表更新
echo   识图后只覆盖 persona\world\schedule.json 的 weekly 字段
echo ============================================
echo.

REM 拖进来的文件一定带扩展名；没扩展名就是双击/只传了参数 -> 收图夹模式
if "%~x1"=="" goto INBOX

set "EXT=%~x1"
if /i "%EXT%"==".png"  goto RUN
if /i "%EXT%"==".jpg"  goto RUN
if /i "%EXT%"==".jpeg" goto RUN
if /i "%EXT%"==".webp" goto RUN
if /i "%EXT%"==".bmp"  goto RUN

echo [提示] 拖进来的看着不是图片，已停手:
echo        %~1
goto END

:INBOX
echo [模式] 收图夹 - 处理 data\schedule_inbox\ 里最新的一张图
echo        想把某一张图直接跑掉，把那张图拖到本文件上就行
echo.

:RUN
echo --------------------------------------------
"%PY%" "%~dp0scripts\update_schedule.py" %*
if errorlevel 1 goto FAIL

echo --------------------------------------------
echo [完成] 周表已更新。
echo        记得【重启 bot】才生效 -- 周表有进程内缓存。
goto END

:FAIL
echo --------------------------------------------
echo [失败] 上面有报错，schedule.json 没有被改动。

:END
echo.
pause
