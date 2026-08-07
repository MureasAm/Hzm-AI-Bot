@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ---------- 配置（按需修改盘符）----------
set DRIVE=D:
set ENV_PATH=%DRIVE%\whisper_env
set HF_CACHE=%DRIVE%\huggingface_cache
set PYTHON_EXE=C:\Users\28916\AppData\Local\Programs\Python\Python312\python.exe
:: ----------------------------------------

:: 设置 Hugging Face 缓存目录，防止模型下载到 C 盘
set HF_HOME=%HF_CACHE%

:: 检查 Python 3.12 是否存在
if not exist "%PYTHON_EXE%" (
    echo [错误] 未找到 Python 3.12：%PYTHON_EXE%
    echo 请修改脚本中的 PYTHON_EXE 为正确路径。
    pause
    exit /b 1
)

:: 检查 whisper 虚拟环境
if not exist "%ENV_PATH%\Scripts\activate.bat" (
    echo [错误] 未找到 whisper 虚拟环境：%ENV_PATH%
    echo 请先创建 D:\whisper_env 并安装 faster-whisper。
    pause
    exit /b 1
)

:: 获取拖入的文件或文件夹；未拖入则让 run_tool 扫描默认位置 assets/audio/
set INPUT=%1
if "%INPUT%"=="" (
    echo 未指定输入，将扫描 assets/audio/ 目录。
    "%ENV_PATH%\Scripts\python.exe" "%~dp0run_tool.py" transcribe
) else (
    "%ENV_PATH%\Scripts\python.exe" "%~dp0run_tool.py" transcribe "%INPUT%"
)

echo.
echo 处理完成！
pause
