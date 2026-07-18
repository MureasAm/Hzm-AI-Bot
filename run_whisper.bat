@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ---------- 配置（按需修改盘符）----------
set DRIVE=D:
set ENV_PATH=%DRIVE%\whisper_env
set HF_CACHE=%DRIVE%\huggingface_cache
set PYTHON_EXE=C:\Users\28916\AppData\Local\Programs\Python\Python312\python.exe
set SCRIPT_NAME=transcribe_whisper.py
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

:: 检查转写脚本（使用批处理文件所在目录）
set SCRIPT_PATH=%~dp0%SCRIPT_NAME%
if not exist "%SCRIPT_PATH%" (
    echo [错误] 未找到转写脚本：%SCRIPT_PATH%
    echo 请确保 transcribe_whisper.py 与本批处理文件放在同一文件夹内。
    pause
    exit /b 1
)

:: 获取拖入的文件或文件夹
set INPUT=%1

:: 如果没有拖入任何内容，默认使用当前目录下的 audio 文件夹
if "%INPUT%"=="" (
    if exist "audio" (
        set INPUT=audio
        echo 未指定输入，使用默认 audio 文件夹。
    ) else (
        echo [提示] 请将音频文件或文件夹拖放到此脚本上。
        echo 也可以在当前目录创建 audio 文件夹并放入 mp3 文件。
        pause
        exit /b 0
    )
)

:: 检查目标盘符是否存在
if not exist "%DRIVE%\" (
    echo [错误] 目标盘符 %DRIVE% 不存在，请修改脚本中的 DRIVE 变量。
    pause
    exit /b 1
)

:: 检查虚拟环境，不存在则创建
if not exist "%ENV_PATH%\Scripts\activate.bat" (
    echo 虚拟环境不存在，正在 %DRIVE% 盘创建...
    "%PYTHON_EXE%" -m venv "%ENV_PATH%"
    if %errorlevel% neq 0 (
        echo [错误] 创建虚拟环境失败。
        pause
        exit /b 1
    )
    echo 正在安装 PyTorch 和 faster-whisper（首次需要几分钟）...
    call "%ENV_PATH%\Scripts\activate.bat"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install faster-whisper
    echo 环境安装完成！
) else (
    call "%ENV_PATH%\Scripts\activate.bat"
)

:: 运行转写（使用虚拟环境内的 python）
echo 开始处理...
python "%SCRIPT_PATH%" "%INPUT%"

echo 处理完成！
pause