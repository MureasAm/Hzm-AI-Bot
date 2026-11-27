@echo off
setlocal
title Hazel bot + GPT-SoVITS TTS

set "LEGACY_DIR=D:\Huizeman-AI-Voice-Cloning\GPT-SoVITS_V4\GPT-SoVITS_V4_250424"
set "TTS_PY=%LEGACY_DIR%\env\python.exe"
set "TTS_DIR=D:\Huizeman-AI-Voice-Cloning\GPT-SoVITS-V5-Turbo"
set "TTS_ARGS=-c GPT_SoVITS/configs/tts_turbo.yaml"
set "SOVITS_PROFILE=v5-turbo"
if /i "%~1"=="v2" (
    set "TTS_DIR=%LEGACY_DIR%"
    set "TTS_ARGS="
    set "SOVITS_PROFILE=v2"
) else (
    set "G2PW=1"
)

if not exist "%TTS_DIR%\api_v2.py" (
    echo Cannot find the GPT-SoVITS runtime at "%TTS_DIR%".
    pause
    exit /b 1
)
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 9880 -State Listen -ErrorAction SilentlyContinue) { exit 1 }"
if errorlevel 1 (
    echo Port 9880 is already in use. Close the current TTS service before starting another profile.
    pause
    exit /b 1
)

echo Starting GPT-SoVITS %SOVITS_PROFILE% on port 9880...
start "GPT-SoVITS %SOVITS_PROFILE%" /D "%TTS_DIR%" cmd /k ""%TTS_PY%" -X utf8 api_v2.py %TTS_ARGS%"
powershell -NoProfile -Command "$deadline = [DateTime]::UtcNow.AddSeconds(45); do { try { $client = New-Object Net.Sockets.TcpClient('127.0.0.1',9880); $client.Dispose(); exit 0 } catch { Start-Sleep -Milliseconds 500 } } while ([DateTime]::UtcNow -lt $deadline); exit 1"
if errorlevel 1 (
    echo GPT-SoVITS did not start within 45 seconds. Check its window for the error.
    pause
    exit /b 1
)

echo Starting Hazel bot. Open the QQ adapter first.
start "Hazel bot" /D "%~dp0." cmd /k "python bot.py"
