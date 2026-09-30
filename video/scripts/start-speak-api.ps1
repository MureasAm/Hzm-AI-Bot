param(
  [string]$ModelRoot = "D:\Huizeman-AI-Voice-Cloning\GPT-SoVITS_V4\GPT-SoVITS_V4_250424",
  [int]$Port = 9881
)

$ErrorActionPreference = "Stop"

$exe = Join-Path $ModelRoot "env\python.exe"
$config = "GPT_SoVITS/configs/tts_infer.yaml"
$logRoot = Resolve-Path (Join-Path $PSScriptRoot "..\out")
$stdout = Join-Path $logRoot "tts-server-$Port.stdout.log"
$stderr = Join-Path $logRoot "tts-server-$Port.stderr.log"

if (-not (Test-Path $exe)) {
  throw "GPT-SoVITS python not found: $exe"
}

$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
  Write-Host "GPT-SoVITS API already listening on $Port (PID $($existing[0].OwningProcess))"
  exit 0
}

$process = Start-Process `
  -FilePath $exe `
  -ArgumentList @("api_v2.py", "-a", "127.0.0.1", "-p", $Port, "-c", $config) `
  -WorkingDirectory $ModelRoot `
  -WindowStyle Hidden `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -PassThru

Write-Host "Started GPT-SoVITS API PID=$($process.Id) port=$Port"
Write-Host "Logs: $stdout"
