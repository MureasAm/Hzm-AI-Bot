param(
  [string]$ApiBase = "http://127.0.0.1:9881",
  [string]$RefAudio = "D:/my_qq_bot/my_qq_bot/assets/voice_refs/ref_voice.wav",
  [string]$PromptText = "有没有可能居委会小区居委会也说会啊，虽然没说开的是小区居委会",
  [int]$Seed = 42,
  [double]$SpeedFactor = 1.0
)

$ErrorActionPreference = "Stop"

$videoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$csvPath = Join-Path $videoRoot "narration.csv"
$audioDir = Join-Path $videoRoot "public\audio"
$reportPath = Join-Path $audioDir "voiceover-report.json"
$ffprobe = Resolve-Path (Join-Path $videoRoot "..\bin\ffprobe.exe")

New-Item -ItemType Directory -Force -Path $audioDir | Out-Null

$rows = Import-Csv $csvPath
$report = @()

foreach ($row in $rows) {
  $outPath = Join-Path $audioDir "$($row.line_id).wav"
  $body = @{
    text = $row.text
    text_lang = "zh"
    ref_audio_path = $RefAudio
    prompt_text = $PromptText
    prompt_lang = "zh"
    text_split_method = "cut0"
    batch_size = 1
    speed_factor = $SpeedFactor
    streaming_mode = $false
    seed = $Seed
    parallel_infer = $true
    repetition_penalty = 1.35
  } | ConvertTo-Json -Depth 4

  Write-Host "Synthesizing $($row.line_id): $($row.text)"
  Invoke-WebRequest `
    -UseBasicParsing `
    -Method Post `
    -Uri "$ApiBase/tts" `
    -ContentType "application/json" `
    -Body $body `
    -OutFile $outPath `
    -TimeoutSec 180

  $durationText = & $ffprobe -v error -show_entries format=duration -of csv=p=0 $outPath
  $durationMs = [int][math]::Round([double]$durationText * 1000)
  $slotMs = [int]$row.duration_ms
  $overrun = $durationMs - $slotMs

  $report += [pscustomobject][ordered]@{
    line_id = $row.line_id
    scene_id = $row.scene_id
    start_ms = [int]$row.start_ms
    end_ms = [int]$row.end_ms
    slot_ms = $slotMs
    audio_ms = $durationMs
    overrun_ms = $overrun
    text = $row.text
    audio_file = "public/audio/$($row.line_id).wav"
  }
}

$report | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 $reportPath

$overruns = @($report | Where-Object { $_.overrun_ms -gt 0 })
Write-Host ""
Write-Host "Generated $($report.Count) voice lines."
Write-Host "Report: $reportPath"
if ($overruns.Count -gt 0) {
  Write-Host "WARNING: $($overruns.Count) line(s) exceed their timeline slot:"
  $overruns | Format-Table line_id, slot_ms, audio_ms, overrun_ms -AutoSize
} else {
  Write-Host "All lines fit their timeline slots."
}
