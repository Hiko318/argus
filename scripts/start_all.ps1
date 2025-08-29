param(
  [string]$WindowTitle = "FORESIGHT_FEED",
  [int]$Fps = 10,
  [string]$FfmpegPath = ""  # set if ffmpeg isn't on PATH
)

# --------- Repo root guard ---------
if (-not (Test-Path "src\backend\app.py")) {
  Write-Host "Run this from the repo root (contains src\backend\app.py)." -ForegroundColor Yellow
  exit 1
}

# --------- Environment for backend ---------
$env:FS_WINDOW_TITLE = $WindowTitle
$env:FS_CAPTURE_FPS  = "$Fps"
if ($FfmpegPath -ne "") { $env:FS_FFMPEG = $FfmpegPath }

# --------- Check ADB device quickly (optional) ---------
try {
  $adbOut = adb devices 2>$null
  if ($LASTEXITCODE -ne 0) {
    Write-Host "ADB not found or no device yet. Continuing, but scrcpy may fail until a phone is connected."
  } else {
    if (-not ($adbOut -match "device`r?$")) {
      Write-Host "No authorized Android device detected. Plug in phone + enable USB debugging." -ForegroundColor Yellow
    }
  }
} catch { }

# --------- Start scrcpy ---------
Write-Host "Starting scrcpy…" -ForegroundColor Cyan
Start-Process scrcpy -ArgumentList @("--max-fps", $Fps, "--stay-awake", "--window-title", $WindowTitle)

# --------- Start backend (in its own terminal) ---------
Write-Host "Starting backend (uvicorn)…" -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit","-Command",". .\.venv\Scripts\Activate.ps1; uvicorn src.backend.app:app --host 127.0.0.1 --port 8000"

Write-Host "`nOpen http://127.0.0.1:8000 in your browser." -ForegroundColor Green
