@echo off
REM Launch scrcpy (window name forced so ffmpeg can grab it)
echo Starting scrcpy...
start "" scrcpy --max-fps 30 --stay-awake --window-title "scrcpy"

REM Wait longer for the window to fully appear and phone to connect
echo Waiting for scrcpy window to appear...
timeout /t 8 >nul

REM Check if window exists before proceeding
echo Looking for scrcpy window...
tasklist /fi "imagename eq scrcpy.exe" 2>nul | find /i "scrcpy.exe" >nul
if errorlevel 1 (
    echo ERROR: scrcpy is not running. Make sure your phone is connected.
    pause
    exit /b 1
)

echo Starting FFmpeg capture...
C:\ffmpeg\ffmpeg.exe -f gdigrab -framerate 30 -i title="scrcpy" ^
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ^
  -fflags nobuffer -flags low_delay ^
  -vcodec libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p ^
  -f rtsp -rtsp_transport tcp -muxdelay 0.1 -listen 1 rtsp://127.0.0.1:8554/scrcpy
