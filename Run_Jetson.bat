@echo off
title Jetson Nano Vehicle Analytics Launcher
color 0a

echo ========================================================
echo    Checking connection to Jetson Nano (192.168.10.2)...
echo ========================================================

ping -n 1 192.168.10.2 >nul 2>&1
if errorlevel 1 (
    echo [!] ERROR: Cannot reach Jetson Nano.
    echo Ensure the LAN cable is connected and the board is powered on.
    pause
    exit /b
)

echo [✓] Jetson Nano reached!
echo [+] Opening Analytics Web Dashboard in Browser...
start http://192.168.10.2:5000

echo [+] Starting Inference Engine over SSH...
echo Please enter your Jetson password below:
ssh -t shinya@192.168.10.2 "~/Desktop/Traffic_Analysis_System/start_pipeline.sh"

pause
