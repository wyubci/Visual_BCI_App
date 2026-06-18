@echo off
title Uploading Optimized Car Code
echo ===================================================
echo  UPLOADING OPTIMIZED CAMERA CODE TO ROBOT
echo ===================================================
echo.
echo Connection: pi@192.168.1.11
echo Target: ~/fast_stream.py
echo.
echo Please enter the robot password when prompted below:
echo.
scp -o ConnectTimeout=10 -o StrictHostKeyChecking=no fast_stream_opt.py pi@192.168.1.11:~/fast_stream.py
if %ERRORLEVEL% EQU 0 (
    color 0A
    echo.
    echo [SUCCESS] Code updated successfully!
    echo You can now restart the main application.
) else (
    color 0C
    echo.
    echo [ERROR] Upload failed. Check password or connection.
)
echo.
pause