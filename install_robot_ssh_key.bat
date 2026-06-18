@echo off
setlocal

echo Installing local SSH public key to robot authorized_keys...
echo You should only need to enter the robot password this one last time.
echo.

type "%USERPROFILE%\.ssh\id_rsa.pub" | ssh yahboom-car "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"

echo.
echo If no error was shown, passwordless SSH is ready.
pause
endlocal