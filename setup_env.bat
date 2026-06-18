@echo off
REM setup_env.bat - create a conda environment and install project dependencies

REM change to script directory to ensure relative paths work
cd /d "%~dp0"

REM Determine conda executable: either on PATH or fallback location
where.exe /q conda && (
    set "CONDA_EXE=conda"
) || (
    set "CONDA_EXE=D:\anaconda\_conda.exe"
)

REM verify that the chosen executable actually exists
if not exist "%CONDA_EXE%" (
    echo Conda not found on PATH and fallback %CONDA_EXE% does not exist.
    echo Please install Miniconda or Anaconda and/or add conda to PATH.
    pause
    exit /b 1
)

echo Using conda executable: %CONDA_EXE%


REM environment name can be changed if desired
set ENV_NAME=visualbci

echo Creating conda environment "%ENV_NAME%" with Python 3.11...
%CONDA_EXE% create -n %ENV_NAME% python=3.11 -y || (
    echo Failed to create conda environment.
    exit /b 1
)

echo (no shell activation) installing dependencies via 'conda run'...
"%CONDA_EXE%" run -n %ENV_NAME% python -m pip install --upgrade pip setuptools wheel
"%CONDA_EXE%" run -n %ENV_NAME% python -m pip install --prefer-binary -r requirements.txt

echo Environment setup complete. To use it later run:
echo    %CONDA_EXE% activate %ENV_NAME%
pause