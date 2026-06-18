param(
    [string]$EnvName = "bci_env",
    [string]$PythonVersion = "3.9",
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ReqFile = Join-Path $PSScriptRoot "requirements-bci-env.txt"
$WheelPath = Join-Path $RepoRoot "interface\deviceControl_interface\neuro_dance-5.0-py3-none-any.whl"

function Invoke-Step {
    param(
        [string]$Title,
        [string[]]$Command
    )

    Write-Host ""
    Write-Host $Title
    Write-Host ("  " + ($Command -join " "))

    if (-not $DryRun) {
        $Executable = $Command[0]
        $Arguments = @()
        if ($Command.Length -gt 1) {
            $Arguments = $Command[1..($Command.Length - 1)]
        }
        & $Executable @Arguments
    }
}

if ((-not $DryRun) -and (-not (Get-Command conda -ErrorAction SilentlyContinue))) {
    throw "conda was not found in PATH. Open an Anaconda Prompt or run 'conda init powershell' first."
}

if (-not (Test-Path $ReqFile)) {
    throw "Missing requirements file: $ReqFile"
}

$ExistingEnv = $null
if (-not $DryRun) {
    $ExistingEnv = conda env list | Select-String -Pattern "^\s*$EnvName\s"
}
if ($ExistingEnv) {
    if (-not $Force) {
        throw "Conda env '$EnvName' already exists. Re-run with -Force to remove and recreate it."
    }

    Invoke-Step "[1/6] Remove existing conda env: $EnvName" @("conda", "env", "remove", "-n", $EnvName, "-y")
} else {
    Write-Host "[1/6] No existing conda env named '$EnvName'"
}

Invoke-Step "[2/6] Create conda env: $EnvName (python=$PythonVersion)" @(
    "conda", "create", "-n", $EnvName, "python=$PythonVersion", "pip", "-y"
)

Invoke-Step "[3/6] Upgrade packaging tools" @(
    "conda", "run", "-n", $EnvName, "python", "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"
)

Invoke-Step "[4/6] Install project dependencies" @(
    "conda", "run", "-n", $EnvName, "python", "-m", "pip", "install", "-r", $ReqFile
)

if (Test-Path $WheelPath) {
    Invoke-Step "[5/6] Install local neuro_dance wheel" @(
        "conda", "run", "-n", $EnvName, "python", "-m", "pip", "install", $WheelPath
    )
} else {
    Write-Host ""
    Write-Host "[5/6] Optional local wheel not found, skipped:"
    Write-Host "  $WheelPath"
}

Write-Host ""
Write-Host "[6/6] Done"
Write-Host "Run app:"
Write-Host "  conda run -n $EnvName python `"$RepoRoot\main.py`""
