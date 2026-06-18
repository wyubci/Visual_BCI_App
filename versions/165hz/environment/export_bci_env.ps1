param(
    [string]$EnvName = "bci_env",
    [switch]$Pack,
    [string]$ArchivePath = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$EnvYaml = Join-Path $PSScriptRoot "$EnvName.lock.yml"
$PipFreeze = Join-Path $PSScriptRoot "requirements-$EnvName.lock.txt"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda was not found in PATH. Open an Anaconda Prompt or run 'conda init powershell' first."
}

$ExistingEnv = conda env list | Select-String -Pattern "^\s*$EnvName\s"
if (-not $ExistingEnv) {
    throw "Conda env '$EnvName' was not found. Create it first with environment\create_bci_env.ps1."
}

Write-Host "[1/3] Export conda environment lock file"
conda env export -n $EnvName --no-builds | Out-File -FilePath $EnvYaml -Encoding utf8
Write-Host "  $EnvYaml"

Write-Host "[2/3] Export pip freeze lock file"
conda run -n $EnvName python -m pip freeze | Out-File -FilePath $PipFreeze -Encoding utf8
Write-Host "  $PipFreeze"

if ($Pack) {
    if ([string]::IsNullOrWhiteSpace($ArchivePath)) {
        $DistDir = Join-Path $RepoRoot "dist"
        New-Item -ItemType Directory -Force -Path $DistDir | Out-Null
        $ArchivePath = Join-Path $DistDir "$EnvName-windows.zip"
    }

    Write-Host "[3/3] Pack conda environment archive"
    conda run -n $EnvName python -m pip install --upgrade conda-pack
    conda run -n $EnvName conda-pack -n $EnvName -o $ArchivePath --force
    Write-Host "  $ArchivePath"
} else {
    Write-Host "[3/3] Binary archive skipped. Add -Pack to create dist\$EnvName-windows.zip."
}
