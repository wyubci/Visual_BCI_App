param(
    [string]$EnvName = "bci_env",
    [string]$PythonVersion = "3.9",
    [switch]$Force,
    [switch]$DryRun
)

$Script = Join-Path $PSScriptRoot "environment\create_bci_env.ps1"

& $Script -EnvName $EnvName -PythonVersion $PythonVersion -Force:$Force -DryRun:$DryRun
