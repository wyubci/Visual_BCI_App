# bci_env Environment

This folder contains the reproducible environment files for Visual_BCI_App.

## Create a fresh environment

Run from the repository root in an Anaconda Prompt or a PowerShell session where `conda` is available:

```powershell
.\environment\create_bci_env.ps1
```

To recreate an existing environment:

```powershell
.\environment\create_bci_env.ps1 -Force
```

Then start the app:

```powershell
conda run -n bci_env python .\main.py
```

## Export or pack the current environment

On a machine that already has the real `bci_env` installed:

```powershell
.\environment\export_bci_env.ps1
```

To also create a portable Windows archive:

```powershell
.\environment\export_bci_env.ps1 -Pack
```

The generated lock files and `dist\` archives are local build artifacts and are intentionally ignored by git.
