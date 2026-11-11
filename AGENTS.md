# Project instructions

## Python environment

- For every Python script, module, package-install, and Python check command, invoke `C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe` by its absolute path. Do not depend on `python` resolving through `PATH` or on an activation prompt. Do not require a specific Python version.
- Example:

  ```powershell
  $python = 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe'
  & $python -u Script\codev_startup_check.py
  & $python -u Script\distortion_grid.py --rotation-min 0 --rotation-count 1 --grid-lines 3 --output-dir "$env:TEMP\distortion_grid_check"
  ```

- If activation is useful in a PowerShell command, invoke `& 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\Activate.ps1'` directly in that same command. The `pyenv-win-venv activate venv` wrapper starts a child `cmd`; its activation does not update the parent shell.