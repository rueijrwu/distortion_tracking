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

## Delegation

- Delegate mechanical file reads, edits, command execution, and checks to a `gpt-6-luna` subagent at low reasoning effort. Keep the main agent focused on audit, planning, and orchestration. If subagents are unavailable, continue directly.

## CODE V distortion workflow

- Use `Script\distortion_grid.py` to collect the 401-angle sweep into a pickle, then use `Script\plot_distortion_grid.py` to plot selected rotations without CODE V. The target model is `Lens\p1_ME.seq`. See [Summary.md](Summary.md) and [Theory.md](Theory.md) for current dataset provenance and fitted formulas (`HANDOFF.md` was merged into `Summary.md` and removed).
- Keep the existing `SRC_ROT` `ADE -20` setting unless the user requests a change. Apply eye-rotation sweep values through `ADE` on the surface labeled `RC`.
- The collector starts CODE V once, sets each rotation with the explicit `S_RC = "s\"RC\""` surface selector, verifies the ADE readback, and writes only a structured NumPy pickle. The plotter reads that pickle and writes a 1 × 5 PNG using Matplotlib Agg without importing or starting CODE V. NumPy and Matplotlib must be installed in the named environment. Allow CODE V startup to finish before issuing optical commands, and clean up with `StopCodeV()` and `pythoncom.CoUninitialize()`.
- A small-grid validation step (11 × 11 grid vs. `dist.seq`) was previously referenced here but its description was never found in the docs; do not assume it exists until confirmed.
