# Distortion tracking handoff

## Goal

Use Python with the CODE V COM API to calculate distortion grids for `Lens/p1_ME.seq` at multiple eye rotations. Apply each eye rotation using `ADE` on the surface labeled `RC`. CODE V performs ray tracing; Python writes the numeric data and creates all visualization.

Continue on another Windows machine with CODE V and access to its license server. Copy this project with its `Lens` and `Script` directories intact. Paths in the distortion script are resolved relative to the script location, so the project can move without editing its root path.

## Files

- `Script/distortion_grid.py`: CODE V collector; opens one COM session and saves only the structured numeric pickle.
- `Script/plot_distortion_grid.py`: standalone plotter; reads the pickle and creates a five-panel PNG without starting CODE V.
- `Script/analyze_distortion.py`: standalone analysis; compares each rotation with the same relative field points at RC 0 and saves pointwise differences plus a summary chart without starting CODE V.
- `Script/raytracing.py`: original user template for COM initialization, `cv_eval`, CODE V commands, and eye rotation.
- `Lens/p1_ME.seq`: target optical model.
- `Script/codev_startup_check.py`: optional standalone check that prints COM creation timing, the return value of `StartCodeV()`, and the CODE V version. It stops its CODE V session when finished.

## Environment

No specific Python version is required. On the machine used for validation, the named `venv` is managed by `pyenv-win-venv` and lives under `%USERPROFILE%\.pyenv-win-venv\envs\venv`; it currently uses Python 3.13.11 and has `pywin32` installed.

In PowerShell, use the environment's interpreter by its absolute path:

```powershell
$distortionPython = "$env:USERPROFILE\.pyenv-win-venv\envs\venv\Scripts\python.exe"
& $distortionPython --version
& $distortionPython -m pip show pywin32
```

Install `pywin32` in that environment only if it is missing:

```powershell
& $distortionPython -m pip install pywin32
```

If activation is useful, invoke its `Activate.ps1` directly in the same PowerShell process. The `pyenv-win-venv activate venv` wrapper starts a child shell in this setup, so its prompt can appear active while the parent process still resolves another Python.

The collector uses NumPy for its structured numeric array and Python's standard `pickle` module to save it. The standalone plotter uses NumPy and Matplotlib with the noninteractive Agg backend to write PNG. The original `raytracing.py` has additional dependencies and an IPython plotting setup.

COM initialization follows the original template:

```python
pythoncom.CoUninitialize()
pythoncom.CoInitialize()
cv = win32com.client.Dispatch("CodeV.Application")
cv.StartingDirectory = str(LENS_DIR)
cv.StartCodeV()
```

The distortion script also sets `CommandTimeout = 600000` and `MaxTextBufferSize = 1000000` before `StartCodeV()`. It calls `StopCodeV()` and `CoUninitialize()` during cleanup. Allow initialization to complete before issuing optical commands.

## Model and calculation

The target model has one wavelength, 850 nm (`REF 1`), millimeter units (`DIM M`), and angular fields extending to ±10 degrees in X and Y. The sequence contains a separate surface `SRC_ROT` with an existing `ADE -20` setting. The current implementation changes only `RC` for the eye rotation sweep; retain the source rotation unless the user requests a change.

The model is loaded using the user's command pattern:

```python
cv.Command(f'run "{SEQ_FILE}";go')
```

`calculate_grid_at_rotation()` follows the installed `dist.seq` calculation:

1. Set `ade s"RC" <angle>;set vig`, then read back `(ade s"RC")` and stop if CODE V did not apply the requested angle. This explicit surface-label selector matches `Script/raytracing.py`'s `S_RC` definition and command.
2. Set two angular fields using the original maximum X/Y semi-fields, then obtain their equivalent `XOB` and `YOB` values and switch to object-height fields.
3. Trace the center chief ray and a chief ray at relative field `(0.05, 0.05)`.
4. Construct the reference grid from the image scale at that central 5% field. Grid increments are `40 * (near_axis - center) / (grid_lines - 1)` for each axis.
5. Trace the grid using `cv.RAYRSI(1, 1, 0, 0, [0.0, 0.0, relative_x, relative_y])`. A return value of zero indicates success. Read image coordinates using `EvaluateExpression("(x si)")` and `EvaluateExpression("(y si)")`.
6. Subtract the center image coordinates and calculate radial/tangential distortion percentages using the convention in `dist.seq`.

Relative grid fields describe equivalent object height after the field conversion. The pickle array's `paraxial_*` fields retain the terminology used by `dist.seq`; their reference scale comes from the central 5% real ray rather than a separate paraxial ray trace.

For reference image position `(px, py)`, residual `(dx, dy)`, and `r² = px² + py²`:

```text
radial_percent     = 100 * (dx*px + dy*py) / r²
tangential_percent = 100 * (dx*py - dy*px) / r²
```

Both values are zero at the center. The tangential sign was corrected to match the installed macro. Failed interior grid rays are represented by NaN, and plotted lines break at missing points. A failed center, reference ray, or edge grid ray raises an error.

This implementation targets the finite image plane in `p1_ME.seq`. It does not implement the macro's afocal direction-coordinate branch or multiple zoom positions.

## Run commands

From the project root, set the named environment's interpreter:

```powershell
$distortionPython = "$env:USERPROFILE\.pyenv-win-venv\envs\venv\Scripts\python.exe"
```

For a small CODE V smoke run, write the pickle and plot to the system temporary directory:

```powershell
& $distortionPython -u Script\distortion_grid.py --rotation-min 0 --rotation-count 1 --grid-lines 3 --output-dir "$env:TEMP\distortion_grid_check"
& $distortionPython -u Script\plot_distortion_grid.py --input "$env:TEMP\distortion_grid_check\distortion_grid.pkl" --output "$env:TEMP\distortion_grid_check\distortion_grid.png" --rotations 0
```

The default collection runs one CODE V session for 401 eye rotations from -20 to +20 degrees in 0.1-degree steps, with an 11 ? 11 grid at each rotation. It writes `data/distortion_grid/distortion_grid.pkl`:

```powershell
& $distortionPython -u Script\distortion_grid.py
```

Plot the requested five rotations from the saved pickle without starting CODE V:

```powershell
& $distortionPython -u Script\plot_distortion_grid.py
```

Analyze radial and tangential distortion changes against RC 0 degrees, matched by relative X/Y field point. The reported differences are percentage points (the distortion percentages are subtracted), and the summary includes the maximum absolute and RMS change at every stored angle:

```powershell
& $distortionPython -u Script\analyze_distortion.py
```

This writes `data/distortion_grid/distortion_change_from_zero.pkl` (pointwise changes and per-angle summary) and `data/distortion_grid/distortion_change_from_zero.png` (maximum absolute and RMS changes versus rotation). Use `--input` and `--output-dir` to select alternate paths.

The plotter defaults to RC -10, -5, 0, 5, and 10 degrees. To select other stored angles or paths, use `--rotations`, `--input`, and `--output`.

Investigate whether five detected points (center and four grid corners) can estimate eye rotation from the saved sweep. This analysis reads the pickle only and does not start CODE V:

```powershell
& $distortionPython -u Script\analyze_eye_rotation.py
```

It writes a separate `data/eye_rotation_analysis` folder containing `report.md`, `eye_rotation_observability.pkl`, and PNG plots of corner-coordinate/distortion changes and conditional precision. The coordinate inverse uses the four corners relative to the measured center; the collected data already recenter coordinates at the center for each rotation. Its seeded Monte Carlo result assumes independent Gaussian noise with 1 µm standard deviation per X and Y coordinate on each of the five measured points. This is an idealized simulation using known field identities and exact CODE V templates, not hardware precision. The reusable `estimate_rotation()` function is documented in the analysis script.

## Outputs

- Collector: `data/distortion_grid/distortion_grid.pkl`, a one-dimensional structured NumPy array with nine `float64` fields in the original CSV column order: `eye_rotation_deg`, `field_x_relative`, `field_y_relative`, `paraxial_x_mm`, `paraxial_y_mm`, `real_x_mm`, `real_y_mm`, `radial_distortion_pct`, and `tangential_distortion_pct`.
- Plotter: `data/distortion_grid/distortion_grid.png`, one 1 × 5 PNG figure with a labeled grid panel for each default selected rotation. Panels share image-coordinate limits and equal aspect for direct visual comparison; reference grids are dashed gray, real grids are blue, and distortion vectors are red.

Rerunning the plotter only reads the pickle; it does not start or connect to CODE V.

## References already reviewed

These paths refer to the original CODE V 2024.03 SR1 installation. Use the corresponding installation directory on the destination machine.

- `C:\CODEV202403_SR1\com\Example_CVApplication.py`: COM methods, properties, events, and direct `RAYRSI` example.
- `C:\CODEV202403_SR1\com\CODEV_PSF_1FLD_Example.py`: simple `Dispatch` / `StartCodeV` connection and Python visualization example.
- `C:\CODEV202403_SR1\macro\dist.seq`: authoritative source for the grid field setup, central reference scale, ray coordinates, and listed distortion values.
- Help: `index.html#page/api/1Overview.2.2.html`.
- Help: `index.html#page/macroplus/Macro-PLUS.html#ww16575`.
- Help: `index.html#page/macroplus/SPHIST.html#ww493` (the supplied anchor leads to the DIST example).
- Help: `index.html#page/diagnostic/HIDD_DISTORTION_GRID.html`.
- Help: `index.html#page/macroplus/HIDC_DISTGRID_ListDistortionValues.html#ww46196`.
- Help: `index.html#page/connect/splash.html`.

The installed `dist.seq` creates CODE V plot output even when numeric listing is enabled. The Python implementation traces directly through COM instead of running that plotting macro. For a one-time numeric comparison, use the macro's list option `YES`, for example:

```text
in cv_macro:dist 0 0 "" 2 "RED" 11 1 "YES"
```

## Validation results

Validation completed on 8 October 2026 with CODE V 2024.03 SR1 Build (42748259), Python 3.13.11 from the named `venv`, `pywin32`, and NumPy 2.4.3. No fixed Python version is required.

Earlier collector runs and the earlier macro comparison are invalid as eye-rotation validation: the collector issued `ade "RC"`, which omitted the surface selector used by `raytracing.py` (`S_RC = "s\"RC\""`). That command did not apply the requested RC rotation, so the prior apparent macro match and identical-angle outputs must not be relied on. The collector now uses the same `ade {S_RC}` command as `raytracing.py`, verifies each requested value by evaluating `(ade s"RC")`, and stops if CODE V reports an error or the readback differs.

A corrected smoke run at RC -10, 0, and +10 degrees confirmed the three readbacks and produced distinct grids. At 3 × 3 resolution, the maximum absolute differences against RC=0 were 0.32846 and 0.35895 percentage points in radial distortion and 0.13095 and 0.14219 percentage points in tangential distortion for RC=-10 and +10, respectively.

The corrected full collection completed in one CODE V session on 8 October 2026. All 401 requested angles from -20 through +20 degrees at 0.1-degree steps passed ADE readback checks. The pickle contains 48,521 rows, exactly 121 per angle; all stored grid coordinates and distortion values are finite, and all 401 center rows have zero radial and tangential distortion. No edge rays failed. The standalone plotter successfully generated the requested -10, -5, 0, +5, and +10 degree panels in one 1 × 5 PNG with shared axes. Their maximum absolute radial distortion values are 2.3001%, 2.4101%, 2.5359%, 2.6743%, and 2.8218%, respectively; maximum absolute tangential values are 2.1173%, 2.1760%, 2.2410%, 2.3097%, and 2.3802%.

The sequence retains its original `SRC_ROT ADE -20` setting; eye rotations are applied only to `RC`. The current output folder also contains an older `distortion_grid.svg` and `validation_report.md` from before the surface-selector correction. Those two historical artifacts are not valid rotation evidence; the scripts do not remove prior outputs automatically.

## Continue on another machine

Copy the project with `Lens` and `Script` intact, confirm CODE V and its license are available, and install `pywin32`, NumPy, and Matplotlib in the Python environment if any are missing. Use the environment's interpreter directly as shown above. Run the small grid first, compare with the installed `dist.seq` listing when validating, then collect the 401-angle sweep and run the standalone plotter. The small-grid commands write to the system temporary directory. Paths in both scripts resolve relative to the project, so its root can move without editing them.
