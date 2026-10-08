# Distortion tracking handoff

## Project goal

Estimate one-axis eye rotation from five detected image points: the center and the top-left, top-right, bottom-left, and bottom-right grid points. The CODE V sweep provides ground-truth rotation labels and the ideal image coordinates for calibration. The main question is how accurately rotation can be estimated when the five point coordinates are noisy.

The current five-point inverse estimator uses the four corner coordinates relative to the measured center. It matches that eight-value X/Y pattern to the saved angle sweep with a weighted piecewise-linear least-squares fit. The quadratic forward model has been fit to the five-point shifts from the ground-truth sweep as described in `Theory.md`; the polynomial has not replaced the inverse estimator.

## Project files

- `Lens/p1_ME.seq`: target CODE V lens model.
- `Script/distortion_grid.py`: collects the rotation sweep in one CODE V COM session and saves a structured NumPy array with Python pickle.
- `Script/plot_distortion_grid.py`: plots selected rotations from the pickle using Matplotlib; it does not start CODE V.
- `Script/analyze_distortion.py`: compares radial and tangential distortion percentages with their values at RC 0 degrees.
- `Script/analyze_eye_rotation.py`: studies five-point rotation observability and 1 µm coordinate-noise performance; it reads the sweep and does not start CODE V.
- `Script/fit_polynomial_distortion.py`: fits the Theory.md quadratic forward model from the saved ground-truth pickle and reports full-sweep fit errors; it does not start CODE V.
- `Script/codev_startup_check.py`: optional CODE V COM startup check.
- `Script/raytracing.py`: original COM/ray-tracing reference, including the required `S_RC = "s\"RC\""` surface selector.
- `Theory.md`: quadratic forward-model notation, measured fit results, and rotation-estimation considerations. The forward polynomial is implemented for evaluation; the rotation inverse remains the piecewise-linear template method.
- `AGENTS.md`: project-specific environment and CODE V workflow instructions.

## Python environment

Use this interpreter by its absolute path for every Python script, module, install, or check command:

```powershell
$distortionPython = 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe'
& $distortionPython --version
```

No particular Python version is required. Python 3.13.11 was present when the latest results were generated. Required packages are NumPy and Matplotlib; collecting from CODE V also requires `pywin32`. Install a missing package into this environment with, for example:

```powershell
& $distortionPython -m pip install numpy matplotlib pywin32
```

If activation is useful, run `& 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\Activate.ps1'` directly in the same PowerShell command. The `pyenv-win-venv activate venv` wrapper starts a child `cmd`, so it does not change the parent PowerShell's Python.

## CODE V model and collector

The target is `Lens\p1_ME.seq`. Keep the sequence's existing `SRC_ROT ADE -20` setting unless the user requests otherwise. Apply eye rotation as `ADE` on surface `RC`, using the exact selector from `raytracing.py`:

```python
S_RC = "s\"RC\""
```

The collector sets the requested `ADE`, runs `set vig`, reads `(ade s"RC")` back for every angle, and stops if CODE V reports an error or the readback does not match. Do not shorten the command to `ade "RC"`; that earlier form failed to select the intended surface and produced misleading, angle-invariant results. The collector starts CODE V once, loads the lens sequence, traces the grid for each requested angle, and cleans up with `StopCodeV()` and `pythoncom.CoUninitialize()`.

The completed sweep uses 401 rotations from -20° to +20° in 0.1° increments and an 11 × 11 field grid at each rotation. Its pickle has 48,521 records (121 per angle), stored as a one-dimensional structured NumPy array with these nine `float64` fields:

`eye_rotation_deg`, `field_x_relative`, `field_y_relative`, `paraxial_x_mm`, `paraxial_y_mm`, `real_x_mm`, `real_y_mm`, `radial_distortion_pct`, `tangential_distortion_pct`.

The `real_x_mm` and `real_y_mm` image coordinates are already relative to the center ray at the same rotation. Radial and tangential distortion are percentages; their changes are reported in percentage points. Those percentages are distinct from the image-coordinate shifts used by the current five-point rotation estimator.

## Existing data and results

The current full sweep is `data\distortion_grid\distortion_grid.pkl`. A CSV export with the same nine fields and 48,521 rows is available at `data\distortion_grid\distortion_grid.csv`. It was collected with the explicit RC surface selector and per-angle readback checks. All 401 angles completed, with 121 finite rows per angle, no failed edge rays, finite coordinates/distortion values, and zero center distortion. The successful collection and 1 × 5 plot were recorded on 8 October 2026 using CODE V 2024.03 SR1 Build 42748259.

The selected grid plot is `data\distortion_grid\distortion_grid.png`; it shows RC -10°, -5°, 0°, +5°, and +10° with shared image-coordinate axes. Generate it from the existing pickle without running CODE V:

```powershell
& $distortionPython -u Script\plot_distortion_grid.py
```

The baseline distortion comparison is saved in `data\distortion_grid\distortion_change_from_zero.pkl` and `data\distortion_grid\distortion_change_from_zero.png`. Recreate it from the pickle with:

```powershell
& $distortionPython -u Script\analyze_distortion.py
```

The five-point study outputs are in `data\eye_rotation_analysis\`: `report.md`, `eye_rotation_observability.pkl`, `dependencies_vs_rotation.png`, and `precision_estimates.png`. Recreate them with:

```powershell
& $distortionPython -u Script\analyze_eye_rotation.py
```

The current inverse model takes five absolute measured points in millimeters, ordered center, top-left, top-right, bottom-left, bottom-right. It subtracts the measured center and compares the four corner coordinates against the sweep templates. Between adjacent 0.1° templates it projects the measurement onto the line segment using weighted least squares, then chooses the segment with the lowest residual. The weight accounts for the shared noise introduced when the measured center is subtracted: `W = I - 11ᵀ/5` for the four corners, up to the common noise scale.

The 1 µm study assumes independent Gaussian noise with standard deviation 0.001 mm on each X and Y coordinate of all five points. It samples 101 true angles with 100 trials per angle (10,100 trials total), assumes known point identities and exact CODE V templates, and omits detector calibration, alignment, and other systematic errors. Results are conditional simulation results, not measured hardware precision:

| Measure | Result |
|---|---:|
| Median absolute rotation error | 0.258893° |
| Pooled 95th-percentile absolute error | 1.89326° |
| RMSE | 0.924109° |
| Trials with error greater than 1° | 13.4% |
| Trials with error greater than 5° | 0.485% |
| Largest observed error | 14.5486° (truth +17.2°, estimate +2.65139°) |
| Worst per-angle 95th percentile | 5.53341° at truth +7.6° |

Performance varies greatly with true angle. The 0.1° grid spacing and interpolation support a continuous numerical estimate, but do not establish 0.1° measurement accuracy. The center anchors the absolute measurement; after center subtraction, it does not add an independent angle feature in this dataset, and its measurement noise is shared across all four relative corners.

The study's leave-one-out interpolation error (median 0.000430581°, maximum 0.00117622°) measures noiseless template interpolation/discretization only. It is not a precision estimate.

## Quadratic forward-model fit

`Script/fit_polynomial_distortion.py` fits the evaluated forward model in `Theory.md` from `data\distortion_grid\distortion_grid.pkl`. It models each point's center-relative real image-coordinate shift from the RC 0° pattern in millimeters:

`Δ_i(θ) = [θ², θ, 1] C_i`, where `Δ_i = [dx_i, dy_i]`.

The fit scales angle as `t=θ/20` for numerical stability, uses `numpy.linalg.lstsq`, and saves the coefficients converted to the physical `[θ², θ, 1]` basis. It fits all 401 ground-truth rotations and reports the residual between the fitted model and those same samples. Outputs are written to `data\polynomial_distortion_fit\`: `report.md`, `polynomial_distortion_fit.pkl`, `truth_vs_quadratic.png`, and `quadratic_residuals.png`.

Recreate those outputs without running CODE V:

```powershell
& $distortionPython -u Script\fit_polynomial_distortion.py
```

Latest full-sweep fit: corner-coordinate RMSE 0.12461 µm, MAE 0.09273 µm, and maximum absolute coordinate residual 0.642517 µm at top-left Y and -20°. The vector RMSE is 0.176225 µm. These deterministic model residuals are below the assumed 1 µm per-coordinate localization noise and describe how well the quadratic approximates this same sweep; they do not establish eye-rotation precision. The existing piecewise-linear inverse in `analyze_eye_rotation.py` remains unchanged; a polynomial inverse and its angle-estimation precision have not been evaluated.

## Validation and historical artifacts

The corrected full sweep's per-angle ADE readbacks and finite records are the relevant collection validation. Earlier collector runs and the earlier purported `dist.seq` match are invalid as eye-rotation validation because the command omitted the explicit RC surface selector; do not use those results.

`data\distortion_grid\distortion_grid.svg` and `data\distortion_grid\validation_report.md` are historical outputs from the earlier workflow. They predate the corrected collector and are not evidence for the 401-angle rotation sweep. The scripts do not delete old outputs automatically. A matching macro comparison has not been completed for the corrected collector.

## Resume checklist

1. Review `data\polynomial_distortion_fit\report.md` and residual plots to assess whether the forward quadratic is adequate for the intended noise level.
2. If proceeding with a polynomial inverse, evaluate angle-estimation error with the same five-point, 1 µm noise assumptions and compare against the current piecewise-linear estimator.
3. Keep the current scope to one-axis RC rotation and the five identified grid points unless the user expands it.

Only rerun CODE V collection if the sweep needs regeneration or settings change. To collect the default 401-angle sweep:

```powershell
& $distortionPython -u Script\distortion_grid.py
```

For a small smoke run that writes to the system temporary folder:

```powershell
& $distortionPython -u Script\distortion_grid.py --rotation-min 0 --rotation-count 1 --grid-lines 3 --output-dir "$env:TEMP\distortion_grid_check"
```

The collector needs Windows, CODE V, a working license, and `pywin32`. Its default output is `data\distortion_grid\distortion_grid.pkl`; all script paths otherwise resolve relative to the project.
