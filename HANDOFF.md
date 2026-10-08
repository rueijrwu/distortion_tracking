# Distortion tracking handoff

## Project goal

Estimate one-axis eye rotation from five detected image points: the center and the top-left, top-right, bottom-left, and bottom-right grid points. The CODE V sweep provides ground-truth rotation labels and the ideal image coordinates for calibration. The main question is how accurately rotation can be estimated when the five point coordinates are noisy.

The current five-point inverse estimator uses the evaluated quadratic forward model. It subtracts the measured center and the zero-degree corner coordinates, then globally minimizes the weighted residual over -20 to +20 degrees by evaluating every real stationary root and both endpoints of the quadratic objective. The fit coefficients are loaded from `data\polynomial_distortion_fit\polynomial_distortion_fit.pkl`.

## Project files

- `Lens/p1_ME.seq`: target CODE V lens model.
- `Script/distortion_grid.py`: collects the rotation sweep in one CODE V COM session and saves a structured NumPy array with Python pickle.
- `Script/plot_distortion_grid.py`: plots selected rotations from the pickle using Matplotlib; it does not start CODE V.
- `Script/analyze_distortion.py`: compares radial and tangential distortion percentages with their values at RC 0 degrees.
- `Script/analyze_eye_rotation.py`: studies five-point rotation observability and evaluates the quadratic inverse under 1 µm coordinate noise; it reads the sweep/model pickles and does not start CODE V.
- `Script/fit_polynomial_distortion.py`: fits the Theory.md quadratic forward model from the saved ground-truth pickle and reports full-sweep fit errors; it does not start CODE V.
- `Script/codev_startup_check.py`: optional CODE V COM startup check.
- `Script/raytracing.py`: original COM/ray-tracing reference, including the required `S_RC = "s\"RC\""` surface selector.
- `Theory.md`: quadratic forward-model notation, measured fit and inverse results, and rotation-estimation considerations.
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

The current sweep uses 401 rotations from -20° to +20° in 0.1° increments and a 3 × 3 field grid at each rotation. The nine nodes include the center and four corners used by the five-point estimator. Its pickle has 3,609 records (9 per angle), stored as a one-dimensional structured NumPy array with these nine `float64` fields:

`eye_rotation_deg`, `field_x_relative`, `field_y_relative`, `paraxial_x_mm`, `paraxial_y_mm`, `real_x_mm`, `real_y_mm`, `radial_distortion_pct`, `tangential_distortion_pct`.

The `real_x_mm` and `real_y_mm` image coordinates are already relative to the center ray at the same rotation. Radial and tangential distortion are percentages; their changes are reported in percentage points. Those percentages are distinct from the image-coordinate shifts used by the current five-point rotation estimator.

## Existing data and results

The current full sweep is `data\distortion_grid\distortion_grid.pkl`. A CSV export with the same nine fields and 3,609 rows is available at `data\distortion_grid\distortion_grid.csv`. It was collected after the optical setup changed, with the explicit RC surface selector and per-angle readback checks. All 401 angles completed, with 9 finite rows per angle, including all four corners and the center. The other four nodes are the edge midpoints. All coordinates and distortion values are finite.

The selected grid plot is `data\distortion_grid\distortion_grid.png`; it shows RC -10°, -5°, 0°, +5°, and +10° with shared image-coordinate axes. Generate it from the existing pickle without running CODE V:

```powershell
& $distortionPython -u Script\plot_distortion_grid.py
```

The baseline distortion comparison is saved in `data\distortion_grid\distortion_change_from_zero.pkl` and `data\distortion_grid\distortion_change_from_zero.png`. Recreate it from the pickle with:

```powershell
& $distortionPython -u Script\analyze_distortion.py
```

The five-point study outputs are in `data\eye_rotation_analysis\`: `report.md`, `eye_rotation_observability.pkl`, `dependencies_vs_rotation.png`, and `precision_estimates.png`. The 1 × 5 grid plot infers the 3 × 3 layout from the pickle and shows RC -10°, -5°, 0°, +5°, and +10°. Recreate the study with the saved quadratic fit using:

```powershell
& $distortionPython -u Script\analyze_eye_rotation.py
```

The inverse function `estimate_rotation_polynomial()` takes five absolute measured points in millimeters, ordered center, top-left, top-right, bottom-left, bottom-right, plus the saved scaled coefficients and zero-degree corner coordinates. It subtracts the measured center and zero-degree pattern, then minimizes the weighted residual of the quadratic forward model. In normalized angle `t=theta/20`, the objective is quartic; its cubic derivative roots and both endpoints are evaluated, so the selected angle is the global minimum on the requested interval. The weight accounts for shared noise after center subtraction: `W = I - 11ᵀ/5` for each coordinate axis.

The analysis defaults to `data\polynomial_distortion_fit\polynomial_distortion_fit.pkl`; override it with `--model-pickle` if needed. It checks the saved scaled monomial basis `(theta/20)^2, theta/20, 1`, point/axis order, predictions, and conversion to the physical `[theta^2, theta, 1]` coefficients before estimating rotation.

The 1 µm study assumes independent Gaussian noise with standard deviation 0.001 mm on each X and Y coordinate of all five points. It samples 101 true angles with 100 trials per angle (10,100 trials total), uses exact CODE V coordinates as the noise-free measurements, and includes the fitted quadratic's approximation bias. It assumes known point identities and omits detector calibration, alignment, localization bias, and other systematic errors. Results are conditional simulation results, not measured hardware precision:

| Measure | Result |
|---|---:|
| Noiseless median / P95 / maximum absolute bias | 2.22169 / 3.22499 / 4.17571 arcmin (maximum at +20°) |
| Median absolute error with 1 µm coordinate noise | 8.79205 arcmin |
| Pooled 95th-percentile absolute error | 36.6595 arcmin |
| RMSE | 17.2944 arcmin |
| Trials with error greater than 1° | 0.7822% |
| Trials with error greater than 5° | 0% |
| Largest observed error | 111.015 arcmin (truth +0.8°, estimate +2.65025°) |
| Worst per-angle 95th percentile | 68.0692 arcmin at truth +0.4° |
| Endpoint clamp rate (-20° / +20°) | 0.2079% / 0.2376% |

Performance varies greatly with true angle. The 0.1° sweep spacing is the ground-truth data sampling interval, not a demonstrated measurement accuracy. The polynomial inverse is continuous in angle, but the reported noise study does not establish sub-sample hardware precision. The center anchors the absolute measurement; after center subtraction, it adds no independent angle feature in this dataset, and its measurement noise is shared across all four relative corners.

## Quadratic forward-model fit

`Script/fit_polynomial_distortion.py` fits the evaluated forward model in `Theory.md` from `data\distortion_grid\distortion_grid.pkl`. It models each point's center-relative real image-coordinate shift from the RC 0° pattern in millimeters:

`Δ_i(θ) = [θ², θ, 1] C_i`, where `Δ_i = [dx_i, dy_i]`.

The fit scales angle as `t=θ/20` for numerical stability, uses `numpy.linalg.lstsq`, and saves the coefficients converted to the physical `[θ², θ, 1]` basis. It fits all 401 ground-truth rotations and reports the residual between the fitted model and those same samples. Outputs are written to `data\polynomial_distortion_fit\`: `report.md`, `polynomial_distortion_fit.pkl`, `truth_vs_quadratic.png`, and `quadratic_residuals.png`.

Recreate those outputs without running CODE V:

```powershell
& $distortionPython -u Script\fit_polynomial_distortion.py
```

Latest full-sweep fit for the changed optical setup: corner-coordinate RMSE 0.174111 µm, MAE 0.141691 µm, and maximum absolute coordinate residual 0.746626 µm at bottom-right Y and +20°. The vector RMSE is 0.246230 µm. These deterministic model residuals are below the assumed 1 µm per-coordinate localization noise; they describe the forward coordinate fit and are separate from angle-estimation error. The refreshed quadratic inverse results are in `data\eye_rotation_analysis\report.md`; the results there are in arcminutes and it returns continuous angles by global weighted residual minimization.

## Validation and historical artifacts

The corrected full sweep's per-angle ADE readbacks and finite records are the relevant collection validation. Earlier collector runs and the earlier purported `dist.seq` match are invalid as eye-rotation validation because the command omitted the explicit RC surface selector; do not use those results.

`data\distortion_grid\distortion_grid.svg` and `data\distortion_grid\validation_report.md` are historical outputs and predate the current optical setup; do not use them as current results. The previous 11 × 11 pickle, fit, and noise-analysis results were replaced by the current 3 × 3 sweep and regenerated downstream outputs. The scripts do not delete old outputs automatically. A matching macro comparison has not been completed for the current collector.

## Resume checklist

1. Review `data\polynomial_distortion_fit\report.md` and `data\eye_rotation_analysis\report.md` with their residual and precision plots. The latter reports both forward coordinate approximation error and angle-estimation error.
2. For further work, compare physical measurements with the conditional 1 µm model results and assess calibration, alignment, and point-localization systematics.
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
