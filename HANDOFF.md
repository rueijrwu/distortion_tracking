# Distortion model handoff

## Project goal

Identify the baseline barrel distortion and the eye-rotation-dependent transformation for the P1 and P4 optical setups. P1 uses its measured real-coordinate grid at zero rotation as the empirical barrel-distorted source and fits a vertical-keystone transform across rotation. P4 fits radial `k1` across accommodation at zero rotation, plus a separate baseline-plus-keystone transform across rotation at 0 D. These fits are setup-specific and are not a jointly calibrated model.

## Core files

- `Lens/p1_ME.seq`, `Lens/p4_ME.len`, and related lens source files: optical models; preserve their current settings.
- `Script/distortion_grid.py` and `Script/distortion_grid_p4.py`: CODE V sweep collectors.
- `Script/plot_distortion_grid.py` and `Script/plot_distortion_grid_p4.py`: plots from saved structured pickles without CODE V.
- `Script/analyze_p1_rotation_transform.py` and `Script/analyze_p4_rotation_transform.py`: baseline-plus-keystone fits.
- `Script/fit_p4_radial_distortion.py`: radial `k1` and accommodation-curve fits for P4.
- `Script/raytracing.py`: COM/ray-tracing reference and explicit RC selector.
- `Script/codev_startup_check.py`: optional CODE V startup check.
- `Theory.md` and `Summary.md`: model definitions, results, and limitations.

## Python environment

Use this interpreter by its absolute path for every Python script, module, install, or check command:

```powershell
$distortionPython = 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe'
& $distortionPython --version
```

Required packages are NumPy and Matplotlib; collecting from CODE V also requires `pywin32`. Use CODE V only for collection. The plotting and fitting scripts consume saved data and do not start CODE V.

## P1 grid and rotation transform

The canonical P1 sweep is `data\distortion_grid\distortion_grid.pkl`; its CSV export is retained at `data\distortion_grid\distortion_grid.csv`. The sweep has 51 absolute THI values from +5 to -5 mm, 401 RC rotations from -20° to +20° in 0.1° increments at each THI, and nine fields per state (184,059 rows total). The structured data includes `z_mm`. Plotting defaults to five rotation panels at -10°, -5°, 0°, +5°, and +10°, overlaying the exact stored z planes -5, -2.6, 0, +2.6, and +5 mm in each panel. Colors distinguish z planes; dashed and solid lines distinguish paraxial reference and real image grids. Use `--rotations` and `--z-values` to select other exact stored values. No CODE V trace is performed by the plotter. Recreate the default comparison with:

```powershell
& $distortionPython -u Script\plot_distortion_grid.py
```

The P1 rotation fit uses the actual real-coordinate grid at theta=0 as its fixed source, so its sampled baseline barrel distortion carries into each prediction. The existing P1 fit outputs were generated from the previous single-z dataset, whose lens-sequence revision had `Cornea_ENT_D` THI at -35 mm; the current user-edited sequence has a 1 mm baseline. The old fit outputs are stale relative to the current multi-z sweep; rerun only when new fit results are requested. The runner defaults to z=0 and accepts `--z-mm` to fit another slice. To recreate its report, pickle, coefficient artifact, and plots:

```powershell
& $distortionPython -u Script\analyze_p1_rotation_transform.py
```

The outputs are in `data\p1_rotation_transform\`. The fitted mapping is `X=sx*x0/(1+q*y0)`, `Y=sy*y0/(1+q*y0)`; its residual includes optical effects outside this reduced model.

## P1 CODE V collection

The target is `Lens\p1_ME.seq`. Keep the existing `SRC_ROT ADE -20` setting unless requested otherwise. Apply eye rotation through `ADE` on surface `RC` with the exact selector `S_RC = "s\"RC\""` from `raytracing.py`. The collector applies absolute THI values through `S_CORNEA_MOV = "s\"Cornea_ENT_D\""`, verifies both ADE and THI readbacks, and cleans up with `StopCodeV()` and `pythoncom.CoUninitialize()`. It checkpoints only complete z groups, validates resume against the sweep configuration and lens SHA-256, and replaces the canonical pickle only after the full sweep completes.

To recollect the default P1 sweep:

```powershell
& $distortionPython -u Script\distortion_grid.py
```

## P4 accommodation grid, radial fit, and rotation transform

### P4 accommodation distortion grid

`Script\distortion_grid_p4.py` collects the P4 model over both accommodation and RC eye rotation. It loads `Lens\p4_ME.len` with CODE V `RES`, targets zoom `z1` (P4), and leaves the second zoom (ME) out of the sweep. The default is 51 accommodation values from 0 to 5 D in 0.1 D steps, 401 RC rotations from -20° to +20° in 0.1° steps, and a 3 × 3 field grid. The completed sweep contains 184,059 rows (9 grid points for each accommodation/rotation state) in `data\distortion_grid_p4\distortion_grid.pkl`. Its metadata in `metadata.json` records the lens hash, settings, baselines, and completed groups. After each accommodation group, the collector writes the pickle and metadata through separate atomic file replacements; the pair is not replaced as one transaction. Resume validates both files against the requested sweep and lens hash before continuing.

Accommodation geometry follows `Script\raytracing.py`: the z1 `CorneaB` and `LensF` thicknesses, `LensF` and `LensB` radii and conics, and GRIN coefficients are set from the pristine lens baseline for each accommodation value. The collector checks geometry readbacks, uses the explicit RC selector `s\"RC\"` and verifies each requested ADE, then traces the same relative 3 × 3 grid. It appends `accommodation_d` to the existing structured grid fields.

The completed collection verified all 20,451 accommodation/rotation states, with nine distinct points per state and all ten numeric fields finite. It includes the center, four corners, and four edge-midpoint fields. CODE V verified the requested RC ADE readback at each rotation. The saved lens SHA-256 is recorded in the metadata alongside all 51 completed accommodation groups.

```powershell
& $distortionPython -u Script\plot_distortion_grid_p4.py --accommodation 0
```

The default plot is one row of five panels at RC -10°, -5°, 0°, +5°, and +10°. Each panel overlays accommodation 0, 2, and 4 D in distinct colors, with dashed reference grids and solid real image grids. All panels share image-coordinate bounds and equal aspect. It is saved as `data\distortion_grid_p4\distortion_grid.png`. Choose another set of D values with `--accommodations`, or one value with `--accommodation`; selected rotations and output path can be set with `--rotations` and `--output`. If a run stops after one or more accommodation groups, continue with the same sweep settings using `--resume`; the checkpoint's model hash and configuration are checked before it continues.

Fit the P4 radial coefficient across accommodation and regenerate its report and plots with:

```powershell
& $distortionPython -u Script\fit_p4_radial_distortion.py
```

Fit the P4 baseline-plus-keystone transform at 0 D across rotation with:

```powershell
& $distortionPython -u Script\analyze_p4_rotation_transform.py
```

The P1 collector needs Windows, CODE V, a working license, and `pywin32`; its default output is `data\distortion_grid\distortion_grid.pkl`. The P4 collector uses the separate `data\distortion_grid_p4\distortion_grid.pkl` output path. All script paths otherwise resolve relative to the project.
