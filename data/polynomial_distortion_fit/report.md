# Quadratic fit of five-point distortion shifts

## Model and data

Fit the saved ground-truth sweep at 401 rotations from -20° to 20°. The input contains 401 five-point samples; no CODE V session was started.

For each point and coordinate, the modeled shift is its center-relative real image coordinate minus the corresponding coordinate at 0°:

```text
Δ_i(θ) = [θ², θ, 1] C_i,     Δ_i = [dx_i, dy_i] in mm
D = B C, with B[n] = [θ_n², θ_n, 1] and D containing 10 X/Y columns.
```

The fit uses the numerically scaled angle t=θ/20 and `numpy.linalg.lstsq`; coefficients are also saved in physical [θ², θ, 1] units. The intercept was fitted freely and is reported below as a check on the zero-shift baseline.
The center and four corners use fields (0,0), (-1,+1), (+1,+1), (-1,-1), (+1,-1).

## Full-sweep fit errors

These residuals compare the quadratic fitted to all 401 ground-truth angles with those same ground-truth samples. They measure approximation error over this sweep and do not measure eye-rotation precision.

| Corner coordinate RMSE | MAE | Maximum absolute coordinate error | Corner vector RMSE | Vector error P95 | Maximum vector error |
|---:|---:|---:|---:|---:|---:|
| 0.000174110876 mm (0.174111 µm) | 0.000141690501 mm (0.141691 µm) | 0.000746625661 mm (0.746626 µm) | 0.000246229962 mm (0.24623 µm) | 0.000364849639 mm (0.36485 µm) | 0.000850413593 mm (0.850414 µm) |

The largest absolute coordinate residual is -0.746626 µm at bottom-right Y, θ=+20°.

## Per-coordinate errors

Values are RMSE / maximum absolute error in µm. The center is included; its ideal shifts are zero by construction.

| Point | Axis | RMSE / max absolute error (µm) |
|---|---|---:|
| center | X | 0 / 0 |
| center | Y | 5.41999e-07 / 3.30397e-06 |
| top-left | X | 0.123882 / 0.407128 |
| top-left | Y | 0.212797 / 0.746626 |
| top-right | X | 0.123882 / 0.407128 |
| top-right | Y | 0.212797 / 0.746626 |
| bottom-left | X | 0.123882 / 0.407128 |
| bottom-left | Y | 0.212797 / 0.746626 |
| bottom-right | X | 0.123882 / 0.407128 |
| bottom-right | Y | 0.212797 / 0.746626 |

## Fitted coefficients

Coefficients use the Theory.md basis `[θ², θ, 1]`. For each coordinate, `dx = a_x θ² + b_x θ + c_x` and similarly for `dy`; units are mm/degree², mm/degree, and mm.

| Point | a_x | b_x | c_x | a_y | b_y | c_y |
|---|---:|---:|---:|---:|---:|---:|
| center | 0 | 0 | 0 | 5.20612031e-14 | 1.19716996e-11 | -1.96805025e-12 |
| top-left | -3.02637086e-05 | 0.000607138103 | -3.64754122e-05 | 0.000105593305 | -0.001420507 | 8.44000675e-05 |
| top-right | 3.02637086e-05 | -0.000607138103 | 3.64754122e-05 | 0.000105593305 | -0.001420507 | 8.44000675e-05 |
| bottom-left | -3.02637086e-05 | -0.000607138103 | -3.64754122e-05 | -0.000105593305 | -0.001420507 | -8.44000715e-05 |
| bottom-right | 3.02637086e-05 | 0.000607138103 | 3.64754122e-05 | -0.000105593305 | -0.001420507 | -8.44000715e-05 |

## Interpretation

Compare these fit residuals with the assumed 1 µm per-coordinate localization noise from the existing eye-rotation study. The corner-coordinate RMSE is 0.174 µm and the maximum absolute error is 0.747 µm. These values describe how closely a quadratic represents the ideal saved sweep. They do not establish rotation-estimation precision and omit detector calibration, alignment, localization bias, and other physical errors.

## Files

- `polynomial_distortion_fit.pkl`: ground truth, fitted coefficients, predictions, residuals, and full-sweep metrics.
- `truth_vs_quadratic.png`: ground-truth shifts and quadratic predictions.
- `quadratic_residuals.png`: full-fit coordinate residuals in µm.
