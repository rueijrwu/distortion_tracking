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
| 0.000124609563 mm (0.12461 µm) | 9.27289124e-05 mm (0.0927289 µm) | 0.000642517114 mm (0.642517 µm) | 0.000176224535 mm (0.176225 µm) | 0.000303720503 mm (0.303721 µm) | 0.000713884038 mm (0.713884 µm) |

The largest absolute coordinate residual is 0.642517 µm at top-left Y, θ=-20°.

## Per-coordinate errors

Values are RMSE / maximum absolute error in µm. The center is included; its ideal shifts are zero by construction.

| Point | Axis | RMSE / max absolute error (µm) |
|---|---|---:|
| center | X | 0 / 0 |
| center | Y | 5.05248e-07 / 4.61312e-06 |
| top-left | X | 0.100871 / 0.31113 |
| top-left | Y | 0.201238 / 0.642517 |
| top-right | X | 0.100871 / 0.31113 |
| top-right | Y | 0.201238 / 0.642517 |
| bottom-left | X | 0.0359165 / 0.132075 |
| bottom-left | Y | 0.10074 / 0.362747 |
| bottom-right | X | 0.0359165 / 0.132075 |
| bottom-right | Y | 0.10074 / 0.362747 |

## Fitted coefficients

Coefficients use the Theory.md basis `[θ², θ, 1]`. For each coordinate, `dx = a_x θ² + b_x θ + c_x` and similarly for `dy`; units are mm/degree², mm/degree, and mm.

| Point | a_x | b_x | c_x | a_y | b_y | c_y |
|---|---:|---:|---:|---:|---:|---:|
| center | 0 | 0 | 0 | -1.2854178e-12 | 1.20020088e-11 | 1.164675e-10 |
| top-left | -1.53887837e-05 | 0.000494301751 | -2.00104407e-05 | 5.00235694e-05 | -0.0014069078 | 4.70714161e-05 |
| top-right | 1.53887837e-05 | -0.000494301751 | 2.00104407e-05 | 5.00235694e-05 | -0.0014069078 | 4.70714161e-05 |
| bottom-left | -1.6517695e-05 | 0.000165260398 | -1.88233503e-05 | -5.52459807e-05 | 0.000730808614 | -4.46763167e-05 |
| bottom-right | 1.6517695e-05 | -0.000165260398 | 1.88233503e-05 | -5.52459807e-05 | 0.000730808614 | -4.46763167e-05 |

## Interpretation

Compare these fit residuals with the assumed 1 µm per-coordinate localization noise from the existing eye-rotation study. The corner-coordinate RMSE is 0.125 µm and the maximum absolute error is 0.643 µm. These values describe how closely a quadratic represents the ideal saved sweep. They do not establish rotation-estimation precision and omit detector calibration, alignment, localization bias, and other physical errors.

## Files

- `polynomial_distortion_fit.pkl`: ground truth, fitted coefficients, predictions, residuals, and full-sweep metrics.
- `truth_vs_quadratic.png`: ground-truth shifts and quadratic predictions.
- `quadratic_residuals.png`: full-fit coordinate residuals in µm.
