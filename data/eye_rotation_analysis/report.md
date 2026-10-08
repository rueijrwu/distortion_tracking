# Five-point eye-rotation analysis with a quadratic forward model

## Data and model

The analysis uses 401 CODE V ground-truth rotations from -20 to 20 degrees in 0.1-degree steps.
The measured points are ordered center, top-left, top-right, bottom-left, bottom-right. The sweep coordinates already subtract the optical center at each rotation, so the center anchors the absolute measurements while the four relative corner X/Y values carry the angle pattern.
The model comes from `C:\Users\rueijrwu\OneDrive\Project\DistortionTracking\data\polynomial_distortion_fit\polynomial_distortion_fit.pkl`. Its scaled basis is `(theta/20)^2, theta/20, 1`, with coefficient order `(power, point, x/y)`, powers `t^2, t, constant`, and points center/TL/TR/BL/BR. The artifact also stores physical coefficients in `[theta^2, theta, 1]` order. Both bases and point order were checked against the artifact predictions (maximum differences 0 mm scaled and 1.39e-17 mm physical).
Full-sweep forward coordinate residual: corner RMSE 0.12461 um, MAE 0.0927289 um, and maximum absolute residual 0.642517 um. These are forward coordinate fit errors, not rotation errors.

## Inverse

For a measured five-point set, subtract the measured center from each corner and subtract the zero-degree corner positions. Fit the resulting four-by-two shift to the quadratic forward model.
With independent noise on the five absolute point coordinates, center subtraction gives each axis corner covariance `sigma^2 (I + 11^T)`. The weighted least-squares precision matrix is `W = I - 11^T/5` for each axis.
The objective is quartic in normalized angle `t=theta/20`, so its derivative is cubic. The inverse evaluates every real stationary root within the angle bounds and both endpoints, then chooses the candidate with the smallest weighted residual. This is a global minimization of the fitted quadratic objective over the bounded interval, without interpolation between angle templates.

## Noiseless inversion

Inverting all 401 exact CODE V corner patterns gives median absolute angle bias 0.0892991 degrees, P95 0.385127 degrees, and maximum 0.432722 degrees at truth +8 degrees (estimate +8.43272 degrees). This bias comes from the quadratic approximation to the CODE V response. It is distinct from the sub-micrometer forward coordinate residual above.
Largest absolute coordinate changes from 0 degrees are dX=-0.0157504 mm at -20 degrees (top-left) and dY=0.0475521 mm at -20 degrees (top-left). Largest radial/tangential distortion changes are 0.683985 and -0.281402 percentage points. These distortion values are diagnostic only; the rotation inverse uses image coordinates.
The closest sampled ground-truth corner patterns at least 1 degree apart are +9.5 and +10.5 degrees, with center-corrected weighted RMS separation 0.000173729 mm per coordinate feature.

## Conditional 1 um coordinate-noise precision

Assume independent Gaussian noise with standard deviation 0.001 mm (1 um) on each X and Y coordinate of all five absolute points, including the center. The seeded Monte Carlo uses 101 true angles, 100 trials per angle, and exact CODE V coordinates as the noise-free means. Thus results include both coordinate noise and quadratic approximation bias.

| Statistic | Result |
|---|---:|
| Median absolute angle error | 0.275162 degrees |
| Pooled 95th-percentile absolute angle error | 1.85545 degrees |
| RMSE | 0.90666 degrees |
| Maximum absolute error | 14.3954 degrees |
| Trials above 1 degree | 13.66% |
| Trials above 5 degrees | 0.4059% |
| Endpoint clamp rate (-20 / +20 degrees) | 0.09901% / 1.059% |
| Worst per-angle P95 | 5.23959 degrees at +7.6 degrees |
| Local Fisher 1-sigma range | 0.150457 to 2.01486 degrees |
| Best / worst local Fisher angle | -20 / +10.1 degrees |

The pooled and per-angle Monte Carlo percentiles have finite-sample uncertainty. The local Fisher estimate describes small errors near the true angle and does not capture wrong-branch outcomes. The 0.1-degree ground-truth spacing is sweep sampling, not demonstrated measurement accuracy.
This simulation assumes known point identities and an exact calibration apart from the fitted quadratic approximation. It omits detector calibration, alignment, localization bias, and other systematic errors; it is not measured hardware precision.

## Reusable estimator

`Script/analyze_eye_rotation.py` exposes `estimate_rotation_polynomial(measured_absolute_xy_mm, coefficients_scaled_by_point_xy, baseline_corner_coordinates_mm, bounds_deg=(-20, 20))`. Supply a `(5, 2)` array ordered center, TL, TR, BL, BR. It returns the fitted angle in degrees and weighted RMS residual in mm.
