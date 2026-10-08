# Five-point eye-rotation observability

## Data and method

Read 401 ground-truth rotations from -20° to 20° in 0.1° steps, using four corner fields plus the center.
The collector already subtracts the center ray at each rotation. The center is therefore an anchor for measured coordinates; its absolute translation carries no rotation information in this saved dataset. The inverse uses the four corner coordinates relative to the measured center.
The four corners are top-left, top-right, bottom-left, and bottom-right at relative fields (-1,+1), (+1,+1), (-1,-1), and (+1,-1). Their image X/Y coordinates form an eight-value fingerprint.
The fingerprint plot shows only five-point image-coordinate shifts in millimeters relative to 0°; radial and tangential distortion percentages are not shown in that plot.

## Dependence and ambiguity

Closest pair of fingerprints separated by at least 1°: RC +9.5° and +10.5°, center-corrected weighted RMS separation 0.000173728741 mm per coordinate feature. This checks sampled global ambiguity; finite measurement noise can still cause errors.
Largest absolute corner-coordinate changes from 0° are ΔX=-0.0157504 mm at RC -20° (top-left) and ΔY=0.0475521 mm at RC -20° (top-left). Largest absolute distortion changes are radial 0.683985 pp at RC +20° (bottom-left) and tangential -0.281402 pp at RC +20° (bottom-left).
Leave-one-out piecewise-linear template inversion across interior sweep samples gives median absolute error 0.000430581072° and maximum 0.00117621646°. This is interpolation/model discretization error for noiseless synthetic templates, not sensor precision.

## Conditional precision

Assume independent, isotropic Gaussian X/Y errors on all five absolute measured points, with σ = 0.001 mm (1 µm standard deviation) for each individual X and Y coordinate, including the center. After subtracting the noisy center, each coordinate's four-corner covariance is σ²(I + 11ᵀ); its inverse apart from σ² is I - 11ᵀ/5. The local Fisher estimate uses the numerical derivative of the sampled corner fingerprint.
The Monte Carlo study uses seed 20261008, 101 evenly spaced ground-truth angles, and 100 trials per angle and noise level. It simulates errors on all five points and fits the piecewise-linear sweep by weighted least squares. The empirical percentiles have Monte Carlo sampling uncertainty because this is a finite 10,100-trial sample. This assumes the optical model/templates are exact and the five field identities are known; it omits detector calibration, alignment, and other systematic errors.
All 401 sampled templates are distinct. This does not prove the underlying continuous physical mapping is injective between samples.
Across all 10,100 trials, 13.4% exceed 1° absolute error and 0.485% exceed 5°. The largest observed error is 14.5486° (truth +17.2°, estimate +2.65139°). Per-angle median errors range 0.02644–1.41132°; per-angle 95th percentiles range 0.232894–5.53341°, with the worst at truth RC +7.6°. The pooled Monte Carlo percentile is not a guaranteed error bound at every eye rotation.

| Assumed per-coordinate point noise | Fisher 1σ range over sweep | Monte Carlo median | Monte Carlo 95th percentile | Monte Carlo RMSE | Max error | Endpoint clamp rate (−20° / +20°) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 µm/coordinate | 0.159841–2.03503° | 0.258893° | 1.89326° | 0.924109° | 14.5486° | 0.416% / 0.703% |
At 1 µm per coordinate, the local Fisher 1σ estimate is best at RC -20° (0.159841°) and worst at RC +10° (2.03503°).

## Limits

These are conditional numerical results from the existing CODE V sweep, not measured hardware precision. Actual accuracy depends on point-localization noise, calibration, alignment, model error, and whether the five points can be detected without bias. The sweep spacing is 0.1°; interpolation permits a continuous estimate but does not establish 0.1° (or finer) experimental accuracy.
The stored corner positions are center-relative by construction. The center adds no angle feature after recentering, but its noisy measurement is subtracted from all four corners and therefore increases their covariance; the inverse accounts for this with W = I − 11ᵀ/5. Any rotation information in absolute center displacement is absent and was not included.

## Reusable inverse

`Script/analyze_eye_rotation.py` exposes `estimate_rotation(measured_absolute_xy_mm, angles_deg, corner_templates_mm)`. Pass a (5, 2) array in center, top-left, top-right, bottom-left, bottom-right order. It returns the fitted rotation in degrees and center-corrected weighted RMS residual in mm.
