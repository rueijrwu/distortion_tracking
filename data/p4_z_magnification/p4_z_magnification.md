# P4 z1 linear magnification versus Z

The collector uses `Lens/p4_ME.len`, zoom z1 (P4), the established 3 × 3 grid, accommodations 0–4 D in 1 D steps, RC rotations -10°, -5°, 0°, +5°, +10°, and 51 absolute THI values from -5 to +5 mm. `z_distance_mm` is the physical Z coordinate, and `z_thickness_mm` records the commanded absolute THI with the same sign. The study resets `Cornea_ENT_D` to each requested value; a fresh loaded LEN readback is recorded in `metadata.json` (the active z1 baseline readback is 0 mm).

For each accommodation and rotation, the measured real-coordinate grid at Z=0 is the fixed source. At each Z, `m = dot(R0, RZ) / dot(R0, R0)` over both coordinates of all nine field points; this least-squares scalar is defined even for the center point. The linear model is `m = 1 + alpha*Z`, with intercept fixed at one, fit to the theta=0 ratios. The same slope curve is compared with all five actual rotation grids.

Collected 1,275 states and 11,475 finite records. Maximum change in m relative to theta=0 across rotations: 0.00294912%. Direct measured-factor coordinate RMSE/max absolute coordinate residual are listed per accommodation below. Pooling Euclidean errors over every accommodation/rotation/Z/field gives direct-factor point RMSE/max 1.99071e-05/8.54393e-05 mm. Linear theta=0 factor prediction applied at all rotations has pooled coordinate RMSE/max absolute coordinate residual ranges 0.000100008–0.000105619 / 0.000375892–0.000407034 mm across accommodation. The per-accommodation slope span is 0.0818504% of its mean.

| Accommodation (D) | alpha (mm^-1) | ratio-angle max variation (%) | Direct factor RMSE (mm) | Direct factor max abs (mm) | Linear RMSE (mm) | Linear max abs (mm) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00295232758 | 0.00171777 | 1.20477e-05 | 6.5299e-05 | 0.000105619 | 0.000375892 |
| 1 | 0.00295172708 | 0.00200342 | 1.30237e-05 | 7.03616e-05 | 0.00010416 | 0.000383346 |
| 2 | 0.00295112891 | 0.00230384 | 1.40079e-05 | 7.5426e-05 | 0.000102744 | 0.000391071 |
| 3 | 0.00295052435 | 0.0026187 | 1.499e-05 | 8.04395e-05 | 0.00010136 | 0.000398961 |
| 4 | 0.00294991207 | 0.00294912 | 1.59703e-05 | 8.54e-05 | 0.000100008 | 0.000407034 |

## Measured endpoints and worst single-grid errors

Endpoint m values are measured theta=0 ratios. Worst-state coordinate RMSE is calculated over nine points and two coordinates in one theta/Z grid, then maximized over all rotations and Z planes. Euclidean point error is the norm across x/y for each grid point.

| Accommodation (D) | Measured m(-5 mm) | Measured m(+5 mm) | Linear worst-state RMSE (mm) | Linear max Euclidean point (mm) | Direct max Euclidean point (mm) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.985455057 | 1.01498075 | 0.000248891 | 0.000485064 | 6.53017e-05 |
| 1 | 0.985457971 | 1.01497766 | 0.000248726 | 0.00049123 | 7.03711e-05 |
| 2 | 0.985460874 | 1.01497458 | 0.000248706 | 0.000497714 | 7.54449e-05 |
| 3 | 0.985463808 | 1.01497147 | 0.000248787 | 0.0005044 | 8.04686e-05 |
| 4 | 0.98546678 | 1.01496832 | 0.000248976 | 0.00051131 | 8.54393e-05 |

## Linear factors at selected physical Z

| Accommodation (D) | Z (mm) | m = 1 + alpha Z |
|---:|---:|---:|
| 0 | -5 | 0.985238362 |
| 0 | -2.6 | 0.992323948 |
| 0 | +0 | 1 |
| 0 | +2.6 | 1.00767605 |
| 0 | +5 | 1.01476164 |
| 1 | -5 | 0.985241365 |
| 1 | -2.6 | 0.99232551 |
| 1 | +0 | 1 |
| 1 | +2.6 | 1.00767449 |
| 1 | +5 | 1.01475864 |
| 2 | -5 | 0.985244355 |
| 2 | -2.6 | 0.992327065 |
| 2 | +0 | 1 |
| 2 | +2.6 | 1.00767294 |
| 2 | +5 | 1.01475564 |
| 3 | -5 | 0.985247378 |
| 3 | -2.6 | 0.992328637 |
| 3 | +0 | 1 |
| 3 | +2.6 | 1.00767136 |
| 3 | +5 | 1.01475262 |
| 4 | -5 | 0.98525044 |
| 4 | -2.6 | 0.992330229 |
| 4 | +0 | 1 |
| 4 | +2.6 | 1.00766977 |
| 4 | +5 | 1.01474956 |

Center point is included in the grid residual but contributes zero to the scalar magnification fit because its baseline coordinate is zero. The directly fitted factors and the constrained linear approximation have separate residual columns above.
