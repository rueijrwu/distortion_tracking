# P1 composed baseline barrel and vertical keystone

All 401 angles from -20 to +20 degrees and all nine fields were used. P1 has no accommodation dimension.

## Model and fit scope

The fixed source is the actual simulated real grid at theta=0. It preserves the P1 baseline barrel distortion at the sampled fields; baseline paraxial coordinates are retained as context and are not the transform source. The rotation-dependent mapping is `X=sx*x0/(1+q*y0)`, `Y=sy*y0/(1+q*y0)`, with homography `[[sx,0,0],[0,sy,0],[0,q,1]]`. The origin is centered, so translation is zero. P1 shows exact left-right reflection parity while top-bottom parity changes with theta, supporting the vertical-keystone orientation. This reduced model excludes shear and horizontal perspective.

The shared P4 implementation supplies the batched algebraic initializer and damped Cartesian Gauss-Newton refinement, plus the ordinary least-squares coefficient fitter. There is one batched solve across 401 angle grids and no per-angle solve loop. The coefficient quadratics use every direct fitted angle in-sample and the scaled basis `[1,t,t^2]`, `t=theta/20`; reported coefficients use physical theta degrees.

## Physical-degree coefficient fits

| Parameter | Units | c0 | c1 per degree | c2 per degree^2 | Coefficient RMSE | Maximum coefficient error |
|---|---|---:|---:|---:|---:|---:|
| sx | unitless | 1.00002080386 | -4.25403635929e-14 | 1.85435557047e-05 | 1.851074e-05 | 5.422594e-05 |
| sy | unitless | 1.00005054265 | 5.03892066961e-14 | 6.46083406774e-05 | 4.5239633e-05 | 0.00013342668 |
| q | mm^-1 | 2.73730848819e-13 | 0.000378982990846 | -1.09741946433e-14 | 7.5720203e-05 | 0.00019728487 |

## Spatial errors

Coordinate RMSE is sqrt(mean(rx^2, ry^2)) over nine points and both coordinates (18 scalar residuals). RMS Euclidean point error is sqrt(mean(rx^2+ry^2)) over nine points. Maximum point error is the largest Euclidean residual among the nine points. Thus point RMS is sqrt(2) times coordinate RMSE for this common 18-coordinate population; they are reported separately for clarity.

| theta deg | Direct coordinate RMSE um | Quadratic coordinate RMSE um | Direct RMS point um | Quadratic RMS point um | Direct max point um | Quadratic max point um |
|---:|---:|---:|---:|---:|---:|---:|
| -20 | 5.9502 | 5.96587 | 8.41486 | 8.43701 | 11.1749 | 11.1511 |
| -15 | 4.56965 | 4.57003 | 6.46246 | 6.463 | 8.57013 | 8.60167 |
| -10 | 3.09703 | 3.10242 | 4.37987 | 4.38748 | 5.80038 | 5.83497 |
| -5 | 1.56362 | 1.56962 | 2.21129 | 2.21978 | 2.92443 | 2.94484 |
| 0 | 0 | 0.0521408 | 0 | 0.0737382 | 0 | 0.089071 |
| 5 | 1.56362 | 1.56962 | 2.21129 | 2.21978 | 2.92443 | 2.94484 |
| 10 | 3.09703 | 3.10242 | 4.37987 | 4.38748 | 5.80038 | 5.83497 |
| 15 | 4.56965 | 4.57003 | 6.46246 | 6.463 | 8.57013 | 8.60167 |
| 20 | 5.9502 | 5.96587 | 8.41486 | 8.43701 | 11.1749 | 11.1511 |

At theta=0, the direct baseline-to-itself fit estimates sx=1, sy=1, q=-8.22825007097e-18 mm^-1. It was not forced to identity.

Across the sweep direct coordinate RMSE is 0–5.9502 um (mean 3.06366); RMS Euclidean point error is 0–8.41486 um (mean 4.33267); maximum point error is 0–11.1749 um. Quadratic-composed coordinate RMSE is 0.0521408–5.96587 um (mean 3.06866); RMS Euclidean point error is 0.0737382–8.43701 um (mean 4.33974); maximum point error is 0.089071–11.1511 um.

The largest direct point residual is 11.1749 um at theta=-20 degrees, relative field (-1, 1).
Initializer rank is [3]; condition number range is 2.11103–2.1511. All nonlinear fits converged: True; accepted refinement steps range 1–6.
Minimum denominator over sampled fields and angles is 0.987473801; quadratic-curve minimum is 0.987139054.

## Artifacts

- `p1_rotation_transform.pkl`: baseline grids, direct and quadratic coefficient series, homographies, predictions, residuals, metrics, and diagnostics.
- `coefficient_quadratics.pkl`: scaled and physical-degree polynomial coefficient artifact.
- `keystone_coefficients.png`: three coefficient panels with quadratic overlays.
- `coordinate_rmse.png`: direct and quadratic-composed spatial RMSE.
