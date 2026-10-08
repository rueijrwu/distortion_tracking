# P4 composed baseline barrel and vertical keystone at 0 D

The analysis selects accommodation with `isclose(0, rtol=0, atol=1e-8)` and matches all 401 rotations to theta=0 by `(field_x_relative, field_y_relative)`.

## Composition and model

The fixed source points are the actual simulated **real** coordinates at accommodation 0 D and theta=0. Those nine coordinates and their paraxial baseline coordinates are saved in the pickle. This preserves the measured baseline barrel distortion exactly at the sampled points; each rotation's paraxial coordinates are not used as the transform source.

The only rotation-dependent mapping is reduced vertical keystone: `X=sx*x0/(1+q*y0)`, `Y=sy*y0/(1+q*y0)`, with homography `[[sx,0,0],[0,sy,0],[0,q,1]]`. Center-relative coordinates set translations to zero; symmetry excludes shear and horizontal perspective. The shared denominator changes horizontal width with y as well as vertical position. `sx` and `sy` are dimensionless; `q` is mm^-1.

A batched algebraic least-squares estimate initializes a damped Gauss-Newton refinement against Cartesian squared reprojection error. All 401 fits are processed together. The theta=0 baseline-to-itself coefficients are fitted from data, not assigned by hand; sampled denominators must remain positive.

## Coefficient dependence

The blue curve is the independently fitted coefficient at each rotation. The dashed curve is a separate unconstrained ordinary least-squares quadratic fitted to all 401 direct coefficients (all ground-truth angles, in-sample). Its internal basis is `[1,t,t²]` with `t=theta/20`; the report table converts this to `c(theta)=c0+c1*theta+c2*theta²`. No symmetry constraints or held-out angles are used, and the fit does not force the theta=0 intercept to identity.

| Parameter | Units | Constant c0 | Linear c1 (per degree) | Quadratic c2 (per degree²) | Coefficient RMSE | Coefficient max error |
|---|---|---:|---:|---:|---:|---:|
| sx | unitless | 0.999841804969 | 4.0488792874e-14 | 0.00015527589542 | 0.00013956886 | 0.00040313667 |
| sy | unitless | 1.00036383697 | 2.8380076067e-14 | 0.000302802819333 | 0.00036378715 | 0.0011886207 |
| q | mm^-1 | -3.18466570347e-12 | -0.00234562287684 | 3.62660769471e-14 | 0.0010868573 | 0.0029633969 |

The reported coefficient order is `(sx, sy, q)`. Coefficient-fit RMSE and maximum error use each parameter's base units. Units are unitless for sx/sy and mm^-1 for q; c1 and c2 carry the corresponding per-degree and per-degree-squared factors. The coefficients are stored both in the scaled basis and in physical-degree form.

## Per-angle spatial errors

Coordinate RMSE is `sqrt(mean(rx², ry²))` over nine points and both coordinates (18 scalar residuals). RMS Euclidean point error is `sqrt(mean(rx²+ry²))` over the nine points. Maximum Euclidean point error is the largest `sqrt(rx²+ry²)` among the nine points. Direct-fit columns come from each independent transform; quadratic columns evaluate the quadratic coefficient transform against the same actual grid.

| theta (degrees) | Direct coordinate RMSE (µm) | Quadratic coordinate RMSE (µm) | Direct RMS point (µm) | Quadratic RMS point (µm) | Direct max point (µm) | Quadratic max point (µm) |
|---:|---:|---:|---:|---:|---:|---:|
| -20 | 5.21325 | 6.57821 | 7.37265 | 9.303 | 10.6716 | 16.0142 |
| -15 | 1.79451 | 1.85938 | 2.53782 | 2.62956 | 3.63294 | 3.5905 |
| -10 | 0.48699 | 1.58129 | 0.688708 | 2.23628 | 1.24405 | 3.82001 |
| -5 | 0.276291 | 1.09945 | 0.390735 | 1.55486 | 0.496854 | 2.15601 |
| 0 | 0 | 0.286062 | 0 | 0.404553 | 0 | 0.490145 |
| 5 | 0.276291 | 1.09945 | 0.390735 | 1.55486 | 0.496854 | 2.15601 |
| 10 | 0.48699 | 1.58129 | 0.688708 | 2.23628 | 1.24405 | 3.82001 |
| 15 | 1.79451 | 1.85938 | 2.53782 | 2.62956 | 3.63294 | 3.5905 |
| 20 | 5.21325 | 6.57821 | 7.37265 | 9.303 | 10.6716 | 16.0142 |

At theta=0, the direct baseline-to-itself fit gives `sx=1`, `sy=1`, `q=-3.45498472109e-17 mm^-1`; these values were estimated rather than forced. The quadratic intercept is reported in the coefficient table without being constrained to these values.

## Residual definitions and fit accuracy

Coordinate RMSE is `sqrt(mean(rx², ry²))` over nine points and both coordinates (18 scalar residuals). RMS Euclidean point error is `sqrt(mean(rx²+ry²))` over the nine points. Maximum Euclidean point error is the largest `sqrt(rx²+ry²)` among the nine points.

Coordinate RMSE is 0–5.21325 µm (mean 1.20933 µm).
RMS Euclidean point error is 0–7.37265 µm (mean 1.71026 µm).
Maximum Euclidean point error is 0–10.6716 µm.
Quadratic-composed coordinate RMSE is 0.286062–6.57821 µm (mean 1.848 µm).
Quadratic-composed maximum Euclidean point error is 0.490145–16.0142 µm.
Largest point residual: 10.6716 µm at theta=20°, field (1, -1).
Batched algebraic initializer rank is 3 at every angle; condition number range 1.59901–1.76785.
All nonlinear fits converged: True; accepted damped steps range 1–8.
Minimum fitted denominator across sampled points: 0.94395166.

## Artifacts

- `p4_rotation_transform.pkl`: baseline grids, direct and quadratic coefficient series, restricted homographies, predictions, residuals, errors, and solver diagnostics.
- `coefficient_quadratics.pkl`: scaled and physical-degree coefficient fits, basis/order/units, and curve errors.
- `keystone_coefficients.png`: direct `sx`, `sy`, and `q` dependence plus quadratic coefficient curves.
- `coordinate_rmse.png`: direct and quadratic-composed coordinate RMSE over the sweep.
