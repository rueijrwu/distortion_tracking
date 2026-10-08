# Distortion model summary

The repository contains distinct P1 and P4 studies. P1 develops a five-point eye-rotation estimator from center-relative coordinate shifts. P4 analyzes a different optical setup with two separate models: radial distortion versus accommodation at zero eye rotation, and a baseline-plus-keystone transform versus eye rotation at zero diopters. Their coefficients are not interchangeable.

For P4, the radial model is `real = paraxial * (1 + k1*r^2)`, fitted independently to nine-point grids at each of 51 accommodation values from 0 to 5 D. At 0 D, `k1=-0.01852833121 mm^-2` and coordinate RMSE is `0.00110191 mm`. Across all groups, coordinate RMSE ranges from `0.000534472` to `0.00110191 mm` (mean `0.000807812 mm`). The maximum absolute scalar-coordinate residual ranges from `0.00106894` to `0.00220383 mm` (mean `0.00161562 mm`); at 0 D it is `0.00220383 mm`. This maximum is distinct from RMSE and from a Euclidean point error. The accommodation curve fits are in-sample summaries of these coefficients; the sampled grid has just two distinct nonzero radii.

The fitted accommodation curves are `k1(A)=d+a*exp(b*A)` with `(d,a,b)=(-0.0195347321373, 0.000993232205102, 0.235926474606)` (`d,a` in mm^-2; `b` in D^-1; coefficient RMSE `4.126091e-6 mm^-2`), and `k1(A)=p0+p1*A+p2*A^2` with `(p0,p1,p2)=(-0.0185179003996, 0.000180197179361, 5.15785422657e-5)` (`p0` in mm^-2, `p1` in mm^-2/D, `p2` in mm^-2/D^2; coefficient RMSE `6.0346857e-6 mm^-2`). These are in-sample fits to the 51 per-accommodation k1 estimates, not independently validated laws.

At 0 D, rotation is modeled from the actual real-coordinate grid at zero rotation with `X=sx*x0/(1+q*y0)` and `Y=sy*y0/(1+q*y0)`. Direct three-parameter fits across 401 angles have coordinate RMSE from 0 to `0.00521325 mm` (mean `0.00120933 mm`) and maximum Euclidean point error up to `0.0106716 mm`. A separate quadratic fit to the parameters gives coordinate RMSE from `0.000286062` to `0.00657821 mm` (mean `0.001848 mm`) on those same angles, with maximum Euclidean point error up to `0.0160142 mm`. The quadratic is in-sample and has no held-out validation. Modeling accommodation effects in this transform requires a separate fit and validation.

The fitted rotation parameter curves use the physical-degree basis `c(theta)=c0+c1*theta+c2*theta^2`, fitted in-sample to all 401 direct estimates at 0 D:

The coefficient RMSE and maximum error columns are in each parameter's base units; the `c1` and `c2` columns carry the corresponding per-degree and per-degree-squared units.

| Parameter | Units | c0 | c1 per degree | c2 per degree^2 | Coefficient RMSE | Coefficient max error |
|---|---|---:|---:|---:|---:|---:|
| sx | unitless | 0.999841804969 | 4.0488792874e-14 | 0.00015527589542 | 0.00013956886 | 0.00040313667 |
| sy | unitless | 1.00036383697 | 2.8380076067e-14 | 0.000302802819333 | 0.00036378715 | 0.0011886207 |
| q | mm^-1 | -3.18466570347e-12 | -0.00234562287684 | 3.62660769471e-14 | 0.0010868573 | 0.0029633969 |

The spatial errors below compare independent direct transforms with the quadratic-composed transform on the same sampled grid. Coordinate RMSE uses 18 scalar coordinate residuals; point errors are Euclidean distances over nine points. Units are um. The quadratic column is an in-sample approximation, not a held-out result.

| theta (degrees) | Direct coordinate RMSE | Quadratic coordinate RMSE | Direct RMS point error | Quadratic RMS point error | Direct max point error | Quadratic max point error |
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

These results describe approximation error on simulated sampled grids, not hardware rotation precision. The P4 radial and rotation reports contain definitions, coefficients, residual conventions, and limitations: `data/p4_radial_distortion_fit/p4_radial_distortion_fit.md` and `data/p4_rotation_transform/report.md`. The existing P1 theory and inverse-estimator results remain documented in [Theory.md](Theory.md).
