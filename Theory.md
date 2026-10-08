# P1 and P4 barrel and keystone models

This note documents the empirical baseline distortion and rotation-dependent transform analyses for the P1 and P4 optical setups. P1 uses a fixed real-coordinate grid at zero rotation as the barrel-distorted baseline, then fits a reduced vertical-keystone transform across eye rotation. P4 separately fits radial `k1` versus accommodation at zero rotation and fits the same type of baseline-plus-keystone transform across rotation at 0 D. These analyses use distinct optical setups and their parameters are not interchangeable.
## P1 fixed-baseline barrel plus vertical keystone

This is the P1 rotation-dependent transform analysis. P1 supplies 401 eye-rotation angles from -20 to +20 degrees and nine field points per angle, with no accommodation dimension. The fixed source is the actual P1 real-coordinate grid at theta=0. This carries the sampled baseline barrel distortion into every prediction; the paraxial theta=0 grid is saved for context but is not the source of this transform. The analysis does not infer or fit P1 accommodation dependence.

The fitted rotation-dependent mapping is

\[
X=\frac{s_x x_0}{1+q y_0},\qquad
Y=\frac{s_y y_0}{1+q y_0},\qquad
H=\begin{bmatrix}s_x&0&0\\0&s_y&0\\0&q&1\end{bmatrix}.
\]

Coordinates are center-relative and the origin is centered, so translations are zero. P1 has exact left-right reflection parity and rotation-dependent top-bottom asymmetry, supporting the vertical denominator orientation. The reduced family excludes shear and horizontal perspective. It is an approximation to the rotation-dependent change beyond the fixed empirical baseline; its residuals can include other optical changes the restricted keystone family cannot represent.

The three parameters at each of the 401 angles are estimated together using the batched algebraic initialization and damped Cartesian Gauss-Newton refinement. A separate unconstrained ordinary least-squares quadratic is fitted in-sample to all 401 estimates. It uses the scaled numerical basis `[1,t,t^2]`, `t=theta/20`, with the following equivalent physical-degree form `c(theta)=c0+c1*theta+c2*theta^2`:

| Parameter | Units | c0 | c1 per degree | c2 per degree^2 | Coefficient RMSE | Coefficient max error |
|---|---|---:|---:|---:|---:|---:|
| sx | unitless | 1.00002080386 | -4.25403635929e-14 | 1.85435557047e-5 | 1.851074e-5 | 5.422594e-5 |
| sy | unitless | 1.00005054265 | 5.03892066961e-14 | 6.46083406774e-5 | 4.5239633e-5 | 0.00013342668 |
| q | mm^-1 | 2.73730848819e-13 | 0.000378982990846 | -1.09741946433e-14 | 7.5720203e-5 | 0.00019728487 |

The `theta=0` direct baseline-to-itself fit estimates `(sx, sy, q)=(1, 1, -8.23e-18 mm^-1)`; identity was not imposed. At the sweep endpoints the direct fits are approximately `(1.00738400, 1.02576045, -0.00738237 mm^-1)` at -20 degrees and `(1.00738400, 1.02576045, +0.00738237 mm^-1)` at +20 degrees. The sign and values are specific to the P1 optical setup and this coordinate convention; they should not be transferred to P4 or treated as universal.

| theta (degrees) | Direct coordinate RMSE (um) | Quadratic-composed coordinate RMSE (um) |
|---:|---:|---:|
| -20 | 5.95020 | 5.96587 |
| -15 | 4.56965 | 4.57003 |
| -10 | 3.09703 | 3.10242 |
| -5 | 1.56362 | 1.56962 |
| 0 | 0 | 0.0521408 |
| +5 | 1.56362 | 1.56962 |
| +10 | 3.09703 | 3.10242 |
| +15 | 4.56965 | 4.57003 |
| +20 | 5.95020 | 5.96587 |

Coordinate RMSE is computed over the nine sampled fields and both coordinates (18 scalar residuals). Direct coordinate RMSE ranges from 0 to 5.95020 um (mean 3.06366 um); the quadratic-composed result ranges from 0.0521408 to 5.96587 um (mean 3.06866 um). Direct maximum Euclidean point error reaches 11.1749 um at -20 degrees, field (-1,+1); the quadratic-composed maximum reaches 11.1511 um. The two-axis point RMS is respectively 0–8.41486 um (mean 4.33267 um) and 0.0737382–8.43701 um (mean 4.33974 um). All direct fits converged; initializer rank is three and condition numbers range from 2.11103 to 2.15110. The minimum sampled denominator is 0.987474.

The keystone residual includes optical changes outside its three-parameter family. The quadratic coefficient curves are an in-sample summary of direct keystone parameters, not held-out validation or a claim that the optical response is exactly quadratic. The report, data, coefficient artifact, and plots are in [`data/p1_rotation_transform`](data/p1_rotation_transform); the runner is [`Script/analyze_p1_rotation_transform.py`](Script/analyze_p1_rotation_transform.py).

## P4 optical-model analyses

The P4 results below describe two separate analyses of the saved P4 grid in `data/distortion_grid_p4/distortion_grid.pkl`. The radial accommodation fit uses zero-rotation grids at multiple accommodation values; the rotation transform uses the zero-diopter baseline and varies eye rotation. Their parameters and residuals answer different questions and should not be combined as if they were one jointly calibrated model.

### Radial distortion versus accommodation at zero rotation

At eye rotation 0 degrees, each accommodation group supplies nine field points with paraxial coordinates `(x,y)` and center-relative real coordinates `(real_x,real_y)`. The tested forward model is

\[
real_x=x(1+k_1r^2),\qquad real_y=y(1+k_1r^2),\qquad r^2=x^2+y^2.
\]

Here `k1` has units mm^-2. A single coefficient is fitted by ordinary pooled least squares across the group's 18 X/Y coordinates. Coordinate RMSE is computed over those 18 scalar residuals, including the center's two zero residuals. The sampled 3 x 3 grid has only two distinct nonzero radii (edge midpoints and corners); the center has zero radius and supplies no leverage. Because the paraxial grid scale changes with accommodation, each group is evaluated using its own paraxial coordinates.

The saved study fits accommodation dependence to 51 per-accommodation estimates from 0 to 5 D in 0.1 D steps. It compares an unconstrained offset exponential, `k1=d+a exp(b A)`, with an unconstrained quadratic, `k1=p0+p1 A+p2 A^2`. For the exponential, `(d,a,b)=(-0.0195347321373, 0.000993232205102, 0.235926474606)`, with `b` in D^-1; its coefficient-curve RMSE is `4.126091e-6 mm^-2` (0.18615% of the observed k1 range). For the quadratic, `(p0,p1,p2)=(-0.0185179003996, 0.000180197179361, 5.15785422657e-5)` with the corresponding powers of D in the coefficient units; its coefficient-curve RMSE is `6.0346857e-6 mm^-2` (0.27226%). These are in-sample fits to the 51 estimated coefficients, not independent validation of an accommodation law.

Across the 51 groups, the one-parameter radial model's coordinate RMSE ranges from `0.000534472` to `0.00110191 mm` (mean `0.000807812 mm`). The maximum absolute scalar-coordinate residual ranges from `0.00106894` to `0.00220383 mm` (mean `0.00161562 mm`). At 0 D, `k1=-0.01852833121 mm^-2`, coordinate RMSE is `0.00110191 mm`, and maximum absolute scalar-coordinate residual is `0.00220383 mm`. The RMSE and maximum are different summaries over the 18 scalar coordinate residuals; the latter is not a Euclidean point error. These residuals describe adequacy on the sampled grid only. With two nonzero radii, they do not establish behavior between sampled radii or prove that radial distortion is the only physical effect.

### Eye rotation at zero diopters: baseline plus vertical keystone

The separate P4 rotation analysis selects accommodation 0 D and matches all 401 rotations from -20 to +20 degrees to the same nine field identities. It fixes the source grid to the actual **real** coordinates at 0 D and zero rotation. This retains the baseline barrel distortion at the sampled points; it does not use each rotated grid's paraxial coordinates as the transform source.

The fitted reduced transform is

\[
X=\frac{s_x x_0}{1+q y_0},\qquad
Y=\frac{s_y y_0}{1+q y_0},\qquad
H=\begin{bmatrix}s_x&0&0\\0&s_y&0\\0&q&1\end{bmatrix}.
\]

The coordinates are center-relative, so translation is omitted. The imposed symmetry also omits shear and horizontal perspective. The shared denominator makes this a vertical-keystone model and changes horizontal width with `y0` as well as vertical position. `sx` and `sy` are dimensionless; `q` is mm^-1. Each angle's three parameters is estimated from all nine points by an algebraic least-squares initialization and damped Gauss-Newton refinement against Cartesian coordinate residuals. At zero rotation the fitted baseline-to-itself values are `sx=1`, `sy=1`, and `q=-3.45e-17 mm^-1` (estimated from the data).

The direct per-angle transforms have coordinate RMSE from 0 to `5.21325 um` (mean `1.20933 um`), with maximum Euclidean point error `10.6716 um` at +20 degrees. A separate unconstrained quadratic is fit in-sample to all 401 direct parameter estimates. Its coordinate RMSE over the same grids ranges from `0.286062` to `6.57821 um` (mean `1.848 um`), with maximum Euclidean point error up to `16.0142 um`. The quadratic coefficient fit does not enforce symmetry or an identity intercept and has no held-out-angle validation. Therefore, it is a compact empirical approximation over this sweep, not evidence that the underlying optical response is exactly quadratic.

This rotation transform is calibrated only at 0 D. Accommodation dependence of `sx`, `sy`, or `q` would require separate fitting and validation across accommodation; the radial `k1(A)` fit does not supply those transform parameters. Neither P4 analysis establishes detector measurement precision: both use simulated grid coordinates and omit detector calibration, alignment, localization noise/bias, and other physical errors. Full model definitions, residual conventions, coefficient tables, and artifact details are in `data/p4_radial_distortion_fit/p4_radial_distortion_fit.md` and `data/p4_rotation_transform/report.md`.

### Conceptual composition across accommodation and rotation (unvalidated)

A possible future model could combine an accommodation-specific paraxial map, a radial correction calibrated against the per-accommodation grids, and a rotation-dependent transform:

\[
P(\theta,A)=\pi\!\left(H(\theta,A)
\begin{bmatrix}
B_A\!\left(P_{\mathrm{para}}(0,A)\right)\\1
\end{bmatrix}\right),
\qquad
\pi\!\left(\begin{bmatrix}X_h\\Y_h\\w_h\end{bmatrix}\right)
=\begin{bmatrix}X_h/w_h\\Y_h/w_h\end{bmatrix}.
\]

Here `P_para(0,A)` denotes paraxial field coordinates at zero rotation and accommodation `A`; `B_A` would denote a baseline correction that maps those coordinates to the measured real-coordinate baseline at that accommodation; `H(theta,A)` is a homogeneous rotation-dependent transform; and `pi` performs homogeneous division. This is a proposed composition notation only. It has not been fitted or validated jointly. The existing rotation fit uses the empirical real-coordinate baseline `B_0` at zero rotation and 0 D, not a fitted radial correction, and has no calibrated accommodation dependence in `H`. The radial `k1(A)` fit is based on zero-rotation grids and does not establish `B_A` as an adequate baseline mapping when combined with rotation. A joint model would require new fitting and validation over accommodation, rotation, and field points before it could be treated as a predictive model.
