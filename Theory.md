# Polynomial model for eye-rotation estimation

## Purpose

The goal is to estimate eye rotation, `theta`, from five measured image points: the center and the four corner points (top-left, top-right, bottom-left, bottom-right). The saved CODE V sweep supplies ground-truth rotations and predicted point positions for calibration. The first question is whether the measured point pattern changes with rotation in a way that is smooth and distinctive enough to estimate `theta` precisely.

This document describes a forward polynomial model: rotation predicts the point shifts. The measured points are then fit to that forward model to estimate rotation. A polynomial fit is a model to investigate; it does not by itself show that the physical response is quadratic or that the inverse estimate will be precise.

## What is meant by distortion shift

In the existing sweep, each stored `real_x_mm`, `real_y_mm` position is already relative to the traced center ray at that same rotation. For field point `i`, let

\[
q_i(\theta)=\begin{bmatrix}x_i(\theta)&y_i(\theta)\end{bmatrix}
\]

be that center-relative image position, in millimeters. Define its change from the zero-rotation pattern as

\[
\Delta_i(\theta)=q_i(\theta)-q_i(0)
=\begin{bmatrix}dx_i(\theta)&dy_i(\theta)\end{bmatrix}.
\]

These `dx` and `dy` values are image-coordinate shifts in mm. They are not the radial and tangential distortion percentages in the pickle. Those percentages can be analyzed separately, but they are different measurements with different units. In the current center-relative data, the center point has `q_center(theta) = (0, 0)` at every angle, so its shift is identically zero. The four corners supply the eight changing coordinate values; the center is still needed as an anchor when the five points are measured in absolute image coordinates.

## Quadratic forward model

An order-2 polynomial means a polynomial of degree two. For each point `i`, model its X and Y shifts as

\[
dx_i(\theta)=a_{ix}\theta^2+b_{ix}\theta+c_{ix},\qquad
dy_i(\theta)=a_{iy}\theta^2+b_{iy}\theta+c_{iy}.
\]

In vector form, using a row of polynomial basis terms,

\[
\Delta_i(\theta)=
\underbrace{\begin{bmatrix}\theta^2&\theta&1\end{bmatrix}}_{\phi(\theta)}
\underbrace{\begin{bmatrix}
a_{ix}&a_{iy}\\
b_{ix}&b_{iy}\\
c_{ix}&c_{iy}
\end{bmatrix}}_{C_i}.
\]

Here `phi(theta)` has shape `1 x 3`, `C_i` has shape `3 x 2`, and `Delta_i` has shape `1 x 2`. The coefficient matrix `C_i` contains the fitted polynomial coefficients for point `i`.

For all five points, concatenate the X and Y shifts into one row, for example `[dx_center, dy_center, dx_TL, dy_TL, ..., dx_BR, dy_BR]`. For the 401 ground-truth angles `theta_n` from -20 to +20 degrees, construct

\[
\mathbf{B}=\begin{bmatrix}
\theta_1^2&\theta_1&1\\
\theta_2^2&\theta_2&1\\
\vdots&\vdots&\vdots\\
\theta_N^2&\theta_N&1
\end{bmatrix}\quad(N\times3),\qquad
\mathbf{D}=\begin{bmatrix}
\Delta(\theta_1)\\
\Delta(\theta_2)\\
\vdots\\
\Delta(\theta_N)
\end{bmatrix}\quad(N\times10).
\]

Then

\[
\mathbf{D}=\mathbf{B}\mathbf{C}+\mathbf{E},\qquad
\widehat{\mathbf{C}}=\operatorname*{argmin}_{\mathbf{C}}
\lVert\mathbf{B}\mathbf{C}-\mathbf{D}\rVert_F^2.
\]

`B` is the design matrix of basis values; `C` is the coefficient matrix. This makes the dimensions explicit and avoids using the same symbol for both. Solve the least-squares problem with a stable QR or SVD based solver (for example, `lstsq`), rather than explicitly forming and inverting `B.T @ B`. Because the center shift is zero by definition in this dataset, its two fitted coefficient columns should be zero apart from numerical roundoff and do not add information.

The sweep angle is measured in degrees, so the coefficients have corresponding units: the quadratic coefficient is mm/degree², the linear coefficient is mm/degree, and the constant is mm. Scaling the angle variable (for example, using `t = theta / 20`) can improve numerical conditioning; coefficients then refer to `t` and predictions use the same scaling.

## Baseline constraint and the cubic notation

Since the modeled quantity is defined as a change from `theta = 0`, it obeys `Delta_i(0) = 0`. A quadratic model can enforce that baseline exactly by omitting the constant term:

\[
\Delta_i(\theta)=\begin{bmatrix}\theta^2&\theta\end{bmatrix}
\begin{bmatrix}a_{ix}&a_{iy}\\b_{ix}&b_{iy}\end{bmatrix}.
\]

Alternatively, fit the constant term and check that it is near zero; a material nonzero intercept can signal numerical issues or inconsistent baseline matching. If modeling the absolute position `q_i(theta)` instead of a shift, keep the constant because `q_i(0)` is generally not zero for a corner.

The basis `[theta^3, theta^2, theta, 1]` is a **cubic** (degree-3) model, not order 2. It is a possible later comparison if the quadratic leaves systematic residual structure. The cubic model has four terms and correspondingly more parameters. For a shift relative to zero, it can enforce zero at the baseline by omitting the constant: `[theta^3, theta^2, theta]`. The polynomial basis vector is the design matrix; the fitted multipliers are the coefficient matrix.

## Estimating rotation from five measured points

Suppose the detector reports absolute measured coordinates `m_center` and `m_i` for the four corners. First remove common translation using the measured center, then subtract the zero-angle corner template:

\[
y_i=(m_i-m_{center})-q_i(0).
\]

The forward model predicts `Delta_i(theta)`, so estimate rotation by finding the angle whose predicted eight corner coordinates best match `y`:

\[
\widehat{\theta}=\operatorname*{argmin}_{-20^\circ\leq\theta\leq20^\circ}
\operatorname{tr}\!\left(E(\theta)^T W E(\theta)\right),\qquad
E(\theta)=Y-\widehat{\Delta}(\theta),
\]

where `Y` and `Delta(theta)` are `4 x 2` arrays (four corners by X/Y), and `W` weights the four corners separately for each axis.

The coefficients are calibrated from the ground-truth sweep first. The saved fit uses the scaled monomial basis `[t^2, t, 1]`, where `t=theta/20`; its physical coefficient version uses `[theta^2, theta, 1]`. Given measured points, the weighted squared residual is a quartic polynomial in `t`. Its derivative is cubic, so the bounded global minimum over -20 to +20 degrees is found by evaluating every real derivative root inside the interval and both endpoints, then choosing the candidate with the lowest residual. This handles multiple local minima directly and provides a residual useful for detecting poor fits.

If each absolute point coordinate has independent Gaussian noise with standard deviation `sigma`, subtracting the noisy center makes the four relative corner errors correlated. For either image axis their covariance is

\[
\Sigma=\sigma^2(I_4+\mathbf{1}\mathbf{1}^T),
\]

so the weight matrix, up to the common factor `1/sigma^2`, is

\[
W=I_4-\frac{1}{5}\mathbf{1}\mathbf{1}^T.
\]

This is the weighting used by the existing five-point analysis. It matters when reporting precision under the assumed 1 micrometer noise per measured X/Y coordinate. Actual precision also depends on calibration errors, alignment, and point-localization bias.

## Local sensitivity and precision

For the quadratic model, the derivative for point `i` is

\[
\frac{d\Delta_i}{d\theta}=\begin{bmatrix}2a_{ix}\theta+b_{ix}&2a_{iy}\theta+b_{iy}\end{bmatrix}.
\]

Stack the four corner derivatives into `g_x(theta)` and `g_y(theta)`, each a four-element vector. With independent absolute-coordinate noise `sigma`, the local one-standard-deviation angle uncertainty predicted by the model is approximately

\[
\sigma_\theta(\theta)\approx
\frac{\sigma}{\sqrt{g_x(\theta)^T W g_x(\theta)+g_y(\theta)^T W g_y(\theta)}}.
\]

Thus, a quadratic turning point can have weak local sensitivity when the weighted derivative becomes small, even though the shift curve itself is visibly nonlinear. If two separated angles produce similar five-point patterns, the inverse can also be ambiguous. A polynomial fit cannot restore information absent from the measured points.

## What the fit can establish

The quadratic forward model has been fit to all 401 ground-truth rotations from -20 to +20 degrees in `Script/fit_polynomial_distortion.py`. The current optical setup supplies a 3 x 3 field grid; the fit uses its five center/corner image positions relative to their values at 0 degrees, with `numpy.linalg.lstsq` on the scaled angle `t=theta/20`. The saved coefficients are also reported in the physical `[theta^2, theta, 1]` basis. On those same 401 samples, the eight corner-coordinate residuals have RMSE 0.174111 micrometers, MAE 0.141691 micrometers, and maximum absolute error 0.746626 micrometers at bottom-right Y and +20 degrees. The two-axis corner-vector RMSE is 0.246230 micrometers, with a 95th percentile of 0.364850 micrometers and maximum of 0.850414 micrometers.

These are full-sweep approximation errors: the same ground-truth angles were used to fit the coefficients and calculate the residuals. They are below the assumed 1 micrometer per-coordinate localization noise, so the quadratic represents this ideal sweep closely at that coordinate scale. The residuals do not establish eye-rotation estimation precision, and omit detector calibration, alignment, localization bias, and other physical errors.

The quadratic inverse is implemented in `Script/analyze_eye_rotation.py`. It first checks the scaled coefficient basis, point and coordinate ordering, stored predictions, and physical-basis conversion from `data/polynomial_distortion_fit/polynomial_distortion_fit.pkl`. On noiseless CODE V coordinates from the current setup, the inverse has median, P95, and maximum absolute angle bias of 2.22169, 3.22499, and 4.17571 arcminutes; the maximum occurs at +20 degrees. Angle errors below are reported in arcminutes, using 1 degree = 60 arcminutes; true and estimated rotations remain in degrees. This is angle bias from the quadratic approximation, distinct from the forward coordinate-fit residual in micrometers.

For independent Gaussian noise with 1 micrometer standard deviation on each X and Y coordinate of all five measured points, the seeded 10,100-trial simulation reports median absolute angle error 8.79205 arcminutes, pooled P95 36.6595 arcminutes, and RMSE 17.2944 arcminutes. Errors vary by angle: the worst per-angle P95 is 68.0692 arcminutes at +0.4 degrees; 0.7822% of trials exceed 60 arcminutes (1 degree), and none exceed 300 arcminutes (5 degrees). The largest observed error is 111.015 arcminutes (truth +0.8 degrees, estimate +2.65025 degrees). These are conditional simulation results, not measured hardware accuracy. The study assumes known point identities and does not include detector calibration, alignment, or other systematic errors.

Rotation is identifiable only where the predicted five-point pattern changes sufficiently with angle and does not repeat at another angle. A polynomial can approximate a nonlinear response, but it cannot remove genuine ambiguity or create sensitivity where the corner pattern changes very little. Nonlinearity is therefore a hypothesis to test against residuals and angle-dependent precision, not an established explanation for all observed estimation error.

The forward quadratic changes no collector behavior. The inverse estimate is continuous in angle, but the 0.1-degree spacing of the training sweep is not a guarantee of 0.1-degree measurement precision.

## P4 optical-model analyses

The P4 results below describe two separate analyses of the saved P4 grid in `data/distortion_grid_p4/distortion_grid.pkl`. They do not replace the P1 five-point inverse model above. The radial accommodation fit uses zero-rotation grids at multiple accommodation values; the rotation transform uses the zero-diopter baseline and varies eye rotation. Their parameters and residuals answer different questions and should not be combined as if they were one jointly calibrated model.

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
