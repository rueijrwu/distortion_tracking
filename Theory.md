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

The coefficients are calibrated from the ground-truth sweep first. The one-dimensional minimization can evaluate the polynomial at candidate angles and refine the best candidate continuously. It is also possible to solve the polynomial equations for `theta`, but explicitly inverting them can produce multiple roots or no real root under noise. Minimizing the measured residual over the allowed rotation range gives a direct way to handle those cases and provides a residual useful for detecting poor fits.

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

The quadratic forward model has been fit to all 401 ground-truth rotations from -20 to +20 degrees in `Script/fit_polynomial_distortion.py`. It uses the five center/corner image positions relative to their values at 0 degrees, with `numpy.linalg.lstsq` on the scaled angle `t=theta/20`; the saved coefficients are also reported in the physical `[theta^2, theta, 1]` basis. On those same 401 samples, the eight corner-coordinate residuals have RMSE 0.12461 micrometers, MAE 0.0927289 micrometers, and maximum absolute error 0.642517 micrometers at top-left Y and -20 degrees. The two-axis corner-vector RMSE is 0.176225 micrometers, with a 95th percentile of 0.303721 micrometers and maximum of 0.713884 micrometers.

These are full-sweep approximation errors: the same ground-truth angles were used to fit the coefficients and calculate the residuals. They are below the assumed 1 micrometer per-coordinate localization noise, so the quadratic represents this ideal sweep closely at that coordinate scale. The residuals do not establish eye-rotation estimation precision, and omit detector calibration, alignment, localization bias, and other physical errors. The polynomial inverse estimator has not been implemented or evaluated; the existing inverse still uses piecewise-linear template matching.

Rotation is identifiable only where the predicted five-point pattern changes sufficiently with angle and does not repeat at another angle. A polynomial can approximate a nonlinear response, but it cannot remove genuine ambiguity or create sensitivity where the corner pattern changes very little. Nonlinearity is therefore a hypothesis to test against residuals and angle-dependent precision, not an established explanation for all observed estimation error.

The forward quadratic changes no collector behavior and does not by itself change the rotation estimator.
