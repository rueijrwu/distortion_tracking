# P1 and P4 distortion transformations

## Purpose and modeling principle

Describe the center-relative image pattern using the fewest dependencies needed at the measurement scale. Rotation is `theta`; P4 also varies with accommodation `A` and the separately studied axial coordinate `Z`.

**A parameter may be held constant when removing its dependence introduces an acceptable image-coordinate error over the intended operating domain.** Approximately **1 micrometer (0.001 mm)** is the project measurement-noise reference. Small coefficient variation alone is not sufficient: its effect must be evaluated after the complete spatial transformation.

The working model treats P1 as accommodation-independent and P4 as accommodation- and Z-dependent where measured. P1 may still require rotation-dependent deformation. The available P1 dataset has no accommodation dimension, so accommodation independence is a modeling premise, not a result of a P1 accommodation sweep.

The existing rotation calibration domain is one-axis RC rotation from -20 to +20 degrees. P1 also has a zero-rotation sweep across absolute THI Z from -5 to +5 mm. The original P4 accommodation/rotation grid covers accommodation from 0 to 5 D and has no Z sweep. A separate P4 z1 study measures absolute THI Z from -5 to +5 mm at five accommodations (0, 1, 2, 3, 4 D) and five rotations (-10, -5, 0, +5, +10 degrees). Each state has nine identified fields in a 3 x 3 grid. The P1 and P4 coefficients are specific to their respective optical setups; their values and signs are not interchangeable.

This document distinguishes existing fitted results from the proposed reduced joint model. Reported results refer to repository revision `f6732f2d523bd3ee13a216de88ad4c1cb174a2a3`. Rewriting this specification does not change the collectors, fitted artifacts, or analysis implementations.

## Coordinates and baseline convention

Let `i` identify a fixed field point and let

$$
\mathbf p_{1,i}(\theta,Z)
=\begin{bmatrix}X_{1,i}(\theta,Z)\\Y_{1,i}(\theta,Z)\end{bmatrix},
\qquad
\mathbf p_{4,i}(\theta,A)
=\begin{bmatrix}X_{4,i}(\theta,A)\\Y_{4,i}(\theta,A)\end{bmatrix},
\qquad
\mathbf p_{4Z,i}(\theta,A,Z)
=\begin{bmatrix}X_{4Z,i}(\theta,A,Z)\\Y_{4Z,i}(\theta,A,Z)\end{bmatrix},
$$

be the setup-specific real image coordinates in millimeters, relative to the center ray at the same state. P1 depends on rotation `theta` and its expanded sweep's absolute THI coordinate `Z`; the original P4 accommodation/rotation study depends on `theta` and `A`; the separate P4 Z study also varies `Z` and uses its own recorded lens revision.

Define the empirical zero-rotation baselines at the selected P1 Z plane and original P4 accommodation as

$$
\mathbf b_{1,i}(Z)=\mathbf p_{1,i}(0,Z),
\qquad
\mathbf b_{4,i}(A)=\mathbf p_{4,i}(0,A).
$$

For the separate P4 Z study, the matched reference is instead

$$
\mathbf b_{4Z,i}(\theta,A)=\mathbf p_{4Z,i}(\theta,A,0),
$$

using the same accommodation and rotation as the requested nonzero-Z plane.

These baselines already contain the sampled barrel distortion. In particular, the new P4 Z reference is the actual same-accommodation, same-rotation real grid at `Z=0`; it is not the theta=0 grid reused for every angle. Applying another radial correction to an empirical real-coordinate baseline would double-count that correction. A fitted radial model is an alternative way to represent the baseline from paraxial coordinates, not an extra correction to the same real baseline.

The rotation source must be the zero-rotation baseline with matching field identities, not the paraxial grid at the rotation being predicted. All quantities here describe relative pattern deformation; absolute center displacement would require a separate translation model.

The existing intended decomposition is

$$
\boxed{
\widehat{\mathbf p}_{1,i}(\theta,Z)
\approx m(Z)K_1(\theta;\mathbf b_{1,i}(0)),
\qquad
\widehat{\mathbf p}_{4,i}(\theta,A)
=K_4(\theta,A;\widehat{\mathbf b}_{4,i}(A)).
}
$$

For P1, use the measured Z=0 baseline and model the zero-rotation Z dependence with the normalized scale `m(Z)=1+alpha_Z Z`, so `b_1(Z)≈m(Z)b_1(0)`. The rotation fit itself is currently available only at Z=0. P4 has an accommodation-dependent baseline. Start with no accommodation dependence in the rotation coefficients, and add it only when coordinate errors require it. The separate P4 Z measurements support an additional, weakly accommodation-dependent scale over their tested domain; details and provenance follow below.

## Reduced rotation mapping

For a baseline point `b = [x_b, y_b]^T`, use the existing vertical-keystone family:

$$
K_j(\theta,A;\mathbf b)=
\begin{bmatrix}
\dfrac{s_{x,j}(\theta,A)x_b}{1+q_j(\theta,A)y_b}\\[2mm]
\dfrac{s_{y,j}(\theta,A)y_b}{1+q_j(\theta,A)y_b}
\end{bmatrix},
\qquad
H_j=\begin{bmatrix}s_{x,j}&0&0\\0&s_{y,j}&0\\0&q_j&1\end{bmatrix}.
$$

The shared denominator changes horizontal width with vertical position as well as vertical position itself. `s_x` and `s_y` are dimensionless; `q` has units mm^-1. Center-relative coordinates remove translation, while the nominal reflection symmetry motivates omitting shear and horizontal perspective. This is a restricted spatial family, not a general model of every optical distortion.

### Minimal coefficient functions

The reported coefficient curves support approximately even scale changes and an odd keystone change with rotation. The first reduced candidate is therefore

$$
\boxed{
 s_{x,j}(\theta)=1+\alpha_j\theta^2,
 \qquad s_{y,j}(\theta)=1+\beta_j\theta^2,
 \qquad q_j(\theta)=\gamma_j\theta.
}
$$

This has three rotation coefficients per reflection instead of three unconstrained quadratic coefficient curves with nine coefficients. It enforces `H_j(0)=I`. For the joint version, preserve `H_4(0,A)=I` so baseline changes are assigned to the baseline rather than absorbed into the rotation scales.

With `theta` in degrees, `alpha` and `beta` have units degree^-2 and `gamma` has units mm^-1 degree^-1. Numerical fitting may use `t=theta/20`, provided the saved basis and conversion are explicit. Existing transform artifacts use `[1,t,t^2]`.

Require positive scales and a denominator bounded away from zero over the accepted domain:

$$
1+q_j(\theta,A)b_{j,i,y}(A)\geq\delta>0.
$$

Check this condition on the field region being modeled, not only at the center. Symmetry and identity are proposed constraints for the reduced fit; the existing coefficient fits were unconstrained.

## P1 fixed-baseline barrel plus vertical keystone

P1 uses its actual real-coordinate grid at zero rotation as its empirical baseline. At the selected Z plane, its rotation calibration contains 401 rotations and nine fields per rotation. The existing analysis independently estimates three keystone parameters at each angle using algebraic initialization and damped Cartesian Gauss-Newton refinement, then fits unconstrained quadratics to the resulting parameter curves. See the [P1 transform report](data/p1_rotation_transform/report.md).

**Dataset status and provenance:** The canonical P1 grid has 51 absolute `Cornea_ENT_D` THI planes from -5 to +5 mm. The P1 rotation report remains the earlier single-slice analysis, associated with a prior sequence revision and a recorded THI baseline of -35 mm; its source hash does not establish identity with the current sequence. A numerical audit found that its entire 3,609-row slice (401 rotations by nine fields) matches the current sweep's Z=0 slice to floating-point roundoff. Therefore its reported rotation fit remains numerically applicable at Z=0. It does not establish rotation behavior at the other 50 Z planes. The current sequence uses a 1 mm baseline; absolute THI Z here is the coordinate recorded by the sweep and is not relabeled as displacement from that baseline.

### Z-dependent zero-rotation barrel baseline and magnification

The zero-rotation sweep isolates the baseline change over the 51 absolute THI Z planes. Let `R_i(Z)` be the measured real-coordinate vector for field `i` and let `R_i(0)` be the actual measured real grid at Z=0. Fit one isotropic scale at each Z by pooled least squares over the nine matched fields:

$$
m(Z)=\frac{\sum_i \mathbf R_i(0)\cdot\mathbf R_i(Z)}
           {\sum_i \|\mathbf R_i(0)\|^2},\qquad
\mathbf R_i(Z)\approx m(Z)\mathbf R_i(0).
$$

The center is harmless in the pooled fit because its center-relative reference vector is zero. For pointwise diagnostics, use the signed projected ratio `R_i(0)·R_i(Z)/||R_i(0)||^2` on the eight noncentral points; do not divide individual X or Y components, which can be zero. The parallel paraxial scale `m_p(Z)` is calculated the same way from the matched paraxial grids. Both ratios are dimensionless; Z is in millimeters.

The working P1 simplification is a linear, zero-normalized scale:

$$
m(Z)=1+\alpha_Z Z,\qquad
\alpha_Z=0.00295165270232\ \mathrm{mm}^{-1}.
$$

Here `Z` is the absolute THI `Cornea_ENT_D` coordinate in millimeters; the 1 mm sequence baseline is not subtracted. The coefficient is the least-squares slope of the 51 saved real-grid magnification ratios with the intercept fixed at one. This enforces `m(0)=1`, consistent with the reference grid. Its ratio RMSE is **1.01275e-4**, and its maximum absolute ratio error is **2.19022e-4** over the 51 planes. The endpoint and selected-plane real-grid ratios are:

| Absolute THI Z (mm) | -5 | -2.6 | 0 | +2.6 | +5 |
|---|---:|---:|---:|---:|---:|
| Real-grid `m(Z)` | 0.9854583393 | 0.9923851807 | 1 | 1.007732594 | 1.014977286 |
| Paraxial `m_p(Z)` | 0.9854551017 | 0.9923834777 | 1 | 1.007734340 | 1.014980683 |

For comparison, the unconstrained linear least-squares fit is `m(Z)=1.00007550219+0.00295165270232 Z`, with ratio RMSE `6.74981e-5` and maximum error `1.43520e-4`. A quadratic fit constrained near identity, `m(Z)=0.999999995615+0.00295165270232 Z+8.71229730269e-6 Z^2`, reduces those errors to `5.14398e-7` and `1.21943e-6`. The simpler `1+alpha_Z Z` relation is the working model; retain the quadratic as an empirical higher-accuracy description when its extra term matters.

The errors above for the unconstrained direct per-plane scale fit are diagnostics of how close each sampled plane is to an isotropic scale; they are not the error of the simplified linear model. Under the working `1+alpha_Z Z` model at theta=0, the largest coordinate RMSE across nine points and both coordinates is **0.29549 micrometers**, and the largest Euclidean point error is **0.50326 micrometers**. For the direct pooled per-plane scale fit, the corresponding largest errors are **0.00120667 micrometers** and **0.00206133 micrometers**. Dividing each real grid by its directly fitted per-plane scale and comparing it with the Z=0 grid gives a largest coordinate RMSE of **0.00118886 micrometers**. These separate measures show that the sampled zero-rotation change is well summarized by common magnification, while the constrained linear scale still has a small, measurable curvature residual.

The magnification factor was also checked at five rotations (`theta=-10,-5,0,+5,+10` degrees), each over all 51 Z planes. Relative to the `theta=0` real-grid scale curve, the largest absolute factor difference was **1.94155e-6**, equivalent to a maximum relative variation `|m(theta,Z)/m(0,Z)-1|` of **0.00019129%**. Applying the exact `theta=0` scale curve to those other-angle grids left at most **0.007220 micrometers coordinate RMSE** and **0.017804 micrometers maximum point error**. This supports theta-independence over these five tested angles only; the other 396 rotation values in the full sweep have not been checked for Z-independence.

The resulting proposed composition is `p_1(theta,Z)≈m(Z) K_1(theta;b_1(0))`: apply the existing keystone map to the Z=0 empirical real baseline, then multiply its output coordinates by `m(Z)`. In homogeneous coordinates the scale is `S(m)=diag(m,m,1)` and the composition is `S(m)H_1(theta)`, in that order. It does not assert that scale commutes with keystone. The keystone fit remains the existing Z=0 fit. At theta=0, the linear scale alone has the residual stated above; using the exact theta=0 scale curve at each tested nonzero angle leaves at most **0.007220 micrometers coordinate RMSE** and **0.017804 micrometers maximum point error**. This composition is a simplified model supported by the zero-rotation sweep and five-angle check, not a full 401-angle joint calibration.

The separate per-Z radial fits use `real = paraxial * (1 + k1*r^2)` with paraxial radius in millimeters and `k1` in mm^-2. The direct quadratic summary is `k1(Z)=-0.0130398764628 + 7.68681341348e-5 Z - 1.13154451225e-7 Z^2` (coefficient units mm^-2, mm^-3, mm^-4). Since paraxial image scale also changes with Z, `k1` alone mixes scale and barrel strength. The sampled dimensionless `k1*r_edge^2` varies by only 0.00898% of its absolute mean; the edge and corner radial percentages vary by less than 0.001 percentage point. As a cross-check, `k1(Z)*m_p(Z)^2` spans 1.17e-6. The per-grid radial-model coordinate RMSE is **2.24–2.31 micrometers**, so it is a distinct and larger fit error than the roughly 0.0012-micrometer residual from the inter-Z common-scale comparison.

See the [expanded P1 radial and magnification report](data/p1_radial_distortion_fit/p1_radial_distortion_fit.md), [magnification plot](data/p1_radial_distortion_fit/p1_z_magnification.png), [saved magnification data](data/p1_radial_distortion_fit/p1_z_magnification.pkl), and [analysis script](Script/fit_p1_radial_distortion.py). The five-angle check used the existing [P1 sweep pickle](data/distortion_grid/distortion_grid.pkl); no new artifact was produced.

### Existing results

| P1 result | Direct per-angle transform | Unconstrained quadratic parameter approximation |
|---|---:|---:|
| Coordinate RMSE range, micrometers | 0-5.95020 | 0.0521408-5.96587 |
| Mean per-angle coordinate RMSE, micrometers | 3.06366 | 3.06866 |
| Maximum Euclidean point error, micrometers | 11.1749 | 11.1511 |

Coordinate RMSE uses all nine points and both coordinates. The reported means are means of per-angle RMSE values, not pooled RMSE. The direct fit at zero rotation is identity to numerical precision. All direct fits converged; the minimum sampled direct denominator is 0.987474.

### Supported coefficient simplification

The dominant terms in the existing P1 coefficient curves provide the following reduced-model starting values:

$$
\alpha_1=1.85435557047\times10^{-5},\qquad
\beta_1=6.46083406774\times10^{-5},\qquad
\gamma_1=3.78982990846\times10^{-4},
$$

with the units defined above. These are extracted from the unconstrained fit, not newly refitted constrained coefficients.

Replacing the fitted scale intercepts by one and removing the nearly zero symmetry-inconsistent terms changes the existing quadratic predictions by **0.052145 micrometers pooled coordinate RMS**, with **0.086877 micrometers maximum absolute coordinate change**. The worst per-angle coordinate RMS change is 0.052152 micrometers.

These are diagnostic calculations using the nine baseline coordinates in the [P1 CSV](data/distortion_grid/distortion_grid.csv), the published physical-degree coefficients, and all 401 sampled angles. For reproducibility, evaluate both the original curves `c0+c1*theta+c2*theta^2` and the reduced functions above through the same `K_1`, then compare their predicted coordinates. No coefficient refit is involved. This comparison measures simplification error, not total error against CODE V.

### A constant baseline is not a constant transformed pattern

Holding the entire P1 pattern at its zero-rotation coordinates is a separate, much stronger reduction:

$$
\widehat{\mathbf p}_{1,i}(\theta)=\mathbf b_{1,i}.
$$

At -20 degrees, direct subtraction of the nine stored P1 coordinates from their zero-rotation baseline gives **30.424416 micrometers coordinate RMSE** and **69.985236 micrometers maximum absolute coordinate error**. Thus the entire P1 transformation cannot be frozen over the full +/-20-degree domain at a 1-micrometer scale.

A constant pattern or individual constant rotation parameters remain candidates for a narrower declared angle/field range. Test them using the coordinate-error rule below; do not infer their acceptability from small percentage changes in `s_x`, `s_y`, or `q`.

### Remaining spatial-model error

The small coefficient-simplification error does not remove the existing keystone residual. For example, at -20 degrees the P1 side-midpoint fields `(field_x,field_y)=(+/-1,0)` have a real Y coordinate of approximately **8.471582 micrometers**, while their baseline Y is zero to numerical precision. The restricted mapping predicts `Y=0` when `y_b=0`, regardless of its three parameter values.

This is a concrete missing spatial deformation, not a polynomial-order problem. A total 1-micrometer requirement over the full domain therefore needs an appropriate spatial-model extension or a smaller accepted domain; changing only the angle dependence of these three parameters cannot represent this component.

## P4 optical-model analyses

The P4 collector has already saved the full 51-accommodation by 401-rotation by nine-field sweep: 184,059 rows covering 20,451 states. See [HANDOFF.md](HANDOFF.md). Existing reported fits use two slices of that data: radial distortion at zero rotation, and rotation transforms at 0 D. A jointly calibrated reduced model has not yet been established by those reports.

### Accommodation-dependent baseline

For diagnostics, first use the empirical baseline `b_4,i(A)=p_4,i(0,A)` at each accommodation. This isolates rotation-model error from baseline-approximation error.

For a compact predictive baseline, fix the reference paraxial coordinates

$$
\mathbf u_i=\mathbf p_{\mathrm{para},4,i}(0,0),
$$

and test

$$
\mathbf z_i(A)=m(A)\mathbf u_i,\qquad m(0)=1,
$$

$$
\boxed{
\widehat{\mathbf b}_{4,i}(A)
=\left[1+k(A)\|\mathbf z_i(A)\|^2\right]\mathbf z_i(A).
}
$$

Here `m(A)` is a dimensionless paraxial scale and `k(A)` is the radial coefficient in mm^-2, called `k1` in the current reports. The radial radius is computed from `z` in millimeters, not from normalized field coordinates or from the distorted output.

The existing analysis uses each accommodation's own paraxial coordinates, whose scale changes with accommodation. Consequently, fitting only `k(A)` while holding the paraxial grid fixed does not describe the complete baseline change. The scalar `m(A)` representation is a candidate to fit and check, not an already validated result. If isotropic scale is insufficient, test a diagonal scale with separate X/Y functions before adding a more general baseline mapping.

Calibrate the paraxial scale from the paraxial data and the radial correction from the real data under this fixed coordinate convention. This avoids allowing baseline scale to drift into `s_x(0,A)` or `s_y(0,A)`.

### Existing radial fit and accommodation curves

The current radial fit uses `real = paraxial*(1+k1*r^2)` independently at each of 51 accommodation values. Its results are:

| P4 radial quantity | Minimum | Mean across accommodation | Maximum |
|---|---:|---:|---:|
| Coordinate RMSE, micrometers | 0.534472 | 0.807812 | 1.10191 |
| Maximum absolute coordinate residual, micrometers | 1.06894 | 1.61562 | 2.20383 |

At 0 D, `k1=-0.01852833121 mm^-2`. The coordinate RMSE includes all 18 scalar coordinates, including the center's zero residuals. A maximum scalar-coordinate residual is not a Euclidean point error.

Two three-parameter accommodation curves were fitted in-sample to the estimated coefficients:

$$
k(A)=d+a\exp(bA),
$$

with `(d,a,b)=(-0.0195347321373, 0.000993232205102, 0.235926474606)` and coefficient RMSE `4.126091e-6 mm^-2`; and

$$
k(A)=p_0+p_1A+p_2A^2,
$$

with `(p0,p1,p2)=(-0.0185179003996, 0.000180197179361, 5.15785422657e-5)` and coefficient RMSE `6.0346857e-6 mm^-2`. `d,a,p0` are in mm^-2; `b` is in D^-1; `p1,p2` carry the corresponding inverse powers of D. These values and their definitions are in the [P4 radial report](data/p4_radial_distortion_fit/p4_radial_distortion_fit.md).

Test constant and linear accommodation functions before requiring either nonlinear alternative. Retain a more complex curve only when it meaningfully improves final coordinate prediction. For a change `Delta k` at fixed paraxial coordinates, the induced baseline displacement is

$$
\Delta\mathbf b_i=(\Delta k)\|\mathbf z_i\|^2\mathbf z_i.
$$

This converts coefficient differences to the relevant coordinate scale; the complete composed prediction must still be checked. The grid has only two distinct nonzero radii, so these fits do not validate arbitrary intermediate field positions or a unique physical accommodation law.

### Rotation at 0 D

The current P4 rotation transform uses the actual real-coordinate baseline at zero rotation and 0 D, not the fitted radial baseline. Its spatial results are:

| P4 result at 0 D | Direct per-angle transform | Unconstrained quadratic parameter approximation |
|---|---:|---:|
| Coordinate RMSE range, micrometers | 0-5.21325 | 0.286062-6.57821 |
| Mean per-angle coordinate RMSE, micrometers | 1.20933 | 1.84800 |
| Maximum Euclidean point error, micrometers | 10.6716 | 16.0142 |

See the [P4 rotation report](data/p4_rotation_transform/report.md). The direct zero-rotation transform is identity to numerical precision, and the minimum sampled direct denominator is 0.943952.

The dominant terms give starting values for the reduced rotation functions at 0 D:

$$
\alpha_4=1.55275895420\times10^{-4},\qquad
\beta_4=3.02802819333\times10^{-4},\qquad
\gamma_4=-2.34562287684\times10^{-3}.
$$

These values are neither a refit with identity constraints nor evidence that the coefficients are accommodation-independent. The parameter-curve approximation adds noticeable error for P4, while the direct spatial family also exceeds 1 micrometer at large rotations. Those two error sources require separate checks.

### Separate P4 Z-dependent magnification study

The dedicated [P4 Z magnification report](data/p4_z_magnification/p4_z_magnification.md) uses `Lens/p4_ME.len`, zoom z1, and the explicit surface selector `s"Cornea_ENT_D"`. It collects accommodations 0, 1, 2, 3, and 4 D; RC rotations -10, -5, 0, +5, and +10 degrees; 51 absolute THI values from -5 to +5 mm at 0.2 mm spacing; and nine fields per state. This is 1,275 states and 11,475 records. The active z1 THI readback after loading this LEN is 0 mm. The study applies `THI = Z` with the same sign and records both `z_distance_mm` and `z_thickness_mm`; Z is an absolute command coordinate, not a shift from another baseline.

For each accommodation and rotation, let `R_i(A,theta,Z)` be the actual center-relative real-coordinate vector at field `i`. The direct scalar factor is measured relative to that same A and theta at Z=0:

$$
m_{4,\mathrm{direct}}(A,\theta,Z)=
\frac{\sum_i \mathbf R_i(A,\theta,0)\cdot\mathbf R_i(A,\theta,Z)}
     {\sum_i \|\mathbf R_i(A,\theta,0)\|^2}.
$$

The center point is valid in this pooled fit and contributes zero to numerator and denominator. The baseline barrel pattern at each A and theta is therefore retained in the test. The constrained linear slope is fit separately at each accommodation using only the theta=0 measured ratios:

$$
m_4(A,Z)=1+\alpha_{4Z}(A)Z,\qquad
\alpha_{4Z}(A)=
\frac{\sum_Z Z\,[m_{4,\mathrm{direct}}(A,0,Z)-1]}{\sum_Z Z^2}.
$$

The resulting per-accommodation slopes and rotation-independence checks are:

| Accommodation (D) | `alpha_4Z(A)` (mm^-1) | Maximum relative change across tested rotations (%) |
|---:|---:|---:|
| 0 | 0.00295232758 | 0.00171777 |
| 1 | 0.00295172708 | 0.00200342 |
| 2 | 0.00295112891 | 0.00230384 |
| 3 | 0.00295052435 | 0.00261870 |
| 4 | 0.00294991207 | 0.00294912 |

Across these five accommodations, the slopes span 0.0818504% of their mean. `m≈1+0.00295 Z` is a convenient common-slope approximation over the sampled domain, while `alpha_4Z(A)` is the more accurate result to use when accommodation-specific accuracy matters. The common slope is not a separate fitted model with an independently measured error bound. At 0 D, measured theta=0 endpoint ratios are 0.9854550569 at -5 mm and 1.014980754 at +5 mm; at 4 D they are 0.9854667801 and 1.014968316. The report lists measured ratios separately from constrained linear predictions.

The comparison of `m_direct(A,theta,Z)` against `m_direct(A,0,Z)` has a maximum relative variation of 0.00294912% over the five tested rotations and 51 Z planes. Applying the theta=0 constrained linear factor to each same-A/theta Z=0 real grid gives a worst-state coordinate RMSE of 0.249 micrometers and maximum Euclidean point error of 0.511 micrometers across the tested states. Pooled over all Z/rotation states, the linear coordinate RMSE is 0.100–0.106 micrometers by accommodation. Direct per-state scalar factors reduce the pooled coordinate RMSE to 0.0120–0.0160 micrometers, with maximum absolute coordinate residuals of 0.0653–0.0854 micrometers. These are residuals against the sampled simulated grids, not hardware accuracy measurements.

This Z dataset has LEN SHA-256 `fb3937e6763f27e331faa10a4863533e994030cbc0ae74364b186ba0d06d474d`. The earlier P4 accommodation/rotation dataset used hash `5e0715c70016fbcec8a49956dd2ec824a50f5c4c52ca472b0225a2523db614ac`. Because these hashes differ, treat the Z magnification result and existing P4 accommodation/radial/rotation fits as separate optical-model revisions. In particular, this study does not establish a jointly calibrated `K_4(theta,A)` across accommodation or justify composing the earlier keystone fit with `m_4(A,Z)` for end-to-end predictions. The P4 theoretical variables now include `theta`, `A`, and `Z`, but the available fits remain separate and cover different subsets/revisions.

See the standalone [structured grid](data/p4_z_magnification/p4_z_magnification.pkl), [metadata and LEN hash](data/p4_z_magnification/metadata.json), [fit artifact](data/p4_z_magnification/p4_z_magnification_fit.pkl), [plot](data/p4_z_magnification/p4_z_magnification.png), and [collector/analysis script](Script/analyze_p4_z_magnification.py).

## Minimal joint model and optional extensions

The first joint candidate is

$$
\boxed{
\widehat{\mathbf p}_{1,i}(\theta)=
\begin{bmatrix}
\dfrac{(1+\alpha_1\theta^2)b_{1,i,x}}{1+\gamma_1\theta b_{1,i,y}}\\[2mm]
\dfrac{(1+\beta_1\theta^2)b_{1,i,y}}{1+\gamma_1\theta b_{1,i,y}}
\end{bmatrix},
}
$$

$$
\boxed{
\widehat{\mathbf p}_{4,i}(\theta,A)=
\begin{bmatrix}
\dfrac{(1+\alpha_4\theta^2)\widehat b_{4,i,x}(A)}{1+\gamma_4\theta\widehat b_{4,i,y}(A)}\\[2mm]
\dfrac{(1+\beta_4\theta^2)\widehat b_{4,i,y}(A)}{1+\gamma_4\theta\widehat b_{4,i,y}(A)}
\end{bmatrix}.
}
$$

Initially, `alpha_4`, `beta_4`, and `gamma_4` are constants with respect to accommodation. P4 still depends jointly on rotation and accommodation through its baseline and the denominator. Thus no explicit accommodation dependence in the rotation coefficients does **not** mean that the complete mapping is additive or has no interaction between the two variables.

Fit or test these shared coefficients against the existing full P4 sweep; do not assume that the 0-D starting values apply unchanged at every accommodation.

If the coordinate-error test requires explicit coupling, add one dependency at a time. For example,

$$
q_4(\theta,A)=(\gamma_{40}+\gamma_{41}A)\theta
$$

adds one `A*theta` coefficient. Similarly,

$$
s_{x,4}(\theta,A)=1+(\alpha_{40}+\alpha_{41}A)\theta^2
$$

adds one `A*theta^2` coefficient. Keep unaffected coefficients constant. Higher accommodation powers are optional, not mandatory.

If the angle curves themselves need more flexibility, test symmetry-compatible terms such as `theta^3` in `q` or `theta^4` in the scales. Do not add these to compensate for residual spatial patterns that the keystone family cannot express.

The reduction hierarchy is: constant parameter; minimal angle dependence; accommodation dependence only where needed; higher-order or spatial extensions only where a measured residual requires them. Removing several terms must pass a combined test, not just separate one-term tests.

## Coordinate-error rule for accepting a simplification

Let `s` denote a state (`theta` for P1; `(theta,A)` for P4), and let `N` be the number of evaluated fields. Define

$$
\mathbf e_{\mathrm{total},i}(s)
=\widehat{\mathbf p}_{\mathrm{simple},i}(s)-\mathbf p_{\mathrm{CODE\,V},i}(s),
$$

$$
\mathbf e_{\mathrm{drop},i}(s)
=\widehat{\mathbf p}_{\mathrm{simple},i}(s)-\widehat{\mathbf p}_{\mathrm{reference},i}(s).
$$

For either residual population, coordinate RMSE at a state is

$$
E(s)=\sqrt{\frac{1}{2N}\sum_{i=1}^{N}\left(e_{i,x}^2+e_{i,y}^2\right)}.
$$

Report the maximum absolute coordinate residual and maximum Euclidean point residual separately. With the same points, RMS Euclidean point error equals `sqrt(2)` times coordinate RMSE. Preserve the nine-point convention for comparison with current reports; separately identify any changed field subset.

**Simplification criterion:** choose a declared budget `epsilon_drop` at or below the approximately 1-micrometer reference and test `max_s E_drop(s) <= epsilon_drop` over the intended domain. Also report maximum coordinate changes; an RMS criterion is not a per-coordinate bound. A constant is acceptable only on the domain where the test passes.

**Total-accuracy criterion:** separately evaluate `max_s E_total(s)` against the application's total error budget. A claim of total 1-micrometer performance requires this test, not merely a passing simplification test. The current full-domain keystone fits do not meet that total target.

Do not automatically allocate 1 micrometer to every omitted term. Deterministic approximation errors can reinforce each other; for the same coordinate norm,

$$
E_{\mathrm{total}}(s)\leq E_{\mathrm{reference}}(s)+E_{\mathrm{drop}}(s).
$$

Quadrature addition is not justified without suitable assumptions. Likewise, 1 micrometer of measurement noise is not a hard noise bound or an angular-precision statement. When applying the model to measured center-subtracted coordinates, use the measured noise covariance; shared center noise correlates the relative points.

## Calibration and validation sequence

1. **Fix coordinates and baselines.** Match field identities and units. Use the empirical zero-rotation baseline for each available accommodation to diagnose rotation fits; calibrate the compact P4 baseline separately. Do not apply the radial correction twice.
2. **Check the spatial family before its coefficient curves.** Fit independent keystone parameters at every relevant state, including all P4 accommodations. Their residual maps reveal missing spatial deformation that higher-order coefficient curves cannot fix.
3. **Fit the smallest global dependence.** Start with the three-coefficient P1 model and accommodation-independent P4 rotation coefficients. Fit constant/linear baseline functions first. Optimize final Cartesian coordinate residuals; parameter-curve RMSE alone is only a diagnostic.
4. **Test reductions and additions against the same data partition.** Refit the remaining free coefficients where appropriate, then report both simplification and total errors. Include the all-constant P1 candidate only for explicitly declared restricted domains. Add individual coupling or spatial terms only when justified.
5. **Validate the complete prediction.** Hold out rotation values and complete accommodation groups. For a compact-model held-out accommodation test, predict its baseline from the trained functions rather than reading that group's real baseline or paraxial grid. An empirical-baseline diagnostic is not end-to-end held-out validation. Inspect boundaries, per-state errors, field residuals, and denominator positivity, then check physical measurements separately.

The original full P4 accommodation/rotation sweep and the separate five-angle P4 Z sweep already exist. Their LEN hashes differ, so the pair does not support joint fitting across `theta`, `A`, and `Z`; collect matched data from one lens revision before calibrating that combined model. Additional field sampling is needed only to substantiate spatial interpolation beyond the current nine fields, and further collection is needed when the optical configuration or intended domain changes. Do not claim extrapolation beyond the calibrated domain.

## Current status and source artifacts

**Established:** separate P1 rotation at the numerically matched Z=0 baseline, P1 zero-rotation radial and magnification summaries over Z (including a linear normalized working approximation), P4 zero-rotation radial accommodation and 0-D rotation fits, plus P4 linear Z magnification by accommodation over five tested rotations. These P4 studies have different lens hashes and remain separate. Setup-specific approximate coefficient parity and the P1 identity/parity simplification calculation above are also established. Direct P1 and P4 Z-dependence checks cover five angles each; neither establishes rotation independence over all 401 angles or a full joint calibration.

**Proposed for fitting and validation:** the compact P4 baseline scale function, shared rotation coefficients across accommodation, a jointly calibrated reduced transformation, and any restricted-domain constant approximation. No new joint-fit or hardware-precision result is claimed here.

The working recommendation is the P1 Z=0 measured real baseline transformed by the Z=0 keystone map and then scaled by `m(Z)=1+0.00295165270232 Z`, as specified above, along with an accommodation-dependent P4 baseline and only the rotation or coupling terms whose omission matters in coordinate space. For the separate P4 Z dataset, use `m_4(A,Z)=1+alpha_4Z(A)Z`, with the per-accommodation slopes above; `alpha_4Z≈0.00295 mm^-1` is a convenient approximation over 0–4 D. Both Z scaling checks cover five angles. The P4 Z scale and existing P4 keystone fit are not yet a matched joint model.

Source artifacts:

- [P1 raw CSV](data/distortion_grid/distortion_grid.csv), [P1 transform report](data/p1_rotation_transform/report.md), [P1 radial/magnification report](data/p1_radial_distortion_fit/p1_radial_distortion_fit.md), and [P1 runners](Script/analyze_p1_rotation_transform.py), [radial/magnification analysis](Script/fit_p1_radial_distortion.py).
- [Original P4 accommodation/rotation metadata](data/distortion_grid_p4/metadata.json), [P4 radial report](data/p4_radial_distortion_fit/p4_radial_distortion_fit.md), and [P4 rotation report](data/p4_rotation_transform/report.md).
- [Separate P4 Z magnification report](data/p4_z_magnification/p4_z_magnification.md), [plot](data/p4_z_magnification/p4_z_magnification.png), [structured grid](data/p4_z_magnification/p4_z_magnification.pkl), [metadata](data/p4_z_magnification/metadata.json), [fit artifact](data/p4_z_magnification/p4_z_magnification_fit.pkl), and [script](Script/analyze_p4_z_magnification.py). This dataset's LEN SHA-256 is `fb3937e6763f27e331faa10a4863533e994030cbc0ae74364b186ba0d06d474d`; the original P4 dataset hash is `5e0715c70016fbcec8a49956dd2ec824a50f5c4c52ca472b0225a2523db614ac`.
- [HANDOFF.md](HANDOFF.md) for collection provenance and environment requirements; [Summary.md](Summary.md) for the existing fit summaries.
