# P1 and P4 distortion transformations

## Purpose and modeling principle

Describe the center-relative image pattern as a function of one-axis eye rotation `theta` and accommodation `A`, using the fewest dependencies needed at the measurement scale.

**A parameter may be held constant when removing its dependence introduces an acceptable image-coordinate error over the intended operating domain.** Approximately **1 micrometer (0.001 mm)** is the project measurement-noise reference. Small coefficient variation alone is not sufficient: its effect must be evaluated after the complete spatial transformation.

The working model treats P1 as accommodation-independent and P4 as accommodation-dependent. P1 may still require rotation-dependent deformation. The available P1 dataset has no accommodation dimension, so accommodation independence is a modeling premise, not a result of a P1 accommodation sweep.

The current calibration domain is one-axis RC rotation from -20 to +20 degrees. P4 additionally covers accommodation from 0 to 5 D. Each state has nine identified fields in a 3 x 3 grid. The P1 and P4 coefficients are specific to their respective optical setups; their values and signs are not interchangeable.

This document distinguishes existing fitted results from the proposed reduced joint model. Reported results refer to repository revision `f6732f2d523bd3ee13a216de88ad4c1cb174a2a3`. Rewriting this specification does not change the collectors, fitted artifacts, or analysis implementations.

## Coordinates and baseline convention

Let `i` identify a fixed field point and let

$$
\mathbf p_{j,i}(\theta,A)
=\begin{bmatrix}X_{j,i}(\theta,A)\\Y_{j,i}(\theta,A)\end{bmatrix},
\qquad j\in\{1,4\},
$$

be its real image coordinates in millimeters, relative to the center ray at the same state. For P1, omit `A`.

Define the empirical zero-rotation baselines as

$$
\mathbf b_{1,i}=\mathbf p_{1,i}(0),
\qquad
\mathbf b_{4,i}(A)=\mathbf p_{4,i}(0,A).
$$

These baselines already contain the sampled barrel distortion. Applying another radial correction to an empirical real-coordinate baseline would double-count that correction. A fitted radial model is an alternative way to represent the baseline from paraxial coordinates, not an extra correction to the same real baseline.

The rotation source must be the zero-rotation baseline with matching field identities, not the paraxial grid at the rotation being predicted. All quantities here describe relative pattern deformation; absolute center displacement would require a separate translation model.

The intended decomposition is

$$
\boxed{
\widehat{\mathbf p}_{1,i}(\theta)
=K_1(\theta;\mathbf b_{1,i}),
\qquad
\widehat{\mathbf p}_{4,i}(\theta,A)
=K_4(\theta,A;\widehat{\mathbf b}_{4,i}(A)).
}
$$

P1 has a fixed baseline. P4 has an accommodation-dependent baseline. Start with no accommodation dependence in the rotation coefficients, and add it only when coordinate errors require it.

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

P1 uses its actual real-coordinate grid at zero rotation as a constant baseline. Its calibration contains 401 rotations and nine fields per rotation. The existing analysis independently estimates three keystone parameters at each angle using algebraic initialization and damped Cartesian Gauss-Newton refinement, then fits unconstrained quadratics to the resulting parameter curves. See the [P1 transform report](data/p1_rotation_transform/report.md).

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

The full P4 sweep already exists; joint fitting does not require another CODE V collection. Additional field sampling is needed only to substantiate spatial interpolation beyond the current nine fields, and further collection is needed when the optical configuration or intended domain changes. Do not claim extrapolation beyond the calibrated domain.

## Current status and source artifacts

**Established:** separate P1 rotation, P4 zero-rotation radial accommodation, and P4 0-D rotation fits; setup-specific approximate coefficient parity; the P1 identity/parity simplification calculation above.

**Proposed for fitting and validation:** the compact P4 baseline scale function, shared rotation coefficients across accommodation, a jointly calibrated reduced transformation, and any restricted-domain constant approximation. No new joint-fit or hardware-precision result is claimed here.

The working recommendation is a fixed P1 baseline, an accommodation-dependent P4 baseline, and only the rotation or coupling terms whose omission matters in coordinate space.

Source artifacts:

- [P1 raw CSV](data/distortion_grid/distortion_grid.csv), [P1 transform report](data/p1_rotation_transform/report.md), and [P1 runner](Script/analyze_p1_rotation_transform.py).
- [P4 sweep metadata](data/distortion_grid_p4/metadata.json), [P4 radial report](data/p4_radial_distortion_fit/p4_radial_distortion_fit.md), and [P4 rotation report](data/p4_rotation_transform/report.md).
- [HANDOFF.md](HANDOFF.md) for collection provenance and environment requirements; [Summary.md](Summary.md) for the existing fit summaries.
