# P4 keystone analysis across accommodation and eye rotation

Input: `C:\Users\rueijrwu\OneDrive\Project\DistortionTracking\data\distortion_grid_p4\distortion_grid.pkl` (19142652 bytes), SHA256 `1d68ca2c237c76fa06539efce4911fe9ff87e203c0b11051eb4f013cbafe4b5a`. The structured array was validated for A=0..5 D by 0.1 D, theta=-20..20 degrees by 0.1 degrees, z=0, exactly the normalized 3x3 field identities in every state, finite coordinates, and repeated absolute centers.

## Empirical baseline and fit

At every accommodation A, the nine measured real coordinates at theta=0 are the source grid. The direct transform is fitted independently for each (A, theta): `X=sx*x0/(1+q*y0)`, `Y=sy*y0/(1+q*y0)`. This treats the measured theta-zero map as the empirical barrel baseline; no additional radial k1 term is added to those baseline coordinates. The fit has three unconstrained parameters and no translation/shear. The fit and quadratic use all angles in-sample.

## Error definitions

Direct error compares each angle's direct fitted prediction with its measured real grid. The order-2 parameter curves are fitted independently in theta for sx, sy, and q. Polynomial total error compares the composition of those coefficient curves with measured truth. Polynomial additional error compares that polynomial-composed prediction with the same-angle direct prediction, isolating coefficient-curve approximation error. They are stored separately. Coordinate RMSE pools both coordinates across the nine points; max absolute coordinate is over all 18 scalar errors; maximum Euclidean point error is the worst of nine 2D point errors. All reported errors below are in micrometers.

| Metric | Minimum | Mean | Maximum |
|---|---:|---:|---:|
| Direct coordinate RMSE (µm) | 0 | 4.06167 | 12.8112 |
| Direct max abs coordinate (µm) | 0 | 6.67824 | 23.834 |
| Direct max Euclidean point (µm) | 0 | 7.60316 | 25.104 |
| Polynomial total coordinate RMSE (µm) | 0.214017 | 4.46151 | 14.5312 |
| Polynomial total max Euclidean point (µm) | 0.367123 | 8.38888 | 25.7348 |
| Polynomial additional max Euclidean point (µm) | 0.363467 | 3.06876 | 14.5049 |

Largest direct max-point error: 25.104 µm at A=5 D, theta=-20 degrees.
Direct algebraic initializer rank values: [3]; condition range 1.46048 to 1.7986. Converged fraction: 100.000000%; accepted steps range 0 to 9. Minimum direct sampled denominator: 0.887473652; polynomial minimum: 0.880433513.

## Order-2 theta coefficient model

Each parameter uses an independent unconstrained least-squares polynomial. Coefficients are stored in scaled basis `[1,t,t²]`, `t=theta/20`, and physical-degree basis `[1,theta,theta²]`; physical coefficients are also plotted against accommodation. This is an in-sample fit over all 401 angles. No even/odd symmetry or identity-at-zero constraint is imposed.

At theta=0 the direct per-A fit maps each empirical baseline to itself. The fitted parameters are estimated and not constrained to identity; their ranges are:
- `sx`: 1 to 1 (unitless).
- `sy`: 1 to 1 (unitless).
- `q`: -6.56135956169e-17 to 8.32667268469e-17 (mm^-1).

Per-A spatial-error summary (mean and worst over theta):

| A (D) | Direct mean max-point (µm) | Direct worst (µm) | Quadratic total mean max-point (µm) | Quadratic total worst (µm) | Quadratic added mean max-point (µm) | Quadratic added worst (µm) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.48878 | 10.6716 | 4.17473 | 16.0142 | 2.42357 | 8.72796 |
| 0.1 | 2.35883 | 10.1175 | 4.09781 | 15.67 | 2.44042 | 8.815 |
| 0.2 | 2.2653 | 9.55856 | 4.03628 | 15.3418 | 2.45898 | 8.90935 |
| 0.3 | 2.20982 | 9.00063 | 3.99021 | 15.0306 | 2.47801 | 9.00591 |
| 0.4 | 2.19226 | 8.44548 | 3.96243 | 14.7375 | 2.49718 | 9.10346 |
| 0.5 | 2.21189 | 7.89291 | 3.95058 | 14.4628 | 2.51643 | 9.20189 |
| 0.6 | 2.26998 | 7.3444 | 3.95364 | 14.2078 | 2.53572 | 9.30076 |
| 0.7 | 2.37109 | 6.8013 | 3.97147 | 13.974 | 2.55514 | 9.4002 |
| 0.8 | 2.51391 | 6.26688 | 4.00445 | 13.7626 | 2.57453 | 9.49923 |
| 0.9 | 2.69683 | 5.7393 | 4.05648 | 13.5743 | 2.59433 | 9.59963 |
| 1 | 2.90992 | 5.55026 | 4.14279 | 13.4102 | 2.61432 | 9.69934 |
| 1.1 | 3.13879 | 5.58377 | 4.27129 | 13.2719 | 2.63518 | 9.80001 |
| 1.2 | 3.3808 | 5.6173 | 4.44672 | 13.1593 | 2.65754 | 9.90079 |
| 1.3 | 3.63563 | 5.65098 | 4.64322 | 13.0738 | 2.68136 | 10.002 |
| 1.4 | 3.90241 | 5.68499 | 4.85213 | 13.0161 | 2.70608 | 10.1035 |
| 1.5 | 4.18105 | 5.76247 | 5.0732 | 12.9856 | 2.7308 | 10.2049 |
| 1.6 | 4.4714 | 6.20283 | 5.30681 | 12.9841 | 2.75583 | 10.3073 |
| 1.7 | 4.76901 | 6.65342 | 5.55095 | 13.01 | 2.78082 | 10.4095 |
| 1.8 | 5.0691 | 7.11461 | 5.80305 | 13.0645 | 2.80608 | 10.5126 |
| 1.9 | 5.37096 | 7.58519 | 6.06168 | 13.1466 | 2.8315 | 10.6159 |
| 2 | 5.67493 | 8.06581 | 6.32682 | 13.2567 | 2.85721 | 10.7203 |
| 2.1 | 5.98021 | 8.55504 | 6.59726 | 13.3932 | 2.88299 | 10.8249 |
| 2.2 | 6.2876 | 9.05452 | 6.87337 | 13.5562 | 2.90897 | 10.9306 |
| 2.3 | 6.59636 | 9.56288 | 7.15409 | 13.7437 | 2.93498 | 11.0363 |
| 2.4 | 6.90657 | 10.0801 | 7.43926 | 13.9558 | 2.96129 | 11.1432 |
| 2.5 | 7.21783 | 10.6053 | 7.72818 | 14.1912 | 2.98792 | 11.2513 |
| 2.6 | 7.53042 | 11.1395 | 8.02079 | 14.4483 | 3.01461 | 11.3598 |
| 2.7 | 7.84446 | 11.6825 | 8.31704 | 14.7266 | 3.04201 | 11.4694 |
| 2.8 | 8.15968 | 12.2339 | 8.61665 | 15.0249 | 3.07064 | 11.5802 |
| 2.9 | 8.47607 | 12.7926 | 8.91921 | 15.3419 | 3.10045 | 11.6919 |
| 3 | 8.7933 | 13.3537 | 9.22432 | 15.6765 | 3.13241 | 11.8051 |
| 3.1 | 9.11188 | 13.9181 | 9.53229 | 16.0283 | 3.16716 | 11.9198 |
| 3.2 | 9.43132 | 14.4847 | 9.84259 | 16.3953 | 3.20381 | 12.0356 |
| 3.3 | 9.75157 | 15.0534 | 10.1551 | 16.7768 | 3.24182 | 12.1531 |
| 3.4 | 10.0728 | 15.6248 | 10.4699 | 17.172 | 3.28077 | 12.2719 |
| 3.5 | 10.3951 | 16.1991 | 10.7885 | 17.5805 | 3.32066 | 12.3927 |
| 3.6 | 10.7182 | 16.7753 | 11.1101 | 18.0009 | 3.36141 | 12.5156 |
| 3.7 | 11.0421 | 17.3543 | 11.4342 | 18.4325 | 3.40286 | 12.64 |
| 3.8 | 11.3669 | 17.9357 | 11.7603 | 18.8746 | 3.44507 | 12.7667 |
| 3.9 | 11.6923 | 18.5193 | 12.0886 | 19.3265 | 3.48803 | 12.8954 |
| 4 | 12.0185 | 19.1053 | 12.4187 | 19.7878 | 3.53181 | 13.027 |
| 4.1 | 12.3455 | 19.6936 | 12.7506 | 20.2578 | 3.57635 | 13.1612 |
| 4.2 | 12.6732 | 20.2847 | 13.0842 | 20.8291 | 3.62157 | 13.2976 |
| 4.3 | 13.0016 | 20.8783 | 13.4194 | 21.4323 | 3.66757 | 13.4367 |
| 4.4 | 13.3308 | 21.4744 | 13.7556 | 22.0385 | 3.7143 | 13.5784 |
| 4.5 | 13.6605 | 22.0727 | 14.0929 | 22.647 | 3.76195 | 13.7239 |
| 4.6 | 13.9911 | 22.6741 | 14.4315 | 23.2591 | 3.81037 | 13.8724 |
| 4.7 | 14.3222 | 23.2775 | 14.771 | 23.8734 | 3.85979 | 14.0252 |
| 4.8 | 14.6541 | 23.8836 | 15.1116 | 24.4909 | 3.91002 | 14.1813 |
| 4.9 | 14.9865 | 24.4924 | 15.4532 | 25.1113 | 3.96113 | 14.3411 |
| 5 | 15.3198 | 25.104 | 15.796 | 25.7348 | 4.01315 | 14.5049 |

Quadratic additional coordinate-RMSE summary (µm):
Minimum 0.203729; mean 1.51779; maximum 7.25432.
Fractions of all A/theta states below 1 µm, by distinct error metric:
- direct coordinate RMSE: 17.236%.
- polynomial total coordinate RMSE: 9.784%.
- polynomial additional coordinate RMSE: 31.641%.
- direct maximum point error: 8.943%.
- polynomial total maximum point error: 3.672%.
- polynomial additional maximum point error: 11.594%.

Representative physical-degree coefficients use `c0 + c1*theta_deg + c2*theta_deg^2`; c1 and c2 include per-degree and per-degree-squared units.

| A (D) | Parameter | c0 | c1 (per degree) | c2 (per degree²) |
|---:|---|---:|---:|---:|
| 0 | sx (unitless) | 0.999841805 | -8.932195261e-14 | 0.0001552758954 |
| 0 | sy (unitless) | 1.000363837 | 1.504886493e-13 | 0.0003028028193 |
| 0 | q (mm^-1) | -2.990926893e-12 | -0.002345622877 | 3.459789746e-14 |
| 1 | sx (unitless) | 0.9997835499 | 1.876242217e-13 | 0.0001732684773 |
| 1 | sy (unitless) | 1.00025696 | -2.172526048e-13 | 0.0003630345936 |
| 1 | q (mm^-1) | -2.472089655e-12 | -0.002831577821 | 4.628224887e-15 |
| 2 | sx (unitless) | 0.9997124267 | -1.043061471e-13 | 0.0001918100541 |
| 2 | sy (unitless) | 1.000110006 | 2.817329703e-13 | 0.000426645301 |
| 2 | q (mm^-1) | -6.790686204e-12 | -0.003355193078 | 6.538044411e-14 |
| 3 | sx (unitless) | 0.9996272074 | 1.237801528e-13 | 0.0002108883579 |
| 3 | sy (unitless) | 0.9999193644 | -4.100469964e-14 | 0.0004936398101 |
| 3 | q (mm^-1) | 1.628466266e-12 | -0.003918254684 | 1.189950916e-14 |
| 4 | sx (unitless) | 0.9995266578 | -1.060915245e-13 | 0.0002305306594 |
| 4 | sy (unitless) | 0.9996850277 | 2.055369763e-13 | 0.0005642392799 |
| 4 | q (mm^-1) | -4.284949451e-12 | -0.004524126537 | 2.676751182e-14 |
| 5 | sx (unitless) | 0.9994093377 | -6.866868185e-14 | 0.0002507548114 |
| 5 | sy (unitless) | 0.9994100841 | -1.967342955e-13 | 0.0006386357893 |
| 5 | q (mm^-1) | 7.934911304e-12 | -0.00517594161 | -7.273088382e-14 |

At each A, the quadratic predictions are compared with direct and measured predictions separately below.

At each A, coefficient fit RMSE by parameter (base parameter units):

| A (D) | sx RMSE | sy RMSE | q RMSE (mm^-1) | max coefficient condition |
|---:|---:|---:|---:|---:|
| 0 | 0.000139569 | 0.000363787 | 0.00108686 | 3.74457 |
| 1 | 0.000191626 | 0.00028357 | 0.00125295 | 3.74457 |
| 2 | 0.00025563 | 0.000175491 | 0.00143406 | 3.74457 |
| 3 | 0.000332913 | 8.34034e-05 | 0.00163271 | 3.74457 |
| 4 | 0.000424835 | 0.000207058 | 0.00185734 | 3.74457 |
| 5 | 0.000533015 | 0.00042213 | 0.00212111 | 3.74457 |

## Ignoring accommodation

For this diagnostic, each A's own measured theta-zero source grid is retained. Parameters fitted at A=0 are applied at every angle to each accommodation's own source. Additional error compares this A=0-parameter prediction to the per-A direct prediction; total error compares it to measured truth.

Ignoring-A additional max point error range: 0–294.802 µm; total max point error range: 0–296.617 µm.

## Absolute center

Absolute center x/y are validated as repeated across the nine rows per state. Raw center curves are stored and plotted separately from same-A changes `c(A,theta)-c(A,0)`. Accommodation baseline is separately stored as `c(A,0)-c(0,0)`. No center polynomial is imposed.
Across all A and theta, the largest absolute same-A theta-zero-relative center component is 5522.14 µm. The largest A-baseline component at theta=0 relative to A=0 is 0 µm.

## Artifacts

- `p4_rotation_transform.pkl`: direct and polynomial predictions, residual/error arrays, parameters, diagnostics, center arrays, and provenance.
- `coefficient_quadratics.pkl`: scaled and physical-degree coefficient fits by A.
- `keystone_coefficients_by_accommodation.png`, `physical_quadratic_coefficients_vs_accommodation.png`.
- Error heatmaps for direct, polynomial total/additional, and ignoring-accommodation total/additional errors; 1 µm contours are cyan.
- `absolute_and_relative_centers.png`.
