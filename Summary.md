# Distortion model summary

**Current validated P4 result (2026-10-13):** The accommodation-specific quadratic-in-rotation keystone model does not meet the 1 um coordinate-error reference across the validated domain: total coordinate RMSE averages 4.46151 um and reaches 14.5312 um, with substantial variation by accommodation. The replacement P4 grid contains 184,059 records: 51 accommodations from 0 to 5 D in 0.1 D steps, 401 RC rotations from -20 to +20 degrees in 0.1 degree steps, and nine normalized fields per state. The pickle SHA-256 is `1d68ca2c237c76fa06539efce4911fe9ff87e203c0b11051eb4f013cbafe4b5a`. Its model, `Lens/p4_ME.len`, has SHA-256 `fb3937e6763f27e331faa10a4863533e994030cbc0ae74364b186ba0d06d474d`; this is the same model hash recorded by the retained P4 Z study. The pre-replacement P4 accommodation/rotation LEN model SHA-256 `5e0715c70016fbcec8a49956dd2ec824a50f5c4c52ca472b0225a2523db614ac` is historical model provenance only.

## P4 radial fit at Z=0, rotation=0

The signed one-term polynomial per-accommodation fit is `real = paraxial * (1 + k1*r^2)`, with no logarithmic term, `r^2=x^2+y^2` in mm^2, and each accommodation's own paraxial coordinates as its reference. The 3 x 3 field grid supplies two nonzero radial classes (edge midpoints and corners). `k1` becomes less negative from -0.0185283312073 mm^-2 at A=0 D to -0.0163117998622 mm^-2 at A=5 D, indicating decreasing barrel distortion over this fitted field. This is separate from the paraxial baseline scale, which changes to 0.896199673581 relative to A=0 at A=5 D; that scale is a diagnostic magnification factor, not another radial coefficient.

Across the 51 accommodations, direct coordinate RMSE is 0.534472 to 1.101915 um (mean 0.807812 um); the largest point error ranges from 1.068945 to 2.203830 um. Accommodation curves fitted to `k1` are `quadratic_A` with coefficients `[-0.0185179003996, 0.000180197179361, 5.15785422657e-5]` and `offset_exponential` with coefficients `[-0.0195347321301, 0.000993232198585, 0.235926475525]`, in the report's stated coefficient order and units. Both are in-sample descriptions of the 51 fitted values. Their maximum added-coordinate RMSE is 0.0366545 um for the quadratic curve (A=5 D) and 0.0428462 um for the exponential curve (A=0 D); maximum added point error is 0.0733090 and 0.0856924 um, respectively. These curve errors are distinct from the larger direct fit-to-grid residual.

See the [radial fit report](data/p4_radial_distortion_fit/p4_radial_distortion_fit.md), [k1 plot](data/p4_radial_distortion_fit/p4_radial_distortion_k1.png), [fit pickle](data/p4_radial_distortion_fit/p4_radial_distortion_fit.pkl), [metrics and provenance](data/p4_radial_distortion_fit/p4_radial_distortion_metrics.pkl), [accommodation curves](data/p4_radial_distortion_fit/p4_radial_distortion_accommodation_models.pkl), and [fit script](Script/fit_p4_radial_distortion.py).

## P4 keystone (rotation) analysis across accommodation

For every accommodation, the measured real-coordinate grid at theta=0 is its empirical barrel-distortion baseline. Rotation is modeled from that measured grid using `X=sx*x0/(1+q*y0)` and `Y=sy*y0/(1+q*y0)`, where `sx` and `sy` are unitless and `q` is in mm^-1. The radial fit is not applied a second time. The direct transform is fit independently at each of the 20,451 accommodation/angle states. Each parameter is then approximated by its own unconstrained quadratic in theta for each accommodation; no additional smooth accommodation law is fit, and no identity, even/odd, or symmetry constraints are imposed. Full coefficients are in the artifacts.

| A (D) | sx theta^2 (degree^-2) | sy theta^2 (degree^-2) | q theta (mm^-1/degree) |
|---:|---:|---:|---:|
| 0 | 0.0001552758954 | 0.0003028028193 | -0.002345622877 |
| 1 | 0.0001732684773 | 0.0003630345936 | -0.002831577821 |
| 2 | 0.0001918100541 | 0.0004266453010 | -0.003355193078 |
| 3 | 0.0002108883579 | 0.0004936398101 | -0.003918254684 |
| 4 | 0.0002305306594 | 0.0005642392799 | -0.004524126537 |
| 5 | 0.0002507548114 | 0.0006386357893 | -0.005175941610 |

These are representative physical-degree terms, not complete parameter curves: each full per-accommodation quadratic has independent `c0`, `c1`, and `c2` coefficients. At theta=0, direct estimates are numerically near (1, 1, 0), but are not constrained to identity.

| Error measure across all A/theta states | Mean | Maximum |
|---|---:|---:|
| Direct coordinate RMSE (um) | 4.06167 | 12.8112 |
| Quadratic total coordinate RMSE (um) | 4.46151 | 14.5312 |
| Quadratic additional coordinate RMSE (um) | 1.51779 | 7.25432 |
| Direct maximum point error (um) | 7.60316 | 25.104 |
| Quadratic total maximum point error (um) | 8.38888 | 25.7348 |
| Quadratic additional maximum point error (um) | 3.06876 | 14.5049 |

The direct worst case is A=5 D, theta=-20 degrees, field (1,0): predicted-minus-measured residual is approximately (+7.884, +23.834) um, with 25.104 um maximum point error. At this field the baseline y0 is approximately zero, so the transform's Y formula predicts approximately zero while the measured grid has a large vertical shift. A richer spatial model is needed to represent this baseline pattern; increasing the theta polynomial order alone cannot correct it. Coordinate-RMSE states below 1 um are 17.236% for the direct fit and 9.784% for the quadratic total fit. Those percentages are coordinate-RMSE counts, not point-error pass rates.

An independent SciPy audit reproduced predictions and metrics and checked field identities and repeated centers; selected states at A=0, 2, and 5 D and theta=-20, -10, 0, +10, and +20 degrees reached the same optimum. All direct nonlinear state fits converged; the per-accommodation quadratic least-squares designs had rank 3. The minimum denominator is 0.887473652 for direct predictions and 0.880433513 for quadratic predictions. Reusing A=0 transform parameters while retaining each accommodation's own baseline produces up to 294.802 um additional and 296.617 um total maximum point error. Center positions are measured and stored separately: the largest same-accommodation shift from theta=0 is 5.52214 mm in one component, while the accommodation-baseline center shift at theta=0 is 0 mm. No center polynomial or hardware-precision claim is implied.

See the [rotation report](data/p4_rotation_transform/report.md), [direct and quadratic predictions, residuals, centers, and provenance](data/p4_rotation_transform/p4_rotation_transform.pkl), [coefficient fits](data/p4_rotation_transform/coefficient_quadratics.pkl), [coefficient plots](data/p4_rotation_transform/keystone_coefficients_by_accommodation.png) and [physical-degree coefficient plot](data/p4_rotation_transform/physical_quadratic_coefficients_vs_accommodation.png), combined [coordinate-RMSE heatmaps](data/p4_rotation_transform/coordinate_rmse_heatmaps.png), the separate [direct keystone fit error](data/p4_rotation_transform/direct_fit_error_heatmap.png), [polynomial total error](data/p4_rotation_transform/polynomial_total_error_heatmap.png), and [polynomial additional error](data/p4_rotation_transform/polynomial_additional_error_heatmap.png), and [analysis script](Script/analyze_p4_rotation_transform.py).

## P4 barrel and keystone analyses compared

These are separate analyses with different scopes. The [barrel-fit script](Script/fit_p4_radial_distortion.py) writes to `data/p4_radial_distortion_fit`; it fits only the theta=0, Z=0 measured grid at each accommodation. The [keystone-analysis script](Script/analyze_p4_rotation_transform.py) writes to `data/p4_rotation_transform`; its full report and artifacts are linked above. It uses each accommodation's measured theta=0 real grid as the empirical baseline and fits the additional changes across rotation. It does not apply the radial fit a second time.

| Analysis | Purpose and input | Scope and fitted parameters | Baseline and error summary |
|---|---|---|---|
| Barrel / radial | Diagnose radial barrel distortion at zero rotation; uses the nine field coordinates and each state's paraxial coordinates from the P4 grid. | 51 accommodations, theta=0, Z=0; one signed `k1` per accommodation in `real = paraxial * (1 + k1*r^2)`, where `r^2=x^2+y^2` in mm^2. `k1` changes from -0.0185283312073 mm^-2 at A=0 D to -0.0163117998622 mm^-2 at A=5 D. | Each accommodation's own paraxial grid. Direct coordinate RMSE across accommodations: mean 0.807812 um, maximum 1.101915 um; maximum point error reaches 2.203830 um. In-sample accommodation curves add at most 0.0366545 um coordinate RMSE (quadratic) or 0.0428462 um (offset exponential). |
| Keystone / rotation | Describe measured real-grid deformation as the eye rotates, relative to its own measured zero-angle grid at each accommodation. | 51 accommodations, 401 angles from -20 to +20 degrees, Z=0; direct transform fits `sx`, `sy`, `q` at each state. Separate unconstrained quadratic-in-angle curves approximate each parameter for each accommodation. | Own measured theta=0 real grid at each A. Direct fit coordinate RMSE across A/angle states: mean 4.06167 um, maximum 12.8112 um; maximum point error reaches 25.104 um. Quadratic total coordinate RMSE: mean 4.46151 um, maximum 14.5312 um; additional coefficient-curve error reaches 7.25432 um coordinate RMSE and 14.5049 um maximum point error. |

The error figures above compare the same metric types (coordinate RMSE with coordinate RMSE, maximum point error with maximum point error), but they are not a controlled head-to-head: the radial fit covers only zero angle, while the rotation fit covers a broad angular range and a different deformation relative to an empirical baseline. The radial analysis diagnoses the zero-angle baseline only. The keystone residual can include angle-dependent radial or other non-projective changes, so a restricted keystone transform does not establish that all observed rotation deformation is keystone. No separate rotation-dependent `k1` was estimated, and no combined compact radial-plus-keystone model was evaluated end to end. The rotation analysis uses the measured theta=0 grid directly, rather than composing the radial fit into its predictions.

Across the rotation range, accommodation dependence matters: applying A=0 transform parameters to other accommodations' own baselines produces up to 294.802 um additional maximum point error. This diagnostic is in the [rotation report](data/p4_rotation_transform/report.md). Additional references: the [radial k1 curve](data/p4_radial_distortion_fit/p4_radial_distortion_k1.png), separate [direct keystone fit error](data/p4_rotation_transform/direct_fit_error_heatmap.png), [polynomial total error](data/p4_rotation_transform/polynomial_total_error_heatmap.png), and [polynomial additional error](data/p4_rotation_transform/polynomial_additional_error_heatmap.png), [physical rotation coefficients versus accommodation](data/p4_rotation_transform/physical_quadratic_coefficients_vs_accommodation.png), and [rotation quadratic coefficient plots](data/p4_rotation_transform/keystone_coefficients_by_accommodation.png).

## P1 dataset status

The validated replacement P1 grid contains 184,059 records across 51 absolute THI planes from -5 to +5 mm, 401 RC rotations, and nine fields per state. Its pickle SHA-256 is `e29d226a0af1e5db8c3cfbf673fa92100917825f8c25d74ae71f994151519f02`; the current sequence baseline is THI=1 mm. P1 radial and rotation fit results are stale because they predate this replacement sweep, and refitting is pending. The old fit output folders have been removed. Retain [the P1 radial fit script](Script/fit_p1_radial_distortion.py) and [the P1 rotation analysis script](Script/analyze_p1_rotation_transform.py) for the refit. THI coordinates are reported as collected, without conversion to displacement from the 1 mm baseline.

## Separate P4 Z magnification study

The retained Z study uses `Lens/p4_ME.len`, zoom z1, `Cornea_ENT_D`, and same-sign absolute `THI=Z`; its loaded z1 THI baseline is 0 mm. It samples accommodations 0, 1, 2, 3, and 4 D; RC rotations -10, -5, 0, +5, and +10 degrees; 51 absolute Z values from -5 to +5 mm in 0.2 mm steps; and nine fields (1,275 states, 11,475 records). Each state's Z=0 real grid is its magnification reference. The pooled dot-product scale is `m=dot(R0,RZ)/dot(R0,R0)`. A linear `m(A,Z)=1+alpha(A)Z` fit uses theta=0 ratios and is checked at the other four angles.

| Accommodation (D) | alpha (mm^-1) | Maximum relative rotation variation (%) |
|---:|---:|---:|
| 0 | 0.00295232758 | 0.00171777 |
| 1 | 0.00295172708 | 0.00200342 |
| 2 | 0.00295112891 | 0.00230384 |
| 3 | 0.00295052435 | 0.00261870 |
| 4 | 0.00294991207 | 0.00294912 |

A common `alpha ~= 0.00295 mm^-1` is convenient over this sampled range; use the per-accommodation slopes when accuracy matters. The worst linear-model state has 0.249 um coordinate RMSE and 0.511 um maximum point error. These results describe simulated-grid consistency, not hardware precision. The matching LEN hash establishes shared model provenance, but does not validate a composed theta/A/Z model; the current replacement P4 calibration directly covers Z=0 and all 51 accommodations and 401 rotation angles.

See the [P4 Z report](data/p4_z_magnification/p4_z_magnification.md), [plot](data/p4_z_magnification/p4_z_magnification.png), [metadata](data/p4_z_magnification/metadata.json), [fit artifact](data/p4_z_magnification/p4_z_magnification_fit.pkl), and [script](Script/analyze_p4_z_magnification.py).

## Validation scope

The radial and rotation fits use all measured states in-sample and have no held-out data or simulated-noise trials. Their residuals therefore do not establish angular precision or hardware accuracy. The 1 um line used in reports and plots is a measurement reference for coordinate error only.
