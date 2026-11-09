# P1/P4 distortion work: handoff (updated 2026-11-08)

This dated handoff summarizes the current implementation and the next user-directed investigation. [Summary.md](Summary.md#current-audited-status-2026-11-08) is the canonical project narrative and result table; [Theory.md](Theory.md#current-hybrid-model-and-geometric-interpretation-2026-11-08) gives the verified equations, physical interpretation, and identifiability limits. The 2026-11-06/07 workstream notes below are retained as **historical snapshot**; their older “current,” model-selection, and waiting-for-confirmation statements do not override this update.

## Current objective and implemented baseline

The requested next step is a **geometry-first absolute nine-ray P1/P4 forward model** that derives the labeled grid center, centroid bias, and P4 accommodation-dependent center motion from its own transformed rays. That design is **not implemented**. Preserve the audited 87-scalar hybrid and its row-level results as the comparison baseline; do not re-label it a complete physical lens forward/inverse model.

At A=0, P1 and P4 currently use the same anchored centroid-bias B3 form (nine fitted scalars per channel) and the same full two-input degree-three gaze form (a separate 10×2 coefficient matrix, ten vector coefficients or 20 scalar components per channel). Their coefficients and saved coordinate normalizers are independent. `G1` and `G4` have no A feature. P1 has **no fitted full nine-ray projective/Brown/radial warp** in this active pipeline. P4 adds a 27-value center-relative field family: 12 frozen PF12 A=0 shape values plus 15 newly fitted A-dependent values. Its nine-field prediction uses an eight-outer-ray centroid anchor to the separately fitted `B4`; a separate two-value M1 diagonal scale maps the P4 central coordinate between A and A=0. The new field and M1 effects vanish at A=0. Total fitted-family count: **58 frozen B/G + 12 frozen PF12 + 15 new P4 field + 2 new M1 = 87**, of which 17 were newly estimated in this extension. Measured references and train-only normalizers are separate constants. The field uses projective/keystone-like, Brown decentering, and radial terms on the raw measured reference; no barrel-removal inversion or fitted P1/P4 registration is used.

P1 independently solves gaze from its observed nine-ray mean. Its output seeds the first P4 accommodation profile from the observed centered grid. The P4 loop estimates its center from the predicted full-field mean, applies M1 inverse normalization, obtains gaze from frozen `G4`, then re-profiles A from the nine-ray shape. It takes no true A, gaze, or labeled center as an inference input and does not fuse P1/P4 gaze. A fixed-A control holds the **observation-derived P1-conditioned A seed**, not true A.

On the full common primary holdout of **83,229 states**, the P1-conditioned A seed has RMS 0.027534 D and the final joint A estimate has bias −0.002192 D, RMS **0.025896 D**, P95 absolute 0.051393 D, maximum absolute 0.180322 D; 82,685/83,229 (99.346%) are within 0.1 D. P1 independent gaze radial RMS is about 0.002954°. Final P4 gaze radial RMS/P95/maximum are **0.015035°/0.028821°/0.034350°**, and labeled P4 center vector RMS is **0.8420 µm**. The fixed-A control has P4 gaze RMS **0.0147033°**, so joint feedback improves A RMS from its seed but slightly worsens gaze RMS versus that control. All primary states converged without alternate rescue or fast-to-strict profile fallback. The 3,249 A-bound estimates occur only at true A endpoints; no gaze-bound hit occurs. Full fast41 wall time was 233.885 s for a batch, **not** a single-frame latency claim.

The forced-**known**-A=0 identity control exactly reproduces the prior independent P4 gaze/center arrays on its own 3,943 primary A=0 holdout. In the separate **unknown**-A run, the true-A=0 common slice has 3,935 states, A RMS 0.004060 D, and P4 gaze RMS 0.00056944°. Do not feed true A=0 to that unknown-A estimator or combine these two scores. The eight-row slice difference comes from the broader common-holdout union exclusion.

## Why the next model is different

The current P4 `H` is center-relative and fixes its labeled central field at zero. It cannot by itself represent absolute central-ray motion, so `B4`, the eight-ray mean anchor, and M1 are **not algebraically redundant inside that parameterization**. Their removal requires a redesigned absolute-coordinate forward family. The labeled central ray, arithmetic nine-ray centroid, and optical principal/distortion pivot are different points. A radial transform about fixed pivot `O` leaves `O` fixed but can move an off-axis labeled central ray; recentering each grid on its own central ray hides this effect. An approximate inverse can be self-consistent while predicting the wrong physical center.

The intended `W1/W4` should use the **same full-grid transformation family with separately fitted coefficients**, in fixed calibrated sensor coordinates, and predict all nine indexed rays including the labeled central ray. P1 uses A=0; P4 A-dependent increments vanish at A=0 and account for accommodation inside the same magnification/projective/radial geometry. Derive `c_j=W_j(r_center)`, `mu_j=mean_k W_j(r_k)`, and `b_j=mu_j-c_j` from that fit, rather than fitting B, an outer-ray anchor, or M1 separately. Keep separate ten-vector `G1/G4` maps as the default readouts/initializers on **derived** centers unless the user changes scope. A direct full geometric inverse for gaze is an optional later architecture, not a decision to remove G now. A projective/Brown/radial W is still a geometry-motivated **surrogate** until it is derived from source, optical surfaces, accommodation mechanics, and sensor projection.

Before fitting, freeze labeled-ray identity, sensor origin/scale/orientation, distortion pivot and translation gauge; do not force equal P1/P4 coefficients or add unvalidated registration. Fit the **absolute** nine-ray positions jointly and score both whole-grid residuals **and** labeled-central-ray error. The first ablation should compare this no-B/no-anchor/no-M1 design with the unchanged 87-value baseline on identical masks. The number of new W coefficients is undecided until its basis and gauges are frozen; do not promise fewer than 87 values or better accuracy. Rank or a large count of correlated ray coordinates alone does not prove identifiability or prevent overfitting. See [Theory.md](Theory.md#intended-absolute-coordinate-replacement-not-implemented) for the W equations and conditions.

## Calibration, evaluation, and limits to preserve

- A=0 P1/P4 B/G and frozen P4 PF12 use **25 unique gaze poses** with each axis in `{−12.4, −6.4, 0, 6.4, 12.4}` degrees. P1 has no A dimension. The P4 accommodation extension and M1 used the matched **75 known-A poses** at A = 0, 2, 4 D; their A=0 components remain frozen. The actual ray support is ±12.4°, inside the requested ±12.5° runtime bound, across 21 A levels from 0 to 4 D in 0.2 D steps.
- The common holdout has **83,250** states after excluding the 99-row union of old/new calibration masks. The user-excluded physical P1 gaze `(ADE,BDE)=(12°,9.6°)` is retained at all 21 A levels as a separate diagnostic, yielding **83,229 primary** states. It has not been independently retraced or proved physically invalid. The own A=0 primary holdout is 3,943; the A=0 common slice is 3,935. Keep source IDs and masks, not just counts.
- Use only calibration data for basis/rank/conditioning and grouped CV decisions; score untouched common/own holdouts afterward. The P4 15-value extension has full Jacobian rank (raw/column-scaled condition 96.54/29.56), but its 2.099 µm coordinate RMS on 75 calibration poses is **training** fit. M1 won a one-SE comparison with M2 using 24 grouped calibration folds (normalized-center CV RMS 7.0416 versus 7.0378 µm). The 512-primary-plus-21-excluded predeclared pilot compared strict401/fast41 final states to within `8.73e−8` D in A and `8.78e−9`° per gaze component. Corrected F-grid orientation was checked on all saved primary final states: 665,832/665,832 triangle signs preserved, minimum area ratio 0.84725; this is sampled-grid evidence, not a global-warp proof.
- The evaluated dataset is dense, correlated, **noiseless ray-traced interpolation** in nominal recorded P1/P4 frames, not hardware precision or independent optical-system validation. P1 patterns repeated across A are not extra independent P1 measurements. Sensor noise, gaze-label error, calibration repeats, and frame/registration perturbations remain **proposed**, not completed. Do not infer general identifiability or immunity to overfit from 87-versus-75 counts: each P4 pose has nine two-coordinate rays, but references and output coordinates are correlated. A 3×3 nine-pose gaze stencil cannot identify an unrestricted ten-monomial `G1` (rank at most nine). Do not substitute old A-expanded C40/C60/129 fits or a symmetry-reduced gaze map for the separately fitted G1/G4 baseline.

## Canonical files and practical restart

- Read [AGENTS.md](AGENTS.md) for the absolute Python executable and delegation rule, then [Summary.md](Summary.md#current-audited-status-2026-11-08) for status/results and [Theory.md](Theory.md#current-hybrid-model-and-geometric-interpretation-2026-11-08) for equations/gauges. The working tree has pre-existing edits and generated artifacts; inspect it before changing files. The historical notes below are not a new approval gate.
- Pinned paired observation subset: [source_subset.npz](data/p4_p1_limited_gaze_12p5/source_subset.npz), SHA-256 `97c6db1b455f9abf7354351fa768cca7eea67198d8a4bfd14bbe684475462daf`. Preserve the user-described mirrored common-source convention and the nominal recorded frames; do not treat `SRC_ROT` naming as independent illumination evidence or infer a validated external registration.
- Frozen independent A=0 channels: [calibration summary](data/p1_p4_a0_independent_gaze/calibration/summary.json) and [runtime summary](data/p1_p4_a0_independent_gaze/runtime/summary.json). Selected P4 accommodation model: [field fit summary](data/p4_unified_combined/calibration/full_field_profile_final/summary.json), [field artifact](data/p4_unified_combined/calibration/full_field_profile_final/p4_full_field_27.npz), [M1 selection](data/p4_unified_combined/calibration/center_motion/summary.json), and [M1 artifact](data/p4_unified_combined/calibration/center_motion/central_motion_M1.npz). Canonical evaluated output: [full fast41 summary](data/p4_unified_combined/runtime/full_fast41/summary.json) and [row-level runtime.npz](data/p4_unified_combined/runtime/full_fast41/runtime.npz). The A=0 PF12 `/12.4` to SparseModel `/20` coefficient conversion is `c_new=c_old(20/12.4)^(i+j)`; omitting it breaks the frozen reduction.
- Current implementation entry points: [frozen A=0 loader](Script/p4_unified_a_gaze_model.py), [P4 full field](Script/p4_unified_fullshape.py), [M1 center motion](Script/p4_unified_center_motion.py), [joint solver](Script/p4_unified_joint_solver.py), and [full evaluation driver](Script/evaluate_p4_unified_combined.py). No new CODE V collection or barrel-removal step is needed to review these saved results. Start the replacement in separate files/artifacts, freeze its W basis and gauges before fitting, then run calibration-only checks and the same heldout comparisons. Use `C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe` explicitly for every Python check or script, as required by AGENTS.md.

## Historical 2026-11-06/07 snapshot

The material below is retained for its experiment provenance and earlier decisions. Statements there that a smaller model was not delivered, that a specific old candidate was preferred, or that confirmation was pending are superseded by the current update above.

# P4 distortion work: historical snapshot (2026-11-06/07)

At this snapshot, the working tree was uncommitted and no CODE V collection or fit process was active. This is historical context, not a current process-status check.
Lens file note from that snapshot: `Lens/p4_ME.len` had hash prefix `c4b7780e…`. The older `fb3937e6…` hash retained in [Summary.md](Summary.md) identifies earlier dataset/lens provenance; it is not an outstanding instruction to overwrite that provenance.

Original snapshot last updated 2026-11-07 after the R1 gate and the averaged-reference question.

## 1. Status by workstream

### 1a. Production two-axis grid (done)
- Output: `data/distortion_grid_p4_biaxial_macro/distortion_grid.pkl` (1,927,989 rows, 21 accommodations 0–4 D × 101 ADE × 101 BDE × 9 fields) and `failure_map.npz` beside it.
- Log: `data/_logs/p4_macro_21x101x101.log`.
- Ray failures: 102 of 214,221 states (0.05%). Every failure is the near-axis reference ray. The center and nine grid rays never failed. Failed states are NaN in `paraxial_*` and `*_pct` (918 rows); `real_*` and `center_*` are valid.
- Not run: 41 × 401 × 401 (about 92 h on 4 workers, about 13 GB of output).

### 1b. Collector and validation
- `Script/distortion_grid_p4_biaxial_macro.py`: one macro per accommodation, `--workers N` (separate CODE V processes), checkpoints and resume, NaN on ray failure, `CommandTimeout = -1`.
- `Script/validate_p4_biaxial_macro.py`: macro vs slow Python path, all 14 columns. PASS (max coordinate difference about 2.5e-9 µm).
- Slow reference collector `Script/distortion_grid_p4_biaxial.py` is unchanged.
- Measured speed: about 155–205 ms/state per session. Four sessions run at about 2.4–3.2× the serial speed.

### 1c. Joint model, all ADE × BDE states
- Script: `Script/fit_p4_biaxial_joint.py`. Report: `data/p4_biaxial_joint/report.md`.
- Form: barrel B_A (power law + measured s(A)), then projective keystone with 2×2 M and q_x, q_y. Coefficients are polynomials of total order 1–4 in t_x = BDE/20, t_y = ADE/20, times a^0..a^2 with a = A/5.
- Results (vertex RMS / coordinate RMSE / max point, µm):

| Model | Split | Vertex | Coord | Max |
|---|---|---:|---:|---:|
| Center-exact, 252 coef | holdout | 9.03 | 6.38 | 53.2 |
| Offset (translation), 336 coef | holdout | 6.43 | 4.55 | 50.8 |
| Offset | full-grid refit | 6.48 | 4.58 | 56.8 |

- Center-exact is the physically consistent model. The offset model moves the center field off zero (about 5.8 µm coordinate RMSE at center).
- q_x, m12, m21 became identifiable with BDE. Fixing them at 0 raises training cost 104×.
- No overfitting within the sampled range (holdout ≈ in-sample; leave-out of 1.0 and 3.0 D costs nothing).
- Error grows with |BDE|, |ADE| and accommodation. Worst states: 3.8–4 D, |ADE| and |BDE| ≈ 20°.

### 1d. Per-state ceiling (model-form test)
- Script: `Script/p4_biaxial_per_state_ceiling.py`. Report: `data/p4_biaxial_joint/improvement.md`. Log: `data/p4_biaxial_joint/per_state_ceiling_run.log`.
- Current form per-state floor = global fit (9.03 µm, ratio 0.997). Higher polynomial order does not help.
- Per-state floors (in-sample, optimistic): + free k1 8.01 µm; + k2, p1, p2 3.62 µm; all four 3.10 µm (not converged, 51,480 states hit the cap).
- This caveat applies to the older all-accommodation per-state ceiling/main experiment; `main()` was edited after its full run and only `--report-only` ran since. Its full rerun (about 37 min) is still needed. The separate canonical A=0 fine gaze-only fit in section 1k is complete; it does not resolve this older all-A verification gap.

### 1e. Accommodation estimation (gaze known exactly)
- Script: `Script/estimate_p4_accommodation_from_model.py`. Report: `data/p4_biaxial_joint/accommodation_estimation.md`. Results npz beside it.
- Noise-free, interior held-out (0.2–3.8 D), center-exact: RMS 13.1 mD, bias −2.9 mD, 99.7% within 0.1 D. Offset model: RMS 15.3 mD.
- Clipping at the 0–4 D search bounds affects only the 0 and 4 D rows; it does not change the interior numbers.
- Cramér–Rao σ_A ≈ 16 mD per µm of coordinate noise. Under 1 µm independent coordinate noise, prior Monte Carlo total RMS was 23.6 mD for the center-exact model and 25.3 mD for the offset model; noise dominates for typical states.
- Corners carry most of the information; the center carries none. A single field is unusable. Four corners give about 80 mD.
- Gaze error, calibration error and lens-to-lens variation are not modelled.

### 1f. Power law (barrel)
- Refit on 0–4 D: c0 = −0.0185005, c1 = 0.00020590, n = 1.4336. The old values were 0–5 D.
- Error in k1 under 0.15% at every accommodation. Position cost about 0.03 µm at 4 D.
- Most of the zero-rotation residual comes from s(A) (measured paraxial scale), not from k1.

### 1g. On-axis real-fitted barrel (ADE = BDE = 0, all accommodations)
- Script: `Script/p4_onaxis_accommodation_model.py`. Report: `data/p4_onaxis/report.md`. Results npz and log beside it.
- Fitting s(A) and k(A) to the real coordinates (not paraxial values) cuts the 4 D error from 0.59 µm to about 0.02 µm (leave-one-out).
- Recommended formula (x = A/4, A in D, coordinates in mm):
  - `u0 = D_c0⁻¹(r0)` (radial Newton inverse of u(1 + c0|u|²)), with `c0 = −0.0185004967`
  - `s(A) = 1 − 0.081595476571·x`
  - `k(A) = c0 + 2.3989235714e-4·x + 1.5756063948e-3·x² − 1.4256378340e-3·x³ + 6.6200789981e-4·x⁴`
  - `real(A) = (1 + k(A)·s(A)²·|u0|²)·s(A)·u0`
- Accommodation estimation with this model: noise-free LOO RMS about 0.6 mD.
- Limits: anchored to the A = 0 real grid r0, which exists only at 9 field points. Only two independent radii per accommodation. s and k depend on the gauge k0; use the k(0) = c0 form.
- Open: a kink at 0.2 D (second difference about 8× the median). The `A^0.81` in the lens model is the semi-aperture law, not part of the distortion fit. Unresolved; treat as caveat.

### 1h. Historical A = 0 D, BDE = 0, ADE sweep (keystone K)
- Script: `Script/fit_p4_ade_fixed_accommodation.py`. Report: `data/p4_ade_fixed_accommodation/report.md`. Plot and npz beside it.
- Model: X = s_x·x0/(1+q·y0), Y = s_y·y0/(1+q·y0), with (x0, y0) = real grid at ADE = 0. q in 1/mm (q_n = q·L, L = 1.235429592627 mm).
- Vertex RMS (µm): no keystone 80.8; per-ADE free fit 2.65 (ceiling); order-4 polynomial in t = ADE/20: 2.65 in-sample, 2.56 holdout (max 10.0). Order 2 gives 3.2 holdout.
- Coefficients (t = ADE/20): q_n = −0.063289·t + 0.008847·t³; s_x = 1 + 0.060561·t² + 0.001794·t⁴; s_y = 1 + 0.125021·t² − 0.004599·t⁴.
- Parity holds (q_n odd, s_x and s_y even).
- Under 1 µm vertex RMS up to |ADE| ≈ 12°; under 1 µm max point up to about 8°.
- Residual shape: the K form cannot put the x-axis edge midpoints at Y ≠ 0, and there is a common Y offset of about 4 µm at ADE = 20°.

### 1i. Historical A = 0 D, BDE = 0, decentering test
- Script: `Script/fit_p4_ade_a0_decenter.py`. Report: `data/p4_ade_a0_decenter/report.md`.
- Per-state ceilings (vertex RMS, µm): K only 2.65; + Brown–Conrady p1 1.36 (4 params); + Y-only smile 1.92; + Y-only general 1.30 (5 params, ill-conditioned near ADE ±10°).
- p2 comes out at 1e-13 (required by x-mirror symmetry).
- p1 is 0.95 correlated with q, so the two trade off.
- Decisive check failed: the Y error at the x-axis edge midpoints does not collapse (+3.32 → −3.18 µm at ADE = 20°). p1 is not the complete missing term.
- Polynomial with p1 = 0.0014017·t − 0.0057599·t³ (8 coefficients): holdout 1.33 µm vertex, 4.09 µm max. Under 1 µm vertex to about 12°, max to about 8.4°.

### 1j. Fine A = 0 biaxial collection (completed)
- Dedicated runner: `Script/distortion_grid_p4_a0_biaxial_macro.py`; output: `data/distortion_grid_p4_biaxial_macro_a0_401x401/`. It is separate from and does not overwrite the production 21 x 101 x 101 all-accommodation dataset.
- P4 only: loaded `Lens/p4_ME.len` unchanged; the runner sets gaze through `ADE` and `BDE` on the existing `RC` surface. The P4 model has no `SRC_ROT` label; no P1 rotation assumptions or optical changes were applied.
- Four persistent CODE V workers completed 81 BDE-row chunks (block size 5), retaining each chunk CSV and SEQ plus checkpoints. Grid: A=0 D, Z=0, ADE and BDE each -20..+20 deg at 0.1 deg (401 x 401 = 160,801 gaze states); final pickle has 1,447,209 nine-field records. Runtime was 2 h 21 min 40 s, ending 2026-11-06 17:42:47 EST. No CODE V processes remain.
- Artifacts: `distortion_grid.pkl`, `failure_map.npz`, `manifest.json`, retained `csv/` and `macros/` within the output folder. Final pickle SHA-256: `a30dd918c1ddb282c380756421dd55195f85c0eef0ce519db5c5211892066445`; lens SHA-256 `c4b7780e582a29860ea58aaf554710a9d21c50a8dfbcde96283280f605e8db73`; full provenance is in `manifest.json`.
- Status/NaN accounting: 20 failed ray statuses (17 near-axis probe failures, 3 grid rays, 0 center rays). Raw data contain 153 paraxial-coordinate NaN records and 3 real-coordinate NaN records in 3 gaze states; 156 rows have a radial/tangential/combined NaN. Rows and failure statuses are preserved. The overlap check against the old A=0 coarse slice passed for 10,201 states / 91,809 records (max coordinate delta 4.0e-9 um, max pct delta 2.83e-10).
- The proposed 41-accommodation x 401 x 401 all-A collection was not run. The original 21 x 101 x 101 production pickle remains unchanged.

### 1k. Fine A = 0 biaxial gaze-only fit (completed; replaces the paused A=0 direction)
- Runner: `Script/fit_p4_ade_a0_decenter.py --biaxial --input ... --output-dir ... --coarse-model ... --batch-size 8192`. Canonical output: `data/p4_ade_a0_decenter_fine401/` (`biaxial_report.md`, `p4_ade_a0_biaxial.npz`, `p4_ade_a0_biaxial_heatmaps.png`, `p4_ade_a0_biaxial_residuals.png`). The previous 101-state ADE-only report and files in `data/p4_ade_a0_decenter/` are preserved. A preliminary run is archived separately at `data/p4_ade_a0_decenter_fine401_preliminary_1AF4/`; use only the canonical folder for results.
- Fit source SHA-256: `1C049660372FE087F9AF8AD76CFA3C2D3D742745B10F9F112DD69E042D087DB4`; input A=0 pickle hash is listed in the fine report.
- Reference/target: use the measured real-ray zero-gaze field `b_j=(z_00,j-c_00)/L`, `L=1.235429592627 mm`, and per-gaze centered target `(z_gj-c_g)/L`; measured centers are supplied at reconstruction. Center displacement is effectively zero at stored precision. This fits centered gaze distortion, not a gaze-dependent absolute center law.
- Model order is H -> Brown -> radial: `h=M b/(1+qx bx+qy by)`; Brown-Conrady maps h to w with p1,p2; the final family is `f=(1+g_r2*||w||^2)w`. Families have 6/8/9 per-state parameters. Each coefficient uses all 15 total-degree <=4 monomials in `tx=BDE/20`, `ty=ADE/20`, no parity constraints. Train on even/even indices (40,401 valid states) and test on the other valid states (120,397).
- Exclude exactly 3 whole states with incomplete real coordinates from all fit/metric masks; retain all 17 probe/paraxial-only failure states whose real coordinates are finite. Valid total is 160,798 states. These are held-out gaze interpolation tests within +/-20 deg on the same nine sampled fields, not validation at unseen field positions or outside the gaze range.
- Heldout test on the eight noncentral fields (xRMSE / yRMSE / vertex RMS / max point, um):

| Model | Fine refit | Saved coarse coefficients on same fine test |
|---|---:|---:|
| Projective H | 4.2079 / 4.1953 / 5.9420 / 28.6357 | 4.2070 / 4.1964 / 5.9421 / 28.5946 |
| H + Brown | 2.4598 / 2.4637 / 3.4815 / 16.9072 | 2.4604 / 2.4634 / 3.4816 / 16.8809 |
| H + Brown + radial r2 | 1.4857 / 1.5235 / 2.1280 / 11.7598 | 1.4864 / 1.5232 / 2.1283 / 11.6340 |

- The fine refit is essentially tied to the coarse coefficient baseline (0.014% vertex-RMS improvement for H+Brown+r2; its max point error is slightly higher). The report also gives train, all-valid, and comparable all-nine metrics. Independent reconstruction, masks, convergence, denominator, nesting, NPZ consistency, UTF-8 report, and figures passed audit.
- The earlier paused A=0 specification is superseded by this measured-reference fine gaze-only study; not every proposed old ceiling/bin/correlation extraction was separately produced. The older all-A per-state-ceiling caveat in section 1d remains.

### 1l. C0 and C1 combined model results (completed, this session)
- C0 zero-gaze composition check PASSED: C0 equals B_A(r0) to round-off; C0 vs measured zero-gaze grid is at most 0.051 um max over A = 0 to 4 D. Source: data/p4_combined_brown/report_c0_verified.md.
- C0 nonzero-gaze error is UNEXPLAINED: held-out 8-field vertex RMS 66.05 um, max 505.9 um at 4 D. The frozen-gaze test that would settle it has NOT run (two low-tier attempts stopped; the forward function was not available in the low-tier scope).
- Shared-coefficient structure: ONE shared set of coefficients serves all gaze states and all accommodations. There are no per-state coefficients in C0 or C1. Per-state parameters exist only in the per-state benchmark (an upper bound, not a model).
- C1 coefficient counts: 9 parameters x 14 gaze monomials x (order + 1). Order 1: 252; order 2: 378; order 3: 504. Source: data/p4_combined_brown/report.md, report_c0_verified.md.
- C1 results (held-out, 7,600 gaze states per A, 8 fields unless stated):
  - C1 order 2 (recommended): vertex RMS 3.33 um, max 26.29 um. All nine fields: vertex RMS 3.14 um, max 26.29 um.
  - C1 order 3: vertex RMS 3.33 um, max 26.43 um.
  - C1 order 1: vertex RMS 3.61 um, max 22.82 um (leave-one-accommodation-out selection values).
  - Joint model reference (252 coefficients, unconstrained, 9 fields): vertex RMS 9.03 um, max 53.2 um.
- Floors (per-state, upper bound): forward (B_A then gaze) 3.28 um; reverse (gaze then B_A) 3.84 um. Floor/C1 ratio 0.98 to 0.99 at every A. The floor rises from 2.14 um at 0 D to 4.43 um at 4 D.
- Gauge: identity gauge at A = 0 costs nothing: 2.15501 um vs stored 2.1552 um.
- Stored A = 0 baseline (use as reference, do not recompute): data/p4_ade_a0_decenter/biaxial_report.md lines 25, 37, 49. 101 x 101 held-out 8-field vertex RMS: projective 6.0418, projective+Brown 3.5227, Brown+radial r2 2.1552 um; max point 27.8451, 16.6221, 11.3488 um. Split: 2,601 training, 7,600 held-out.
- Fine 401 x 401 numbers are NOT used for comparison (the user said they do not show a significant difference).

### 1m. R1 radial merge (completed, gate FAILED)
- Formula: r = L*(1 + kappa*|w|^2)*w, where kappa = k(A)*L^2 + g(t), w = Brown(H(s(A)*u0/L)), u0 = D_c0^-1(r0), with t_x = BDE/20, t_y = ADE/20. Source: data/p4_r1_radial_merge/report.md.
- Symmetric gaze basis, 33 terms, total degree 1 to 4, no constant: m11, m22, g even-even (5 terms each: x^2, y^2, x^4, x^2y^2, y^4); m12, m21 odd-odd (3 each: xy, x^3y, xy^3); qx, p2 odd-even (3 each: x, x^3, xy^2); qy, p1 even-odd (3 each: y, x^2y, y^3). m11 and m22 include a constant 1. Symmetry check passed on the full grid (both flips).
- Zero-gaze check: R1 equals B_A(r0) to 2.2e-13 um at every A.
- A = 0 gate: R1 8-field held-out vertex RMS 2.2260 um, max 9.857 um, 9-field vertex RMS 2.0987 um. Same-basis current-chain control: 2.1550 um, max 11.531 um, 9-field 2.0318 um. The gate allowed at most 0.05 um over 2.1552 um; R1 failed by 0.071 um.
- Step 2 (R1-A0 at all A, S0-frozen, R1-all) was NOT run because the gate failed.
- Archive: data/p4_r1_radial_merge/step1_a0.npz. Implementation: Script/fit_p4_r1_radial_merge.py. Job: task-mw6qgvtj-wvhl9g.

## 2. Next investigation

- Open question: a real system cannot measure zero gaze and 0 D exactly. The protocol takes an averaged reference under fixation, with gaze and accommodation drifting, and assumes gaze = 0 and A = 0.
- Two errors follow. (1) Spread: averaging over drift is not the image at the mean state; even terms leave a bias that grows as sigma^2. (2) Mean offset: the reference sits at the true mean gaze and accommodation, not at zero; transforms do not commute, so the offset is first order.
- Planned test (not launched; waiting for the user's yes): (a) commutation test on stored coefficients; (b) closed-form spread bias; (c) end-to-end check of control and R1 with the averaged reference. Sweep ranges and report where errors cross 0.5 um and 1 um.
- Literature reference: Ronda, J. I., and Valdes, A. (2019). Geometrical Analysis of Polynomial Lens Distortion Models. Journal of Mathematical Imaging and Vision, 61(3), 252-268. https://oa.upm.es/64244/ ; arXiv 1804.03584 ; https://doi.org/10.1007/s10851-018-0833-x. Read from abstract only; full text not checked.

## 3. Decisions and unresolved items

1. Lens provenance: current hash c4b7780e...; Summary.md still records fb3937e6...; do not treat as identical.
2. Reference handling for the real system: the averaged-reference test in section 2 must run before choosing a model form. Ask the user for typical fixation spread and mean offset only if needed; the test does not block on them.
3. Model choice: C1 order 2 (378 coefficients) is the recommended fitted model. The 33-term symmetric control reproduces the A = 0 baseline. R1 is not yet validated across accommodation. The C0 nonzero-gaze error is open.
4. Symmetry reduction to fewer than 50 coefficients: the user requires it; not yet delivered as a fitted model.
5. Accuracy criterion: 1 um is a measurement-reference line, not a specification.
6. On-axis 0.2 D kink: unresolved; caveat.
7. Full 41 x 401 x 401 accommodation grid: not run.

## 4. Known risks

- The old slow collectors use plain `shutil.rmtree` at publish. On OneDrive this can fail with WinError 5. Only the macro collector handles it.
- Hard-killing Python leaves CODE V sessions running and holding licences. Use Ctrl+C.
- CODE V writes `codev*.rec` files into `C:\CODEV202403_SR1` for every COM session. We do not clean these.
- `Lens/p4_ME.1.len` is untracked and was not created by this work. Left alone.
- `Script/__pycache__` shows as modified in git. Harmless.
- Unverified: `p4_biaxial_per_state_ceiling.py` `main()` has not been run end to end since its last edit.
- Accommodation estimation assumes gaze is known exactly.
- Scratch prompt files: eight files named .scratch_*.md sit in the project root and must be deleted. Names: .scratch_codex_amend.md, .scratch_codex_c0_fix.md, .scratch_codex_combined_prompt.md, .scratch_codex_final_amend.md, .scratch_codex_plan_impl.md, .scratch_codex_r1.md, .scratch_luna_frozen_check.md, .scratch_luna_frozen_check2.md.
- Codex job cancellations and failures earlier in this session are logged in the job records; not all produced output.

## 5. Workflow notes

- Coordinator only. Every plan goes to the advisor first.
- Implementation goes to gpt-6.1-sol at xhigh, launched with --fresh unless a resume-by-ID is verified.
- Mechanical work (reads, commands, file edits, diffs, deletions, reruns) goes to gpt-6-luna at low.
- Audit goes to the advisor before any number is reported.
- Reports to the user are written in ASD-STE100.
- Earlier jobs ran on other tiers or without advisor planning. Each report says so.

## 6. Key file index

| Purpose | Path |
|---|---|
| Production grid | `data/distortion_grid_p4_biaxial_macro/distortion_grid.pkl` |
| Failure map | `data/distortion_grid_p4_biaxial_macro/failure_map.npz` |
| Collector | `Script/distortion_grid_p4_biaxial_macro.py` |
| Validator | `Script/validate_p4_biaxial_macro.py` |
| Joint model | `Script/fit_p4_biaxial_joint.py`, `data/p4_biaxial_joint/report.md` |
| Model-form ceiling | `Script/p4_biaxial_per_state_ceiling.py`, `data/p4_biaxial_joint/improvement.md` |
| Accommodation estimation | `Script/estimate_p4_accommodation_from_model.py`, `data/p4_biaxial_joint/accommodation_estimation.md` |
| On-axis real-fitted barrel | `Script/p4_onaxis_accommodation_model.py`, `data/p4_onaxis/report.md` |
| A = 0 keystone (ADE) | `Script/fit_p4_ade_fixed_accommodation.py`, `data/p4_ade_fixed_accommodation/report.md` |
| A = 0 decentering | `Script/fit_p4_ade_a0_decenter.py`, `data/p4_ade_a0_decenter/report.md` |
| A = 0 fine biaxial collector and dataset | `Script/distortion_grid_p4_a0_biaxial_macro.py`, `data/distortion_grid_p4_biaxial_macro_a0_401x401/` |
| A = 0 fine biaxial gaze fit | `data/p4_ade_a0_decenter_fine401/biaxial_report.md`, `data/p4_ade_a0_decenter_fine401/p4_ade_a0_biaxial.npz` |
| Macro log | `data/_logs/p4_macro_21x101x101.log` |
| Theory | `Theory.md` |
| Data provenance | `Summary.md` |
