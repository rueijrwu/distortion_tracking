# P4 model record

## Scope and geometry

All P4 model results and numerical parameters in this record are for **167.93 mm, zero working distance**, from the historical P4 full-detail trace. They do not describe the file-thickness geometry. The current input path is `data\p4_raytrace\p4_raytrace_251x251_12p5_A4.pkl` (524,441,352 bytes); its hash and contents are unverified. The previous input had SHA-256 `aceb43c9744dec4898df558d4e21ab786dede141e464a47303b1dd13c0892c03`: 214,221 states × 9 rays, A=0–4 D at 0.2 D, ADE/BDE=−12.5°–12.5° at 0.25°, and a 3×3 field at 20° on each axis. Those dimensions and zero ray statuses describe the historical input only. The unchanged P4 validators still require that 21×101×101×9 lattice; the current `251x251`/`A4` filename may indicate incompatible dimensions or a single A=4 D slice. H1/H2/H3 are repeatedly used validation strata, not blind tests; no field holdout exists.

The trace set `Cornea_ENT_D` thickness to zero. `runs\p4_geometry_check\z13_readback.txt` records S13 at z=167.9259434838 mm from the S16 stop origin. The lens file value is 334.8586486894943 mm. The `.len`/`.seq` match is user-confirmed but was not independently verified because the `.len` is binary. No measurement or validated prediction at 334.86 mm exists; any transport there is model-only and unverified. The earlier proposal for a 50-state CODE V pilot at file thickness was not approved or run.

## Status and current model pointer

The current accepted P4 model is the E2 pinhole plus radial k1/k2 camera linear in accommodation, validated over the full traced A=0–4 D range at **167.93 mm, zero working distance**. Its canonical equations, coefficients, validation tables, and interpretation are maintained in `CURRENT_MODEL.md`, the current source of truth. The earlier plan to settle a physical A=0 eye model before restoring accommodation is historical and superseded by this validated camera result. The user rejected the old empirical and mirror eye-model directions documented below; none is an accepted physical eye model.

The stop tangent is `t=(-l16/n16,-m16/n16)`. Rays pass through the S16 origin and the camera is fixed after S16. In the all-A camera fit, E0/E1/E2 holdout maxima were 25.44/0.502/0.0175 µm; E3/E4 matched E2 by symmetry. The E1 residual was outer-tangent radial error, addressed by k2. An audited per-accommodation E2 report (five A values, run `p4_camera_per_accom_20261110T235949Z_ea13bf95`) found each own-fit holdout maximum 0.01405–0.01493 µm and the A=0 E2 camera at every A no worse than 0.0149 µm; the maximum drift gate was 0.0016 µm against a 0.01 µm limit. This supports camera invariance over 0–4 D at **167.93 mm, zero working distance**. That report notes its run used a different script hash than the current file, so the subsequent full validation below supplies the reproducible canonical result.

### Linear E2 accommodation-law source note

The six-parameter law first appeared as a historical proposal in `p4_model_proposal.md` and was subsequently validated by canonical full run `runs\p4_camera_linear_accom_20261111T011655Z_09cb5314`; V1–V6 passed over all 21 A levels, with 16 interpolation levels and no quadratic fallback. The accepted equations, coefficients, detailed errors, and interpretation are recorded only in `CURRENT_MODEL.md` to keep one current-model source of truth.

## Historical fitted models

### Nested empirical eye model

At **167.93 mm, zero working distance**, run `runs\p4_fit_20261110T153705Z_4f23532d` selected camera E2 plus equivariant S7 eye mapping with a degree-2 Legendre accommodation law: 107 parameters total. Composite maxima were H1 0.716 µm, H2 0.698 µm, H3 0.611 µm, and train 0.811 µm, within the 1 µm rule. It was superseded as a direction because of parameter count and the later user decision to pursue a physical A=0 model.

### Compact rational direction-cosine surrogate

This empirical surrogate maps gaze, field, and accommodation to the measured stop tangent, then uses a separately fitted camera. It is not a first-principles eye model, anatomical-distance estimate, or formula copied from a cited paper. Let `g` be gaze, `phi` field, `u=g·g`, `v=g·phi`, `w=phi·phi`; vector numerator terms use the O(2)-equivariant basis `(a*g+b*phi)u^i v^j w^k`, and scalar invariant terms form a shared denominator. The unnormalised outgoing ray `(T*Nx,T*Ny,-D)` gives `t=T*N/ D`; `D>0` enforces the observed negative z direction. The denominator intercept is fixed at 1 to remove scale ambiguity. Accommodation uses Legendre polynomials in `(A−2)/2`. These symmetry and low-order expansions are empirical hypotheses.

At **167.93 mm, zero working distance**, canonical run `runs\p4_compact_20261110T160451Z_a2061018` selected camera E1 plus eye p=2, q=2 and common accommodation degree 2: 57 retained numerator coefficients + 27 denominator coefficients + 4 camera coefficients = 88. Sensor holdout RMS was 0.097911 µm; max 0.955176 µm (H1/H2/H3: 0.851112/0.955176/0.914225 µm; per-A maximum 0.955176 µm). Traced denominator minimum was 0.402744. A 100,000-point off-grid pole check over the traced A, gaze and source-angle ranges found minimum 0.486871 and no nonpositive denominator; this is a pole check, not an off-grid accuracy test. The rational-tangent Jacobian rank was 84/84, column-scaled condition number 262,364. Four Sanathanan–Koerner passes were not fully converged (last relative L2 step 5.78e−4), so global optimality is not established. E1 camera parameters were f=425.143461 mm/tangent and k1=2.855657 tangent⁻², with principal point numerically zero. A 71-coefficient per-angular-order cap extension had 1.09146 µm max and failed the 1 µm criterion. Two separately reconstructed held-out states (18 rays) differed by at most 1.76e−6 in tangent and 0.832 µm on sensor. No field holdout exists; no off-grid accuracy is claimed. The cited camera/zoom papers motivate structure but do not validate P4 eye physics. This direction is abandoned.

### Fused effective zoom lens

This is one fused effective map from gaze, source field, and accommodation directly to sensor XY; it does not fit eye and camera maps separately. Its pupil/focal parameters are effective quantities, not anatomical distances or a recovered physical stop. For `R=Rx(ADE)Ry(BDE)`, source direction `s=(phi_x,phi_y,sqrt(1−|phi|²))`, local `l=Rᵀs`, `h=l_xy/l_z`, `r²=h·h`, and `tau=(A−2)/2`, the proposed family is:

```text
q(A)     = 1 + q1 P1(tau) + q2 P2(tau) + q3 P3(tau)
Delta(A) = ax - an = d0 P0(tau) + d1 P1(tau) + d2 P2(tau)
mu(A)    = sigma * (1 - Delta(A)/q(A))
F(A)     = F0 P0(tau) + F1 P1(tau) + F2 P2(tau) + F3 P3(tau)
hd       = mu(A) * (1 + k1(A) r^2 + k2(A) r^4) * h
v        = R (hd_x, hd_y, 1)
sensor   = F(A) * (v_x/v_z, v_y/v_z) mm
```

The intercept q0=1 fixes q scale; sigma is discrete ±1. Principal point and extra sensor tilt are fixed to zero, with gaze rotation supplying the effective lens tilt. F is distance normal to the fixed sensor at reference zero gaze; the tilted projective denominator includes the optical-axis intersection shift. This was structurally analogous to a zoom-camera model, with accommodation used as zoom; the analogy does not make P4 the cited camera system.

At **167.93 mm, zero working distance**, run `runs\p4_effective_zoom_20261110T164337Z_ed34cd66` fit raw sensor coordinates from the full training split (28,611 states/257,499 rays), with analytic-Jacobian nonlinear least squares. Both parities were warm-fit; costs differed by only 9.6e−9 mm², so parity was unidentifiable and +1 chosen deterministically. q coefficients were bounded to [−0.3,0.3]. Tested families had 10–18 parameters; the best H1/H2/H3 maxima were still 572.893/561.491/551.843 µm for the selected 16-parameter k1/k2-quadratic family. No family passed the 1 µm criterion. For the selected family F ranged −38.0423 to −36.5746 mm and mu 0.19797–0.19851; Jacobian rank 16/16, scaled condition 4.10e6, q minimum 0.9306, and minimum absolute vz 0.9431 on the full trace. The fit converged in five evaluations; analytic Jacobian finite-difference discrepancy was 2.09e−9. H strata were reused for comparison, with no field holdout. The failure establishes a large mismatch for this proposed structure; its dominant cause was not isolated. Do not add an unplanned generic correction. This direction is abandoned.

### Physical, single-system, and mirror experiments

At **167.93 mm, zero working distance**, the physical full-prescription reference R0 (nL=1.4092, no eye fit) had H1 max 1,595 µm. Homogeneous-index P1 with linear nL(A) had H1 max 87.2 µm/RMS 24.6 µm; P2 adding A^0.81 had 87.8 µm and no gain. P3 adding dR,dK0,dK1 was the best all-A physical-family fit (train max/RMS 9.75/2.42 µm; H1/H2/H3 max 9.55/9.18/8.98 µm), but those extra parameters are empirical, not physical. Single-system diagnostics were also poor: fixed-A0 F0 H1 max 544 µm; F0 linear in A H1/H2/H3 933/935/913 µm; F1 cubic at A=0/2/4 H1 maxima 39.4/46.2/65.3 µm; F1 linear in A H1/H2/H3 67.7/65.6/64.7 µm. The P3 residual was attributed to replacing the gradient-index lens by a homogeneous index, but that attribution is inferred, not proven.

At **167.93 mm, zero working distance**, all-A rotating conic mirror fitting failed Newton convergence for 8,208 rays at held A=3.0; cause undiagnosed (`runs\p4_mirror_20261110T221731Z_735febf6`). The A=0 ladder (`runs\p4_mirror_a0_20261110T225100Z_cf7bb24e`) gave S0 sphere max/RMS 535.3/91.0 µm, S1 plus conic K 76.8/26.8 µm, S2 plus b4 77.2/26.7 µm, and S3 plus refracting cornea 6.88/0.94 µm. S3 fit n_e=1.448, not a physical corneal index, with strong parameter correlations (r–K 0.99, r–n_e −0.96); verdict was “Base model not sufficient.” One conic K could not meet both gaze-cubic response and edge/corner gain. Cornea refraction reduced error but did not establish the fixed-physics model. These are historical diagnostics, not accepted eye models.

## Reproduction and source coverage

The compact and effective implementations are `Script\p4_compact_model.py` and `Script\p4_effective_zoom.py`; their canonical run summaries retain coefficients and diagnostics. The validated linear E2 camera run is `runs\p4_camera_linear_accom_20261111T011655Z_09cb5314`; see `CURRENT_MODEL.md` for its accepted definition and metrics. This record consolidates the substantive historical method, result, and limitation content of `p4_compact_model.md` and `p4_effective_zoom_model.md`. It also folds in the key historical result/status tables from `p4_fit_plan.md`, `p4_model_proposal.md`, `p4_peraccom_e2_spec.md`, `p4_peraccom_e2_results.md`, and `p4_a0_e2_spec.md`; those source documents remain intact. `Summary.md` and `Theory.md` contain broader P1 and historical distortion work not reproduced here.
