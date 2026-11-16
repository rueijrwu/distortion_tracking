# P4 model record

**Current handoff, 2026-11-16: goal closed; FL is the accepted default and F/quadratic is the fallback or future option.** The user accepts approximately 1 arcmin FL gaze error. Current theory is [P4_MODEL.md](P4_MODEL.md), procedures/constants/results are [P4_IMPLEMENTATION.md](P4_IMPLEMENTATION.md), and the current documentation contract is `CURRENT_PLAN.md` Section 17. The dated closure at the end records the completed comparison and distinguishes the unchanged strict `CLOSE: F` verdict from this later user acceptance. Earlier model/status/protocol statements below are historical unless explicitly restated in that closure.

## Scope and geometry

All P4 model results and numerical parameters in this record are for **167.93 mm, zero working distance**, from the historical P4 full-detail trace. They do not describe the file-thickness geometry. The then-current historical input was named `data\p4_raytrace\p4_raytrace_251x251_12p5_A4.pkl` (524,441,352 bytes); its hash and contents were unverified in that review. The preceding input had SHA-256 `aceb43c9744dec4898df558d4e21ab786dede141e464a47303b1dd13c0892c03`: 214,221 states × 9 rays, A=0–4 D at 0.2 D, ADE/BDE=−12.5°–12.5° at 0.25°, and a 3×3 field at 20° on each axis. Those dimensions and zero ray statuses describe the historical input only. The older validators required that 21×101×101×9 lattice; the `251x251`/`A4` filename raised an unresolved compatibility question at the time and did not establish its contents. These names and concerns are historical, not current execution prerequisites. The current FL/F input requirement is the 21 full 5x5 shards plus manifest defined in `P4_IMPLEMENTATION.md` Section 2. H1/H2/H3 were repeatedly used validation strata, not blind tests; no field holdout exists.

The trace set `Cornea_ENT_D` thickness to zero. The historical geometry-check readback records S13 at z=167.9259434838 mm from the S16 stop origin. The lens file value is 334.8586486894943 mm. The `.len`/`.seq` match is user-confirmed but was not independently verified because the `.len` is binary. No measurement or validated prediction at 334.86 mm exists; any transport there is model-only and unverified. The earlier proposal for a 50-state CODE V pilot at file thickness was not approved or run.

## Historical E2 camera status

The then-accepted E2 pinhole plus radial k1/k2 camera linear in accommodation was validated over the traced A=0–4 D range at **167.93 mm, zero working distance**. Its former canonical document, `CURRENT_MODEL.md`, is obsolete and absent; it is not a current source or prerequisite. This paragraph preserves the historical camera result, not today's estimator selection. The earlier physical A=0 and mirror/empirical eye-model directions remain historical. Current FL theory and fixed camera definition are consolidated in `P4_MODEL.md` and `P4_IMPLEMENTATION.md`.

The stop tangent is `t=(-l16/n16,-m16/n16)`. Rays pass through the S16 origin and the camera is fixed after S16. In the all-A camera fit, E0/E1/E2 holdout maxima were 25.44/0.502/0.0175 µm; E3/E4 matched E2 by symmetry. The E1 residual was outer-tangent radial error, addressed by k2. An audited per-accommodation E2 report (five A values, run `p4_camera_per_accom_20261110T235949Z_ea13bf95`) found each own-fit holdout maximum 0.01405–0.01493 µm and the A=0 E2 camera at every A no worse than 0.0149 µm; the maximum drift gate was 0.0016 µm against a 0.01 µm limit. This supports camera invariance over 0–4 D at **167.93 mm, zero working distance**. That report notes its run used a different script hash than the current file, so the subsequent full validation below supplies the reproducible canonical result.

### Linear E2 accommodation-law source note

The six-parameter law first appeared as a historical proposal in `p4_model_proposal.md` and was subsequently validated by canonical full run `p4_camera_linear_accom_20261111T011655Z_09cb5314`; V1–V6 passed over all 21 A levels, with 16 interpolation levels and no quadratic fallback in that historical camera experiment. Its former `CURRENT_MODEL.md` source is obsolete and absent. This history is distinct from the accepted FL estimator and its F fallback; current camera constants are stated directly in `P4_IMPLEMENTATION.md`.

## Historical fitted models

### Nested empirical eye model

At **167.93 mm, zero working distance**, run `p4_fit_20261110T153705Z_4f23532d` selected camera E2 plus equivariant S7 eye mapping with a degree-2 Legendre accommodation law: 107 parameters total. Composite maxima were H1 0.716 µm, H2 0.698 µm, H3 0.611 µm, and train 0.811 µm, within the 1 µm rule. It was superseded as a direction because of parameter count and the later user decision to pursue a physical A=0 model.

### Compact rational direction-cosine surrogate

This empirical surrogate maps gaze, field, and accommodation to the measured stop tangent, then uses a separately fitted camera. It is not a first-principles eye model, anatomical-distance estimate, or formula copied from a cited paper. Let `g` be gaze, `phi` field, `u=g·g`, `v=g·phi`, `w=phi·phi`; vector numerator terms use the O(2)-equivariant basis `(a*g+b*phi)u^i v^j w^k`, and scalar invariant terms form a shared denominator. The unnormalised outgoing ray `(T*Nx,T*Ny,-D)` gives `t=T*N/ D`; `D>0` enforces the observed negative z direction. The denominator intercept is fixed at 1 to remove scale ambiguity. Accommodation uses Legendre polynomials in `(A−2)/2`. These symmetry and low-order expansions are empirical hypotheses.

At **167.93 mm, zero working distance**, canonical run `p4_compact_20261110T160451Z_a2061018` selected camera E1 plus eye p=2, q=2 and common accommodation degree 2: 57 retained numerator coefficients + 27 denominator coefficients + 4 camera coefficients = 88. Sensor holdout RMS was 0.097911 µm; max 0.955176 µm (H1/H2/H3: 0.851112/0.955176/0.914225 µm; per-A maximum 0.955176 µm). Traced denominator minimum was 0.402744. A 100,000-point off-grid pole check over the traced A, gaze and source-angle ranges found minimum 0.486871 and no nonpositive denominator; this is a pole check, not an off-grid accuracy test. The rational-tangent Jacobian rank was 84/84, column-scaled condition number 262,364. Four Sanathanan–Koerner passes were not fully converged (last relative L2 step 5.78e−4), so global optimality is not established. E1 camera parameters were f=425.143461 mm/tangent and k1=2.855657 tangent⁻², with principal point numerically zero. A 71-coefficient per-angular-order cap extension had 1.09146 µm max and failed the 1 µm criterion. Two separately reconstructed held-out states (18 rays) differed by at most 1.76e−6 in tangent and 0.832 µm on sensor. No field holdout exists; no off-grid accuracy is claimed. The cited camera/zoom papers motivate structure but do not validate P4 eye physics. This direction is abandoned.

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

At **167.93 mm, zero working distance**, run `p4_effective_zoom_20261110T164337Z_ed34cd66` fit raw sensor coordinates from the full training split (28,611 states/257,499 rays), with analytic-Jacobian nonlinear least squares. Both parities were warm-fit; costs differed by only 9.6e−9 mm², so parity was unidentifiable and +1 chosen deterministically. q coefficients were bounded to [−0.3,0.3]. Tested families had 10–18 parameters; the best H1/H2/H3 maxima were still 572.893/561.491/551.843 µm for the selected 16-parameter k1/k2-quadratic family. No family passed the 1 µm criterion. For the selected family F ranged −38.0423 to −36.5746 mm and mu 0.19797–0.19851; Jacobian rank 16/16, scaled condition 4.10e6, q minimum 0.9306, and minimum absolute vz 0.9431 on the full trace. The fit converged in five evaluations; analytic Jacobian finite-difference discrepancy was 2.09e−9. H strata were reused for comparison, with no field holdout. The failure establishes a large mismatch for this proposed structure; its dominant cause was not isolated. Do not add an unplanned generic correction. This direction is abandoned.

### Physical, single-system, and mirror experiments

At **167.93 mm, zero working distance**, the physical full-prescription reference R0 (nL=1.4092, no eye fit) had H1 max 1,595 µm. Homogeneous-index P1 with linear nL(A) had H1 max 87.2 µm/RMS 24.6 µm; P2 adding A^0.81 had 87.8 µm and no gain. P3 adding dR,dK0,dK1 was the best all-A physical-family fit (train max/RMS 9.75/2.42 µm; H1/H2/H3 max 9.55/9.18/8.98 µm), but those extra parameters are empirical, not physical. Single-system diagnostics were also poor: fixed-A0 F0 H1 max 544 µm; F0 linear in A H1/H2/H3 933/935/913 µm; F1 cubic at A=0/2/4 H1 maxima 39.4/46.2/65.3 µm; F1 linear in A H1/H2/H3 67.7/65.6/64.7 µm. The P3 residual was attributed to replacing the gradient-index lens by a homogeneous index, but that attribution is inferred, not proven.

At **167.93 mm, zero working distance**, all-A rotating conic mirror fitting failed Newton convergence for 8,208 rays at held A=3.0; cause undiagnosed (`p4_mirror_20261110T221731Z_735febf6`). The A=0 ladder (`p4_mirror_a0_20261110T225100Z_cf7bb24e`) gave S0 sphere max/RMS 535.3/91.0 µm, S1 plus conic K 76.8/26.8 µm, S2 plus b4 77.2/26.7 µm, and S3 plus refracting cornea 6.88/0.94 µm. S3 fit n_e=1.448, not a physical corneal index, with strong parameter correlations (r–K 0.99, r–n_e −0.96); verdict was “Base model not sufficient.” One conic K could not meet both gaze-cubic response and edge/corner gain. Cornea refraction reduced error but did not establish the fixed-physics model. These are historical diagnostics, not accepted eye models.

## Reproduction and source coverage

The historical compact and effective implementations are `archive\2026-11-11\Script\p4_compact_model.py` and `archive\2026-11-11\Script\p4_effective_zoom.py`; these pre-existing historical source pointers are explicit exceptions to the cleanup archive-pointer restriction. Their historical run identifiers preserve provenance, not current document dependencies. The validated linear E2 camera run identifier is `p4_camera_linear_accom_20261111T011655Z_09cb5314`; the former `CURRENT_MODEL.md` pointer is obsolete and absent. This record consolidates substantive historical methods/results from the earlier model, proposal, specification and result documents named above. The former `Theory.md` pointer is also obsolete and absent. The current FL definition, camera constants and comparison are self-contained in `P4_MODEL.md` and `P4_IMPLEMENTATION.md`.

## P4 polynomial-gaze historical record (from P4_POLY_FIT.md, 2026-11-13)

Historical companion documents were `archive\CURRENT_STATUS.md`, `archive\CURRENT_PLAN.md`, the earlier `P4_MODEL.md` lens-map text and `P4_SPEC.md`. These pre-existing historical archive references are explicit exceptions to the cleanup archive-pointer restriction. These roles are superseded: the earlier model text is history, and the root specification is obsolete and absent. Current companions are `P4_MODEL.md` for accepted FL theory, `P4_IMPLEMENTATION.md` for procedures/constants/evidence, and `CURRENT_PLAN.md` Section 17 for this closure.

**All results in this section: noise-free design traces, ground-truth gaze, unaudited.** Recovered-gaze results use ground truth for scoring. A=0 and fixed E2 camera unless stated. Per-state fits and sampled successes do not establish full-grid shared-law performance.

### Data, diagnostics and lens conventions

The 5x5 trace is complete: all 21 A00-A20 shards, 255,025 rows each, 5,355,525 total. The old status reports matching manifest SHA-256 values and zero failed rays for every shard; the pilot subset discrepancy was zero. These checks were not repeated for this edit. All-shard comparison to the 3x3 subset remains open.

The historical 3x3 pickle was renamed to p4_raytrace_21x101x101_12p5_A4.pkl from p4_raytrace_251x251_12p5_A4.pkl. Size stayed 524,441,352 bytes; unchanged contents were assumed from size/write time, not a new hash. The contemporary warning to check loader defaults, comparison defaults and overwrite guards concerns that old input only; current FL/F requirements are in `P4_IMPLEMENTATION.md` Section 2.

D0 interpolates node coefficients exactly; D1 fits shift 2, L 4 and keystone 2 per state. On 3x3, x^3=x; on 5x5, x^5=1.25x^3-0.25x and x^6=1.25x^4-0.25x^2. Denser nodes reduce aliasing but do not identify physical aberrations. The old ring null mode (1-x^2)(1-y^2) leaves outer predictions unchanged.

(The reference lens map and orders A/B/C belong to the historical model text; current camera constants are consolidated in `P4_IMPLEMENTATION.md`. The obsolete root `P4_SPEC.md` is not required.)

| Per-state result, um unless counts | A | B | C |
|---|---:|---:|---:|
| 3x3 worst nine-point max | 7.6 | Up to about 11 | Up to about 11 |
| 3x3 ring-only centre RMS / max | 2.16 / 3.60 | 2.88 / 3.74 | 3.59 / 4.53 |
| 3x3 failed nine / ring fits of 10,201 | 8 / 22 | 107 / 111 | 260 / 270 |
| 5x5 all-node RMS / max | 2.6 / 13.7 | 3.2 / 18.7 | 3.3 / 18.3 |
| 5x5 centre withheld, fit 24: RMS / max | 1.47 / 2.22 | 2.12 / 2.66 | 2.58 / 3.18 |
| 5x5 failed all-node fits of 10,201 | 42 | 147 | 432 |

These are converged-only. The 3x3 binned RMS of per-state nine-point maxima was about 1.1-1.4, 2.2-2.7, 2.8-3.7 and 5.4-5.6 um; this is not vector RMS. Orders disagreed on centre by 0.8-1.8 um RMS, up to 4.3 um. The 5x5 outer-eight centre result matched 3x3. Q and prism failed the old 10% gain rule; the old ladder stopped before trefoil, r^4, quartic or lag.

### Zero-gaze radial law and sampled offset fits

| Radial multiplier, t is squared pattern radius | Max residual, um | Fitted values |
|---|---:|---|
| 1+qt | 2.10 | H,q free |
| 1+qt+q4 t^2 | 0.039 | H=2.601372, q=-0.035514, q4=0.001855, rho=1.47 |
| Add q6 t^3 | 0.0012 | q6=-0.000141 |
| (1+c_s t)^(-1/2) | 0.39 | c_s=0.0699 |
| Pade (1+aP t)/(1+bP t), selected | 0.0076 | H=2.601646, aP=0.028143, bP=0.064079 |

For this historical selected Padé law, the Taylor values are $q=-0.035936$, $q_4=0.002303$, $q_6=-0.000148$ and $\rho=1.78$. Selection first requires maximum residual $\le0.1\ \mu\mathrm m$, then minimizes parameter count, maximum residual and table order in that sequence. The fit uses five nonzero radii: $0.5,\sqrt{0.5},1,\sqrt{1.25},\sqrt2$. Its gauges are zero shift, $L=I$ and $k=0$. The fixed-$H,q$ control discrepancy is below $0.0045\ \mu\mathrm m$. These are historical radial-fit facts, not the current FL estimator definition.

Old radial residual norms at radii 0.5, sqrt(0.5), 1, sqrt(1.25), sqrt(2) were 3.09,2.37,0.27,1.25,0.34 um. Inner signed residuals pointed outward. Camera controls changed q4 by 0.46% with identity camera and 0.046% with K1 +/-10%. Exact Pade de-aliasing using H_cal=2.591652, J=-0.077876 and fixed bP gives h=2.601632 and effective a=-0.035934, reproducing axial/corner pre-camera values 2.513776/2.435900 mm.

On the stride-5 441-state lattice, order-A all-node RMS/max was 2.70/13.7 um for old q-only, 2.00/7.87 corrected frozen, 1.57/5.43 free-q4 and 1.72/7.44 free offset. Frozen correction reduced RMS 26% and max 43%; free q4 added about 22% RMS gain, opposing strict gaze independence.

Free-offset centre RMS/max from 24 nodes was 0.379/0.701 um, versus frozen 1.52/2.05; from eight it was 1.08/2.48 versus 2.28/3.12. Offset RMS/max was 0.023/0.031 pattern units. Delta_x=-0.0224u_x and delta_y=-0.0225u_y explained about 70% of variance, not a validated shared law. B/C reached 17-18 um corner maxima and up to 15/21 failures. The radial selftest reported 94/94, validity gates passed, status completed_with_failures. Sampled Stage 2 took 67 minutes; full-grid conclusions remain unavailable.

### Joint ladder and earlier additive context

G0 uses six cubic shift coefficients; G1 adds six L coefficients; G2 adds two rational-keystone coefficients (14 total). G3 selected q variation and lag. G4 replaces rational keystone with six quadratic shape coefficients (18 total). G5 adds a design centre law to G4, leaving its outer predictions unchanged. T0 reproduces additive M14.

| Model | Training outer RMS / max, um | Odd/odd outer RMS / max, um | Recovered-gaze training RMS / max, arcmin |
|---|---:|---:|---:|
| T0 | 3.62 / 18.85 | 3.48 / 17.03 | Not comparable |
| G0 | 119.3 / 507.0 | Not measured | 0.86 / 1.60 |
| G1 | 66.9 / 171.9 | 65.5 / 169.2 | 0.87 / 1.55 |
| G2 | 3.08 / 10.59 | 2.98 / 9.95 | 0.035 / 0.154 |
| G3 | 2.05 / 15.17 | 1.95 / 13.10 | 0.021 / 0.162 |
| Order control | 2.97 / 29.01 | 2.80 / 26.13 | 0.034 / 0.147 |
| G4 and G5 | 4.64 / 16.43 | 4.46 / 15.41 | 0.022 / 0.058 |

Training centre known/recovered RMS/max, um: G0 64.6/105.3 and 68.1/105.3; G1 64.3/104.3 and 68.1/104.3; G2 2.62/6.01 and 3.05/7.38; G3 2.54/11.35 and 2.64/12.81; order control 3.60/5.89 and 3.86/7.20; G4 2.30/6.32 and 2.57/6.83; G5 0.065/0.377 and 0.149/0.248. G5 odd/odd known/recovered values were 0.059/0.300 and 0.144/0.253. These historical centre targets are retired.

The optimized joint run took 2,551 s and ended completed_with_errors, with 56 recorded errors. Separate coverage reports list 105 G3 failures on 2,600 training states, and 16 G2/25 G4 failures in sampled coverage; other models used 120 states per scope. These counts are not interchangeable. Maxima are conditional on successful solves. T0 seed rejection affected 9,224/10,200 states; its recovered results are not comparable. G0 state 2448 still failed its equivalence gate (6.18e-9 versus 1e-10). G4 alternative starts remain unresolved.

Earlier additive M12 -> M14 reduced outer training RMS 67.85 -> 3.624 um by adding keystone; N18 reached 2.748 um. Shift-only mean-to-centre RMS was about 65 um, M14 about 4.9 um and larger additive models about 8.2 um. Sampled additive gaze RMS/max was about 0.017-0.023/0.037-0.089 arcmin with zero sampled failures, using a different solver and scope. None of these facts resolves the later joint failures or identifies a physical mechanism.


## P4 termwise and lens-mean tests (2026-11-13/14)

**Noise-free design traces, ground-truth gaze.** This section records the A=0 termwise tests and the Increment 3 implementation/selftest handoff. Increment 1/2 results were reported by the coordinator as **gpt-6-astra: ACCEPT AS PROVISIONAL**; the numerical artifacts remain labelled "unaudited". That audit history is user-provided and no independent audit-acceptance artifact was found during this documentation review. Nothing here establishes a physical lens law, real-eye accuracy, noisy-data accuracy, or an A=4 result for these increments.

Evidence from the 2026-11-13/14 A=0 termwise experiments and full lens-map selftests was inspected read-only and copied here, with no Python, script, experiment or selftest executed for that documentation update. The first termwise experiment supplied the A=0 T0/T1 results; the subsequent T2/T3 experiment supplied its report, results and ladder. That ladder identifies itself as Increment 2 even though its tag is simply a00. The completed lens-map selftest supplied the recorded failures below; later partial-progress observations did not establish a passing rerun. The numerical results, failed cases, audit attribution and provenance limitations are preserved inline in this section. These obsolete outputs are scheduled for removal under `CURRENT_PLAN.md` Section 18; their removal has not occurred in this documentation task, and this record requires none of them as an input.

### Increment 1: T0 polynomial and T1 tangent law, A=0

M8 is the eight-outer-point mean and is primary. E1 is the traced-centre diagnostic reference, never a runtime measurement. Gaze metrics below are all-state vector errors over 10,201 A=0 states (training plus holdouts plus zero); coefficients use raw centred sensor mm, with the prescribed training fit.

Full-precision T0 six allowed coefficients:

| Coefficient | Unit | M8 | E1 (diagnostic) |
|---|---|---:|---:|
| a_x | deg/mm | -1.8208250052386186 | -1.8429408129919429 |
| a_y | deg/mm | -1.8208107940462983 | -1.8429447439505338 |
| c30 | deg/mm^3 | 3.5458687885390263e-5 | 5.477137043263538e-5 |
| c03 | deg/mm^3 | 3.6310814081856698e-5 | 5.606294995061301e-5 |
| c12 | deg/mm^3 | 3.40713796941756e-4 | 3.7207804426297856e-4 |
| c21 | deg/mm^3 | -5.84973652116694e-4 | -5.8569724969299386e-4 |

Largest forbidden coefficient, in the report's normalised deg convention: M8 7.6881293308505536e-15, E1 7.5888683818348125e-15 (about 7.7e-15). All six M8 coefficients reproduce the rounded reference table. This is design symmetry/round-off evidence, not identification of a physical mechanism.

T1 distinguishes the Taylor gain **G0=-31.467056384020118 mm** (the handoff's -31.46706) from the primary LS-matched **G1=-31.458191809621919 mm**. Their relative difference is -0.00028170968; they are not two conflicting zero-gaze radial calibrations. The primary tangent law leaves the following M8 cubic content:

| Term | Actual [deg/mm^3] | T1 LS prediction [deg/mm^3] | Actual minus prediction [deg/mm^3] | Implied epsilon (gap/c_s) |
|---|---:|---:|---:|---:|
| c30 | 3.5458687885390263e-5 | 5.783437124137227e-4 | -5.428850245283324e-4 | -0.2952253080 |
| c03 | 3.6310814081856698e-5 | 5.93356283472545e-4 | -5.570454693906883e-4 | -0.3029258735 |
| c12 | 3.40713796941756e-4 | 8.644978605537215e-4 | -5.237840636119655e-4 | -0.2848380495 |
| c21 | -5.84973652116694e-4 | -6.226576192029231e-18 | -5.849736521166878e-4 | -0.3181134473 |

The four implied epsilons average -0.3002756696 (about -0.300); spread 0.0332754, about 11% of the mean magnitude. The M8 convention indicator D12/c_s is +0.5034, supporting the primary tangent convention. T1 gaze RMS/P95/max is **9.1174966 / 18.8845315 / 29.9979893 arcmin**. The supplied historical summary quotes 9.118 / 18.885 / 29.998; 9.118 is retained here as that summary value rather than represented as the exact artifact rounding. Error is almost entirely inward radial (radial/tangential RMS about 9.117/0.129 arcmin), small inside 4 deg and large at outer gaze. E1 has the same structure (RMS about 8.83 arcmin), so it is not solely a ring effect.

T1 does not reproduce the observed x/y splits: actual a_x-a_y=-1.4211192320e-5 deg/mm versus prediction +5.9783327406e-4, and actual c30-c03=-8.5212619647e-7 deg/mm^3 versus prediction -1.5012571059e-5. The next term was therefore T2, a shared isotropic epsilon.

### Increment 2: T2 isotropic cubic and T3 camera, A=0

T2 matches the polynomial's mean linear coefficient and c21, solving G and epsilon together. The constants are obtained from the polynomial; this is not an independent lens-constant prediction.

| A=0 model/estimator | G [mm] | epsilon | Gaze RMS / P95 / max [arcmin] |
|---|---:|---:|---|
| Sensor T2, M8 | -31.457582292478403 | -0.2975535522902155 | 0.0528242 / 0.0857337 / 0.181184 |
| Sensor T2, E1 diagnostic | -31.08045260447947 | -0.28821090426209633 | 0.0685850 / 0.111304 / 0.227920 |
| M8 own cubic baseline | n/a | n/a | 0.0154565 / 0.0286880 / 0.0926449 |
| Sensor T3, M8, K1 only | not tabulated here | not tabulated here | 0.0590811 / 0.0959369 / 0.203181 |
| Sensor T3, M8, K1+K2 | not tabulated here | not tabulated here | 0.0601655 / 0.0976936 / 0.206671 |
| Frame-Z T2, M8 | not tabulated here | not tabulated here | 0.0602830 / 0.0978612 / 0.208383 |

T2 predicts c12 to a relative discrepancy of 0.0004169 (0.0417%, approximately 0.04%); c30 and c03 are respectively 9.20% and 9.53% high. The supplied summary says "x/y splits not reproduced"; quantitatively, T2 brings the splits close but not exactly to the observed values: a_x-a_y=-1.0159982970e-5 versus actual -1.4211192320e-5 deg/mm, and c30-c03=-1.0500866951e-6 versus actual -8.5212619647e-7 deg/mm^3. Thus a blanket claim that T2 predicts no split would be incorrect.

The fixed camera has kappa1=K1/F^2=1.6285934367381e-5 mm^-2. Its M8 single-point cubic increment is epsilon_cam=+0.01611622652265034 and delta_c_cam=+2.96358672305469e-5 deg/mm^3, about 84% of the observed c30. These are the stage's single-point camera comparison, not a proof that a ring mean can be passed through the camera as one point.

Camera identities I1-I4 are diagnostics without scientific thresholds: I1=-0.0014991388778534155; ring-camera gain I2=0.0003307333243709465 against its hand value 0.0002952918205794984; I3 K1/K1+K2=3.1910109560828743e-5 / 0.0003217909536580854; I4 K1/K1+K2=0.00030312791181130727 / 0.00029810640535532507. Their historical definitions are in `archive\CURRENT_PLAN.md`, "T3 computations (per A, per estimator)"; M8 carries ring-camera coupling whereas E1 is the single-point diagnostic. Frame Z inverses each camera point before averaging, as specified.

The remaining T2 gaze residual is predominantly inward radial and grows beyond about 8 deg. M8 T2 forward-mean vector RMS/P95/max is 0.482773 / 0.783454 / 1.640352 um, with the largest error at corners. Isotropic cubic F2 and anisotropic cubic forward remainders have maxima about 2.50/2.44 um for M8; the quintic F3 maximum is about 0.072 um. E1 similarly has 3.16/3.08/0.119 um. This favours studying **T4b quintic before T4a anisotropy**, rather than treating the small split as established optical anisotropy. The later user decision deferred T4a/T4b/T5 and chose the full lens-map experiment first.

Increment 1/2 were reported ACCEPT AS PROVISIONAL by gpt-6-astra. The requested preservation of Increment 1 functions was verified only behaviourally: a reported default-T1 rerun matched the earlier baseline except timings. A textual preservation comparison is not established by that result. A=0/A=4 Run 2 for both increments remains pending; the A=0 results supply no conclusion about their change with A.

### Increment 3: full lens-map script and failed selftest

The supplied 2026-11-14 handoff history says the Section 12 plan in `archive\CURRENT_PLAN.md` reached PLAN READY after three gpt-6-astra rounds and the sol implementation was written. The contemporary read-only inspection recorded the full lens-map script snapshot as 144,541 bytes and 2,436 total lines; `Script\p4_termwise_test.py` was 120,578 bytes and 2,054 total lines. Both were untracked in the Git status inspected then. No earlier committed lens-map script baseline was available for a textual comparison. The contemporary recommendation to commit audited scripts is historical, not an instruction for this cleanup; no Git action is authorized today. The history's model self-report identified GPT-6 only; the exact worker model/effort is not independently verified.

The existing completed lensmean selftest report and result say **selftest_failed, exit_code=4**, elapsed_seconds=2037.142860100008 (about 33 min 57 s). U1-U8 complete; P2/P6/P8/P9 complete; P1/P3/P4/P5/P7 fail. The retained pipeline messages are:

| Case | Saved failure |
|---|---|
| P1 | q4_truth unexpected shape failure ('A00', 'q0', 'L3c') |
| P3 | prism unexpected shape failure ('A00', 'pade', 'L2b') |
| P4 | all_terms unexpected shape failure ('A00', 'q4', 'L3b') |
| P5 | L2b_truth unexpected shape failure ('A00', 'q4', 'L2a') |
| P7 | rotation unexpected shape failure ('A00', 'pade', 'L2a') |

The failed cases' saved numbers blocks are empty; these messages do not identify the original solver/gate reason. According to the supplied handoff history, diagnostic saving was then added and hardened by sol, with a narrow astra re-audit pending. Source inspection supports the presence of bounded diagnostic-saving code, but does not establish audit acceptance or a passing rerun. A later changing progress log is only partial evidence of another attempt; it cannot replace the completed failure report. This documentation task did not start, stop or modify that process.

No A=0 real-trace L0-L3 lens-map result was available at that handoff. Track E/E1 was diagnostic only, not a runtime path; exit 4 for a failed required selftest was specified by the historical plan. Its then-proposed order was narrow re-audit -> one-command luna/low selftest -> audit selftest results and resolve failures -> A=0 lens run -> audit A=0 results -> consider A=0/A=4. A=4 execution was not cleared before the A=0 audit. `archive\CURRENT_PLAN.md`, "Handoff (2026-11-14)", holds those historical commands and dispatch lessons; its Section 12 retains that technical plan. The later closing status below supersedes this sequence.

The two historical A=0 radial calibrations remain an open provenance issue: (H,q)=(2.591652,-0.030049), with the latter derived from rounded J/H, versus (2.591699,-0.029967). They are distinct from the T1 G0/G1 Taylor-versus-LS gain distinction. No production solver, fit objective or threshold was changed in this documentation task.

### Post-diagnostics selftest rerun and open gate decision (2026-11-14)

This appendix supersedes only the earlier pending narrow re-audit/partial-progress status and next-action sequence in "Increment 3: full lens-map script and failed selftest" above; the historical text and earlier run are preserved. The supplied 2026-11-14 user handoff reports the post-diagnostics/hardening full lens-map script snapshot as 2,436 lines, 144,541 bytes, SHA-256 `83ED686987FC62293E3558EA2C16B4B2269E7623245B5E6B366E21109DEC540D`, with a gpt-6-astra static verdict **READY TO RUN**. That verdict concerns the code snapshot, not result acceptance or current authorisation for production runs. The fingerprint, audit verdict, rerun details and diagnostics below are user-reported; this documentation task did not independently verify them. The earlier read-only evidence review described above is separate from these newly supplied facts.

The user reports that luna reran `--selftest` for approximately **2076 s** and obtained **selftest_failed**: U1-U8 and P2/P6/P8/P9 pass; P1/P3/P4/P5/P7 fail. This is distinct from the earlier completed **2037.142860100008 s** selftest with saved `exit_code=4`. The latest observed process exit was reportedly **1**; historical Sections 12.5/12.8 of `archive\CURRENT_PLAN.md` specify failed-selftest exit **4**, and the user reports the audited code agrees. The discrepancy's cause is unresolved. A wrapper/console explanation is only a possibility, not a finding. Any separately authorized future mechanical task would need to capture the actual process exit separately from report status and JSON exit fields; this historical note authorizes no rerun.

User-reported latest failing-fit diagnostics (precision retained):

| Case / fixture | Failure key (shard, radial law, stage) | Scaled gradient | Rank | Scaled condition | Reported acceptance/termination detail |
|---|---|---:|---:|---:|---|
| P1 / q4_truth | `('A00', 'q0', 'L3c')` | 1.627e-10 | 5 | 56.1 | `gradient`; `step_and_relative_objective` |
| P3 / prism | `('A00', 'pade', 'L2b')` | 1.389e-8 | 3 | 2.37 | `gradient`; other termination detail not supplied here |
| P4 / all_terms | `('A00', 'q4', 'L3b')` | 4.371e-10 | 4 | 2.49 | `solver_unsuccessful` and `gradient`; `damping_stalled` |
| P5 / L2b_truth | `('A00', 'q4', 'L2a')` | 6.663e-10 | 1 | 1 | `gradient`; other termination detail not supplied here |
| P7 / rotation | `('A00', 'pade', 'L2a')` | 1.514e-10 | 1 | 1 | `gradient`; other termination detail not supplied here |

All five exceed the unchanged scaled-gradient acceptance gate **1e-10**. **P4 additionally has a solver failure (`solver_unsuccessful`, `damping_stalled`); it did not fail only the gradient gate.** The reported fits stopped near a stationary point above the gate; true convergence and the cause remain unproved. Small-step/objective termination was not acceptance under historical Section 12.5 of `archive\CURRENT_PLAN.md`. Some compact case summaries omit stage/key/exception details; the supplied history says the full saved fixture diagnostics retained those details. No missing termination or exception detail is inferred here.

The finite-difference-floor hypothesis concerns the shared `p4_a0_joint_fit.py` central-difference step `eps**(1/3)` and half-step consistency bound `5e-5`. These settings do not establish a gradient-error floor or prove that the `1e-10` gate is unattainable. The shared code may similarly reject production shape fits, but no production rejection is established by this synthetic selftest.

**Solver/gate decision: OPEN; no alternative selected.** The unresolved alternatives supplied by the user are:

- Analytic or higher-accuracy complex-step/Richardson derivatives, if suitable and reviewed, retaining `1e-10`.
- An error-aware acceptance criterion tied to a demonstrated derivative floor, requiring an explicit plan amendment and astra audit.
- A fixed relaxation, forbidden without explicit user approval under the supplied never-relax-to-pass instruction; no such approval is present.
- An astra check of whether the fits truly converged.

At the time of this entry, the solver/objective/gate requirements were recorded in the former `P4_SPEC.md` and the plan now at `archive\CURRENT_PLAN.md` Section 12.5. Those pointers are historical and do not define the accepted FL estimator. The no-relax rule was supplied by the user; current unchanged FL gates are documented in `P4_IMPLEMENTATION.md` Section 3. `.claude\CLAUDE.md` supplies the role/order rules.

At this historical entry, no A=0 or A=4 real-trace L0-L3 lens run had started, and no lens-map result existed. Historical polynomial A=0..4 tables are not evidence for those unstarted runs. Increment 1/2 A=0 results retain their reported ACCEPT AS PROVISIONAL scope and unaudited numerical labels; their independent A=0/A=4 Run 2 commands were then pending under the historical Section 8.1/8.5 prerequisites in `archive\CURRENT_PLAN.md`. E1/track E remains diagnostic only. Nothing here establishes a physical lens law, real-eye/noisy-data accuracy, or a passing selftest. The later closing status withdraws those pending runs.

**HISTORICAL CLOSING STATUS (2026-11-15), clarified 2026-11-16:** the next paragraph's order, earlier "next order" sentences and OPEN solver/gate alternatives are superseded and are not to be acted on. The lens ladder L0-L3, selftest reruns, A=0/A=4 lens runs and pending termwise Run 2 commands are closed, not required. The solver/gate decision closed without a relaxed gate; the closing estimator uses no nonlinear least-squares fit. Section 16 revision 4 subsequently governed the completed gaze-focused F/FL comparison. Its ring/mean scope withdrawal does not claim those former goals passed. Current documentation follows Section 17; FL is accepted as default and the dated closure below supplies the measured evidence. The earlier no-FL-results/edit-only statements described a pre-run state and are superseded. Everything in this older section remains history.

Historical proposed order, superseded and not execution authorisation: user decision on the open solver/gate issue -> astra/high checks convergence and updates/reviews historical Section 12 as needed -> sol/xhigh edit-only implementation against that plan -> astra/high static audit before any run -> luna/low one-command selftest -> astra/high results audit and resolution of all required failures -> A=0 lens run -> astra A=0 results audit -> consider A=0/A=4 lens run. An unchanged failing rerun was not the automatic next action. Those historical commands and dispatch lessons are in `archive\CURRENT_PLAN.md`, "Handoff (2026-11-14)"; its Section 12 was the detailed technical plan. No action in that sequence is required today.

Historical pointer clarification: the bare `CURRENT_STATUS.md` companion pointer in "P4 polynomial-gaze historical record" above refers to `archive\CURRENT_STATUS.md`, not a current root-level status file. Current status is the handoff in `CURRENT_PLAN.md`.

This documentation update executed no Python, script, experiment or selftest and edited no script or run output. The final astra audit of these four documents is separate from code/result acceptance and cannot clear a lens run.

Historical-source clarification (2026-11-14), updated 2026-11-16: former `CURRENT_MODEL.md`, `Theory.md` and root `P4_SPEC.md` pointers are obsolete and absent, and no replacement archive path for the first two is asserted. Earlier "current source"/"current model" wording describes history, not current prerequisites; this does not withdraw or recertify the historical E2 camera audit. Current roles are `CURRENT_PLAN.md` Section 17 for documentation status, `P4_MODEL.md` for accepted FL theory, and `P4_IMPLEMENTATION.md` for definitions, constants, procedures and measured evidence. Historical run identifiers preserve provenance only; this record requires no run artifacts as inputs.

### Console-exit and current-input clarifications (2026-11-14)

This historical appendix clarifies the exit/input attribution above. The coordinator reports mechanical-console `Exit code 1` for both completed selftests: first **2037.9 s**, then approximately **2076 s**. The earlier saved JSON records `exit_code=4`, `elapsed_seconds=2037.142860100008`; the first console duration and saved elapsed time are distinct reports of that run. The coordinator reports the same 1-versus-4 split in both runs; the rerun's saved JSON was not independently verified in that update. The repeated console reporting is consistent with a wrapper/console reporting process code, but does not prove that explanation. Actual process exit and the discrepancy's cause remain unresolved. The then-conditional mechanical task would have captured the actual process exit immediately after Python in the same shell, preserved it before any subsequent command, and compared it with saved JSON `exit_code` and report status. No run occurred in that documentation update. Its four solver/gate alternatives and prerequisites were OPEN at the time; the later closing status superseded them and authorizes no rerun today.

The published input contract is the **5x5** shard set `data\p4_raytrace\p4_raytrace_5x5_101x101_12p5_A00.pkl` through `p4_raytrace_5x5_101x101_12p5_A20.pkl`, with `data\p4_raytrace\p4_raytrace_5x5_manifest.json`; current loading/gates are consolidated in `P4_IMPLEMENTATION.md` Section 2. The old Section 4 pointer refers to `archive\CURRENT_PLAN.md` and is historical. Both older input names `p4_raytrace_251x251_12p5_A4.pkl` and `p4_raytrace_21x101x101_12p5_A4.pkl` are historical and are not current execution prerequisites. The earlier console-exit discrepancy concerns old nonlinear-lens selftests, not the completed FL comparison below. Its OPEN alternatives do not reopen the closed lens task.

## 2026-11-16 closure: accept FL as default and hand off

**Goal closed by user acceptance.** The decision is: "accept this linear A model as default. quadratic will be fallback or future use. rewrite/consolidate @P4_MODEL.md and @P4_IMPLEMENTATION.md. need close the current goal for next new goal (as handoff)." The user accepts approximately 1 arcmin FL gaze error. FL is now the default; F/quadratic remains the fallback or future option. The current contract is `CURRENT_PLAN.md` Section 17; Section 16 preserves the original experiment.

The accepted gaze law is the same spatial order-3 polynomial with ten monomials per axis and twenty coefficient positions at each accommodation. FL forces fourteen positions to exact zero; output $w_x$ retains $x,x^3,xy^2$ and output $w_y$ retains $y,x^2y,y^3$. Each allowed coefficient is $c(A)=c_0+c_1A$. The size laws and full joint surface are

$$
\mathbf E(\boldsymbol\tau)=(\tau_x^2,\tau_y^2,\tau_x^4,\tau_x^2\tau_y^2,\tau_y^4),
\qquad \gamma(A)=\gamma_0+A\gamma_1,\qquad \mu_{\rm cal}(A)=u_0+u_1A,
$$

$$
M(A,\boldsymbol\tau)=(u_0+u_1A)[1+\mathbf E(\boldsymbol\tau)\mathbin{\cdot}(\gamma_0+A\gamma_1)].
$$

The induced $A^2$ product is required and is not a separately fitted quadratic law. FL stores 12 gaze-law, 10 gamma-law and 2 mu-law values, giving 24 law values, 30 with scales/origin/$H_0,q_0$, and 35 with the fixed camera. F counts are 60/15/5, then 80/86/91. These are independent stored-value counts, not polynomial sizes; structural zeros, gamma padding and derived mu ordinates do not add law parameters. `P4_MODEL.md` now gives the theory first; `P4_IMPLEMENTATION.md` gives the source map, exact calibration order, unchanged gates, runtime, compact constants, comparison and reproduction.

The completed comparison is **noise-free design traces; ground-truth gaze used for calibration targets and scoring only; development data, non-anchor accommodation previously inspected, not fresh validation**. All $21\times10{,}201=214{,}221$ states are valid for both models. The F gaze reproduction checks passed within $10^{-6}$ arcmin, the quoted interior A checks within $10^{-8}$ D, and median/maximum passes exactly 6/12. The recorded synthetic selftest passed. These are copied results; no new execution occurred for this consolidation.

| All-state measure | FL default | F fallback |
|---|---:|---:|
| Coverage; failures | 100%; 0 | 100%; 0 |
| Gaze RMS / maximum, arcmin | 0.1544 / 1.0176 | 0.0195 / 0.1904 |
| A error RMS / maximum, D | 0.019 / 0.168 | 0.0019 / 0.033 |
| Passes median / maximum | 6 / 11 | 6 / 12 |
| Clipped states | 11,209 | 6,180 |
| States with gaze error $>0.1$ arcmin | 93,258 | 120 |
| States with gaze error $>1$ arcmin | 8 | 0 |

FL interior A01-A19 gaze RMS/maximum is 0.1537/1.0176 arcmin. A00 is 0.0652/0.0956 arcmin with near-zero A error (RMS $2.66\times10^{-6}$ D, maximum $7.49\times10^{-5}$ D). A20 is 0.2188/0.8403 arcmin, with A RMS/maximum approximately 0.026/0.105 D. All eight FL states above 1 arcmin are at interior accommodation values. Clips are 10,164 at A00 and 1,045 at A20; F clips only 6,180 A20 states. Maximum FL gaze error is 1.0175524717148585 arcmin at A10, state 10200, gaze $(12.5,-12.5)^\circ$; maximum A error is 0.16788211350886328 D at A10, state 0, gaze $(-12.5,12.5)^\circ$.

**Original strict experiment result: `L0=True`, `L1=True`, `L2=False`, `CLOSE: F`.** The 1.0176 arcmin maximum did not pass the strict $<1.0$ arcmin gate. The user's later approximately 1 arcmin acceptance selects FL after seeing that result; it does not change the experiment verdict, numerical gates or saved output. No accommodation-error acceptance threshold has been set. The condition gate $10^8$ and nonlinear scaled-gradient gate $10^{-10}$ remain unchanged; FL uses linear fits rather than nonlinear least-squares fits.

Evidence provenance: primary source `Script\p4_linA.py`; recorded script SHA-256 `e5341f7dcf26bc17ad761b372a18a5da648cf752362b85ee5dd40d666df626c1`. Reproduction is documented, not executed, from the repository root:

```powershell
& 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe' .\Script\p4_linA.py
```

Reproduction retains the original strict-gate verdict. This consolidation changes documentation only, with no code/constants/gates/outputs changed. Astra/high independent documentation and source-read audit passed with no blocking findings. The audit confirms documentation consistency with inspected sources and recorded evidence; it does not certify code or a new execution. No scripts, tests or rerun were executed.

Limits and next-goal handoff: there is no blind, noisy or real-eye validation, contraction/uniqueness proof, or continuous/off-grid accuracy guarantee. Endpoint clipping can leave size mismatch. AE7 simultaneous fitting has not been performed. Candidate separate goals are blind/noisy/real-eye validation, accommodation accuracy criteria, iteration robustness/convergence, and AE7 simultaneous fitting. The current goal is closed; these candidates do not reopen its withdrawn physical-lens, ring-coordinate or mean-fit criteria.
