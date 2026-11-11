# CURRENT MODEL — decision record

## 1. Goal and scope

**Numerical scope for every fit/result below: 167.93 mm, zero working distance.** These fits characterize traced rays on that setup only.

The target is an empirical **geometry-only / ray-only** description: field and gaze angles to traced ray tangent coordinates, plus compact sensor mapping where useful. Current camera fits do not explain how gaze or field produces the tangents. The labels are **E1 = P1** and **E2 = P4**. The user confirmed `.len` and `.seq` refer to the same lens; that file identity has not been independently verified here.

For P4, RC is 13.8 mm behind `Cornea_ENT` in the sequence (reported surfaces 49–52). The full ray path and geometric transforms have not been independently audited here.

The camera-stage question is answered within the stated numerical scope. Section 6 records the P1 gaze, scale and missing-centre results.

## 2. Verified facts

P1 camera refit: recorded result; source run archived. The current user-supplied input is `data/p1_raytrace/p1_raytrace_251x251_12p5.pkl` (127,015,917 bytes). P4 sequence: `Lens/p4_ME_no_DM.seq`. The current P4 input is `data/p4_raytrace/p4_raytrace_251x251_12p5_A4.pkl` (524,441,352 bytes). These files exist, but their hashes and contents have not been verified. The historical P1 hash `914e240cc5580821f4e79a580840339513f2c5cd4f892fbed499c8c3bdfec296` and P4 hash `aceb43c9744dec4898df558d4e21ab786dede141e464a47303b1dd13c0892c03` identify the previous inputs, not the current files. The recorded P4 results and unchanged validators assume 21 accommodation levels × 101 ADE × 101 BDE × 9 rays; the current `251x251`/`A4` filename may describe a different lattice or a single A=4 D slice. Compatibility is unverified. The seven fit/analysis entry points accept optional `--expected-sha256`; the default skips hash enforcement and prints the hash found. Schema, shape, metadata, lens and science checks remain enforced. Reported tangent inputs for P4 are `tx=-l16/n16`, `ty=-m16/n16`. Training uses states where ADE and BDE indices are both even; holdout is all other states. T is computed from training rays only.

**Run evidence is summary/report evidence.** The linear run summary records `completed`, `accepted=true`, `fit_health_passed=true`, `validation_eligible=true`; its run log says completed and accepted. No process exit code is saved in the summary, so a process exit code is not independently established by this audit. The JSON's `status` is a run-record status, not an OS exit status.

## 3. P1 — E1 camera refit

**167.93 mm, zero working distance.** Model definition: pinhole `f,cx,cy` plus radial `k1`; residual is predicted minus traced. Full 251 × 251 lattice; train 15,876 states / 142,884 rays, holdout 47,125 states / 424,125 rays. T = 0.007831188170431441, defined as max(|tx|,|ty|) over active training rays; normalized `u=tx/T`, `v=ty/T`. Conversion: `f=f_norm/T`, `k1=k1_norm/T²`; centers unchanged.

| Physical parameter | Estimate | Formal SE | Unit |
|---|---:|---:|---|
| f | −425.1378570992409 | not included in summary physical table | mm/tan |
| cx | −1.1406923614785756e−16 | not included in summary physical table | mm |
| cy | 9.095143002416344e−17 | not included in summary physical table | mm |
| k1 | 2.9287122874005385 | not included in summary physical table | tan⁻² |

Holdout Euclidean RMS = 0.00014742686224599716 µm; maximum point error = 0.0015814139116305877 µm. Dataset gates passed; fit converged. Historical E1 holdout max was 0.0015829611284427967 µm, for earlier data/lens and reference only.

## 4. P4 — E2 camera and accommodation law

**167.93 mm, zero working distance.** Physical radial mapping is `x̂=cx+f tx (1+k1 r²+k2 r⁴)`, `ŷ=cy+f ty (1+k1 r²+k2 r⁴)`, `r²=tx²+ty²`; `f` in mm/tan, `cx,cy` in mm, `k1` in tan⁻², `k2` in tan⁻⁴. Centered laws fix `cx=cy=0`. Per-A own fits allow centers; constant camera keeps all five A=0 own-fit physical parameters including free centers.

For the linear-law fit, `T_a=max(|tx|,|ty|)` on each A's full even/even training rays; normalized inputs `u=tx/T_a`, `v=ty/T_a`; normalized coefficients convert by `f=f_norm/T_a`, `k1=k1_norm/T_a²`, `k2=k2_norm/T_a⁴`. Fit indices 0,5,10,15,20 (A=0,1,2,3,4 D); 16 intermediate levels were not used to estimate the six coefficients and serve as interpolation holdouts for that law. Own-fit diagnostics were nevertheless computed at all 21 A levels, so these levels are not untouched by the entire analysis. Per-A diagnostics and V2/V3 use own fits at all levels; V6 compares the five anchor results with the earlier per-A run.

### Six-parameter law

Ordering: `(f0, f1, k1_0, k1_1, k2_0, k2_1)`. Equations: `f(A)=f0+f1 A`, `k1(A)=k1_0+k1_1 A`, `k2(A)=k2_0+k2_1 A`, with A in diopters. Formal SEs are deterministic-data uncertainty summaries, not measurement-noise estimates. OLS slope SE is five-point line scatter with 3 df; joint/own SE uses the sensor-residual Jacobian covariance `RSS/(2N-p)`.

| Parameter | Unit | OLS | OLS SE | Joint | Joint SE |
|---|---|---:|---:|---:|---:|
| f0 | mm/tan | 425.13755805853566 | 3.18788509e−7 | 425.1375580469268 | 6.67733076e−7 |
| f1 | mm/tan/D | 5.461957588731658e−6 | 1.30144864e−7 | 5.4675382302626545e−6 | 2.72663482e−7 |
| k1_0 | tan⁻² | 2.9435451777208224 | 4.96278390e−6 | 2.9435454608804155 | 8.09189158e−6 |
| k1_1 | tan⁻²/D | −8.05515375648903e−5 | 2.02604804e−6 | −8.069035357733996e−5 | 3.34867424e−6 |
| k2_0 | tan⁻⁴ | −106.47414624992594 | 0.00649396003 | −106.47462088323039 | 0.00928189918 |
| k2_1 | tan⁻⁴/D | 0.09274142769605395 | 0.00265114808 | 0.09297609201985432 | 0.00388257077 |

Joint solver succeeded; rank 6; column-scaled condition number 46.2284; 117,045 fit rays. Reported proposal values equal OLS values when rounded to six significant digits.

### A=0 E1 diagnostic and independent per-accommodation E2 fits

**167.93 mm, zero working distance.** Separate P4 A=0 E1 diagnostic (recorded result; source run archived) has holdout maximum 0.459290 µm, corner dominated. This is the P4 E1 diagnostic, not the P1 E1 refit result.

Per-accommodation fits: recorded result; source run archived; A levels are 0, 1, 2, 3, 4 D. **167.93 mm, zero working distance.** Entries are estimate (formal SE); center estimates and SEs are in µm as recorded in table_A. Holdout is Euclidean RMS / point maximum in µm.

| A (D) | T_a (tan) | f (mm/tan) | cx (µm) | cy (µm) | k1 (tan⁻²) | k2 (tan⁻⁴) | Holdout RMS/max (µm) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0228424594374 | 425.137557657 (9.0563e−7) | −8.6203e−13 (3.8858e−6) | 3.3439e−13 (3.8858e−6) | 2.943551420 (1.0902e−5) | −106.482286730 (0.0124386) | 0.000788779 / 0.014930523 |
| 1 | 0.0227826138565 | 425.137563759 (8.8200e−7) | 5.1675e−13 (3.7814e−6) | 5.4044e−13 (3.7814e−6) | 2.943460929 (1.0730e−5) | −106.376627862 (0.0123504) | 0.000766857 / 0.014659132 |
| 2 | 0.0227311604907 | 425.137569348 (8.6059e−7) | −9.9400e−13 (3.6880e−6) | −4.2811e−13 (3.6880e−6) | 2.943378392 (1.0576e−5) | −106.281224866 (0.0122716) | 0.000747201 / 0.014428685 |
| 3 | 0.0226854025193 | 425.137574604 (8.4065e−7) | 1.2238e−12 (3.6021e−6) | 2.5182e−13 (3.6021e−6) | 2.943301013 (1.0433e−5) | −106.192567983 (0.0121987) | 0.000729054 / 0.014225165 |
| 4 | 0.0226453627774 | 425.137579544 (8.2212e−7) | −5.9991e−13 (3.5232e−6) | −3.6350e−13 (3.5232e−6) | 2.943228620 (1.0301e−5) | −106.110609531 (0.0121316) | 0.000712341 / 0.014047547 |
### Accommodation drift and gates

Across A=0,1,2,3,4 D, same-domain fixed-camera total drift `d_total_um` is 0, 0.000446290, 0.000840256, 0.001198845, 0.001520977 µm. A0 fixed-camera holdout point maxima at those anchors are 0.014930523, 0.014252578, 0.013680412, 0.013179560, 0.012745993 µm. The practical trend flag is not a test of whether the fitted coefficient slope is nonzero: it flags only when the 4-D sensor effect exceeds both 0.01 µm and 3× its propagated SE. Thus f has slope 5.44907e−6 per D (SE 2.72841e−7), but 4-D sensor effect is only 0.000498631 ± 0.000024967 µm; k1 slope −8.04338e−5/D (SE 3.35046e−6), effect −0.00163026 ± 0.000067908 µm; k2 slope 0.0926734/D (SE 0.00388405), effect +0.000980074 ± 0.000041076 µm. cx/cy slopes are effectively zero. These coefficient slopes are precisely estimated; their sensor effects stay below the 0.01 µm practical threshold. The measured T trend is −4.91405e−5/D; T is the exact domain scale, not a camera parameter.

All per-accommodation gates G1–G6 passed at each of the five requested levels. G1 checks the center/stop ray within 1e−6 mm in x/y/z, LMN norm error ≤1e−12, all n16 negative, all ray statuses zero, and center-ray statuses zero. G2 requires E2 holdout point maximum ≤0.025 µm, holdout/train max ratio <1.5, solver success, and full rank. G3 requires k2 ≥10 SE. G4 requires fixed-A0 versus per-A matched-domain maximum drift ≤0.01 µm. G5 requires fixed-A0 camera holdout maximum ≤0.025 µm. G6 rejects any significant practical parameter trend using the dual threshold above. This run's overall verdict is “single A-independent E2 camera (A=0 parameters) is valid for A=0–4 D” at the stated scope.
### V1–V6 gates

All six passed in the full run. Exact thresholds:

| Gate | Criterion | Result |
|---|---|---|
| V1 | OLS linear holdout max ≤ 0.025 µm at each of 21 levels | PASS; range 0.0140818479664–0.0149704141802 µm |
| V2 | OLS linear holdout max minus own-fit holdout max ≤ 0.0005 µm (signed) at every level | PASS; range −3.52418e−5 to +3.98913e−5 µm |
| V3 | OLS law-to-own vector displacement max ≤ 0.0005 µm across all rays at every level | PASS; largest 4.23716e−5 µm |
| V4 | OLS six coefficients match proposal rounded to 6 significant digits | PASS |
| V5 | Each own-fit abs(cx) and abs(cy) strictly < 1e−6 µm | PASS |
| V6 | Five anchors reproduce the earlier per-accommodation result within listed absolute tolerances, relative tolerance zero (recorded result; source run archived) | PASS; baseline identity flag true |

V6 absolute tolerances: T 5e−14; f 5e−9; cx/cy 1e−12; k1 5e−8; k2 5e−5; point maximum 5e−7 µm; Euclidean RMS 5e−8 µm. This establishes numerical equivalence under declared tolerances, not reproducibility with the historical executable (the recorded source hashes differ; see Audit). No V2/V3 failures were confined to intermediate levels; no quadratic fallback was fitted.

### Per-level holdout maxima

Values are µm; own, constant_A0, OLS-law, and joint-law models are compared at every A. These own-fit diagnostics cover all levels; only the five anchor levels estimate the linear-law coefficients.

| A (D) | Own | Constant A0 | OLS law | Joint law |
|---:|---:|---:|---:|---:|
| 0.0 | 0.014930523 | 0.014930523 | 0.014970414 | 0.014967361 |
| 0.2 | 0.014868003 | 0.014774822 | 0.014884896 | 0.014882159 |
| 0.4 | 0.014812790 | 0.014636605 | 0.014816157 | 0.014813732 |
| 0.6 | 0.014759775 | 0.014503964 | 0.014752315 | 0.014750200 |
| 0.8 | 0.014708988 | 0.014376975 | 0.014693489 | 0.014691681 |
| 1.0 | 0.014659132 | 0.014252578 | 0.014636617 | 0.014635114 |
| 1.2 | 0.014610739 | 0.014132005 | 0.014582971 | 0.014581770 |
| 1.4 | 0.014563491 | 0.014014504 | 0.014531810 | 0.014530910 |
| 1.6 | 0.014517735 | 0.013900859 | 0.014483961 | 0.014483359 |
| 1.8 | 0.014472593 | 0.013789021 | 0.014437351 | 0.014437046 |
| 2.0 | 0.014428685 | 0.013680412 | 0.014393455 | 0.014393444 |
| 2.2 | 0.014385871 | 0.013574697 | 0.014351951 | 0.014352232 |
| 2.4 | 0.014343910 | 0.013471297 | 0.014312265 | 0.014312837 |
| 2.6 | 0.014303510 | 0.013371844 | 0.014276102 | 0.014276963 |
| 2.8 | 0.014263841 | 0.013274397 | 0.014241479 | 0.014242627 |
| 3.0 | 0.014225165 | 0.013179560 | 0.014209038 | 0.014210472 |
| 3.2 | 0.014187407 | 0.013087142 | 0.014178602 | 0.014180319 |
| 3.4 | 0.014150954 | 0.012998016 | 0.014151102 | 0.014153102 |
| 3.6 | 0.014115503 | 0.012911478 | 0.014125827 | 0.014128107 |
| 3.8 | 0.014080847 | 0.012827052 | 0.014102295 | 0.014104856 |
| 4.0 | 0.014047547 | 0.012745993 | 0.014081848 | 0.014084687 |

## 5. What gaze does

**167.93 mm, zero working distance.** The camera-fit question is **ANSWERED**: E2 was fitted separately at A=0,1,2,3,4 D. Section 4's values imply only negligible accommodation dependence, represented by a linear law: over 0–4 D, `f` changes about 5e−8 relative, `k1` 0.011%, and `k2` 0.35%; the net corner effect is ≤0.0016 µm. The centred model fixes `cx=cy=0`; free-centre fits are numerically zero. Each A has its own 101 × 101 (ADE,BDE) eye-rotation grid × nine field rays and its own traced tangents. Its E2 fit uses that A's `(tx,ty) → (xsi,ysi)` pairs: tangents are camera-fit **inputs**, not a separate camera-fit goal. Effective coefficient slopes do not imply physical camera motion.

`tx=-l16/n16`, `ty=-m16/n16` are tangents of chief-ray angles at camera stop S16 (the camera entrance pupil). CODE V traces them; in the real measurement only image positions are measured. The user's clarification reports every ray passing through the S16 origin to about 1e−13 mm; that precision was not independently rechecked here. With training-ray maximum tangent `T≈0.0228`, define `u=tx/T`, `v=ty/T`, `r²=u²+v²`. The normalized mapping is `x=cx+f_norm u(1+k1_norm r²+k2_norm r⁴)` and likewise for y using v; `r` is a radial viewing-angle coordinate in the tangent plane, **not a pupil radius**. Section 4 gives the physical coefficients and conversion. Almost all mapping is pinhole: `x≈f tx`, `f≈425.14 mm/tan`, so `tx=0.01` gives about 4.25 mm; radial terms are small corrections.

The composition is `sensor = C(E(field,gaze,A))`. The physical camera stage knows nothing about the eye, gaze, or accommodation; their dependence enters through the eye stage's tangents. This bookkeeping separates the nearly exact camera mapping from the eye stage, whose physical model remains unresolved. Earlier direct `(field,gaze,A) → sensor` empirical fits with 88–107 parameters in `Record.md` lacked physical meaning. These are the user's 2026-11-10/11 clarifications; no new trace or fit verification accompanies them.

**Measurement and target (user clarification, 2026-11-11):** Calibration uses a fixed nine-point design at ADE=BDE=0 for each A. The source cannot overlap the camera, so its central point is unobservable: the available measurement is the other eight image positions, including during calibration. Real `tx,ty` are also unmeasured. The missing labelled image centre `c`, nine-point arithmetic mean `m`, eight-point mean and camera principal point are distinct quantities. The prior `c=m−B(g)` correction in [Theory.md](Theory.md#frozen-a0-centroid-and-gaze-channels) depends on unknown gaze; its identifiability as a joint estimator has not been established.

## 6. P1 gaze, scale and missing-centre results

Scope: P1 only, 167.93 mm, zero working distance, A=0. These are CODE V design-trace results with E1, not real measurements. Calculated findings were reported by Fable 5.2 / Opus 5.5 from saved traced arrays; none has been independently audited. Noise results cover one noise seed and 200 Monte Carlo trials. P4 and accommodation are not covered. The eye-stage formula is empirical and has no lens-model derivation. Replacing tangent inputs with image positions was analysed, but not tested as a fit.

### Tangent definition and camera

P1 uses tx=l8/n8 and ty=m8/n8 at S8, with no minus sign. Since n8<0, tx has the opposite sign to l8. P4 instead uses tx=-l16/n16 and ty=-m16/n16 at S16. T=0.007831188170431441 is max(|tx|,|ty|) over even/even training rays: 15,876 states and 142,884 rays. Set u=tx/T, v=ty/T and r²=u²+v². T is a fixed scale, so u,v contain exactly the same information as tx,ty.

E1 is x=cx+f*tx*(1+k1*(tx²+ty²)), with the corresponding y equation using ty. Here f=-425.1379 mm/tan, k1=2.9287 tan^-2 and cx≈cy≈0. Calculated normalised values are f_norm=f*T≈-3.329 mm and k1_norm=k1*T²≈1.796e-4. The camera fitter takes u,v; it does not take ADE, BDE or field labels as inputs.

### Gaze dependence

The study contains 63,001 gaze states: ADE and BDE each span -12.5 to +12.5 degrees in 0.1-degree steps, giving 567,009 valid rays. Nine-ray mean slopes are du/dBDE=+0.0389914/deg, dv/dADE=-0.0386808/deg, dtx/dBDE=+3.05349e-4 tan/deg and dty/dADE=-3.02917e-4 tan/deg. Centre-ray gain a=3.1069e-4 tan/deg, or 0.03967 normalised units/deg and 0.0178 rad/rad. BDE drives tx,u; ADE drives ty,v with a negative sign.

At zero gaze the field-label order is (-1,+1), (0,+1), (+1,+1), (-1,0), (0,0), (+1,0), (-1,-1), (0,-1), (+1,-1). Tangent signs oppose field-label signs. Edge magnitudes are 0.003991148 tan (0.5096 normalised); each corner component has magnitude 0.003833097 tan (0.4895 normalised); the centre is zero. Per-ray cross-axis linear slopes are about 0.5% of main slopes and cancel in the nine-ray mean. Mean u is odd in BDE and mean v is odd in ADE to about 1e-13; reversing both gaze angles and selecting the opposite ray gives point symmetry.

The pattern moves almost as a rigid shift. The non-common part is 0.8% of the shift at (5,0), 1.2% at (5,5), and 2.0% at (12.5,12.5) degrees. Rigid-rotation projection terms are about 500 times too small and have the opposite sign.

For w=(BDE,-ADE) in degrees and zero-gaze tangent p_i, the empirical model is t_i(w)=p_i+S(w)+c2|w|²p_i+c3(w·p_i)w+alpha|p_i|²w+beta(w·p_i)p_i. Reported coefficients are c2=1.915e-5 and c3=4.583e-5 deg^-2, alpha=-0.0665 and beta=-0.1028 (deg*tan)^-1. The anisotropic shift is Sx=a*wx+sx3*wx³+sxy*wx*wy² and Sy=a*wy+sy3*wy³+syx*wy*wx², with a=3.1069e-4 tan/deg, sx3=sy3=-2.60e-8, sxy=-1.02e-8 and syx=-5.76e-8 tan/deg³. Symmetry forbids gaze-times-field and gaze-squared shift terms in this expansion. Shape terms reach 30–60 µm at 12.5 degrees; the non-common k1 contribution is only 0.08–0.16 µm at 5 degrees.

### Scale and calibration limits

For zero-gaze image positions alone, t→s*t, f→f/s and k1→k1/s² is an exact scale gauge. Only c, m_i=f*t_i and kappa=k1/f² are identifiable. With design tangents supplied, f is recovered to 0.0175% with fixed k1 or 0.069% with free k1. k1 is not usefully recovered: about 930% uncertainty at 1 µm noise, against only 0.08 µm edge and 0.2 µm corner signal. Without the centre point, the zero-gaze centre is the mean of eight symmetric points, with sigma=0.354 µm. E1 has a unique inverse because k1>0 makes the radial derivative positive. At 1 µm image noise, calculated sigma(tx)=2.35e-6 tan and sigma(u)=3.0e-4.

A 3×3 gaze calibration at -12.5, 0 and +12.5 degrees does not remove the scale gauge even when gaze is known. One absolute quantity must come from outside the image fit: design gain a, an absolute nonlinear eye-stage amplitude, design tangents, or f. With gain known, reported relative f accuracy is 1.1e-4 and tangent-difference accuracy is 1.2e-6 tan. k1 uncertainty remains 49% (34% for 5×5; 43% for four-corner calibration). Gain error propagates as df/f=-da/a and dk1/k1=-2*da/a. k1 depends on the eye-stage shape terms: omitting them gives a 0.94% f error and k1≈30 instead of 2.93. A design quantity remains necessary to identify scale.

### Recovering the missing centre

Only the eight outer image points are observable. Their mean has a gaze-dependent bias of about 1.1 µm/deg: 5.6 µm at (5,0) and 19.1 µm at (12.5,12.5). The estimator 2*mean(four edges)-mean(four corners), called E2 in this centre study only and unrelated to the P4 E2 camera, leaves a null-mode bias of 0.126 µm/deg: 0.63 µm and 2.16 µm at those gazes. The term (1-x²)(1-y²) vanishes at all eight nominal nodes but equals one at the centre, so unrestricted outer-point interpolation cannot determine the centre.

A parametric fit to the eight points over 5×5 calibration gazes uses the true zero-gaze tangent p as field variable, not nominal node coordinates. With f and k1 fixed, reported centre bias is 0.16–0.25 µm and calibration-noise contribution 0.29–0.49 µm at 1 µm noise. f has an exact scale gauge and k1 is weakly identifiable. The 3×3 calibration fails at 5 degrees, with 7.3 µm centre error, because linear gain and pure cubic shift terms alias.

When the centre ray is aligned with the principal point, the predicted image centre is c+f*S(w)*(1+k1|S(w)|²), where c is the camera principal point. For ring means expressed in tangent space after E1 inversion, the reported relation is S=(rho*M_edge-M_corner)/(rho-1), with rho=|p_corner|²/|p_edge|²=1.8447, not 2. Its validity depends on the stated eye-stage model and has not been independently audited. Assumptions include a gaze-dependent shift of field degree at most two in true tangent p, no |p|^4*w term, a grid symmetric about the centre ray, and centre-ray alignment with the principal point (untested). The reported fourth-order contribution is 0.0035 µm/deg. Allowing an unconstrained |p|^4*w term makes the centre non-identifiable from the eight points.

The reported runtime calculation solves gaze w from the eight points, then evaluates the centre law. Reported Jacobian condition numbers are 1.002–1.077; gaze uncertainty is about 0.17 arcmin per axis and centre uncertainty about 0.35 µm per coordinate, plus 0.2–0.3 µm bias at 1 µm noise. A shift-law error giving 3.3 arcmin gaze error changed the centre by about 0.12 µm. A four-unknown free-centre form gives gaze errors about 100 times larger, 0.14–0.47 degrees. Gaze→centre→gaze iteration has reported contraction factor 0.0085 (about 117-fold error reduction per step), conditional on an accurate inverse centre law. A constant pattern offset looks like 0.0076 degrees of gaze per µm; a common tangent offset separates from gaze only through weak shape terms, with one-frame gaze sigma 0.16–0.58 degrees.

E1 and E2 cameras are accepted at the stated scope. These P1-only calculations do not establish the end-to-end targets of accommodation error <0.1 D and gaze error <1 arcmin with unknown runtime A and gaze at 1 µm image noise.

## 7. Scripts and data layout

The 11 kept scripts in `Script\` are:

- `p1_retrace_no_dm_full_detail.py`
- `p4_retrace_full_detail.py`
- `p1_camera_refit.py`
- `p1_camera_fit.py`
- `p1_uv_gaze_study.py`
- `p4_camera_per_accom.py`
- `p4_camera_linear_accom.py`
- `p4_fit.py`
- `p4_a0_e1_fit.py`
- `distortion_grid_p4_biaxial.py`
- `distortion_grid_p4_biaxial_macro.py`

Run outputs use `runs\<script name>\`, a fixed folder overwritten on re-run; smoke outputs use `runs\<script name>_smoke\`. The A=0 E1 fit exception is `p4_a0_e1_fit.py` → `runs\p4_a0_e1\` (smoke: `runs\p4_a0_e1_smoke\`). Full-detail data destinations and fit inputs are `data\p1_raytrace\p1_raytrace_251x251_12p5.pkl` and `data\p4_raytrace\p4_raytrace_251x251_12p5_A4.pkl`; pilot and file-thickness retraces retain distinct filenames in those same data folders. The biaxial collectors retain their separate `distortion_grid.pkl` outputs within their run folders.

The old `runs\` folder was archived on 2026-11-11 to `archive\2026-11-11\runs\`; moved scripts are in `archive\2026-11-11\Script\`. The three excluded scripts (`p4_a0_camera_ladder.py`, `distortion_grid_p4.py`, `distortion_grid.py`) are still in `Script\` awaiting the user's move.

## 8. Model limits and audit notes

- Historical `p4_fit.py` review found a wrong `d_mm` around line 779 and an incorrect pilot distance-transport formula/scaling around lines 909–912. These findings apply to the dropped pilot only; the transported pilot model is not a validated result.
- Linear accommodation law: recorded result; source run archived. Historical source provenance is retained in the archive; this record does not claim that the edited scripts match the historical executable.
- P1 refit and P4 per-accommodation fits: recorded result; source run archived. Historical executable hashes differ from the sources reviewed at handoff; exact historical-code reproducibility is not established.
- Run acceptance and gate flags are recorded in reports/JSON. No explicit process exit field was found. Linear full run is accepted; smoke mode is not validation-eligible (V4/V6 do not have full-run meaning). Report audit notes include a potential wording inconsistency where V gates pass but fit health can fail in another run, and unguarded report writes; automated acceptance has those recorded limitations.
- Sequence text confirms RC is 13.8 mm behind `Cornea_ENT` at file lines 49–52. `.len`/`.seq` equivalence is user-confirmed, not independently checked in this audit. An approximately 0.0003 µm k3 floor was only a scratch note, unverified; Fable/canonical status is also unverified. Neither is established evidence.
- Smoke runs are designed to exit 1 when their own gates/acceptance fail; V4/V6 have full-run meaning only. Audit verdict: accepted with nonblocking notes; no fit rerun was performed.
