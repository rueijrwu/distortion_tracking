# SUGGESTION — missing-centre recovery, scale, and accommodation for P1 and P4

Status: mathematical proposal only. Nothing in this document has been run, traced or fitted.
**Every number is hand arithmetic and UNVERIFIED** unless it is listed in section 2.1 as a recorded value, and the recorded values are themselves unaudited design-trace results.
No numerical success is claimed anywhere.

---

## 1. Purpose, scope, measurement model

### 1.1 Purpose

Decide how to recover the missing grid centre from the eight measurable image points, what can and cannot be identified from image-only data, what an independently calibrated camera (flat grid), a matched model eye and CODE V add, and how accommodation is handled for P4.

### 1.2 Scope

- P1 (corneal reflection) and P4 (posterior lens reflection), 167.93 mm, zero working distance.
- Only relative changes are needed; absolute scale matters only where stated.
- Targets carried from the baseline: accommodation error < 0.1 D and gaze error < 1 arcmin at 1 µm image noise. This document does not establish them.

### 1.3 Measurement model

- The grid is a 3×3 nine-point design. The central point cannot be measured (the source overlaps the camera). Only the 8 outer image points (x_i, y_i) are available, always.
- Ray tangents (u, v) = (tx, ty) are never measurable in the real system.
- Reference frame: ADE = BDE = 0, 8 points.
- Calibration frames: known gaze (ADE, BDE), 8 points each.
- Accommodation A is known only during calibration (0, 2, 4 D) and unknown at runtime. Gaze is unknown at runtime.

### 1.4 Notation

| Symbol | Meaning |
|---|---|
| X | image position (mm) |
| t | chief-ray tangent (tx, ty); r = \|t\| |
| p_i | zero-gaze tangent of grid point i |
| w | (BDE, −ADE) in degrees |
| S(w) | gaze shift of the pattern in tangent space |
| c | camera principal point |
| f, k1, k2 | camera focal length and radial coefficients |
| q = f·t | image-scaled tangent (mm); m_i = f·p_i |
| κ1 = k1/f², κ2 = k2/f⁴ | image-only distortion invariants |
| R_e, R_c | tangent radii of the edge ring and the corner ring |
| Q_e, Q_c | the same radii in the image (mm) |
| ρ = R_c²/R_e² | ring ratio |
| R̄² = (R_e² + R_c²)/2 | mean squared ring radius |
| X̄_e, X̄_c | image means of the 4 edge points and the 4 corner points |
| M8 = (X̄_e + X̄_c)/2 | mean of the 8 observed points |
| D = X̄_c − X̄_e | ring-mean difference |
| Δ = M8 − M8_ref | displacement of the 8-mean from the reference frame |
| K = (1 + ρ)/(2(ρ − 1)) | two-ring leverage |
| γ = α + β/2 | eye-stage shape combination that biases ring means |
| G = f·a | image gain (mm/deg) |
| Γ = f·γ_img·R̄² | 8-mean bias per degree, γ_img = γ + 2·k1·a |
| λ1, λ3 | coefficients of the calibrated law D̂(Δ) |
| b_null | centre bias from the unobservable null mode |
| o | static pattern offset or eye translation |

### 1.5 Four centres that must be kept distinct

1. **c_cam = c**: the camera principal point.
2. **c_grid(w, A)**: the image of the (absent) centre ray. This is the target.
3. **M8**: the mean of the 8 observed points. The 9-point mean is M9 = (8·M8 + c_grid)/9 and is unobservable.
4. **c_T**: the centre used in the undistortion d = (X − c_T)/f (transformation centre).

---

## 2. Facts used and quantities not available

### 2.1 Recorded values (from CURRENT_MODEL.md and related records; unaudited)

**P1, E1 camera, A = 0.**
- x = cx + f·tx·(1 + k1·r²), same for y.
- f = −425.1379 mm/tan, k1 = 2.92871 tan⁻², cx ≈ cy ≈ 0.
- T = 0.0078312 (maximum tangent component on training rays, gaze included).
- Tangents: tx = l8/n8, ty = m8/n8 at S8.
- Zero-gaze grid: edge magnitude e = 0.003991148, corner component k = 0.003833097. Hence ρ = 2k²/e² = 1.8447.
- Gain a = 3.1069e-4 tan/deg.
- Empirical eye-stage law (section 3.3) with c2 = 1.915e-5, c3 = 4.583e-5 deg⁻², α = −0.0665, β = −0.1028 (deg·tan)⁻¹, sx3 = sy3 = −2.60e-8, sxy = −1.02e-8, syx = −5.76e-8 tan/deg³.
- Reported 8-mean bias about 1.1 µm/deg; reported 2·edges − corners bias 0.126 µm/deg; reported fourth-order (null-mode) contribution 0.0035 µm/deg.
- Reported k1 uncertainty about 930% from zero-gaze images with design tangents; 49% with gain known and a 3×3 gaze calibration.

**P4, E2 camera.**
- x = cx + f·tx·(1 + k1·r² + k2·r⁴), same for y; tx = −l16/n16, ty = −m16/n16 at S16 (camera entrance pupil, downstream of the eye stage). cx = cy = 0.
- A = 0: T_a = 0.0228425, f = 425.137558, k1 = 2.943551, k2 = −106.4823.
- A = 4 D: T_a = 0.0226454, f = 425.137580, k1 = 2.943229, k2 = −106.1106.
- Linear accommodation law: f0 = 425.1375580, f1 = 5.4675e-6; k1_0 = 2.9435455, k1_1 = −8.069e-5; k2_0 = −106.47462, k2_1 = 0.092976 (per dioptre).
- Camera drift over 0–4 D at the sensor: f 0.0005 µm, k1 −0.0016 µm, k2 +0.00098 µm. The camera stage is effectively accommodation-independent.
- Trace field: 3×3 nine-field grid, 20 deg per axis.

### 2.2 NOT FOUND in the records (P4)

- Zero-gaze tangents p_i.
- Ring ratio ρ.
- Gaze gain per degree at each A.
- Any dependence of eye-stage tangents on A. Stable camera coefficients do not establish accommodation-invariant eye-stage tangents.
- Any bound on the P4 eye-stage null mode.
- A barrel or pincushion label.

### 2.3 Assumptions made here (all replaceable by test P4-1)

| Id | Assumption | Used for |
|---|---|---|
| A1 | Scenario A: zero-gaze grid fills T_a. Edge (±T_a, 0), corner (±T_a, ±T_a), corner radius √2·T_a = 0.032304, ρ = 2. Inconsistent with T_a containing gaze shift, so an upper bound only. | Static distortion magnitudes |
| A2 | Scenario B: P1 geometry scaled by T_a/T_P1 = 2.917. R_e = 0.01164, corner component 0.01118, R_c = 0.01581, ρ = 1.8447, a4 = 9.06e-4 tan/deg (f·a4 = 385 µm/deg, S = 0.01133 at 12.5 deg). | All P4 bias numbers |
| A3 | P4 eye-stage shape scaled from P1 (γ4 ≈ γ1/2.917, Γ4 ≈ 3.3 µm/deg). Pure guess. | Orders of magnitude |
| A4 | P4 ring radii scale with A like T_a: −0.216 %/D. Pure guess. | Accommodation illustration |
| A5 | Grid is 4-fold symmetric with two rings. | Ring-mean algebra |
| A6 | An eye translation or pattern offset shifts P1 and P4 equally. | Joint P1 + P4 counts |
| A7 | Camera magnification about 2.5 image mm per object mm at the eye. | Corneal-irregularity estimate only |

---

## 3. Equation sheet

### 3.1 Camera: forward, inverse, Jacobian, branch

**Forward.**
- E1: X = c + f·t·(1 + k1·r²)
- E2: X = c + f·t·(1 + k1·r² + k2·r⁴)

**Inverse.** d = (X − c)/f, ρ_d = |d|, g(r) = r(1 + k1·r² + k2·r⁴), solve g(r) = ρ_d, then t = d·r/ρ_d.
- Series: r = ρ_d − k1·ρ_d³ + (3k1² − k2)·ρ_d⁵ − (12k1³ − 8·k1·k2)·ρ_d⁷ + … (set k2 = 0 for E1).
- Newton: r ← r − (g(r) − ρ_d)/g'(r), start r = ρ_d, 2–3 iterations.
- Cardano (E1) is exact but ill-conditioned for small k1·ρ_d²; do not use it.

**Jacobian.** ∂X/∂t = f[(1 + k1·r² + k2·r⁴)·I + (2k1 + 4k2·r²)·t·tᵀ]
- Radial eigenvalue f·g'(r), with g'(r) = 1 + 3k1·r² + 5k2·r⁴.
- Tangential eigenvalue f·g(r)/r.
- det = f²·g'(r)·g(r)/r.

**Unique branch.**
- Need f ≠ 0 and g'(r) > 0 on the domain. The sign of f is irrelevant (negative f is a 180-degree rotation; det > 0).
- E1 with k1 > 0: monotone everywhere. With k1 < 0: r < 1/sqrt(3|k1|).
- E2 (P4 values): g' changes sign at r² = 0.05242, r = 0.229 (12.9 deg). The inverse is unique for |d| < g(0.229) ≈ 0.197. The tangential factor vanishes at r = 0.334. The field (≤ 0.032) is about 7 times inside the fold.

**Truncation sizes.**
- P1: cubic inverse error about 1e-3 µm. Adequate.
- P4 (Scenario A corner): fifth-order term 2.0 µm, seventh-order term 0.044 µm. Use fifth order plus a check, or Newton.

### 3.2 Scale gauge and image-only invariants

- Exact gauge: t → s·t, f → f/s, k1 → k1/s², k2 → k2/s⁴.
- Under the gauge the eye stage transforms as p → s·p, a → s·a, cubic shifts → s×, c2 and c3 unchanged, α → α/s, β → β/s.
- Invariants: c, m_i = f·p_i, κ1 = k1/f², κ2 = k2/f⁴, G = f·a, f × cubic shifts, c2, c3, α/f, β/f.
- Values: κ1 ≈ 1.62e-5 mm⁻² (P1 E1), 1.6286e-5 mm⁻² (P4 E2); κ2 ≈ −3.26e-9 mm⁻⁴.

**Image-space undistortion (no f).** With Q = |X − c|:
- q = (X − c)[1 − κ1·Q² + (3κ1² − κ2)·Q⁴ − (12κ1³ − 8κ1κ2)·Q⁶ + …]
- Forward: X = c + q(1 + κ1|q|² + κ2|q|⁴)

### 3.3 Eye-stage law (baseline empirical form, P1)

t_i(w) = p_i + S(w) + c2|w|²·p_i + c3(w·p_i)·w + α|p_i|²·w + β(w·p_i)·p_i

- S_x = a·w_x + sx3·w_x³ + sxy·w_x·w_y²; S_y = a·w_y + sy3·w_y³ + syx·w_y·w_x².
- For P4 the same form is a working hypothesis with coefficients depending on A.

### 3.4 Ring means

For a 4-fold symmetric ring of radius R: mean(p) = 0, mean(p·pᵀ) = (R²/2)·I. Then:
- Any field dependence of degree ≤ 3 in p gives M_ring = F(0) + (R²/2)·tr A(w). Odd degrees vanish; every quadratic form pᵀA p averages to (R²/2)·tr A.
- Tangent space: M_ring(w) = S(w) + γ·R²·w.
- Rigid shift t = p + S, exact:
  - mean(|t|²·t) = S(|S|² + 2R²)
  - mean(|t|⁴·t) = S(|S|⁴ + 6R²|S|² + 3R⁴)
- Image ring mean, camera only: X̄_R = c + f·S[1 + k1(|S|² + 2R²) + k2(|S|⁴ + 6R²|S|² + 3R⁴)]
- Centre: c_grid = c + f·S[1 + k1|S|² + k2|S|⁴] (when the centre ray has p_0 = 0).
- Image ring mean with eye-stage shape, to cubic order: X̄_R = c_grid + f·R²(2k1·S + γ·w).

### 3.5 Two-ring identity and what it leaves

- Identity: c_grid = (ρ·X̄_e − X̄_c)/(ρ − 1) = M8 − K·D.
- Exact through field degree 3 in p, for any shift S, for any c, f, k1, in image space or tangent space.
- For a field term |p|⁴ it returns −R_e²·R_c² instead of 0. For p_x⁴ + p_y⁴ it returns +0.170·R_e⁴ (ρ = 1.8447).
- **Camera residual (exact for a rigid shift): −3·f·k2·R_e²·R_c²·S.** In image units the correction is ĉ ← ĉ + 3·κ2·Q_e²·Q_c²·(ĉ − c).
- Bias of simpler estimators (P1):

| Estimator | Bias |
|---|---|
| M8 (plain mean) | f·γ_img·R̄²·w ≈ 1.12–1.14 µm/deg |
| M9 | 8/9 of the above; unobservable |
| 2·X̄_e − X̄_c (ρ = 2 on this grid) | f·γ_img·R_e²(2 − ρ)·w ≈ 0.124 µm/deg |
| Two-ring with ρ = 1.8447 | null mode only, about 0.0035 µm/deg (recorded) |

- Camera share of the P1 8-mean bias: 2·f·k1·R̄²·S ≈ 0.0175 µm/deg (1.5%, opposite sign to the eye-stage term).

### 3.6 Estimators

**E_A — instantaneous two-ring (no gaze, no A, no camera, no reference frame).**
- ĉ = (ρ·X̄_e − X̄_c)/(ρ − 1)
- Weights: ρ/(4(ρ − 1)) per edge point, −1/(4(ρ − 1)) per corner point.
- Noise: σ·sqrt(ρ² + 1)/(2(ρ − 1)) = 1.24σ at ρ = 1.8447; 1.12σ at ρ = 2.
- ρ from design, from the reference image (Q_c²/Q_e² about the reference 8-mean), or from the current frame.

**E_B — calibrated, low noise (recommended at runtime).**
- Fit the odd law D̂(Δ) = λ1·Δ + λ3|Δ|²·Δ (plus anisotropic cubic terms if needed) on reference and calibration frames.
- ĉ = M8 − K·D̂(Δ) + b_null(Δ)
- First order: ĉ − c_ref ≈ (1 − K·λ1)·Δ. Predicted for P1: λ1 ≈ −5.07e-3, K·λ1 ≈ −8.5e-3 (the centre moves about 0.85% more than the 8-mean).
- Noise: about 0.36σ per frame; 0.50σ against a single reference frame.
- Calibration term: K·|Δ|·σ(λ1), with σ(λ1) ≈ 0.707σ/(Δ_rms·sqrt(2N)).
- The law uses image quantities only. Known gaze is not required for the centre; gaze diversity with at least two magnitudes per axis is.
- Assumes D depends on gaze only (per A level for P4).

**E_T — model-based tangent-space estimator (needs known c, f, k1, k2).**
- Invert every frame to tangents. Take p_i = t_i(0). Fit θ = (a, cubic shifts, c2, c3, α, β) on calibration frames.
- Runtime: ŵ = argmin Σ_i |t_i − T(p_i, w; θ)|² (16 equations, 2 unknowns). First order: ŵ ≈ mean_i(t_i − p_i)/(a + γ·R̄²).
- Centre: ĉ = c + f·t_0(1 + k1|t_0|² + k2|t_0|⁴), with t_0 the law evaluated at p_0 = mean of the measured p_i.
- This is the two-ring identity with noise averaged over calibration gazes; it is not an independent principle. Noise about 0.36σ; gaze noise σ_t/(a·sqrt(8)) ≈ 0.16 arcmin per axis (P1).

**Map form (image only, reference-guided).**
- ξ = X0 − ĉ_0 with ĉ_0 = M8_ref. Fit Φ_w with Φ_w(ξ_i) = X_i(w); the carried centre is Φ_w(0).
- Unrestricted full quadratic: 12 parameters, 16 data, 4 check degrees of freedom. Its constant term equals E_A with ρ̂ from the reference image.
- On nominal node coordinates it is the 8-node serendipity value 2·mean(edges) − mean(corners).
- Physics-restricted map: Φ_w(ξ) = ξ + Σ(w) + c2|w|²ξ + c3(w·ξ)w + α'|ξ|²w + β'(w·ξ)ξ, 2 runtime unknowns. This is E_B / E_T written gauge-invariantly.

### 3.7 MAP prior form

- θ ~ N(θ_CV(R̂, â), Σ_θ), with Σ_θ = J_τ·Σ_τ·J_τᵀ + Σ_model.
  - θ_CV: CODE V values as a function of the measured subject radius R̂ (and gain â if needed).
  - J_τ: CODE V finite-difference Jacobian with respect to tolerances τ (radius, distances, decentre, tilt).
  - Σ_model: floor for asphericity or toricity mismatch.
- Estimate: minimise Σ|r|²/σ² + (θ − θ_CV)ᵀ·Σ_θ⁻¹·(θ − θ_CV).
- Posterior covariance: (JᵀJ/σ² + Σ_θ⁻¹)⁻¹.

---

## 4. What is and is not identifiable

### 4.1 Camera parameters from eye images

- f and k1 (and k2) cannot be fitted from reference plus known-gaze calibration images. The scale gauge is exact even with known gaze, because the gain a is free.
- Minimal absolute anchor: one scalar with tangent units from outside the image fit (a, f, one |p|, or absolute α or β). Then f = G/a, k1 = κ1·f², df/f = −da/a, dk1/k1 = dκ1/κ1 − 2·da/a.
- For relative work (centre in µm, gaze through G in mm/deg) no anchor is needed.

### 4.2 The k1 alias with the eye stage

Expanding the camera cubic with t ≈ p + a·w:

k1|t|²·t = k1[ |p|²p + 2a(w·p)p + a²|w|²p + a|p|²w + 2a²(w·p)w + a³|w|²w ]

So k1 → k1 + δ is compensated by β → β − 2aδ, c2 → c2 − a²δ, α → α − aδ, c3 → c3 − 2a²δ, all cubic shifts → − a³δ, p_i → p_i(1 − δ|p_i|²). The residual is fifth order, about 0.03 µm at 12.5 deg (P1).

- With free m_i and free eye-stage coefficients, κ1 is unidentifiable at 1 µm noise.
- Eye-stage terms as "equivalent k1": α/a ≈ −214, β/(2a) ≈ −165, c2/a² ≈ +198, c3/(2a²) ≈ +237, s3/a³ ≈ −867, against k1 = 2.93.
- **Conflict to resolve:** the recorded "49% with gain known" must have held eye-stage coefficients or p_i fixed. Check which.
- Best case (zero gaze, design tangent shape supplied): residual signal about 0.108 µm, σ(k1)/k1 ≈ 925% at 1 µm. Matches the recorded 930%.

### 4.3 Parameter counts and ranks

| Fit | Unknowns | Data | Rank |
|---|---|---|---|
| Zero-gaze frame, free m_i | 19 (c 2, m 16, κ1 1) | 16 | c and κ1 not identifiable |
| Zero-gaze frame, 4-fold grid, free radii | c 2, two radii, κ1 | 16 | κ1 not identifiable (2 radii for 3 unknowns) |
| Reference + N known gazes, camera free (P1) | 29 raw | 16(1 + N) | 28 exact (gauge); 27 numerical (k1 alias) |
| Same, camera known | 25 (p_i 16, eye coefficients 9) | 16(1 + N) | 25 with the gaze design below |
| Same, plus a free \|p\|⁴w coefficient | 26 | 16(1 + N) | 25 (null mode) |

- c is weakly determined (leverage 0.3–0.6% through the shape terms) and only by assuming the symmetry-forbidden terms vanish. Rough σ(c) ≈ 10 µm for N = 25. Conversely a 1 µm error in c matters at 0.003–0.006 µm.
- Gaze design needed for full rank:
  - At least 2 distinct nonzero |w| per axis. Levels {−W, 0, +W} alias a with s3 (the recorded 3×3 failure).
  - Off-axis gazes with w_x·w_y ≠ 0.
  - Both signs.
  - Minimal set: (±W, 0), (±W/2, 0), (0, ±W), (0, ±W/2), (±W, ±W) = 12 gazes plus the reference. 5×5 is the safe choice.
- Which data determine which coefficient (camera known):
  - β, c2, c3: within-ring structure.
  - α against S (hence a and the cubic shifts): only the R² dependence of the ring means, i.e. the two-ring extrapolation.

### 4.4 The null mode

- N(p) = (1 − |p|²/R_e²)(1 − |p|²/R_c²) is 0 on all 8 points and 1 at the centre. On the nominal square grid it is (1 − x²)(1 − y²).
- T' = T + δ·w·N(p) gives identical 8 tangents and identical 8 images for every δ and every camera, while the centre moves by f·δ·w.
- Known f, k1, k2, c do not break it. It is a field-shape ambiguity of the eye stage. E1/E2 is a pointwise bijection, so it cannot add information about an unobserved field point.
- On the nominal tensor grid x⁴ = x² and x²y² = x² + y² − 1 at the nodes, so quartic content aliases silently into the constant and quadratic terms of any fitted map.
- The centre is therefore constrained by the 8 points only as a polynomial extrapolation of degree ≤ 3. Only the design trace (which contains the centre ray) or a third ring radius can bound the rest.
- Physical form of the null mode in a real eye: a local surface slope error at the centre glint that the other 8 glints do not share. Rough size about 0.02 µm per µrad (assumption A7), so about 50 µrad gives 1 µm.

### 4.5 Static offsets

- Ring centroid versus centre ray: the centre source does not exist, so c_grid is defined as the image of the ring-centroid field point. The static offset is a definition and cancels in relative changes.
- Centre ray versus principal point: image-space estimators are immune through cubic order.
- Only a gaze-dependent leak survives: (c2|w|² + c3·w·wᵀ + O(α|p||w|))·o ≈ (3–9)e-3 × o at 12.5 deg.
- For gaze, an unreferenced offset costs 0.0076 deg/µm (recorded), i.e. 0.45 arcmin per µm. Always difference against the reference frame.

### 4.6 The definition of f (open)

- Recorded: tangents are chief-ray tangents at S8 (P1) and S16 (P4), described as the camera stop or entrance pupil. Under that definition f, k1, k2 are camera-side constants, the mirror lives in the eye stage (p_i, a, α, β), and a model-eye f transfers to any eye.
- User's description: f is the combined mirror + camera focal length. If the tangent were defined before the mirror, f would change with corneal radius (δf/f ≈ δR/R, several percent across subjects) and the gauge would reopen for every subject.
- Which holds depends on where S8 sits relative to the mirror. Not checked.
- Paraxial combination, if the combined definition is used: f_m = −R_m/2 (convex mirror, converging-positive convention; |f_m| ≈ 3.9 mm for 7.8 mm), f_comb = f_m·f_c/(f_m + f_c − d), ∂ln f/∂ln f_m = (f_c − d)/(f_m + f_c − d) ≈ 1. The system is at finite conjugates, so the measured quantity is a magnification-like m_i, not an infinite-conjugate focal length.

### 4.7 Flat-grid κ versus E1/E2 k1

A flat grid gives X − c = M·h(1 + κ1·M²h² + κ2·M⁴h⁴) with relative object heights h known. It identifies M, κ1, κ2, c from images alone and returns κ natively, not k1. It equals the E1/E2 coefficient only if:
- (a) the S8/S16 tangent is the object-space tangent at the entrance pupil (no pupil distortion in between);
- (b) image positions behave as chief-ray intercepts (then the stop-direction-to-sensor map is conjugate-independent; blur-centroid shifts break this);
- (c) the entrance-pupil position is known (otherwise a scale error, or a κ error if it is field-dependent).

Further:
- Mirror-induced distortion is never in the flat-grid value, and in the stop-tangent formulation need not be.
- Precision needs field coverage: about 925% on the P1-sized field; about 10% for a grid filling roughly ±4 mm with a few hundred points (rough).
- P1's k1 = 2.9287 is close to the P4 law truncated over the P1 field: 2.9436 − 106.5 × 1.4e-4 ≈ 2.929. One camera law (κ1, κ2) probably serves both channels.

### 4.8 Sign and naming

- With x − c = f·u·(1 + k1·r²) and k1 > 0, |x − c| grows faster than r. That is **pincushion** in the tangent-to-image direction, for either sign of f.
- The image-to-tangent inverse has coefficient −κ1 and looks barrel.
- P4: k1 > 0 with k2 < 0 is pincushion with a barrel-going fifth-order roll-off. The distortion ratio peaks at r = 0.118 (+2.03%), outside the field.
- A barrel-like appearance of the glint pattern is consistent with the convex-mirror eye stage (γ < 0 compresses the outer ring), not with the camera.

---

## 5. Matched model eye, flat grid and CODE V

### 5.1 What each instrument gives

| Instrument | Gives | Does not give |
|---|---|---|
| Flat grid | κ1, κ2, c (image-only) | f; anything about the eye stage |
| Model eye of known radius | m_i = f·p_i^ME; f after computing p_i^ME from known geometry; check of the E1/E2 form; D_ref = 0 symmetry; with a gimbal, the D(Δ) law against known geometry | the real eye's p_i or a; the centre (unobservable on the model eye too) |
| CODE V matched to both | null-mode bound for design surfaces; dimensionless shape ratios; the form of the A-law; tolerance Jacobian | real corneal or lens irregularity, tilt, decentre; the subject's A-law |

- Uncertainty of a model-eye f: δf/f = δQ/Q ⊕ δp/p, with δQ/Q ≈ 2e-4 at 1 µm and δp/p ≈ δR_m/R_m ⊕ δL/L. A 0.1% radius tolerance gives about 1e-3.
- If f is camera-side, the absolute anchor is closed: real-eye p_i come from inverting the real reference frame, real-eye a from the known-gaze calibration, and p_i(real)/p_i(model eye) is a keratometric radius estimate.

### 5.2 Tolerance propagation (centre budget 0.1 µm at 12.5 deg)

n_R and n_L are the log-sensitivities of Γ/G to radius and distance. Expected O(1–2) but not derived; they must come from CODE V finite differences.

| Tolerance | Enters through | P1 | P4 (assumptions A2–A4) |
|---|---|---|---|
| Centroiding σ | M8 | 0.36σ per frame | same |
| Camera f | nothing in image space | any | any |
| k1 / κ1 | fifth order only | ≤ 0.03 µm × δk1/k1 | loose with undistortion; raw two-ring leaves 0.02–0.06 µm cross terms |
| k2 / κ2 | null projection | not applicable | 0.052 µm × δk2/k2 (scales as R⁴) |
| Principal point / decentre | leak | 11–33 µm allowed | similar; immune inside the two-ring step |
| Radius, if a CODE V Γ is used unscaled | Γ | 14 µm × n_R × δR/R, so δR/R ≤ 0.7%/n_R | about 41 µm × n_R × δR/R, so about 0.25%/n_R |
| Working / source distance | Γ | δL/L ≤ 0.7%/n_L | about 0.25%/n_L |
| Ring-radius (grid placement) error | ρ | 0.945 µm/deg per unit δρ; none if ρ is taken from the reference image | same form |
| Grid anisotropy ξ | β term | about 1 µm/deg × ξ; none with point-wise measured p_i | same form |
| Accommodation error in Γ(A) | Γ | none | about 0.17 µm/D, so A to about 0.5 D |
| Null mode | prior | 0.044 µm in total; 100% uncertainty acceptable | unknown; must be known to 0.1 µm absolute |
| Local surface irregularity | null mode | about 0.02 µm/µrad (A7); not boundable | not boundable |
| Inverse series order | undistortion | cubic adequate | fifth order or Newton required |

### 5.3 Priors versus subject-specific quantities

**Subject-specific (measure; never fix at design).**
- p_i: scale with corneal (or lens-surface) radius; read from the reference frame at about 2e-4 relative.
- a: scales with rotation-centre geometry; from the known-gaze calibration (about 4e-5 relative for 5×5).
- Γ in absolute terms: a ±8% radius spread would give about 1.1–2.3 µm error at 12.5 deg if fixed at design.
- For P4: everything as a function of A, and the static offset o(A).

**Usable as CODE V priors, in dimensionless gauge-invariant form, tabulated against measured R̂.**
- s3/a, sxy/a and the other cubic-shift ratios (the sin w expansion alone gives −1.58e-8 of the reported −2.6e-8).
- c2, c3.
- α|p|, β|p|, or equivalently λ1, λ3.
- b_null.
- The functional form of the A-dependence.

**Conjecture to test.** λ = dD/dΔ may be nearly independent of rotation-centre distance, since both the shift and the shape terms are driven by displacement of the centre of curvature. If so, the Δ-based law transfers across subjects better than a w-based law.

**Gain from priors (P1, centre at 12.5 deg; per-frame noise 0.354 µm in every case).**

| Calibration | Calibration term | Total |
|---|---|---|
| Free 5×5 | 0.24 µm | 0.43 µm |
| 5×5 with 1% prior on λ | 0.12 µm | 0.38 µm |
| Free 4-point (±W per axis) | 0.60 µm | 0.70 µm |
| 4-point with 1–2% prior | 0.14–0.26 µm | 0.38–0.44 µm |
| Any, with 5% prior | no real help over 5×5 | |

- Centre noise reduction is at most about 20%. The value of priors is bias control, fewer calibration gazes, removal of the a/s3 alias, and the null-mode bound.
- Gaze: per-frame noise (0.16 arcmin) is image-limited and unchanged. A prior on a is useless. Priors do not rescue the free-offset fit (0.14–0.47 deg).

### 5.4 Validation protocol

**Setup.** Model eye on a gimbal with known radius and rotation centre; ideally 2–3 radii plus one aspheric or toric surface. Reference frame plus at least 5 gaze levels per axis plus diagonals.

**Compare image-only invariants with CODE V.**
1. Ring radii Q_e, Q_c and ρ.
2. G (mm/deg) and the cubic shift.
3. The D(Δ) law (λ1, λ3).
4. Within-ring second harmonic (β', c3) and the |w|² scaling (c2).
5. D_ref = 0 and residual maps.
6. Scaling of items 1–4 with radius against the CODE V family.

**Agreement proves:** camera chain, geometry and prescription are correct on the 8 observable field points for spherical surfaces across the tested radii, so the priors and the tolerance Jacobian can be trusted for that family.

**Agreement does not prove:**
- The centre itself; every centre statement is inherited from the model. A test-only grid with a third ring radius would measure the R⁴ coefficient; an R⁶ null mode would remain.
- Real-eye asphericity, toricity or irregularity.
- Tear-film effects.
- A non-fixed rotation centre (eye translation).
- Accommodation behaviour of a real lens.

---

## 6. P1 and P4 specifics, accommodation, joint estimation

### 6.1 P1

- Camera distortion is tiny: 0.08 µm at the edge, 0.14 µm per corner component at zero gaze, about 0.6 µm at the field limit.
- The 8-mean bias (1.12–1.14 µm/deg) is 98.5% eye-stage shape. Undistortion changes it by 0.0175 µm/deg and cannot remove it.
- The two-ring identity cancels f, k1, c through cubic order; residual about 1e-3 µm plus the recorded null mode 0.0035 µm/deg (0.044 µm at 12.5 deg).
- Undistortion is optional. E1 inversion buys nothing for the centre.
- Physically the corneal reflection should not depend on A, but the records cover A = 0 only.

### 6.2 P4 camera distortion

| Location (Scenario A) | k1·r² | k2·r⁴ | Net | In µm |
|---|---|---|---|---|
| Edge, r = T_a | +1.536e-3 | −2.90e-5 | +1.507e-3 | 14.6 µm on 9.711 mm |
| Corner, r = √2·T_a | +3.072e-3 | −1.160e-4 | +2.956e-3 | 28.7 µm per component; 40.6 µm radial |

About 25 times P1 (2.917³).

**Camera-induced centre bias (Scenario B).**

| Term | Per degree | At 12.5 deg, one axis |
|---|---|---|
| 8-mean, k1 part | 0.437 µm/deg | 5.5 µm |
| 8-mean, 3k2R⁴ part | −0.005 µm/deg | −0.06 µm |
| 8-mean, 6k2R²\|S\|² part | cubic in gaze | −0.08 µm |
| Two-ring, k2 residual | +0.0042 µm/deg | 0.052 µm (0.074 µm radial on the diagonal) |

- Under Scenario A radii the two-ring residual is about 16 times larger (about 0.8 µm).
- A 0.1 µm budget at f·S = 4.8 mm needs R_e²·R_c² ≤ 6.5e-8, i.e. R_e ≤ about 0.0137. Scenario B has a margin of only 1.9.
- Camera × eye-stage cross terms: about 0.02–0.06 µm under A3 (they were 1e-3 µm in P1).
- The wider field does not change the order of exactness of the identity (still degree 3). It multiplies the degree-4 residual by about 72.
- Remedies: the closed-form κ2 correction (section 3.5), or undistort, apply the two-ring step in q-space, and map back. Undistortion is recommended for P4.
- The camera share of the 8-mean bias rises to about 13% (A3). The dominant part is still eye-stage shape.
- The P4 eye-stage null mode is unknown. If the |p|⁴w coefficient equalled P1's, R⁴ scaling would give about 0.25 µm/deg. Speculative, but potentially the largest term.

### 6.3 Accommodation

- The camera is A-independent, so one undistortion map serves all A. Every A-dependence is in the eye stage: p_i(A), a(A), γ(A), cubic shifts, c2(A), c3(A), and for a tilted or decentred lens a static offset o(A).
- **E_A is A-free and gaze-free.** No reference or re-referencing. ρ is unchanged if p(A) scales uniformly; otherwise take ρ̂ from the current frame. Its only bias is b_null(A, w).
- **E_B must be re-referenced per A:** λ1(A), λ3(A), M8_ref(A).
  - Three calibration levels give a linear law with 1 check degree of freedom, or an exact quadratic with none. Validating a quadratic needs a fourth level or a CODE V prior on the form.
- **Observables for A from the 8 points:** ring radii Q_e(A), Q_c(A); gain G4(A); λ(A). Under A4 the edge radius changes about −10.7 µm/D against 0.35 µm radius noise, so σ_A ≈ 0.03 D (illustration only).
- **Reference frame at unknown A:** usable, because it labels its own A through its ring radii, provided Q(A) was calibrated and is monotone.
- **Confound:** c2|w|² also rescales the pattern (about 3e-3 at 12.5 deg with the P1 value, equivalent to about 1.4 D). Solve jointly with w; a 1% error in c2 costs about 0.014 D.

### 6.4 Runtime identifiability with A and gaze both unknown

| Case | Unknowns | Data | Rank condition | Outcome |
|---|---|---|---|---|
| P4 alone, no pattern offset | w (2), A (1) | 16 | dQ/dA ≠ 0 | Identifiable |
| P4 alone, with offset o | w, o, A = 5 | 16 | w and o separate only through 1–2% shape terms | Gaze poor (0.14–0.47 deg as in P1) |
| P1 + P4 jointly | w, o, A = 5 | 32 | G1 ≠ G4 and dQ4/dA ≠ 0 | Identifiable at leading order |

- Joint leading-order equations: M8_P1 = o + G1·w; M8_P4 = o + G4(A)·w; plus the P4 scale.
- Joint gaze noise: √2 × 0.354 µm / |G4 − G1| ≈ 0.12 arcmin (A2).
- **A robust unknown-A, unknown-gaze, unknown-offset estimator needs P1 and P4 jointly.** P4 alone suffices only if the offset is known to be zero.
- Counterexamples: Q(A) flat or non-monotone (A unobservable); G1 = G4 (w and o inseparable); o(A) of a tilted lens mimicking gaze through a stale M8_ref.

---

## 7. Staged procedure, noise budget, test matrix

### 7.1 Stages

Nothing below is authorised. Each stage starts only on the user's explicit approval, and each stage's exit is the user's acceptance of its report. Roles are generic: the user approves; a trace operator runs CODE V; an analyst runs fits on saved arrays; an independent auditor checks the result. Assignment to specific agents is the user's choice.

| Stage | Content | Entry criterion | Exit criterion | Stays unauthorised until approved |
|---|---|---|---|---|
| 0 | Read-only checks of existing records and lens files: S8/S16 position versus the mirror, f definition, sign convention, what was fixed in the "49%" fit | User approval of this document | Written answers to the three questions; no trace, no fit | Any CODE V run |
| 1 | P4-1 extraction from existing saved traces: zero-gaze p_i, ρ, gain per A, eye-stage A-dependence | Stage 0 accepted | Assumptions A1–A4 replaced by recorded values, or declared unavailable | New traces |
| 2 | P1 analytic checks on saved arrays: ring identities, biases, gauge and alias, estimator noise | Stage 1 accepted | Each P1 test reported as prediction versus result, with audit | P4 fits, new traces |
| 3 | P4 camera and centre checks on saved arrays: undistortion order, camera residual, null-mode measurement at A = 0, 2, 4 | Stage 2 accepted | Null-mode value n(w, A) reported with audit; decision on undistortion | Joint estimator, new traces |
| 4 | New CODE V perturbation traces: radius, distance, decentre, tilt, conic, toric, rotation centre, intermediate A, third ring | Stage 3 accepted and trace list approved | Tolerance Jacobian J_τ and A-law form reported | Hardware work |
| 5 | Joint P1 + P4 estimator with unknown A, gaze, offset; Monte Carlo | Stage 4 accepted | Noise and bias versus targets reported with audit | Hardware work |
| 6 | Model-eye and flat-grid protocol (section 5.4) | Stage 5 accepted | Protocol document only | Any claim about real eyes |

### 7.2 Noise budget at 1 µm per coordinate

| Quantity | P1 | P4 |
|---|---|---|
| Tangent per component, σ/\|f\| | 2.35e-6 | 2.35e-6 |
| Plain 8-mean (biased) | 0.354 µm | 0.354 µm |
| E_A | 1.24 µm | 1.12–1.24 µm |
| E_B / E_T per frame | 0.36 µm | 0.36 µm |
| E_B against a single reference frame | 0.50 µm | 0.50 µm |
| Calibration term, free 5×5 | 0.10 µm (5 deg) to 0.24 µm (12.5 deg) | the same × Γ4/Γ1, per A level, plus A-interpolation |
| Calibration term with 1% prior | about 0.12 µm at 12.5 deg | not estimated |
| ρ̂ from the reference image | about 0.016 µm at 12.5 deg | not estimated |
| Gaze, 2-unknown fit | 0.16 arcmin per axis | not estimated |
| Gaze, free-offset fit, single channel | 0.14–0.47 deg | similar expected |
| Gaze, joint P1 + P4 | about 0.12 arcmin (A2) | |
| A from P4 ring scale | not applicable | about 0.03 D (A4) |

### 7.3 Consolidated minimal test matrix (design traces and saved arrays only)

All predictions are unverified hand arithmetic.

**Stage 0 — read-only.**

| Id | Test | Expected outcome |
|---|---|---|
| S0-1 | Locate S8 (P1) and S16 (P4) relative to the mirror and the camera | Decides camera-only versus combined f |
| S0-2 | Evaluate E1 at the corner | Pincushion for k1 > 0 |
| S0-3 | Identify what was held fixed in the recorded "k1 49%" fit | Eye-stage coefficients or p_i fixed |

**P1.**

| Id | Test | Prediction |
|---|---|---|
| P1-1 | Zero gaze | D_ref = 0 and M8 = c_grid |
| P1-2 | Single-axis gaze at W/2 and W, and diagonal (W, W): E_A versus traced centre | Null mode about 0.0035 µm/deg; diagonal isolates 4-fold quartic |
| P1-3 | ρ = 2, 1.8447, image-derived | 0.124 µm/deg, ~0, ~0 |
| P1-4 | Camera only, synthetic rigid shift, k1 on/off | M8 bias = f·k1·S(R_e² + R_c²); two-ring bias 0 |
| P1-5 | Raw versus undistorted 8-mean and two-ring | 1.12, 1.14, about 0.0035 µm/deg |
| P1-6 | Quadratic map in reference coordinates versus E_A versus serendipity | First two identical; third 0.124 µm/deg |
| P1-7 | Gauge s = 1.1; k1 + δ with compensated eye coefficients | Identical images; residual about 0.03 µm |
| P1-8 | Consistent perturbation in calibration and runtime: f × 1.001, k1 × {0, 1, 2, 10}, c + 100 µm | Centre change < 0.1 µm; gaze unchanged |
| P1-9 | Inconsistent use: design θ with perturbed f, k1, and γ ± 10% | 0.03 µm, 0.02 µm, 1.4 µm at 12.5 deg |
| P1-10 | Decentre: pattern offset from the principal point and from the ring centroid; asymmetric grid | Leak (3–9)e-3 × offset; point-wise model unaffected; design-ρ two-ring biased |
| P1-11 | Calibration design 3×3 versus 5 levels, scored on hold-out gazes | a/s3 and λ1/λ3 alias in 3×3 |
| P1-12 | Monte Carlo at 1 µm: E_A, E_B, E_T (2 and 4 unknowns), free and MAP calibration (1%, 2%, 5% priors; 5×5 and 4-point) | Section 7.2 and 5.3 values |

**P4.**

| Id | Test | Prediction |
|---|---|---|
| P4-1 | Extract zero-gaze p_i, ρ, a4 at each A | Replaces A1–A4 |
| P4-2 | Pure-camera rigid shift | Ring bias f·S[2k1R² + k2(6R²\|S\|² + 3R⁴)]; two-ring residual −3·f·k2·R_e²·R_c²·S |
| P4-3 | Raw two-ring versus undistorted two-ring versus traced centre, on-axis and diagonal, A = 0, 2, 4 | Isolates n(w, A) |
| P4-4 | Inverse accuracy: cubic, fifth order, Newton | 2 µm and 0.04 µm truncation at the Scenario A corner |
| P4-5 | Q_e(A), Q_c(A), ρ(A), G4(A), λ(A), with one intermediate A (1 or 3 D) | Tests the A-law form |
| P4-6 | c2\|w\|² versus A confound | Rank 3 of the (w, A) fit |
| P4-7 | k2 ± 100% and c ± 100 µm in the undistortion | 0.052 µm × δk2/k2; c immune in the two-ring step |

**Joint and CODE V perturbations.**

| Id | Test | Prediction or output |
|---|---|---|
| J-1 | Joint P1 + P4 with injected common offset o | Rank 5; gaze noise about 0.12 arcmin (A2) |
| J-2 | One (κ1, κ2) fitted to P1 and P4 together; flat-target trace at two distances | P1 k1 equals truncated E2; same κ under section 4.7 conditions |
| J-3 | Lens tilt or decentre perturbation | o(A) drift and stale-reference gaze error |
| J-4 | Monte Carlo at 1 µm, E_A and E_B per A level, joint estimator | Section 7.2 values |
| C-1 | CODE V perturbations: radius ±1% and ±8%, distance ±1 mm, decentre ±50 µm, tilt, conic Q = −0.26, toric, rotation centre ±1 mm; tabulate G, ρ, λ1, λ3, b_null | n_R, n_L and J_τ |
| C-2 | Two corneal radii (e.g. 7.4 and 8.2 mm), refit camera | f, k1 unchanged if camera-only; p_i and a scale with radius |
| C-3 | λ versus rotation-centre distance | Invariance conjecture |
| C-4 | Third-ring field points (both channels) | R⁴ coefficient and b_null |

---

## 8. Open questions and risks, by priority

1. **P4-1: the true P4 zero-gaze tangents, ρ and gain per A.** Every P4 bias number here rests on assumptions A1–A4. The camera residual scales as R⁴ and has only a factor 1.9 margin under Scenario B.
2. **Position of S8/S16 relative to the mirror, and hence the definition of f.** If f contains the mirror, a model-eye f does not transfer to a subject and the anchor question reopens.
3. **Null-mode bound for P4.** No recorded bound. It may be the largest centre error and cannot be measured from 8 points on any eye.
4. **Eye-stage accommodation dependence for P4.** Unknown form, unknown monotonicity of Q(A). Three calibration levels cannot validate a quadratic law.
5. **Real-eye departures from the design surfaces.** Local irregularity (the physical null mode), lens tilt and decentre giving o(A), eye translation, tear film. None is boundable by CODE V or by a model eye.
6. **The recorded "k1 49%" result.** It conflicts with the alias unless eye-stage quantities were held fixed.
7. **Whether the P1 empirical eye-stage form (field degree ≤ 2 in the gaze-dependent shift) holds for P4.** If not, λ needs more terms and the two-ring order statement must be rechecked.
8. **Flat-grid κ equivalence.** Conditions (a)–(c) of section 4.7 are unchecked. The consequence is small for the centre (tolerances are loose) but matters for absolute tangents.
9. **Sign convention.** "Barrel" versus the pincushion sign of E1/E2 should be settled so that priors and corrections are applied with the right sign.
10. **Equal-offset assumption A6** for P1 and P4 under eye translation. If the two channels respond differently, the joint rank argument changes.
11. **Log-sensitivities n_R, n_L** are guessed as O(1–2). The radius and distance tolerances in section 5.2 scale directly with them.
12. **End-to-end targets** (accommodation < 0.1 D, gaze < 1 arcmin, both unknown at runtime) are not established by anything in this document.
