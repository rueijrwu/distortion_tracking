# P1 and P4 distortion transformations

**Collection status (2026-10-10):** Both replacement sweeps have validated. P1 contains 184,059 records across 51 absolute THI planes, 401 RC rotations, and nine fields per state; its pickle SHA-256 is `e29d226a0af1e5db8c3cfbf673fa92100917825f8c25d74ae71f994151519f02`. P4 contains 184,059 records across 51 accommodations (0 to 5 D at 0.1 D steps), fixed THI = 0 mm, 401 RC rotations, and nine fields per state; its pickle SHA-256 is `1d68ca2c237c76fa06539efce4911fe9ff87e203c0b11051eb4f013cbafe4b5a`. The prior P1/P4 radial and rotation fits are historical/stale; their generated result folders and old P4 grid PNG have been removed, while source scripts remain. The separate P4 Z magnification study and script remain useful prior work on their recorded lens revision.

**Consolidation note (2026-11-05):** This document now also incorporates the former `MODEL.md` (the fitted P4 joint corner model) and `MODEL_FORMULAS_REVIEW.md` (the power-law barrel variant and the general projective-keystone framework). Both files have been removed; their content lives in the "P4 joint corner model" and "Power-law barrel variant" sections below.

## Current hybrid model and geometric interpretation (2026-11-08)

This section describes the [implemented 87-scalar calibration](Summary.md#current-audited-status-2026-11-08), including its empirical corrections and limits. The older four-corner, radial, and A=0 analyses below retain their original equations and dated results, but their use of “current” is historical. The implemented model is **not** a complete absolute-coordinate physical lens forward/inverse model. It is a geometry-motivated P4 relative-field family combined with independent empirical P1/P4 centroid and gaze maps and a separate P4 central-motion normalization.

### Frozen A=0 centroid and gaze channels

Let `j=1,4` identify the channel and `g=(ADE,BDE)`. Each recorded nine-ray grid has arithmetic mean `m_j` and a **labeled central ray** `c_j`; neither is automatically the optical principal point or distortion pivot. The current A=0 runtime estimates `c_j=m_j-B_j(g)` with a separately fitted, anchored centroid-bias model. With `x=BDE/12.4` and `y=ADE/12.4`, both channels use the same form but their own measured zero-gaze anchors and coefficients:

$$
\begin{aligned}
B_{j,x}(g)&=b_{j0,x}+\beta_{j1}x+\beta_{j2}xy+\beta_{j3}xy^2+\beta_{j4}x^3,\\
B_{j,y}(g)&=b_{j0,y}+\gamma_{j1}y+\gamma_{j2}y^2+\gamma_{j3}x^2+\gamma_{j4}y^3+\gamma_{j5}x^2y.
\end{aligned}
$$

The nine `β/γ` scalars per channel were fitted independently on 25 unique A=0 gaze poses. The measured anchor is separate: P1 `b_10≈(0,−0.05801502493)` mm and P4 `b_40≈(0,0)` mm. Both calibration grids pass the BDE reflection parity check; no ADE parity was imposed. The P1 data have no accommodation dimension, so repeating a P1 grid at 21 P4 accommodation levels adds no independent P1 information.

Each channel also has its **own** direct gaze calibration on its corrected absolute central-ray coordinate. With its own training-only coordinate mean and scale, let `z_j=(c_j−μ_j)/σ_j` and

$$
\phi(z_x,z_y)=(1,z_x,z_y,z_x^2,z_xz_y,z_y^2,z_x^3,z_x^2z_y,z_xz_y^2,z_y^3),\qquad
G_j(c_j)=\phi(z_j)C_j,
$$

where each `C_j` is an independently fitted 10×2 matrix: ten vector coefficients, or 20 scalar components. `G1` and `G4` are empirical inverse calibration maps, **not** lens-distortion equations and not a single P4−P1 difference map. Together B1/B4/G1/G4 contain 58 fitted scalars. The current P1 channel has no fitted full nine-ray projective/Brown/radial field transformation. The equal-form statement for P1/P4 currently applies to these B/G channels only.

### P4 relative-field family: projective, decentering, and radial terms

The P4 field module starts from the **measured real-ray** A=0, zero-gaze center-relative nine-ray reference `R_{0,k}` (field 4 is zero), with fixed `L=1.235429592627` mm. Define `a=A/4`, `t_x=BDE/20`, and `t_y=ADE/20`. The selected SparseModel composition is **projective first**, then Brown decentering, then a one-power radial factor. Its on-axis magnification is `s(a)=1+s_1a`, and for each reference field:

$$
\begin{aligned}
v_k&=s(a)R_{0,k}/L, &D_k&=1+q_xv_{k,x}+q_yv_{k,y},\\
u_{k,x}&=(m_{11}v_{k,x}+m_{12}v_{k,y})/D_k,
&u_{k,y}&=(m_{21}v_{k,x}+m_{22}v_{k,y})/D_k,\\
w_{k,x}&=u_{k,x}+2p_1u_{k,x}u_{k,y}+p_2(3u_{k,x}^2+u_{k,y}^2),\\
w_{k,y}&=u_{k,y}+p_1(u_{k,x}^2+3u_{k,y}^2)+2p_2u_{k,x}u_{k,y},\\
\kappa(a,g)&=g_{\rm rad}(t_x,t_y,a)+k_1a+k_2a^2,
&H_k(A,g)&=Lw_k\bigl(1+\kappa\lVert w_k\rVert^2\bigr).
\end{aligned}
$$

Here `m11,m22` start at 1, and `m12,m21,qx,qy,p1,p2,g_rad` start at 0. Their 12 gaze-spatial terms are `t_x²,t_y²` for each of `m11,m22,g_rad`; `t_xt_y` for `m12,m21`; `t_x` for `qx,p2`; and `t_y` for `qy,p1`. Each coefficient function equals its **frozen A=0 base** plus `a` times one newly fitted interaction of the same spatial form. The additional three fitted values are `s1,k1,k2`; all 15 new terms vanish at A=0. The 12 frozen PF12 coefficients were originally expressed in powers of gaze `/12.4`; SparseModel uses `/20`, so a base coefficient multiplying `t_x^i t_y^j` is converted as `c_new=c_old(20/12.4)^(i+j)` without refitting. That conversion, the fixed reference, `L`, and the projective-first order give exact A=0 PF12 reduction.

`qx,qy` are projective-denominator, **keystone-like** terms; `m` contributes affine stretch/shear; `p1,p2` are Brown tangential/decentering terms. The sign of `κ` contracts or expands radius locally after decentering and can express barrel-like or pincushion-like behavior. The radial factor here has `\lVert w\rVert²` only: `k2 a²` changes radial **strength with accommodation** and is **not** a radial `r⁴` coefficient. No inverse barrel-removal step or fitted registration is applied to the raw reference. These transform families are physically suggestive but their fitted coefficients are empirical surrogates for this ray-traced grid, not identified lens prescriptions or a derivation through optical surfaces and the sensor projection.

The actual P4 field prediction used for calibration and accommodation profiling is not bare `H`. With `\bar H(A,g)=\frac19\sum_k H_k(A,g)` and the frozen independently fitted `B4(g)`, it is

$$
F_k(A,g)=H_k(A,g)+\omega_k\bigl[B_4(g)-\bar H(0,g)\bigr],\qquad
\omega_{\rm center}=0,\quad \omega_{\rm eight\ outer}=9/8.
$$

Consequently `F_center=0` and `mean_k F_k=B4(g)+mean_k H_k(A,g)−mean_k H_k(0,g)`. At A=0 the mean is exactly the frozen B4 prediction. This is a **discrete correction to eight sampled rays** to anchor a nine-ray centroid, not a continuous distortion field or physical registration transform. It changes outer-ray positions, so the final `F` grid itself was checked for cell orientation: all 665,832 primary final-state triangle signs were preserved relative to the raw A=0 reference, with minimum absolute area ratio 0.84725. That sampled-grid check does not prove a globally invertible warp between sampled fields.

The separate P4 central-motion map is a two-value diagonal normalization about the **measured** A=0 on-axis central ray `o4=(0,0)` mm:

$$
T_A(c_0)=o_4+\operatorname{diag}(1+k_xa,1+k_ya)(c_0-o_4),\qquad
k_x=-0.0413830971,\quad k_y=-0.0414280673.
$$

It has `T_0=identity` and positive sampled 0–4 D scales (minimum about 0.95857). M1 was selected over a four-value quadratic scale by one-SE grouped calibration CV (normalized-center RMS 7.0416 versus 7.0378 µm). `T_A` is an empirical **absolute-center normalization**, distinct from the relative-grid transform `F`. It should not be interpreted as an independently established physical movement of an optical center. The measured on-axis labeled P4 ray is exactly `[0,0]` mm at A=0, 2, and 4 D in the 75 calibration poses; that supports the no-translation fit on those sampled points, not at every possible state.

### Implemented inverse and validation boundaries

P1 first solves its own centroid fixed point `g1=G1(m1−B1(g1))`, without P4 gaze fusion. The observed P1 estimate seeds an accommodation profile of the **mean-centered observed P4 nine-ray grid** against `F(A,g1)`. The P1-conditioned profile gives `(A_seed,g1)`, the primary start. For a current P4 state `(A,g)`, the update is

$$
c_4=m_4-\bar F(A,g),\qquad c_{40}=T_A^{-1}(c_4),\qquad
g^*=\operatorname{clip}_{\pm12.5^\circ}G_4(c_{40}),\qquad
A^*=\arg\min_{0\le A'\le4}\sum_{k=1}^9
\left\|(y_{4,k}-m_4)-\bigl(F_k(A',g^*)-\bar F(A',g^*)\bigr)\right\|^2.
$$

The solver damps unstable updates, requires two consecutive undamped residuals within `1e−5` D in A and `1e−4`° in gaze, and keeps a converged P1-conditioned primary path. A=0 and A=4 starts are diagnostics/rescues only; their choice uses observed shape residuals, never truth labels. A fixed-A control holds the **observed P1-conditioned** `A_seed`, not true A, and iterates only the P4 center/gaze update. The forced-**known**-A=0 identity control reproduces the frozen independent P4 arrays exactly; the separate unknown-A run on true-A=0 cases can estimate nonzero A. The 87 fitted scalar components are 58 frozen B/G + 12 frozen PF12 + 15 new field + 2 new M1; references and coordinate normalizers are distinct constants.

The [full common-holdout results](data/p4_unified_combined/runtime/full_fast41/summary.json) score 83,229 primary states across all 21 A levels after keeping 21 repetitions of the explicitly excluded P1 gaze as a separate diagnostic. The 15-value P4 extension has full calibration Jacobian rank (raw/column-scaled condition 96.54/29.56), but its 2.099 µm coordinate RMS on the 75 known-A training poses is **training fit**, not heldout accuracy. M1/M2 grouped CV is calibration-only. On the primary holdout, initial A RMS 0.027534 D becomes final 0.025896 D, while P4 gaze radial RMS is 0.015035° versus 0.0147033° with the fixed-A control. All primary paths converged. A predeclared 512+21 pilot matched strict401 and fast41 profiles closely; this verifies the acceleration on those paths, not general numerical convergence or model correctness. The holdout is dense, correlated interpolation in one noiseless ray-traced system with nominal recorded channel frames. It supplies no instrument-noise, gaze-label, frame-registration, independent-optical-system, or extrapolation evidence. Rank and a large number of correlated ray coordinates do not by themselves rule out overfitting.

### Intended absolute-coordinate replacement (not implemented)

The current separate B maps, outer-ray anchor, and M1 are **not algebraically redundant inside center-relative H**: `H_center=0`, and H has no translation or absolute central-ray motion. They could become redundant as independently fitted blocks in a redesigned **complete absolute-coordinate** transform that predicts all nine indexed rays, including the labeled central ray, in a fixed calibrated sensor frame. For channel `j`, reference ray `r_k`, and `A_1=0`, `A_4=A`, the intended form is

$$
y_{j,k}=W_j(r_k;g,A_j,\theta_j),\qquad
c_j=W_j(r_c;g,A_j,\theta_j),\qquad
\mu_j=\tfrac19\sum_{k=1}^9 W_j(r_k;g,A_j,\theta_j),\qquad
b_j=\mu_j-c_j.
$$

P1 and P4 would use the **same W family with independent coefficients** at A=0. P4 A-dependent increments would vanish at A=0, and magnification/radial/projective geometry would also account for central-ray motion; no separate fitted B, discrete mean anchor, or T would be used. The separately calibrated ten-vector `G1/G4` maps remain the default readouts/initializers, now applied to centers derived from W; a complete joint geometric inverse estimating gaze directly is an optional additional design choice, not part of the chosen refactor. A geometry-motivated W remains an empirical surrogate unless its parameters are actually derived from source, reflecting surfaces, accommodation mechanics, and sensor projection.

The **labeled central ray**, **arithmetic centroid**, and **optical principal/distortion center** are different objects. A radial transform about fixed distortion pivot `O` leaves `O` fixed but can move an off-axis labeled central ray; recentering every observed grid on its own central ray suppresses that effect. The pivot must be fixed by physical calibration or explicit convention and then validated. Sensor origin, scale, orientation, ray labels (especially `r_c`), and a common recorded-coordinate convention must be fixed; freely interchangeable center and translation parameters would destroy identifiability. Different P1/P4 origins or mirror geometry do not justify equal coefficients or an unvalidated fitted registration.

For the relative decomposition `y_k=c+F_k(g,A)` with `F_c=0`, the least-squares center at a **fixed** state is algebraically `\hat c=mean(y)-mean(F)`. Thus a separate mean-to-center bias polynomial is unnecessary **if** one adequate full-grid F and a meaningful state estimate are available. This conditional identity does not establish physical central-ray accuracy: an approximate warp can invert itself consistently while predicting the wrong labeled center. The proposed ablation must therefore fit absolute observed nine-ray positions jointly, compare both whole-field residuals **and** labeled-central-ray error on the existing 3,943 A=0 and 83,229 full-primary masks, preserve A=0 identity, check calibration-only rank/conditioning/grouped CV, and then test sensor noise, gaze-label uncertainty, and frame/registration perturbations. The P1 25-pose stencil supports a 10-monomial design; a nine-pose stencil has rank at most nine and cannot identify that unrestricted polynomial. These ablations are **not completed** and no lower parameter count, improved accuracy, or immunity to overfitting is promised.

## Historical transformation analyses

The following analyses retain their original dates, formulas, and data provenance. Their “current” labels refer to their original publication within this project, not to the 2026-11-08 hybrid or the proposed replacement above.

## Purpose and modeling principle

Describe the center-relative image pattern using the fewest dependencies needed at the measurement scale. Rotation is `theta`; P4 also varies with accommodation `A` and the separately studied axial coordinate `Z`.

**A parameter may be held constant when removing its dependence introduces an acceptable image-coordinate error over the intended operating domain.** Approximately **1 micrometer (0.001 mm)** is the project measurement-noise reference. Small coefficient variation alone is not sufficient: its effect must be evaluated after the complete spatial transformation.

The working model treats P1 as accommodation-independent and P4 as accommodation- and Z-dependent where measured. P1 may still require rotation-dependent deformation. The available P1 dataset has no accommodation dimension, so accommodation independence is a modeling premise, not a result of a P1 accommodation sweep.

The existing rotation calibration domain is one-axis RC rotation from -20 to +20 degrees. P1 also has a zero-rotation sweep across absolute THI Z from -5 to +5 mm. The P4 accommodation/rotation grid covers accommodation from 0 to 5 D and has no Z sweep at that resolution. A separate P4 z1 study measures absolute THI Z from -5 to +5 mm at five accommodations (0, 1, 2, 3, 4 D) and five rotations (-10, -5, 0, +5, +10 degrees). Each state has nine identified fields in a 3 x 3 grid. The P1 and P4 coefficients are specific to their respective optical setups; their values and signs are not interchangeable.

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

be the setup-specific real image coordinates in millimeters, relative to the center ray at the same state.

Define the empirical zero-rotation baselines at the selected P1 Z plane and original P4 accommodation as

$$
\mathbf b_{1,i}(Z)=\mathbf p_{1,i}(0,Z),
\qquad
\mathbf b_{4,i}(A)=\mathbf p_{4,i}(0,A).
$$

For the separate P4 Z study, the matched reference is instead `b_4Z,i(theta,A) = p_4Z,i(theta,A,0)`, using the same accommodation and rotation as the requested nonzero-Z plane.

**These baselines already contain the sampled barrel distortion.** Applying another radial correction to an empirical real-coordinate baseline would double-count that correction. A fitted radial model is an alternative way to represent the baseline from paraxial coordinates, not an extra correction to the same real baseline. (This is why the P4 joint corner model below never applies two radial maps in series: when one barrel map is substituted for another, the model's own keystone-fitted barrel term is zeroed first — see "Power-law barrel variant.")

The rotation source must be the zero-rotation baseline with matching field identities, not the paraxial grid at the rotation being predicted. All quantities here describe relative pattern deformation; absolute center displacement would require a separate translation model (see Summary.md's "Candidate absolute-coordinate composition").

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

For P1, use the measured Z=0 baseline and model the zero-rotation Z dependence with the normalized scale `m(Z)=1+alpha_Z Z`. The P4 joint model fitted at that stage (below) superseded the separate accommodation-dependent-baseline-plus-rotation-coefficients sketch for the P4 corner points; this decomposition remained the working P1 model and the general template for P4 field points away from the four corners.

### General rotation-mapping family and reduced coefficient functions

For a baseline point `b = [x_b, y_b]^T`, the vertical-keystone family used throughout is

$$
K_j(\theta,A;\mathbf b)=
\begin{bmatrix}
\dfrac{s_{x,j}(\theta,A)x_b}{1+q_j(\theta,A)y_b}\\[2mm]
\dfrac{s_{y,j}(\theta,A)y_b}{1+q_j(\theta,A)y_b}
\end{bmatrix},
\qquad
H_j=\begin{bmatrix}s_{x,j}&0&0\\0&s_{y,j}&0\\0&q_j&1\end{bmatrix}.
$$

`s_x` and `s_y` are dimensionless; `q` has units mm^-1 (or is dimensionless `q_n` in corner-normalized coordinates, with physical `q=q_n/L`). Center-relative coordinates remove translation; the nominal reflection symmetry motivates omitting shear and horizontal perspective. This is a restricted spatial family, not a general model of every optical distortion.

A **minimal, proposed** (not yet fit to the replacement data) reduction of the coefficient functions is

$$
s_{x,j}(\theta)=1+\alpha_j\theta^2,
\quad s_{y,j}(\theta)=1+\beta_j\theta^2,
\quad q_j(\theta)=\gamma_j\theta,
$$

enforcing `H_j(0)=I`. The P4 joint corner model below uses a richer version of this family (order-4 in theta, order-2 in A, with its own fitted barrel term) and was the authority for P4's four corner points at that stage; this minimal form remained a candidate simplification for P1 and for P4 field points not covered by the corner fit. With `theta` in degrees, `alpha` and `beta` have units degree^-2 and `gamma` has units mm^-1 degree^-1.

Require positive scales and a denominator bounded away from zero over the accepted domain: `1+q_j(theta,A) b_{j,i,y}(A) >= delta > 0`. Check this condition on the field region being modeled, not only at the center.

## P4 joint corner model (historical, fitted)

This was the authoritative P4 model for the four corner field points at the time of that analysis, fit directly against ray-traced data in `Script/analyze_p4_joint_corner_transform.py` (schema `p4-joint-corner-inverse-v2`). Its reference pattern `R_0` is the computed real-ray (nonparaxial) pattern at zero accommodation and zero eye rotation. It was jointly fit to simulated corners over `A in [0,5] D` and `theta in [-20,20] deg`, 20,451 states, all 38 coefficients fit together. No experimental measurements were used for validation — these are in-sample simulation residuals.

For a reference corner `u_0 = (u_0x, u_0y) in {(-1,-1),(-1,1),(1,-1),(1,1)}`, the complete four-corner forward equation is

$$
\widehat{\mathbf R}_{\theta,A}=L M_B(A)
\begin{bmatrix}
\dfrac{s_x(\theta,A)u_{0x}}{1+q_n(\theta,A)M_B(A)u_{0y}}\\[6pt]
\dfrac{s_y(\theta,A)u_{0y}}{1+q_n(\theta,A)M_B(A)u_{0y}}
\end{bmatrix},
\qquad L=1.235429592626371\ \mathrm{mm}.
$$

The accommodation-driven barrel multiplier `M_B` acts on the reference corner before the keystone transform, so it appears in the keystone denominator as well as the overall radial scale. There is no fitted center translation. At `A=0`, `M_B=1`; at `theta=0`, the keystone map is the identity.

**Normalization and barrel coefficient.** Normalized inputs are `a=A/(5 D)` and `t=theta/(20 deg)`, so `0<=a<=1`, `-1<=t<=1`. The reference dimensionless corners are `u_0 = R_0/L`, with `||u_0||^2=2`. The relative barrel coefficient is

$$
\beta(A)=-0.0428500980586a-9.70843337206\times10^{-5}a^2,
$$
$$
M_B(A)=1+2\beta(A)=1-0.0857001961172a-0.0001941686674412a^2.
$$

`beta` acts on normalized coordinates; its millimeter-coordinate equivalent is `beta/L^2` in mm^-2. At `A=5 D`, `M_B ≈ 0.9141056352`. The corner data constrain only the net relative radial effect at `||u_0||^2=2`; they do not identify an absolute physical barrel coefficient separately from isotropic scale.

**General map and keystone.** For a general dimensionless reference-plane coordinate `u`,

$$
\mathbf r=K_{s_x,s_y,q_n}\bigl(B_\beta(\mathbf u)\bigr),\qquad
B_\beta(\mathbf u)=(1+\beta\|\mathbf u\|^2)\mathbf u,
\qquad
K_{s_x,s_y,q_n}(x,y)=\left(\frac{s_xx}{1+q_ny},\frac{s_yy}{1+q_ny}\right).
$$

The homography is normalized to `H_33=1`; its two scale factors are independent and `s_x s_y` is not constrained to one. `q_n` is dimensionless; its millimeter equivalent is `q=q_n/L`.

**Fitted coefficient functions** (dominant parity-structured approximation; the full fit retains all 38 coefficients, this omits only small cross/roundoff terms and is not a separate fit):

$$
s_x\approx1+f_{x2}(a)t^2+f_{x4}(a)t^4,\quad
s_y\approx1+f_{y2}(a)t^2+f_{y4}(a)t^4,\quad
q_n\approx f_{q1}(a)t+f_{q3}(a)t^3,
$$

$$
\begin{aligned}
f_{x2}(a)&=0.0594091260+0.0342887416a+0.000354140600a^2,\\
f_{x4}(a)&=0.00191095740+0.000448252512a+0.00287105952a^2,\\
f_{y2}(a)&=0.122914116+0.111542515a+0.00729162267a^2,\\
f_{y4}(a)&=-0.00353495801+0.00351620358a+0.00833102320a^2,\\
f_{q1}(a)&=-0.0630503261-0.0603090233a-0.0137920738a^2,\\
f_{q3}(a)&=0.00811845052+0.00575111136a+0.00296676077a^2.
\end{aligned}
$$

`B_beta` carries the fitted zero-gaze accommodation-dependent radial scale at the reference-corner radius; `s_x, s_y` describe fitted gaze-dependent stretch with accommodation-dependent amplitudes, and their geometric mean can include isotropic gain, so `beta(A)` does not represent every isotropic-scale contribution.

**Inverse** (keystone first, then radial — the two do not commute, so inversion reverses the forward order):

$$
y=\frac{Y_n}{s_y-q_nY_n},\qquad x=\frac{X_ns_y}{s_x(s_y-q_nY_n)},
$$

then solve the monotone radial equation `r_v = r_u(1+beta(A) r_u^2)` for `r_u`, `u = (r_u/r_v)(x,y)` (zero-radius case: `u=(0,0)`), on the branch where `1+beta r_u^2>0` and `1+3 beta r_u^2>0`.

**Fit objective and accuracy.** All coefficients are fit together minimizing the pooled four-corner vertex RMS

$$
E=\sqrt{\frac{1}{4N}\sum_{s=1}^{N}\sum_{i=1}^{4}\left\|L\hat{\mathbf u}_{si}-\mathbf R_{0i}\right\|^2}
$$

over all 20,451 states. **Mean per-state four-corner vertex RMS is 6.815869 µm; worst per-state is 21.348234 µm.** These are in-sample residuals against the optical simulation on the four fitted corners only; they do not establish experimental hardware precision or accuracy away from the sampled states or at the five non-corner field points.

## Power-law barrel variant (used in `plot_p4_transformation_effects.py`)

A second barrel parameterization — fit independently per accommodation rather than jointly with the keystone — is used when composing and validating the full transform against the raw ray-traced grid in `Script/plot_p4_transformation_effects.py`. For physical reference-plane coordinates `u` in millimeters,

$$
D_k(\mathbf u)=(1+k\|\mathbf u\|^2)\mathbf u,
\qquad
k_1(A)=c_0+c_1A^n,
$$
$$
c_0=-0.0184792556675\ \mathrm{mm^{-2}},\quad
c_1=0.000180236991874\ \mathrm{mm^{-2}D^{-n}},\quad
n=1.53545703384,
$$

where `n` is itself a fitted parameter (bounded `1e-6 <= n <= 10` in `Script/fit_p4_radial_distortion.py`), not a chosen exponent. Let `r_0` be the fixed measured real-ray reference grid at `A=0, theta=0`, and let `s(A)` be the saved paraxial scale ratio (`s(0)=1`, `s(5 D)=0.896199673581`). The accommodation barrel map is

$$
B_A(\mathbf r_0)=D_{k_1(A)}\!\left[s(A)\,D_{k_1(0)}^{-1}(\mathbf r_0)\right],
\qquad
B_A^{-1}(\mathbf r)=D_{k_1(0)}\!\left[\frac{1}{s(A)}D_{k_1(A)}^{-1}(\mathbf r)\right].
$$

`B_0(r_0)=r_0` by construction despite `k_1(0)=c_0`. Either radial inverse must solve `rho_out = rho_in(1+k rho_in^2)` on the near-axis monotonic branch, requiring `1+3k rho_in^2 > 0`; the plotting script raises if this is violated.

**How it composes with the joint corner model.** `plot_p4_transformation_effects.py` builds its "Joint" panel as `model_transform(B_A(r_0), L, beta=0, s_x(A,theta), s_y(A,theta), q_n(A,theta))` — i.e. it reuses only `s_x, s_y, q_n` from the joint corner fit above, and zeroes that fit's own `beta(A,theta)` barrel term, substituting `B_A` in its place. This is deliberate, not a double-count (the fit's `beta` is zeroed exactly where `B_A` is substituted in), and it is an improvement over using the joint fit's own four-corner-only `beta`: at `A=5 D, theta=0`, the self-consistent `beta`-barrel alone gives 41.28 µm all-nine-field RMSE (it was fit only against corners and fails the other five points), while the power-law `B_A` gives 0.754606 µm all-nine-field RMSE, 1.127786 µm max (meeting the ~1 µm reference). The full composed joint prediction at `A=5 D, theta=20 deg`, however, gives 19.033962 µm all-nine-field RMSE and 26.438233 µm max point error — roughly 20x the reference, at this one checked state. `s_x, s_y, q_n` were fit against the four corners only; the other five field points are out-of-sample checks. This single-state figure is not directly comparable to the pooled 6.82/21.35 µm four-corner numbers above (different field subset, different pooling).

## General projective keystone (two-axis form; fitted specialization below)

For a barrel output `r_b=(x_b,y_b)`, a general center-preserving two-axis projective keystone is

$$
K(\mathbf r_b)
=\frac{M(A,E_x,E_y)\,\mathbf r_b}
       {1+q_x(A,E_x,E_y)x_b+q_y(A,E_x,E_y)y_b},
\qquad
M=\begin{bmatrix}m_{11}&m_{12}\\m_{21}&m_{22}\end{bmatrix},
\qquad
H=\begin{bmatrix}m_{11}&m_{12}&0\\m_{21}&m_{22}&0\\q_x&q_y&1\end{bmatrix}.
$$

The zero translation terms keep the center fixed. The denominator is linear in `x_b, y_b` — not a radial function — and rotation about one physical axis often produces a denominator term on the *other* image coordinate, but the sign/axis assignment depends on optical and coordinate geometry. This two-axis family has now been fitted for P4 at `A=0` over the sampled ADE/BDE gaze grid below. Its coefficient functions over accommodation, other lens setups, and gaze outside the sampled range remain unvalidated. The P4 joint corner model and power-law variant above still use a single rotation variable `theta` and a diagonal `M`; assigning that simulation's `theta` to physical `E_x` or `E_y` requires optical-coordinate evidence beyond those fits.

## P4 A=0 biaxial gaze-induced distortion (fitted)

This fit describes the center-relative nine-field pattern at zero accommodation over the sampled two-axis gaze domain. It uses the measured A=0 pattern at `ADE=BDE=0` as its fixed reference. Let `z_{g,j}` be the measured two-coordinate position in millimeters for gaze state `g` and field `j`, and let `c_g=z_{g,center}` be that state's measured center point. Write `00` for zero gaze, `c_{00}` for its measured center, and `L=1.235429592627 mm`. The fixed normalized reference field is

$$
\mathbf b_j=\frac{\mathbf z_{00,j}-\mathbf c_{00}}{L},\qquad
\mathbf y_{g,j}=\frac{\mathbf z_{g,j}-\mathbf c_g}{L}.
$$

For a fitted dimensionless map `f_g`, the reconstructed positions are

$$
\widehat{\mathbf z}_{g,j}=\mathbf c_g+L f_g(\mathbf b_j).
$$

Thus the model fits the centered distortion and supplies each measured `c_g`; it does not estimate gaze-dependent center motion. The measured center displacement over the grid is negligible (about `3.4×10^-9 mm`). The center field is excluded from the fit objective, and the primary errors below use the eight noncentral fields.

**Spatial map and fitted variants.** For `h=H_g(b)`,

$$
\mathbf h=\frac{M_g\mathbf b}{1+q_{x,g}b_x+q_{y,g}b_y},\qquad
M_g=\begin{bmatrix}m_{11,g}&m_{12,g}\\m_{21,g}&m_{22,g}\end{bmatrix}.
$$

The 6-parameter projective model uses `f_g(b)=h`. A uniform magnification is already expressible in `M_g` (take `M_g=m_g I` when isotropic); it is distinct from the optional fitted radius-dependent term below.

The 8-parameter projective-plus-Brown model applies the two Brown-Conrady tangential terms after `H`. With `h=(h_x,h_y)`,

$$
\mathbf w=\operatorname{BC}_{p_1,p_2}(\mathbf h)=
\begin{bmatrix}
h_x+2p_{1,g}h_xh_y+p_{2,g}(3h_x^2+h_y^2)\\
h_y+p_{1,g}(h_x^2+3h_y^2)+2p_{2,g}h_xh_y
\end{bmatrix},\qquad f_g(\mathbf b)=\mathbf w.
$$

The 9-parameter model applies a residual radial gain after this Brown map:

$$
f_g(\mathbf b)=\bigl(1+k_{\rm rad,g}\|\mathbf w\|^2\bigr)\mathbf w.
$$

Here `k_rad` is dimensionless because `w` is normalized by `L`. This radius-dependent gain is not a uniform magnification: uniform `S_m v=m v` can be included in `M_g`, while `R_k(v)=(1+k\|v\|^2)v` changes scale with field radius. Their order matters; for example, `R_k(S_m v)=S_m R_{km^2}(v)`, whereas `S_mR_k(v)` generally differs. The fitted residual term is empirical and may absorb interaction between scale and barrel distortion; the fit does not establish a separate physical cause for `k_rad`.

**Gaze dependence and split.** Here `tx=BDE/20°` and `ty=ADE/20°`; the coefficient functions for every free parameter use all 15 monomials `tx^i ty^j` with `i+j≤4`, with no parity constraints. The source grid and saved state arrays are BDE-major and ADE-minor. The polynomial coefficients were fit from per-state parameters at the 51 even-index BDE samples crossed with the 51 even-index ADE samples (2,601 training states); the other 7,600 gaze states form the held-out set. This tests gaze interpolation within `ADE,BDE∈[-20°,20°]` at the same nine field locations. It does not test new field positions or extrapolation beyond ±20°.

**Held-out performance.** Errors are ray-traced simulation residuals in micrometers; the primary metrics pool the eight noncentral fields. Vertex RMS is `sqrt(mean(e_x²+e_y²))` and the maximum is the largest Euclidean point error.

| Model | Parameters per gaze state | X RMSE | Y RMSE | Vertex RMS | Max point |
|---|---:|---:|---:|---:|---:|
| Projective `H` | 6 | 4.2778 µm | 4.2665 µm | 6.0418 µm | 27.8451 µm |
| Projective + Brown `p1,p2` | 8 | 2.4892 µm | 2.4928 µm | 3.5227 µm | 16.6221 µm |
| Projective + Brown + radial `k_rad` | 9 | 1.5048 µm | 1.5429 µm | 2.1552 µm | 11.3488 µm |

The radial variant improves held-out vertex RMS by 64.3% relative to projective alone. Its pooled held-out vertex RMS is still above 1 µm; 2,868 of 7,600 held-out gaze states are below 1 µm vertex RMS and 1,952 are below 1 µm maximum point error. The per-state fits are empirical local-fit benchmarks, not proven global minima. They were checked for nesting, denominator validity, small normalized gradients on stalled states, and independent refits of the most weakly stationary examples. This two-axis A=0 fit supersedes the earlier BDE=0-only K/decentering analysis as the current characterization of A=0 gaze, but the restricted one-axis result remains a different model and its metrics are not directly comparable.

See the [fitting script](Script/fit_p4_ade_a0_decenter.py), [full report](data/p4_ade_a0_decenter/biaxial_report.md), [coefficients and residuals](data/p4_ade_a0_decenter/p4_ade_a0_biaxial.npz), [held-out error maps](data/p4_ade_a0_decenter/p4_ade_a0_biaxial_heatmaps.png), and [representative residual vectors](data/p4_ade_a0_decenter/p4_ade_a0_biaxial_residuals.png).

## Display-coordinate swap (plotting convention only)

Figures display keystone-bearing panels after swapping physical coordinates: plotted `x` is physical `y`, plotted `y` is physical `x`. Under this swap, numerator scale labels exchange places and the denominator term appears on plotted `x`:

$$
\begin{bmatrix}x_{\rm plot}'\\y_{\rm plot}'\end{bmatrix}
=\frac{\begin{bmatrix}s_y&0\\0&s_x\end{bmatrix}
        \begin{bmatrix}x_{\rm plot}\\y_{\rm plot}\end{bmatrix}}
       {1+q\,x_{\rm plot}}.
$$

This is a plotting-coordinate swap only; it does not alter the saved physical-coordinate model or ray data.

## P1 fixed-baseline barrel plus vertical keystone

P1 uses its actual real-coordinate grid at zero rotation as its empirical baseline. At the selected Z plane, the prior rotation calibration contained 401 rotations and nine fields per rotation. The source analysis independently estimates three keystone parameters at each angle and fits unconstrained quadratics to the resulting parameter curves. Its generated result folder has been removed after the replacement sweep validated; see `Script/analyze_p1_rotation_transform.py` to regenerate current results.

**Dataset status and provenance:** The validated canonical P1 grid has 51 absolute `Cornea_ENT_D` THI planes from -5 to +5 mm. The prior P1 rotation fit came from an earlier single-slice analysis associated with a prior sequence revision and a recorded THI baseline of -35 mm; its source hash does not establish identity with the current sequence. A numerical audit found that its 3,609-row slice (401 rotations by nine fields) matches the previous multi-Z sweep Z=0 slice to floating-point roundoff, but does not validate that fit against the replacement grid or establish rotation behavior at other Z planes. The prior fit is stale pending refresh. The current sequence uses a 1 mm baseline; absolute THI Z is reported as collected, not relabeled as displacement from that baseline.

### Z-dependent zero-rotation barrel baseline and magnification

Let `R_i(Z)` be the measured real-coordinate vector for field `i` and `R_i(0)` the actual measured real grid at Z=0. Fit one isotropic scale at each Z by pooled least squares over the nine matched fields:

$$
m(Z)=\frac{\sum_i \mathbf R_i(0)\cdot\mathbf R_i(Z)}
           {\sum_i \|\mathbf R_i(0)\|^2},\qquad
\mathbf R_i(Z)\approx m(Z)\mathbf R_i(0).
$$

The center is harmless in the pooled fit (its center-relative reference vector is zero). For pointwise diagnostics, use the signed projected ratio `R_i(0)*R_i(Z)/||R_i(0)||^2` on the eight noncentral points. The parallel paraxial scale `m_p(Z)` is calculated the same way. Both ratios are dimensionless; Z is in millimeters.

The working P1 simplification is `m(Z)=1+alpha_Z Z`, `alpha_Z=0.00295165270232 mm^-1`, the least-squares slope of the 51 saved real-grid magnification ratios with the intercept fixed at one (`m(0)=1`). Ratio RMSE is 1.01275e-4, max absolute ratio error 2.19022e-4 over the 51 planes.

| Absolute THI Z (mm) | -5 | -2.6 | 0 | +2.6 | +5 |
|---|---:|---:|---:|---:|---:|
| Real-grid `m(Z)` | 0.9854583393 | 0.9923851807 | 1 | 1.007732594 | 1.014977286 |
| Paraxial `m_p(Z)` | 0.9854551017 | 0.9923834777 | 1 | 1.007734340 | 1.014980683 |

For comparison, the unconstrained linear fit is `m(Z)=1.00007550219+0.00295165270232 Z` (ratio RMSE 6.74981e-5, max 1.43520e-4); a near-identity-constrained quadratic, `m(Z)=0.999999995615+0.00295165270232 Z+8.71229730269e-6 Z^2`, reduces those to 5.14398e-7 and 1.21943e-6. The simpler `1+alpha_Z Z` relation is the working model; retain the quadratic when its extra term matters.

Under the working `1+alpha_Z Z` model at theta=0, the largest coordinate RMSE across nine points and both coordinates is 0.29549 µm, largest Euclidean point error 0.50326 µm. For the direct pooled per-plane scale fit, the corresponding largest errors are 0.00120667 µm and 0.00206133 µm.

The magnification factor was checked at five rotations (`theta=-10,-5,0,+5,+10` degrees) over all 51 Z planes. Relative to the `theta=0` real-grid scale curve, the largest absolute factor difference was 1.94155e-6 (max relative variation 0.00019129%). Applying the exact `theta=0` scale curve to those other-angle grids left at most 0.007220 µm coordinate RMSE and 0.017804 µm maximum point error — supporting theta-independence over these five tested angles only; the other 396 rotation values have not been checked.

The resulting proposed composition is `p_1(theta,Z) ≈ m(Z) K_1(theta; b_1(0))`: apply the existing keystone map to the Z=0 empirical real baseline, then multiply by `m(Z)`. In homogeneous coordinates, `S(m)=diag(m,m,1)` and the composition is `S(m) H_1(theta)`, in that order — scale does not commute with keystone. This is a simplified model supported by the zero-rotation sweep and five-angle check, not a full 401-angle joint calibration.

The separate per-Z radial fits use `real = paraxial * (1 + k1*r^2)`. The direct quadratic summary is `k1(Z) = -0.0130398764628 + 7.68681341348e-5 Z - 1.13154451225e-7 Z^2` (mm^-2, mm^-3, mm^-4). Since paraxial image scale also changes with Z, `k1` alone mixes scale and barrel strength; the sampled dimensionless `k1*r_edge^2` varies by only 0.00898% of its absolute mean. The per-grid radial-model coordinate RMSE is 2.24-2.31 µm — a distinct, larger fit error than the ~0.0012 µm inter-Z common-scale residual. These per-Z radial/magnification values are historical pre-replacement P1 results; their generated artifacts have been removed. Use `Script/fit_p1_radial_distortion.py` to regenerate from the validated P1 sweep.

### Existing P1 results and supported simplification

| P1 result | Direct per-angle transform | Unconstrained quadratic parameter approximation |
|---|---:|---:|
| Coordinate RMSE range, µm | 0-5.95020 | 0.0521408-5.96587 |
| Mean per-angle coordinate RMSE, µm | 3.06366 | 3.06866 |
| Maximum Euclidean point error, µm | 11.1749 | 11.1511 |

The direct fit at zero rotation is identity to numerical precision. All direct fits converged; minimum sampled direct denominator is 0.987474.

Dominant terms give reduced-model starting values `alpha_1=1.85435557047e-5`, `beta_1=6.46083406774e-5`, `gamma_1=3.78982990846e-4` (extracted from the unconstrained fit, not refitted). Replacing the fitted scale intercepts by one and removing near-zero symmetry-inconsistent terms changes the existing quadratic predictions by 0.052145 µm pooled coordinate RMS (0.086877 µm max). These diagnostics used the pre-replacement P1 transform artifact (now removed) and all 401 sampled angles; they measure simplification error on the prior dataset, not total error against the replacement sweep.

**A constant baseline is not a constant transformed pattern.** Holding the entire P1 pattern at its zero-rotation coordinates (`p_1,i(theta) = b_1,i`) gives, at -20 degrees, 30.424416 µm coordinate RMSE and 69.985236 µm max absolute coordinate error directly against the stored baseline — the entire P1 transform cannot be frozen over the full ±20-degree domain at 1 µm scale. A constant pattern or individual constant parameters remain candidates only for a narrower declared range, tested against the coordinate-error rule below.

**Remaining spatial-model error:** at -20 degrees the P1 side-midpoint fields `(field_x,field_y)=(±1,0)` have real Y ≈ 8.471582 µm while baseline Y is zero to numerical precision; the restricted mapping predicts `Y=0` when `y_b=0` regardless of parameter values. This is a concrete missing spatial deformation, not a polynomial-order problem.

## Historical pre-joint-model P4 analyses (superseded)

Before the joint corner model above was fit, P4 barrel and keystone were calibrated as two separate per-accommodation analyses. They are retained here for their accommodation-curve detail, which the corner-only joint model does not reproduce.

**Zero-rotation barrel baseline (per-accommodation, independent of keystone):** `b_i(A) = [1+k1(A) r_i^2] u_i(A)`, nine-field grid. `k1` changes from -0.0185283312073 mm^-2 at 0 D to -0.0163117998622 mm^-2 at 5 D. Coordinate RMSE across accommodation: mean 0.807812 µm, max 1.101915 µm; max point error 2.203830 µm. Three in-sample accommodation curves have been fit to the 51 `k1` values: quadratic `k1(A)=p0+p1A+p2A^2`, `(p0,p1,p2)=(-0.0185179003996, 0.000180197179361, 5.15785422657e-5)`, coordinate-RMSE addition up to 0.0366545 µm; offset-exponential `k1(A)=d+a*exp(bA)`, `(d,a,b)=(-0.0195347321301, 0.000993232198585, 0.235926475525)`, addition up to 0.0428462 µm; and the power-law form used in the "Power-law barrel variant" section above. The grid has only two distinct nonzero radii, so none of these validate arbitrary intermediate field positions or a unique physical accommodation law.

**Rotation keystone (per-accommodation, pre-joint-fit):** fit independently at each of 20,451 (A,theta) states using `X=sx*x0/(1+q*y0)`, `Y=sy*y0/(1+q*y0)` against each accommodation's own measured theta=0 real grid (no radial correction applied a second time), then each of `sx, sy, q` approximated by its own unconstrained quadratic in theta per accommodation (9 coefficients per accommodation; no accommodation law, no identity/symmetry constraint).

| A (D) | sx coeff. of theta^2 (deg^-2) | sy coeff. of theta^2 (deg^-2) | q coeff. of theta (mm^-1 deg^-1) |
|---:|---:|---:|---:|
| 0 | 0.0001552758954 | 0.0003028028193 | -0.002345622877 |
| 1 | 0.0001732684773 | 0.0003630345936 | -0.002831577821 |
| 2 | 0.0001918100541 | 0.0004266453010 | -0.003355193078 |
| 3 | 0.0002108883579 | 0.0004936398101 | -0.003918254684 |
| 4 | 0.0002305306594 | 0.0005642392799 | -0.004524126537 |
| 5 | 0.0002507548114 | 0.0006386357893 | -0.005175941610 |

| Fit summary over accommodation and rotation states | Mean | Maximum |
|---|---:|---:|
| Direct-transform coordinate RMSE (µm) | 4.06167 | 12.8112 |
| Quadratic-parameter total coordinate RMSE (µm) | 4.46151 | 14.5312 |
| Added error from parameter quadratics, coordinate RMSE (µm) | 1.51779 | 7.25432 |
| Direct-transform maximum point error (µm) | 7.60316 | 25.104 |
| Quadratic-parameter total maximum point error (µm) | 8.38888 | 25.7348 |
| Added error from parameter quadratics, maximum point error (µm) | 3.06876 | 14.5049 |

This per-accommodation keystone is evaluated against each accommodation's own measured 9-field baseline, so its numbers are not directly comparable to the joint corner model's pooled four-corner numbers (different field subset and baseline convention). The largest residual occurs at A=5 D, theta=-20°, field (1,0): predicted-minus-measured ≈(+7.884,+23.834) µm — a missing spatial component that raising the theta-polynomial order cannot fix. Reusing A=0 keystone parameters at other accommodations gives up to 294.802 µm additional and 296.617 µm total max point error — accommodation dependence is essential. The measured absolute image center's same-accommodation shift from theta=0 reaches 5.52214 mm in one component; no continuous center law is established (see Summary.md's "Candidate absolute-coordinate composition").

### Historical separate P4 Z-dependent magnification study

Uses `Lens/p4_ME.len`, zoom z1, surface selector `s"Cornea_ENT_D"`; samples A in {0,1,2,3,4} D, theta in {-10,-5,0,5,10} deg, 51 absolute THI values from -5 to +5 mm (0.2 mm spacing), nine fields (1,275 states, 11,475 records). Active z1 THI readback after loading this LEN is 0 mm; `THI=Z` with matching sign, both `z_distance_mm` and `z_thickness_mm` recorded.

For each A, theta, let `R_i(A,theta,Z)` be the center-relative real-coordinate vector. The direct scalar factor relative to the same A, theta at Z=0 is `m_direct(A,theta,Z) = sum_i[R_i(A,theta,0)·R_i(A,theta,Z)] / sum_i||R_i(A,theta,0)||^2`. The constrained linear slope fit at theta=0 only is `m_4(A,Z)=1+alpha_4Z(A)Z`:

| Accommodation (D) | `alpha_4Z(A)` (mm^-1) | Max relative change across tested rotations (%) |
|---:|---:|---:|
| 0 | 0.00295232758 | 0.00171777 |
| 1 | 0.00295172708 | 0.00200342 |
| 2 | 0.00295112891 | 0.00230384 |
| 3 | 0.00295052435 | 0.00261870 |
| 4 | 0.00294991207 | 0.00294912 |

Across these five accommodations the slopes span 0.0818504% of their mean; `m≈1+0.00295 Z` is a convenient common-slope approximation, `alpha_4Z(A)` more accurate when accommodation-specific accuracy matters. Applying the theta=0 constrained linear factor to each same-A/theta Z=0 real grid gives a worst-state coordinate RMSE of 0.249 µm, max point error 0.511 µm, across the tested states; pooled over all Z/rotation states the linear coordinate RMSE is 0.100-0.106 µm by accommodation, and direct per-state scalar factors reduce pooled coordinate RMSE to 0.0120-0.0160 µm.

This Z dataset's LEN SHA-256 is `fb3937e6763f27e331faa10a4863533e994030cbc0ae74364b186ba0d06d474d`, differing from the P4 joint-corner-model dataset's own provenance; this study does not establish a jointly calibrated model across `theta`, `A`, and `Z`, and does not justify composing the joint corner model with `m_4(A,Z)` for end-to-end predictions without further checking.

## Coordinate-error rule for accepting a simplification

Let `s` denote a state (`theta` for P1; `(theta,A)` for P4), and `N` the number of evaluated fields. Define

$$
\mathbf e_{\mathrm{total},i}(s)
=\widehat{\mathbf p}_{\mathrm{simple},i}(s)-\mathbf p_{\mathrm{CODE\,V},i}(s),
\qquad
\mathbf e_{\mathrm{drop},i}(s)
=\widehat{\mathbf p}_{\mathrm{simple},i}(s)-\widehat{\mathbf p}_{\mathrm{reference},i}(s).
$$

For either residual population, coordinate RMSE at a state is

$$
E(s)=\sqrt{\frac{1}{2N}\sum_{i=1}^{N}\left(e_{i,x}^2+e_{i,y}^2\right)}.
$$

Report the maximum absolute coordinate residual and maximum Euclidean point residual separately; RMS Euclidean point error equals `sqrt(2)` times coordinate RMSE for the same points.

**Simplification criterion:** choose a declared budget `epsilon_drop` at or below the ~1 µm reference and test `max_s E_drop(s) <= epsilon_drop` over the intended domain. A constant is acceptable only on the domain where the test passes.

**Total-accuracy criterion:** separately evaluate `max_s E_total(s)` against the application's total error budget. A claim of total 1-µm performance requires this test, not merely a passing simplification test. Neither the per-accommodation keystone fit nor the composed power-law-barrel joint prediction meets that total target at the checked worst states.

Do not automatically allocate 1 µm to every omitted term: `E_total(s) <= E_reference(s) + E_drop(s)`; quadrature addition is not justified without suitable assumptions. 1 µm of measurement noise is not a hard noise bound or angular-precision statement.

## Calibration and validation sequence

1. **Fix coordinates and baselines.** Match field identities and units. Use the empirical zero-rotation baseline for each available accommodation to diagnose rotation fits. Do not apply the radial correction twice.
2. **Check the spatial family before its coefficient curves.** Fit independent keystone parameters at every relevant state; residual maps reveal missing spatial deformation that higher-order coefficient curves cannot fix.
3. **Fit the smallest global dependence.** Optimize final Cartesian coordinate residuals; parameter-curve RMSE alone is a diagnostic only.
4. **Test reductions and additions against the same data partition.** Report both simplification and total errors. Add individual coupling or spatial terms only when justified.
5. **Validate the complete prediction.** Hold out rotation values and accommodation groups; predict a held-out group's baseline from trained functions rather than reading its real baseline. Inspect boundaries, per-state errors, field residuals, and denominator positivity, then check physical measurements separately.

The P4 joint corner model and the separate five-angle P4 Z sweep use different LEN-revision provenance, so their results do not yet support joint fitting across `theta`, `A`, and `Z`. Additional field sampling is needed only to substantiate spatial interpolation beyond the current nine (or four-corner) fields. Do not claim extrapolation beyond the calibrated domain.

## Historical status and source artifacts (through 2026-11-05)

**Fitted by 2026-11-05:** the P4 joint corner model (four corners, all `theta`/`A`), its power-law-barrel composed variant, and the narrower P4 `A=0` two-axis gaze fit (nine sampled fields, with the eight noncentral points in its objective; ADE/BDE in ±20°), described above. The A=0 fit does not validate accommodation dependence or field positions beyond the sampled 3×3 grid. **Already historical/stale then:** the P1 rotation/radial fits, the pre-joint-model separate P4 radial and per-accommodation rotation fits, and the P1 and P4 Z-dependence checks (five angles each, not all 401). **Proposed at that time:** broader two-axis projective coefficient functions over accommodation and minimal reduced coefficient functions as a P1/off-corner candidate. The [2026-11-08 status above](#current-hybrid-model-and-geometric-interpretation-2026-11-08) supersedes this dated inventory.

Source artifacts:

- [Validated P1 sweep pickle](data/distortion_grid/distortion_grid.pkl) and retained analysis scripts: [P1 rotation](Script/analyze_p1_rotation_transform.py), [P1 radial/magnification](Script/fit_p1_radial_distortion.py).
- Historical validated P4 sweep (source archived; distinct from the [current P4 full-detail pickle](data/p4_raytrace/p4_raytrace_251x251_12p5_A4.pkl)) and archived analysis scripts: [P4 radial fit](Script/fit_p4_radial_distortion.py) (also fits the quadratic_A, offset_exponential, and power_law accommodation curves for `k1(A)`), [P4 rotation transform](Script/analyze_p4_rotation_transform.py) (pre-joint, per-accommodation), [P4 joint corner transform](Script/analyze_p4_joint_corner_transform.py) (historical joint model), [composition/validation plot](Script/plot_p4_transformation_effects.py).
- [Separate P4 Z magnification report](data/p4_z_magnification/p4_z_magnification.md), [plot](data/p4_z_magnification/p4_z_magnification.png), [structured grid](data/p4_z_magnification/p4_z_magnification.pkl), [metadata](data/p4_z_magnification/metadata.json), [fit artifact](data/p4_z_magnification/p4_z_magnification_fit.pkl), and [script](Script/analyze_p4_z_magnification.py).
- [Summary.md](Summary.md) for data provenance, hashes, and the candidate absolute-coordinate composition.

## Camera-side lens model (p1_camera_fit)

The camera-side fit in [Script/p1_camera_fit.py](Script/p1_camera_fit.py) models traced image coordinates $(x,y)$ from the direction cosines at surface 8. Define physical tangent inputs $t_{x}=\frac{l_{8}}{n_{8}}$ and $t_{y}=\frac{m_{8}}{n_{8}}$. On the training set, let $T=\max(|t_{x}|,|t_{y}|)$ over all training rays, then normalize as $u=\frac{t_{x}}{T}$ and $v=\frac{t_{y}}{T}$. The fit uses the full-grid interleaved training/holdout split recorded in [runs/p1_camera_fit_v2/summary.json](runs/p1_camera_fit_v2/summary.json).

For parameters $f,c_{x},c_{y},k_{1},k_{2},\theta_{s},\varphi_{s},q_{1},q_{2}$, the exact forward map is

$$
t_{x}=\frac{l_{8}}{n_{8}},\qquad t_{y}=\frac{m_{8}}{n_{8}}
$$

$$
T=\max_{\text{training rays}}\max(|t_{x}|,|t_{y}|),\qquad u=\frac{t_{x}}{T},\quad v=\frac{t_{y}}{T}
$$

$$
r^{2}=u^{2}+v^{2}
$$

$$
\rho=1+k_{1}\cdot r^{2}+k_{2}\cdot r^{4},\qquad u_{r}=\rho\cdot u,\quad v_{r}=\rho\cdot v
$$

$$
s=q_{1}\cdot(u_{r}^{2}+v_{r}^{2}),\qquad u_{p}=u_{r}+s,\quad v_{p}=v_{r}+q_{2}\cdot(u_{r}^{2}+v_{r}^{2})
$$

$$
D=1+\theta_{s}\cdot v_{p}-\varphi_{s}\cdot u_{p}
$$

$$
x=\frac{f\cdot u_{p}}{D}+c_{x},\qquad y=\frac{f\cdot v_{p}}{D}+c_{y}
$$

The nested levels activate parameters as follows; all terms not listed for a level are zero:

| Level | Active parameters |
|---|---|
| E0 | $f,c_{x},c_{y}$ |
| E1 | E0 plus $k_{1}$ |
| E2 | E1 plus $k_{2}$ |
| E3 | E2 plus $\theta_{s},\varphi_{s}$ |
| E4 | E3 plus $q_{1},q_{2}$ |

The physical-tangent coefficient conversions from normalized fit coefficients are $f_{\text{physical}}=\frac{f}{T}$, $k_{1,\text{physical}}=\frac{k_{1}}{T^{2}}$, and $k_{2,\text{physical}}=\frac{k_{2}}{T^{4}}$; each of $\theta_{s,\text{physical}}=\frac{\theta_{s}}{T}$, $\varphi_{s,\text{physical}}=\frac{\varphi_{s}}{T}$, $q_{1,\text{physical}}=\frac{q_{1}}{T}$, and $q_{2,\text{physical}}=\frac{q_{2}}{T}$ converts by division by $T$. The image center offsets $c_{x}$ and $c_{y}$ remain in millimeters.

Model selection uses the smallest level whose holdout maximum Euclidean point error is at most $1\,\text{µm}$. The E1 holdout maximum is **$0.0015829611284427967\,\text{µm}$** in the saved summary. A user override selects E1 downstream (2026-11-09). E3 and E4's $\theta_{s}$, $\varphi_{s}$, $q_{1}$, and $q_{2}$ coefficients are exactly zero for this dataset: its chief rays are symmetric and the holdout is interleaved. Consequently, the data do not validate the tilt or prism terms' signs or functional forms.

For implementation details and the dataset summary, see [Script/p1_camera_fit.py](Script/p1_camera_fit.py) and [runs/p1_camera_fit_v2/summary.json](runs/p1_camera_fit_v2/summary.json).
