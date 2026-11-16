# P4 model: default FL gaze and accommodation

## 1. Accepted default and scope

**FL is the accepted default as of 2026-11-16.** The user accepts approximately 1 arcmin gaze error and selects linear accommodation-dependent laws. Algorithm F, with quadratic coefficient trends, is retained as a fallback or future option. This closes the present model-selection goal under [CURRENT_PLAN.md, Section 17](CURRENT_PLAN.md#17-accept-fl-as-default-consolidate-p4-documentation-close-the-goal).

FL estimates gaze and accommodation from eight measured ring points. The sensor mean supplies gaze conditional on accommodation; ring spacing supplies accommodation conditional on gaze. Alternating these two conditional estimates resolves their coupled dependence empirically. The accepted default is a calibrated estimator, not an identified physical eye model or a reconstruction of all ring coordinates.

The recorded comparison is noise-free development evidence. Its original strict maximum-gaze test failed for FL: 1.0176 arcmin is above the required $<1.0$ arcmin. The unchanged experiment therefore reports `CLOSE: F`. The later user decision accepts the approximately 1 arcmin FL result and selects FL as default; it does not change that gate or verdict. Numerical procedures, constants, counts and detailed results are consolidated in [P4_IMPLEMENTATION.md](P4_IMPLEMENTATION.md).

## 2. Measurements, coordinates and notation

Select the four edge midpoints and four corners of the original $3\times3$ pattern by label from the $5\times5$ traces. The dimensionless labels and signed reference directions are

$$
\mathcal L=\{(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)\},
\qquad \mathbf d_i=-\boldsymbol\ell_i.
$$

The sign reversal fixes the traced pattern's orientation. Let $r_i^2=\lVert\mathbf d_i\rVert_2^2$: edges have $r_i^2=1$ and corners have $r_i^2=2$. Each sensor position $\mathbf X_i$, its arithmetic mean $\overline{\mathbf X}$, the fixed A00 zero-gaze mean origin $\mathbf o_0$ and the mean displacement $\mathbf m$ are in mm:

$$
\overline{\mathbf X}=\frac18\sum_{i=1}^8\mathbf X_i,
\qquad \mathbf o_0=\overline{\mathbf X}(A=0,\mathbf w=\mathbf0),
\qquad \mathbf m=\overline{\mathbf X}-\mathbf o_0.
$$

The origin stays fixed for every accommodation. The measured mean is not the traced centre E1; neither E1 nor a centre-minus-mean correction is a runtime input. Shared positive A00 axis scales $s_x,s_y$ in mm give dimensionless polynomial coordinates

$$
x=m_x/s_x,\qquad y=m_y/s_y.
$$

Accommodation $A$ and its estimate $\widehat A$ are in dioptres (D). Gaze targets and outputs are $\mathbf w=(\mathrm{BDE},-\mathrm{ADE})^T$ in degrees. Convert to radians before evaluating trigonometric functions. The dimensionless size coordinates are

$$
\boldsymbol\theta=\frac{\pi}{180}\mathbf w,
\qquad \tau_x=\frac{\tan\theta_x}{\cos\theta_y},
\qquad \tau_y=\tan\theta_y.
$$

Thus $\boldsymbol\tau=(\tan(\mathrm{BDE})/\cos(\mathrm{ADE}),-\tan(\mathrm{ADE}))^T$ when the named angles are in radians. Signs and the secant factor are part of the model convention.

| Symbol | Meaning | Units |
|---|---|---|
| $\mathbf X_i,\overline{\mathbf X},\mathbf o_0,\mathbf m$ | Sensor points, mean, fixed mean origin, displacement | mm |
| $s_x,s_y$; $x,y$ | Shared axis scales; normalized mean coordinates | mm; dimensionless |
| $A,\widehat A$; $\mathbf w$ | Accommodation; two-axis gaze | D; deg |
| $\mathbf c_{\rm cam},\mathbf z_i,\mathbf S_i$ | Camera centre, pre-camera point, mean-free point | mm |
| $H_A,q_A$ | Zero-gaze radial scale and relative curvature | mm, dimensionless |
| $\mu,\beta,\boldsymbol\tau,\boldsymbol\gamma(A)$ | Size, relative corner-shape change, tangents, size-response coefficients | dimensionless |

## 3. Default FL gaze polynomial

Spatial cubic order and accommodation-law degree are separate choices. At every fixed $A$, FL evaluates the same order-3, two-output polynomial with ten monomials per output, or twenty coefficient positions in total:

$$
\boldsymbol\phi(x,y)=(1,x,y,x^2,xy,y^2,x^3,x^2y,xy^2,y^3)^T,
\qquad \widehat{\mathbf w}=P_A(\mathbf m)=C(A)\boldsymbol\phi(x,y).
$$

The full two-axis expression is

$$
\begin{aligned}
\widehat w_x={}&c^x_{00}+c^x_{10}x+c^x_{01}y+c^x_{20}x^2+c^x_{11}xy+c^x_{02}y^2
+c^x_{30}x^3+c^x_{21}x^2y+c^x_{12}xy^2+c^x_{03}y^3,\\
\widehat w_y={}&c^y_{00}+c^y_{10}x+c^y_{01}y+c^y_{20}x^2+c^y_{11}xy+c^y_{02}y^2
+c^y_{30}x^3+c^y_{21}x^2y+c^y_{12}xy^2+c^y_{03}y^3.
\end{aligned}
$$

Every $c^\alpha_{ij}=c^\alpha_{ij}(A)$ multiplies $x^iy^j$ in output axis $\alpha$. Because $x,y$ are dimensionless, the evaluated coefficients are in degrees. Each term has the following role; the role does not assert a uniquely identified optical mechanism.

| Monomial | Contribution to each output |
|---|---|
| $1$ | Constant offset |
| $x$ | Linear sensitivity to horizontal mean displacement |
| $y$ | Linear sensitivity to vertical mean displacement |
| $x^2$ | Horizontal-coordinate quadratic curvature |
| $xy$ | Mixed quadratic coupling |
| $y^2$ | Vertical-coordinate quadratic curvature |
| $x^3$ | Horizontal-coordinate cubic curvature |
| $x^2y$ | Vertical displacement modulated by horizontal displacement squared |
| $xy^2$ | Horizontal displacement modulated by vertical displacement squared |
| $y^3$ | Vertical-coordinate cubic curvature |

FL enforces independent-axis reflection symmetry. Only six positions are allowed:

$$
\widehat w_x=c^x_{10}(A)x+c^x_{30}(A)x^3+c^x_{12}(A)xy^2,
\qquad
\widehat w_y=c^y_{01}(A)y+c^y_{21}(A)x^2y+c^y_{03}(A)y^3.
$$

The first output is odd in $x$ and even in $y$; the second is even in $x$ and odd in $y$. The fourteen forbidden positions are exactly zero: both constants; $y$ in output $x$ and $x$ in output $y$; all six quadratic positions; $x^2y,y^3$ in output $x$ and $x^3,xy^2$ in output $y$. No equality between swapped x/y coefficients is imposed. Antipodal oddness alone would leave twelve positions; the separate reflections yield six.

Each of the six allowed coefficients has a linear accommodation law:

$$
c(A)=c_0+c_1A.
$$

Intercepts are in deg and slopes in deg/D. With tensor axis order `(A power, monomial, gaze axis)`, runtime stores a $2\times10\times2$ view; forbidden intercepts and slopes remain exactly zero. Six lines require twelve independently fitted values. **Twenty polynomial positions at each $A$, twelve stored gaze-law values, and twenty-four total FL law values are different counts.** The fallback F retains all twenty positions, each with a quadratic trend, giving sixty gaze-law values.

Historical raw-mm coefficients require basis conversion: $c^{\rm raw}_{ij,\alpha}(A)=c^\alpha_{ij}(A)/(s_x^is_y^j)$, with units $\mathrm{deg}/\mathrm{mm}^{i+j}$. They cannot be inserted directly into the normalized basis.

## 4. Camera, lens reference and ring size

The two measurement branches use different frames. Gaze uses the sensor mean directly. Size removes the fixed camera distortion from each point before removing translation. The camera centre $\mathbf c_{\rm cam}$ is distinct from $\mathbf o_0$; shifting an origin and changing magnification are different operations.

For pre-camera point $\mathbf z$ in mm, the camera map is

$$
\mathbf X=\mathbf c_{\rm cam}+\operatorname{Cam}(\mathbf z),
\qquad \operatorname{Cam}(\mathbf z)=\mathbf z(1+\kappa_1\lVert\mathbf z\rVert^2+\kappa_2\lVert\mathbf z\rVert^4),
$$

$$
\kappa_1=K_1/F_{\rm cam}^2,\qquad \kappa_2=K_2/F_{\rm cam}^4.
$$

Here $F_{\rm cam}$ is in mm, $K_1,K_2$ are dimensionless, and $\kappa_1,\kappa_2$ have units $\mathrm{mm}^{-2},\mathrm{mm}^{-4}$. The quartic-radius term means whole-field behaviour cannot be classified from $K_1$ alone.

At zero gaze, the two-radius empirical lens reference is

$$
\mathbf z_i^{\,0}(A)=H_A\mathbf d_i(1+q_A\lVert\mathbf d_i\rVert^2).
$$

$H_A$ describes radial scale or magnification and $q_A$ describes the relative edge/corner spacing. Negative $q_A$ gives barrel-like compression relative to the linear reference on a valid monotone branch; positive $q_A$ gives pincushion-like expansion. Two radii do not identify the full physical lens law away from the samples. The runtime reference is the fixed A00 pair $H_0,q_0$.

Earlier angular-response interpretations used, with $\rho^2=\lVert\boldsymbol\tau\rVert^2$,

$$
\mathbf m\approx G_A\boldsymbol\tau(1+\epsilon_A\rho^2+\epsilon_{5,A}\rho^4),
\qquad
\mathbf k\approx\kappa\boldsymbol\tau,\quad
\boldsymbol\delta\approx\delta_0\boldsymbol\tau,\quad
L\approx I+m_1\rho^2I+m_2\boldsymbol\tau\boldsymbol\tau^T.
$$

$G_A$ is a mean-motion gain in mm, the epsilon terms describe empirical angular curvature, $\mathbf k$ is a candidate dimensionless keystone vector, $\boldsymbol\delta$ a barrel-input offset in pattern units, and $L$ a dimensionless stretch map. These historical forms describe possible mechanisms, not FL runtime equations or accepted fitted physical laws. Mean shift, radial curvature, keystone-like perspective deformation and stretch can mix under camera composition; offset, prism and keystone can mimic one another at low order. Fitted inverse-polynomial coefficients are least-squares coefficients, not Taylor identities for those mechanisms. FL neither estimates an identifiable physical keystone law nor runs the separate lens-form LF mean inverse.

For size, compute

$$
\mathbf z_i=\operatorname{Cam}^{-1}(\mathbf X_i-\mathbf c_{\rm cam}),
\qquad \overline{\mathbf z}=\frac18\sum_i\mathbf z_i,
\qquad \mathbf S_i=\mathbf z_i-\overline{\mathbf z}.
$$

Pointwise camera inversion comes **before** mean removal. Nonlinear camera inversion and averaging do not commute. Normalize the two radii by their own positive A00 baselines:

$$
B_i=H_0(1+q_0r_i^2),\qquad \mathbf Y_i=\mathbf S_i/B_i.
$$

$B_i$ is in mm and $\mathbf Y_i$ is dimensionless. Project onto an overall radial pattern and an extra corner pattern:

$$
(\widehat u,\widehat v)=\arg\min_{u,v}\sum_{i=1}^8
\lVert\mathbf Y_i-u\mathbf d_i-v\mathbf d_i(r_i^2-1)\rVert_2^2,
\qquad \mu_{\rm raw}=\widehat u,\quad \beta_{\rm raw}=\widehat v/\widehat u.
$$

$\beta_{\rm raw}$ is defined only when $\widehat u\ne0$. The reconstructed radial part is

$$
\widehat{\mathbf S}_i=B_i\mu_{\rm raw}\mathbf d_i[1+\beta_{\rm raw}(r_i^2-1)].
$$

Thus the edge factor is $\mu_{\rm raw}$ and the corner factor is $\mu_{\rm raw}(1+\beta_{\rm raw})$, relative to their respective A00 baselines. Only $\mu_{\rm raw}$ drives accommodation inference. The relative shape and projection residual are descriptive; this statistic is not an RMS radius.

Zero-gaze anchor rings supply measured ratios

$$
\mu_j=\frac{H_{A_j}(1+q_{A_j})}{H_0(1+q_0)},
\qquad
\beta_j=\frac{H_{A_j}(1+2q_{A_j})}{H_0(1+2q_0)\mu_j}-1.
$$

The measured A00 ordinate is $\mu_0=1$. It measures the product $H_A(1+q_A)$ relative to A00, not $H_A$ alone. The fitted FL size-line intercept is estimated from all five anchors and need not equal one exactly.

## 5. Default FL size surface

Gaze changes the measured ring size as well as its mean. The five-term even, dimensionless basis is

$$
\mathbf E(\boldsymbol\tau)=(\tau_x^2,\tau_y^2,\tau_x^4,\tau_x^2\tau_y^2,\tau_y^4).
$$

The squared terms express second-order size response, the pure fourth powers express fourth-order axis response, and the mixed fourth-order term couples both axes. There is no intercept, so $\mathbf E(\mathbf0)=0$. The code name `E4` denotes this fourth-order size basis, not the historical four-corner estimator with the same name.

FL fits a line for each of the five size-response components and one line for zero-gaze size:

$$
\boldsymbol\gamma(A)=\boldsymbol\gamma_0+A\boldsymbol\gamma_1,
\qquad \mu_{\rm cal}(A)=u_0+u_1A,
$$

$$
M(A,\boldsymbol\tau)=(u_0+u_1A)
\left[1+\mathbf E(\boldsymbol\tau)\mathbin{\cdot}(\boldsymbol\gamma_0+A\boldsymbol\gamma_1)\right].
$$

$\boldsymbol\gamma_0,u_0$ are dimensionless and $\boldsymbol\gamma_1,u_1$ have units $\mathrm D^{-1}$. The five gamma lines contain ten fitted values; the mu line contains two. The complete size surface generally contains

$$
u_1[\mathbf E(\boldsymbol\tau)\mathbin{\cdot}\boldsymbol\gamma_1]A^2.
$$

This induced product term is required even though every separately fitted law is linear in $A$. It is neither an independently fitted quadratic trend nor an additional stored coefficient. Do not flatten $M$ into a straight line in $A$.

The empirical observation relation is $\mu_{\rm raw}=M(A,\boldsymbol\tau)+\varepsilon_\mu$, with dimensionless mismatch $\varepsilon_\mu$ even for noise-free traces. The size-response coefficients are not uniquely identified barrel/keystone constants and are distinct from $q_A$ and the camera coefficients. A gaze correction frozen at A00 followed by pure size correction is the rejected separable baseline; the response must vary with candidate accommodation inside the inversion.

## 6. Two-stage inference and iteration

At a current accommodation guess $a$, estimate gaze and its tangents:

$$
\mathbf w_a=P_a(\mathbf m),\qquad \boldsymbol\tau_a=\boldsymbol\tau(\mathbf w_a).
$$

Then hold that gaze fixed and solve the sampled fixed-gaze equation

$$
M(A,\boldsymbol\tau_a)=\mu_{\rm raw}.
$$

Candidate $A$ varies **both** $u_0+u_1A$ and $\boldsymbol\gamma_0+A\boldsymbol\gamma_1$. The observed mean and size remain fixed. Let $T_{\rm clip}(a)$ denote this complete conditional gaze, table construction and endpoint-clipped inverse update. Iterate

$$
A^{(0)}=0\ \mathrm D,\qquad A^{(k)}=T_{\rm clip}(A^{(k-1)}),
\qquad |A^{(k)}-A^{(k-1)}|\le10^{-6}\ \mathrm D.
$$

At most twenty passes are allowed. Each table has 401 nodes from 0 to 4 D at 0.01 D spacing and must be finite and strictly monotone. In-range inversion interpolates between sampled nodes; finite out-of-range sizes select the nearest endpoint, with ties selecting 0 D. Invalid tables and states that fail to stop are failures. Final gaze is recomputed at the returned accommodation:

$$
\widehat A=A^{(k)},\qquad \widehat{\mathbf w}=P_{\widehat A}(\mathbf m).
$$

This is alternating conditional inference, not simultaneous nonlinear least squares. Stopping controls an accommodation update; a clipped endpoint may be stable while retaining size mismatch. Fixed-gaze table monotonicity does not prove contraction, uniqueness or convergence of the coupled map. See [implementation Section 5](P4_IMPLEMENTATION.md#5-table-inversion-and-runtime-iteration) for failure and diagnostic rules.

## 7. Calibration and fallback

Only the five anchors $A_j=j\ \mathrm D$, $j=0,\ldots,4$, calibrate the runtime laws. Full cubics are first fitted on training gazes at each anchor. FL records the largest forbidden coefficient at each anchor, sets the fourteen forbidden positions to exact zero, then fits the six allowed coefficient lines on the gated design $[1,A]$.

The five measured zero-gaze size ordinates fit $u_0,u_1$. Runtime uses ordinates generated by that fitted line. Anchor gamma vectors are then refitted using true-gaze tangents and targets $\mu_{\rm raw}/\mu_{\rm FL}(A_j)-1$, followed by five gated $[1,A]$ gamma trends. Gamma is not obtained by deleting a quadratic row from F: the fitted size-line denominator changes its anchor targets. True gaze is used for calibration targets and scoring, not as a runtime input.

Shared preparation also fits own-accommodation cubics and performs checks on non-anchor shards, but no non-anchor fitted quantity calibrates either FL or F. All numerical/data gates remain unchanged. The FL laws require $12+10+2=24$ independently fitted values; fixed scales, mean origin and radial reference raise the count to 30, and the fixed camera raises it to 35. These are stored law/calibration counts, not the number of monomials at fixed accommodation.

Fallback F retains twenty gaze coefficients with quadratic accommodation trends (60 values), five quadratic gamma trends (15 values), and piecewise-linear interpolation through five measured size ordinates (5 values): 80 law values, or 86/91 with the same reference/camera additions. F's size ordinates are not a fitted quadratic law. Its greater development accuracy remains available without changing FL's accepted default status.

## 8. Recorded performance and limits

The completed comparison covers $21\times10{,}201=214{,}221$ noise-free design states over $A=0$ to $4$ D and a $\pm12.5^\circ$ field per axis. Both models have 100% coverage. FL gaze RMS/maximum is 0.1544/1.0176 arcmin; F is 0.0195/0.1904 arcmin. FL accommodation RMS/maximum is approximately 0.019/0.168 D; F is approximately 0.0019/0.033 D. Eight FL states exceed 1 arcmin, all at interior accommodation values. The original strict result is `L0=True`, `L1=True`, `L2=False`, `CLOSE: F`; the user subsequently accepts approximately 1 arcmin and selects FL.

These are development traces, with non-anchor accommodations previously inspected. No measurement-noise, real-eye or blind validation has been performed. Anchor/all-state summaries include training gazes. No accommodation-error acceptance threshold has been set. Fixed camera/reference assumptions, endpoint clipping and empirical mismatch limit interpretation. There is no contraction or unique-fixed-point proof, off-grid accuracy claim or guarantee over a continuous domain. AE7 simultaneous fitting has not been performed. [Implementation Section 7](P4_IMPLEMENTATION.md#7-recorded-results-and-acceptance) contains the numerical comparison and maxima locations.

## 9. HANDOFF: goal closed

The current goal is closed by the user's 2026-11-16 acceptance: **FL default; F/quadratic fallback or future use.** The completed comparison and its strict verdict remain intact. These two P4 documents define the accepted estimator and its implementation; `CURRENT_PLAN.md` Section 17 records this documentation/handoff contract, and the dated `Record.md` closure preserves the decision and evidence.

Candidate next goals are blind/noisy/real-eye validation, accommodation accuracy criteria, iteration robustness and convergence analysis, or AE7 simultaneous fitting. They are separate goals; none is a prerequisite left open for this closure. Reproduction is documented in [implementation Section 8](P4_IMPLEMENTATION.md#8-reproduction-and-source-references); no execution is part of this documentation update.
