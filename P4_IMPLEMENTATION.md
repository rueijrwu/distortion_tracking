# P4 implementation: default FL procedures and calibration

**FL is the accepted default; F is the quadratic fallback or future option.** The user accepted approximately 1 arcmin gaze error on 2026-11-16. The original strict experiment remains `L0=True`, `L1=True`, `L2=False`, `CLOSE: F`. This document records the existing implementation and completed development comparison; this documentation update executes no scripts/tests and changes no code, constants, gates or outputs. See [P4_MODEL.md](P4_MODEL.md) for theory and [CURRENT_PLAN.md Section 17](CURRENT_PLAN.md#17-accept-fl-as-default-consolidate-p4-documentation-close-the-goal) for the current documentation contract.

## 1. Runtime overview and source map

`Script\p4_linA.py` is the primary reproduction/calibration entry point. It builds F and FL from the same validated data, then scores both with unchanged helpers. FL runtime consumes measured mean displacement and projected ring size plus its fixed calibration; ground-truth gaze/accommodation are calibration/scoring information, not runtime inputs.

The complete local source closure is `p4_linA` -> `p4_final` -> `p4_accom_test`, plus eight helper modules: `p4_a0_data`, `p4_a0_gazepoly`, `p4_a0_ninept5`, `p4_termwise_test`, `p4_a0_ninept`, `p4_a0_joint_fit`, `p4_a0_gain` and `p4_a0_tilt_fit`. These eleven `Script\*.py` files use static local imports; no dynamic import or script scheduled for removal is needed by this closure. Retaining a helper does not authorize running its separate historical experiment.

The eleven retained scripts form an import closure, not an assurance that every helper runs independently with its legacy defaults. The legacy default input name `p4_raytrace_251x251_12p5_A4.pkl` is stale for `p4_a0_gain.py`, `p4_a0_tilt_fit.py`, `p4_a0_joint_fit.py` and `p4_a0_ninept.py`; production FL/F routes (`p4_linA.py` and `p4_final.py`) do not use that default and instead use the documented 21 full 5x5 shards plus manifest under `data\p4_raytrace`.

The operational order is:

1. Compute the eight-point sensor mean and subtract the fixed A00 mean origin.
2. Invert the camera at each of the eight sensor points, remove the pre-camera mean, normalize the two radii and obtain $\mu_{\rm raw}$ by radial projection.
3. Start at $A^{(0)}=0$ D. Evaluate the FL gaze polynomial at the current accommodation.
4. Hold that gaze fixed, build the joint size table while varying candidate accommodation, and invert it with endpoint clipping.
5. Repeat until the accommodation update stops or the state fails. Recompute final gaze and the size residual at the returned accommodation.

| Source | Responsibility |
|---|---|
| `Script\p4_linA.py`, `calibrate` | Masked linear gaze trends, fitted size line, refitted anchor gamma and linear gamma trends |
| `Script\p4_linA.py`, `real_run`, `scored`, `pooled`, `verdict` | Shared preparation, F/FL comparison, full-denominator scoring and original strict verdict |
| `Script\p4_accom_test.py`, `prepare_all`, `shared_preparation`, `size_calibration` | Shard/data gates, fixed camera and A00 reference, own-accommodation cubics and measured anchor sizes |
| `Script\p4_accom_test.py`, `linear_fit`, `evaluate_gaze`, `two_radius_projection` | Gated regressions, conditional gaze and radial size statistic |
| `Script\p4_accom_test.py`, `s1_surface_table`, `s1_table_inverse`, `s2_map`, `s2_loop` | Fixed-gaze table, strict sampled inverse, complete update and stopping |
| `Script\p4_final.py`, `run_loop`, `size_value`, `score` | Shared clipped runtime, final gaze/size evaluation, pooled diagnostics |
| `Script\p4_termwise_test.py`, `TERMS3`, `ALLOWED`, `tangent_coordinates`, `checked_cubic` | Feature order, six allowed positions, tangent signs and cubic checks |
| `Script\p4_a0_gazepoly.py`, `fit_polynomial`; `Script\p4_a0_joint_fit.py`, `camera_inverse`, `radial_inverse` | Retained anchor cubic fit; camera radial inversion |
| `Script\p4_a0_data.py` | Shared structured-data schema and serialization helpers |
| `Script\p4_a0_ninept5.py` | Manifest-complete 5x5 shard loading, node geometry and validation |
| `Script\p4_a0_ninept.py` | Shared nine-point labels, geometry and numerical conventions |
| `Script\p4_a0_gain.py` | Shared input-validation helpers and fixed camera conventions |
| `Script\p4_a0_tilt_fit.py` | Retained shared polynomial/basis helpers from the historical tilt diagnostics |

FL uses the polynomial branch of `fin.run_loop` with `law=None`. The separately named lens-form LF branch is not the default FL algorithm.

## 2. Data, coordinates and normalization

The published data are the 21 `p4_raytrace_5x5_101x101_12p5_A00.pkl` through `p4_raytrace_5x5_101x101_12p5_A20.pkl` shards and `p4_raytrace_5x5_manifest.json` under `data\p4_raytrace`. The existing loader validates manifest/hashes, schema, geometry, ray status, state ordering and split consistency before scoring. Each shard has 10,201 gaze states, giving $21\times10{,}201=214{,}221$ states. Accommodation is $0.2i$ D for shard index $i$; gaze spans $\pm12.5^\circ$ per axis at $0.25^\circ$ spacing on a $101\times101$ grid.

All 25 existing files in this data directory are retained unchanged; the three files in addition to the 21 shards and manifest are not additional runtime inputs. Older 3x3/`251x251` filenames are historical and do not replace the current shard contract. The FL gaze-law tensor has shape $2\times10\times2$ (intercept/slope, monomial, output); F has $3\times10\times2$ (constant/linear/quadratic accommodation row, monomial, output). Both evaluate the same ten-monomial, two-output polynomial with twenty coefficient positions at each accommodation; fourteen FL positions are exact zeros.

Select the four edge midpoints and four corners by the labels and direction convention $\mathbf d_i=-\boldsymbol\ell_i$ in [model Section 2](P4_MODEL.md#2-measurements-coordinates-and-notation). Use all eight points for the sensor mean. Gaze inputs are

$$
\mathbf m=\frac18\sum_i\mathbf X_i-\mathbf o_0,
\qquad \mathbf o_0=\overline{\mathbf X}(0,\mathbf0),
\qquad x=m_x/s_x,\quad y=m_y/s_y.
$$

All points, origin, displacement and axis scales are in mm; $x,y$ are dimensionless. The shared scales are the per-axis maximum absolute A00 training displacement. Both must be finite and positive. These axis scales are distinct from regression-column scaling. Never replace the fixed mean origin with a per-accommodation origin or camera-invert the sensor mean for gaze.

The camera centre is fixed to the A00 loader calibration, obtained from the horizontal-pair midpoint. It is a separate reference from the mean origin and is not multiplied by the radial scale. Each point is camera-inverted before subtracting the pre-camera mean to obtain `Zshape`. Shared preparation also checks the camera round trip and fits a full own-accommodation cubic on every shard. Those non-anchor fits/checks do not calibrate either runtime model.

| Calibration shard | A00 | A05 | A10 | A15 | A20 |
|---|---:|---:|---:|---:|---:|
| Accommodation, D | 0 | 1 | 2 | 3 | 4 |

Each anchor uses 2,600 nonzero even/even training gazes; zero gaze supplies reference calibration. The other parity splits contain 2,550, 2,550 and 2,500 states. Only these five anchors feed the runtime laws. Ground-truth gaze is $\mathbf w=(\mathrm{BDE},-\mathrm{ADE})^T$ in degrees. The size design uses $\theta=(\pi/180)\mathbf w$, $\tau_x=\tan\theta_x/\cos\theta_y$, $\tau_y=\tan\theta_y$; calibration uses true gaze and runtime uses estimated gaze.

Before size diagnostics, `real_run` adds `field_lt12` as the logical OR of the first three existing field bins on every state. It computes $\mu_{\rm raw}$ with `two_radius_projection(Zshape,H0,q0)` using the unchanged fixed reference.

## 3. FL calibration and immutable numerical gates

The following is the order in `Script\p4_linA.py:43` onward. It uses five anchors and the two-column design

$$
V=\begin{pmatrix}1&0\\1&1\\1&2\\1&3\\1&4\end{pmatrix},
\qquad V\in\mathbb R^{5\times2}.
$$

First, shared preparation fits full anchor cubics. The retained coefficients are the $10\times2$ output of `fit_polynomial` on axis-normalized inputs with SVD least squares and `rcond=1e-12`. `checked_cubic` requires rank 10, finite coefficients and finite predictions. A separate `linear_fit` call checks the column-scaled cubic design; its checking solution does not replace the retained coefficients. Ordinary cubic condition and scaled-design condition are separate records.

**Gaze trends.** Copy the five retained coefficient matrices, record the largest absolute forbidden coefficient at each anchor, then set the fourteen forbidden positions to exact zero. Fit the $5\times20$ flattened targets on $V$ using `linear_fit`, reshape to $2\times10\times2$, and assert forbidden intercepts/slopes are exactly zero. Only output-x terms $x,x^3,xy^2$ and output-y terms $y,x^2y,y^3$ remain. There is no x/y coefficient equality. The recorded dropped magnitudes range from approximately $7.39\times10^{-15}$ to $1.11\times10^{-14}$ deg in the normalized basis.

**Zero-gaze size line.** Fit the five measured anchor ordinates from `size_calibration` on $V$ to obtain $u_0,u_1$. The measured observations are

$$
\mu_j=\frac{H_{A_j}(1+q_{A_j})}{H_0(1+q_0)}.
$$

Build a separate FL calibration view whose complete mu array is $u_0+u_1A$; keep measured anchor ordinates separately as observations. Runtime helpers interpolate its five anchor samples. Samples of one line evaluate that line between anchors; they are not five independent FL size parameters. The fitted intercept is not constrained to equal the measured $\mu_0=1$.

**Gamma anchor fits, then trends.** At each anchor, use true-gaze tangent coordinates on training states and refit five components with

$$
\mathbf E(\boldsymbol\tau)=(\tau_x^2,\tau_y^2,\tau_x^4,\tau_x^2\tau_y^2,\tau_y^4),
\qquad
\gamma_j=\arg\min_{g\in\mathbb R^5}\sum_{n\in\mathcal T_j}
\left[\mathbf E(\boldsymbol\tau_{jn})\mathbin{\cdot}g-
\left(\frac{\mu_{{\rm raw},jn}}{u_0+u_1A_j}-1\right)\right]^2.
$$

Fit the five anchor vectors on $V$ to obtain the actual $2\times5$ gamma law. The FL denominator is the fitted size line, so these anchor fits must be recomputed; F's gamma values cannot simply be truncated. FL does not call `s1_fit_surface`, which unconditionally fits quadratic trends. Store actual two-row laws and rank-2 records. Append a third exact-zero gamma row only to the runtime compatibility view, because unchanged helpers index three rows. This padding is neither a fitted quadratic term nor an independent parameter. `gamma_A00` retains the first direct anchor fit; the joint runtime uses the trend.

The resulting table and final residual both evaluate the full product

$$
M(A,\boldsymbol\tau)=(u_0+u_1A)
\left[1+\mathbf E(\boldsymbol\tau)\mathbin{\cdot}(\gamma_0+A\gamma_1)\right].
$$

Its induced $u_1(\mathbf E\mathbin{\cdot}\gamma_1)A^2$ term is required. No separately fitted FL law has degree two.

For every `linear_fit`, require finite inputs, matching row counts and finite positive column norms. For a design $D$, use column scaling

$$
s_k=\lVert D_{:k}\rVert_2,\qquad \widetilde D=D\operatorname{diag}(s_k^{-1}),
\qquad \operatorname{cond}_2(\widetilde D)\le10^8.
$$

SVD least squares uses `rcond=1e-12`; require full column rank and finite coefficients, then undo scaling. Required ranks are 10 for the scaled full cubic check, 2 for each FL accommodation-trend design, 5 for each gamma anchor design, and 2 for the radial projection. The recorded FL trend condition is approximately 3.1463; gamma anchor condition is approximately 12.8922. No lower-rank solution, reduced anchor set or substitute constants is accepted after failure.

The existing nonlinear-fit threshold $10^{-10}$ is unchanged. For an unbounded nonlinear least-squares problem with residual $r$, Jacobian $J$, column scales $s$ and $\widetilde J=J\operatorname{diag}(s^{-1})$, its normalized scaled-gradient criterion is

$$
\frac{\lVert\widetilde J^Tr\rVert_\infty}
{\max(1,\sigma_{\max}(\widetilde J)\lVert r\rVert_2)}\le10^{-10}.
$$

It measures first-order stationarity after parameter scaling, rather than residual magnitude or gaze accuracy. The bounded inverse solver uses its established projected-gradient convention. FL calibration calls linear least squares and the runtime calls polynomial/table evaluation; neither invokes those nonlinear least-squares fits. Camera bracketed Newton is a separate root solve. **Neither the scaled-condition gate $10^8$ nor the nonlinear-gradient gate $10^{-10}$ was relaxed.**

Fallback F uses `coefficient_model` and `s1_fit_surface` unchanged: full twenty-position quadratic gaze trends, quadratic gamma trends and piecewise-linear interpolation of five measured mu ordinates. Quadratic fitting belongs only to that branch.

## 4. Camera inversion and ring-size numerics

The fixed forward camera is

$$
\operatorname{Cam}(\mathbf z)=\mathbf z(1+a r^2+b r^4),
\quad r=\lVert\mathbf z\rVert_2,
\quad a=K_1/F_{\rm cam}^2,\quad b=K_2/F_{\rm cam}^4,
\quad \mathbf X=\mathbf c_{\rm cam}+\operatorname{Cam}(\mathbf z).
$$

The scalar inverse preserves direction. For $\mathbf y=\mathbf X-\mathbf c_{\rm cam}$ and $\rho=\lVert\mathbf y\rVert_2$, solve

$$
f(r)=r(1+ar^2+br^4)=\rho,
\qquad
\mathbf z=\begin{cases}(r/\rho)\mathbf y,&\rho>0,\\\mathbf0,&\rho=0.\end{cases}
$$

Radii and vectors are in mm. The origin-connected branch requires positive tangential and radial eigenvalues throughout the interval from zero to the recovered radius:

$$
1+ar^2+br^4>0,\qquad 1+3ar^2+5br^4>0.
$$

`radial_inverse` brackets below the first positive radial-derivative zero when one exists; otherwise it expands the upper bound, with at most eighty expansions and the existing finite/radius checks. It checks exact eigenvalue minima in squared radius at endpoints and stationary points. A Newton proposal is used only with positive derivative and a proposal strictly inside the bracket; otherwise use its midpoint. Permit at most one hundred updates and require

$$
|f(r)-\rho|\le2\times10^{-14}\max(1,\rho).
$$

The 1 denotes the numerical 1 mm reference magnitude. Unbracketable targets, failed convergence and nonpositive branch eigenvalues fail. This is inversion of the full radial polynomial, not a truncated inverse series or coefficient-sign reversal. Preparation additionally requires each forward/inverse round-trip norm, converted from mm to micrometres, to be strictly below $10^{-9}\ \mu\mathrm m$.

For zero-gaze calibration, the loader measures edge half-span $a_A$ and the selected corner's positive x coordinate $b_A$, both in mm, then uses

$$
H_A=2a_A-b_A,\qquad q_A=(b_A-a_A)/H_A,
\qquad a_A=H_A(1+q_A),\quad b_A=H_A(1+2q_A).
$$

The corner radius is $\sqrt2b_A$. Retain positivity, symmetry, branch and $10^{-6}$ mm calibration-defect checks. Fix $H_0,q_0$ to A00.

After pointwise inversion, remove the pre-camera mean, divide each point by $B_i=H_0(1+q_0r_i^2)$ and stack each point's x and y components consecutively. The $16\times2$ dimensionless radial design has columns $\mathbf d_i$ and $\mathbf d_i(r_i^2-1)$. Its gated rank-2 least-squares solution gives $\widehat u,\widehat v$, with $\mu_{\rm raw}=\widehat u$ and $\beta_{\rm raw}=\widehat v/\widehat u$ when the denominator is nonzero. Baselines must be finite and positive. The radial reconstruction and residual are defined in [model Section 4](P4_MODEL.md#4-camera-lens-reference-and-ring-size); only mu enters runtime accommodation inference.

Measured zero-gaze edge/corner factors must be finite and positive and measured primary anchor sizes strictly monotone. The beta inverse is a diagnostic and may be unavailable. FL replaces runtime mu ordinates with its fitted line; every state/pass table must still pass the unchanged strict monotonicity gate.

## 5. Table inversion and runtime iteration

`fin.run_loop` calls `ac.s2_loop` with $A^{(0)}=0$ D. At each active pass evaluate $\mathbf w=P_a(\mathbf m)$ using the current accommodation $a$ and convert it to tangents. Hold that gaze fixed while forming

$$
a_\ell=0.01\ell\ \mathrm D,\quad \ell=0,\ldots,400,
\qquad M_\ell=M(a_\ell,\boldsymbol\tau(\mathbf w)).
$$

There are 401 nodes from 0 to 4 D. Candidate accommodation varies both mu and gamma; table entries change with the state and current gaze and are not stored calibration parameters.

The target and all table entries must be finite. All adjacent differences must be strictly positive or all strictly negative; flat segments fail. Reverse a decreasing table for interpolation. For an in-range target, return

$$
A_{\rm next}=a_\ell+(a_{\ell+1}-a_\ell)
\frac{\mu_{\rm raw}-M_\ell}{M_{\ell+1}-M_\ell}.
$$

This is the inverse of a sampled piecewise-linear table, not an analytic inverse of continuous $M$. For a finite out-of-range target in a valid monotone table, choose the endpoint closest in size. Equal distances select the first endpoint, 0 D. Endpoint clipping handles range failure only; it does not repair nonfinite/nonmonotone tables.

Stop when $|A_{\rm next}-a|\le10^{-6}$ D, with at most twenty passes and no restart. A pass failure records its reason and pass number. A state still active after twenty passes receives reason 6 and a NaN final estimate; the last iterate is diagnostic only. Final gaze is recomputed at returned $\widehat A$, and nonfinite final gaze fails the state. Failed estimates/gazes become NaN and their clipping flags are cleared.

The clipped flag describes the stopping pass, not whether any earlier pass clipped. One extra complete accommodation update evaluates the fixed-point residual $|T_{\rm clip}(\widehat A)-\widehat A|$ without changing the estimate or pass count. Its diagnostic failure is separate from the estimate status. The final size residual uses both fitted FL factors at the returned accommodation:

$$
r_{\rm size}=|M(\widehat A,\boldsymbol\tau(\widehat{\mathbf w}))-\mu_{\rm raw}|.
$$

The reported micrometre diagnostic multiplies this dimensionless residual by $R_c=1000\sqrt2H_0(1+2q_0)$. A clipped fixed point may have zero update residual but nonzero size residual. No contraction or uniqueness claim follows from stopping or table monotonicity.

Failure reasons are: 0 valid; 1 nonfinite input/evaluation (including an accommodation outside gaze bounds); 2 out of calibrated range in the strict comparison route; 3 unavailable secondary calibration; 4 nonmonotone table; 5 unavailable estimator/calibration; 6 not stopped within twenty passes. FL/F use the clipped primary route; reasons for unrelated shared routes are retained for schema compatibility.

## 6. Stored constants and parameter counts

The polynomial at each accommodation still has twenty coefficient positions: ten monomials times two outputs. FL forces fourteen positions to zero and fits the six remaining positions linearly in accommodation, requiring twelve stored gaze-law values. The following counts describe independent stored law/calibration values, **not polynomial size**:

| Independent values | FL default | F fallback |
|---|---:|---:|
| Gaze laws | $6\times2=12$ | $20\times3=60$ |
| Gamma laws | $5\times2=10$ | $5\times3=15$ |
| Mu calibration | 2 line coefficients | 5 measured ordinates |
| Law total | **24** | **80** |
| Add axis scales (2), mean origin (2), $H_0,q_0$ (2) | **30** | **86** |
| Add camera $F_{\rm cam},K_1,K_2$, centre (2) | **35** | **91** |

Structural zeros, the compatibility gamma row, line-derived ordinates, fixed anchor abscissae, grid/loop settings, derived camera kappas and fit/provenance metadata are not independent law parameters. Serialized arrays can therefore contain more numbers than these counts. Full anchor cubics and gamma vectors are intermediate/diagnostic calibration results, not extra runtime trends.

The saved schema includes `FL.gaze` ($2\times10\times2$ with structural zeros), `FL.gamma` ($2\times5$), `FL.mu` (2), the six reference values, fixed camera, feature/axis ordering, anchors, grid/loop settings, fit checks, dropped-coefficient magnitudes, script hash, pooled scores and original verdict. Runtime alone pads gamma to $3\times5$ with an exact-zero final row. Long serialized arrays are omitted here; the compact tables below give the accepted laws rounded for reading, not replacement machine-precision constants.

| FL gaze position | Intercept, deg | Slope, deg/D |
|---|---:|---:|
| $w_x:x$ | -12.6161413922 | -0.128762585601 |
| $w_x:x^3$ | 0.00971792148118 | 0.0237640024880 |
| $w_x:xy^2$ | 0.109410571639 | 0.0257883408587 |
| $w_y:y$ | -12.5083151466 | -0.128145266142 |
| $w_y:x^2y$ | -0.194976150448 | 0.0188309263787 |
| $w_y:y^3$ | 0.00969049517256 | 0.0237497615599 |

| FL size-response component | Intercept | Slope, $\mathrm D^{-1}$ |
|---|---:|---:|
| $\tau_x^2$ | 0.799216962631 | 0.146348760346 |
| $\tau_y^2$ | 0.799200545644 | 0.146309189922 |
| $\tau_x^4$ | -0.589189681090 | 0.0256744075843 |
| $\tau_x^2\tau_y^2$ | -1.13725204395 | 0.0463610653434 |
| $\tau_y^4$ | -0.588782030251 | 0.0266536012423 |

The fitted mu law is $\mu_{\rm cal}(A)=1.0000014269192838-0.019044052033249284A$, with $A$ numerically in D. Its intercept is dimensionless and slope in $\mathrm D^{-1}$.

| Fixed reference | Recorded value |
|---|---|
| $s_x,s_y$, mm | 6.9304104979129635; 6.871249659937712 |
| $\mathbf o_0$, mm | $(0,-5.551115123125783\times10^{-17})$ |
| $H_0$, mm; $q_0$ | 2.591651929169045; -0.030048720379678887 |
| $F_{\rm cam}$, mm; $K_1$; $K_2$ | 425.137557657; 2.943551420; -106.482286730 |
| $\mathbf c_{\rm cam}$, mm | $(0,0)$ |

## 7. Recorded results and acceptance

These values are copied from the completed comparison and are not a new execution. Evidence label: **noise-free design traces; ground-truth gaze used for calibration targets and scoring only; development data, non-anchor accommodation previously inspected, not fresh validation**. Script SHA-256: `e5341f7dcf26bc17ad761b372a18a5da648cf752362b85ee5dd40d666df626c1`.

The separately retained `p4_final` comparison record preserves quadratic F alongside earlier lens-form LF-L2 and LF-L3 comparisons. In its constants schema, `polynomial.coefficient` has shape `[3,10,2]` and is quadratic F evidence. Its F-labelled A00 row and F pool rows for A00, A01-A19, A20 and all21 likewise belong to F. The separate historical terminal `selected_form=L3` / `CLOSE: FULL` records the earlier LF selection after that experiment's ring goal narrowed to ring size; it is neither the later FL acceptance nor a reinterpretation of those F coefficients or F-labelled rows. The original records remain unchanged, and the relevant interpretation is stated here without an output-file dependency.

For each state, $e_w=60\lVert\widehat{\mathbf w}-\mathbf w\rVert_2$ in arcmin and $e_A=|\widehat A-A|$ in D. Pool state arrays rather than rounded shard RMS values. Failures remain in the denominator and make the full-group RMS/maximum infinite; threshold counts count valid states only, with failures reported separately. The completed comparison has no failed states for either model.

| All 214,221 states | FL default | F fallback |
|---|---:|---:|
| Valid / failed; coverage | 214,221 / 0; 100% | 214,221 / 0; 100% |
| Gaze RMS / maximum, arcmin | 0.1544 / 1.0176 | 0.0195 / 0.1904 |
| A error RMS / maximum, D | 0.0190 / 0.1679 | 0.0019 / 0.0330 |
| Passes median / maximum | 6 / 11 | 6 / 12 |
| Clipped on stopping pass | 11,209 | 6,180 |
| Valid states with gaze error $>0.1$ arcmin | 93,258 | 120 |
| Valid states with gaze error $>1$ arcmin | 8 | 0 |

| Model / pool | States | Gaze RMS / max, arcmin | A RMS / max, D | Passes median / max | Clips |
|---|---:|---:|---:|---:|---:|
| FL / A00 | 10,201 | 0.0652 / 0.0956 | $2.66\times10^{-6}$ / $7.49\times10^{-5}$ | 1 / 2 | 10,164 |
| FL / A01-A19 | 193,819 | 0.1537 / 1.0176 | 0.0190 / 0.1679 | 6 / 11 | 0 |
| FL / A20 | 10,201 | 0.2188 / 0.8403 | 0.0260 / 0.1046 | 6 / 11 | 1,045 |
| FL / five anchors | 51,005 | 0.1622 / 1.0176 | 0.0197 / 0.1679 | 6 / 11 | 11,209 |
| FL / sixteen non-anchors | 163,216 | 0.1519 / 1.0043 | 0.0187 / 0.1663 | 6 / 11 | 0 |
| F / A00 | 10,201 | 0.0184 / 0.0519 | 0.0004 / 0.0042 | 3 / 6 | 0 |
| F / A01-A19 | 193,819 | 0.0193 / 0.1249 | 0.0020 / 0.0330 | 6 / 12 | 0 |
| F / A20 | 10,201 | 0.0252 / 0.1904 | 0.0014 / 0.0044 | 4 / 10 | 6,180 |

Every pool has 100% coverage. All eight FL states above 1 arcmin are in A01-A19: four anchor and four non-anchor states. Here “interior” means accommodation between the endpoints, not a held-out calibration set. The FL gaze maximum is 1.0175524717148585 arcmin at A10 (2 D), state 10200, gaze $(12.5,-12.5)^\circ$. The maximum accommodation error is 0.16788211350886328 D at A10, state 0, gaze $(-12.5,12.5)^\circ$. Its A00 maximum gaze error is at state 1666, gaze $(0,8.5)^\circ$; its A20 gaze/A maxima occur at state 10100, gaze $(-12.5,-12.5)^\circ$.

F's overall gaze maximum is 0.1903982095933611 arcmin at A20, state 100, gaze $(12.5,12.5)^\circ$. Its maximum A error is 0.032997891682490454 D at A19, state 10200, gaze $(12.5,-12.5)^\circ$. F reproduction passed all recorded checks: quoted gaze maxima and interior RMS within $10^{-6}$ arcmin; interior A RMS/maximum within $10^{-8}$ D of the quoted eight-decimal references; exact median/maximum passes 6/12; and 100% full-state coverage.

FL's largest reported size-equation residual is approximately $2.80\times10^{-5}\ \mu\mathrm m$ for unclipped states and 2.9361 $\mu\mathrm m$ for clipped states. F's corresponding maxima are approximately $1.56\times10^{-5}$ and 0.4967 $\mu\mathrm m$. A stopped/clipped estimate therefore need not solve the continuous size equation.

The original strict verdict is preserved:

| Check | Recorded outcome |
|---|---|
| L0: synthetic selftest, data/preparation/calibration and F reproduction | True |
| L1: all 214,221 FL states valid | True |
| L2: FL maximum gaze error strictly $<1.0$ arcmin | **False** |
| Original experiment verdict | **`CLOSE: F`** |

The exact FL maximum is greater than 1, so it cannot pass that strict gate. The user's subsequent acceptance of approximately 1 arcmin selects **FL as default**, with F/quadratic as fallback or future use. This is a model-selection decision after the measured result, not a rewritten test, relaxed numerical gate or claim that L2 passed. No A-error acceptance threshold exists.

Limits remain: no real-eye or measurement-noise validation; no blind accommodation validation because non-anchor values were previously inspected; training gazes included in anchor/all-state summaries; no off-grid accuracy, contraction, uniqueness or continuous-domain convergence proof; endpoint clipping can conceal a size-equation root outside the calibration interval. AE7 simultaneous fitting has not been performed.

## 8. Reproduction and source references

From the repository root, the recorded comparison can be reproduced with the required absolute Python interpreter:

```powershell
& 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe' .\Script\p4_linA.py
```

This command is documented only; it is not executed in this consolidation. With the same data and unchanged source, it retains the original strict test and `CLOSE: F` verdict. It does not encode the later user acceptance as `CLOSE: FL`. The script runs its synthetic selftest before production trace I/O and writes its summary, constants and progress record only to its dedicated script output folder. These documents contain the acceptance, equations and key evidence directly and require no output artifact to be read.

The script distinguishes completed comparison (exit 0, including strict `CLOSE: F`), missing/invalid data (2), numerical/preparation/calibration/reproduction failure (3) and synthetic-selftest failure (4). The saved status, process exit and final strict verdict describe different things. `--selftest` is synthetic only and is not a production comparison. No reproduction or selftest was run for this documentation update.

Cleanup verification is static only under `CURRENT_PLAN.md` Section 18. Do not execute either `p4_linA.py` or `p4_final.py`, including `--selftest`: both entry points' selftests overwrite their saved output artifacts. The reproduction command above describes a separate scientific rerun, not a cleanup verification step.

Source anchors for inspection are `Script\p4_linA.py:43` (`calibrate`), `Script\p4_accom_test.py:243` (`linear_fit`), `:398` (`shared_preparation`), `:475` (`two_radius_projection`), `:491` (`size_calibration`), `:852` (`s1_surface_table`), and `:2040` (`s2_map`); `Script\p4_final.py:164` (`size_value`), `:196` (`run_loop`) and `:231` (`score`); `Script\p4_a0_joint_fit.py:196` (`radial_inverse`) and `:464` (`solve`). Function names take precedence if line numbers move. `Record.md` preserves earlier model attempts as history and appends the dated FL closure; older specification/model pointers there are historical, not runtime prerequisites.

**HANDOFF: this goal is closed.** FL is the accepted default and F the fallback/future option. Candidate separate goals are blind/noisy/real-eye validation, accommodation accuracy criteria, iteration robustness/convergence, and AE7 simultaneous fitting. None was performed or promised as part of the present closure.
