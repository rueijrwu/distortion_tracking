# P1 geometry, simulation model, and identifiability

This note separates the idealized first-order geometry from the current CODE V simulation and its fitted surrogate. The analytic sphere result is not a fit to CODE V. The fitted surrogate is not a claim that real-world calibration noise or drift has been modeled.

## Fixed global geometry for the idealized theory

Use transverse coordinate $Y$ and axial coordinate $Z$, with global $+Z$ pointing from the cornea toward the objects. The nominal eye rotation center is the origin and the nominal corneal vertex is $(0,R)$, where $R=13.8\,\mathrm{mm}$. The ideal corneal radius is $r=7.77\,\mathrm{mm}$, giving paraxial focal-length magnitude

$$
f=\frac{r}{2}=3.885\,\mathrm{mm}.
$$

For the idealized geometry, the objects are fixed at

$$
O_1=(0,\infty),\qquad O_2=(5,113.8)\,\mathrm{mm},\qquad
O_3=(5,38.8)\,\mathrm{mm}.
$$

Thus the finite objects have height $H=5\,\mathrm{mm}$ and neutral vertex-relative distances $p_2=100\,\mathrm{mm}$ and $p_3=25\,\mathrm{mm}$. These are theory inputs; they are not the current CODE V zoom settings below. The infinity source direction from cornea to source is global $+Z$; incident light travels along $-Z$.

Let $t$ be physical head translation along global $+Y$, with the objects held fixed. Let $\theta$ be eye rotation in radians, positive when the outward corneal normal tilts toward $+Y$. The rotation center, normal, tangent, and vertex are

$$
C(t)=(t,0),\qquad
\mathbf n=(\sin\theta,\cos\theta),\qquad
\mathbf e=(\cos\theta,-\sin\theta),
$$

$$
V=C+R\mathbf n=(t+R\sin\theta,R\cos\theta).
$$

For the ideal sphere, its curvature center, distinct from both vertex and pivot, is

$$
S=V-r\mathbf n=C+(R-r)\mathbf n
=\left(t+(R-r)\sin\theta,(R-r)\cos\theta\right).
$$

With $L=R-r=6.03\,\mathrm{mm}$, $S_Y=t+L\theta+O(\theta^3)$ and $S_Z=L+O(\theta^2)$. This motivates the shorthand $g=t+L\theta$ for the ideal sphere-center displacement; it does not imply that a complete simulated relay has identical zoom gains.

At zero rotation the finite object's local transverse offset is $H-t=5-t$: positive head translation decreases that offset. The saved collector's `h_translation_mm` stores this physical translation $t$.

For each finite fixed object $O_i=(H,R+p_i)$, set $D_i=O_i-V$. Its exact local axial and transverse coordinates are

$$
P_i=D_i\cdot\mathbf n=(H-t)\sin\theta+(p_i+R)\cos\theta-R,
$$

$$
U_i=D_i\cdot\mathbf e=(H-t)\cos\theta-(p_i+R)\sin\theta.
$$

At $t=\theta=0$, $P_i=p_i$ and $U_i=H$. Their first derivatives are

$$
\left.\frac{\partial P_i}{\partial\theta}\right|_0=H,\quad
\left.\frac{\partial P_i}{\partial t}\right|_0=0,\quad
\left.\frac{\partial U_i}{\partial\theta}\right|_0=-(p_i+R),\quad
\left.\frac{\partial U_i}{\partial t}\right|_0=-1.
$$

Since $H=5\,\mathrm{mm}$ is fixed and nonzero, $H\theta$ in $P_i$ is a first-order term in a Taylor expansion about the actual field point. Dropping it under a jointly paraxial approximation is not consistent with retaining exact finite-field derivatives as one physical correction.

## Ideal-sphere symmetry and general system response

For a perfect spherical mirror, if the fixed external optics, apertures, and ray selection carry no orientation cues, rotating the sphere without moving its center cannot change its physical surface. An image observable that depends only on sphere center has the form $F_i(S_Y,S_Z)$. With finite-object parallax retained inside $F_i$, its first-order response is

$$
y_i=F_i(S_Y,S_Z)\approx y_{i0}+A_i\left(t+L\theta\right)
+O(t^2,t\theta,\theta^2),\qquad L=6.03\,\mathrm{mm}.
$$

The gains $A_i$ may differ across objects, but the translation and rotation columns remain proportional. This idealized center-only response has rank one: it observes $g=t+L\theta$, not $t$ and $\theta$ separately. The result need not hold exactly for an asphere, moving pupil or stop, orientation-dependent ray selection, or a complete camera/relay.

For the actual finite-field traced system, write the image-surface output for object $i$ as

$$
y_i=G_i(\theta,t;\text{fixed objects, prescription, stop/ray selection, relay, SI}).
$$

About $\theta=t=0$, with the field height and object configuration fixed, its local first-order model is

$$
y_i\approx y_{i0}+B_i\theta+A_i t,\qquad
B_i=\left.\frac{\partial G_i}{\partial\theta}\right|_0,\quad
A_i=\left.\frac{\partial G_i}{\partial t}\right|_0.
$$

The sensitivity matrix has rows $(B_i,A_i)$. Local separation requires rank two; differing absolute gains alone do not prove identifiability. Compare column direction, pairwise determinants, singular values, or condition number. These derivatives must describe the actual traced SI observable.

The signed paraxial mirror relation

$$
\frac{1}{P}+\frac{1}{Q}=-\frac{1}{f},\qquad Q=-\frac{fP}{P+f},\qquad \mu=-\frac{Q}{P}=\frac{f}{P+f},\quad M=|\mu|.
$$

Here $P>0$ denotes a real object and $P<0$ a virtual object; $Q>0$ denotes a real image and $Q<0$ a virtual image. The degenerate $P=0$ and image-at-infinity $P=-f$ cases are excluded. This centered paraxial relation is not an exact finite-field rotating-axis image solution for $G_i$.

For the paraxial corneal conjugate coordinate with eye rotation held fixed, translation gives

$$
\delta q=(1-\mu)\delta t.
$$

A virtual source at $P=-r$ has $Q=-r$, $\mu=-1$, and unit image-size magnitude $M=1$, therefore $\delta q=2\delta t$. Magnitude $M=1$ alone must not be used as signed $+1$ in the displacement formula. This is source-to-corneal-conjugate magnification, not relay or camera magnification; the independent SI gain $A_i$ above is not determined by size magnification alone. See the [first-order derivation](P1_MODEL_FIRST_ORDER.md) for the signed derivation and ideal-sphere numerical check.

## Current CODE V simulation configuration

The latest raw sweep contains $201\times401\times3=241{,}803$ rows: 201 physical translations from $-1$ to $+1\,\mathrm{mm}$, 401 RC rotations from $-20^\circ$ to $+20^\circ$, and three zoom configurations. All traces succeeded with finite SI $Y$. The recorded `theta` convention is $\theta=-\mathrm{ADE}$, with degrees converted to radians when used in the theory. The sequence was read as follows:

| Zoom | S0 THI (mm) | S1 THI (mm) | S1 ADE (deg) | S1 YDE (mm) | F1 YOB (mm) | Configuration |
|---|---:|---:|---:|---:|---:|---|
| z1 | $1\times10^{13}$ | 50 | -15 | 0 | 0 | tilted infinity source |
| z2 | 0 | 50 | 0 | +5 | 0 | finite object |
| z3 | 0 | 25 | 0 | -5 | 0 | finite object |

Here the tilt is part of the z1 source configuration; it is distinct from the swept eye rotation ADE at RC. The finite-source vertical offsets have opposite signs. This simulation therefore differs from the idealized theory inputs above, which used $p_2=100\,\mathrm{mm}$ and both finite object heights $+5\,\mathrm{mm}$.

The corrected pivot spacing remains CORNEA_ENT S2 to RC S3 $+13.8\,\mathrm{mm}$ and RC S3 to CORNEA S4 $-13.8\,\mathrm{mm}$, with the existing pickup $\mathrm{THI}(S3)=-\mathrm{THI}(S2)$. The traced cornea prescription is radius $7.7\,\mathrm{mm}$ and conic $K=0.0221$. It is not the ideal $r=7.77\,\mathrm{mm}$ sphere ($K=0$) used for the symmetry derivation.

The installed CODE V Lens System Setup Reference Manual describes the sign conventions in `help/lens_system/lensdata.03.22.html` (Local Coordinate Systems), `help/lens_system/lensdata.03.28.html` (Curvature Sign Convention), and `help/lens_system/tilts.11.08.html` (Tilt Definitions). The mirror sign convention is consistent with [OpenStax University Physics, Volume 3, Section 2.2](https://openstax.org/books/university-physics-volume-3/pages/2-2-spherical-mirrors).

## Fitted surrogate and evaluation protocol

The raw CODE V data were represented separately for each zoom by

$$
G_i(u,h)=b_i(u)+h q_i(u)+h^2r_i(u),\qquad
u=\frac{\theta}{20^\circ},\quad h=t.
$$

The baseline $b_i$ is a degree-6 polynomial fitted to the $h=0$ data. That calibration cannot identify translation response by itself: $q_i$ and $r_i$ were fit using nonzero-translation simulator data. Their selected degrees are 6 and 3 respectively. This is a simulator-trained response model, not an independent physical calibration.

The h slices were split by sorted index: even-index slices were the training pool and odd-index slices the held-out test pool. Twenty interior, nonzero even-index h slices were reserved to select q/r polynomial order; after selection the model was refitted on all even-index training slices. The odd-h test reuses the same rotation angles as calibration, so it tests unseen translations, not unseen angles. Held-out forward prediction RMSE was $0.00800221\,\mathrm{\mu m}$.

Measurement evaluation used all three zooms jointly. It sampled 2,000 held-out states uniformly, 700 additional distinct held-out states within $|\theta|\le1^\circ$, and 300 exact-$\theta=0$ stress trials (100 each at $h=-1,0,+1\,\mathrm{mm}$). The exact-zero stress h levels include training values, so these trials are stress tests, not held-out generalization. The noisy scenario added independent Gaussian noise with $\sigma=1\,\mathrm{\mu m}$ to each zoom channel; this is an assumed measurement scale, not measured detector noise. The inverse was bounded to the modeled $\theta$ and h ranges and initialized from model templates, not truth values.

| Evaluation group | Noise | Theta RMSE (deg) | h RMSE (mm) |
|---|---|---:|---:|
| Uniform held-out states | none | 0.0193923 | 0.00206123 |
| Uniform held-out states | 1 $\mathrm{\mu m}$ per zoom | 3.19636 | 0.339394 |
| Extra held-out $|\theta|\le1^\circ$ | none | 0.0469769 | 0.00500452 |
| Extra held-out $|\theta|\le1^\circ$ | 1 $\mathrm{\mu m}$ per zoom | 3.0775 | 0.327359 |
| Exact $\theta=0$ stress states | none | 0.0424576 | 0.00451322 |
| Exact $\theta=0$ stress states | 1 $\mathrm{\mu m}$ per zoom | 3.07028 | 0.326613 |

The large difference between noiseless and noisy inversion errors is consistent with weak local separation, not a contradiction. Direct finite differences in the raw sweep at $\theta=0,h=0$ gave per-zoom angular slopes $m_i=\partial y_i/\partial\theta$ of $[-0.134702,-0.125189,-0.117425]\,\mathrm{mm/deg}$ and translation secants $s_i\approx[y_i(+1)-y_i(-1)]/2$ of $[-1.264347,-1.175771,-1.102617]\,\mathrm{mm/mm}$. These are nearly collinear ($|\cos(m,s)|=0.999999966$). The angular component orthogonal to translation is only about $0.0566\,\mathrm{\mu m/deg}$; a $1\,\mathrm{\mu m}$ per-channel noise scale implies an unbounded local $\theta$ standard deviation of about $17.66^\circ$. The translation values are finite secants over the available h range, not independently measured infinitesimal derivatives.

An example near the center shows how bounded nonlinear estimation can turn this weak direction into a large error. For one noisy trial whose true values were $(\theta,h)=(0^\circ,0\,\mathrm{mm})$, the global best fit was $(-8.0711^\circ,0.8570\,\mathrm{mm})$ with $\chi^2=1.1317$, while the true parameters had $\chi^2=8.520$. Starting at truth converged to the same optimum, and the profile audit confirmed the alternative. Across the uniform noisy samples, median absolute theta error was $0.872^\circ$, p95 was $7.335^\circ$, p99 was $13.207^\circ$, and maximum was $18.760^\circ$. A local linear noise-propagation estimate is not a literal confidence interval for these bounded, nonlinear, sometimes multimodal fits.

## Interpretation and limits

The ideal perfect-sphere model predicts a rank-one center displacement. The actual simulation has different source tilt, finite-object offsets, and a mildly aspheric prescription; its observed near-collinearity is established directly in the raw sweep, so it is not merely a polynomial-fit artifact. The surrogate's low forward prediction error does not imply precise inverse recovery: along the weak direction, small image changes permit large compensating changes in $\theta$ and h.

The results apply to separate observations at individual modeled states. A prospective experiment that shares one h across many rotation measurements could pool that constraint, but such a batch estimator has not been implemented or evaluated here. Other possible ways to improve separation include an independent measurement of translation or rotation, additional optical observables, or a configuration whose zoom responses have more independent sensitivities. These remedies require their own validation.

The model is limited to the simulated prescription, three zoom settings, finite sampled range, and assumed noise. The h response is learned from the same simulator sweep; its validation is on discrete held-out h slices, not on independent hardware or intermediate calibration conditions. Real calibration errors, drift, and detector noise have not been characterized.

## Related artifacts

- Raw sweep: [`data/p1_only_sigma_depth_raytracing/p1_only_sigma_depth_raytracing.pkl`](data/p1_only_sigma_depth_raytracing/p1_only_sigma_depth_raytracing.pkl)
- Measurement evaluation script: [`Script/evaluate_p1_sigma_depth_random_measurements.py`](Script/evaluate_p1_sigma_depth_random_measurements.py)
- Evaluation report: [`data/p1_only_sigma_depth_random_measurements/p1_sigma_depth_random_measurements_report.md`](data/p1_only_sigma_depth_random_measurements/p1_sigma_depth_random_measurements_report.md)
- Estimates figure: [`data/p1_only_sigma_depth_random_measurements/p1_sigma_depth_random_estimates.svg`](data/p1_only_sigma_depth_random_measurements/p1_sigma_depth_random_estimates.svg)
- Error figure: [`data/p1_only_sigma_depth_random_measurements/p1_sigma_depth_random_errors.svg`](data/p1_only_sigma_depth_random_measurements/p1_sigma_depth_random_errors.svg)
- Binned error figure: [`data/p1_only_sigma_depth_random_measurements/p1_sigma_depth_random_binned_errors.svg`](data/p1_only_sigma_depth_random_measurements/p1_sigma_depth_random_binned_errors.svg)
