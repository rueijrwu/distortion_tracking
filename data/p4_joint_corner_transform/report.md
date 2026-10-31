# P4 joint real-ray four-corner transform

This single joint fit uses the 51 by 401 sampled states at $Z=0$ and only the four real-ray field corners. Its fixed reference is the measured corner pattern at $A=0$, $\theta=0$. Coordinates are normalized by $L=1.23542959263\,\mathrm{mm}$; no paraxial coordinates or extra magnification parameter enter the objective.

The forward map is $K\circ B_\beta$: relative barrel first, then keystone. For a real corner $\mathbf R=(X,Y)$, recovery applies the inverse keystone and then inverse barrel:

$$
\mathbf v=K^{-1}(\mathbf R),\qquad \mathbf u=B_\beta^{-1}(\mathbf v),\qquad \min \operatorname{RMS}_{corners}(\mathbf u-\mathbf u_0).
$$

The normalized keystone is $X=s_xx/(1+q_ny)$, $Y=s_yy/(1+q_ny)$ with $H_{33}=1$. The fitted scales $s_x,s_y$ are independent; no unit-determinant constraint is imposed. The inverse uses $y=Y/(s_y-q_nY)$ and $x=Xs_y/[s_x(s_y-q_nY)]$. The physical keystone coefficient is $q=q_n/L$ in $\mathrm{mm}^{-1}$.

The relative barrel is $B_\beta(\mathbf u)=(1+\beta r^2)\mathbf u$, where $\beta(A)=b_1(A/5)+b_2(A/5)^2$. Here $\beta$ is dimensionless in normalized coordinates and converts to $\mathrm{mm}^{-2}$ by dividing by $L^2$. Since all fitted corners share $r^2=2$, these data constrain the relative radial effect at that radius; they do not identify absolute physical $k_1$ separately from isotropic scale.

The joint parameter family is

$$
s_x=1+\sum_{l=0}^{2}\sum_{j=1}^{4}c^x_{lj}(A/5)^l(\theta/20)^j,\qquad
s_y=1+\sum_{l=0}^{2}\sum_{j=1}^{4}c^y_{lj}(A/5)^l(\theta/20)^j,
$$

$$
q_n=\sum_{l=0}^{2}\sum_{j=1}^{4}c^q_{lj}(A/5)^l(\theta/20)^j.
$$

The order-4 gaze terms and order-2 accommodation terms are fitted together against the recovered-corner residuals. The theta-zero keystone is the identity by construction, and $\beta(0)=0$.

## Parameter dependence

The dominant fitted dependence, with rounded coefficients for interpretation, is

$$
s_x\approx1+f_{x2}(a)t^2+f_{x4}(a)t^4,\qquad s_y\approx1+f_{y2}(a)t^2+f_{y4}(a)t^4,
$$

$$
q_n\approx f_{q1}(a)t+f_{q3}(a)t^3,\qquad a=A/5,\qquad t=\theta/20.
$$

Each function is quadratic in normalized accommodation:

$$
\begin{aligned}
f_{x2}(a)&=0.0594091+0.0342887a+0.000354141a^2,\quad&f_{x4}(a)&=0.00191096+0.000448253a+0.00287106a^2,\\
f_{y2}(a)&=0.122914+0.111543a+0.00729162a^2,\quad&f_{y4}(a)&=-0.00353496+0.00351620a+0.00833102a^2,\\
f_{q1}(a)&=-0.0630503-0.0603090a-0.0137921a^2,\quad&f_{q3}(a)&=0.00811845+0.00575111a+0.00296676a^2.
\end{aligned}
$$

These are approximate summaries of the dominant gaze terms; every coefficient in the full fitted model remains in the saved prediction. The relative-barrel accommodation polynomial is $\beta=-0.0428500981a-9.70843337\times10^{-5}a^2$. At $A=5$ D ($a=1$), its quadratic coefficient contributes about 0.23% of the linear coefficient magnitude. This coefficient comparison does not validate removing the quadratic term.

## Sampled fit summary

The fit converged: True (`ftol` termination condition is satisfied.); 4 evaluations for 38 coefficients. Mean state four-corner vertex RMS is 6.815869 $\mu\mathrm{m}$ and the worst sampled state is 21.348234 $\mu\mathrm{m}$. The minimum inverse-keystone denominator is 0.98468247; the minimum inverse-barrel radial derivative is 0.74215947. The analytic Jacobian finite-difference audit error is 3.23e-10, and the maximum forward/inverse round-trip discrepancy is 9.89e-13 $\mu\mathrm{m}$.

These are in-sample residuals on the simulated grid. They summarize this corner-only transform family and do not establish a unique physical barrel mechanism.

## Figures

`parameters_vs_gaze.png` shows $s_x,s_y,q_n$ versus gaze rotation at accommodation overlays. `parameters_vs_accommodation.png` shows the same parameters versus accommodation at selected gaze angles. `joint_parameter_heatmaps.png` shows their full joint surfaces; `barrel_parameter_vs_accommodation.png` isolates the fitted relative-barrel dependence; `joint_vertex_rms.png` shows residuals over the sampled domain.

## Full physical coefficient tables

The complete fitted coefficients on physical monomials $A^l\theta^j$ appear below. Scale coefficients have units $\mathrm{D}^{-l}\mathrm{degree}^{-j}$; $q=q_n/L$ coefficients have units $\mathrm{mm}^{-1}\mathrm{D}^{-l}\mathrm{degree}^{-j}$. The full scaled-basis coefficient vector is stored in the pickle.

### sx

Physical coefficient units: $\mathrm{D}^{-l}\mathrm{degree}^{-j}$.

| $A$ power $l$ | $\theta$ power $j$ | Physical coefficient |
|---:|---:|---:|
| 0 | 1 | -0.002551757158 |
| 0 | 2 | -2.844045449e-14 |
| 0 | 3 | 8.214197887e-07 |
| 0 | 4 | 1.388273217e-16 |
| 1 | 1 | -0.0004881623659 |
| 1 | 2 | -1.13093071e-14 |
| 1 | 3 | 1.163787761e-07 |
| 1 | 4 | 3.148723364e-17 |
| 2 | 1 | -2.232757558e-05 |
| 2 | 2 | 4.568627778e-15 |
| 2 | 3 | 1.20070006e-08 |
| 2 | 4 | -1.709494722e-17 |

### sy

Physical coefficient units: $\mathrm{D}^{-l}\mathrm{degree}^{-j}$.

| $A$ power $l$ | $\theta$ power $j$ | Physical coefficient |
|---:|---:|---:|
| 0 | 1 | -0.002551757158 |
| 0 | 2 | -2.844045449e-14 |
| 0 | 3 | 8.214197887e-07 |
| 0 | 4 | 1.388273217e-16 |
| 1 | 1 | -0.0004881623659 |
| 1 | 2 | -1.13093071e-14 |
| 1 | 3 | 1.163787761e-07 |
| 1 | 4 | 3.148723364e-17 |
| 2 | 1 | -2.232757558e-05 |
| 2 | 2 | 4.568627778e-15 |
| 2 | 3 | 1.20070006e-08 |
| 2 | 4 | -1.709494722e-17 |

### q=q_n/L

Physical coefficient units: $\mathrm{mm}^{-1}\mathrm{D}^{-l}\mathrm{degree}^{-j}$.

| $A$ power $l$ | $\theta$ power $j$ | Physical coefficient |
|---:|---:|---:|
| 0 | 1 | -0.002551757158 |
| 0 | 2 | -2.844045449e-14 |
| 0 | 3 | 8.214197887e-07 |
| 0 | 4 | 1.388273217e-16 |
| 1 | 1 | -0.0004881623659 |
| 1 | 2 | -1.13093071e-14 |
| 1 | 3 | 1.163787761e-07 |
| 1 | 4 | 3.148723364e-17 |
| 2 | 1 | -2.232757558e-05 |
| 2 | 2 | 4.568627778e-15 |
| 2 | 3 | 1.20070006e-08 |
| 2 | 4 | -1.709494722e-17 |

### Relative-barrel accommodation polynomial

| $A$ power | Scaled coefficient | Physical coefficient ($\mathrm{mm}^{-2}\,\mathrm{D}^{-l}$) |
|---:|---:|---:|
| 1 | -0.0428500980586 | -0.005614949 |
| 2 | -9.70843337206e-05 | -2.544328285e-06 |