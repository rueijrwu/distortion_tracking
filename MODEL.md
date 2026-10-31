# P4 joint barrel-and-scale plus keystone model

## Final joint model

This is the simulation-fitted four-corner model relative to the P4 simulated central ray at fixed axial coordinate $Z=0$. Its reference pattern $\mathbf R_0$ is the computed real-ray (nonparaxial) pattern at zero accommodation and zero eye rotation. The model was jointly fit to simulated corners over accommodation $A\in[0,5]$ D and eye rotation $\theta\in[-20^\circ,20^\circ]$. No experimental measurements were used for validation.

For a reference corner $\mathbf u_0=(u_{0x},u_{0y})\in\{(-1,-1),(-1,1),(1,-1),(1,1)\}$, the complete four-corner forward equation is

$$
\widehat{\mathbf R}_{\theta,A}=L M_B(A)
\begin{bmatrix}
\displaystyle\frac{s_x(\theta,A)u_{0x}}{1+q_n(\theta,A)M_B(A)u_{0y}}\\[6pt]
\displaystyle\frac{s_y(\theta,A)u_{0y}}{1+q_n(\theta,A)M_B(A)u_{0y}}
\end{bmatrix},
\qquad L=1.235429592626371\,\mathrm{mm}.
$$

The accommodation-driven barrel multiplier $M_B$ acts on the reference corner before the keystone transform; hence it appears in the keystone denominator as well as in the overall radial scale. The model has no fitted center translation. The coefficient expressions below give the dominant parity-structured approximation to the jointly fitted gaze and accommodation dependence. The joint fit retains all 38 fitted coefficients; the displayed approximation omits small cross and roundoff terms and is not a separate fit. At $A=0$, $M_B=1$; at $\theta=0$, the keystone map is the identity.

## Data, coordinates, and parameterization

The simulated corner coordinates are relative to the simulated central ray and are normalized by $L$. The reference dimensionless corners are $\mathbf u_0=\mathbf R_0/L$, with $\|\mathbf u_0\|^2=2$. Define normalized inputs

$$
a=A/(5\,\mathrm D),\qquad t=\theta/(20^\circ),
$$

so $0\le a\le1$ and $-1\le t\le1$.

The relative barrel coefficient is

$$
\beta(A)=-0.0428500980586a-9.70843337206\times10^{-5}a^2,
\qquad a=A/(5\,\mathrm D),
$$

and the corner-radius barrel multiplier is

$$
M_B(A)=1+2\beta(A)=1-0.0857001961172a-0.0001941686674412a^2.
$$

Since $\beta$ acts on normalized coordinates, its equivalent coefficient on physical millimeter coordinates is $\beta/L^2$ in $\mathrm{mm}^{-2}$. At $A=5$ D, $M_B\approx0.9141056352$. The corner data constrain this net relative radial effect at $\|\mathbf u_0\|^2=2$; they do not identify an absolute physical barrel coefficient separately from isotropic scale.

The full map for a general dimensionless reference-plane coordinate $\mathbf u$ is

$$
\mathbf r=K_{s_x,s_y,q_n}\bigl(B_\beta(\mathbf u)\bigr),\qquad
B_\beta(\mathbf u)=(1+\beta\|\mathbf u\|^2)\mathbf u,
$$

where the barrel map is applied first, followed by

$$
K_{s_x,s_y,q_n}(x,y)=\left(\frac{s_xx}{1+q_ny},\frac{s_yy}{1+q_ny}\right).
$$

The homography is normalized to $H_{33}=1$. Its two scale factors are independent, and $s_xs_y$ is not constrained to one. The normalized keystone coefficient $q_n$ is dimensionless; its millimeter-coordinate equivalent is $q=q_n/L$ in $\mathrm{mm}^{-1}$.

The fitted keystone parameters have the dominant accommodation and gaze dependence

$$
\begin{aligned}
s_x&\approx1+f_{x2}(a)t^2+f_{x4}(a)t^4,\\
s_y&\approx1+f_{y2}(a)t^2+f_{y4}(a)t^4,\\
q_n&\approx f_{q1}(a)t+f_{q3}(a)t^3,
\end{aligned}
$$

with

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

In this fitted model, $B_\beta$ carries the fitted zero-gaze accommodation-dependent radial scale at the reference-corner radius. The keystone scales $s_x,s_y$ describe fitted gaze-dependent stretch with accommodation-dependent amplitudes. Their geometric mean can include isotropic gain as gaze changes, so $\beta(A)$ does not represent every isotropic scale contribution in the complete map.

## Inverse and fitting criterion

For an observed normalized corner $\mathbf r=(X_n,Y_n)$, undo keystone first:

$$
y=\frac{Y_n}{s_y-q_nY_n},\qquad x=\frac{X_ns_y}{s_x(s_y-q_nY_n)}.
$$

Then undo the radial barrel. If $r_v=\sqrt{x^2+y^2}$, solve the monotone radial equation

$$
r_v=r_u(1+\beta(A)r_u^2),\qquad \mathbf u=\frac{r_u}{r_v}(x,y),
$$

with the zero-radius case set to $\mathbf u=(0,0)$, using the physical nonnegative branch where $1+\beta(A)r_u^2>0$ and $1+3\beta(A)r_u^2>0$. The inverse order is the reverse of the forward order because keystone and barrel do not generally commute.

For $N$ sampled accommodation and gaze states, the optimized pooled vertex RMS is

$$
E=\sqrt{\frac{1}{4N}\sum_{s=1}^{N}\sum_{i=1}^{4}\left\|L\hat{\mathbf u}_{si}-\mathbf R_{0i}\right\|^2},
$$

where $\hat{\mathbf u}_{si}$ is recovered from corner $i$ at state $s$, and $\mathbf R_{0i}$ is its reference corner in millimeters. All joint coefficients are fit together by minimizing this pooled objective over 20,451 simulated states and four corners per state. The mean and worst per-state four-corner vertex RMS are $6.815869\,\mu\mathrm m$ and $21.348234\,\mu\mathrm m$, respectively. These are in-sample residuals measuring agreement with the optical simulation; they do not establish experimental hardware precision, accuracy for new eye data, or performance away from the sampled simulation states.
