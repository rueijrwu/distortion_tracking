# Distortion model results

## Scope and conventions

The current P4 calibration covers accommodation $A=0,0.1,\ldots,5\,\mathrm{D}$, eye rotation $\theta=-20,-19.9,\ldots,20^\circ$, and $Z=0$. Each accommodation/rotation state has nine normalized field locations on a $3\times3$ grid. Paraxial and real image coordinates are in millimeters; reported residuals are in micrometers. The results below describe in-sample simulated coordinates over this sampled domain. The $1\,\mu\mathrm{m}$ value is a reference line, not a precision or accuracy claim.

For a state with nine field points, coordinate RMSE is

$$
\operatorname{RMSE}=\sqrt{\frac{1}{18}\sum_{i=1}^{9}\left[(\Delta x_i)^2+(\Delta y_i)^2\right]},
$$

and maximum point error is the largest of the nine Euclidean errors $\sqrt{(\Delta x_i)^2+(\Delta y_i)^2}$. “Mean RMSE” below is the arithmetic mean of state RMSE values, not a pooled residual RMSE.

## Zero-rotation barrel baseline

At each accommodation, let $\mathbf u_i(A)=(u_{ix},u_{iy})$ be the actual paraxial reference coordinate for field $i$, and let $r_i^2=\mathbf u_i\cdot\mathbf u_i$. The signed radial model is

$$
\widehat{\mathbf b}_i(A)=\left[1+k_1(A)r_i^2\right]\mathbf u_i(A),
$$

where $k_1$ has units $\mathrm{mm}^{-2}$. Negative $k_1$ gives barrel distortion. The paraxial magnification is approximately isotropic; this is separate from the radial real-grid fit residuals reported below.

An in-sample quadratic accommodation description is

$$
k_1(A)=p_0+p_1A+p_2A^2,
$$

with $(p_0,p_1,p_2)=(-0.0185179003996,\,0.000180197179361,\,5.15785422657\times10^{-5})$; coefficient units are respectively $\mathrm{mm}^{-2}$, $\mathrm{mm}^{-2}\mathrm{D}^{-1}$, and $\mathrm{mm}^{-2}\mathrm{D}^{-2}$. A fitted signed offset exponential is also an empirical alternative:

$$
k_1(A)=d+a e^{bA},\qquad(d,a,b)=(-0.0195347321301,\,0.000993232198585,\,0.235926475525\,\mathrm{D}^{-1}),
$$

where $d,a$ are in $\mathrm{mm}^{-2}$. Since $k_1$ is negative, no logarithm of $k_1$ is used.

The coefficient becomes less negative from $-0.0185283312073\,\mathrm{mm}^{-2}$ at $0\,\mathrm{D}$ to $-0.0163117998622\,\mathrm{mm}^{-2}$ at $5\,\mathrm{D}$. Both accommodation terms matter: at $5\,\mathrm{D}$, the linear and quadratic contributions are about $0.000901$ and $0.001289\,\mathrm{mm}^{-2}$. The paraxial reference also changes with accommodation: its measured isotropic scale relative to $0\,\mathrm{D}$ is $0.896199673581$ at $5\,\mathrm{D}$. This is distinct from the radial coefficient; no continuous law for the paraxial scale has been fitted.

Direct per-accommodation barrel fits have mean state RMSE $0.807812\,\mu\mathrm{m}$, maximum state RMSE $1.101915\,\mu\mathrm{m}$, and maximum point error $2.203830\,\mu\mathrm{m}$. Replacing direct $k_1$ values with the quadratic curve adds at most $0.0366545\,\mu\mathrm{m}$ coordinate RMSE; the exponential adds at most $0.0428462\,\mu\mathrm{m}$. These are sampled-field, in-sample results.

## Rotation-dependent keystone

For each accommodation, the input is its measured real grid at $\theta=0$, denoted $\mathbf b_i(A)=(x_{0i},y_{0i})$. It already contains the zero-angle barrel distortion, so applying the radial model again would double-count it. The fitted transformation is

$$
X=\frac{s_x x_0}{1+qy_0},\qquad
Y=\frac{s_y y_0}{1+qy_0},\qquad
H=\begin{bmatrix}
s_x&0&0\\
0&s_y&0\\
0&q&1
\end{bmatrix}.
$$

Here $s_x,s_y$ are dimensionless and $q$ has units $\mathrm{mm}^{-1}$. At each accommodation, each of the three directly fitted parameters is then represented by an unconstrained quadratic in rotation. In compact form,

$$
\begin{bmatrix}s_x(\theta,A)&s_y(\theta,A)&q(\theta,A)\end{bmatrix}
=
\begin{bmatrix}1&\theta&\theta^2\end{bmatrix}C(A),
\qquad
C(A)=
\begin{bmatrix}
c_{0x}&c_{0y}&c_{0q}\\
c_{1x}&c_{1y}&c_{1q}\\
c_{2x}&c_{2y}&c_{2q}
\end{bmatrix}.
$$

Rows of $C(A)$ correspond to basis terms $1,\theta,\theta^2$ and columns to parameters $s_x,s_y,q$. Each accommodation has its own fitted matrix; no accommodation polynomial for its entries has been fit. The dominant measured terms in physical-degree units are:

| $A$ (D) | $s_x$ coefficient of $\theta^2$ ($\mathrm{degree}^{-2}$) | $s_y$ coefficient of $\theta^2$ ($\mathrm{degree}^{-2}$) | $q$ coefficient of $\theta$ ($\mathrm{mm}^{-1}\,\mathrm{degree}^{-1}$) |
|---:|---:|---:|---:|
| 0 | 0.0001552758954 | 0.0003028028193 | -0.002345622877 |
| 1 | 0.0001732684773 | 0.0003630345936 | -0.002831577821 |
| 2 | 0.0001918100541 | 0.0004266453010 | -0.003355193078 |
| 3 | 0.0002108883579 | 0.0004936398101 | -0.003918254684 |
| 4 | 0.0002305306594 | 0.0005642392799 | -0.004524126537 |
| 5 | 0.0002507548114 | 0.0006386357893 | -0.005175941610 |

A reduced interpretation is

$$
s_x\approx c_x(A)+\alpha(A)\theta^2,\qquad
s_y\approx c_y(A)+\beta(A)\theta^2,\qquad
q\approx\gamma(A)\theta.
$$

The scale intercepts $c_x,c_y$ are near one, but are not fixed to one. Direct fits at zero rotation are numerically near $(1,1,0)$; the subsequent quadratics do not enforce that identity or even/odd symmetry. Omitted scale-linear and $q$-quadratic terms are numerically negligible in these fits. These observed dominant orders do not establish that higher-order terms are unnecessary. The rotation and keystone coefficients vary materially with accommodation.

| Fit summary over accommodation and rotation states | Mean | Maximum |
|---|---:|---:|
| Direct-transform coordinate RMSE ($\mu\mathrm{m}$) | 4.06167 | 12.8112 |
| Quadratic-parameter total coordinate RMSE ($\mu\mathrm{m}$) | 4.46151 | 14.5312 |
| Added error from parameter quadratics, coordinate RMSE ($\mu\mathrm{m}$) | 1.51779 | 7.25432 |
| Direct-transform maximum point error ($\mu\mathrm{m}$) | 7.60316 | 25.104 |
| Quadratic-parameter total maximum point error ($\mu\mathrm{m}$) | 8.38888 | 25.7348 |
| Added error from parameter quadratics, maximum point error ($\mu\mathrm{m}$) | 3.06876 | 14.5049 |

The barrel and keystone errors cover different scopes: barrel is evaluated at zero rotation against paraxial coordinates, while keystone covers the rotation range relative to each accommodation's measured zero-angle real grid. They are not a controlled head-to-head comparison. Reusing the $0\,\mathrm{D}$ keystone parameters at other accommodations produces a maximum pointwise prediction difference of $294.802\,\mu\mathrm{m}$ relative to the accommodation-specific direct fit; the resulting maximum point error against measurements is $296.617\,\mu\mathrm{m}$. Accommodation dependence is essential.

The largest direct keystone residual occurs at $A=5\,\mathrm{D}$, $\theta=-20^\circ$, field $(1,0)$, with predicted-minus-measured residual about $(+7.884,+23.834)\,\mu\mathrm{m}$. There $y_0$ is near zero, so this projective form predicts $Y$ near zero despite the measured vertical shift. A missing spatial component is implicated; raising only the rotation-polynomial order cannot represent that residual.

The measured absolute image center is a separate observable. Its same-accommodation change from zero rotation reaches $5.52214\,\mathrm{mm}$ in one component. No continuous center law is established.

## Candidate composition and limits

A possible absolute-coordinate model is

$$
\widehat{\mathbf p}_{\mathrm{abs},i}(\theta,A)
=
\mathbf c_{\mathrm{abs}}(\theta,A)
+
K_{\theta,A}\!\left(\widehat{\mathbf b}_i(A)\right),
$$

where $K$ is the keystone transform above. This composition has not been validated end to end: reported keystone errors use the measured baseline $\mathbf b_i(A)$, not the fitted $\widehat{\mathbf b}_i(A)$, and the center term has no fitted law. A possible continuous accommodation description of the keystone coefficient matrix is $C(A)=C_0+C_1A$, with a quadratic term considered only if composed Cartesian errors warrant it; this is a proposal, not a fitted result. The accommodation dependence of paraxial scale likewise remains without a fitted law.

A separate limited $Z$ study found an approximately linear scale $m_Z\approx1+0.00295Z$, where $Z$ is in millimeters and the slope has units $\mathrm{mm}^{-1}$. It sampled only $A=0,1,2,3,4\,\mathrm{D}$ and five rotations from $-10^\circ$ to $+10^\circ$. Its worst linear-model state had coordinate RMSE $0.249\,\mu\mathrm{m}$ and maximum point error $0.511\,\mu\mathrm{m}$. It does not establish a joint accommodation/rotation/$Z$ model over the full P4 rotation domain or whether the scale transformations commute.

Current P1 fit results are not established for the current P1 data. An eventual eye-rotation inference using the center and four corners, with $1\,\mu\mathrm{m}$ Gaussian measurement noise, has not been evaluated; the residual metrics above use all nine field points. The remaining scientific questions are how to represent the observed spatial residuals, how many rotation terms are useful, what accommodation dependence is needed for keystone coefficients and paraxial scale, and how the barrel, keystone, and center components perform together in Cartesian coordinates.
