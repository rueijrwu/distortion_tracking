# P4 coupled physical-barrel and two-q keystone fit

Fit uses four real-ray corners from every one of the 51 x 401 simulated accommodation/gaze states. All five noncorner fields are checks and are excluded from the objective; no train/test split is used. The existing canonical ray grid, v2 joint fit, radial fit, and separable fit artifacts are read-only.

## Integrated transformation

Let $D_k(\mathbf u)=(1+k\|\mathbf u\|^2)\mathbf u$, $k_0=k_1(0)$, $w=s(A)D_{k_0}^{-1}(\mathbf r_0)$, and $\gamma=1+k_1(A)\|w\|^2$. The complete physical forward map is implemented as the single expression

$$
F(A,E;\mathbf r_0)=\frac{\gamma\begin{bmatrix}s_x(A,E)w_x\\s_y(A,E)w_y\end{bmatrix}}{1+\gamma\left(q_{n,x}(A,E)w_x/L+q_{n,y}(A,E)w_y/L\right)}.
$$

Equivalently $F=K_{A,E}(B_A(\mathbf r_0))$, where $B_A=D_{k_1(A)}\circ S_{s(A)}\circ D_{k_0}^{-1}$ and $K$ has diagonal numerator $\operatorname{diag}(s_x,s_y)$ and both $q_xx_b+q_yy_b$ denominator terms. Physical $q_x=q_{n,x}/L$, $q_y=q_{n,y}/L$. The single simulated gaze variable E remains one axis; no separate physical $E_x,E_y$ interpretation is inferred.

Inverse correction applies $K_{A,E}^{-1}$ first, then $B_A^{-1}$, and minimizes recovered-corner coordinates against the fixed real A=0,E=0 reference. The inverse projective branch must have positive denominator; the radial inverse stays on $1+3k\rho^2>0$.

For each parameter family independently, $f(A,E)=\sum_{l=0}^2\sum_{j=1}^4c_{lj}(A/5)^l(E/20)^j$. Then $s_x=1+f_{s_x}$, $s_y=1+f_{s_y}$, and $q_{n,x}=f_{q_{n,x}}$, $q_{n,y}=f_{q_{n,y}}$. This gives 48 coefficients and exact K identity at E=0.

## Fit and validation

Converged=True (`ftol` termination condition is satisfied.), nfev=5, first-order optimality=0.000671968, objective cost=3171357.75. Active coefficient bounds (0-based) lower/upper: [] / [].
The integrated expression and staged B-then-K composition agree over every A/E state and all nine reference fields to max 8.88e-16 mm. New forward predictions round-trip through inverse K then inverse B to max 1.33e-15 mm. Minimum inverse projective denominator=0.89789556; minimum radial derivative=0.8066429.

New coupled inverse-fit corners: vertex RMS/max=8.805425/21.516309 um; coordinate RMSE=6.226376 um; mean/max state RMS=6.822130/21.348827 um.
New coupled forward corners: vertex RMS/max=8.921008/24.638373 um; coordinate RMSE=6.308105 um; mean/max state RMS=6.878640/22.232607 um.
New coupled forward five noncorner checks: vertex RMS/max=6.454286/26.451199 um; coordinate RMSE=4.563869 um; mean/max state RMS=5.063587/18.677279 um.
New coupled forward all nine: vertex RMS/max=7.649449/26.451199 um; coordinate RMSE=5.408977 um; mean/max state RMS=6.035090/19.340743 um.
New coupled inverse five noncorner checks: vertex RMS/max=6.087740/22.858174 um; coordinate RMSE=4.304682 um; mean/max state RMS=4.856179/16.369716 um.

## Same-B model comparison

Every variant below uses the identical corrected physical $B_A$, canonical ray data, and all 51 x 401 states. `old_hybrid` reuses A/E-dependent v2 K; `separable` reuses the new gaze-only 16-coefficient K; `joint` is the new 48-coefficient K(A,E). RMS/max are over actual grid vertices.

### old_hybrid

Forward corners: vertex RMS/max=8.920282/24.598265 um; coordinate RMSE=6.307592 um; mean/max state RMS=6.887281/22.211575 um.
Forward five noncorner checks: vertex RMS/max=6.466943/26.438234 um; coordinate RMSE=4.572819 um; mean/max state RMS=5.100993/18.651834 um.
Forward all nine: vertex RMS/max=7.655009/26.438234 um; coordinate RMSE=5.412909 um; mean/max state RMS=6.059823/19.309926 um.
Inverse corners: vertex RMS/max=8.806810/21.513570 um; coordinate RMSE=6.227355 um; mean/max state RMS=6.830363/21.346656 um.
Inverse five noncorner checks: vertex RMS/max=6.099822/22.845692 um; coordinate RMSE=4.313225 um; mean/max state RMS=4.891659/16.348171 um.
Inverse all nine: vertex RMS/max=7.425771/22.845692 um; coordinate RMSE=5.250813 um; mean/max state RMS=5.926519/17.774092 um.

### separable

Forward corners: vertex RMS/max=33.780195/161.745809 um; coordinate RMSE=23.886205 um; mean/max state RMS=24.691591/115.541514 um.
Forward five noncorner checks: vertex RMS/max=20.080153/151.040055 um; coordinate RMSE=14.198812 um; mean/max state RMS=14.400261/75.553051 um.
Forward all nine: vertex RMS/max=27.040030/161.745809 um; coordinate RMSE=19.120188 um; mean/max state RMS=19.690897/95.345545 um.
Inverse corners: vertex RMS/max=27.274239/119.683873 um; coordinate RMSE=19.285799 um; mean/max state RMS=20.879066/86.178702 um.
Inverse five noncorner checks: vertex RMS/max=16.818506/115.308540 um; coordinate RMSE=11.892479 um; mean/max state RMS=12.470746/61.598033 um.
Inverse all nine: vertex RMS/max=22.085307/119.683873 um; coordinate RMSE=15.616670 um; mean/max state RMS=16.788889/73.544138 um.

### joint

Forward corners: vertex RMS/max=8.921008/24.638373 um; coordinate RMSE=6.308105 um; mean/max state RMS=6.878640/22.232607 um.
Forward five noncorner checks: vertex RMS/max=6.454286/26.451199 um; coordinate RMSE=4.563869 um; mean/max state RMS=5.063587/18.677279 um.
Forward all nine: vertex RMS/max=7.649449/26.451199 um; coordinate RMSE=5.408977 um; mean/max state RMS=6.035090/19.340743 um.
Inverse corners: vertex RMS/max=8.805425/21.516309 um; coordinate RMSE=6.226376 um; mean/max state RMS=6.822130/21.348827 um.
Inverse five noncorner checks: vertex RMS/max=6.087740/22.858174 um; coordinate RMSE=4.304682 um; mean/max state RMS=4.856179/16.369716 um.
Inverse all nine: vertex RMS/max=7.419531/22.858174 um; coordinate RMSE=5.246401 um; mean/max state RMS=5.902858/17.794914 um.

## A=5 D, E=20 deg pointwise-state comparison

A=5 D,E=20 deg old_hybrid forward corners: RMS/max=19.501095/21.320110 um.
A=5 D,E=20 deg old_hybrid forward five noncorner checks: RMS/max=18.651834/26.438233 um.
A=5 D,E=20 deg old_hybrid forward all nine: RMS/max=19.033962/26.438233 um.
A=5 D,E=20 deg old_hybrid inverse corners: RMS/max=17.833528/18.174175 um.
A=5 D,E=20 deg old_hybrid inverse five noncorner checks: RMS/max=16.348171/22.845691 um.
A=5 D,E=20 deg old_hybrid inverse all nine: RMS/max=17.024337/22.845691 um.
A=5 D,E=20 deg separable forward corners: RMS/max=115.407582/161.745808 um.
A=5 D,E=20 deg separable forward five noncorner checks: RMS/max=75.553050/151.040054 um.
A=5 D,E=20 deg separable forward all nine: RMS/max=95.345545/161.745808 um.
A=5 D,E=20 deg separable inverse corners: RMS/max=86.178701/119.683873 um.
A=5 D,E=20 deg separable inverse five noncorner checks: RMS/max=61.598033/115.308540 um.
A=5 D,E=20 deg separable inverse all nine: RMS/max=73.544138/119.683873 um.
A=5 D,E=20 deg joint forward corners: RMS/max=19.514516/21.358488 um.
A=5 D,E=20 deg joint forward five noncorner checks: RMS/max=18.677279/26.451198 um.
A=5 D,E=20 deg joint forward all nine: RMS/max=19.053927/26.451198 um.
A=5 D,E=20 deg joint inverse corners: RMS/max=17.832171/18.158296 um.
A=5 D,E=20 deg joint inverse five noncorner checks: RMS/max=16.369716/22.858174 um.
A=5 D,E=20 deg joint inverse all nine: RMS/max=17.035203/22.858174 um.

## Fitted coefficient table

Rows are independent parameter families; columns are accommodation powers l=0,1,2, each containing gaze powers j=1..4:

### sx-1

| l | j=1 | j=2 | j=3 | j=4 |
|---:|---:|---:|---:|---:|
| 0 | 1.28004215339e-12 | 0.0605616233598 | -1.6581731487e-12 | 0.000882933539129 |
| 1 | -1.1422721364e-11 | 0.0318115828895 | 2.02908653695e-11 | 0.00266514916381 |
| 2 | 1.17375629789e-11 | 0.0013382097159 | -2.23258361987e-11 | 0.00199684546946 |

### sy-1

| l | j=1 | j=2 | j=3 | j=4 |
|---:|---:|---:|---:|---:|
| 0 | -7.78293415323e-13 | 0.124094806656 | 2.16066697506e-12 | -0.00458496590214 |
| 1 | -7.14631893303e-12 | 0.109027333764 | 1.56633787392e-11 | 0.00578085098323 |
| 2 | 1.01665745296e-11 | 0.00830502913418 | -2.40967816348e-11 | 0.00738405116093 |

### qnx

| l | j=1 | j=2 | j=3 | j=4 |
|---:|---:|---:|---:|---:|
| 0 | -2.2392675421e-12 | 2.63363652753e-12 | 3.22495604882e-12 | -3.20746992318e-12 |
| 1 | 1.2200121616e-11 | -1.48657075873e-11 | -1.74048065975e-11 | 1.80628005488e-11 |
| 2 | -1.15210518536e-11 | 1.39623906529e-11 | 1.63479820632e-11 | -1.68973445279e-11 |

### qny

| l | j=1 | j=2 | j=3 | j=4 |
|---:|---:|---:|---:|---:|
| 0 | -0.0630552484493 | -1.56599423321e-11 | 0.00811383030655 | 3.07468529282e-11 |
| 1 | -0.0603536897403 | -1.75012166729e-11 | 0.00579121607676 | 9.76142447734e-12 |
| 2 | -0.0137211149005 | 4.56739248612e-11 | 0.00292270946349 | -6.25637965557e-11 |

Saved physical barrel: $k_1(A)=0.000180236991874A^{1.53545703384}-0.0184792556675$ mm$^{-2}$; saved $s(0)=1$ and $s(5)=0.896199673581$.
The fitted $q_{n,x}$ surface max absolute value is 1.55952e-12; the fitted $q_{n,y}$ surface max absolute value is 0.120302.

Figures include the 3 x 3 state sweep, A-overlay by E, stages, A/E coefficient surfaces, and inverse corner residual surface. Keystone panels swap physical x/y for display only. Each grid shows its outer perimeter and center dot; middle native cross-lines and other markers are omitted.

The fit is empirical over this one-angle P4 simulation domain and does not establish independent physical rotation axes or performance beyond the sampled conditions.
