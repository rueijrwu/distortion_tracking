# First-order global-frame model for a convex corneal mirror

This is a paraxial model of the cornea's reflected conjugate image (real or virtual) in a fixed global frame. It is an analytic reference, not a model of the full relay to the CODE V image surface `SI`.

## Coordinates and notation

Use global transverse coordinate $y$ (positive upward) and axial coordinate $z$ (positive toward the objects). The eye's rotation center is the origin. The nominal corneal vertex is at $z=R$ and $y=0$. Let $t_j$ be the head translation along global $+y$, and let $e_j$ be eye rotation in radians, positive when the local optical axis tilts toward $+y$. To first order, the outward optical-axis unit vector is

$$
\mathbf n_j\simeq(e_j,1).
$$

The translated and rotated vertex has lateral coordinate

$$
m_j=t_j+R e_j.
$$

Here $m_j$ is a lateral vertex coordinate; it is not the object's local transverse height. The cornea is modeled as a convex spherical mirror with radius magnitude $r$ and positive focal-length magnitude $f=r/2$. In the signed Gaussian mirror relation its focal length is $F=-f$; the positive $f$ below denotes only the magnitude.

For finite object $i$, denote its fixed global lateral coordinate by $H_i$. In the user's earlier notation, this lateral position was called $p_{i,0}$; it is not an axial object distance. Let $D_i$ be its signed axial distance from the nominal vertex, positive for a real source in front and negative for a virtual source behind it. Thus the object is at $(H_i,R+D_i)$. Let signed image distance $Q_i$ be positive for a real image in front and negative for a virtual image behind it.

$$
\frac{1}{D_i}+\frac{1}{Q_i}=-\frac{1}{f},\qquad
\mu_i=-\frac{Q_i}{D_i}=\frac{f}{D_i+f},\qquad
M_i=|\mu_i|,\quad \sigma_i=\operatorname{sign}(\mu_i),\quad \mu_i=\sigma_iM_i.
$$

The degenerate $D_i=0$ case and the $D_i=-f$ image-at-infinity case are excluded. A finite-curvature sphere has no nondegenerate finite-distance conjugate with $\mu_i=+1$; that is only the excluded formal $D_i\to0$ limit. Positive magnitude $M_i$ alone does not determine orientation. Some virtual sources with $-f<D_i<0$ have $\mu_i>1$ and form real reflected images. In the original real-source regime $D_i>0$, the earlier $M$ denoted the positive signed magnification $\mu$; generalized formulas use $\mu$ for signed magnification and reserve $M$ for magnitude.

For an infinity object, use the global direction from the cornea toward the source, with signed angle $\alpha_{\infty}$ from global $+z$ toward $+y$. Incident light propagates in the opposite direction. An on-axis infinity source has $\alpha_{\infty}=0$.

| Symbol | Meaning |
|---|---|
| $R$ | rotation-center to nominal-vertex distance |
| $r$ | positive magnitude of spherical corneal radius |
| $f=r/2$ | positive magnitude of focal length; signed convex mirror focal length is $-f$ |
| $t_j$ | physical head translation along global $+y$ |
| $e_j$ | eye tilt in radians |
| $m_j=t_j+Re_j$ | global lateral position of the corneal vertex |
| $H_i$ | fixed global lateral position of finite object $i$ |
| $D_i$ | signed axial object distance from nominal vertex (positive real, negative virtual) |
| $Q_i$ | signed axial image distance (positive real, negative virtual) |
| $\mu_i,M_i,\sigma_i$ | signed transverse magnification, magnitude, and sign |

## Finite-object derivation

In the local corneal frame, the transverse object height relative to the tilted vertex is, to first order,

$$
U_{ij}=H_i-m_j-D_i e_j.
$$

The term $-D_i e_j$ is the transverse projection of the fixed axial object offset into the tilted local frame. The local transverse image coordinate is

$$
v_{ij}=\mu_iU_{ij}=\mu_i(H_i-m_j-D_i e_j).
$$

This $v_{ij}$ is a local transverse coordinate only. Its global lateral coordinate is

$$
q_{ij}=m_j+v_{ij}+Q_i e_j.
$$

Substitution gives

$$
\begin{aligned}
q_{ij}
&=m_j+\mu_i(H_i-m_j-D_i e_j)+Q_i e_j\\
&=\mu_iH_i+(1-\mu_i)m_j+(Q_i-\mu_iD_i)e_j\\
&=\mu_iH_i+(1-\mu_i)m_j+2Q_i e_j.
\end{aligned}
$$

Since $2Q_i=r(\mu_i-1)$ and $m_j=t_j+Re_j$,

$$
\boxed{q_{ij}=\mu_iH_i+(1-\mu_i)\left[t_j+(R-r)e_j\right].}
$$

The object contributes baseline $\mu_iH_i$. Translation and rotation enter through $t_j+(R-r)e_j$, multiplied by gain $1-\mu_i$. Different gains do not make translation and rotation independently identifiable in this ideal model.

## Virtual source with unit image-size magnification

Pure translation gives

$$
\delta q=(1-\mu)\delta t=(1-\sigma M)\delta t.
$$

The real-source upright branch has gain $1-M$; the inverted branch has gain $1+M$. For $D=Q=-r=-2f$, $\mu=-1$, $M=1$, and $\sigma=-1$, so $\delta q=2\delta t$. The original formula using positive magnitude $M$ assumed upright orientation; substituting $M=1$ into the generalized signed factor $1-\mu$ as though $\mu=+1$ would give an incorrect cancellation.

An independent ideal-sphere reflection check used $r=7.77\,\mathrm{mm}$ and a source at $(0,-r)$ in fixed laboratory coordinates whose origin is the nominal vertex, not the moving vertex. Sampling ray heights $\pm0.01\,\mathrm{mm}$ relative to the translated vertex, a $1\,\mu\mathrm m$ head translation moved the reflected-line backward intersection by about $2\,\mu\mathrm m$. This is not CODE V or full-relay validation.

## Infinity-object limit and sign check

For an infinity source, the source direction relative to the tilted local optical axis is $\alpha_{\infty}-e_j$. With the sign convention above, the local transverse focal-plane image displacement is

$$
v_{\infty j}=f(\alpha_{\infty}-e_j).
$$

Thus an on-axis source has $v_{\infty j}=-fe_j$, not $+fe_j$. Including the virtual image's axial depth $f$ behind the vertex gives the global lateral coordinate

$$
q_{\infty j}=m_j+v_{\infty j}-fe_j
=f\alpha_{\infty}+t_j+(R-r)e_j.
$$

Relative to the moving vertex, the global lateral displacement is $v_{\infty j}-fe_j=f\alpha_{\infty}-2fe_j$. For the on-axis case it is $-2fe_j$. Adding the vertex coordinate $m_j$ yields the absolute global image coordinate above. The finite-object expression has this infinity limit when $D_i\to\infty$ and $H_i/D_i\to\alpha_{\infty}$: then $\mu_i\to0$, $Q_i\to-f$, and $\mu_iH_i\to f\alpha_{\infty}$. If $H_i$ is fixed as $D_i\to\infty$, the limit is instead the on-axis case $\alpha_{\infty}=0$.

## Calibration across known gaze angles

Consider a calibration in which the true eye angles $e_k$ are known and, at each angle, repeated measurements have head translations distributed as

$$
t_{kn}\mid e_k\sim\mathcal N(0,\sigma_{t,\mathrm{cal}}^2),\qquad
\mathbb E[t_{kn}\mid e_k]=0.
$$

The zero-mean condition is conditional on each known angle; individual calibration observations need not have zero translation. Let $i$ index the fixed-angle infinity source or either finite object, and define

$$
k_{\infty}=1,\qquad k_i=1-\mu_i\quad\text{for a finite object},
$$

$$
b_{\infty}=f\alpha_{\infty},\qquad b_i=\mu_iH_i\quad\text{for a finite object}.
$$

Under the ideal model, an observation at known angle $e_k$ is

$$
q_{i,kn}=b_i+k_i\left(ae_k+t_{kn}\right)+\epsilon_{i,kn},
\qquad \mathbb E[\epsilon_{i,kn}\mid e_k]=0.
$$

Therefore the conditional calibration curve is

$$
C_i(e_k)=\mathbb E[q_{i,kn}\mid e_k]=b_i+s_i e_k,
\qquad s_i=k_i a.
$$

With known $a=R-r\ne0$, calibration over multiple distinct known gaze angles estimates the slope $s_i$ and hence the ideal translation gain $k_i=s_i/a$, as well as the intercept $b_i$; a single zero-angle calibration point cannot estimate a slope. Repeated measurements reduce the effect of random calibration head motion on those estimates. The zero-mean calibration assumption is a model for that protocol; it does not imply that unknown deployment translations are zero-mean or automatically average away.

## Identifiability and scope

For all finite objects and the infinity object, the first-order lateral image shifts depend on the same combination

$$
a=R-r,\qquad g_j=t_j+a e_j.
$$

If $\mu_i,H_i,R,r$ are calibrated, subtract each known nominal image baseline. For a fixed-angle infinity source and two finite objects, the resulting three equations are

$$
\begin{aligned}
\Delta q_{\infty j}&=q_{\infty j}-q_{\infty,0}=t_j+a e_j,\\
\Delta q_{2j}&=q_{2j}-q_{2,0}=(1-\mu_2)(t_j+a e_j),\\
\Delta q_{3j}&=q_{3j}-q_{3,0}=(1-\mu_3)(t_j+a e_j),
\end{aligned}
\qquad
\begin{aligned}
q_{\infty,0}&=f\alpha_\infty,\\
q_{2,0}&=\mu_2H_2,\\
q_{3,0}&=\mu_3H_3.
\end{aligned}
$$

Their sensitivity matrix is

$$
\begin{bmatrix}
1 & a\\
1-\mu_2 & a(1-\mu_2)\\
1-\mu_3 & a(1-\mu_3)
\end{bmatrix}.
$$

The second column is $a$ times the first, so this matrix has rank one, including for virtual sources. The calibrated angle-response curves have slopes $s_i=k_i a$ and translation gains $k_i$; dividing each baseline-subtracted deployment measurement by its calibrated slope gives

$$
\frac{q_{ij}-b_i}{s_i}=e_j+\frac{t_j}{a}+\frac{\epsilon_{ij}}{s_i}.
$$

All three calibrated angle estimates share the same unknown head-motion term $t_j/a$. Known-angle calibration curves can estimate intercepts and slopes correctly, but do not create an independent instantaneous constraint on deployment $t_j$ and $e_j$. For any $\delta$, the pair $e'_j=e_j+\delta$, $t'_j=t_j-a\delta$ produces the same three responses. If $t_j$ is independently known, then $e_j=(g_j-t_j)/a$; if $e_j$ is known, then $t_j=g_j-ae_j$. Otherwise an additional measurement whose translation-to-rotation sensitivity ratio differs is needed to separate them. The on-axis special case has $\alpha_\infty=0$. This is a geometric result for the reflected conjugate image of the ideal sphere under the stated paraxial approximations.

The approximation assumes small $e_j$, small transverse field angles (including $|H_i/D_i|\ll1$), fixed object coordinates, and paraxial conjugates. It neglects aberrations, asphericity, pupil or stop motion, ray-selection changes, and relay optics. In particular, a finite source offset or substantial source tilt can fall outside the small-field approximation. The conjugate-image coordinate $q_{ij}$ is not generally the lateral coordinate at a downstream `SI` surface; a relay model is needed to map between them.

The signed mirror relation is consistent with [OpenStax University Physics, Volume 3, Section 2.2](https://openstax.org/books/university-physics-volume-3/pages/2-2-spherical-mirrors) and its [key equations](https://openstax.org/books/university-physics-volume-3/pages/2-key-equations). See [P1_MODEL.md](P1_MODEL.md) for the broader comparison with the current simulated prescription and inverse-estimation results.
