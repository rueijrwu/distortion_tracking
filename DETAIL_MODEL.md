# P1/P4 hybrid calibration model: detailed definition (87 values)

This file defines the implemented 87-value hybrid model: every parameter, every formula, and the physical or interpolative role of each term. It describes the model only. Results, performance, and provenance are in [Record.md](Record.md). The canonical model text is in [Theory.md](Theory.md#current-hybrid-model-and-geometric-interpretation-2026-11-08); this file expands it.

The model is **not** a complete physical forward or inverse lens model. It is a P4 relative-field family plus independently fitted empirical centroid and gaze maps for P1 and P4, and a separate P4 centre-motion normalization.

---

## 0. Notation and conventions

| Symbol | Meaning | Units |
|---|---|---|
| $j \in \{1,4\}$ | channel: P1 or P4 | — |
| $g = (\mathrm{ADE}, \mathrm{BDE})$ | eye gaze: ADE (vertical), BDE (horizontal) | degrees |
| $A$ | accommodation | D |
| $a = A/4$ | normalized accommodation, $0 \le a \le 1$ | — |
| $x = \mathrm{BDE}/12.4$, $y = \mathrm{ADE}/12.4$ | normalized gaze, used by the centroid-bias and gaze maps | — |
| $t_x = \mathrm{BDE}/20$, $t_y = \mathrm{ADE}/20$ | normalized gaze, used by the P4 field family | — |
| $k = 1,\dots,9$ | index of the nine sampled rays (field points) | — |
| $c_j$ | labeled central-ray coordinate of channel $j$ | mm |
| $m_j$ | arithmetic mean of the nine measured ray coordinates | mm |
| $L = 1.235429592627$ | reference scale of the P4 field family | mm |
| $R_{0,k}$ | measured real-ray A=0, zero-gaze reference position of ray $k$ (centre-relative) | mm |

Gaze normalizers 12.4° and 20° are scale constants, not fitted. The ±12.4° range is the support of the ray data. The runtime allows ±12.5°.

---

## 1. Parameter inventory (87 fitted values)

| Block | Channel | Count | Symbols | Fitted? |
|---|---|---:|---|---|
| B (centroid bias) | P1 | 9 | $\beta_{1,1..4}$, $\gamma_{1,1..5}$ | yes |
| B (centroid bias) | P4 | 9 | $\beta_{4,1..4}$, $\gamma_{4,1..5}$ | yes |
| G (inverse gaze map) | P1 | 20 | $C_1$ (10 × 2) | yes |
| G (inverse gaze map) | P4 | 20 | $C_4$ (10 × 2) | yes |
| PF12 field base (A=0 shape) | P4 | 12 | frozen base of the 12 spatial pairs in Table 2 | **frozen** from an earlier fit |
| Field interaction | P4 | 12 | $\lambda_p$, one per spatial pair $p$ | yes |
| Field accommodation | P4 | 3 | $s_1$, $k_1$, $k_2$ | yes |
| M1 centre scale | P4 | 2 | $k_x$, $k_y$ | yes |
| **Total** | | **87** | 58 B/G + 12 PF12 + 15 new field + 2 M1 | |

Of the 87 values, 17 were newly estimated in the P4 extension (15 field + 2 M1). The 58 B/G values and 12 PF12 values come from the earlier A=0 calibration.

**Constants that are measured or chosen, not fitted:**

| Constant | Value | Role |
|---|---|---|
| $b_{1,0}$ | $(0,\ -0.05801502493)$ mm | P1 zero-gaze anchor for B1 |
| $b_{4,0}$ | $(0,0)$ mm | P4 zero-gaze anchor for B4 |
| $\mu_j,\ \sigma_j$ | per-channel training mean and scale of $c_j$ | normalizers for G |
| $o_4$ | $(0,0)$ mm | on-axis P4 pivot for M1 |
| $\omega_{\rm center}$ | 0 | anchor weight, centre ray |
| $\omega_{\rm outer}$ | 9/8 | anchor weight, each of eight outer rays |
| $12.4^\circ,\ 20^\circ$ | — | gaze normalizers |
| $L$ | 1.235429592627 mm | field scale |
| $[0,4]$ D | — | accommodation search bounds |
| $\pm 12.5^\circ$ | — | gaze clip in the solver |

---

## 2. Channel B: centroid bias (9 fitted values per channel)

**Why it exists.** The nine measured rays have an arithmetic mean $m_j$, but the optical reference is the labeled central ray $c_j$. The two are different points. The bias $B_j(g)$ maps the mean to the centre: $c_j = m_j - B_j(g)$.

**Formula.** With $x = \mathrm{BDE}/12.4$ and $y = \mathrm{ADE}/12.4$:

$$
\begin{aligned}
B_{j,x}(g) &= b_{j0,x} + \beta_{j1}\,x + \beta_{j2}\,xy + \beta_{j3}\,xy^2 + \beta_{j4}\,x^3,\\[2pt]
B_{j,y}(g) &= b_{j0,y} + \gamma_{j1}\,y + \gamma_{j2}\,y^2 + \gamma_{j3}\,x^2 + \gamma_{j4}\,y^3 + \gamma_{j5}\,x^2y.
\end{aligned}
$$

**Terms.**
- $b_{j0}$: measured zero-gaze anchor (not fitted). It sets the offset of the mean from the centre at zero gaze.
- $\beta_{j1..4}$, $\gamma_{j1..5}$: gaze-dependent offset. Each term is a low-order polynomial in gaze.
- Parity: the form is odd in $x$ for the $\beta$ terms and odd in $y$ for the $\gamma$ terms. A reflection check on BDE passed. No ADE parity was imposed.

**Interpretation.** Interpolation. The bias is an empirical correction for the arrangement of the nine-ray grid. It is not a lens law.

---

## 3. Channel G: inverse gaze map (20 fitted values per channel)

**Why it exists.** The gaze is read from the corrected centre coordinate. The map is fitted separately for each channel.

**Formula.** Let $z_j = (c_j - \mu_j)/\sigma_j$ with $z_j = (z_{j,x}, z_{j,y})$. Define the ten monomials

$$
\phi(z) = \bigl(1,\ z_x,\ z_y,\ z_x^2,\ z_xz_y,\ z_y^2,\ z_x^3,\ z_x^2z_y,\ z_xz_y^2,\ z_y^3\bigr),
$$

and the map

$$
G_j(c_j) = \phi(z_j)\, C_j, \qquad C_j \in \mathbb{R}^{10\times 2}.
$$

The output is the estimated gaze $(\mathrm{ADE},\mathrm{BDE})$ in degrees. $C_1$ and $C_4$ have 20 scalars each.

**Terms.** Each column of $C_j$ is a cubic polynomial in the normalized centre. There is no accommodation dependence. $G_1$ and $G_4$ are empirical inverse maps, not lens-distortion equations.

**Interpretation.** Interpolation. Gaze is read from centre position by a fitted cubic. The 25 calibration poses support a 10-monomial design. A 3 × 3 pose stencil (nine poses) cannot identify all ten monomials.

---

## 4. Channel P4 field family: 27 values (12 frozen + 15 fitted)

The P4 field family describes how the nine-ray pattern changes with gaze and accommodation. It is built from the measured reference $R_{0,k}$.

### 4.1 Spatial structure

Gaze enters through the 12 spatial pairs in Table 2. Each pair $p$ has one coefficient function, with a frozen A=0 base and one fitted accommodation interaction:

$$
\text{coef}_p(t, a) = c^{\,0}_p + a\,\lambda_p, \qquad p = 1,\dots,12.
$$

**Table 2: spatial pairs (12).** $c^0_p$ are the frozen PF12 base values; $\lambda_p$ are the fitted interactions.

| # | Coefficient | Monomial in $t$ | Role |
|---:|---|---|---|
| 1 | $m_{11}$ | $t_x^2$ | affine stretch, horizontal |
| 2 | $m_{11}$ | $t_y^2$ | affine stretch, horizontal |
| 3 | $m_{22}$ | $t_x^2$ | affine stretch, vertical |
| 4 | $m_{22}$ | $t_y^2$ | affine stretch, vertical |
| 5 | $m_{12}$ | $t_xt_y$ | shear |
| 6 | $m_{21}$ | $t_xt_y$ | shear |
| 7 | $q_x$ | $t_x$ | projective denominator, horizontal |
| 8 | $q_y$ | $t_y$ | projective denominator, vertical |
| 9 | $p_1$ | $t_y$ | Brown tangential term |
| 10 | $p_2$ | $t_x$ | Brown tangential term |
| 11 | $g_{\rm rad}$ | $t_x^2$ | radial gain |
| 12 | $g_{\rm rad}$ | $t_y^2$ | radial gain |

The base value of $m_{11}$ and $m_{22}$ is 1 at zero gaze and accommodation. The base values of $m_{12}, m_{21}, q_x, q_y, p_1, p_2, g_{\rm rad}$ are 0 at zero gaze. The parity structure (even/odd in $t_x$, $t_y$) is inherited from the frozen PF12 base.

**Coefficient conversion.** The PF12 base was originally expressed in powers of $\mathrm{gaze}/12.4$. The field family uses $\mathrm{gaze}/20$. A base coefficient for monomial $t_x^i t_y^j$ is converted by $c_{\rm new} = c_{\rm old}\,(20/12.4)^{i+j}$. Omitting this conversion breaks the exact A=0 reduction.

### 4.2 Accommodation scale and radial strength

$$
s(a) = 1 + s_1\,a, \qquad \kappa(a,g) = g_{\rm rad}(t) + k_1\,a + k_2\,a^2.
$$

$s_1, k_1, k_2$ are fitted and vanish at $A=0$ only in the sense that their contributions are zero at $a=0$.

### 4.3 The field map

For each reference ray $k$ (with $R_{0,k}$ measured and $k=$ centre excluded from the warp):

$$
\begin{aligned}
v_k &= \frac{s(a)\,R_{0,k}}{L}, &
D_k &= 1 + q_x\,v_{k,x} + q_y\,v_{k,y},\\[3pt]
u_{k,x} &= \frac{m_{11}v_{k,x} + m_{12}v_{k,y}}{D_k}, &
u_{k,y} &= \frac{m_{21}v_{k,x} + m_{22}v_{k,y}}{D_k},\\[3pt]
w_{k,x} &= u_{k,x} + 2p_1u_{k,x}u_{k,y} + p_2\bigl(3u_{k,x}^2 + u_{k,y}^2\bigr),\\[3pt]
w_{k,y} &= u_{k,y} + p_1\bigl(u_{k,x}^2 + 3u_{k,y}^2\bigr) + 2p_2u_{k,x}u_{k,y},\\[3pt]
H_k(A,g) &= L\,w_k\bigl(1 + \kappa\,\lVert w_k\rVert^2\bigr).
\end{aligned}
$$

The steps run in this order: scale by accommodation, projective divide, affine matrix (inside the divide), Brown tangential map, then radial gain. The order matters. Projective and radial steps do not commute.

### 4.4 Field prediction with the centroid anchor

Let $\bar H(A,g) = \frac19\sum_k H_k(A,g)$. The predicted field is

$$
F_k(A,g) = H_k(A,g) + \omega_k\bigl[B_4(g) - \bar H(0,g)\bigr],
\qquad \omega_{\rm center}=0,\quad \omega_{\rm outer}=\tfrac98.
$$

Consequences: $F_{\rm center}=0$, and $\frac19\sum_k F_k = B_4(g) + \bar H(A,g) - \bar H(0,g)$. At $A=0$ the mean equals the frozen $B_4$ prediction.

**Terms.**
- $v_k$: magnification with accommodation, $s(a)$.
- $D_k$: projective denominator (keystone-like). Physical role: a perspective or tilt effect that makes the pattern lean with gaze.
- $m_{ij}$: affine stretch and shear.
- $p_1, p_2$: Brown–Conrady tangential terms, which describe decentring of optical elements.
- $g_{\rm rad}, k_1, k_2$: radial gain. The gain is $1 + \kappa\|w\|^2$. $\kappa$ contracts or expands the pattern with radius. It can express barrel or pincushion behaviour. $k_1 a$ and $k_2 a^2$ let the barrel strength change with accommodation.
- $\omega$ anchor: fixes the mean of the nine rays to $B_4$. It is a constraint for centroid consistency, not a physical law.

**Interpretation.** Mixed. The projective, Brown, and radial forms are physically suggestive. Their coefficients are empirical surrogates for the ray-traced grid. They are not identified lens prescriptions, and they do not come from a derivation through optical surfaces.

---

## 5. Channel P4 centre motion: M1 (2 fitted values)

The centre moves with accommodation. A diagonal scale about the on-axis ray $o_4 = (0,0)$ models this:

$$
T_A(c_0) = o_4 + \operatorname{diag}(1 + k_x a,\ 1 + k_y a)\,(c_0 - o_4),
\qquad k_x = -0.0413830971,\quad k_y = -0.0414280673.
$$

At $A=4$ D ($a=1$) the scales are about 0.9586 and 0.9586. $T_A$ is an empirical absolute-centre normalization. It is distinct from the relative field map $F$. It is not established as the physical movement of an optical centre.

**Interpretation.** Interpolation, calibrated on sampled accommodations. The on-axis labelled ray is at $(0,0)$ mm at 0, 2, and 4 D in the calibration poses.

**Alternative (not in the 87 model).** A separate physical candidate for the centre motion uses an archived lever $\ell = 8.10207652$ mm and the on-axis scale $s(A) = 1 - 0.081595476571\,(A/4)$. It is distilled from the M1 teacher, not identified from optics. It is documented in [Theory.md](Theory.md) and [Record.md](Record.md).

---

## 6. Accommodation and gaze readout

**Channel P1 (no accommodation).**
1. Solve the centroid fixed point: $g_1 = G_1\bigl(m_1 - B_1(g_1)\bigr)$.
2. Use $g_1$ to seed the accommodation profile of the P4 nine-ray grid. This gives a starting pair $(A_{\rm seed}, g_1)$.

**Channel P4 update.** For a current state $(A,g)$:

$$
\begin{aligned}
c_4 &= m_4 - \bar F(A,g), &
c_{40} &= T_A^{-1}(c_4),\\
g^* &= \operatorname{clip}_{\pm 12.5^\circ}\,G_4(c_{40}), &
A^* &= \operatorname*{arg\,min}_{0\le A'\le 4}\ \sum_{k=1}^{9}\bigl\|(y_{4,k} - m_4) - \bigl(F_k(A',g^*) - \bar F(A',g^*)\bigr)\bigr\|^2.
\end{aligned}
$$

$y_{4,k}$ is the measured $k$-th ray coordinate. The solver iterates the gaze and accommodation updates until they converge. The convergence test is two consecutive undamped residuals within $10^{-5}$ D in $A$ and $10^{-4}$° in gaze.

**Terms.**
- $c_4$: centre estimate from the measured mean minus the predicted mean field.
- $T_A^{-1}$: removes the accommodation-dependent centre motion.
- $G_4$: reads gaze from the corrected centre.
- $A^*$: the accommodation whose predicted shape best matches the measured shape of the nine rays, after removing the common centre.

**Interpretation.** The solver is a fixed-point inverse. It uses no true $A$, true gaze, or labeled centre as input.

---

## 7. Summary of roles

| Block | Count | Role | Physical basis |
|---|---:|---|---|
| $b_{j0}$ anchors | (measured) | zero-gaze offset of the centroid | measured |
| B1, B4 | 18 | gaze-dependent centroid bias | interpolation (empirical) |
| G1, G4 | 40 | centre → gaze readout | interpolation (empirical) |
| PF12 base | 12 | A=0 gaze-dependent field shape | suggestive (projective, Brown, radial), empirical |
| $\lambda_p$ | 12 | accommodation change of field shape | interpolation |
| $s_1$ | 1 | accommodation magnification | suggestive (compare on-axis scale) |
| $k_1, k_2$ | 2 | accommodation change of radial gain | suggestive (compare on-axis barrel fit) |
| $k_x, k_y$ | 2 | accommodation centre motion | interpolation, teacher-calibrated |
| $\omega$ anchor | (fixed) | centroid consistency | constraint, not physical |

---

## 8. Limits and caveats

1. **Count is not validity.** 87 values do not show overfitting or accuracy. Those are assessed by held-out error on identical masks, reported in [Record.md](Record.md).
2. **Empirical surrogates.** The B, G, and field coefficients are fitted to a noiseless ray-traced grid in nominal recorded frames. They are not identified optical parameters.
3. **Frozen PF12 base.** The 12 PF12 values come from an earlier A=0 fit. Their physical meaning is not validated.
4. **Shared sensor frame.** P1 and P4 share a nominal sensor frame. This is not independently verified.
5. **Domain.** Gaze support is ±12.4° (runtime ±12.5°). Accommodation is 0–4 D. Outside these ranges the model is not validated.
6. **Not a forward/inverse lens model.** The model predicts ray positions and reads gaze and accommodation from them. It does not model the optical surfaces or the eye.
7. **Identifiability.** A nine-ray grid has 18 coordinates per gaze state, but the references and outputs are correlated. Rank and count alone do not establish identifiability.
