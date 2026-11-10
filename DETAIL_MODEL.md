# P1/P4 gaze readout model: candidate 67-value form

This file defines the candidate model: parameters, formulas, and the role of each term. It describes the model only. Results, performance, and provenance are in [Record.md](Record.md). The canonical hybrid text is in [Theory.md](Theory.md#current-hybrid-model-and-geometric-interpretation-2026-11-08).

**Status:** candidate, not qualified. Nothing here is validated.

---

## 1. Notation

| Symbol | Meaning | Units |
|---|---|---|
| $g = (\mathrm{ADE}, \mathrm{BDE})$ | eye gaze: ADE (vertical), BDE (horizontal) | degrees |
| $A$ | accommodation | D |
| $a = A/4$ | normalized accommodation, $0 \le a \le 1$ | — |
| $x = \mathrm{BDE}/12.4$, $y = \mathrm{ADE}/12.4$ | normalized gaze for the centroid correction | — |
| $t_x = \mathrm{BDE}/20$, $t_y = \mathrm{ADE}/20$ | normalized gaze for the field correction | — |
| $k = 1,\dots,9$ | index of the nine measured rays | — |
| $m_j$ | arithmetic mean of the nine measured ray coordinates, channel $j \in \{1,4\}$ | mm |
| $c_j$ | corrected labelled central-ray coordinate of channel $j$ | mm |
| $R_{0,k}$ | measured real-ray reference of ray $k$ at $A=0$, zero gaze, centre-relative | mm |
| $L = 1.235429592627$ | reference scale of the P4 field correction | mm |
| $o_4 = (0,0)$ | on-axis P4 pivot | mm |

The gaze scales 12.4° and 20° are fixed constants. The ±12.4° range is the support of the ray data. The runtime allows ±12.5°.

---

## 2. Parameter inventory

| Block | Count | Symbols | Fitted? |
|---|---:|---|---|
| B1: P1 centroid bias | 9 | $\beta_{1,1..4}$, $\gamma_{1,1..5}$ | yes |
| B4: P4 centroid bias | 9 | $\beta_{4,1..4}$, $\gamma_{4,1..5}$ | yes |
| PF12 field base | 12 | $c^0_p$ | frozen from an earlier fit |
| Field interaction | 12 | $\lambda_p$ | yes |
| Field accommodation | 3 | $s_1$, $k_1$, $k_2$ | yes |
| M1 centre scale | 2 | $k_x$, $k_y$ | yes |
| Difference gaze map | 20 nominal (10 effective) | $C_d$ (10 × 2) | yes |
| **Total** | **67 nominal / 57 effective** | | |

The 10 effective difference-map values remain after the exact BDE mirror-symmetry reduction. Only the non-zero coefficients are counted.

**Fixed constants (not counted):** $b_{1,0} = (0,\,-0.05801502493)$ mm, $b_{4,0} = (0,0)$ mm, the training-only centre and scale constants $\mu_d, \sigma_d$, the gaze scales, the accommodation search range $[0,4]$ D, and the gaze clip $\pm 12.5^\circ$.

---

## 3. Centroid correction (B1, B4)

The nine measured rays have arithmetic mean $m_j$. The labelled central ray $c_j$ is a different point. The bias $B_j(g)$ maps the mean to the centre.

$$
\begin{aligned}
B_{j,x}(g) &= b_{j0,x} + \beta_{j1}\,x + \beta_{j2}\,xy + \beta_{j3}\,xy^2 + \beta_{j4}\,x^3,\\[2pt]
B_{j,y}(g) &= b_{j0,y} + \gamma_{j1}\,y + \gamma_{j2}\,y^2 + \gamma_{j3}\,x^2 + \gamma_{j4}\,y^3 + \gamma_{j5}\,x^2y.
\end{aligned}
$$

The corrected centre is $c_j = m_j - B_j(g)$.

**Terms.** $b_{j0}$ is a measured zero-gaze anchor. The $\beta$ and $\gamma$ terms are a low-order gaze-dependent offset. **Role:** interpolation. It is an empirical correction for the arrangement of the nine-ray grid, not a lens law.

---

## 4. P4 field correction (PF12, interaction, accommodation)

### 4.1 Spatial pairs

Gaze enters through 12 spatial pairs. Each pair $p$ has a coefficient function with a frozen base and one fitted accommodation interaction:

$$
\text{coef}_p(t, a) = c^0_p + a\,\lambda_p, \qquad p = 1,\dots,12.
$$

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

The base of $m_{11}$ and $m_{22}$ is 1 at zero gaze and accommodation. The base of the other coefficients is 0 at zero gaze.

**Coefficient conversion.** The frozen PF12 base was written in powers of gaze/12.4. The field uses gaze/20. A base coefficient for $t_x^i t_y^j$ is converted by $c_{\rm new} = c_{\rm old}\,(20/12.4)^{i+j}$. Omitting this breaks the exact A=0 reduction.

### 4.2 Accommodation terms

$$
s(a) = 1 + s_1 a, \qquad \kappa(a,g) = g_{\rm rad}(t) + k_1 a + k_2 a^2.
$$

### 4.3 Field map

For each reference ray $k$ (the centre is not warped):

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

The order is: accommodation scale, projective divide, Brown tangential map, radial gain. Projective and radial steps do not commute.

### 4.4 Centroid anchor

Let $\bar H(A,g) = \frac19\sum_k H_k(A,g)$. The predicted field is

$$
F_k(A,g) = H_k(A,g) + \omega_k\bigl[B_4(g) - \bar H(0,g)\bigr],
\qquad \omega_{\rm center}=0,\quad \omega_{\rm outer}=\tfrac98.
$$

This makes $F_{\rm center}=0$. At $A=0$ the mean of $F$ equals $B_4(g)$.

**Terms.**
- $s(a)$: magnification with accommodation.
- Projective $D_k$: a perspective or tilt effect, keystone-like. Its physical role is suggested, not identified.
- $m_{ij}$: affine stretch and shear.
- $p_1, p_2$: Brown–Conrady tangential terms, which describe decentring of optical elements.
- $g_{\rm rad}$, $k_1$, $k_2$: radial gain, which can express barrel or pincushion behaviour. $k_1$ and $k_2$ let it change with accommodation.
- $\omega$ anchor: a constraint that fixes the mean of the nine rays to $B_4$. Not a physical law.

**Role:** mixed. The projective, Brown, and radial forms are physically suggestive, but their coefficients are empirical surrogates for the ray-traced grid.

---

## 5. Centre motion (M1)

$$
T_A(c_0) = o_4 + \operatorname{diag}(1 + k_x a,\ 1 + k_y a)\,(c_0 - o_4),
\qquad k_x = -0.0413830971,\quad k_y = -0.0414280673.
$$

At $A=4$ D the scales are about 0.9586. **Role:** interpolation, calibrated on sampled accommodations. It is an empirical absolute-centre normalization, not an established physical movement of an optical centre.

---

## 6. Difference readout (the gaze map)

### 6.1 Chain

**Corrected centres.** For P1 the gaze is needed for $B_1$, so the chain runs in a loop:

$$
c_1 = m_1 - B_1(g), \qquad c_4 = m_4 - \bar F(A,g).
$$

**P4 centre inverse.** Remove the accommodation-dependent centre motion, with $s(a)$ as in §5:

$$
c_{4,\rm ref} = o_4 + \frac{c_4 - o_4}{s(a)}\quad\text{(component-wise)}.
$$

**Difference.**

$$
d = c_{4,\rm ref} - c_1 \quad\text{(mm, 2-vector)}.
$$

### 6.2 Degree-3 polynomial

Normalize $d$ with training-only constants $\mu_d, \sigma_d$, and let $\tilde d = (d - \mu_d)/\sigma_d = (\tilde d_x, \tilde d_y)$:

$$
\phi(\tilde d) = \bigl(1,\ \tilde d_x,\ \tilde d_y,\ \tilde d_x^2,\ \tilde d_x\tilde d_y,\ \tilde d_y^2,\ \tilde d_x^3,\ \tilde d_x^2\tilde d_y,\ \tilde d_x\tilde d_y^2,\ \tilde d_y^3\bigr),
\qquad
g = \phi(\tilde d)\, C_d,
$$

where $C_d$ is 10 × 2 and $g = (\mathrm{ADE}, \mathrm{BDE})$ in degrees. Only the 10 non-zero coefficients are kept after the BDE mirror-symmetry reduction.

**Role:** interpolation. The difference map is an empirical fit from centre difference to gaze.

### 6.3 Accommodation and closed loop

Accommodation is estimated from the nine-ray shape, with the current gaze, by minimizing

$$
A^* = \operatorname*{arg\,min}_{0\le A'\le 4}\ \sum_{k=1}^{9}\bigl\|(y_{4,k} - m_4) - \bigl(F_k(A',g) - \bar F(A',g)\bigr)\bigr\|^2,
$$

where $y_{4,k}$ are the measured ray coordinates. The loop is:

1. Estimate $A$ from the shape, with the current gaze.
2. Recompute $c_1$, $c_4$, $c_{4,\rm ref}$, and $d$.
3. Update gaze with $g = \phi(\tilde d)\,C_d$, clipped to $\pm 12.5^\circ$.
4. Repeat until the updates are below $10^{-5}$ D in $A$ and $10^{-4}$° in gaze.

---

## 7. Terms and roles

| Term | Count | Role | Physical basis |
|---|---:|---|---|
| $b_{j0}$ anchors | (fixed) | zero-gaze offset of the centroid | measured |
| B1, B4 | 18 | gaze-dependent centroid bias | interpolation, empirical |
| PF12 base | 12 | A=0 gaze-dependent field shape | suggestive, empirical |
| $\lambda_p$ | 12 | accommodation change of field shape | interpolation |
| $s_1$ | 1 | accommodation magnification | suggestive |
| $k_1, k_2$ | 2 | accommodation change of radial gain | suggestive |
| $k_x, k_y$ | 2 | accommodation centre motion | interpolation, teacher-calibrated |
| $C_d$ | 20 nominal (10 effective) | centre difference → gaze | interpolation, empirical |
| $\omega$ anchor | (fixed) | centroid consistency | constraint, not physical |

---

## 8. Decisions for you

1. **Keep or drop the 47 correction parameters.** Dropping them gives the strict 20-parameter P4 − P1 form (10 effective). It has not been run.
2. **Degree of the difference map.** Degree 3 gives 10 monomials per output. Degree 2 gives 6.
3. **Accommodation source.** Keep the shape-profile estimate, or estimate accommodation from the difference.
4. **Shared sensor frame.** The difference assumes P1 and P4 share a frame. This is not verified, and it limits every version.

## 9. Limits

- Count is not validity. Performance is reported in [Record.md](Record.md).
- The correction and difference coefficients are fitted to a noiseless ray-traced grid in nominal frames. They are not identified optical parameters.
- The frozen PF12 base comes from an earlier fit, and its physical meaning is not validated.
- Gaze support is ±12.4° (runtime ±12.5°). Accommodation is 0–4 D.
