# Model formulas review draft

Slide-ready equations and definitions for review. The barrel/accommodation equations below use the newer physical radial fit. The general two-axis keystone is a proposed extension; it is not the transformation currently fitted to the simulation.

## 1. Physical barrel law

For physical reference-plane coordinates $\mathbf u=(u_x,u_y)^T$ in millimeters, define the radial map

$$
D_k(\mathbf u)=\left(1+k\,\|\mathbf u\|^2\right)\mathbf u,
\qquad
k_1(A)=0.000180236991874\,A^{1.53545703384}-0.0184792556675\;\mathrm{mm}^{-2}.
$$

Here $A$ is accommodation in diopters; the coefficient on $A^{1.53545703384}$ carries the corresponding $\mathrm{mm}^{-2}\,\mathrm D^{-1.53545703384}$ units. This power law was fitted to the 51 per-accommodation radial coefficients from the simulated $E=0$ real-ray grids.

## 2. Fixed-real-reference accommodation map

Let $\mathbf r_0$ be a point in the fixed real-ray reference grid at $A=0$, $E=0$. Let $s(A)$ be the saved paraxial scale ratio derived from the simulated $E=0$ grids, relative to $A=0$. Then the accommodation barrel map is

$$
B_A(\mathbf r_0)
=D_{k_1(A)}\!\left[s(A)\,D_{k_1(0)}^{-1}(\mathbf r_0)\right],
\qquad s(0)=1,
\qquad s(5\,\mathrm D)=0.896199673581.
$$

This composition first removes the baseline physical barrel from the real reference, scales paraxial coordinates by $s(A)$, and applies the physical barrel law at the requested accommodation. It keeps $B_0(\mathbf r_0)=\mathbf r_0$ despite the fitted power-law offset $k_1(0)=c_0$.

The inverse order is

$$
B_A^{-1}(\mathbf r)
=D_{k_1(0)}\!\left[\frac{1}{s(A)}D_{k_1(A)}^{-1}(\mathbf r)\right].
$$

For either radial inverse, solve $\rho_{\rm out}=\rho_{\rm in}(1+k\rho_{\rm in}^2)$ on the near-axis, monotonic branch, which requires

$$
1+3k\rho_{\rm in}^2>0.
$$

## 3. Center-preserving projective keystone

For a barrel output $\mathbf r_b=(x_b,y_b)^T$, a general center-preserving two-axis projective keystone is

$$
K(\mathbf r_b)
=\frac{M(A,E_x,E_y)\,\mathbf r_b}
       {1+q_x(A,E_x,E_y)x_b+q_y(A,E_x,E_y)y_b},
\qquad
M=\begin{bmatrix}m_{11}&m_{12}\\m_{21}&m_{22}\end{bmatrix}.
$$

Its homogeneous matrix is

$$
H=
\begin{bmatrix}
m_{11}&m_{12}&0\\
m_{21}&m_{22}&0\\
q_x&q_y&1
\end{bmatrix}.
$$

The zero translation terms keep the center fixed. The denominator is linear in $x_b$ and $y_b$; it is not a radial function of $\sqrt{x_b^2+y_b^2}$. Rotation about one physical axis often produces a denominator term associated with the other image coordinate, but the sign and axis assignment depend on the optical and coordinate geometry.

## 4. Current fitted keystone and display coordinates

The saved single-angle simulation fit uses a diagonal physical-coordinate numerator, $M=\operatorname{diag}(s_x,s_y)$, and a single denominator coefficient on physical $y$:

$$
K_{\rm saved}(x_b,y_b)
=\left(\frac{s_xx_b}{1+q y_b},\frac{s_yy_b}{1+q y_b}\right),
\qquad q_x=0,\quad q_y=q_n/L.
$$

Here $q_n$ is the saved dimensionless coefficient and $L$ is its coordinate normalization length, so $q$ has units $\mathrm{mm}^{-1}$. This saved fit does not have separate fitted $E_x$ and $E_y$ inputs. The simulation's gaze variable $\theta$ is not identified here with either physical rotation component.

The figures display keystone-bearing panels after swapping physical coordinates: plotted $x$ is physical $y$, and plotted $y$ is physical $x$. Under this display-only swap, the numerator scale labels exchange places and the denominator term appears on plotted $x$:

$$
\begin{bmatrix}x_{\rm plot}'\\y_{\rm plot}'\end{bmatrix}
=\frac{\begin{bmatrix}s_y&0\\0&s_x\end{bmatrix}
        \begin{bmatrix}x_{\rm plot}\\y_{\rm plot}\end{bmatrix}}
       {1+q\,x_{\rm plot}}.
$$

This is a plotting-coordinate swap only; it does not alter the saved physical-coordinate model or ray data.

## 5. Joint forward and inverse maps

The joint forward transformation from the fixed real reference is

$$
\mathbf r=K\!\left(B_A(\mathbf r_0)\right).
$$

To invert a general projective keystone, given $\mathbf r$, recover the barrel-plane point from

$$
\mathbf r_b=(M-\mathbf r\,\mathbf q^T)^{-1}\mathbf r,
\qquad \mathbf q=(q_x,q_y)^T,
$$

when the inverse exists. Then undo the accommodation map in reverse order:

$$
\mathbf r_0=B_A^{-1}\!\left(K^{-1}(\mathbf r)\right).
$$

## 6. Scope of the proposed two-rotation extension

The current simulation varies one gaze angle and fits a one-denominator-term keystone with diagonal scale terms. The full $M(A,E_x,E_y)$ and both $q_x(A,E_x,E_y)$ and $q_y(A,E_x,E_y)$ above are a general proposed extension for separately controlled rotations. They have not been fitted or validated by the current single-angle simulation. Assigning its stored $\theta$ to physical $E_x$ or $E_y$ requires optical-coordinate evidence beyond the saved fit.
