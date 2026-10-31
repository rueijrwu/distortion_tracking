# P4 separable power-barrel and two-q keystone fit

This new fit uses all 51 x 401 simulated states, fitting the four real-ray corners at every state. The five noncorner fields are checks and are not included in the objective. The saved canonical ray grid, original v2 fit artifact, and radial-fit artifacts are read-only inputs.

## Model

The accommodation map is fixed to the saved physical radial law and paraxial scale: $B_A=D_{k_1(A)}\circ S_{s(A)}\circ D_{k_1(0)}^{-1}$. The keystone depends only on the single sampled gaze angle $E$:

$$
K_E(x_b,y_b)=\frac{1}{1+q_{n,x}(E)x_b/L+q_{n,y}(E)y_b/L}\begin{bmatrix}s_x(E)x_b\\s_y(E)y_b\end{bmatrix}.
$$

The plotted panels swap physical x/y for display; the saved coefficients and fit use physical axes. The denominator has two fitted physical-coordinate terms, $q_xx_b+q_yy_b$, with $q_x=q_{n,x}/L$ and $q_y=q_{n,y}/L$ in mm$^{-1}$. No separate $E_x/E_y$ rotations are inferred: $E$ is the dataset's one gaze-angle variable.

The four coefficient vectors are independent: $f_{s_x}(E)=\sum_{j=1}^4c_{s_x,j}(E/20)^j$, $f_{s_y}(E)=\sum_{j=1}^4c_{s_y,j}(E/20)^j$, $f_{q_{n,x}}(E)=\sum_{j=1}^4c_{q_{n,x},j}(E/20)^j$, and $f_{q_{n,y}}(E)=\sum_{j=1}^4c_{q_{n,y},j}(E/20)^j$. Then $s_x=1+f_{s_x}$, $s_y=1+f_{s_y}$; the q curves have zero baseline. Thus $K_0$ is exactly identity. Fit residuals use inverse mapping $B_A^{-1}(K_E^{-1}(\mathbf r))$ against the fixed real A=0,E=0 reference.

## Fit result

Converged: True (`ftol` termination condition is satisfied.); 6 evaluations, 16 fitted coefficients, objective cost=30426347.6, first-order optimality=0.00120141. Coefficients at lower/upper bounds (0-based): [] / []. Reference uses L=1.23542959263 mm. Baseline B0 identity max error=2.22e-13 um. Minimum inverse projective denominator=0.92178201; minimum inverse radial derivative=0.80349689.

Forward predictions passed through $K_E^-1$ then $B_A^-1$ return the fixed reference with max all-nine roundtrip error 1.26e-12 um. On those predicted grids, the minimum inverse projective denominator is 0.91775622 and the minimum inverse radial derivative is 0.80665333.

Inverse-fit corners: vertex RMS/max=27.274239/119.683873 um; coordinate RMSE=19.285799 um; mean/max state RMS=20.879066/86.178702 um.
Forward corners: vertex RMS/max=33.780195/161.745809 um; coordinate RMSE=23.886205 um; mean/max state RMS=24.691591/115.541514 um.
Forward five noncorner checks: vertex RMS/max=20.080153/151.040055 um; coordinate RMSE=14.198812 um; mean/max state RMS=14.400261/75.553051 um.
Forward all nine points: vertex RMS/max=27.040030/161.745809 um; coordinate RMSE=19.120188 um; mean/max state RMS=19.690897/95.345545 um.
Inverse five noncorner checks: vertex RMS/max=16.818506/115.308540 um; coordinate RMSE=11.892479 um; mean/max state RMS=12.470746/61.598033 um.

## Comparison against previous K on the same new B

Both variants use the same fixed physical power-law barrel B_A and score all 51 x 401 actual grids. The previous K uses its saved accommodation-dependent sx/sy/qy and qx=0; the new K uses gaze-only sx/sy/qx/qy.

Old K forward corners: vertex RMS/max=8.920282/24.598265 um; coordinate RMSE=6.307592 um; mean/max state RMS=6.887281/22.211575 um.
New K forward corners: vertex RMS/max=33.780195/161.745809 um; coordinate RMSE=23.886205 um; mean/max state RMS=24.691591/115.541514 um.
Old K forward five noncorner checks: vertex RMS/max=6.466943/26.438234 um; coordinate RMSE=4.572819 um; mean/max state RMS=5.100993/18.651834 um.
New K forward five noncorner checks: vertex RMS/max=20.080153/151.040055 um; coordinate RMSE=14.198812 um; mean/max state RMS=14.400261/75.553051 um.
Old K forward all nine points: vertex RMS/max=7.655009/26.438234 um; coordinate RMSE=5.412909 um; mean/max state RMS=6.059823/19.309926 um.
New K forward all nine points: vertex RMS/max=27.040030/161.745809 um; coordinate RMSE=19.120188 um; mean/max state RMS=19.690897/95.345545 um.
Old K inverse corners: vertex RMS/max=8.806810/21.513570 um; coordinate RMSE=6.227355 um; mean/max state RMS=6.830363/21.346656 um.
New K inverse corners: vertex RMS/max=27.274239/119.683873 um; coordinate RMSE=19.285799 um; mean/max state RMS=20.879066/86.178702 um.
Old K inverse five noncorner checks: vertex RMS/max=6.099822/22.845692 um; coordinate RMSE=4.313225 um; mean/max state RMS=4.891659/16.348171 um.
New K inverse five noncorner checks: vertex RMS/max=16.818506/115.308540 um; coordinate RMSE=11.892479 um; mean/max state RMS=12.470746/61.598033 um.

## Coefficients

Rows are parameter families sx, sy, qnx, qny; columns are powers j=1..4 of E/20:

| Parameter | j=1 | j=2 | j=3 | j=4 |
|---|---:|---:|---:|---:|
| sx-1 | -5.38598555978e-13 | 0.0768204378532 | 1.14597824901e-12 | 0.00276708070792 |
| sy-1 | -1.40561845433e-12 | 0.182098007346 | 3.07151857143e-12 | 0.00137795763328 |
| qnx | 4.33879023453e-15 | 1.12468936607e-14 | -6.04973550706e-15 | -1.77062757612e-14 |
| qny | -0.0962925058499 | -8.52653263906e-12 | 0.00948033919223 | 1.35841814408e-11 |

Both q vectors were fit independently. The fitted $q_{n,x}$ curve is effectively zero by symmetry (maximum absolute value 8.17e-15); $q_{n,y}$ carries the observed gaze-dependent perspective term.

## Fixed barrel inputs

$k_1(A)=(0.000180236991874)A^{1.53545703384}+(-0.0184792556675)$ mm$^{-2}$. Saved paraxial scales include s(0)=1 and s(5)=0.896199673581.

## Files

`p4_transformation_stages.png/.svg`, `p4_transformation_sweep.png/.svg`, and `p4_transformation_accommodation_overlay.png/.svg` show reference, actual, and refit predictions. All keystone panels swap physical axes for display, show only outer perimeters and center dots, and use the established shared limits. `keystone_parameter_curves.png/.svg` shows the four fitted gaze curves; `inverse_corner_rms.png/.svg` maps inverse-fit corner RMS across A/E.

The model is empirical for this one-angle P4 simulation domain. The separate qx and qy terms do not establish independent physical rotation axes or generalize beyond the sampled conditions without new validation.
