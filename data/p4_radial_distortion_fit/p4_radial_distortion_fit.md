# P4 radial distortion fit

This fit uses only the existing P4 grids at eye rotation 0° and accommodation 0–5 D. Each accommodation group contains nine field points.

The forward model is `real_x = x * (1 + k1*r^2)` and `real_y = y * (1 + k1*r^2)`, where `(x, y)` are paraxial image coordinates and `r^2 = x^2 + y^2`. This maps ideal/paraxial coordinates to real distorted coordinates. Coordinates are in millimeters, so `k1` is in mm^-2. One coefficient is fit by pooled least squares over all 18 x/y coordinates; coordinate RMSE is sqrt(mean of 18 squared coordinate residuals), including the center's two zero residuals. The stored real coordinates are already relative to the traced center ray.
The 3 × 3 grid has only two distinct nonzero sampled radii (edge midpoints and corners); the center has zero radius and contributes no leverage. The paraxial grid scale also varies with accommodation, so the model is evaluated against each group's own stored ideal coordinates.

## Accommodation models for fitted k1

The accommodation curves are fit to the 51 per-accommodation k1 estimates. The offset exponential is `k1 = d + a*exp(b*A)` with signed, unconstrained coefficients and direct least squares in k1 space (coefficient order d, a, b). It is optimized by scanning b across [-2, 2] D^-1, refining every local minimum, and expanding the interval if the best result is at a boundary; the selected optimum is interior. A stable profiled basis `expm1(b*A)/b` is used, with linear limit A at b=0. For the exponential, d and a are in mm^-2 and b is in D^-1. The quadratic is `k1 = p0 + p1*A + p2*A^2` (coefficient order p0, p1, p2; units mm^-2, mm^-2/D and mm^-2/D^2). Model RMSE is sqrt(mean((prediction-k1)^2)) in mm^-2; the percent column divides RMSE by the observed k1 range (max-min). These are coefficient-curve errors, distinct from coordinate RMSE in mm above. The lower plot panel shows coefficient errors in units of 10^-4 mm^-2.

| Model | Coefficient 0 | Coefficient 1 | Coefficient 2 | RMSE k1 (mm^-2) | RMSE (% of k1 range) | Maximum absolute k1 error (mm^-2) |
|---|---:|---:|---:|---:|---:|---:|
| offset_exponential | -0.0195347321373 | 0.000993232205102 | 0.235926474606 | 4.126091e-06 | 0.18615% | 1.3168725e-05 |
| quadratic_A | -0.0185179003996 | 0.000180197179361 | 5.15785422657e-05 | 6.0346857e-06 | 0.27226% | 1.5651084e-05 |

Across 51 accommodation groups, coordinate RMSE ranges from 0.000534472 to 0.00110191 mm (mean 0.000807812 mm). Maximum absolute coordinate residual ranges from 0.00106894 to 0.00220383 mm (mean 0.00161562 mm). These residuals show the adequacy of this one-parameter radial model for the sampled nine-point grids.

| Accommodation (D) | k1 (mm^-2) | Coordinate RMSE (mm) | Maximum absolute residual (mm) |
|---:|---:|---:|---:|
| 0.0 | -0.01852833121 | 0.00110191 | 0.00220383 |
| 0.1 | -0.01850908193 | 0.00108946 | 0.00217891 |
| 0.2 | -0.01848757484 | 0.00107698 | 0.00215396 |
| 0.3 | -0.01846482412 | 0.00106453 | 0.00212907 |
| 0.4 | -0.01844106214 | 0.00105214 | 0.00210428 |
| 0.5 | -0.01841644591 | 0.00103977 | 0.00207954 |
| 0.6 | -0.01839112337 | 0.00102744 | 0.00205487 |
| 0.7 | -0.01836494013 | 0.00101514 | 0.00203029 |
| 0.8 | -0.01833755631 | 0.00100295 | 0.0020059 |
| 0.9 | -0.01830982527 | 0.000990726 | 0.00198145 |
| 1.0 | -0.01828098946 | 0.000978608 | 0.00195722 |
| 1.1 | -0.01825133076 | 0.000966514 | 0.00193303 |
| 1.2 | -0.01822096259 | 0.000954458 | 0.00190892 |
| 1.3 | -0.0181897887 | 0.00094244 | 0.00188488 |
| 1.4 | -0.01815757967 | 0.000930497 | 0.00186099 |
| 1.5 | -0.01812455952 | 0.000918605 | 0.00183721 |
| 1.6 | -0.01809053502 | 0.000906756 | 0.00181351 |
| 1.7 | -0.01805587632 | 0.000894951 | 0.0017899 |
| 1.8 | -0.01802025137 | 0.000883182 | 0.00176636 |
| 1.9 | -0.01798376328 | 0.000871475 | 0.00174295 |
| 2.0 | -0.01794619424 | 0.00085981 | 0.00171962 |
| 2.1 | -0.01790767949 | 0.000848214 | 0.00169643 |
| 2.2 | -0.01786835333 | 0.000836641 | 0.00167328 |
| 2.3 | -0.0178281768 | 0.000825123 | 0.00165025 |
| 2.4 | -0.01778678495 | 0.000813663 | 0.00162733 |
| 2.5 | -0.01774420802 | 0.000802273 | 0.00160455 |
| 2.6 | -0.0177009096 | 0.000790923 | 0.00158185 |
| 2.7 | -0.01765655085 | 0.000779617 | 0.00155923 |
| 2.8 | -0.01761109538 | 0.000768363 | 0.00153673 |
| 2.9 | -0.01756468188 | 0.000757159 | 0.00151432 |
| 3.0 | -0.01751709235 | 0.000746017 | 0.00149203 |
| 3.1 | -0.01746842501 | 0.000734914 | 0.00146983 |
| 3.2 | -0.01741875853 | 0.000723868 | 0.00144774 |
| 3.3 | -0.01736773743 | 0.000712888 | 0.00142578 |
| 3.4 | -0.01731579967 | 0.000701951 | 0.0014039 |
| 3.5 | -0.01726268473 | 0.000691061 | 0.00138212 |
| 3.6 | -0.01720816794 | 0.000680236 | 0.00136047 |
| 3.7 | -0.01715267104 | 0.000669457 | 0.00133891 |
| 3.8 | -0.01709592404 | 0.000658734 | 0.00131747 |
| 3.9 | -0.01703793107 | 0.000648069 | 0.00129614 |
| 4.0 | -0.01697854417 | 0.000637461 | 0.00127492 |
| 4.1 | -0.01691785946 | 0.000626909 | 0.00125382 |
| 4.2 | -0.01685608222 | 0.000616406 | 0.00123281 |
| 4.3 | -0.01679292057 | 0.000605962 | 0.00121192 |
| 4.4 | -0.01672849062 | 0.000595574 | 0.00119115 |
| 4.5 | -0.0166625825 | 0.000585247 | 0.00117049 |
| 4.6 | -0.01659535931 | 0.000574971 | 0.00114994 |
| 4.7 | -0.01652655206 | 0.000564759 | 0.00112952 |
| 4.8 | -0.0164564051 | 0.000554604 | 0.00110921 |
| 4.9 | -0.01638486663 | 0.000544509 | 0.00108902 |
| 5.0 | -0.01631179986 | 0.000534472 | 0.00106894 |
