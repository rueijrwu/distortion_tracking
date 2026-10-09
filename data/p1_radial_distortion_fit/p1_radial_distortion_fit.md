# P1 radial distortion versus absolute THI Z

Scope: all 51 in-sample Z levels from -5 to +5 mm, eye rotation 0° only; each group has nine field points. Z is the absolute THI `Cornea_ENT_D` coordinate. The source sweep baseline is 1 mm; this report does not convert Z to baseline displacement.

At each Z, one coefficient is fit by pooled least squares over all 18 x/y coordinates using `real_x = x*(1+k1*r²)` and `real_y = y*(1+k1*r²)`, where `(x,y)` are that Z group's paraxial image coordinates and `r²=x²+y²` in mm². Thus k1 has units mm^-2. Coordinate RMSE is over the 18 coordinate residuals. These spatial residuals measure the radial model fit to each sampled grid. Since paraxial image scale varies with Z, k1 alone does not measure scale-independent barrel strength. Dimensionless normalized K1 is `k1*r_edge²`, with `r_edge` the paraxial radius of the two horizontal noncorner axis-edge points. Edge and corner radial percentages are the stored sweep values averaged over symmetric field points.

The curves fit the 51 k1 estimates directly in k1 space. The offset exponential is `k1=d+a*exp(b*Z)` (coefficient order d, a, b; d/a in mm^-2, b in mm^-1). Its signed b is scanned across both signs and interior minima refined using a stable profiled basis `expm1(b*Z)/b`; its b→0 limit is reported as a line to avoid unstable huge d/a values. The quadratic is `k1=c0+c1*Z+c2*Z²` with coefficient units mm^-2, mm^-3, mm^-4. Curve RMSE/max error are in mm^-2 and are distinct from coordinate grid RMSE in mm.

## Z-dependent curve fits

| Model | Coefficient 0 | Coefficient 1 | Coefficient 2 | k1 RMSE (mm^-2) | k1 RMSE (% of span) | Max absolute k1 error (mm^-2) |
|---|---:|---:|---:|---:|---:|---:|
| offset_exponential | 0.0130687505049 | -0.0261086269851 | -0.00294409977182 | 2.2210535e-09 | 0.00028894% | 5.2370438e-09 |
| quadratic_Z | -0.0130398764628 | 7.68681341348e-05 | -1.13154451225e-07 | 2.9748261e-14 | 3.87e-09% | 7.0877332e-14 |

Across 51 Z groups, grid coordinate RMSE ranges 0.0022423689–0.0023099044 mm (mean 0.00227581 mm); maximum absolute coordinate residual ranges 0.0044847378–0.0046198088 mm (mean 0.00455162 mm). Normalized K1 ranges -0.04104581311 to -0.04104212775 (span 3.68536e-06, 0.0089791% of absolute mean). Mean edge radial distortion spans -4.3611351% to -4.3607275% (0.000407528 percentage points); mean corner distortion spans -8.1450242% to -8.1442969% (0.000727324 points). These dimensionless measures show how the sampled barrel distortion changes after accounting for paraxial scale.

## Selected Z samples

| Absolute THI Z (mm) | k1 (mm^-2) | Edge radius (mm) | normalized K1 | Edge radial distortion (%) | Corner radial distortion (%) | Coordinate RMSE (mm) | Max absolute residual (mm) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| -5.0 | -0.01342704599 | 1.7483352 | -0.04104212775 | -4.3607275 | -8.1442969 | 0.0022423689 | 0.0044847378 |
| -2.6 | -0.01324049854 | 1.7606271 | -0.04104299972 | -4.360824 | -8.1444689 | 0.0022582155 | 0.0045164309 |
| 0.0 | -0.01303987646 | 1.7741399 | -0.04104395321 | -4.3609294 | -8.1446571 | 0.0022756367 | 0.0045512734 |
| 2.6 | -0.01284078424 | 1.7878617 | -0.04104491602 | -4.3610359 | -8.1448471 | 0.0022933283 | 0.0045866566 |
| 5.0 | -0.01265836465 | 1.8007177 | -0.04104581311 | -4.3611351 | -8.1450242 | 0.0023099044 | 0.0046198088 |

## Matched-field Z magnification

The fixed reference is the actual nine-point real-coordinate grid at Z=0. For each Z, one isotropic scale is fit by pooled XY least squares: `m(Z)=sum(R0·RZ)/sum(R0·R0)`, with predicted points `m(Z)R0`. The center is included in the pooled fit (it contributes zero when centered); per-point projected ratios use only the eight noncentral points, avoiding division by a zero reference coordinate. The parallel paraxial ratio uses the same fit against the Z=0 paraxial grid. Coordinate RMSE is over all 18 predicted coordinate residuals; maximum point error is the largest Euclidean residual across nine points.

| Curve | c0 | c1 (mm^-1) | c2 (mm^-2) | Ratio RMSE | Maximum absolute ratio error |
|---|---:|---:|---:|---:|---:|
| linear_Z | 1.00007550219 | 0.00295165270232 | 0 | 6.7498122e-05 | 0.00014352029 |
| quadratic_Z | 0.999999995615 | 0.00295165270232 | 8.71229730269e-06 | 5.1439781e-07 | 1.2194349e-06 |

Across all 51 fitted Z planes, m(Z) spans 0.9854583393 to 1.014977286; the largest real-grid coordinate RMSE is 0.00120667 µm and largest point error is 0.00206133 µm. The edge-to-corner difference in mean projected point ratio ranges from -1.7850004e-06 to 1.8733529e-06, assessing whether a common scale describes the full field. After dividing each Z grid by m(Z), the normalized-grid coordinate RMSE relative to Z=0 ranges from 0 to 0.00118886 µm. This directly tests the residual barrel-shape change after removing the fitted scale.
The paraxial ratio spans 0.9854551017 to 1.014980683. The real-grid ratio is therefore an empirical image magnification relative to the actual Z=0 image, while the paraxial ratio isolates the corresponding reference-ray scale. `normalized_k1_scaled = k1(Z)*m_paraxial(Z)^2` has span 1.1708581e-06; its comparison with k1(0) checks the expected inverse-square coefficient scaling under a common image scale.

| Z (mm) | Real m(Z) | Paraxial m(Z) | Coordinate RMSE (µm) | Max point error (µm) | Edge ratio mean | Corner ratio mean | Edge-corner difference | Normalized-grid RMSE (µm) | k1·mP² |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -5.0 | 0.9854583393 | 0.9854551017 | 0.00114976 | 0.001964109 | 0.9854571817 | 0.9854589667 | -1.7850004e-06 | 0.001166727 | -0.0130392965 |
| -2.6 | 0.9923851807 | 0.9923834777 | 0.000604793 | 0.001033154 | 0.9923845718 | 0.9923855108 | -9.3893969e-07 | 0.0006094337 | -0.01303957353 |
| 0.0 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | -0.01303987646 |
| 2.6 | 1.007732594 | 1.00773434 | 0.0006201787 | 0.001059437 | 1.007733218 | 1.007732255 | 9.6282598e-07 | 0.0006154199 | -0.01304018235 |
| 5.0 | 1.014977286 | 1.014980683 | 0.00120667 | 0.002061327 | 1.014978501 | 1.014976627 | 1.8733529e-06 | 0.001188864 | -0.01304046736 |
