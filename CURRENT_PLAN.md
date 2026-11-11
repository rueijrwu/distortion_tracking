# CURRENT_PLAN

## NEXT PLAN

Planning only. No mathematical handoff, fitting, implementation, execution, scoring or new files are authorised. Each future stage needs separate authorisation; acceptance of a scoring protocol does not authorise execution.

Scope is 167.93 mm, zero working distance. Preserve P1 SRC_ROT ADE -20 (this is not a P4 rule) and, for P4, RC at 13.8 mm behind Cornea_ENT. E1 for P1 and E2 for P4 are accepted camera mappings. Do not build distance-transport pilot models, transport a fit to another distance, or revive abandoned eye-stage formulas and centre corrections, including the earlier 88–107-parameter fits. A two-stage model is justified only if its eye stage is small and physically supported.

The first objective is P1 at A=0: fit the model to the eight measured outer-point image positions at known calibration gazes, using the E1 camera composed with an eye-stage model. Use the fitted model and its inverse to recover the missing labelled image centre. At runtime, infer gaze from the eight outer points and then evaluate that centre. An image-position replacement fit has not yet been tested. Extend to P4 and accommodation only after the P1 gates pass.

### Measurement, routing and boundaries

Only the eight labelled outer image points are measured. The missing central-ray position and frame-specific traced tangents are diagnostic truth, never estimator inputs. Explicitly declared design-derived calibration constants may be supplied with uncertainty; they must not be presented as measurements. Keep the camera principal point c_camera, missing labelled grid centre c_grid, observed-point mean and transformation centre distinct unless a derivation establishes their relation.

Gaze is known during calibration. A is known only during calibration at 0, 2 and 4 D; both gaze and A are unknown at runtime. P1 at A=0 is a limited proof, not a general-accommodation solution. Plan 3–5 s fixation holds, with labels, timestamps, stable-window selection and repeated holds/sessions. Treat fixation at the stimulus distance setting A as a calibration-label assumption. Quantify the effective number of independent samples; a hold duration alone does not establish noise reduction.

Default future routing, subject to the user's task-specific overrides: gpt-6-astra high plans and audits; Fable 5.2 does mathematics only when explicitly requested; gpt-6.1-sol xhigh edits only and never runs Python; gpt-6-luna low performs authorised mechanical reads, runs and checks. Haiku is for a real sandbox failure only; Opus requires an explicit user request. These are future roles, not instructions to start work. The standing instruction `/codex:review --background` is recorded, not invoked by this edit.

### Working rules

Close stale `cvgui` before invoking CODE V. Every authorised Python script, module, package-install or check command must invoke `C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe` by absolute path; do not rely on PATH or require a specific Python version.

Write outputs under `runs\<script name>\`, a fixed folder overwritten on re-run; use `runs\<script name>_smoke\` for smoke runs. The exception is `p4_a0_e1_fit.py` → `runs\p4_a0_e1\` (smoke: `runs\p4_a0_e1_smoke\`). Create missing output parents and allow folder re-use. Canonical full-detail pickles use `data\p1_raytrace\p1_raytrace_251x251_12p5.pkl` and `data\p4_raytrace\p4_raytrace_251x251_12p5_A4.pkl`; the user supplies them without copying or renaming data.

Keep all planning in CURRENT_PLAN.md. Create no temporary plan `.md` files or second plan file. CURRENT_MODEL.md records modeling content and current layout only. Report provenance accurately: edited sources do not establish historical executable reproducibility; uncommitted work does not imply a repository without history. Before relying on automated acceptance, review the recorded V-gate/fit-health wording inconsistency and unguarded report writes. All stages below are proposals.

### Gate 0 — user acceptance before scoring

Before scoring, the user must accept the observable inputs, supplied design constants and their uncertainty, gaze domain, calibration budget, split rules, noise definition, centre tolerance, success interpretation, sample sizes and confidence method. The proposed baseline is independent zero-mean Gaussian image noise with per-coordinate sigma=1 µm. Correlated noise, common offsets and calibration-label errors are separate stress cases. No prior result counts as a pass under this new protocol.

The user's gaze target is strict radial two-axis error <1 arcmin; later the accommodation target is absolute error <0.1 D. A proposed supporting centre gate is radial centre-error P95 ≤1 µm, with per-coordinate bias also reported. For P1, define success by valid convergence and gaze <1 arcmin. For the full estimator, require valid convergence, gaze <1 arcmin and A error <0.1 D together. A proposed statistical gate is a one-sided 95% confidence lower bound of at least 95% for the success fraction, separately for each predeclared gaze/accommodation stratum. The user must accept this probabilistic interpretation before scoring; it is not an all-case guarantee. An all-case claim on a finite test set instead requires every case to meet the strict physical targets.

Report signed per-axis bias, RMS, P95 and maximum, along with radial errors, convergence/failure rates and ambiguity counts. Include every evaluated case and all failures in the denominator. A finite maximum gives no universal population guarantee. Acceptance also requires no leakage, an identifiable supported inverse, and the convergence gates below; image residuals alone cannot establish physical accuracy.

### Gate 1 — proposed Fable math-only brief and Astra audit

Ask Fable for a physical derivation of the P1 eye-stage shift and shape terms from the optics. Use the S8 definitions tx=l8/n8 and ty=m8/n8 with their correct signs, ray propagation about RC, surface normals, refraction and stop geometry. Derive the low-order expansion in gaze and true zero-gaze tangent p, including the anisotropic cubic shift terms. Explain the small centre-ray gain and why a rigid-rotation projection does not explain the observed shift.

Distinguish symmetry-forbidden terms from terms allowed by symmetry but expected to be small. Derive or bound the x²*y² and |p|^4*w contributions and the outer-point centre null mode. Symmetry alone must not be used to remove quartic terms. Check whether the reported empirical eye-stage family has a physical basis and whether its omitted terms fit within the proposed centre and gaze error budgets.

Also derive the scale gauge and minimum absolute anchor, the E1 inverse and its full Jacobian, and identifiability of centre-ray versus principal-point offsets. Define the radius r in the assertion that zero-gaze calibration “gives r” and derive and verify that assertion. Define the ring-mean formula's coordinates and units, and test its assumptions. Establish that the eight observed points determine the intended centre under those assumptions; design symmetry alone is insufficient. Give parameter counts, rank and global-ambiguity conditions, counterexamples where needed, and conditions for centre inversion and iteration. Deliver one equation sheet and a minimal test matrix only: no code, fitting, lens execution or numerical success claim.

Astra audits this brief before calibration work. Advance only with derived or bounded terms, or an explicitly justified empirical remainder within the accepted error budgets. An unresolved physical assumption or identifiability failure stops its dependent branch.

### Gate 2 — freeze the P1 calibration and absolute scale

Use a symmetric gaze lattice of 5×5 or larger, covering both axes, the ±12.5-degree boundaries and intermediate magnitudes. Avoid a 3×3 lattice, which aliases linear gain and pure cubic shift. Add corner-weighted repeated fixations while retaining axis and interior coverage. Compare uniform and corner-weighted designs at the same declared total calibration budget, and require full rank for all retained cubic terms.

For the existing design-trace lattice, preserve the even/even training rule exactly. Its angles are -12.5+0.1*i degrees; zero is index 125 and is odd. A symmetric 6×6 training design can use {-12.5,-6.3,-0.1,+0.1,+6.3,+12.5} degrees on each axis, all at even indices. Keep zero gaze and every state with any odd index as holdout; do not relabel them to obtain a 5×5 design. Report even/odd, odd/even and odd/odd holdout groups separately, keeping every ray of a state together. The previously reported 5×5 calculation is an unaudited result, not evidence that this new split has passed.

A new physical-measurement calibration may use a 5×5 or larger lattice including zero, with separately frozen calibration and validation splits. Include the fixed nine-point design at zero gaze for each calibration A, while measuring only its eight outer points. Keep whole holds and sessions out of fitting to avoid frame leakage, and reserve independent gaze locations and boundary checks. No final holdout may select a model family or tune parameters. Freeze T from training information only; do not use holdout or runtime maxima.

For the primary P1 proof, fix the accepted E1 f and k1 and explicitly supply the design zero-gaze tangents p as calibration constants. This inherits an absolute scale from the design and camera; it does not recover that scale from images alone. Use true tangent p as the field variable, not nominal grid-node coordinates. Keep any frame-specific tangent truth solely for diagnostics.

A separate sensitivity arm may supply design gain a instead, fit f and latent zero-gaze tangents from the calibration images, and fix or constrain the weakly identifiable k1. List every supplied and fitted quantity and its uncertainty. Where camera parameters are refitted, a gain error propagates as df/f=-da/a and dk1/k1=-2*da/a. In the fixed-camera arm, propagate gain or tangent errors into the fitted eye stage, gaze and centre instead; do not apply those camera-refit formulas blindly. Sweep agreed gain, design-tangent and camera uncertainties. Fixing f must not be described as independent scale recovery. Astra freezes this complete protocol before any future fit.

### Gate 3 — fit outer image positions and invert on held-out gazes

Fit the forward composition E1(Eye(p,w)) to the eight outer image positions at known calibration gaze w=(BDE,-ADE). Freeze a small candidate family and parameter count before fitting, and estimate parameters only on training data. Keep f and k1 fixed in the primary arm. Audit camera inversion for units, the negative P1 focal sign, unique supported branch and Jacobian conditioning. Image coordinates cannot simply replace tx,ty without this mapping.

For each held-out eight-point image, solve for unknown gaze, then evaluate the inferred eye-stage centre at p=0, with any tested centre-ray offset included. Compare against the traced missing central ray only for evaluation. Compare the ring estimator, the eight-point mean and the nominal 2*edge-mean minus corner-mean estimator as baselines; none is automatically the true centre.

Separate noiseless model error from calibration noise and independent runtime noise. Propose multiple independent noise seeds and at least 200 Monte Carlo trials per seed; the user must accept the sample allocation and statistical power before scoring. Refit each noisy calibration trial, then evaluate independent noisy held-out images. Preserve covariance from shared images and fitted constants. Compare candidates using identical cases and noise realisations.

Report centre errors in µm and gaze errors in arcmin (1 arcmin=1/60 degree), both per-axis and radial. Report bias, RMS, P95, maximum, failures, ambiguity and conditioning overall, by holdout group, gaze magnitude, boundary and midpoint. Check parameter identifiability as well as output error. Low reprojection residuals or a forward–inverse round trip alone cannot establish accurate gaze or centre recovery.

### Gate 4 — test assumptions and model-error sensitivity

Test point symmetry, field asymmetry, grid centring and centre-ray alignment with the camera principal point. Fit explicit offset nuisance models where identifiable. Compare true design tangent p with nominal node coordinates, and perturb x²*y² and |p|^4*w terms, higher shift orders and anisotropic cubic coefficients. Stress common image/tangent offsets, registration, camera constants, gain and gaze labels. Fit only on training data and evaluate on the held-out cases.

Construct outer-point patterns that are indistinguishable at the noise level but predict different centres. If a permitted quartic term leaves the eight observations unchanged while changing the centre beyond tolerance, stop the centre claim or identify the extra external information needed. A small measured quartic contribution is evidence to audit, not permission to set it identically to zero.

Report centre robustness and gaze sensitivity separately. Test the weak information in a free-centre model, multiple inverse roots, branch boundaries and the full declared domain under all noise seeds. Gates 3 and 4 must meet the user-accepted identifiability, error and convergence criteria before extending to P4.

### Gate 5 — P4 first, then accommodation

After the P1 gates pass, repeat the derivation and outer-point calibration for P4 at A=0 with E2. Use tx=-l16/n16 and ty=-m16/n16 at S16, not the P1 S8 convention. Audit symmetries, the supported E2 inverse branch, positive radial factor and radial derivative 1+3*k1*r²+5*k2*r⁴>0 over the supported tangent domain. Record eight-point availability and correspondence in each channel, their shared-coordinate registration, common gaze and each recovered centre's uncertainty before combining P1 and P4. Freeze deployable calibration information and runtime accommodation availability before scoring, together with Gate 0's noise assumptions and physical tolerances.

Only after P4 at A=0 passes, calibrate matched gaze fixations at A=0,2,4 D. Randomise or counterbalance gaze/A order, repeat holds across sessions, and quantify fixation-label uncertainty and drift. Verify P1 accommodation coverage; stable camera coefficients do not establish accommodation-invariant eye-stage tangents. Derive or fit bounded eye-stage accommodation dependence without assuming it is linear.

Reserve independent checks at A=1 and 3 D and held-out odd-index gaze groups. Previously inspected intermediate-accommodation data are diagnostic checks, not untouched final validation. Match channels by actual gaze, accommodation and field labels rather than array indices. Fix accepted camera mappings and deployable normalisations explicitly; no true-runtime-A camera selection or true-A-dependent T is allowed.

Before attempting runtime iteration, require a nuisance-adjusted observation Jacobian that identifies ADE, BDE and A, check global collisions, and propagate the complete image/calibration/registration covariance into gaze and A error. Stop if the 1 µm noise floor or label uncertainty consumes the accepted budget. Historical C20 is the 20-parameter P4−P1 inverse map for a different feature; its applicability to recovered centres is unverified. Refit and evaluate any centre-based replacement on held-out data using the actual observed or inferred feature. Do not transplant historical coefficients or reuse the historical c=m−B(g) correction with unknown gaze without an explicitly identifiable joint solution.

### Gate 6 — joint unknown-A runtime iteration and final audit

After all preceding gates pass, initialise gaze from a calibration-fitted inverse of the observed P4-minus-P1 mean feature only if that feature is identifiable with unknown A. If it is ambiguous, revise the plan before proceeding. Use a declared calibration-based A seed or bounded multistart search; never use the true runtime A, gaze or missing centre.

Then repeat the requested order: estimate A from the observed points and current gaze; estimate the centres conditional on that A and gaze; apply the calibrated centre correction and re-estimate gaze. Refit the gaze inverse for the actual centre/observed-point features. Write the complete update as z(k+1)=F(z(k);images), with z=(gaze,A,centres), and require its fixed point to agree with the observation model within the declared noise tolerance.

Audit spectral radius of the update Jacobian below one for local convergence and test the initializer's basin over all held-out gaze/A cases. Test multiple roots, false convergence, cycles, boundaries and nonconvergence. Damping is a candidate only if its stability is established. Freeze iteration caps, residual/conditioning gates, stopping tolerances small relative to the accepted error budgets, and out-of-domain handling. Do not hide failures through clipping or omitted cases.

Propagate fixed-point sensitivity through (I-D_z F)^(-1)*D_images F where valid, preserving reused-image covariance. Iteration supplies no new independent information. Report initial versus final gaze, centre and A errors, iteration counts and failures, separately and jointly across held-out gazes and accommodation levels. Apply the user-accepted Gate 0 criteria without changing them after seeing results.

Astra performs the final audit. Distinguish mathematical identifiability, design-trace accuracy, noise robustness and real-measurement validation. No success is claimed in advance, and no future work is authorised by this plan.
