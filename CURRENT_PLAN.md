# Current plan

**Current operational contract: Section 18 (Project cleanup); current scientific/documentation contract: Section 17.** Section 18 cleanup MOVE and DELETE are complete, and the cleanup goal is closed. The harmless Windows-permission leftovers `.codex\AGENTS.md` (move denied) and empty `.tmp` (delete denied) are intentionally retained for optional user manual removal. Section 17 accepts FL as the default, consolidates the two P4 documents, and closes the scientific goal for handoff. The opening extraction record and Section 16 below are historical.

Historical document-only extraction of the frozen S2 / algorithm F record (2026-11-16):

1. Preserve archived `P4_MODEL.md` lines 7-52 verbatim in a new root `P4_MODEL.md`, beneath the iterative gaze/accommodation title and three short status lines: S2 / algorithm F frozen, noise-free development results, and the separate future linear-in-A-only test.
2. Remove archived source lines 5-53 only: S2 pointer (5), S2 text (7-52), and adjacent blank separators (6 and 53); preserve every other source byte.
3. Append `## Future: linear-A-only test` and `Results will be added after the test run.` after the copied S2 section. Do not run scripts, tests, builds, or experiments.
4. Submit the document changes for independent audit and report line counts, removed source ranges, resulting archive opening, and preservation limits.

Implementation complete; independent content audit passed with historical preservation limitations.
Historical restoration of the archived opening cannot be certified without a pre-S2 baseline.

## 18. Project cleanup

**Current operational closure record, 2026-11-16; MOVE complete, DELETE complete, cleanup goal closed.** Preserve the accepted FL default and F/quadratic fallback, the complete scientific handoff in Section 17, and the historical strict experiment contract in Section 16. The present edit is documentation only, limited to line 3, this section and the root `AGENTS.md` cleanup pointer; it performs no script execution, imports, tests, Git operations, moves or deletions. gpt-6.1-sol/xhigh edits the authorized documents; gpt-6-astra/high audits this documentation before it is accepted. Native background role agents substitute for the unavailable `/codex:rescue --background` command. All cleanup inventories and new cleanup archive destinations are consolidated here only. Scientific documents must remain self-contained and require no saved output to be read; this section's operational inventory is the exception to the no-output-path documentation rule. The pre-existing historical source references explicitly retained in `Record.md` are separate from this cleanup archive.

### 18.1 Retained runtime closure, inputs and comparison evidence

Keep exactly these eleven Python files in `Script\`, byte-for-byte unchanged:

| Entry points and retained helpers |
|---|
| `Script\p4_linA.py` |
| `Script\p4_final.py` |
| `Script\p4_accom_test.py` |
| `Script\p4_a0_data.py` |
| `Script\p4_a0_gazepoly.py` |
| `Script\p4_a0_ninept5.py` |
| `Script\p4_termwise_test.py` |
| `Script\p4_a0_ninept.py` |
| `Script\p4_a0_joint_fit.py` |
| `Script\p4_a0_gain.py` |
| `Script\p4_a0_tilt_fit.py` |

The local source closure is `p4_linA` -> `p4_final` -> `p4_accom_test`, plus the eight helper modules listed above. It uses static local imports, with no dynamic imports and no dependency on a script scheduled to move. The primary entry point compares the linear-in-A FL default with quadratic F. FL's gaze tensor is $2\times10\times2$ and F's is $3\times10\times2$; at each accommodation both have ten monomials times two outputs, or twenty coefficient positions. FL's fourteen exact-zero positions leave six linear laws, not a smaller polynomial definition.

Current runtime inputs are the 21 full 5x5 shards `data\p4_raytrace\p4_raytrace_5x5_101x101_12p5_A00.pkl` through `data\p4_raytrace\p4_raytrace_5x5_101x101_12p5_A20.pkl`, plus `data\p4_raytrace\p4_raytrace_5x5_manifest.json`. Retain all 25 existing files in `data\p4_raytrace` unchanged; the other three retained files are not additional FL/F runtime prerequisites. No input regeneration, schema/manifest change or data move is planned for that directory.

The eleven retained scripts form an import closure, not an assurance that every helper runs independently with its legacy defaults. The legacy default input name `p4_raytrace_251x251_12p5_A4.pkl` is stale for `p4_a0_gain.py`, `p4_a0_tilt_fit.py`, `p4_a0_joint_fit.py` and `p4_a0_ninept.py`; production FL/F routes (`p4_linA.py` and `p4_final.py`) do not use that default and instead use the documented 21 full 5x5 shards plus manifest under `data\p4_raytrace`.

The user-verified final output tree contains only `runs\p4_linA` (linear-in-A FL) and `runs\p4_final` (quadratic F), both based on the full twenty-coefficient polynomial. Each contains exactly `constants.json`, `progress.log` and `summary.txt`: six retained files in total. The folder totals 868,322 bytes for `runs\p4_linA` and 96,269 bytes for `runs\p4_final` are historical recorded totals, not newly verified byte counts. The former preserves the F/FL comparison and original strict result. The latter preserves quadratic F evidence together with separate historical LF comparisons and an L3 terminal selection. Its `polynomial.coefficient` tensor has shape `[3,10,2]` and belongs to quadratic F. F-labelled A00 and pooled A00/A01-A19/A20/all21 rows are F evidence. LF-L2/LF-L3 are a separate historical branch: terminal `selected_form=L3` / `CLOSE: FULL` belongs to that earlier LF comparison after the ring goal narrowed to ring size. It neither accepts FL nor reinterprets the F coefficients/rows. Preserve all these original records without rewriting them.

### 18.2 Complete filesystem dispositions

Retain the six root files `.gitignore`, `AGENTS.md`, `CURRENT_PLAN.md`, `P4_IMPLEMENTATION.md`, `P4_MODEL.md` and `Record.md`. The present documentation exception is limited to `CURRENT_PLAN.md` line 3 and Section 18 and the root `AGENTS.md` cleanup pointer; no other finishing pointer repairs are authorized here, and concurrent document changes must remain protected. Do not edit `.gitignore`. Leave `.git` entirely untouched, including `.git\index`; perform no Git commands.

The following sixteen script moves are complete; they were ordinary same-volume filesystem moves:

| Source | Destination |
|---|---|
| `Script\p1_camera_fit.py` | `archive\202611_16_cleanup\Script\p1_camera_fit.py` |
| `Script\p1_camera_refit.py` | `archive\202611_16_cleanup\Script\p1_camera_refit.py` |
| `Script\p1_retrace_no_dm_full_detail.py` | `archive\202611_16_cleanup\Script\p1_retrace_no_dm_full_detail.py` |
| `Script\p1_uv_gaze_study.py` | `archive\202611_16_cleanup\Script\p1_uv_gaze_study.py` |
| `Script\p4_a0_camera_ladder.py` | `archive\202611_16_cleanup\Script\p4_a0_camera_ladder.py` |
| `Script\p4_a0_e1_fit.py` | `archive\202611_16_cleanup\Script\p4_a0_e1_fit.py` |
| `Script\p4_a0_joint_fast.py` | `archive\202611_16_cleanup\Script\p4_a0_joint_fast.py` |
| `Script\p4_a0_radial0.py` | `archive\202611_16_cleanup\Script\p4_a0_radial0.py` |
| `Script\p4_a0_unwarp.py` | `archive\202611_16_cleanup\Script\p4_a0_unwarp.py` |
| `Script\p4_camera_linear_accom.py` | `archive\202611_16_cleanup\Script\p4_camera_linear_accom.py` |
| `Script\p4_camera_per_accom.py` | `archive\202611_16_cleanup\Script\p4_camera_per_accom.py` |
| `Script\p4_fit.py` | `archive\202611_16_cleanup\Script\p4_fit.py` |
| `Script\p4_lensmean_test.py` | `archive\202611_16_cleanup\Script\p4_lensmean_test.py` |
| `Script\p4_polyA.py` | `archive\202611_16_cleanup\Script\p4_polyA.py` |
| `Script\p4_retrace_full_detail.py` | `archive\202611_16_cleanup\Script\p4_retrace_full_detail.py` |
| `Script\p4_unwarp_accom.py` | `archive\202611_16_cleanup\Script\p4_unwarp_accom.py` |

The whole `data\p1_raytrace` directory has been moved to `archive\202611_16_cleanup\data\p1_raytrace`, with SHA256 `914E240CC5580821F4E79A580840339513F2C5CD4F892FBED499C8C3BDFEC296` verified before and after the move. The move of `.codex\AGENTS.md` to `archive\202611_16_cleanup\.codex\AGENTS.md` failed with access denied; the 2,011-byte obsolete workflow file remains in place. Astra audit accepts this exception. Existing archive content must remain protected; never merge into or overwrite a destination collision. Retain `.vscode\settings.json`, `.claude\CLAUDE.md` and `.claude\scheduled_tasks.lock` unchanged. Retain the three PDF files in `docs\` and the seventeen optical files in `Lens\` unchanged. The empty `.tmp` directory also remains because Windows permissions denied its deletion. Both harmless permission-denied leftovers are intentionally retained for optional user manual removal; neither blocks closure.

The following eighteen obsolete output directories and their contents were deleted after Astra ACCEPT, per the user-verified cleanup record:

| Explicit directory allowlist |
|---|
| `runs\docs_text` |
| `runs\p4_a0_joint` |
| `runs\p4_a0_tilt` |
| `runs\p4_accom` |
| `runs\p4_data` |
| `runs\p4_gain` |
| `runs\p4_gazepoly` |
| `runs\p4_history` |
| `runs\p4_lensmean` |
| `runs\p4_ninept` |
| `runs\p4_ninept5` |
| `runs\p4_radial0` |
| `runs\p4_retrace_5x5_audit` |
| `runs\p4_retrace_full_detail_full5` |
| `runs\p4_retrace_full_detail_pilot5_smoke` |
| `runs\p4_termwise` |
| `runs\p4_unwarp` |
| `runs\p4_unwarp_accom` |

The following seventeen loose log files were also deleted after Astra ACCEPT, per the user-verified cleanup record; their names were read from the output directory during the earlier documentation stage:

| Explicit loose-file allowlist |
|---|
| `runs\full5b_stderr.txt` |
| `runs\full5b_stdout.txt` |
| `runs\full5_stderr.txt` |
| `runs\full5_stdout.txt` |
| `runs\n5full_stderr.txt` |
| `runs\n5full_stdout.txt` |
| `runs\p5pilot_stderr.txt` |
| `runs\p5pilot_stdout.txt` |
| `runs\p5subset_stderr.txt` |
| `runs\p5subset_stdout.txt` |
| `runs\r0full2_stderr.txt` |
| `runs\r0full2_stdout.txt` |
| `runs\r0full_stderr.txt` |
| `runs\r0full_stdout.txt` |
| `runs\r0s5_stderr.txt` |
| `runs\r0s5_stdout.txt` |
| `runs\subagent_status.log` |

The obsolete-output deletion's approximately 16.8 GB (15.66 GiB) was a pre-deletion inventory estimate, not permission to broaden the allowlist. The completed operation's guardrails were: never recursively delete the `runs` root; resolve and verify every absolute target remains inside the intended project/output directory before any recursive move or deletion; stop the affected stage for review on inventory drift, an unexpected writer, a destination collision or any ambiguous target; do not infer a replacement target or add files to these lists. These historical guardrails authorize no new filesystem operation.

### 18.3 Evidence preconditions, execution order and acceptance checklist

Historical pre-deletion evidence/pointer requirements (not new edits): Before deletion, `Record.md` must preserve the dated termwise and lens-map experiment facts, metrics, failed cases, uncertainty, provenance limits and audit attribution inline, without output-folder citations. Replace the obsolete input requirement with the current contract in `P4_IMPLEMENTATION.md` Section 2; correct the historical compact/effective source locations and historical plan references; remove both obsolete lens-map script pointers while preserving their dated line/byte snapshots, fingerprint and audit attribution. Keep the valid termwise source pointer. The Padé Taylor values, selection rule, five nonzero radii, gauges, scale and fixed-control discrepancy must be inline rather than referred to the current FL theory. Historical staging, untracked-file and commit observations remain dated facts, not instructions to act today. Do not recreate the obsolete root `CURRENT_MODEL.md`, `Theory.md` or `P4_SPEC.md`.

Completed cleanup chronology and present documentation-only audit; filesystem completion is the user-verified record:

1. Per the user-verified cleanup record, Astra/high returned ACCEPT before DELETE proceeded. This historical acceptance is separate from the audit of the present documentation update.
2. The original pre-mutation plan called for Luna/low to recheck the exact inventories, active writers and destination absence, and record pre-mutation hashes for all eleven retained scripts, all six retained output files, `.git\index`, every move-source file, all retained inputs and protected files, plus current-document baselines. This was the planned baseline requirement, not a claim that it was fully obtained; see the provenance limits below. Inventory, active-writer, collision and concurrent-document checks were historical pre-deletion guardrails, not new verification claimed by this update.
3. MOVE is complete for the sixteen scripts and the P1 data directory; the P1 SHA256 was verified before and after the move. The attempted `.codex\AGENTS.md` move remains the accepted access-denied exception described above.
4. DELETE initially hit a Codex sandbox execution-policy block, then completed through the Haiku sandbox fallback under the project rules after Astra ACCEPT, per the user-verified cleanup record.
5. The user verified removal of the exact eighteen directory targets and seventeen loose logs listed above. The empty `.tmp` directory could not be deleted because of Windows permissions and is intentionally retained for optional user manual removal; it does not block closure.
6. The user-verified final inventory is eleven kept scripts in `Script\`, all 25 files in `data\p4_raytrace`, and only `runs\p4_linA` and `runs\p4_final`, each with `constants.json`, `progress.log` and `summary.txt`. `archive\202611_16_cleanup` contains the sixteen moved scripts and `data\p1_raytrace`.
7. The present closure-status update is documentation only, implemented by sol/xhigh and independently audited by Astra/high before this documentation is accepted. The cleanup goal is closed; no new filesystem verification or final documentation-audit acceptance is claimed here.

Recorded MOVE evidence and provenance limits: `Script\` contains exactly eleven kept scripts. The sixteen script moves were same-volume filesystem moves, but no pre-move per-file baseline hashes were retained, so a byte-preservation comparison for those scripts cannot be certified. A claim that nothing else was altered in pre-existing archive folders also cannot be certified.

Retained acceptance constraints and provenance limits:

- Exact eleven-file static import closure, exact dispositions and allowlists subject to the accepted `.codex\AGENTS.md` move-denied and empty `.tmp` delete-denied exceptions, and no moved dependency. P1 data has a verified pre/post-move SHA256; certifications of script byte preservation and unchanged pre-existing archive content remain bounded by the provenance limits above.
- All 25 P4 input-directory files intact; current 21-shard-plus-manifest loading preserved; both FL/F twenty-position configurations documented correctly.
- Six retained comparison-output files unchanged, including quadratic F evidence, separate historical LF/L3 records and the original `L0=True`, `L1=True`, `L2=False`, `CLOSE: F` result.
- User acceptance of approximately 1 arcmin selects FL default/F fallback and remains distinct from the failed strict $<1.0$ gate; no scientific goal is reopened or numerical gate relaxed.
- Self-contained theory, procedures and historical evidence; no scientific dependency on deleted outputs; no new cleanup archive pointers outside this section except the explicitly retained pre-existing historical source references.
- No `.git`/index or `.gitignore` changes, no script execution/imports/tests, no concurrent document edits overwritten, and no scope expansion after inventory drift.

All verification for this cleanup is static. Do not run either `p4_linA.py` or `p4_final.py`, including either entry point's selftest: both selftests overwrite saved output artifacts. No new scientific execution is needed for filesystem cleanup.

Implementation status: MOVE and DELETE are complete, and the cleanup goal is closed. The harmless `.codex\AGENTS.md` move-denied and empty `.tmp` delete-denied leftovers are intentionally retained for optional user manual removal; the provenance limits above remain. This status edit is documentation only; no script, test, import, Git operation, move or deletion is performed by it.

## 17. Accept FL as default, consolidate P4 documentation, close the goal

**Current implementation contract, 2026-11-16.** Documentation only: do not change code, constants, numerical gates or recorded outputs, and do not execute scripts or tests. gpt-6.1-sol/xhigh implements this plan; gpt-6-astra/high audits every implementation. User decision: "accept this linear A model as default. quadratic will be fallback or future use. rewrite/consolidate @P4_MODEL.md and @P4_IMPLEMENTATION.md. need close the current goal for next new goal (as handoff)." The user accepts approximately 1 arcmin FL gaze error. This decision selects FL without changing the original strict test or its recorded verdict.

1. Consolidate `P4_MODEL.md` as a theory-first document: accepted FL default and F/quadratic fallback/future scope; eight labelled measurements, their mean, fixed A00 origin, displacement, axis normalization, gaze signs/units and tangent coordinates; all ten cubic monomials and the full 20-coefficient two-axis polynomial, the six allowed terms ($w_x$: $x,x^3,xy^2$; $w_y$: $y,x^2y,y^3$), $c(A)=c_0+c_1A$, fourteen exact zeros, and the distinction between spatial cubic order and linear accommodation trends. Explain camera distortion, centering/shift, magnification, radial/barrel effects, empirical angular response including $\mathbf z=H_A\mathbf d(1+q_A\lVert\mathbf d\rVert^2)$, pointwise camera inversion before mean removal, reference normalization, two-radius projection, $\mu/\beta$, and the existing keystone/barrel interpretation without inventing an identifiable physical keystone law. Define $E(\boldsymbol\tau)$, $\gamma(A)=\gamma_0+\gamma_1A$, $\mu_{\rm cal}(A)=u_0+u_1A$, and $M(A,\boldsymbol\tau)=(u_0+u_1A)[1+E(\boldsymbol\tau)\mathbin{\cdot}(\gamma_0+\gamma_1A)]$; its induced $A^2$ product is not a fitted quadratic law. Explain conditional gaze, fixed-gaze A inversion varying both factors of $M$, iteration and final gaze recomputation; anchors $A=0,1,2,3,4$ D and gated $[1,A]$ fits; F quadratic trends and piecewise-linear sizes only as fallback. End with recorded performance/limits and a closed-goal handoff. Remove future-linear-A-only framing.
2. Consolidate `P4_IMPLEMENTATION.md`: runtime/source map centered on `Script\p4_linA.py` and unchanged helpers; data/coordinates/normalization; exact FL calibration and immutable gates; camera inversion by bracketed Newton and radial size projection; table inversion and iteration; stored constants/counts; results/acceptance; reproduction/source references. Calibration order is full anchor cubic fits, record forbidden magnitudes, zero forbidden positions, $[1,A]$ gaze trends; fit the mu line to measured anchor ordinates and use line-derived runtime ordinates; refit anchor gamma using true-gaze tangents and $\mu_{\rm raw}/\mu_{\rm FL}-1$ targets, then $[1,A]$ gamma trends. The zero quadratic gamma row is compatibility padding only. Explain finite checks, column scaling, full rank, SVD rcond $10^{-12}$, scaled condition $10^8$, and scaled nonlinear gradient $10^{-10}$; neither gate was relaxed. Runtime starts at $A_0=0$, uses 401 nodes from 0 to 4 D by 0.01 D, strict monotonicity, interpolation and nearest-endpoint clipping (ties choose zero), $10^{-6}$ D tolerance, at most 20 passes and final gaze recomputation.
3. Give independent stored-value counts for FL/F: gaze 12/60, gamma 10/15, mu 2/5, law total 24/80; add scales (2), origin (2), $H_0,q_0$ to obtain 30/86; add camera $F,K_1,K_2$, centre (2) to obtain 35/91. Explicitly answer that each A still has the same order-3, 20-coefficient two-dimensional polynomial (ten monomials times two axes), with fourteen forced zeros and six coefficients fitted linearly in A giving twelve stored values. The 24/30/35 counts describe stored law/calibration values, not polynomial size. Structural zeros, padding and derived ordinates are not independently stored law parameters. Do not include long arrays.
4. Copy evidence accurately from the completed noise-free development comparison: $21\times10{,}201=214{,}221$ states, 100% coverage, F reproduced within $10^{-6}$ arcmin. FL gaze RMS/max 0.1544/1.0176 arcmin versus F 0.0195/0.1904; A RMS/max 0.019/0.168 D versus 0.0019/0.033; passes median/max 6/11 versus 6/12; clipped 11,209 versus 6,180; states above 0.1 arcmin 93,258 versus 120. FL A01-A19 gaze RMS/max 0.1537/1.0176; A00 0.0652/0.0956 with A approximately zero; A20 0.2188/0.8403 with A RMS/max 0.026/0.105 D; eight states above 1 arcmin, all interior; clips A00 10,164 and A20 1,045. Preserve the original strict result `L0=True`, `L1=True`, `L2=False`, `CLOSE: F`; distinguish later user acceptance of approximately 1 arcmin and FL default selection. Never claim that 1.0176 passed strict $<1.0$.
5. Close this goal and hand off FL default/F fallback. Document, but do not execute, reproduction from the repository root: `& 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe' .\Script\p4_linA.py`. The command retains the original strict-gate verdict. Limits: noise-free design traces; held-out accommodation values previously inspected, not blind; no real-eye/noise validation, contraction proof or A-error acceptance threshold; AE7 simultaneous fit not performed. Candidate next goals are blind/noisy/real-eye validation, accommodation criteria, iteration robustness/convergence and AE7 simultaneous fitting.
6. Repair stale documentation pointers to obsolete `CURRENT_MODEL.md`, `Theory.md` and `P4_SPEC.md`, future-FL framing and Section 16-current claims. Do not recreate those obsolete documents. Preserve history and clarify superseded status. Append a dated closure after the current `Record.md` tail. Documents must have no run-artifact path dependencies/references. Keep all plans in this file only and all math in dollar-delimited LaTeX.
7. Independent Astra audit: documentation follows Section 17; theory and procedures match source; counts and metrics agree; original strict failure and user acceptance remain distinct; gates/code/constants/outputs are unchanged; no unsupported physical interpretation or convergence guarantee; no run paths or long arrays. Return any defects for correction and re-audit before accepting the implementation.

Implementation status: the two P4 documents are consolidated, stale root pointers clarified, and a dated FL closure appended to `Record.md`. The scientific model-selection goal is closed by user acceptance. Astra/high independent documentation and source-read audit passed with no blocking findings. This audit covers documentation consistency with the inspected sources and recorded evidence; it does not certify code or a new execution. No scripts, tests, rerun or new experiment were executed.

## 16. Closing test: linear-in-A fitted laws (model FL), revision 4, 2026-11-16

**Historical contract, superseded by Section 17 for current documentation and default selection.** This section preserves the original experiment contract and strict verdict rules. The experiment has since been completed; statements below about a future run or absent FL results describe its pre-run state. The later user acceptance does not rewrite those historical gates.

**Historical pre-run implementation contract: Section 16, revision 4; then EDIT ONLY, including no selftest execution.** The completed document-extraction plan above is preserved. The former numbered plan is archived; its old Section 15 diagnostic and ring/mean criteria were withdrawn. This section governed the new script, including its six allowed gaze coefficients and fourteen exact-zero positions; the model document's then-broader twenty-coefficient proposal was not this experiment's contract. Frozen S2 / algorithm F and its historical results remained unchanged. FL results now exist and are consolidated in `P4_IMPLEMENTATION.md`; Section 17 governs current documentation/default selection.

**User-directed scope change, resolved rather than "To confirm with user".** The user's 2026-11-16 messages, "S2 passes the metric; why keep extending; linear A only; show me the performance", withdraw ring-coordinate, ring-shape and forward-mean criteria from this closing test. They authorize the gaze-focused F-versus-FL comparison below, not a relaxation of numerical gates or a claim that the former ring/mean goals passed. Accommodation errors are report-only: no A-error threshold has been set. No extra degree, alternative form, new traces, noise study or lens-model ladder is authorized. The current request permits plan/pointer edits and script implementation plus static audit only; the future command below is documentation, not permission to execute now.

### 16.1 Model, fitted laws and parameter counts

FL retains F's eight-point sensor mean, fixed A00 mean origin and normalization, camera inverse, two-radius measured size, tangent convention, five-term E4 size basis, and S2 runtime iteration. At each pass estimate gaze from the current A; hold that gaze fixed while building a size table at 0, 0.01, ..., 4 D; invert the finite strictly monotone table with F's endpoint clipping. Start at 0 D, stop when the absolute A update is <= 1e-6 D, allow at most 20 passes, fail a state that does not stop, and reevaluate final gaze and size residual at the returned A. Use the unchanged helpers for these steps.

For normalized sensor coordinates `(x,y)=m/scale`, retain six gaze positions: output x uses `x,x^3,xy^2`; output y uses `y,y^3,x^2y`. Their indices come from `tw.ALLOWED` and `tw.TERMS3`. Each allowed coefficient has the fitted law `c(A)=c0+c1*A`; the other fourteen positions have exactly zero intercept and slope. Do not impose x/y swap equality.

The other fitted laws are `gamma(A)=gamma0+gamma1*A` (five components) and `mu_cal(A)=u0+u1*A`. With `E(tau)=(tau_x^2,tau_y^2,tau_x^4,tau_x^2*tau_y^2,tau_y^4)`, the full size surface remains

`M(A,tau) = (u0+u1*A) * [1 + E(tau) dot (gamma0+gamma1*A)]`.

**Only the separately fitted laws are linear in A.** Their product generally contains the induced term `u1 * (E(tau) dot gamma1) * A^2`. That term is required by the model; it is neither a separately fitted quadratic law nor an independent stored coefficient. Do not flatten this product into a straight line. F remains the quadratic-trend baseline with piecewise-linear anchor sizes; `CLOSE: F` retains that baseline and does not certify linear-only success.

| Independent runtime values | FL | F |
|---|---:|---:|
| Gaze laws | 6 x 2 = 12 | 20 x 3 = 60 |
| Gamma laws | 5 x 2 = 10 | 5 x 3 = 15 |
| Size calibration | 2 | 5 |
| Law values | **24** | **80** |
| Including scale (2), mean origin (2), H0 and q0 | **30** | **86** |
| Including camera F, K1, K2 and centre (2) | **35** | **91** |

These count independent stored law/reference values, not every serialized scalar or the algebraic degree of M. Exact structural zeros, compatibility padding, metadata, fixed anchor abscissae, grid/procedure constants and redundant derived camera kappas are not independent values. The `(2,10,2)` FL gaze array may serialize structural zeros without changing the count.

### 16.2 Data preparation and calibration

Use unmodified `p4_accom_test` (`ac`), `p4_final` (`fin`) and `p4_termwise_test` (`tw`). Call the same data path as F: `ac.arguments(['--stages','S2'])`, `ac.new_report`, `ac.prepare_all` with `ac.five.load_shard`, `ac.shared_preparation(..., stage_s1=True)`, and `ac.size_calibration`. Use this script's progress callback. Validate all 21 published shards before scoring: A00 through A20, A = 0.2 times shard index, 10,201 states each and 214,221 total. Preserve every manifest, hash, schema, geometry, state-order, split, camera-inverse and numerical gate. No substitute data or prior output reads.

For every state, prepare `state['groups']['field_lt12'] = np.logical_or.reduce([state['groups'][name] for name in ac.gp.BIN_NAMES[:3]])`, exactly as `p4_final.py` does, before any surface diagnostic can access it. Compute `mu_raw` by `ac.two_radius_projection(state['Zshape'], cal['H0'], cal['q0'])`; preserve projection checks and the unchanged fixed H0/q0.

Only A00, A05, A10, A15, A20 (A = 0,1,2,3,4 D), training-gaze fits and anchor zero-gaze references calibrate F/FL. Shared preparation does incidental own-A fits and projection/data checks on non-anchor shards: these are allowed, but **no non-anchor fitted quantity calibrates either runtime model**. Do not inaccurately promise that no non-anchor fit is performed.

F reproduction uses `ac.coefficient_model(states,args)` and `ac.s1_fit_surface(states,cal,args,'E4')` unchanged. Its quadratic fits are confined to the F branch.

FL calibration, in this order:

1. **Gaze line.** Copy five training-only `state['fit']['coef']` arrays (5 x 10 x 2), record each anchor's largest absolute dropped normalized coefficient in deg, then set the fourteen forbidden positions to exact zero. Fit the resulting 5 x 20 target matrix with `ac.linear_fit` on the 5 x 2 design `[1,A]`. Reshape to `(2,10,2)`, assert forbidden intercepts and slopes are exactly zero, and supply scale, bounds `(0.,4.)` and a truthful degree-1 record. `ac.evaluate_gaze` handles this shape unmodified.
2. **Size line.** Fit the five measured anchor `cal['mu'][cal['anchors']]` values on that same `[1,A]` design using `ac.linear_fit`, obtaining `u0,u1`. Build a separate `cal_FL`: replace its entire 21-value `mu` array with `u0+u1*cal['A']` and rebuild the consumed/reported runtime size metadata to describe this fitted line. The unmodified table/residual helpers interpolate its five anchor samples; interpolation of samples of one line evaluates that line. Retain the measured anchor values separately as calibration observations, not runtime ordinates. Do not copy contradictory piecewise-measured or quadratic metadata into FL's saved record.
3. **Gamma anchor fits, then line.** Do **not** call `ac.s1_fit_surface` for FL: it unconditionally fits `[1,A,A^2]`. Instead reproduce only its existing anchor fit: for each of the five anchors, compute true-gaze tangents, target `state['mu_raw']/cal_FL['mu'][index]-1`, and call the unchanged `ac.linear_fit(ac.s1_size_design(tau[train],'E4'), target[train], context)`. These five 5-component vectors form a 5 x 5 target matrix. Fit them directly on the 5 x 2 `[1,A]` design using `ac.linear_fit`, with rank 2 and a linear-only trend record. Retain each anchor's fit record and coefficients. Any diagnostic using `ac.s1_diagnostic` has `field_lt12` available from preparation.
4. Build the FL surface with E4 basis, `gamma_A00` from the first anchor fit, grid `np.arange(401)*.01`, and truthful anchor/trend records. Store the actual gamma law as `(2,5)`. Append a third **exact zero** row only in the runtime compatibility view, since unmodified `ac.s1_surface_table` and `fin.size_value` index three rows. No quadratic fit is performed for FL and no padding row is counted as an independent value. Metadata must identify the genuine rank-2, two-column fit, not a copied quadratic fit.

All fits use unchanged full-rank and column-scaled condition <= 1e8 gates, with the existing rcond; report ordinary cubic and scaled-design checks separately. Fit failure stops with exit 3: no lower rank, refit on fewer anchors or fallback constants. Changing mu_cal affects both table generation **and** final size-residual evaluation, so FL coverage, clipping and stopping statistics must be measured anew; none can be copied from F.

### 16.3 Scoring, reproduction and verdict

For F and FL use `fin.run_loop(state,cal_row,model_row,surface_row)` with its `law` argument left None, then `fin.score` with F's same view and `Rc=1000*sqrt(2)*H0*(1+2*q0)`. Never call the unrelated LF calibration or runtime. Score every state. Gaze error is the two-axis norm in degrees times 60 (arcmin); A error is absolute estimated-minus-true A in D. Any failed/nonfinite state remains in the denominator and makes its group RMS and maximum infinite. Pool state arrays or exact sums of squares/counts, never rounded shard RMS or maxima.

Produce 42 shard rows (F/FL x 21), explicitly marking `anchor`/`nonanchor`, and 12 pool rows (F/FL x A00, A01-A19, A20, anchors, nonanchors, all21). Pool sizes are 10,201; 193,819; 10,201; 51,005; 163,216; 214,221. Every row reports total/valid/failed/coverage, failures by reason, clipped count, gaze RMS/max with location, A RMS/max with location, passes median/max, and counts of valid states with gaze error >1 and >0.1 arcmin. Failed states are separately counted, never silently treated as meeting a threshold. Report FL-minus-F comparisons for the pooled readouts. For an infinite metric report the first failure context, not a finite-only maximum disguised as the full maximum.

| Check | Requirement |
|---|---|
| L0 | In-process selftest, all data/preparation/calibration gates and F reproduction pass; 100% F coverage over 214,221 states |
| F gaze reproduction | A00/interior/A20 maxima 0.05193063 / 0.12489784 / 0.19039821 arcmin (`fin.AUDITED`); overall max 0.1903982096; interior RMS 0.01926065 arcmin; absolute tolerance 1e-6 **arcmin** for these gaze metrics |
| F additional cross-checks | Interior A RMS/max 0.00197210 / 0.03299789 D, compare to their quoted 8-decimal precision (absolute tolerance 1e-8 **D**); all21 passes median/max exactly 6 / 12; report these checks separately from gaze tolerance |
| L1 | FL valid 214,221 / 214,221 (100%); preserve failure denominators and report clips |
| L2 | FL all21 maximum gaze error strictly <1 arcmin (exactly 1 fails) |
| Readouts only | FL A RMS/max, gaze RMS and >0.1 counts, passes, clips and size-equation residuals; no A accuracy threshold or ring/mean criterion |

The A cross-check above verifies **F reproduction**, not scientific acceptance of FL accommodation accuracy. Supplied F precision is a reference for the future run, not newly measured evidence. If reproduction fails, investigate; do not weaken its tolerance or infer FL failure from an invalid baseline.

Emit exactly one final verdict line: `CLOSE: FL` if L0/L1/L2 pass; `CLOSE: F` if L0 passes and L1 or L2 fails; `NOT CLOSED: INVALID RUN` if L0 fails. `CLOSE: F` keeps the quadratic 80-law-value baseline and explicitly means FL did not satisfy this contract. No additional form is tried. Completed scientific FL coverage/gaze failure can still close on valid F; a calibration/numerical failure that prevents a valid comparison is an invalid run.

Every output labels the experiment **noise-free design traces; ground-truth gaze used for calibration targets and scoring only; development data, non-anchor A previously inspected, not fresh validation**. Coverage applies to A=0..4 D, field +/-12.5 deg and the specified endpoint clip rule. No noise, real-eye, off-grid, contraction or unique-fixed-point claim follows.

### 16.4 Implementation and output contract

Add only `Script/p4_linA.py`, **fewer than 200 physical lines including blanks/comments**; use small readable functions and unmodified helpers. Set `sys.dont_write_bytecode=True` before project imports. Edit no existing helper/script; do not monkeypatch imported globals, use another script's main/real_run/writer/log/guard, or generate companion code. Report a specific conflict rather than silently expand scope if the line budget or helper reuse is impossible.

Only writes are `summary.txt`, `constants.json`, `progress.log` in the dedicated script output folder, protected by a local guard equivalent to `fin.guard` for redirected/reparse/symlink/hardlink targets. No prior-result reads, per-state arrays, maps, plots, NPZ or extra output files. Documents are self-contained and never require run artifacts as inputs. The folder path is omitted in this documentation consolidation; the original script/output policy is unchanged.

`summary.txt` contains labels, hash, settings/units, all 24 independent FL law values, six reference constants, fit rank/condition checks, dropped-coefficient readouts, the shard/pool tables and comparisons, reproduction checks, L0/L1/L2, independent stored counts (FL 24/30/35; F 80/86/91), status and the single final verdict. Record per-shard maxima/locations and both threshold counts. `constants.json` saves full-precision `(2,10,2)` gaze law with structural zeros, `(2,5)` gamma law, two mu constants, terms/axes, fixed scale/origin/H0/q0/camera, anchors/grid/loop settings, fit records, script hash and run status. Any runtime padding is identified as structural, not an estimated quadratic law. Serialize with `fin.plain` and `allow_nan=False`. Progress records selftest, loading/preparation, calibration, each model/shard and terminal status.

Implement `--selftest` with synthetic arrays only, and invoke the same selftest in process before any production trace I/O. Do not execute it in this edit-only task. It must exercise production helpers and establish:

1. Linear fits recover six synthetic intercepts/slopes and off-anchor gaze to 1e-12; nonzero forbidden anchor entries are masked and their fitted trends remain exactly zero. Fit design is rank 2 `[1,A]`, not `[1,A,A^2]`, with no fitted quadratic result. Include a non-collinear anchor perturbation leaving a residual rather than fitting it with an added degree.
2. Both `ac.s1_surface_table` and `fin.size_value` match the full product `(u0+u1*A)*(1+E dot (gamma0+gamma1*A))` to 1e-12 off anchor, using a case with a nonzero induced A^2 product. Verify compatibility padding is zero and saved/fit records remain two-row linear laws.
3. Synthetic exact-FL states, including non-anchor A, recover at 100% coverage with A within 1e-5 D under `fin.run_loop`; an out-of-range size clips and is counted; a nonfinite state fails and makes full-denominator RMS/max infinite. Include pooled bookkeeping and threshold counts.
4. Stored independent counts are 12+10+2=24, then 30/35; all three verdict branches behave correctly, including max exactly 1 arcmin failing L2. Unchanged rank-deficient-fit rejection remains effective.

Exit codes: 0 completed comparison (either closing verdict), 2 missing/invalid data, 3 numerical/preparation/calibration/reproduction failure, 4 selftest failure (as in the original Section 16/final-script convention). Preserve the distinction between a state failure measured by L1 and a global run-invalid failure. A stopped run never reports completion.

Future production command, documented only, after independent static audit and subsequent execution authorization:

```powershell
& 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe' Script\p4_linA.py
```

The future runner captures `$LASTEXITCODE` in the same shell. No script, selftest, experiment, build or import-based check is executed in the present task.

### 16.5 Audit and later documentation

Sequence now: gpt-6-astra/high plan/pointer audit; gpt-6.1-sol/xhigh edit-only implementation against revision 4; gpt-6-astra/high independent static audit. Return defects for implementation and re-audit. Static audit checks the FL/F boundary, field_lt12 preparation, anchor-only calibration despite incidental non-anchor checks, linear-only FL fit records, exact forbidden zeros, induced product, mu line in both helper paths, gates, denominator/threshold/pooling correctness, three guarded outputs, <200 lines and no imported-helper edits. No execution is part of this request.

After later execution authorization: gpt-6-luna/low one run, then gpt-6-astra/high results audit. The prescribed Haiku 5.5 build/run-only sandbox fallback is unavailable here; report an actual block instead of silently substituting a model. Native background agents preserve the prescribed role/model/effort because the documented slash-command dispatcher is unavailable in this session.

Only after a valid results audit, append a self-contained closing record to `Record.md` with copied constants, equations, F/FL pooled/per-shard metrics, coverage/failures/clips/passes, maxima locations, thresholds, counts, caveats, script hash and audit/verdict. Do not fabricate FL results or overwrite concurrently rewritten model documentation. `CLOSE: F` must explicitly retain F's quadratic baseline. A physical forward lens model, ring and mean goals remain outside this closure.

Historical pointer handling for this experiment: `Record.md` then pointed to Section 16 revision 4. The absent root `P4_SPEC.md` was not recreated; its archived copy remained historical. The conditional old `P4_MODEL.md` status-line edit did not apply to the concurrently rewritten document. Current pointers now use Section 17, accepted FL theory in `P4_MODEL.md`, and procedures/constants/evidence in `P4_IMPLEMENTATION.md`. The obsolete root `CURRENT_MODEL.md`, `Theory.md` and `P4_SPEC.md` remain absent and are not prerequisites.
