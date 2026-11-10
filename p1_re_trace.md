# P1 ray-trace re-run: handoff

Date: 2026-11-09 (updated). Project: `C:\Users\rueijrwu\OneDrive\Project\DistortionTracking`.

## 1. Goal

Trace the P1 rays through `Lens\p1_ME_no_DM.seq`, and get per gaze state and per ray:
- the cornea position (S4) in the global frame: X, Y, Z;
- the stop position (S8) in the global frame: X, Y, Z;
- the direction cosines L, M, N after S4 and after S8, in the global frame;
- the sensor (SI) position in the global frame: X, Y, Z. The full-detail retrace now includes it.

The data feeds the camera-side P1 model. The model is not fitted yet.

## 2. Working rules (from the user)

- The coordinator only coordinates. The advisor plans and audits. Code is written by the code-writing agent (gpt-6.1-sol xhigh). Read-only checks and runs go to a mechanical agent (gpt-6-luna low). Code is not written by the coordinator.
- If a run fails in the Codex sandbox, rerun it with Haiku 5.5 at low effort.
- CODE V runs go outside the sandbox. The sandbox cannot run CODE V.
- Use the absolute Python `C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe` for every Python run.
- Keep every read and write inside the project. Do not use the scratchpad for project files.
- No commits. No edits to existing files without a plan.
- User reports use ASD-STE100.

## 3. Facts established

**CODE V start (COM).**
- `win32com.client.Dispatch("CodeV.Application")`, then `StartCodeV()`, then `StopCodeV()` and `pythoncom.CoUninitialize()`.
- COM starts `cvgui` with no main window and no visible window, 10 seconds after launch, in one snapshot. Headless is confirmed for that snapshot only.
- Launch and seq load work outside the sandbox. `Script\p1_cv_launch_check.py` ran to `StopCodeV: done`. `y` at S8 was about 8e-25 (origin).
- The sandbox run of the launch check stalled in CODE V startup and was stopped. Sandbox support is unknown.

**Lens.**
- `Lens\p1_ME_no_DM.seq` (SHA-256 `1d0c27fb…4dbe`): the fold mirror is removed, the cornea radius sign is flipped (`CorneaF` = -7.77 mm), and the return `Cornea_ENT_D` (S5) returns to the `RC` frame.
- `GLO S8 0.0 0.0 0.0` is in the file, so the global origin is at the stop (S8).
- Gaze ADE and BDE are set on `RC` (`ADE S"RC" <v>`, `BDE S"RC" <v>`). The current native-source retrace changes only RC ADE/BDE and leaves `SRC_ROT` ADE/BDE/CDE at their loaded 0° values. Earlier historical runs set source ADE to −20° in memory. The lens file itself sets no ADE, BDE, or CDE anywhere.
- Fields: XAN and YAN at ±20°, nine fields.
- Paraxial: EFL 78.4 mm. The reflex plane does not image onto the sensor (image 2.76 mm past S45; sensor at 75.02 mm). β must be a free parameter.
- **Sensor centre (user-supplied): fixed at (0, 0, 45.564056516201) mm in the global frame.** It does not move with gaze.
- Thickness sums (S4 to S8) disagree between the two lenses. Codex's sums include negative thicknesses and do not separate the fold-mirror path. The 85 µm figure is not confirmed. Not verified.

**Query syntax.**
- Help item form: `X | Y | Z | L | M | N | AOI ... [Sk] [Zn] [Fm] [Wj] [Gj] [Ri]`. Qualifier order does not matter (user).
- If the ray qualifier `[Ri]` is omitted, the value is taken from the last `RSI` or `SIN` ray traced.
- `G` (global) applies to X, Y, Z, and to L, M, N. Use `G8` on every read. `g8` is the surface number of the global reference surface (S8, the stop).
- `L`, `M`, `N` are optical direction cosines: cosines of the angles to the axes, multiplied by the index. In air, `L² + M² + N² = 1`.
- `AOI` is local. Do not use it.
- User-specified sensor read form: `z si r1 g8 f1`. The native chief-ray probe confirmed this qualified form and the unqualified current-ray `si g8` form return the same global SI position. Qualifier order does not matter (user).
- Surface `si` (image) is used in the old collector as `(x si)`.

**Probe, single angle ADE 0 / BDE 0 (passed, after the fix).**
- Macro: `Script\cv_macro\p1_probe_one_ray.seq`. Reads in the form `(x s4 g8)` and so on, with the old collector's field reset. Run headless through COM, outside the sandbox, at 14:05:08.
- Output (order: st, x4, y4, z4, l4, m4, n4, x8, y8, z8, l8, m8, n8):
  - S4: X = 0, Y = 0, Z = −334.79282 mm; L = 0, M = 0, N = −1.
  - S8: X = 0, Y = 0, Z = 0; L = 0, M = 0, N = −1.
- Checks: the S4 Z matches the thickness sum from the cornea to the stop (166.8669 + 167.9259 = 334.7928 mm, agreement to about 1e-9 mm). The stop is at the origin. The direction is unit length.
- Limits: one state (zero gaze), one ray. Gaze dependence is not tested.
- The earlier `R1 F1` probe (13:52:52) gave zeros and N = −1 with no field reset. It is not evidence of a correct trace.

**Old collector (`Script\collect_p1_biaxial_macro.py`, historical reference only).**
- Its historical per-state procedure set ADE/BDE on `RC`, set source ADE on `SRC_ROT` to −20°, changed `XAN`/`YAN`, converted field-2 `xob`/`yob` to surface 0, then traced centre, near-axis, and nine grid rays. This is not the current RC-only procedure.
- One CODE V session per run. The P1 collector runs with one worker.

**Prior 25-column 251 × 251 retrace (historical baseline).**
- Run `20261109T201039Z_e76ad3b8`: 63,001 states, 567,009 rays, 253 chunks, one CODE V session.
- ADE and BDE each from −12.5° to +12.5° in 0.1° steps. Nine field rays per state, pupil input (0, 0). Source ADE set to −20° in memory.
- Failed rays: 0. RC readback errors: 0.
- Maximum S8 XY radius: 3.2e-12 mm. LMN norm error: 5.8e-15.
- Stop θx range: ±0.4520°. Stop θy range: −0.6698° to +0.2197°.
- Files: `data\p1_retrace_full_251x251\20261109T201039Z_e76ad3b8\p1_retrace_251x251.csv` (189 MB, hash not obtained); `p1_retrace_251x251.pkl` (113 MB; SHA-256 `2456c0a2…a561`); `manifest.json`; `report.md`.
- The pickle named by the user, `data\distortion_grid\p1_full_101_125.pkl`, has the same size and hash as the run's pickle.
- **CSV columns (25):** `state_index`, `ray_index`, `ade_deg`, `bde_deg`, `field_x_normalized`, `field_y_normalized`, `rc_ade_readback_deg`, `rc_bde_readback_deg`, `rayrsi_status`, `x4`, `y4`, `z4`, `l4`, `m4`, `n4`, `x8`, `y8`, `z8`, `l8`, `m8`, `n8`, `theta_x_deg`, `theta_y_deg`, `geometric_theta_x_deg`, `geometric_theta_y_deg`.
- **No sensor (SI) columns in this historical baseline.** Use the full-detail run below for sensor positions.
- Field index map: 0=(−1,1), 1=(0,1), 2=(1,1), 3=(−1,0), 4=(0,0), 5=(1,0), 6=(−1,−1), 7=(0,−1), 8=(1,−1).

**Sensor query probe (passed).**
- Run `20261109T210233Z_3f8d3c9f` at RC ADE/BDE 0° and source ADE −20° traced one native `rsi r1 f1` ray. Qualified `(x/y/z si r1 g8 f1)` and unqualified `(x/y/z si g8)` agreed within 1e-9 mm.
- The ray hit SI at global (0, −1.697196930606539, 45.56405651620101) mm. Its Z agrees with the user-supplied fixed sensor plane 45.564056516201 mm. The sensor centre (0, 0, Z) is the plane origin, not necessarily a ray intersection.
- Results: `data\p1_retrace_sensor_probe\20261109T210233Z_3f8d3c9f\result.json` and `report.md`.

**Historical source−20° full-detail retrace (superseded for the current intent).**
- The 5 × 5 gate run `20261109T210649Z_844e7154` used RC ADE/BDE −12.5°, −6.25°, 0°, 6.25°, 12.5° and nine field rays per state: 225/225 valid. SI Z matched the fixed plane exactly. All 81 keys overlapping the historical 25-column full run matched its coordinate, direction, and angle columns exactly. Results: `data\p1_retrace_detail_5x5\20261109T210649Z_844e7154\report.md`.
- Full run `20261109T210803Z_88259c44`: 251 × 251 RC ADE/BDE states from −12.5° to +12.5° in 0.1° steps, nine normalized field rays per state, 63,001 states and 567,009/567,009 valid rays. One CODE V session, 253 resumable chunks, zero ray failures and zero RC readback errors. Source ADE was −20° in memory; only ADE/BDE settings changed. Native field settings were unchanged.
- Each ray records global S4 XYZLMN, S8 XYZLMN, and SI XYZ. The three new CSV/pickle columns are exactly `xsi`, `ysi`, `zsi`, appended to the historical 25-column schema for 28 columns total. Per-ray SI reads are unqualified `(x si g8)`, `(y si g8)`, `(z si g8)` immediately after `rayrsi`; the qualified `R1 F1` form was used only in the native probe.
- SI Z was 45.564056516201 mm for every ray (maximum plane error 0). SI X ranged from −3.3549865086937 to +3.3549865086937 mm; SI Y ranged from −4.9722108680533 to +1.6299509662149 mm. The old 25 columns matched the historical full run on all 567,009 keys exactly. The final CSV and pickle were compared field by field across all rows and matched exactly.
- Full run files: `data\p1_retrace_full_detail_251x251\20261109T210803Z_88259c44\p1_retrace_detail_251x251.csv` (SHA-256 `752378cbe14326830906dd07256e0393ff99c7bca621fd47ffba150242c0eacf`), `p1_retrace_detail_251x251.pkl` (SHA-256 `61a89af4e30f7ba51fb8de4525bd62f0103ec34caabad5335c54f946229095dc`), `report.md`, and `manifest.json`.
- This pickle was previously copied to `data\distortion_grid\p1_retrace_detail_251x251.pkl`; that destination has since been atomically replaced with the validated native-source pickle below. The historical source−20° pickle remains in its original run folder. `data\distortion_grid\p1_full_101_125.pkl` is currently absent; it was not changed by the replacement.
- Production collector: `Script\p1_retrace_full_detail.py`. The run restored RC ADE/BDE to 0° and closed its own CODE V backend; pre-existing CODE V sessions were not touched.

**Current native-source full-detail retrace (use this dataset).**
- Explicit user correction: keep the lens's loaded `SRC_ROT` ADE/BDE/CDE at 0°. Only `ADE S"RC"` and `BDE S"RC"` may change optical settings. The collector uses the same nine normalized field rays, global S4/S8 XYZLMN reads, and global SI XYZ reads as the historical run.
- The 5 × 5 gate `20261109T221227Z_8748422b` passed with 25 states and 225 valid rays. At ADE/BDE 0°, center ray index 4 had S4/S8/SI XY effectively 0, stop angles effectively 0°, and SI Z = 45.564056516201 mm. Source ADE/BDE/CDE remained 0° before and after. Results: `data\p1_retrace_native_detail_5x5\20261109T221227Z_8748422b\report.md`.
- Full run `20261109T221348Z_1f596ed9` passed: 251 × 251 RC ADE/BDE states from −12.5° to +12.5° in 0.1° steps, 63,001 states, 567,009/567,009 valid rays, 28 columns, zero ray failures and RC readback errors. SI XYZ was finite for every ray; SI Z was 45.564056516201 mm throughout. All 81 keys shared with the new 5 × 5 gate matched exactly. CSV and pickle matched field by field across all rows. Source ADE/BDE/CDE remained 0° before and after, RC ADE/BDE were restored to 0°, and the run's CODE V backend exited.
- The run has 253 completed chunks and 256 attempts. Three earlier attempts stopped with `PermissionError(13, 'Access is denied')` before their chunk directories were created. All affected chunks succeeded on resume. The cause of those filesystem interruptions was not established. The retained manifest error text is historical; final status is `complete` and validation passed. The diagnostic resume wrapper and its provenance are retained in the run folder.
- Current files: `data\p1_retrace_native_detail_251x251\20261109T221348Z_1f596ed9\p1_retrace_detail_251x251.csv` (SHA-256 `77e123145ce106b71bd0abf4a08c950e2ba5a2cc46b372a336ec25bf2931f860`), `p1_retrace_detail_251x251.pkl` (SHA-256 `2249c93a9b8ad59e254f4941015fb9108b8a22459ab361320ad3cf1d64273736`), `report.md`, and `manifest.json`.
- The verified pickle was atomically copied to `data\distortion_grid\p1_retrace_detail_251x251.pkl` (127,010,718 bytes; SHA-256 `2249c93a9b8ad59e254f4941015fb9108b8a22459ab361320ad3cf1d64273736`). Collector: `Script\p1_retrace_native_source_detail.py`; diagnostic resume wrapper: `Script\p1_retrace_resume_mkdir.py`.

**Grids.**
- Stored P1 grid: ADE and BDE from −20 to +20 in 0.4° steps, 101 values per axis.
- The retrace grid (±12.5 in 0.1° steps) shares the stored states at multiples of 0.4° within ±12.4°.

## 4. Status

| Item | Status |
|---|---|
| Launch and seq load, headless, outside sandbox | Done (one snapshot) |
| Single-angle probe, ADE 0 / BDE 0, old-collector read form | **Passed** (values in §3) |
| Historical 25-column and source−20° full-detail retraces | **Done**; preserved in their run folders, superseded for the current intent |
| Sensor centre (0, 0, 45.564056516201) mm, fixed | Given by the user; SI Z verified by the probe and all 567,009 full-detail rays |
| Sensor-position read `z si r1 g8 f1` | **Passed** in the historical source−20° native chief-ray probe; qualified and current-ray forms agreed |
| Per-ray `(x/y/z si g8)` reads | **Done** in the full-detail collector |
| Native-source 5 × 5 detail gate | **Passed**; 225 valid rays, centered zero-gaze ray index 4 |
| Current native-source full-detail 251 × 251 retrace | **Done**; 567,009 valid rays, 28 columns, CSV/pickle equality verified; 81 shared keys matched the new 5 × 5 gate |
| Lens equivalence: p1_ME vs p1_ME_no_DM at the sensor | **Not verified** |
| Macro-PLUS help read | **Done.** Confirms item syntax, `[Gj]`, `RAYRSI`, and the omitted-ray rule. Does not confirm `si` with `G`, or qualifier order (user says order does not matter). |
| `Script\p1_lens_collect.py` (202 lines) | Historical unvalidated draft; not the production collector. `Script\p1_retrace_full_detail.py` produced the completed dataset. |
| Older 11 × 11 / 2-worker and 4-worker plans | Superseded by the passed 5 × 5 gate and single-worker full run. |
| Camera-side model fit | Not started. Complete sensor columns are now available. |
| Eye-side (mirror) check | Not started. Needs user approval. |

## 5. Next steps (in order)

1. Use the current native-source 28-column full-detail dataset, copied to `data\distortion_grid\p1_retrace_detail_251x251.pkl`, for the camera-side P1 model. The source−20° data remains historical. No further retrace is currently needed for the handoff's S4/S8/SI outputs.
2. Before claiming a sensor ground-truth match to the earlier `p1_ME.seq` B1 bar, decide which lens defines ground truth or test sensor equivalence between `p1_ME.seq` and `p1_ME_no_DM.seq`. That equivalence is still unverified.
3. The eye-side mirror check remains a separate step and has not been authorized or started.
4. Keep advisor audit before using new modeled numbers in user reports.

## 6. Open items and housekeeping

- **Codex route:** the advisor says `gpt-6.1-sol` may be reachable through `/codex:rescue --model gpt-6.1-sol --effort xhigh`. The earlier "Sonnet" answer came from a prompt that told the agent not to use tools. The luna-flagged jobs have not been confirmed to run on luna. Check the job log before relying on either.
- Codex status still lists two stale jobs as running (`task-mw9k3453-61s3c6`, `task-mw9k1d3t-3872nj`). No process is behind them.
- About 100 `node` and `node_repl` processes have built up since 11:31. Possibly leaked Codex companions. Not killed; ask first.
- `Script\p1_cv_launch_check.py` was written by the coordinator, which broke the rule. The code-writing agent should take it over or fold it in. Do not delete without asking.
- Untracked files: `Lens\p1_ME_no_DM.1.seq`, `Lens\p1_ME_no_DM.seq`, `Script\cv_macro\*`, `Script\p1_lens_collect.py`, `Script\p1_cv_launch_check.py`, `gen_py\`. `DETAIL_MODEL.md` is modified.
- `Script\cv_macro\p1_collect_biaxial_state0.seq` contains 1° placeholder field-of-view values. Replace them before use.
- The Codex sandbox cannot run CODE V.
