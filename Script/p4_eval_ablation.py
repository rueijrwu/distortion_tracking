"""Noise-free evaluation-only ablations of the frozen input-(b) paired readout.

Writes only data/p4_eval_ablation/*.npz and report.md. No baseline rerun.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

import p4_closed_loop_readout as base
from p4_unified_fullshape import UnifiedP4Field, ProfileAdapter
from p4_unified_center_motion import CentralMotion

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/p4_eval_ablation"
SAVED = ROOT / "data/p4_closed_loop_readout/closed_loop_readout.npz"
STEP1 = ROOT / "data/p4_phys/step1.npz"
REFERENCE_GAZE = np.array([.92, 1.77, 4.27])
REFERENCE_A = np.array([.028, .055, .185])
PARITY_TOL_ARCMIN = 1e-6
SLOPE = -.081595476571
TRUSTED_INITIAL_NUMERICAL_SCRIPT_SHA256 = "fb2e4289b1d7bece0e4f5ddcdc286fda25cecaf532db1ae2dfd65dd9fd3ac364"
ADE_TERMS = ((0, 0), (0, 1), (2, 0), (0, 2), (2, 1), (0, 3))
BDE_TERMS = ((1, 0), (1, 1), (3, 0), (1, 2))


class ZeroBias:
    def predict(self, gaze):
        return np.zeros_like(np.asarray(gaze, float))

    def jacobian(self, gaze):
        return np.zeros((len(gaze), 2, 2))


class ShapeMeanBias:
    """A0 mean of the existing frozen H; adds no physics or fitted values."""
    def __init__(self, sparse):
        self.sparse = sparse

    def predict(self, gaze):
        g = np.asarray(gaze)
        return self.sparse.evaluate(np.zeros(len(g)), g[:, 0], g[:, 1])["prediction"].mean(axis=1)

    def jacobian(self, gaze):
        g = np.asarray(gaze)
        result = self.sparse.evaluate(np.zeros(len(g)), g[:, 0], g[:, 1], state_jac=True)
        return result["state_jacobian"][..., 1:3].mean(axis=1)


def parity_fit(d, gaze):
    terms = base.powers(3)
    coefficients = np.zeros((len(terms), 2))
    conditions = []
    for axis, allowed in enumerate((ADE_TERMS, BDE_TERMS)):
        design = base.features(d, allowed)
        scale = np.linalg.norm(design, axis=0)
        co, _, rank, sv = np.linalg.lstsq(design / scale, gaze[:, axis], rcond=None)
        if rank != len(allowed):
            raise RuntimeError("Parity fit rank deficient")
        for ij, v in zip(allowed, co / scale):
            coefficients[terms.index(ij), axis] = v
        conditions.append(sv[0] / sv[-1])
    return {"terms": terms, "coefficients": coefficients, "condition": np.array(conditions)}


def predict(model, d, accommodation, gain=None):
    value = base.predict(model, d)
    if gain is not None:
        value *= 1 + (np.asarray(accommodation)[:, None] - 2.) * gain
    return value


def corrected_displacement(p4_mean, p1_mean, a, g, frozen, field, motion):
    return base.corrected_displacement(p4_mean, p1_mean, a, g, frozen, field, motion)


def solve(observed_p4, observed_p1_mean, model, frozen, field, motion, seed,
          *, gain=None, oracle_A=None, label="variant"):
    """Frozen solver mechanics; only the explicitly labelled oracle receives truth."""
    n = len(observed_p4)
    means = observed_p4.mean(axis=1)
    current = np.column_stack((seed["estimated_A_d"], np.zeros((n, 2))))
    if oracle_A is not None:
        current[:, 0] = oracle_A
    alive = seed["converged"] & seed["valid_map"] & np.isfinite(current).all(axis=1)
    converged = np.zeros(n, bool)
    invalid = ~alive.copy()
    consecutive = np.zeros(n, np.int16)
    iterations = np.zeros(n, np.int16)
    alpha = np.ones(n)
    previous_norm = np.full(n, np.inf)
    previous_state = np.full((n, 3), np.nan)
    best_norm = np.full(n, np.inf)
    best_state = current.copy()
    growth_count = np.zeros(n, np.int16)
    cycle_count = np.zeros(n, np.int16)
    profile_fallbacks = np.zeros(n, np.int16)
    bound_pressure = np.zeros((n, 2), bool)
    adapter = ProfileAdapter(field)
    started = time.perf_counter()
    for iteration in range(1, base.MAX_ITER + 1):
        rows = np.flatnonzero(alive)
        if not len(rows):
            break
        st = current[rows]
        d, _, _, _, valid = corrected_displacement(means[rows], observed_p1_mean[rows],
                                                  st[:, 0], st[:, 1:], frozen, field, motion)
        raw = predict(model, d, st[:, 0], gain)
        bounded = np.clip(raw, -base.BOUND, base.BOUND)
        bound_pressure[rows] |= np.abs(raw) > base.BOUND
        valid &= np.isfinite(raw).all(axis=1)
        proposal = np.full_like(st, np.nan)
        proposal[:, 1:] = bounded
        if oracle_A is not None:
            proposal[:, 0] = oracle_A[rows]
        elif valid.any():
            prof, _ = base.profile(adapter, observed_p4[rows[valid]], bounded[valid])
            proposal[valid, 0] = prof["estimated_A_d"]
            profile_fallbacks[rows[valid]] += prof["reference401_fallback"]
            ok = valid.copy()
            ok[valid] = prof["converged"] & prof["valid_map"]
            valid = ok
        delta = proposal - st
        norm = np.linalg.norm(delta / base.STATE_SCALE, axis=1)
        valid &= np.isfinite(norm)
        iterations[rows] = iteration
        invalid[rows[~valid]] = True
        alive[rows[~valid]] = False
        improved = valid & (norm < best_norm[rows])
        take = rows[improved]
        best_norm[take] = norm[improved]
        best_state[take] = st[improved]
        step_ok = valid & (np.abs(delta[:, 0]) <= base.A_TOL) & (
            np.linalg.norm(delta[:, 1:], axis=1) <= base.GAZE_TOL)
        consecutive[rows] = np.where(step_ok, consecutive[rows] + 1, 0)
        just = valid & (consecutive[rows] >= 2)
        converged[rows[just]] = True
        best_state[rows[just]] = st[just]
        best_norm[rows[just]] = norm[just]
        growth = valid & np.isfinite(previous_norm[rows]) & (norm > 1.2 * previous_norm[rows])
        cycle = (valid & np.isfinite(previous_state[rows]).all(axis=1)
                 & (np.linalg.norm((proposal - previous_state[rows]) / base.STATE_SCALE, axis=1) <= 1e-5)
                 & (norm > 1e-5))
        growth_count[rows[growth]] += 1
        cycle_count[rows[cycle]] += 1
        damp = rows[growth | cycle]
        alpha[damp] = np.maximum(alpha[damp] / 2., 1. / 32.)
        previous_norm[rows[valid]] = norm[valid]
        previous_state[rows[valid]] = st[valid]
        alive[rows[just]] = False
        updating = valid & ~just
        current[rows[updating]] += alpha[rows[updating], None] * delta[updating]
        print(f"{label}: iteration {iteration}; active {len(rows)}; converged {just.sum()}; remaining {alive.sum()}", flush=True)
    d, c1, c4, c4ref, final_valid = corrected_displacement(means, observed_p1_mean,
        best_state[:, 0], best_state[:, 1:], frozen, field, motion)
    raw = predict(model, d, best_state[:, 0], gain)
    if oracle_A is None:
        final_profile, _ = base.profile(adapter, observed_p4, np.clip(raw, -base.BOUND, base.BOUND))
        delta_a = final_profile["estimated_A_d"] - best_state[:, 0]
        final_profile_valid = final_profile["converged"] & final_profile["valid_map"]
        shape_rms = final_profile["fit_vertex_rms_um"]
    else:
        delta_a = oracle_A - best_state[:, 0]
        final_profile_valid = final_valid.copy()
        prediction = field.evaluate(oracle_A, np.clip(raw, -base.BOUND, base.BOUND))
        residual = (observed_p4 - means[:, None]) - (prediction - prediction.mean(axis=1)[:, None])
        shape_rms = np.sqrt(np.mean(np.sum(residual * residual, axis=2), axis=1)) * 1000
    delta_g = np.linalg.norm(np.clip(raw, -base.BOUND, base.BOUND) - best_state[:, 1:], axis=1)
    if np.any(converged & ((np.abs(delta_a) > base.A_TOL) | (delta_g > base.GAZE_TOL))):
        raise RuntimeError(f"{label}: final residual contradicts convergence")
    return dict(estimated_state=best_state, converged=converged, invalid=invalid,
                valid=final_valid & np.isfinite(best_norm), iterations=iterations, alpha=alpha,
                growth_events=growth_count, cycle_events=cycle_count,
                profile_fallbacks=profile_fallbacks, bound_pressure=bound_pressure,
                d_mm=d, c1_mm=c1, c4_mm=c4, c4_A0_mm=c4ref, raw_map_gaze_deg=raw,
                final_delta_A_d=delta_a, final_delta_gaze_deg=delta_g,
                final_profile_valid=final_profile_valid, shape_fit_vertex_rms_um=shape_rms,
                wall_seconds=np.array(time.perf_counter() - started))


def stats(x):
    x = np.asarray(x)
    if not len(x):
        return np.full(3, np.nan)
    return np.array([np.sqrt(np.mean(x * x)), np.percentile(np.abs(x), 95), np.max(np.abs(x))])


def fmt(x):
    return " / ".join(f"{v:.7g}" for v in np.asarray(x))


def deviations(result, truth):
    return base.errors(result, truth)


def dy_even(model, d):
    reflected = d.copy()
    reflected[:, 1] = 2 * -1.697 - d[:, 1]
    even = .5 * (base.predict(model, d) + base.predict(model, reflected)) * 60
    plane = d.copy()
    plane[:, 1] = -1.697
    curvature = even - base.predict(model, plane) * 60
    return even, curvature


def pf12_check(frozen, field):
    """Compare the saved Step-1 candidate with unconverted/converted teachers."""
    with np.load(STEP1, allow_pickle=False) as z:
        states = z["P4_states"].copy()
        candidate = z["P4_candidate_raw_H"].copy()
        saved_teacher = z["P4_teacher_raw_H"].copy()
        trace = z["P4_trace_raw_H"].copy()
        focal = z["PF_focal_mm"].copy()
    old = UnifiedP4Field(frozen, field.extension.copy())
    old.base12 = np.asarray(frozen.p4_shape.coefficients).copy()
    g = states[:, 1:]
    before = old.sparse_model().evaluate(states[:, 0], g[:, 0], g[:, 1])["prediction"]
    after = field.sparse_model().evaluate(states[:, 0], g[:, 0], g[:, 1])["prediction"]
    op = np.max(np.abs(g), axis=1) <= 12.4 + 1e-10
    def metrics(pred, mask, target=candidate):
        radial = np.linalg.norm((target - pred)[mask], axis=2).ravel() * 1000
        return stats(radial)
    return dict(before_full=metrics(before, np.ones(len(states), bool)),
                saved_physical_focal_mm=focal,
                after_full=metrics(after, np.ones(len(states), bool)),
                before_operating=metrics(before, op), after_operating=metrics(after, op),
                teacher_trace_before_full=metrics(before, np.ones(len(states), bool), trace),
                teacher_trace_after_full=metrics(after, np.ones(len(states), bool), trace),
                teacher_trace_before_operating=metrics(before, op, trace),
                teacher_trace_after_operating=metrics(after, op, trace),
                saved_teacher_vs_after_max_um=np.array(np.max(np.linalg.norm(saved_teacher - after, axis=2)) * 1000),
                saved_teacher_vs_before_max_um=np.array(np.max(np.linalg.norm(saved_teacher - before, axis=2)) * 1000))


def make_report(saved, model, results, designs, parity, gain, pf, hashes, elapsed):
    truth = saved["truth_state_scoring_only"]
    p = saved["primary_mask"]
    ids, rows = saved["source_indices"], saved["source_subset_rows"]
    reference = results["reference_saved"]
    ge, ae = deviations(reference, truth)
    reference_gaze = stats(ge[p])
    reference_A = stats(ae[p])
    text = ["# P4 evaluation ablation", "", "Noise-free, input (b) only. Evaluation items (a)-(e); no baseline rerun, distortion refit, CODE V run, sensor-noise simulation, or additional physics.", "",
        "## Conventions and design", "",
        "All non-oracle variants retain the same 75-pose teacher-conditioned calibration, frozen 10-value parity readout, observed P4 nine-ray grids, observed P1 means, recorded nominal coordinates, solver bounds and damping. One component is replaced at a time; the readout and other components are not refitted. The saved observed-only A seed at zero gaze is reused. Changed field components are used consistently in both centre correction and the mean-centred A profiler. Truth enters held-out scoring only, except the explicitly isolated true-A oracle.", "",
        "The unrestricted 20-value saved readout was reported as 67 total values; parity reduction gives 57 = 18 B3/B4 + 12 PF12 + 15 A terms + 2 M1 + 10 readout. The requested 57-value reference is quoted as gaze 0.92 / 1.77 / 4.27 arcmin and A 0.028 / 0.055 / 0.185 D. Its saved degree-3 numerical outputs are used for reference statistics. Earlier degree-1 and degree-2 gaze results are quoted only: 6.91 / 10.78 / 16.0 and 6.14 / 10.10 / 11.96 arcmin.", "",
        "Every gaze error is radial Euclidean error in degrees times 60; A uses signed error for RMS and absolute error for P95/max. Triples are RMS / P95 / max. All attempted primary states, including failed convergence, remain in metrics. There are 83,229 primary and 21 separately excluded diagnostic states, with no new exclusions. |gaze| means max(|ADE|,|BDE|), following the frozen report.", "",
        "| Row | Design |", "|---|---|"]
    for name, design in designs.items():
        text.append(f"| {name} | {design} |")
    text += ["| physical_B3 | Unavailable; not run. Step 1 provides sampled P1 nine-ray means, but no continuous derived P1 field law suitable for arbitrary estimated gaze. Inventing or fitting such a law would add physics beyond the permitted scope. |", "",
        "The M1 physical substitution uses the specified scalar Step-1 scale about the saved pivot, not the separate Step-1 gaze-dependent lever-displacement candidate. B4 physical substitution means the natural mean of frozen raw H at A=0, with the existing accommodation-dependent mean shift retained. Field-shape identity means the 12 PF12 distortion coefficients are zero, with the reference ray lattice and the 15 A terms retained. A-term identity means all 15 extension values are zero. These choices preserve the frozen model decomposition.", "",
        "The `identity_A_terms` row removes all accommodation dependence from the relative-field shape used by the A profiler. Accommodation is therefore unidentifiable from that shape objective. The profile is flat across the bounded 0–4 D interval; deterministic tie handling selects the lower endpoint A=0. This boundary selection is a numerical tie rule, not an accommodation estimate.", "",
        "## (a) Closed-loop ablation and substitution", "",
        "Changes are measured against the actual saved reference, avoiding artefacts from the rounded quoted values.", "",
        "| Row | N | Gaze arcmin | A D | Gaze change vs saved reference | A change vs saved reference | Converged | Invalid |", "|---|---:|---|---|---|---|---:|---:|"]
    for name, result in results.items():
        g, a = deviations(result, truth)
        text.append(f"| {name} | {p.sum()} | {fmt(stats(g[p]))} | {fmt(stats(a[p]))} | {fmt(stats(g[p])-reference_gaze)} | {fmt(stats(a[p])-reference_A)} | {result['converged'][p].sum()} | {result['invalid'][p].sum()} |")
    text += ["| physical_B3 | - | unavailable | unavailable | not judged | not judged | - | - |", "",
        "Component decisions below use closed-loop gaze and A results, not micrometre discrepancies. A substitution is described as worse if any gaze summary exceeds the actual saved reference by more than 1e-6 arcmin; no hardware acceptance is inferred.", ""]
    for name, result in results.items():
        if name in ("reference_saved", "A_linear_gain"):
            continue
        g, a = deviations(result, truth)
        worse_g = np.any(stats(g[p]) > reference_gaze + 1e-6)
        worse_a = np.any(stats(a[p]) > reference_A + 1e-8)
        text.append(f"- {name}: gaze {'worsens' if worse_g else 'does not worsen'} and A {'worsens' if worse_a else 'does not worsen'} against the saved reference (tolerances 1e-6 arcmin and 1e-8 D); {'retain the frozen component' if worse_g or worse_a else 'no degradation in these summaries'}.")
    oracle_g, _ = deviations(results["oracle_true_A"], truth)
    text += ["", f"True-A tail split: gaze P95/max change from {reference_gaze[1]:.7g}/{reference_gaze[2]:.7g} to {stats(oracle_g[p])[1]:.7g}/{stats(oracle_g[p])[2]:.7g} arcmin. The remaining oracle tail is conditional centre-correction/readout error; the change isolates the effect of A estimation in this nonlinear loop and is not an additive variance decomposition."]
    text += ["", "## (b) Parity readout and dy diagnostic", "",
        "ADE uses exactly 1, dy, dx², dy², dx²dy, dy³; BDE uses dx, dxdy, dx³, dxdy². The two outputs are fitted independently on the 75 calibration poses with column-scaled least squares in physical d coordinates. No dy parity is imposed.", "",
        f"Maximum radial prediction difference from the saved 20-value map: calibration {parity['cal_max_arcmin']:.9g} arcmin; saved primary runtime d {parity['primary_max_arcmin']:.9g} arcmin; all saved runtime d {parity['all_max_arcmin']:.9g} arcmin. Declared tolerance is {PARITY_TOL_ARCMIN:g} arcmin; {'PASS' if parity['pass'] else 'FAIL'}. This is map equivalence on calibration and recorded runtime covariates, without rerunning the reference loop.", "",
        "| Physical monomial | ADE deg coefficient | BDE deg coefficient |", "|---|---:|---:|"]
    for ij, co in zip(model["terms"], model["coefficients"]):
        text.append(f"| dx^{ij[0]} dy^{ij[1]} | {co[0]:.14g} | {co[1]:.14g} |")
    text += ["", "Report-only dy-even diagnostic after d0=(0,-1.697 mm): u=dy-d0y, E(dx,u)=[P(dx,d0y+u)+P(dx,d0y-u)]/2. Reflection is a polynomial diagnostic and can leave the sampled support. BDE's dy-even part includes the desired horizontal gaze response, so it is not itself an error metric. The ADE dy-even part and ADE nonconstant curvature E-P(dx,d0y) quantify different offsets; neither is constrained in fitting.", "",
        "| Covariates | ADE dy-even arcmin | BDE dy-even arcmin | ADE dy-even curvature arcmin |", "|---|---|---|---|"]
    for label, d in (("75 calibration", saved["calibration_d_teacher_mm"]), ("saved primary runtime", saved["degree3_d_mm"][p])):
        even, curvature = dy_even(model, d)
        text.append(f"| {label} | {fmt(stats(even[:,0]))} | {fmt(stats(even[:,1]))} | {fmt(stats(curvature[:,0]))} |")
    text += ["", "## (c) PF12 conversion audit", "",
        "The `before` case places legacy `/12.4` coefficients directly on SparseModel's `/20` gaze basis, a mismatched representation. A valid `/12.4` model with legacy coefficients is mathematically equivalent to the converted `/20` model with c_new=c_old*(20/12.4)^(i+j); this coordinate conversion does not change the physical teacher law. Only the frozen-teacher comparator changes here: the Step-1 physical candidate and fitted focal value are held fixed, with no candidate refit.", "",
        "The identical saved Step-1 physical candidate is compared with teachers differing only in PF12 coefficient conversion c_new=c_old*(20/12.4)^(i+j). Its fitted focal value is held fixed: no new physical-candidate fit. Values below are radial per-ray deviations in µm (RMS / P95 / max).", "",
        "| Domain | Legacy coefficients incorrectly used on `/20` | Converted coefficients used on `/20` |", "|---|---|---|",
        f"| Full Step-1 domain | {fmt(pf['before_full'])} | {fmt(pf['after_full'])} |",
        f"| Operating max-axis gaze ≤12.4° | {fmt(pf['before_operating'])} | {fmt(pf['after_operating'])} |", "",
        "The coefficient conversion also repairs agreement of the frozen teacher itself with recorded raytrace data (same radial per-ray triples):", "",
        "| Teacher vs raytrace | Legacy coefficients incorrectly used on `/20` | Converted coefficients used on `/20` |", "|---|---|---|",
        f"| Full Step-1 domain | {fmt(pf['teacher_trace_before_full'])} | {fmt(pf['teacher_trace_after_full'])} |",
        f"| Operating axes ≤12.4° | {fmt(pf['teacher_trace_before_operating'])} | {fmt(pf['teacher_trace_after_operating'])} |", "",
        f"Saved Step-1 teacher vs recomputed converted teacher max radial difference {float(pf['saved_teacher_vs_after_max_um']):.9g} µm; vs unconverted teacher {float(pf['saved_teacher_vs_before_max_um']):.9g} µm. "
        "The saved Step-1 raw teacher already uses the handoff conversion. Thus the approximately 3961 µm Step-1 failure was not caused by a missing conversion; conversion does not rescue the physical candidate.", "",
        f"The held-fixed Step-1 physical PF focal value is {float(pf['saved_physical_focal_mm']):.9g} mm, at its 1000 mm search upper bound; this limitation is inherited, with no candidate refit.", "",
        "## (d) A-linear readout gain", "",
        "Exactly one extra coefficient per gaze axis: g_j=P_j(d)*[1+k_j*(A-2 D)]. The anchor A=2 D matches the mean of the balanced 0/2/4 D calibration. Each k is fitted by scalar least squares to calibration gaze residuals with P fixed; held-out truth does not enter the fit. At runtime the current estimated A enters the gain. The profiler, corrections, bounds and solver are unchanged. The evaluation truth grid has 21 A planes at 0.2 D spacing and contains no A=1.5 D states; A=1.5 performance is unavailable. A=1.4 and A=1.6 are reported as observed bracketing proxies, with no interpolation to 1.5.", "",
        f"Gain [k_ADE,k_BDE] in D^-1: `{gain.tolist()}`. These are two added readout values (59 total).", "",
        "| True A | Gaze bin | N | Before gaze arcmin | After gaze arcmin | Before A D | After A D |", "|---:|---|---:|---|---|---|---|"]
    mag = np.max(np.abs(truth[:,1:]), axis=1)
    gain_ge, gain_ae = deviations(results["A_linear_gain"], truth)
    for a in (0., 1.4, 1.6, 4.):
        for lo, hi in ((0,4),(4,8)):
            mask = p & np.isclose(truth[:,0], a, rtol=0, atol=1e-10) & (mag>=lo) & (mag<hi)
            before, after = stats(ge[mask]), stats(gain_ge[mask])
            text.append(f"| {a:g} | [{lo},{hi}) | {mask.sum()} | {fmt(before)} | {fmt(after)} | {fmt(stats(ae[mask]))} | {fmt(stats(gain_ae[mask]))} |")
    text += ["| 1.5 | unavailable | 0 | unavailable | unavailable | unavailable | unavailable |", "",
        "Each observed A/bin cell is judged separately. In both gaze bins, the A=0 and A=4 endpoint RMS/P95 worsen with the gain. At A=1.4, all three gaze summaries worsen in both bins. At A=1.6, the low-gaze bin improves slightly, while all three summaries worsen in the 4–8° bin. This is mixed behavior at the observed bracketing proxies, not evidence that the A-shaped error pattern is reduced. Globally the gain worsens RMS and P95 (0.9163/1.7710 to 0.9427/1.8253 arcmin), while lowering max error (4.2713 to 4.1429 arcmin); it is not a recommended replacement.", ""]
    text += ["", "## (e) Reference gaps and clipping", "",
        f"Saved reference primary fraction within 1 arcmin (≤): {np.mean(ge[p]<=1):.9g} ({np.sum(ge[p]<=1)}/{p.sum()}); within 3 arcmin (≤): {np.mean(ge[p]<=3):.9g} ({np.sum(ge[p]<=3)}/{p.sum()}).", "",
        "Endpoint estimates are reported as search-bound contacts (the profiler compares exact endpoints), not proof of an unconstrained optimum beyond the interval. Numerical contact tolerance is 1e-7 D; exact contacts are also counted.", "",
        "| Row | A≈0 primary | A≈4 primary | Exact A=0 | Exact A=4 | True A=4 A error D |", "|---|---:|---:|---:|---:|---|"]
    for name, result in results.items():
        est = result["estimated_state"][:,0]
        aerr = est-truth[:,0]
        mask = p & np.isclose(truth[:,0],4,rtol=0,atol=1e-10)
        text.append(f"| {name} | {np.sum(p & np.isclose(est,0,atol=1e-7,rtol=0))} | {np.sum(p & np.isclose(est,4,atol=1e-7,rtol=0))} | {np.sum(p & (est==0))} | {np.sum(p & (est==4))} | {fmt(stats(aerr[mask]))} |")
    text += ["", "Reference true-A=4 D rows with P95=0 are clipping artefacts of the 4 D upper search bound, not perfect accommodation results. The aggregate A=4 row need not have P95=0. The true-A oracle's zero A error is imposed by design and is also not an estimator result.", ""]
    ref_est_A = reference["estimated_state"][:,0]
    for lo,hi,label in ((0,4,"[0,4)"),(4,8,"[4,8)"),(8,12,"[8,12)"),(12,12.400001,"[12,12.4]")):
        mask=p & np.isclose(truth[:,0],4,atol=1e-10,rtol=0) & (mag>=lo) & (mag<hi)
        p95=stats(ae[mask])[1]
        contacts=np.sum(mask & np.isclose(ref_est_A,4,atol=1e-7,rtol=0))
        text.append(f"- Reference true A=4, gaze bin {label}: A P95={p95:.9g} D; upper-bound contacts {contacts}/{mask.sum()}. {'The zero P95 is boundary censoring, not perfect estimation.' if p95==0 else 'P95 is nonzero in this bin.'}")
    text += ["",
        "## |gaze| × A tables", ""]
    for name, result in results.items():
        g, aerr = deviations(result, truth)
        text += [f"### {name}", "", "| True A D | Gaze bin ° | N | Gaze arcmin | A D | Converged |", "|---:|---|---:|---|---|---:|"]
        for a in np.unique(truth[:,0]):
            for lo, hi, label in ((0,4,"[0,4)"),(4,8,"[4,8)"),(8,12,"[8,12)"),(12,12.400001,"[12,12.4]")):
                mask = p & np.isclose(truth[:,0],a,rtol=0,atol=1e-10) & (mag>=lo) & (mag<hi)
                text.append(f"| {a:g} | {label} | {mask.sum()} | {fmt(stats(g[mask]))} | {fmt(stats(aerr[mask]))} | {result['converged'][mask].sum()} |")
        text += ["", f"### Ten worst primary states: {name}", "", "| Source ID | Source row | True A D | True ADE ° | True BDE ° | Estimated A D | Estimated ADE ° | Estimated BDE ° | Gaze arcmin | A error D | Converged |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        worst = np.flatnonzero(p)[np.argsort(g[p])[-10:][::-1]]
        for k in worst:
            s, est = truth[k], result["estimated_state"][k]
            text.append(f"| {ids[k]} | {rows[k]} | {s[0]:g} | {s[1]:g} | {s[2]:g} | {est[0]:.9g} | {est[1]:.9g} | {est[2]:.9g} | {g[k]:.9g} | {aerr[k]:.9g} | {bool(result['converged'][k])} |")
        text.append("")
    text += ["## Conditional-floor caveat and problems", "",
        "All results, and any conditional noise-floor interpretation from the prior report, are conditional on nominal recorded coordinates. The shared sensor frame is not independently verified: common origin, axis signs, scale and physical registration remain unproved. This noise-free interpolation evaluation does not validate hardware, eye translation, registration robustness, or feedback response to noise.", "",
        "B3 physical substitution is unavailable for the reason recorded above. PF12 physical candidate remains discrepant. Unconverged and invalid states are retained and flagged in each NPZ; convergence alone is not an accuracy claim. The saved source excludes the same 21 diagnostic states from primary scores; their estimates are retained. Oracle A isolates conditional centre-correction/readout error but does not imply additive error variances, because the full loop is nonlinear. During implementation the natural-B4 wrapper's required gaze Jacobian was added from the existing sparse model and checked against finite differences before the final full run.", "",
        "## Reproduction and provenance", "", "```powershell", "$env:PYTHONDONTWRITEBYTECODE='1'", "& 'C:\\Users\\rueijrwu\\.pyenv-win-venv\\envs\\venv\\Scripts\\python.exe' -B -u Script/p4_eval_ablation.py", "```", "",
        f"Elapsed wall time {elapsed:.3f} s. Frozen hashes: `{json.dumps(hashes,sort_keys=True)}`. Saved reference SHA-256 `{base.digest(SAVED)}`; Step-1 SHA-256 `{base.digest(STEP1)}`; source SHA-256 `{base.digest(base.SOURCE)}`. Original numerical-run script SHA-256 `{TRUSTED_INITIAL_NUMERICAL_SCRIPT_SHA256}`; current report-rendering script SHA-256 `{base.digest(__file__)}`. The numerical solver and variant checkpoints are unchanged; the current script accepts only this recorded original checkpoint hash or its own current hash.", "",
        "`evaluation.npz` contains scoring labels, masks, parity coefficients, gain coefficients, PF12 diagnostics and every variant's numerical output; `variant_*.npz` are run checkpoints. Existing files were read only; all generated files are within the allowed project paths. No commit was made.", ""]
    return "\n".join(text)


def main():
    started = time.perf_counter()
    frozen, field, motion, hashes = base.load_models(base.FIELD, base.MOTION)
    if base.digest(base.SOURCE) != "97c6db1b455f9abf7354351fa768cca7eea67198d8a4bfd14bbe684475462daf":
        raise RuntimeError("Frozen source checksum changed")
    with np.load(SAVED, allow_pickle=False) as z:
        saved = {k: z[k].copy() for k in z.files}
    if json.loads(str(saved["model_hashes_json"])) != hashes:
        raise RuntimeError("Saved reference and loaded frozen models differ")
    if saved["primary_mask"].sum() != 83229 or saved["excluded_mask"].sum() != 21:
        raise RuntimeError("Reference primary/excluded population changed")
    with np.load(base.SOURCE, allow_pickle=False) as z:
        rows = saved["source_subset_rows"]
        if not np.array_equal(z["source_indices"][rows], saved["source_indices"]):
            raise RuntimeError("Observed-source ID join failed")
        if not np.array_equal(z["states"][rows], saved["truth_state_scoring_only"]):
            raise RuntimeError("Observed-source state join failed")
        observed_p4 = z["absolute_p4_mm"][rows].copy()
        observed_p1 = z["mean_p1_absolute_mm"][rows].copy()
    cald, cal = saved["calibration_d_teacher_mm"], saved["calibration_states_teacher"]
    model = parity_fit(cald, cal[:,1:])
    unrestricted = {"terms": [tuple(v) for v in saved["degree3_powers"]],
                    "coefficients": saved["degree3_coefficients"]}
    caldiff = np.linalg.norm(base.predict(model,cald)-base.predict(unrestricted,cald),axis=1)*60
    runtime_d = saved["degree3_d_mm"]
    diff = np.linalg.norm(base.predict(model,runtime_d)-base.predict(unrestricted,runtime_d),axis=1)*60
    p = saved["primary_mask"]
    parity = dict(cal_max_arcmin=float(caldiff.max()),primary_max_arcmin=float(diff[p].max()),
                  all_max_arcmin=float(diff.max()),pass_=bool(max(caldiff.max(),diff.max())<=PARITY_TOL_ARCMIN))
    parity["pass"] = parity.pop("pass_")
    print(f"Parity map check: {parity}",flush=True)
    pred = base.predict(model,cald)
    feature = (cal[:,0,None]-2)*pred
    residual = cal[:,1:]-pred
    gain = np.sum(feature*residual,axis=0)/np.sum(feature*feature,axis=0)
    print(f"A-linear gain /D: {gain}",flush=True)
    seed = dict(estimated_A_d=saved["initial_observed_only_A_d"],
                converged=saved["initial_profile_valid"],valid_map=saved["initial_profile_valid"],
                reference401_fallback=saved["initial_profile_fallback"])
    results = {"reference_saved": {k[len("degree3_"):]:v for k,v in saved.items()
                                   if k.startswith("degree3_")}}
    designs = {"reference_saved":"Saved degree-3 closed-loop outputs; quoted reference, not rerun.",
        "identity_M1":"M1 scale coefficients zero; pivot unchanged.",
        "identity_B4":"B4(g)=0, preserving raw-H accommodation mean shift and central-ray convention.",
        "identity_B3":"P1 B3(g)=0; all P4 models unchanged.",
        "identity_field_shape":"PF12 coefficients zero; reference lattice, A terms and B4 anchor retained.",
        "identity_A_terms":"All 15 A extension values zero; PF12, B4, B3 and M1 retained.",
        "physical_M1":"Isotropic scale 1-0.081595476571*(A/4) about the saved M1 pivot.",
        "physical_B4":"B4(g)=mean(raw H(0,g)), so F=raw H(A,g), without the discrete B4 anchor.",
        "oracle_true_A":"True A held inside every centre correction and M1 step; bypasses A estimation. Other inputs remain observed.",
        "A_linear_gain":"Parity map times 1+k_axis*(estimated A-2); two calibration-only fitted gain values."}
    zero4 = replace(frozen,p4_bias=ZeroBias())
    zero3 = replace(frozen,p1_bias=ZeroBias())
    shape0 = UnifiedP4Field(frozen,field.extension.copy())
    shape0.base12[:] = 0
    natural4 = replace(frozen,p4_bias=ShapeMeanBias(field.sparse_model()))
    variants = [
        ("identity_M1",frozen,field,CentralMotion(1,motion.pivot_mm.copy(),np.zeros((2,1))),{}),
        ("identity_B4",zero4,UnifiedP4Field(zero4,field.extension.copy()),motion,{}),
        ("identity_B3",zero3,UnifiedP4Field(zero3,field.extension.copy()),motion,{}),
        ("identity_field_shape",frozen,shape0,motion,{}),
        ("identity_A_terms",frozen,UnifiedP4Field(frozen,np.zeros(15)),motion,{}),
        ("physical_M1",frozen,field,CentralMotion(1,motion.pivot_mm.copy(),np.full((2,1),SLOPE)),{}),
        ("physical_B4",natural4,UnifiedP4Field(natural4,field.extension.copy()),motion,{}),
        ("oracle_true_A",frozen,field,motion,{"oracle_A":saved["truth_state_scoring_only"][:,0]}),
        ("A_linear_gain",frozen,field,motion,{"gain":gain})]
    OUT.mkdir(parents=True,exist_ok=True)
    for name,f,fld,mot,options in variants:
        checkpoint = OUT/f"variant_{name}.npz"
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as z:
                if str(z["evaluation_script_sha256"]) not in (base.digest(__file__), TRUSTED_INITIAL_NUMERICAL_SCRIPT_SHA256):
                    raise RuntimeError(f"Checkpoint script mismatch: {checkpoint}")
                results[name] = {k:z[k].copy() for k in z.files if k!="evaluation_script_sha256"}
            print(f"Loaded own checkpoint {name}",flush=True)
        else:
            print(f"Running {name}",flush=True)
            result = solve(observed_p4,observed_p1,model,f,fld,mot,seed,label=name,**options)
            results[name]=result
            np.savez_compressed(checkpoint,evaluation_script_sha256=np.array(base.digest(__file__)),**result)
        g,a = deviations(results[name],saved["truth_state_scoring_only"])
        print(f"{name}: primary gaze {fmt(stats(g[p]))}; A {fmt(stats(a[p]))}",flush=True)
    print("Checking PF12 conversion",flush=True)
    pf = pf12_check(frozen,field)
    arrays = {"source_indices":saved["source_indices"],"source_subset_rows":saved["source_subset_rows"],
        "truth_state_scoring_only":saved["truth_state_scoring_only"],"primary_mask":p,
        "excluded_mask":saved["excluded_mask"],"calibration_states_teacher":cal,
        "calibration_d_teacher_mm":cald,"parity_coefficients":model["coefficients"],
        "parity_powers":np.array(model["terms"]),"A_linear_gain_per_D":gain,
        "parity_comparison_json":np.array(json.dumps(parity)),"frozen_hashes_json":np.array(json.dumps(hashes,sort_keys=True))}
    for key,value in pf.items():
        arrays["PF12_"+key]=value
    for name,result in results.items():
        for key,value in result.items():
            arrays[name+"_"+key]=value
        g,a=deviations(result,saved["truth_state_scoring_only"])
        arrays[name+"_gaze_error_arcmin_scoring_only"]=g
        arrays[name+"_A_error_D_scoring_only"]=a
    np.savez_compressed(OUT/"evaluation.npz",**arrays)
    report=make_report(saved,model,results,designs,parity,gain,pf,hashes,time.perf_counter()-started)
    (OUT/"report.md").write_text(report,encoding="utf-8")
    print(f"Wrote {OUT/'report.md'}",flush=True)


if __name__ == "__main__":
    main()
