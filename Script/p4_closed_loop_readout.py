"""Noise-free, observed-grid-only replacement of G1/G4 by one paired readout.

Only this script's data/p4_closed_loop_readout directory is written. Frozen
distortion models are reused, while saved hybrid estimates are never loaded.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

from evaluate_p4_unified_combined import load_models
from p4_fast_accommodation_profile import estimate_accommodation_fast
from p4_unified_fullshape import ProfileAdapter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/p4_p1_limited_gaze_12p5/source_subset.npz"
RUNTIME = ROOT / "data/p4_unified_combined/runtime/full_fast41/runtime.npz"
FIELD = ROOT / "data/p4_unified_combined/calibration/full_field_profile_final/summary.json"
MOTION = ROOT / "data/p4_unified_combined/calibration/center_motion/summary.json"
P1_CAL = ROOT / "data/p1_p4_a0_independent_gaze/calibration/calibration_data.npz"
MOTION_CAL = ROOT / "data/p4_unified_combined/calibration/center_motion/calibration_data.npz"
BASELINE = ROOT / "data/p4_gaze_p4p1_baseline/report.md"
OUT = ROOT / "data/p4_closed_loop_readout"
MAX_ITER = 40
A_TOL = 1e-5
GAZE_TOL = 1e-4
BOUND = 12.5
STATE_SCALE = np.array([4., 20., 20.])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def powers(degree):
    return [(i, total-i) for total in range(degree+1)
            for i in range(total, -1, -1)]


def features(x, terms):
    return np.stack([x[:, 0]**i*x[:, 1]**j for i, j in terms], axis=1)


def fit_readout(x, gaze, degree):
    """Fit in baseline z-score coordinates, then absorb them into coefficients."""
    terms = powers(degree)
    center, scale = x.mean(axis=0), x.std(axis=0)
    if np.any(scale <= 0):
        raise RuntimeError("Degenerate calibration displacement")
    design = features((x-center)/scale, terms)
    coef, _, rank, sv = np.linalg.lstsq(design, gaze, rcond=None)
    if rank != len(terms):
        raise RuntimeError("Rank deficient calibration readout")
    physical = np.zeros_like(coef)
    for k, (i, j) in enumerate(terms):
        for p in range(i+1):
            for q in range(j+1):
                factor = (math.comb(i, p)*math.comb(j, q)*(-center[0])**(i-p)
                          *(-center[1])**(j-q)/scale[0]**i/scale[1]**j)
                physical[terms.index((p, q))] += coef[k]*factor
    check = features(x, terms)@physical
    if np.max(np.abs(check-design@coef)) > 1e-9:
        raise RuntimeError("Normalizer folding changed predictions")
    return dict(terms=terms, coefficients=physical, normalized_coefficients=coef,
                center=center, scale=scale, rank=int(rank),
                condition=float(sv[0]/sv[-1]))


def predict(model, d):
    return features(d, model["terms"])@model["coefficients"]


def jacobian(model, x):
    result = np.zeros((len(x), 2, 2))
    for coefficient, (i, j) in zip(model["coefficients"], model["terms"]):
        if i:
            result[:, :, 0] += (i*x[:, 0]**(i-1)*x[:, 1]**j)[:, None]*coefficient
        if j:
            result[:, :, 1] += (j*x[:, 0]**i*x[:, 1]**(j-1))[:, None]*coefficient
    return result


def corrected_displacement(p4_mean, p1_mean, a, g, frozen, field, motion):
    forward = field.evaluate(a, g, diagnostics=True)
    c1 = p1_mean-frozen.p1_bias.predict(g)
    c4 = p4_mean-forward["prediction"].mean(axis=1)
    c4_ref = motion.inverse(c4, a)
    return c4_ref-c1, c1, c4, c4_ref, forward["valid"]


def profile(adapter, observed, gaze):
    return estimate_accommodation_fast(adapter, observed, gaze, chunk=1024,
                                        mean_centered=True, refinement_iterations=32)


def solve(observed_p4, observed_p1_mean, model, frozen, field, motion, seed):
    """Runtime boundary: observed grids, fitted parameters, observed-only seed.

    Truth, source IDs, saved A/gaze, G1 and G4 are absent from this function.
    The seed is profiled from observed_p4 at zero gaze outside this function.
    """
    n = len(observed_p4)
    means = observed_p4.mean(axis=1)
    current = np.column_stack((seed["estimated_A_d"], np.zeros((n, 2))))
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
    for iteration in range(1, MAX_ITER+1):
        rows = np.flatnonzero(alive)
        if not len(rows):
            break
        st = current[rows]
        d, _, _, _, valid = corrected_displacement(
            means[rows], observed_p1_mean[rows], st[:, 0], st[:, 1:], frozen, field, motion)
        raw = predict(model, d)
        bounded = np.clip(raw, -BOUND, BOUND)
        bound_pressure[rows] |= np.abs(raw) > BOUND
        valid &= np.isfinite(raw).all(axis=1)
        proposal = np.full_like(st, np.nan)
        proposal[:, 1:] = bounded
        if valid.any():
            prof, _ = profile(adapter, observed_p4[rows[valid]], bounded[valid])
            proposal[valid, 0] = prof["estimated_A_d"]
            profile_fallbacks[rows[valid]] += prof["reference401_fallback"]
            ok = valid.copy()
            ok[valid] = prof["converged"] & prof["valid_map"]
            valid = ok
        delta = proposal-st
        norm = np.linalg.norm(delta/STATE_SCALE, axis=1)
        valid &= np.isfinite(norm)
        iterations[rows] = iteration
        invalid[rows[~valid]] = True
        alive[rows[~valid]] = False
        improved = valid & (norm < best_norm[rows])
        take = rows[improved]
        best_norm[take] = norm[improved]
        best_state[take] = st[improved]
        # Check the full, undamped fixed-point proposal, never the damped step.
        step_ok = valid & (np.abs(delta[:, 0]) <= A_TOL) & (
            np.linalg.norm(delta[:, 1:], axis=1) <= GAZE_TOL)
        consecutive[rows] = np.where(step_ok, consecutive[rows]+1, 0)
        just = valid & (consecutive[rows] >= 2)
        converged[rows[just]] = True
        best_state[rows[just]] = st[just]
        best_norm[rows[just]] = norm[just]
        growth = valid & np.isfinite(previous_norm[rows]) & (norm > 1.2*previous_norm[rows])
        cycle = (valid & np.isfinite(previous_state[rows]).all(axis=1)
                 & (np.linalg.norm((proposal-previous_state[rows])/STATE_SCALE, axis=1) <= 1e-5)
                 & (norm > 1e-5))
        growth_count[rows[growth]] += 1
        cycle_count[rows[cycle]] += 1
        damp = rows[growth | cycle]
        alpha[damp] = np.maximum(alpha[damp]/2., 1./32.)
        previous_norm[rows[valid]] = norm[valid]
        previous_state[rows[valid]] = st[valid]
        alive[rows[just]] = False
        updating = valid & ~just
        current[rows[updating]] += alpha[rows[updating], None]*delta[updating]
        print(f"degree {max(sum(ij) for ij in model['terms'])}: iteration {iteration}; active {len(rows)}; "
              f"new converged {just.sum()}; remaining {alive.sum()}", flush=True)
    # Retain every attempted state; failed states keep their best observable
    # fixed-point residual, as in the frozen hybrid. They are never removed.
    d, c1, c4, c4ref, final_valid = corrected_displacement(
        means, observed_p1_mean, best_state[:, 0], best_state[:, 1:], frozen, field, motion)
    raw = predict(model, d)
    final_profile, final_timing = profile(adapter, observed_p4, np.clip(raw, -BOUND, BOUND))
    delta_a = final_profile["estimated_A_d"]-best_state[:, 0]
    delta_g = np.linalg.norm(np.clip(raw, -BOUND, BOUND)-best_state[:, 1:], axis=1)
    if np.any(converged & ((np.abs(delta_a) > A_TOL) | (delta_g > GAZE_TOL))):
        raise RuntimeError("Final recomputed residual contradicts convergence flag")
    return dict(estimated_state=best_state, converged=converged, invalid=invalid,
                valid=final_valid & np.isfinite(best_norm), iterations=iterations,
                alpha=alpha, growth_events=growth_count, cycle_events=cycle_count,
                profile_fallbacks=profile_fallbacks, bound_pressure=bound_pressure,
                d_mm=d, c1_mm=c1, c4_mm=c4, c4_A0_mm=c4ref, raw_map_gaze_deg=raw,
                final_delta_A_d=delta_a, final_delta_gaze_deg=delta_g,
                final_profile_valid=final_profile["converged"] & final_profile["valid_map"],
                shape_fit_vertex_rms_um=final_profile["fit_vertex_rms_um"],
                wall_seconds=np.array(time.perf_counter()-started))


def triple(x):
    x = np.asarray(x)
    if not len(x):
        return "n/a"
    if not np.isfinite(x).all():
        return "nonfinite (see NPZ)"
    return "/".join(f"{v:.6g}" for v in
                    (np.sqrt(np.mean(x*x)), np.percentile(np.abs(x), 95), np.max(np.abs(x))))


def errors(result, truth):
    return (np.linalg.norm(result["estimated_state"][:, 1:]-truth[:, 1:], axis=1)*60.,
            result["estimated_state"][:, 0]-truth[:, 0])


def make_report(models, results, states, ids, rows, primary, excluded, cal, cald,
                calids, p1ids, seed, hashes, probe_error, elapsed):
    headline = results[3]
    ge, ae = errors(headline, states)
    p = primary
    terms = models[3]["terms"]
    m = models[3]
    linear = np.stack([m["coefficients"][terms.index((1, 0))],
                       m["coefficients"][terms.index((0, 1))]], axis=1)
    zero_cal = np.flatnonzero(np.all(np.isclose(cal[:, 1:], 0.), axis=1) & np.isclose(cal[:, 0], 0.))
    j0 = jacobian(m, cald[zero_cal])[0]
    je = jacobian(m, headline["d_mm"][p])
    z = (cald-m["center"])/m["scale"]
    even_idx = [i for i, ij in enumerate(terms) if sum(ij) % 2 == 0]
    odd_idx = [i for i, ij in enumerate(terms) if sum(ij) % 2 == 1]
    even = features(z, terms)[:, even_idx]@m["normalized_coefficients"][even_idx]
    ratio = np.max(np.abs(m["normalized_coefficients"][even_idx])) / np.max(
        np.abs(m["normalized_coefficients"][odd_idx]))
    text = ["# P4/P1 closed-loop paired readout", "",
        "Noise-free evaluation using only input (b), the frozen hybrid-corrected centres. "
        "The new runtime is run from observed P4 grids and observed P1 nine-ray means. "
        "No saved A/gaze estimates, G1/G4 readout calls, raw-centroid readout results, or oracle-centre results enter it.", "",
        "## Design and count", "",
        "At each current estimate (A,g): c1=mean(y1)-B3(g); "
        "c4=mean(y4)-mean(F4(A,g)), where F4 includes B4, PF12 and the 15 A terms; "
        "d=M1^-1(c4,A)-c1; g=polynomial(d). B3 is the frozen P1 bias model named `p1_bias` in code. "
        "The degree-3 map has [1,dx,dy,dx²,dxdy,dy²,dx³,dx²dy,dxdy²,dy³] per output: 20 values. "
        "87-40+20=67: 18 B3/B4 + 12 PF12 + 15 A terms + 2 M1 + 20 readout. "
        "All distortion parameters remain frozen. Fitted readout z-score constants are analytically folded into "
        "the 10×2 physical-d coefficient array, so deployment needs exactly 20 readout values. "
        "Reference geometry and coordinate conventions remain the frozen constants.", "",
        "## Gates and calibration", "",
        "**Shared frame: cannot independently verify a common origin, axis signs and scale.** "
        "The stored baseline report (`data/p4_gaze_p4p1_baseline/report.md`, line 5) quotes the paired-source "
        "assumption: “conditional common-SI-coordinate simulation; identical nominal rear-camera chain/FOV, "
        "no fitted registration transform; physical registration independently unproved.” "
        "It also states that no explicit mirror matrix is documented and that the reconstructed P1 mean check "
        "does not independently verify shared sensor origin, signs or scale. "
        "Thus d uses nominal recorded coordinates exactly as the baseline; no registration or reflection is invented. "
        "The numerical performance below is conditional on that stored-frame assumption.", "",
        "**Teacher-consistent calibration:** centres are constructed using true gaze and true A, as required by "
        "the user's protocol and baseline definitions (baseline report line 11; baseline script calibration formulas). "
        "Runtime centre corrections, M1 and the A solver use estimated values only. "
        f"The frozen P1 calibration has {len(p1ids)} A=0 poses; the matched readout/P4 calibration has {len(cal)} poses "
        "(25 gazes at each A=0,2,4 D). These are the handoff's 25+75 roles, not 100 independent readout rows; "
        f"{len(np.intersect1d(p1ids, calids))} source identities overlap. "
        "Readouts of degrees 1,2,3 all fit the same 75 teacher-conditioned d values. "
        "No new distortion fit or calibration-pose selection is performed.", "",
        "## Closed-loop method", "",
        "One deterministic observed-only start: g=(0,0); A is estimated from the observed mean-centred "
        "P4 grid at that gaze. The saved joint estimate is not used even as a start. "
        "Each iteration recomputes both corrected centres, applies M1 inverse, forms d, evaluates the new map, "
        "then re-estimates A conditional on that new gaze. "
        "The existing frozen hybrid `estimate_accommodation_fast` with `ProfileAdapter(UnifiedP4Field)` "
        "minimizes sum of squared residuals over all nine mean-centred 2D rays. "
        "It scans 41 points over 0–4 D (0.1 D), refines with 32 golden iterations, compares exact endpoints, "
        "and falls back to the existing 401-point profiler for irregular/multiple/invalid profiles. "
        "The nonlinear forward transform is evaluated in full; its A dependence is not replaced by a linear interpolation.", "",
        f"Convergence: two consecutive full **undamped** proposals with |ΔA|≤{A_TOL:g} D and "
        f"radial ||Δg||₂≤{GAZE_TOL:g} degrees; maximum {MAX_ITER} iterations. "
        "Frozen hybrid bounds are A∈[0,4] D and each gaze axis ∈[-12.5,12.5] degrees. "
        "The bounded gaze enters the A solver and scoring state. Raw final polynomial outputs are saved and scored separately below. "
        "Damping follows the frozen hybrid: normalized state scale [4 D,20°,20°], residual growth >1.2× previous "
        "or a two-cycle (≤1e-5 normalized distance, residual >1e-5) halves alpha, minimum 1/32. "
        "Converged rows retain the tested current state; unconverged rows retain their best observable full-proposal "
        "residual state, are flagged, and remain in all attempted-state metrics.", "",
        f"Runtime/source joins reproduce the baseline's {p.sum():,} primary states and {excluded.sum()} separately "
        "excluded diagnostic states (no new exclusions). Truth and IDs enter scoring/calibration only, never `solve`. "
        f"Batched-versus-single-row profiler spot-check max |ΔA|={probe_error:.3g} D. "
        f"Total wall time {elapsed:.1f} s.", "",
        "## Degree comparison on the same primary holdout", "",
        "Gaze is radial Euclidean error ×60 arcmin; A is signed error for RMS and absolute error for P95/max. "
        "Every table's triples are RMS / P95 / max, including unconverged attempted states.", "",
        "| Degree | Readout values | N | Gaze (arcmin) | A (D) | Converged | Invalid | Iterations median/P95/max |",
        "|---:|---:|---:|---|---|---:|---:|---|"]
    for degree in (1, 2, 3):
        r = results[degree]
        gerr, aerr = errors(r, states)
        it = r["iterations"][p]
        text.append(f"| {degree} | {2*len(models[degree]['terms'])} | {p.sum()} | {triple(gerr[p])} | "
                    f"{triple(aerr[p])} | {r['converged'][p].sum()} | {r['invalid'][p].sum()} | "
                    f"{np.median(it):g}/{np.percentile(it,95):g}/{it.max()} |")
    text += ["", f"Degree 3 final undamped residual maxima: |ΔA|={np.max(np.abs(headline['final_delta_A_d'][p])):.6g} D; "
        f"radial Δg={np.max(headline['final_delta_gaze_deg'][p]):.6g} deg. "
        f"Final A-profile failures: {(~headline['final_profile_valid'][p]).sum()}. "
        f"Any-iteration gaze-bound pressure: {np.any(headline['bound_pressure'][p],axis=1).sum()} rows; "
        f"growth/cycle damping events: {headline['growth_events'][p].sum()}/{headline['cycle_events'][p].sum()}; "
        f"runtime irregular-profile fallbacks: {headline['profile_fallbacks'][p].sum()} "
        f"(initial profile fallbacks {seed['reference401_fallback'][p].sum()}).", "",
        f"Final unbounded degree-3 polynomial-output gaze error on the same primary rows: "
        f"{triple(np.linalg.norm(headline['raw_map_gaze_deg'][p]-states[p,1:],axis=1)*60.)} arcmin. "
        f"Separately excluded 21 rows: gaze {triple(ge[excluded])} arcmin; A {triple(ae[excluded])} D; "
        f"converged {headline['converged'][excluded].sum()}/{excluded.sum()}.", "",
        "## Stored baseline comparison (quoted, not rerun)", "",
        "Same primary held-out states and radial arcmin metric; input (b). "
        "Source: `data/p4_gaze_p4p1_baseline/report.md`, primary table (lines 33,40,45). "
        "The stored baseline is not a closed-loop replacement of G1/G4; it uses the saved hybrid A/centres.", "",
        "| Stored readout | RMS / P95 / max (arcmin) |", "|---|---|",
        "| P4 − P1, degree 3 | 0.91 / 1.77 / 4.19 |",
        "| P4 alone with M1 | 0.47 / 0.92 / 2.21 |", "| P1 alone | 0.18 / 0.30 / 0.37 |", "",
        "## Degree-3 error by |gaze| bin × true A", "",
        "For continuity with the baseline, |gaze| means max(|ADE|,|BDE|), not radial gaze magnitude. "
        "Bins are [0,4), [4,8), [8,12), [12,12.4] degrees. A and bin labels are scoring only.", "",
        "| A (D) | |gaze| bin (deg) | N | Gaze RMS/P95/max (arcmin) | A RMS/P95/max (D) | Converged |",
        "|---:|---|---:|---|---|---:|"]
    mag = np.max(np.abs(states[:, 1:]), axis=1)
    for a in np.unique(states[:, 0]):
        for lo, hi, label in ((0,4,"[0,4)"),(4,8,"[4,8)"),(8,12,"[8,12)"),(12,12.400001,"[12,12.4]")):
            mask = p & np.isclose(states[:, 0], a, atol=1e-10, rtol=0) & (mag>=lo) & (mag<hi)
            text.append(f"| {a:g} | {label} | {mask.sum()} | {triple(ge[mask])} | {triple(ae[mask])} | {headline['converged'][mask].sum()} |")
    text += ["", "## Ten worst primary states by degree-3 gaze error", "",
        "| Source ID | Source row | A true | ADE true | BDE true | A estimated | ADE estimated | BDE estimated | Gaze error (arcmin) | A error (D) | Iterations | Converged |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    worst = np.flatnonzero(p)[np.argsort(ge[p])[-10:][::-1]]
    for k in worst:
        s, est = states[k], headline["estimated_state"][k]
        text.append(f"| {ids[k]} | {rows[k]} | {s[0]:g} | {s[1]:g} | {s[2]:g} | {est[0]:.8g} | "
                    f"{est[1]:.8g} | {est[2]:.8g} | {ge[k]:.8g} | {ae[k]:.8g} | "
                    f"{headline['iterations'][k]} | {bool(headline['converged'][k])} |")
    text += ["", "## Linear gain and conditional single-frame 1 μm floor", "",
        "Matrices below map detector [x,y] displacement in mm to gaze [ADE,BDE] in degrees. "
        "The raw-d linear coefficient is not the operating-point derivative when d has an offset.", "",
        f"Raw-d linear coefficient matrix (deg/mm): `{linear.tolist()}`; singular gains "
        f"`{np.linalg.svd(linear,compute_uv=False).tolist()}` deg/mm.", "",
        f"At the teacher A=0, g=(0,0) displacement `{cald[zero_cal][0].tolist()}` mm, "
        f"Jacobian (deg/mm) is `{j0.tolist()}`; singular gains "
        f"`{np.linalg.svd(j0,compute_uv=False).tolist()}` deg/mm.", "",
        "For independent σd=0.001 mm per d axis, radial RMS = 60 σd ||J||F. "
        f"The linear-term floor is {60*.001*np.linalg.norm(linear):.6g} arcmin; "
        f"on-axis local floor {60*.001*np.linalg.norm(j0):.6g} arcmin; "
        f"primary-state local-Jacobian aggregate {60*.001*np.sqrt(np.mean(np.sum(je*je,axis=(1,2)))):.6g} arcmin. "
        "If each of two already-normalized channel centres instead has independent 1 μm/axis uncertainty, "
        "σd=√2 μm and these floors multiply by √2: "
        f"{60*.001*np.sqrt(2)*np.linalg.norm(linear):.6g}, "
        f"{60*.001*np.sqrt(2)*np.linalg.norm(j0):.6g}, "
        f"{60*.001*np.sqrt(2)*np.sqrt(np.mean(np.sum(je*je,axis=(1,2)))):.6g} arcmin respectively. "
        "These are conditional one-frame propagation calculations with A and corrections fixed; "
        "they do not simulate sensor noise, the full feedback/noise response, or 1 μm per ray. "
        "If uncertainty is specified before M1 inverse, the P4 covariance must also be scaled by 1/s(A)².", "",
        "## Parity check", "",
        f"The calibration d mean is `{m['center'].tolist()}` mm, and scale `{m['scale'].tolist()}` mm. "
        "Raw d is not centred on the odd-symmetry origin. Constant/even raw-d coefficients therefore need not vanish. "
        "Both raw-d and baseline-normalized coefficients are listed to avoid claiming parity from a coordinate shift. "
        f"In z=(d-mean)/std, max absolute even (constant+quadratic) coefficient / max absolute odd coefficient "
        f"is {ratio:.6g}; even contribution on calibration has radial RMS/max "
        f"{np.sqrt(np.mean(np.sum(even*even,axis=1)))*60:.6g}/{np.max(np.linalg.norm(even,axis=1))*60:.6g} arcmin. "
        f"Using an explicit relative 1e-3 coefficient threshold, even coefficients are "
        f"{'near zero' if ratio < 1e-3 else 'not all near zero'}. This is a descriptive check; no coefficients are forced to zero.", "",
        "| Monomial | Physical-d coefficients [ADE,BDE] | Baseline-normalized coefficients [ADE,BDE] | Parity |",
        "|---|---|---|---|"]
    for ij, co, zn in zip(terms, m["coefficients"], m["normalized_coefficients"]):
        text.append(f"| dx^{ij[0]} dy^{ij[1]} | {co.tolist()} | {zn.tolist()} | {'even' if sum(ij)%2==0 else 'odd'} |")
    text += ["", "## Problems and scope", "",
        f"Degree 3 has {(~headline['converged'][p]).sum()} unconverged primary states and "
        f"{headline['invalid'][p].sum()} invalid-loop states; these remain in the metrics. "
        "Convergence alone does not establish gaze accuracy. "
        "The physical shared-frame gate remains unverified, and teacher-conditioned calibration can differ "
        "from estimated-state runtime covariates. This is a noise-free interpolation test of frozen simulation artifacts; "
        "it does not validate hardware, registration robustness, eye translation or noise sensitivity. "
        "The quoted standalone P1 baseline can outperform this paired model in a simulation without eye translation. "
        "No unrelated pipeline, CODE V process, distortion refit or stored-baseline rerun was performed.", "",
        "## Reproduction and provenance", "",
        "Run from the project with bytecode disabled:", "", "```powershell",
        "$env:PYTHONDONTWRITEBYTECODE='1'",
        "& 'C:\\Users\\rueijrwu\\.pyenv-win-venv\\envs\\venv\\Scripts\\python.exe' -u Script/p4_closed_loop_readout.py",
        "```", "", "All numerical states, coefficients, errors, convergence telemetry and calibration inputs "
        "are in `closed_loop_readout.npz` (no object arrays; truth is labeled scoring-only).", "",
        f"Source SHA-256: `{digest(SOURCE)}`. Runtime-mask artifact SHA-256: `{digest(RUNTIME)}`. "
        f"Baseline report SHA-256: `{digest(BASELINE)}`. Frozen model hashes: `{json.dumps(hashes,sort_keys=True)}`.", ""]
    return "\n".join(text)


def main():
    started = time.perf_counter()
    if digest(SOURCE) != "97c6db1b455f9abf7354351fa768cca7eea67198d8a4bfd14bbe684475462daf":
        raise RuntimeError("Frozen source checksum changed")
    frozen, field, motion, hashes = load_models(FIELD, MOTION)
    if motion.order != 1 or motion.coefficient_count != 2:
        raise RuntimeError("Expected frozen two-value M1")
    with np.load(SOURCE) as z:
        allids, allstates = z["source_indices"].copy(), z["states"].copy()
        p4, p1 = z["absolute_p4_mm"].copy(), z["mean_p1_absolute_mm"].copy()
        calmask, common = z["cal75"].copy(), z["common_holdout"].copy()
    # Load identity/masks only. No saved hybrid A/gaze/centre arrays are read.
    with np.load(RUNTIME) as z:
        rows, ids = z["source_subset_rows"].copy(), z["source_indices"].copy()
        primary, excluded = z["primary_mask"].copy(), z["excluded_mask"].copy()
    if (not np.array_equal(rows, np.flatnonzero(common)) or not np.array_equal(ids, allids[rows])
            or primary.sum() != 83229 or excluded.sum() != 21
            or np.any(primary & excluded) or not np.all(primary | excluded)):
        raise RuntimeError("Frozen baseline holdout identity/masks changed")
    calrows = np.flatnonzero(calmask)
    cal, calids = allstates[calrows], allids[calrows]
    if (len(cal) != 75 or len(np.unique(cal[:, 1:], axis=0)) != 25
            or not np.array_equal(np.unique(cal[:, 0]), [0., 2., 4.])):
        raise RuntimeError("Frozen 75-pose calibration changed")
    with np.load(MOTION_CAL) as z:
        lookup = {int(i): s for i, s in zip(z["source_indices"], z["states"])}
        if any(int(i) not in lookup or not np.array_equal(s, lookup[int(i)]) for i, s in zip(calids, cal)):
            raise RuntimeError("Frozen motion calibration join changed")
    with np.load(P1_CAL) as z:
        p1ids = z["calibration_p4_source_indices"].copy()
        p1g = z["calibration_gaze_deg"].copy()
    if len(p1ids) != 25 or len(np.unique(p1g, axis=0)) != 25:
        raise RuntimeError("Frozen 25-pose P1 calibration changed")
    # True labels used only for teacher-consistent calibration construction.
    cald, _, _, _, calvalid = corrected_displacement(p4[calrows].mean(axis=1), p1[calrows],
                         cal[:, 0], cal[:, 1:], frozen, field, motion)
    if not calvalid.all():
        raise RuntimeError("Invalid teacher calibration grid map")
    models = {d: fit_readout(cald, cal[:, 1:], d) for d in (1, 2, 3)}
    observed_p4, observed_p1 = p4[rows], p1[rows]
    adapter = ProfileAdapter(field)
    print("Initial A profile from observed P4 grids at zero gaze", flush=True)
    seed, _ = profile(adapter, observed_p4, np.zeros((len(rows), 2)))
    # Small numerical implementation check of the reused solver; no new model fit.
    probe = np.unique(np.r_[np.linspace(0, len(rows)-1, 7, dtype=int), np.flatnonzero(excluded)[:1]])
    probe_errors = []
    for k in probe:
        single, _ = profile(adapter, observed_p4[k:k+1], np.zeros((1, 2)))
        probe_errors.append(abs(single["estimated_A_d"][0]-seed["estimated_A_d"][k]))
    results = {}
    for degree in (1, 2, 3):
        print(f"Running closed loop degree {degree}", flush=True)
        results[degree] = solve(observed_p4, observed_p1, models[degree], frozen, field, motion, seed)
    # Truth is first consumed after inference for these heldout rows.
    states = allstates[rows]
    arrays = dict(source_indices=ids, source_subset_rows=rows,
                  truth_state_scoring_only=states, primary_mask=primary, excluded_mask=excluded,
                  calibration_states_teacher=cal, calibration_source_indices=calids,
                  calibration_d_teacher_mm=cald, p1_calibration_source_indices=p1ids,
                  initial_observed_only_A_d=seed["estimated_A_d"],
                  initial_profile_valid=seed["converged"] & seed["valid_map"],
                  initial_profile_fallback=seed["reference401_fallback"],
                  model_hashes_json=np.array(json.dumps(hashes,sort_keys=True)))
    for degree, model in models.items():
        prefix = f"degree{degree}_"
        for name in ("coefficients", "normalized_coefficients", "center", "scale", "rank", "condition"):
            arrays[prefix+name] = np.asarray(model[name])
        arrays[prefix+"powers"] = np.array(model["terms"])
        for name, values in results[degree].items():
            arrays[prefix+name] = values
        ge, ae = errors(results[degree], states)
        arrays[prefix+"gaze_error_arcmin_scoring_only"] = ge
        arrays[prefix+"A_error_d_scoring_only"] = ae
    report = make_report(models, results, states, ids, rows, primary, excluded, cal, cald,
                         calids, p1ids, seed, hashes, max(probe_errors), time.perf_counter()-started)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT/"closed_loop_readout.npz", **arrays)
    (OUT/"report.md").write_text(report, encoding="utf-8")
    print(f"Wrote {OUT/'report.md'} and closed_loop_readout.npz", flush=True)


if __name__ == "__main__":
    main()
