"""P1-centred corrected-M1 check using frozen, noise-free artifacts only.

Writes only data/p4_pivot_pf12_check/. The pivot law translates the archived
Step-1 additive M1 increment from the zero detector origin to corrected P1 c1.
"""
from __future__ import annotations

import json
import hashlib
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

import p4_closed_loop_readout as base
import p4_eval_ablation as ab
import p4_m1_pivot_fix as fix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/p4_pivot_pf12_check"
REPORT = OUT / "report.md"
SAVED = ROOT / "data/p4_closed_loop_readout/closed_loop_readout.npz"
EVAL = ROOT / "data/p4_eval_ablation/evaluation.npz"
STEP = ROOT / "data/p4_phys/step1.npz"
PIVOT_NAME = "physical_M1_P1_pivot"
SLOPE = fix.SLOPE


def read(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


class PivotP1Motion(fix.PhysicalMotion):
    pass


def corrected_displacement(p4_mean, p1_mean, a, g, frozen, field, motion):
    if not isinstance(motion, PivotP1Motion):
        return fix.corrected_displacement(p4_mean, p1_mean, a, g, frozen, field, motion)
    forward = field.evaluate(a, g, diagnostics=True)
    c1 = p1_mean - frozen.p1_bias.predict(g)
    c4 = p4_mean - forward["prediction"].mean(axis=1)
    scale_minus_one = SLOPE * np.asarray(a) / 4.0
    delta_origin = fix.physical_delta(a, g, motion.lever, motion.signs)
    delta_c1 = delta_origin - scale_minus_one[:, None] * c1
    c4ref = c4 - delta_c1
    return c4ref - c1, c1, c4, c4ref, forward["valid"]


def signed_bias(saved, results):
    truth = saved["truth_state_scoring_only"]
    primary = saved["primary_mask"]
    radius = np.max(np.abs(truth[:, 1:]), axis=1)
    grid = np.arange(21) / 5
    rows, data = [], []
    summary = []
    for name in ("reference_saved", "physical_M1_corrected", PIVOT_NAME):
        result = results[name]
        err = (result["estimated_state"][:, 1:] - truth[:, 1:]) * 60
        for lo, hi in ((0, 4), (4, 8)):
            means = []
            for a in grid:
                mask = (primary & (radius >= lo) & (radius < hi)
                        & np.isclose(truth[:, 0], a, atol=1e-10, rtol=0))
                mu = err[mask].mean(axis=0)
                mag = float(np.linalg.norm(mu))
                means.append(mu)
                data.append(f"{name}|{a:.1f}|{lo}|{hi}|{int(mask.sum())}|{mu[0]:.17g}|{mu[1]:.17g}|{mag:.17g}")
                rows.append(f"| {name} | {a:.1f} | [{lo},{hi}) | {mask.sum()} | {mu[0]:.10g} | {mu[1]:.10g} | {mag:.10g} |")
            means = np.asarray(means)
            # Summarize each component separately; BDE's near-zero mean is
            # expected from the balanced signed gaze population.
            for axis, axis_name in enumerate(("ADE", "BDE")):
                values = means[:, axis]
                coef = np.polyfit(grid, values, 1)
                fitted = np.polyval(coef, grid)
                denom = np.sum((values - values.mean()) ** 2)
                r2 = float(1 - np.sum((values - fitted) ** 2) / denom) if denom else float("nan")
                crossing = bool(values.min() <= 0 <= values.max())
                zero = float(-coef[1] / coef[0]) if coef[0] else float("inf")
                summary.append(f"| {name} | [{lo},{hi}) | {axis_name} | {values.min():.10g} | {values.max():.10g} | {coef[1]:.10g} | {coef[0]:.10g} | {r2:.8g} | {zero:.9g} | {crossing} |")
    return rows, summary, np.asarray(data, dtype="U256")


def pivot_rows(result, saved):
    states = saved["truth_state_scoring_only"]
    p = saved["primary_mask"]
    rows = ["| True A D | N | c1 x min / median / max mm | c1 y min / median / max mm |",
            "|---:|---:|---|---|"]
    data = []
    for a in np.arange(21) / 5:
        m = p & np.isclose(states[:, 0], a, atol=1e-10, rtol=0)
        c = result["c1_mm"][m]
        sx = np.min(c[:, 0]), np.median(c[:, 0]), np.max(c[:, 0])
        sy = np.min(c[:, 1]), np.median(c[:, 1]), np.max(c[:, 1])
        rows.append(f"| {a:.1f} | {m.sum()} | {' / '.join(f'{x:.9g}' for x in sx)} | {' / '.join(f'{x:.9g}' for x in sy)} |")
        data.append([a, m.sum(), *sx, *sy])
    return rows, np.asarray(data)


def metric_line(name, result, saved, count):
    p = saved["primary_mask"]
    gaze, acc = base.errors(result, saved["truth_state_scoring_only"])
    return (f"| {name} | {count} | {ab.fmt(ab.stats(gaze[p]))} | "
            f"{ab.fmt(ab.stats(acc[p]))} | {int(result['converged'][p].sum())} | "
            f"{int(result['invalid'][p].sum())} |")


def worst_rows(result, saved):
    gaze, acc = base.errors(result, saved["truth_state_scoring_only"])
    p = saved["primary_mask"]
    order = np.flatnonzero(p)[np.argsort(gaze[p])[-10:][::-1]]
    rows = ["| Source ID | Source row | True A/ADE/BDE | Estimated A/ADE/BDE | Gaze arcmin | A error D | Converged |",
            "|---:|---:|---|---|---:|---:|---|"]
    for i in order:
        t = " / ".join(f"{v:.9g}" for v in saved["truth_state_scoring_only"][i])
        e = " / ".join(f"{v:.9g}" for v in result["estimated_state"][i])
        rows.append(f"| {saved['source_indices'][i]} | {saved['source_subset_rows'][i]} | {t} | {e} | {gaze[i]:.9g} | {acc[i]:.9g} | {bool(result['converged'][i])} |")
    return rows


def pf12_audit(frozen, field, step, saved):
    states = step["P4_states"]
    candidate = step["P4_candidate_raw_H"]
    teacher = step["P4_teacher_raw_H"]
    trace = step["P4_trace_raw_H"]
    a, g = states[:, 0], states[:, 1:]
    sparse = field.sparse_model()
    h = sparse.evaluate(a, g[:, 0], g[:, 1])["prediction"]
    h0 = sparse.evaluate(a, np.zeros(len(a)), np.zeros(len(a)))["prediction"]
    h_a0 = sparse.evaluate(np.zeros(len(a)), g[:, 0], g[:, 1])["prediction"]
    gaze_delta = h - h0
    accommodation_delta = h - h_a0
    dev = candidate - teacher
    op = np.max(np.abs(g), axis=1) <= 12.4 + 1e-10
    radial = np.linalg.norm(dev, axis=2) * 1000
    stats_full = ab.stats(radial.ravel())
    stats_op = ab.stats(radial[op].ravel())
    gain = .0069 * 60  # 6.9 deg/mm => 0.414 arcmin/µm
    budget_um = 3.0 / gain
    delta_norm_um = np.linalg.norm(gaze_delta, axis=2) * 1000
    ratio = radial / np.maximum(delta_norm_um, 1e-12)
    flatworst = np.argsort(radial[op].max(axis=1))[-10:][::-1]
    operating_ids = np.flatnonzero(op)
    worst = operating_ids[flatworst]
    table = ["| Source state row | A/ADE/BDE | Field label | Teacher gaze displacement µm | Candidate-teacher deviation µm | Ratio | Deviation arcmin at 6.9 deg/mm |",
             "|---:|---|---|---:|---:|---:|---:|"]
    for i in worst:
        j = int(np.argmax(radial[i]))
        row = " / ".join(f"{x:.6g}" for x in states[i])
        label = " / ".join(f"{x:.6g}" for x in step["P4_fields"][j])
        table.append(f"| {i} | {row} | {label} | {delta_norm_um[i,j]:.8g} | {radial[i,j]:.8g} | {ratio[i,j]:.8g} | {radial[i,j]*.414:.8g} |")
    # Explicit handoff basis parity: saved teacher must match the model evaluated
    # with the already converted PF12 coefficients.
    parity_um = float(np.max(np.linalg.norm((teacher - h), axis=2)) * 1000)
    # Candidate relation to the ray-trace is reported independently of teacher.
    trace_dev = np.linalg.norm(candidate - trace, axis=2) * 1000
    trace_op = ab.stats(trace_dev[op].ravel())
    # The physical construction's transferred sensor centre is the radial pivot.
    origin = step["G4_origin"]
    signs = step["G4_sensor_sign"]
    lever = step["G4_effective_lever_xyz_mm"]
    from p4_phys_step1 import rotation
    centre = origin + np.einsum("ij,nj->ni", signs,
        np.einsum("nij,j->ni", rotation(g)-np.eye(3), lever)[:, :2])
    k0 = -.018500496700
    radius = np.linalg.norm(centre, axis=1)
    tangential = 1 + k0 * radius**2
    radial_gain = 1 + 3*k0*radius**2
    min_rad = float((np.sqrt(-1/(3*k0))) if k0 < 0 else np.inf)
    worst_i = int(np.flatnonzero(op)[np.argmax(np.linalg.norm(centre[op], axis=1))])
    sections = ["## (c) PF12 audit: corrected teacher versus ray-trace", "",
        "The saved Step-1 candidate is compared directly with the converted frozen teacher and with the saved ray-trace, using all nine labelled rays and no fit. At each P4 state/A, gaze displacement is H(A,g)-H(A,0), using that same A and each corresponding labelled ray. This establishes the scale of the field signal alongside candidate error; it does not turn this candidate into a validated PF12 model.", "",
        f"Per-ray radial candidate-minus-teacher deviation RMS/P95/max is {ab.fmt(stats_full)} µm over the full Step-1 grid and {ab.fmt(stats_op)} µm for max-axis gaze ≤12.4°. Per-ray candidate-minus-raytrace deviation on the operating region is {ab.fmt(trace_op)} µm. Saved converted teacher parity maximum is {parity_um:.9g} µm.", "",
        f"Using the independently quoted 6.9 deg/mm scale (0.414 arcmin/µm), the 3-arcmin displacement allowance is {budget_um:.6g} µm; the handoff uses the conservative rounded budget of 7.2 µm. This is an approximate sensitivity conversion, not a loop result. The candidate exceeds the 7.2-µm budget by {stats_op[2]/7.2:.6g}× at the operating maximum.", "",
        f"The transferred radial-distortion centre uses G4 origin {origin.tolist()} mm, sensor sign matrix {signs.tolist()}, and effective lever {lever.tolist()} mm. For the existing on-axis radial coefficient k0={k0}, the radial derivative changes sign beyond radius {min_rad:.8g} mm. Worst operating center row {worst_i} has pivot radius {radius[worst_i]:.8g} mm, tangential gain {tangential[worst_i]:.8g}, radial gain {radial_gain[worst_i]:.8g}. The saved convention report says the sensor-centre displacement to distortion-plane pivot transfer is unresolved and the on-axis law was only validated at two sampled nonzero radii; this model can fold under large corner offsets. Accordingly, the very large residual is an invalid pivot-frame transfer / radial extrapolation diagnostic, not evidence of a valid PF12 candidate failure or a corrected PF12 result. No PF12 substitution or arbitrary scaling was run.", "",
        "Ten largest operating states (one worst ray per state):", "",
        "\n".join(table), "",
        "At A=0 and zero gaze the candidate has the same reference identity as the teacher. At A=0 and combined gaze (+12.4,+12.4), the recorded candidate/teacher vector comparison is retained in the row ranking; there is no claim this corner is representative of the whole population."]
    arrays = {"PF12_deviation_full_um": stats_full, "PF12_deviation_operating_um": stats_op,
              "PF12_candidate_trace_operating_um": trace_op,
              "PF12_teacher_parity_max_um": np.array(parity_um),
              "PF12_gaze_displacement_um": delta_norm_um,
              "PF12_accommodation_displacement_um": np.linalg.norm(accommodation_delta, axis=2)*1000,
              "PF12_candidate_teacher_deviation_um": radial,
              "PF12_deviation_to_gaze_displacement_ratio": ratio,
              "PF12_transferred_pivot_mm": centre,
              "PF12_radial_derivative_min_radius_mm": np.array(min_rad),
              "PF12_radial_gain_worst_operating": np.array(radial_gain[worst_i]),
              "PF12_worst_operating_state_row": np.array(worst_i)}
    return "\n".join(sections), table, arrays


def report_write(sections):
    body = ["# P4 corrected M1 with P1-centred pivot", "",
            "Noise-free, observed-grid input-(b) check. All frozen models and saved source artifacts are read-only. The primary score includes all 83,229 attempted states; 21 diagnostic states remain excluded. Shared P1/P4 detector registration is assumed and has not been independently verified.", "",
            "Pivot convention used here: keep the archived Step-1 additive M1 law and translate its zero-origin magnification increment to the same-state corrected P1 centre c1(g)=mean(P1)-B3(g). Let delta0=(s(A)-1) S[(R(g)-I)(0,0,ell)]xy and s(A)=1-0.081595476571(A/4). The translated increment is delta_c1=delta0-(s(A)-1)c1; runtime uses c4ref=c4-delta_c1. This preserves the archived lever, sensor signs, fixed slope, and additive inverse; it does not refit M1 or the 10-value readout. It is an operational P1-centred pivot hypothesis, not an anatomically identified eye pivot: the detector-to-eye registration is provisional and the 3D lever remains unidentified. It is distinct from the frozen CentralMotion affine 2D center law.", "",
            "The relative pivot in the corrected detector frame is exactly [0,0] mm. Absolute c1 varies by gaze and accommodation; its per-accommodation coordinate summaries follow.", ""]
    for key in sorted(sections):
        body.extend([sections[key], ""])
    REPORT.write_text("\n".join(body), encoding="utf-8")


def main():
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUT}")
    if (OUT.parent / "p4_m1_pivot_fix").exists() is False:
        raise FileNotFoundError("Saved M1/PF12 diagnostic artifacts are required")
    OUT.mkdir(parents=True, exist_ok=True)
    input_paths = [SAVED, EVAL, STEP, base.SOURCE, base.FIELD, base.MOTION,
                   ROOT / "Script/p4_m1_pivot_fix.py", ROOT / "Script/p4_eval_ablation.py",
                   ROOT / "Script/p4_closed_loop_readout.py",
                   ROOT / "Script/p4_phys_step1.py", ROOT / "Script/p4_unified_center_motion.py",
                   ROOT / "data/p4_unified_combined/calibration/center_motion/central_motion_M1.npz",
                   ROOT / "data/p4_m1_pivot_fix/diagnostics.npz",
                   ROOT / "data/p4_m1_pivot_fix/variant_physical_M1_corrected.npz",
                   ROOT / "data/p4_phys/step1_report.md", ROOT / "data/p4_phys/conventions.md"]
    input_hashes_before = {str(p.relative_to(ROOT)): base.digest(p) for p in input_paths}
    frozen, field, motion, hashes = base.load_models(base.FIELD, base.MOTION)
    saved = read(SAVED)
    evaluation = read(EVAL)
    step = read(STEP)
    if json.loads(str(saved["model_hashes_json"])) != hashes:
        raise RuntimeError("Saved reference and frozen-model hashes disagree")
    if int(saved["primary_mask"].sum()) != 83229 or int(saved["excluded_mask"].sum()) != 21:
        raise RuntimeError("Frozen primary/excluded masks changed")
    results = {"reference_saved": {k[len("degree3_"):]: v for k, v in saved.items()
                                   if k.startswith("degree3_")}}
    for path in sorted((ROOT / "data/p4_eval_ablation").glob("variant_*.npz")):
        results[path.stem[len("variant_"):]] = read(path)
    old_physical = results["physical_M1"]
    old_ge, old_ae = base.errors(old_physical, saved["truth_state_scoring_only"])
    old_primary = saved["primary_mask"]
    old_metric = np.r_[ab.stats(old_ge[old_primary]), ab.stats(old_ae[old_primary])]
    expected_old = np.array([28.90906,55.97492,75.25108,.1872435,.4,.6])
    if not np.allclose(old_metric, expected_old, rtol=2e-5, atol=2e-6):
        raise RuntimeError(f"Old physical_M1 checkpoint differs from its report: {old_metric}")
    corrected_checkpoint = read(ROOT / "data/p4_m1_pivot_fix/variant_physical_M1_corrected.npz")
    corr_ge, corr_ae = base.errors(corrected_checkpoint, saved["truth_state_scoring_only"])
    corr_metric = np.r_[ab.stats(corr_ge[old_primary]), ab.stats(corr_ae[old_primary])]
    expected_corr = np.array([.8650331,1.647548,4.291233,.02859844,.05610867,.1863149])
    if not np.allclose(corr_metric, expected_corr, rtol=2e-5, atol=2e-6):
        raise RuntimeError(f"Corrected M1 checkpoint differs from its report: {corr_metric}")
    results["physical_M1_corrected"] = corrected_checkpoint
    if not np.array_equal(evaluation["reference_saved_estimated_state"],
                          results["reference_saved"]["estimated_state"]):
        raise RuntimeError("Saved reference arrays disagree")
    model = {"terms": [tuple(x) for x in saved["degree3_powers"]],
             "coefficients": evaluation["parity_coefficients"]}
    del evaluation
    with np.load(base.SOURCE, allow_pickle=False) as z:
        source_rows = saved["source_subset_rows"]
        if not np.array_equal(z["source_indices"][source_rows], saved["source_indices"]):
            raise RuntimeError("Source ID join changed")
        if not np.array_equal(z["states"][source_rows], saved["truth_state_scoring_only"]):
            raise RuntimeError("Source state join changed")
        observed_p4 = z["absolute_p4_mm"][source_rows].copy()
        observed_p1 = z["mean_p1_absolute_mm"][source_rows].copy()
    seed = dict(estimated_A_d=saved["initial_observed_only_A_d"],
                converged=saved["initial_profile_valid"],
                valid_map=saved["initial_profile_valid"],
                reference401_fallback=saved["initial_profile_fallback"])
    pivot_motion = PivotP1Motion(float(step["M1_lever_mm"][0]), step["G4_sensor_sign"])
    ab.corrected_displacement = corrected_displacement
    result = ab.solve(observed_p4, observed_p1, model, frozen, field, pivot_motion,
                      seed, label=PIVOT_NAME)
    results[PIVOT_NAME] = result
    sections = {}
    pivot_table, pivot_data = pivot_rows(result, saved)
    sections["10"] = "## (a) P1-centred pivot and closed loop\n\n| Variant | Values | Gaze RMS / P95 / max arcmin | A RMS / P95 / max D | Converged | Invalid |\n|---|---:|---|---|---:|---:|\n" + metric_line(PIVOT_NAME, result, saved, "57 (55 + fixed slope + archived lever)") + "\n\n" + "\n".join(pivot_table)
    bias_rows, bias_fits, bias_data = signed_bias(saved, results)
    sections["20"] = "## (b) Signed gaze bias by A and gaze bin\n\nComponents are mean estimated-minus-true gaze in arcmin, followed by the magnitude of that mean vector. The scalar ADE trend is shown separately from component crossing; BDE symmetry means near-zero component means should not be read as a pivot-direction diagnosis.\n\n| Variant | True A D | Gaze bin | N | Mean ADE | Mean BDE | Vector magnitude |\n|---|---:|---|---:|---:|---:|---:|\n" + "\n".join(bias_rows) + "\n\n| Variant | Gaze bin | Component | Min | Max | Intercept | Slope/D | R² | Linear zero D | Crossing |\n|---|---|---|---:|---:|---:|---:|---:|---:|---|\n" + "\n".join(bias_fits)
    pf, pf_rows, pf_arrays = pf12_audit(frozen, field, step, saved)
    sections["30"] = pf
    counts = {"reference_saved": "57", "identity_M1": "55", "identity_B4": "48",
              "identity_B3": "48", "identity_field_shape": "45", "identity_A_terms": "42",
              "physical_M1": "57 declared; invalid scalar-centre law", "physical_B4": "48",
              "oracle_true_A": "57; true A supplied", "A_linear_gain": "59",
              "physical_M1_corrected": "57 (55 + fixed slope + archived lever)",
              PIVOT_NAME: "57 (55 + fixed slope + archived lever)"}
    lines = ["## (d) Consolidated saved and new variants", "",
             "| Variant | Values | Gaze RMS / P95 / max arcmin | A RMS / P95 / max D | Converged | Invalid |",
             "|---|---|---|---|---:|---:|"]
    for name, value in results.items():
        if name in counts:
            lines.append(metric_line(name, value, saved, counts[name]))
    lines += ["| physical_B3 | unavailable | not run | — | — | — |",
              "| physical_PF12 | not deployed | invalid pivot-frame transfer; no valid substitution | — | — | — |",
              "| pivot_fitted | not deployed | no fitted pivot run | — | — | — |"]
    lines += ["", "Endpoint A estimates count as clipped when within 1e-7 D of 0 or 4 D. Primary headline values retain all states.",
              "", "| Variant | True A | N | Clipped | All A RMS/P95/max D | Unclipped N | Unclipped A RMS/P95/max D |",
              "|---|---:|---:|---:|---|---:|---|"]
    truth = saved["truth_state_scoring_only"]; primary = saved["primary_mask"]
    for name, value in results.items():
        if name not in counts: continue
        est = value["estimated_state"][:, 0]
        err = est - truth[:, 0]
        clipped = np.isclose(est, 0., atol=1e-7, rtol=0) | np.isclose(est, 4., atol=1e-7, rtol=0)
        for a in (0., 4.):
            m = primary & np.isclose(truth[:, 0], a, atol=1e-10, rtol=0); u = m & ~clipped
            lines.append(f"| {name} | {a:g} | {m.sum()} | {(m&clipped).sum()} | {ab.fmt(ab.stats(err[m]))} | {u.sum()} | {ab.fmt(ab.stats(err[u]))} |")
    lines += ["", "Counts use retained nonzero/fitted law values. Reference=18 B3/B4+12 PF12+15 A+2 M1+10 readout=57; readout is 6 ADE and 4 BDE terms. Identity counts: M1 55, B3/B4 each 48, field shape 45, A terms 42. The old physical_M1 row is the invalid absolute scalar-centre law; corrected and P1-pivoted laws each retain the archived lever and fixed slope (57 law values). Oracle uses true A and is not deployable. A_linear_gain adds two values (59)."]
    lines += ["", "Constants and provenance: `SLOPE=-0.081595476571` is the noise-free on-axis ray-trace fitted scale coefficient documented in `data/p4_onaxis/report.md` and used as fixed in Step 1; s(4)=0.918404523429 (direct A=4 fit 0.9183505584). The frozen M1 diagonal coefficients are [-0.04138309709608498,-0.04142806731949131] about the [0,0] mm A=0 sensor pivot; at A=4 this gives scales [0.9586169029,0.9585719327], and doubling them gives [0.9172338058,0.9171438654], close to but not exactly the fixed physical scale. The 8.10207652-mm effective axial lever was distilled against frozen M1 teacher increments, not anatomically identified; sensor signs/origin are inherited calibration conventions."]
    sections["40"] = "\n".join(lines)
    sections["50"] = "## (e) Ten worst primary states for new pivot loop\n\n" + "\n".join(worst_rows(result, saved))
    hashes = input_hashes_before
    if input_hashes_before != {str(p.relative_to(ROOT)): base.digest(p) for p in input_paths}:
        raise RuntimeError("An input artifact changed during the check")
    gaze_err, acc_err = base.errors(result, saved["truth_state_scoring_only"])
    base_metric = np.array([.8650331,1.647548,4.291233])
    new_metric = ab.stats(gaze_err[primary])
    ratios = new_metric / base_metric
    rounded_base_metric = np.array([.865, 1.648, 4.291])
    rounded_ratios = new_metric / rounded_base_metric
    accepted = bool(np.all((rounded_ratios >= .95) & (rounded_ratios <= 1.05)))
    sections["60"] = ("## Decision and provenance\n\n"
        f"Within strict ±5% corrected-M1 envelope using requested rounded baseline [.865,.1.648,4.291]: **{accepted}**. Ratios vs exact saved corrected-M1 metrics are {ab.fmt(ratios)}; vs rounded gate they are {ab.fmt(rounded_ratios)}. Degradation-only <=5% gate is {bool(np.all(ratios <= 1.05))}; an improvement exceeding 5% lies outside a strict symmetric envelope but is not an accuracy degradation. The 3-arcmin maximum target is {'met' if new_metric[2] <= 3 else 'not met'}. No fitted second pivot was run.\n\n"
        f"New pivot metrics: gaze {ab.fmt(ab.stats(gaze_err[primary]))} arcmin; accommodation {ab.fmt(ab.stats(acc_err[primary]))} D.\n\n"
        f"New script SHA-256: `{base.digest(__file__)}`; input hashes: `{json.dumps(hashes,sort_keys=True)}`.\n\n"
        f"Elapsed {time.perf_counter()-started:.3f}s. All reported results remain conditional on shared P1/P4 registration.")
    report_write(sections)
    out = dict(source_indices=saved["source_indices"], source_subset_rows=saved["source_subset_rows"],
               truth_state_scoring_only=truth, primary_mask=primary, excluded_mask=saved["excluded_mask"],
               signed_bias_table=bias_data, pivot_by_A=pivot_data,
               new_gaze_error_arcmin_scoring_only=gaze_err,
               new_A_error_D_scoring_only=acc_err, accepted_within_5percent=np.array(accepted),
               input_hashes_json=np.array(json.dumps(hashes, sort_keys=True)),
               new_script_sha256=np.array(base.digest(__file__)),
               new_converged=result["converged"], new_invalid=result["invalid"],
               new_estimated_state=result["estimated_state"], new_c1_mm=result["c1_mm"],
               new_d_mm=result["d_mm"], new_c4_mm=result["c4_mm"], new_c4_A0_mm=result["c4_A0_mm"],
               **pf_arrays)
    np.savez_compressed(OUT / "evaluation.npz", **out)
    print(f"Completed {PIVOT_NAME}: gaze {ab.fmt(ab.stats(gaze_err[primary]))}; A {ab.fmt(ab.stats(acc_err[primary]))}; accepted={accepted}", flush=True)


if __name__ == "__main__":
    main()
