"""Direct raw P4-P1 gaze: separate detector-x/BDE and detector-y/ADE cubics.

Eight polynomial coefficients total. Ground-truth gaze is a calibration target
and scoring label only; predict() receives the measured mean difference only.
No affine, bivariate cross-term, accommodation-input or corrected-center model.
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fit_p4_limited_gaze import DEFAULT_OUT as CALIBRATION_ROOT
from fit_p4_sparse_calibration import write_json
from evaluate_p4_p1_centroid_iteration import P1_PATH, P1_SHA

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT/"data/p4_p1_separate_axis_cubic_gaze"
PAIR_THRESHOLD = .054280660145285264
CENTER_THRESHOLD = .08014970816572936
FLAG_TOL = 1e-12


def design(u):
    return np.column_stack((np.ones(len(u)), u, u*u, u*u*u))


def matrix_stats(X):
    sv = np.linalg.svd(X, compute_uv=False)
    scaled = np.linalg.svd(X/np.maximum(np.linalg.norm(X, axis=0), 1e-30), compute_uv=False)
    return {"singular_values": sv.tolist(), "column_scaled_singular_values": scaled.tolist(),
        "condition": float(sv[0]/sv[-1]) if sv[-1] else None,
        "column_scaled_condition": float(scaled[0]/scaled[-1]) if scaled[-1] else None,
        "rank_relative_1e_10": int(np.sum(sv > sv[0]*1e-10))}


@dataclass
class AxisCubic:
    mean_mm: np.ndarray
    scale_mm: np.ndarray
    x_to_BDE: np.ndarray
    y_to_ADE: np.ndarray

    def predict(self, measured_mean_difference_mm):
        d = np.asarray(measured_mean_difference_mm, float)
        u = (d-self.mean_mm)/self.scale_mm
        return np.column_stack((design(u[:, 1])@self.y_to_ADE, design(u[:, 0])@self.x_to_BDE))

    def save(self, path, **metadata):
        np.savez_compressed(path, mean_mm=self.mean_mm, scale_mm=self.scale_mm,
            x_to_BDE=self.x_to_BDE, y_to_ADE=self.y_to_ADE, powers=np.arange(4),
            parameter_count=np.array(8), input_order=np.array(["d_x_mm", "d_y_mm"]),
            output_order=np.array(["ADE_deg", "BDE_deg"]), **metadata)

    @classmethod
    def load(cls, path):
        with np.load(path) as stored:
            return cls(stored["mean_mm"], stored["scale_mm"], stored["x_to_BDE"], stored["y_to_ADE"])


def fit(measured_d, ground_truth_gaze):
    d, target = np.asarray(measured_d), np.asarray(ground_truth_gaze)
    mean, scale = np.mean(d, axis=0), np.std(d, axis=0, ddof=0)
    if np.any(scale <= 1e-15):
        raise ValueError("Calibration detector coordinate has no usable variance")
    u = (d-mean)/scale
    coefficients, diagnostics = [], {}
    for axis, target_axis, name in ((0, 1, "detector_x_to_BDE"), (1, 0, "detector_y_to_ADE")):
        X = design(u[:, axis])
        beta, _, rank, _ = np.linalg.lstsq(X, target[:, target_axis], rcond=None)
        coefficients.append(beta)
        diagnostics[name] = {"columns": 4, "rows": len(d), "SVD_rank": int(rank),
            "full_rank": bool(rank == 4), "normalized_design": matrix_stats(X),
            "raw_mm_design": matrix_stats(design(d[:, axis])),
            "calibration_input_min_max_mm": [float(np.min(d[:, axis])), float(np.max(d[:, axis]))]}
    model = AxisCubic(mean, scale, coefficients[0], coefficients[1])
    correlation = np.corrcoef(np.column_stack((d, target)), rowvar=False)
    diagnostics["calibration_only_detector_vs_ADE_BDE_correlation"] = correlation[:2, 2:].tolist()
    diagnostics["axis_correspondence"] = "detector x predicts BDE; detector y predicts ADE; signs learned from ground-truth targets"
    diagnostics["full_rank"] = all(diagnostics[name]["full_rank"] for name in ("detector_x_to_BDE", "detector_y_to_ADE"))
    diagnostics["warning"] = "Formal4-column rank on3x3 may rely on accommodation/cross-gaze signal drift; rank alone does not prove stable interpolation"
    return model, diagnostics


def metrics(prediction, truth):
    error = prediction-truth
    radius = np.linalg.norm(error, axis=1)
    axes = {}
    for j, name in ((0, "ADE_y"), (1, "BDE_x")):
        e = error[:, j]
        tss = np.sum((truth[:, j]-np.mean(truth[:, j]))**2)
        axes[name] = {"bias_deg": float(np.mean(e)), "RMSE_deg": float(np.sqrt(np.mean(e*e))),
            "MAE_deg": float(np.mean(np.abs(e))), "P95_abs_deg": float(np.percentile(np.abs(e), 95)),
            "max_abs_deg": float(np.max(np.abs(e))), "inverse_target_R2": float(1.-np.sum(e*e)/tss) if tss else None}
    return {"states": len(truth), "axes": axes, "radial_RMSE_deg": float(np.sqrt(np.mean(radius*radius))),
        "radial_median_deg": float(np.median(radius)), "radial_P95_deg": float(np.percentile(radius, 95)),
        "radial_P99_deg": float(np.percentile(radius, 99)), "radial_max_deg": float(np.max(radius)),
        "mean_radial_error_deg": float(np.mean(radius)), "fraction_radial_error_le_0p5deg": float(np.mean(radius <= .5)),
        "bias_vector_ADE_BDE_deg": np.mean(error, axis=0).tolist(),
        "gaze_magnitude_signed_bias_deg": float(np.mean(np.linalg.norm(prediction, axis=1)-np.linalg.norm(truth, axis=1)))}


def cv(d, gaze):
    predictions = np.full_like(gaze, np.nan)
    rows = []
    for location in np.unique(gaze, axis=0):
        held = np.all(np.isclose(gaze, location, atol=1e-9, rtol=0), axis=1)
        model, identity = fit(d[~held], gaze[~held])
        predictions[held] = model.predict(d[held])
        error = predictions[held]-gaze[held]
        rows.append({"heldout_gaze_ADE_BDE": location.tolist(), "heldout_calibration_rows": int(np.sum(held)),
            "train_rows": int(np.sum(~held)), "coordinate_MSE_deg2": float(np.mean(error**2)),
            "radial_RMSE_deg": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))), "fit": identity,
            "train_mean_mm": model.mean_mm.tolist(), "train_scale_mm": model.scale_mm.tolist()})
    errors = predictions-gaze
    return {"folds": rows, "fold_count": len(rows), "all_folds_full_rank": all(r["fit"]["full_rank"] for r in rows),
        "pooled_coordinate_MSE_deg2": float(np.mean(errors**2)),
        "pooled_radial_RMSE_deg": float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))),
        "fold_coordinate_MSE_mean": float(np.mean([r["coordinate_MSE_deg2"] for r in rows])),
        "fold_coordinate_MSE_SD": float(np.std([r["coordinate_MSE_deg2"] for r in rows], ddof=1)),
        "policy": "leave every physical gaze location out with all3A; normalization andfit recomputed withinfold"}, predictions


def frozen_quality_flags(states):
    raw = P1_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != P1_SHA:
        raise RuntimeError("Pinned P1 data changed")
    p1 = pickle.loads(raw)
    ade, bde = np.unique(p1["eye_rotation_deg"]), np.unique(p1["eye_rotation_x_deg"])
    ai, bi = np.searchsorted(ade, p1["eye_rotation_deg"]), np.searchsorted(bde, p1["eye_rotation_x_deg"])
    p1 = p1[np.lexsort((p1["field_y_relative"], p1["field_x_relative"], ai, bi))]
    xy = np.stack((p1["real_x_mm"]+p1["center_absolute_x_mm"], p1["real_y_mm"]+p1["center_absolute_y_mm"]), -1).reshape(101, 101, 9, 2)
    midpoint = .5*(xy[:, :, [0, 1, 2, 3]]+xy[:, :, [8, 7, 6, 5]])
    center = np.mean(midpoint, axis=2)
    pair_error = np.sqrt(np.mean(np.sum((midpoint-center[:, :, None])**2, axis=3), axis=2))
    center_error = np.linalg.norm(xy[:, :, 4]-center, axis=2)
    ii, jj = np.searchsorted(bde, states[:, 2]), np.searchsorted(ade, states[:, 1])
    return (pair_error[ii, jj] > PAIR_THRESHOLD+FLAG_TOL) | (center_error[ii, jj] > CENTER_THRESHOLD+FLAG_TOL)


def self_check():
    rng = np.random.default_rng(20261107)
    d = rng.uniform(-2., 2., (100, 2))+[.4, -.3]
    bx, ay = np.array([1., -2., .3, -.02]), np.array([-.5, 3., -.2, .04])
    truth = np.column_stack((design(d[:, 1])@ay, design(d[:, 0])@bx))
    model, identity = fit(d, truth)
    test = rng.uniform(-2., 2., (50, 2))+[.4, -.3]
    expected = np.column_stack((design(test[:, 1])@ay, design(test[:, 0])@bx))
    assert identity["full_rank"] and np.max(np.abs(model.predict(test)-expected)) < 1e-10
    mu = model.mean_mm.copy()
    model.predict(test*10.)
    assert np.array_equal(mu, model.mean_mm)
    deficient = np.column_stack((np.tile([-1., 0., 1.], 3), np.repeat([-1., 0., 1.], 3)))
    _, rank = fit(deficient, deficient)
    assert not rank["full_rank"]


def plot(path, states, common, cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for name, (prediction, clipped) in cases.items():
        error = prediction[common]-states[common, 1:]
        clipped_error = clipped[common]-states[common, 1:]
        for values, style, label in ((np.linalg.norm(error, axis=1), "-", name),
                                     (np.linalg.norm(clipped_error, axis=1), "--", name+" clipped")):
            values = np.sort(values)
            axes[0, 0].plot(np.maximum(values, 1e-10), np.arange(1, len(values)+1)/len(values), style, label=label)
        av = np.unique(states[:, 0])
        rms = [np.sqrt(np.mean(np.sum((prediction[common & np.isclose(states[:, 0], a)]-states[common & np.isclose(states[:, 0], a), 1:])**2, axis=1))) for a in av]
        axes[0, 1].plot(av, rms, "o-", label=name)
        for j, ax in ((1, axes[1, 0]), (0, axes[1, 1])):
            values = np.sort(np.abs(error[:, j]))
            ax.plot(np.maximum(values, 1e-10), np.arange(1, len(values)+1)/len(values), label=name)
    axes[0, 0].axvline(.5, color="k", ls=":", label="0.5deg")
    axes[0, 0].set(xscale="log", xlabel="Paired gaze-parameter error (deg)", ylabel="Cumulative fraction", title="All common holdout; tails retained")
    axes[0, 1].set(xlabel="Ground-truth A (D), scoring only", ylabel="Paired gaze RMSE (deg)", title="Residual accommodation dependence")
    axes[1, 0].set(xscale="log", xlabel="Absolute BDE/x error (deg)", ylabel="Cumulative fraction")
    axes[1, 1].set(xscale="log", xlabel="Absolute ADE/y error (deg)", ylabel="Cumulative fraction")
    for ax in axes.ravel():
        ax.legend(fontsize=8)
    fig.suptitle("Separate-axis cubic gaze from raw P4−P1 means; no runtime A input")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=CALIBRATION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.check_only:
        print("separate-axis cubic synthetic recovery/normalization/rank checks PASS")
        return
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    source = args.calibration_dir/"source_subset.npz"
    with np.load(source) as saved:
        states, y4, m1, common, cal27, cal75, indices = (saved[key] for key in
            ("states", "absolute_p4_mm", "mean_p1_absolute_mm", "common_holdout", "cal27", "cal75", "source_indices"))
    assert len(states) == 83349 and np.sum(common) == 83250 and np.sum(cal27) == 27 and np.sum(cal75) == 75
    d = np.mean(y4, axis=1)-m1
    flags = frozen_quality_flags(states)
    summary = {"source_subset": str(source), "source_subset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "P1_sha256": P1_SHA,
        "input": "raw mean(all9 absoluteP4)-mean(all9 absoluteP1), x/y inmm", "runtime_inputs": ["d_x_mm", "d_y_mm"],
        "targets": "known ground-truth calibration BDE for detectorx; ADE for detectory", "polynomial_coefficients": 8,
        "normalization": "train-only mean/std saved separately; powers0,1,2,3 peraxis", "no_runtime_A_or_center_correction": True,
        "common_holdout_states": 83250, "requested_domain_deg": 12.5, "actual_ray_support_deg": 12.4,
        "quality_diagnostic": {"frozen_new75_pair_threshold_mm": PAIR_THRESHOLD, "center_threshold_mm": CENTER_THRESHOLD,
            "equality_tolerance_mm": FLAG_TOL, "flagged_common_states": int(np.sum(flags & common)), "primary_filters": "none"}, "cases": {}}
    cases = {}
    for budget, cal in ((27, cal27), (75, cal75)):
        model, identity = fit(d[cal], states[cal, 1:])
        cv_info, cv_prediction = cv(d[cal], states[cal, 1:])
        prediction = model.predict(d)
        clipped = np.clip(prediction, -12.5, 12.5)
        outside_input = np.any((d < np.min(d[cal], axis=0)-1e-12) | (d > np.max(d[cal], axis=0)+1e-12), axis=1)
        error = prediction-states[:, 1:]
        model.save(out/f"axis_cubic_{budget}.npz", calibration_source_indices=indices[cal],
            calibration_states=states[cal], calibration_raw_d_mm=d[cal], ground_truth_gaze_deg=states[cal, 1:])
        np.savez_compressed(out/f"predictions_{budget}.npz", source_indices=indices, truth_state=states,
            measured_raw_d_mm=d, unbounded_gaze_ADE_BDE_deg=prediction, clipped_gaze_ADE_BDE_deg=clipped,
            signed_error_deg=error, calibration_mask=cal, common_holdout_mask=common, quality_flag=flags,
            outside_calibration_detector_axis_intervals=outside_input, calibration_CV_prediction=cv_prediction)
        row = {"calibration_poses": budget, "coefficients": 8, "fit": identity, "calibration_CV": cv_info,
            "calibration_inverse_target_metrics": metrics(prediction[cal], states[cal, 1:]),
            "primary_unbounded_common": metrics(prediction[common], states[common, 1:]),
            "clipped_domain_prior_common": metrics(clipped[common], states[common, 1:]),
            "unbounded_outside_12p5_frames": int(np.sum(common & np.any(np.abs(prediction) > 12.5, axis=1))),
            "unbounded_outside_12p5_by_ADE_BDE": np.sum(np.abs(prediction[common]) > 12.5, axis=0).tolist(),
            "signal_axis_interval_extrapolation_common": int(np.sum(common & outside_input)),
            "quality_groups": {}, "by_A": {}}
        total_sse = np.sum(error[common]**2)
        for group, mask in (("all", common), ("unflagged", common & ~flags), ("flagged", common & flags)):
            if np.any(mask):
                row["quality_groups"][group] = {"coverage_of_common": float(np.sum(mask)/np.sum(common)),
                    "unbounded": metrics(prediction[mask], states[mask, 1:]), "clipped": metrics(clipped[mask], states[mask, 1:]),
                    "gaze_squared_error_share": float(np.sum(error[mask]**2)/total_sse) if total_sse else 0.}
        for av in np.unique(states[:, 0]):
            take = common & np.isclose(states[:, 0], av)
            row["by_A"][f"{av:.1f}"] = metrics(prediction[take], states[take, 1:])
        summary["cases"][str(budget)] = row
        cases[f"{budget}cal poses"] = prediction, clipped
        write_json(out/"summary.json", summary)
        print(f"{budget}poses axis-cubic: common radial RMSE{row['primary_unbounded_common']['radial_RMSE_deg']:.8g}deg; fullrank={identity['full_rank']}", flush=True)
    plot(out/"gaze_error_comparison.png", states, common, cases)
    lines = ["# Separate-axis cubic gaze from raw P4−P1 means", "",
        "Only the two requested univariate cubics are fitted: detectorx→ground-truthBDE and detectory→ground-truthADE, four coefficients each. The runtime input is only the measured raw nine-point mean difference; no true A, true gaze or P4-center correction is supplied. Training normalization is recomputed inside each calibration-only CV fold. No affine or bivariate model is used.", "",
        "| Cal poses | Coefficients | Common N | BDE/x RMSE(deg) | ADE/y RMSE(deg) | Radial RMSE(deg) | Median | P95 | P99 | Max | ≤0.5deg |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for budget, row in summary["cases"].items():
        q = row["primary_unbounded_common"]
        lines.append(f"|{budget}|8|{q['states']:,}|{q['axes']['BDE_x']['RMSE_deg']:.7g}|{q['axes']['ADE_y']['RMSE_deg']:.7g}|{q['radial_RMSE_deg']:.7g}|{q['radial_median_deg']:.7g}|{q['radial_P95_deg']:.7g}|{q['radial_P99_deg']:.7g}|{q['radial_max_deg']:.7g}|{100*q['fraction_radial_error_le_0p5deg']:.3f}%|")
    lines += ["", "Primary results retain every common holdout observation and use unbounded predictions. The separate±12.5 clipping comparison applies the same known domain prior to both calibration budgets; pre-clip values/counts remain saved. Frozen new75 quality flags and detector-signal extrapolation are diagnostic only. Rank/singular values/raw andscaled conditions, CV folds, inverse-target R², by-A errors and all signed predictions are in summary.json/NPZ. This direct raw-mean cubic is distinct from the iterative D15 center-correction pipeline."]
    (out/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
