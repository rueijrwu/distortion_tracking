"""Calibrate new gaze-only P4 shape/C8 models using A0 observations exclusively."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import numpy as np
from scipy.optimize import least_squares

from p4_a0_only_models import (CenterCubic, GazeOnlyShape, OUTER,
                               design_geometry, self_check)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"data/p4_p1_limited_gaze_12p5/source_subset.npz"
SOURCE_SHA = "97c6db1b455f9abf7354351fa768cca7eea67198d8a4bfd14bbe684475462daf"
QUALITY = ROOT/"data/p4_p1_limited_gaze_12p5/quality_diagnostics/quality_flags.npz"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coordinate_metrics(predicted, observed):
    e = (np.asarray(predicted)-np.asarray(observed))*1000.0
    radial = np.linalg.norm(e, axis=-1)
    return {"x_RMSE_um": float(np.sqrt(np.mean(e[..., 0]**2))),
            "y_RMSE_um": float(np.sqrt(np.mean(e[..., 1]**2))),
            "vertex_RMS_um": float(np.sqrt(np.mean(radial**2))),
            "max_point_um": float(radial.max())}


def fit_shape(reference, gaze, target, count, order):
    model = GazeOnlyShape(reference, np.zeros(count), order, count)
    begun = time.perf_counter()
    residual = lambda c: ((model.evaluate(gaze, c)[:, OUTER]-target[:, OUTER])*1000.0).ravel()
    result = least_squares(residual, np.zeros(count), jac="cs", x_scale="jac",
                           ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=300)
    model.coefficients = result.x.copy()
    evaluated = model.evaluate(gaze, diagnostics=True)
    geometry = design_geometry(result.jac/1000.0)
    valid = bool(evaluated["valid"].all())
    good = bool(result.success and geometry["rank"] == count and valid
                and np.isfinite(result.fun).all())
    diagnostics = {
        "success": bool(result.success), "eligible": good,
        "message": str(result.message), "nfev": int(result.nfev),
        "njev": int(result.njev), "cost_um2_half": float(result.cost),
        "optimality": float(result.optimality), "wall_seconds": time.perf_counter()-begun,
        "jacobian_physical_output_mm": geometry, "all_calibration_maps_valid": valid,
        "regularity_minima": {key: float(np.min(evaluated[key])) for key in
            ("denominator", "brown_determinant", "radial_tangential", "radial_radial", "homography_determinant")},
        "calibration_outer8": coordinate_metrics(evaluated["prediction"][:, OUTER], target[:, OUTER]),
        "zero_gaze_identity_max_abs_mm": float(np.max(np.abs(model.evaluate(np.zeros((1, 2)))[0]-reference))),
    }
    return model, diagnostics


def shape_candidate(reference, gaze, target, count, order):
    model, full = fit_shape(reference, gaze, target, count, order)
    folds, losses = [], []
    nonzero = np.flatnonzero(np.any(gaze != 0, axis=1))
    for test_row in nonzero:
        train = np.arange(len(gaze)) != test_row
        try:
            fold_model, info = fit_shape(reference, gaze[train], target[train], count, order)
            prediction = fold_model.evaluate(gaze[test_row:test_row+1])
            loss = float(np.mean(((prediction[:, OUTER]-target[test_row:test_row+1, OUTER])*1000.0)**2))
            if not info["eligible"] or not np.isfinite(loss):
                loss = None
            folds.append({"test_calibration_row": int(test_row), "test_gaze_deg": gaze[test_row].tolist(),
                          "coordinate_MSE_um2": loss, "fit": info})
        except Exception as error:
            loss = None
            folds.append({"test_calibration_row": int(test_row), "test_gaze_deg": gaze[test_row].tolist(),
                          "coordinate_MSE_um2": None, "error": repr(error)})
        losses.append(loss)
    eligible = full["eligible"] and all(value is not None for value in losses)
    values = np.asarray(losses, dtype=float) if eligible else None
    summary = {"spatial_coefficients": count, "order": order, "full_fit": full,
               "folds": folds, "fold_count": len(folds), "eligible": bool(eligible),
               "CV_coordinate_MSE_um2": float(values.mean()) if eligible else None,
               "CV_SE_um2": float(values.std(ddof=1)/np.sqrt(len(values))) if eligible else None}
    return model, summary


def cubic_calibration(q, gaze):
    cubic, full = CenterCubic.fit(q, gaze)
    full_rank = all(v["standardized"]["rank"] == 4 for v in full.values())
    folds, losses = [], []
    for test_row in np.flatnonzero(np.any(gaze != 0, axis=1)):
        train = np.arange(len(gaze)) != test_row
        try:
            fitted, geometry = CenterCubic.fit(q[train], gaze[train])
            prediction = fitted.predict(q[test_row:test_row+1])[0]
            loss = float(np.mean((prediction-gaze[test_row])**2))
            rank_ok = all(v["standardized"]["rank"] == 4 for v in geometry.values())
            folds.append({"test_calibration_row": int(test_row), "test_gaze_deg": gaze[test_row].tolist(),
                          "prediction_gaze_deg": prediction.tolist(), "coordinate_MSE_deg2": loss,
                          "full_rank": rank_ok, "geometry": geometry})
            losses.append(loss if rank_ok else None)
        except Exception as error:
            folds.append({"test_calibration_row": int(test_row), "error": repr(error), "full_rank": False})
            losses.append(None)
    good = full_rank and all(value is not None for value in losses)
    return cubic, {"coefficient_count": 8, "full_rank": full_rank, "eligible": bool(good),
                   "full_geometry": full, "folds": folds,
                   "CV_coordinate_MSE_deg2": float(np.mean(losses)) if good else None,
                   "CV_radial_RMS_deg": float(np.sqrt(2*np.mean(losses))) if good else None,
                   "q_mean_mm": cubic.mean_mm.tolist(), "q_scale_mm": cubic.scale_mm.tolist(),
                   "calibration_correlation_dx_BDE": float(np.corrcoef(q[:, 0], gaze[:, 1])[0, 1]),
                   "calibration_correlation_dy_ADE": float(np.corrcoef(q[:, 1], gaze[:, 0])[0, 1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--include-nine", action="store_true", help="Optional9-pose feasibility case")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"data/p4_p1_a0_only/calibration")
    args = parser.parse_args()
    checks = self_check()
    print(json.dumps({"model_checks": checks}), flush=True)
    if args.check_only:
        return
    if digest(SOURCE) != SOURCE_SHA:
        raise RuntimeError("Pinned paired observations changed")
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Fresh calibration directory required: {out}")
    # Filter to A0 before obtaining reference, any calibration/scaling, or fitting.
    with np.load(SOURCE) as raw:
        a0_rows = np.flatnonzero(raw["states"][:, 0] == 0)
        gaze = raw["states"][a0_rows, 1:3].copy()
        observed = raw["absolute_p4_mm"][a0_rows].copy()
        mean_p1 = raw["mean_p1_absolute_mm"][a0_rows].copy()
        source_ids = raw["source_indices"][a0_rows].copy()
        prior_common = raw["common_holdout"][a0_rows].copy()
    if len(gaze) != 3969 or len(np.unique(gaze[:, 0])) != 63 or len(np.unique(gaze[:, 1])) != 63:
        raise AssertionError("Expected only3,969 A0 gaze observations")
    if not np.isfinite(observed).all() or not np.isfinite(mean_p1).all():
        raise AssertionError("A0 observations contain invalid real points")
    zero = np.flatnonzero(np.all(gaze == 0, axis=1))
    if len(zero) != 1:
        raise AssertionError("Expected one measured zero-gaze reference")
    centered = observed-observed[:, 4:5, :]
    reference = centered[zero[0]].copy()
    stencil25 = np.array([-12.4, -6.4, 0., 6.4, 12.4])
    stencil9 = np.array([-12.4, 0., 12.4])
    on_stencil = lambda values: np.all(np.any(np.isclose(gaze[:, :, None], values, atol=1e-9, rtol=0), axis=2), axis=1)
    cal25, cal9 = on_stencil(stencil25), on_stencil(stencil9)
    if cal25.sum() != 25 or cal9.sum() != 9 or (~cal25).sum() != 3944:
        raise AssertionError("Physical calibration/holdout stencils differ")
    with np.load(QUALITY) as quality:
        flags = quality["new_calibration_flag"][a0_rows].astype(bool)
        if not np.array_equal(quality["source_indices"][a0_rows], source_ids):
            raise AssertionError("Quality row identities differ")
    if flags.sum() != 1 or np.any(flags & cal25):
        raise AssertionError("Expected one retained heldout P1 anomaly")
    out.mkdir(parents=True)
    (out/"models").mkdir()
    np.savez_compressed(out/"a0_observations.npz", gaze_deg=gaze, absolute_p4_mm=observed,
                        mean_p1_absolute_mm=mean_p1, source_indices=source_ids,
                        raw_source_subset_rows=a0_rows, calibration25=cal25, calibration9=cal9,
                        holdout25=~cal25, quality_flagged=flags, prior_comparison3936=prior_common,
                        reference_mm=reference, reference_row=np.array(zero[0]))
    summaries = {}
    for budget, mask in ((25, cal25), (9, cal9)):
        if budget == 9 and not args.include_nine:
            continue
        cal_gaze, cal_target = gaze[mask], centered[mask]
        candidates = {}
        fitted = {}
        for count in ((12, 33) if budget == 25 else (12,)):
            for order in ("PF", "DF"):
                name = f"{order}{count}"
                print(json.dumps({"calibration_poses": budget, "candidate": name, "phase": "calibration_CV"}), flush=True)
                try:
                    model, info = shape_candidate(reference, cal_gaze, cal_target, count, order)
                    path = out/"models"/f"cal{budget}_{name}.npz"
                    model.save(path)
                    info.update(path=str(path), sha256=digest(path))
                    candidates[name], fitted[name] = info, model
                except Exception as error:
                    candidates[name] = {"eligible": False, "error": repr(error), "spatial_coefficients": count, "order": order}
        (out/f"cal{budget}_candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
        eligible = [name for name, value in candidates.items() if value["eligible"]]
        if not eligible:
            summaries[str(budget)] = {"calibration_poses": budget, "shape_candidates": candidates,
                                     "failure": "No successful full-rank shape candidate; no reduced-rank deployment"}
            if budget == 9:
                continue
            raise RuntimeError(f"No successful full-rank shape candidate for{budget} poses")
        best = min(eligible, key=lambda name: candidates[name]["CV_coordinate_MSE_um2"])
        threshold = candidates[best]["CV_coordinate_MSE_um2"]+candidates[best]["CV_SE_um2"]
        within = [name for name in eligible if candidates[name]["CV_coordinate_MSE_um2"] <= threshold]
        selected = min(within, key=lambda name: (candidates[name]["spatial_coefficients"], candidates[name]["CV_coordinate_MSE_um2"]))
        selection = {"selected": selected, "best_CV": best, "one_SE_threshold_um2": threshold,
                     "eligible_candidates": eligible, "within_one_SE": within,
                     "rule": "Smallest coefficient count within oneSE of best calibration-only fold MSE; order tie by lower CV MSE"}
        # Save/freeze the calibration-only decision before scoring any heldout ray.
        (out/f"cal{budget}_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
        cubic, cubic_info = cubic_calibration(observed[mask, 4]-mean_p1[mask], cal_gaze)
        cubic_path = out/"models"/f"cal{budget}_center_C8.npz"
        cubic.save(cubic_path)
        cubic_info.update(path=str(cubic_path), sha256=digest(cubic_path))
        # Evaluation only: same3,944 new-model test rows even for optional9 case.
        test = ~cal25
        heldout_forward = {}
        for name, model in fitted.items():
            prediction = model.evaluate(gaze[test], diagnostics=True)
            heldout_forward[name] = {"outer8": coordinate_metrics(prediction["prediction"][:, OUTER], centered[test][:, OUTER]),
                                      "all9": coordinate_metrics(prediction["prediction"], centered[test]),
                                      "invalid_maps": int((~prediction["valid"]).sum())}
        summaries[str(budget)] = {"calibration_poses": budget, "calibration_rows": np.flatnonzero(mask).tolist(),
                                 "shape_candidates": candidates, "shape_selection": selection,
                                 "center_C8": cubic_info, "heldout_forward": heldout_forward,
                                 "total_selected_values": candidates[selected]["spatial_coefficients"]+8,
                                 "cubic_9pose_caution": "Three intentional per-axis gaze levels can give fragile cubic identification through nuisance cross-axis drift" if budget == 9 else None}
    summary = {"experiment": "All calibration/reference/scaling/fits use only true A0 observations",
               "source_subset_sha256": SOURCE_SHA, "source_subset_path": str(SOURCE),
               "raw_P4_grid_sha256": "9d2e2f9da71360dea3583b9da180d40703679de9123af0d1358767af7f8ce490",
               "raw_P1_grid_sha256": "1a57a8d850d021f3d23dda201f2fc47bd1aec6eb60860e42990be990cfe7f835",
               "A0_rows": len(gaze), "primary_holdout": int((~cal25).sum()), "retained_flagged": int(flags.sum()),
               "reference_source_id": int(source_ids[zero[0]]), "reference_A0_gaze_deg": [0, 0],
               "gaze_scale_deg": 12.4, "requested_runtime_bound_deg": 12.5,
               "no_accommodation_parameters": True, "no_prior_learned_coefficients": True,
               "model_checks": checks, "cases": summaries,
               "observation_artifact_sha256": digest(out/"a0_observations.npz"),
               "model_source_sha256": digest(ROOT/"Script/p4_a0_only_models.py"),
               "calibration_source_sha256": digest(__file__),
               "frame_assumption": "Pinned P1/P4 nominal SI coordinates; external registration not independently proved"}
    (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "selections": {key: value.get("shape_selection", value.get("failure"))
                                                         for key, value in summaries.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
