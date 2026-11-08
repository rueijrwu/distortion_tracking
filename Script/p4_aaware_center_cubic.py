"""75-pose corrected-center C8/C12/C16, with explicit estimated-A runtime input."""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fit_p4_limited_gaze import DEFAULT_OUT as SOURCE_ROOT
from fit_p4_p1_axis_cubic_gaze import matrix_stats, metrics
from fit_p4_sparse_calibration import write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT/"data/p4_p1_aaware_center_cubic"
FEATURE_NAMES = {8: ("1", "z", "z^2", "z^3"),
    12: ("1", "z", "z^2", "z^3", "a", "a*z"),
    16: ("1", "z", "z^2", "z^3", "a", "a*z", "a*z^2", "a*z^3")}


class FrozenC8Adapter:
    """Reuse the exact frozen C8 coefficients while accepting current A."""
    parameter_count = 8

    def __init__(self, frozen_model):
        self.frozen_model = frozen_model

    def predict(self, corrected_q_mm, current_estimated_A_d):
        return self.frozen_model.predict(corrected_q_mm)


def features(z, a, count):
    base = np.column_stack((np.ones(len(z)), z, z*z, z*z*z))
    if count == 8:
        return base
    if count == 12:
        return np.column_stack((base, a, a*z))
    if count == 16:
        return np.column_stack((base, a[:, None]*base))
    raise ValueError("Only predeclared C8/C12/C16 are supported")


@dataclass
class AwareCenterCubic:
    mean_mm: np.ndarray
    scale_mm: np.ndarray
    x_to_BDE: np.ndarray
    y_to_ADE: np.ndarray
    parameter_count: int

    def predict(self, corrected_q_mm, current_estimated_A_d):
        q, A = np.asarray(corrected_q_mm), np.asarray(current_estimated_A_d).ravel()
        if q.shape != (len(A), 2):
            raise ValueError("Expected corrected q(N,2) and estimated A(N)")
        z, a = (q-self.mean_mm)/self.scale_mm, A/4.
        return np.column_stack((features(z[:, 1], a, self.parameter_count)@self.y_to_ADE,
                                features(z[:, 0], a, self.parameter_count)@self.x_to_BDE))

    def save(self, path, **metadata):
        np.savez_compressed(path, mean_mm=self.mean_mm, scale_mm=self.scale_mm,
            x_to_BDE=self.x_to_BDE, y_to_ADE=self.y_to_ADE, parameter_count=np.array(self.parameter_count),
            features=np.array(FEATURE_NAMES[self.parameter_count]), input_A_scale_D=np.array(4.), **metadata)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(data["mean_mm"], data["scale_mm"], data["x_to_BDE"], data["y_to_ADE"], int(data["parameter_count"]))


def fit(q, known_calibration_A, target_gaze, count):
    q, A, target = np.asarray(q), np.asarray(known_calibration_A), np.asarray(target_gaze)
    mean, scale = np.mean(q, axis=0), np.std(q, axis=0, ddof=0)
    if np.any(scale <= 1e-15):
        raise ValueError("Calibration q has no coordinate variance")
    z, a = (q-mean)/scale, A/4.
    coefficients, identity = [], {}
    for axis, output, name in ((0, 1, "x_BDE"), (1, 0, "y_ADE")):
        X = features(z[:, axis], a, count)
        beta, _, rank, _ = np.linalg.lstsq(X, target[:, output], rcond=None)
        coefficients.append(beta)
        covariance = np.linalg.pinv(X.T@X, rcond=1e-14)
        sd = np.sqrt(np.maximum(np.diag(covariance), 0.))
        correlation = covariance/np.maximum(sd[:, None]*sd[None, :], 1e-30)
        identity[name] = {"rank": int(rank), "columns": X.shape[1], "full_rank": bool(rank == X.shape[1]),
            "normalized_design": matrix_stats(X), "raw_q_mm_design": matrix_stats(features(q[:, axis], a, count)),
            "unit_target_noise_design_covariance": covariance.tolist(), "parameter_correlation_geometry": correlation.tolist(),
            "scope": "design geometry; not an errors-in-variables or real sensor-noise covariance"}
    identity["full_rank"] = identity["x_BDE"]["full_rank"] and identity["y_ADE"]["full_rank"]
    return AwareCenterCubic(mean, scale, coefficients[0], coefficients[1], count), identity


def calibration_cv(q, A, gaze, count):
    prediction = np.full_like(gaze, np.nan)
    folds = []
    for location in np.unique(gaze, axis=0):
        held = np.all(np.isclose(gaze, location, atol=1e-9, rtol=0), axis=1)
        model, identity = fit(q[~held], A[~held], gaze[~held], count)
        prediction[held] = model.predict(q[held], A[held])
        loss = float(np.mean(np.sum((prediction[held]-gaze[held])**2, axis=1)))
        folds.append({"gaze_ADE_BDE": location.tolist(), "heldout_rows": int(np.sum(held)),
            "squared_vector_error_mean_deg2": loss, "fit": identity,
            "train_mean_mm": model.mean_mm.tolist(), "train_scale_mm": model.scale_mm.tolist()})
    losses = np.array([row["squared_vector_error_mean_deg2"] for row in folds])
    return {"folds": folds, "all_folds_full_rank": all(row["fit"]["full_rank"] for row in folds),
        "mean_squared_vector_error_deg2": float(np.mean(losses)),
        "SE_over25_fold_MSEs_deg2": float(np.std(losses, ddof=1)/np.sqrt(len(losses))),
        "pooled_radial_RMSE_deg": float(np.sqrt(np.mean(np.sum((prediction-gaze)**2, axis=1))))}, prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT/"calibration")
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    source = args.source_dir/"source_subset.npz"
    with np.load(source) as stored:
        states, y4, m1, cal, indices = (stored[key] for key in
            ("states", "absolute_p4_mm", "mean_p1_absolute_mm", "cal75", "source_indices"))
    assert np.sum(cal) == 75
    q, A, gaze = y4[cal, 4]-m1[cal], states[cal, 0], states[cal, 1:]
    summary = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "calibration_poses": 75, "feature": "labeled central P4 ray minus measured P1mean; a=knowncalA/4, runtime currentestimatedA/4",
        "normalization": "q mean/std from training rows only, recomputed perCVfold", "models": {}}
    for count in (8, 12, 16):
        model, identity = fit(q, A, gaze, count)
        cv, prediction = calibration_cv(q, A, gaze, count)
        path = out/f"corrected_C{count}.npz"
        model.save(path, calibration_states=states[cal], calibration_q_mm=q, calibration_source_indices=indices[cal])
        summary["models"][f"C{count}"] = {"parameter_count": count, "fit": identity, "CV": cv,
            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "calibration_metrics": metrics(model.predict(q, A), gaze)}
        np.savez_compressed(out/f"CV_C{count}.npz", truth_state=states[cal], predicted_gaze=prediction)
    eligible = [name for name, row in summary["models"].items() if row["fit"]["full_rank"] and row["CV"]["all_folds_full_rank"]]
    best = min(eligible, key=lambda name: summary["models"][name]["CV"]["mean_squared_vector_error_deg2"])
    threshold = summary["models"][best]["CV"]["mean_squared_vector_error_deg2"]+summary["models"][best]["CV"]["SE_over25_fold_MSEs_deg2"]
    selected = min([name for name in eligible if summary["models"][name]["CV"]["mean_squared_vector_error_deg2"] <= threshold],
                   key=lambda name: summary["models"][name]["parameter_count"])
    summary["selection"] = {"selected": selected, "eligible": eligible, "bestCV": best,
        "one_SE_threshold_deg2": threshold, "policy": "smallest full-rank count within oneSE, calibration only, before any denseholdout scoring"}
    write_json(out/"summary.json", summary)
    print(f"75-pose A-aware corrected-cubic selection:{selected}", flush=True)


if __name__ == "__main__":
    main()
