"""Fit only the 15 P4 accommodation extension values on 75 known-A poses."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

from p4_a0_only_models import design_geometry
from p4_unified_a_gaze_model import OUTER, load_frozen_a0
from p4_unified_fullshape import UnifiedP4Field, self_check

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/p4_p1_limited_gaze_12p5/source_subset.npz"
A0_OBS = ROOT / "data/p4_p1_a0_only/calibration/a0_observations.npz"
A0_SUMMARY = ROOT / "data/p4_p1_a0_only/calibration/summary.json"
INDEPENDENT_CAL = ROOT / "data/p1_p4_a0_independent_gaze/calibration/calibration_data.npz"
OUT = ROOT / "data/p4_unified_combined/calibration/full_field"
STENCIL = np.array([-12.4, -6.4, 0., 6.4, 12.4])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coordinate_metrics(prediction, target):
    error_um = (np.asarray(prediction)-np.asarray(target))*1000.
    radial = np.linalg.norm(error_um, axis=-1)
    return {"coordinate_RMS_um": float(np.sqrt(np.mean(error_um*error_um))),
            "vertex_RMS_um": float(np.sqrt(np.mean(radial*radial))),
            "vertex_median_um": float(np.median(radial)),
            "vertex_P95_um": float(np.percentile(radial, 95)),
            "vertex_max_um": float(radial.max())}


def bounded_validity(model, extension):
    levels = np.array([-12.5, -6.25, 0., 6.25, 12.5])
    gaze = np.array([(ade, bde) for ade in levels for bde in levels])
    all_a = np.repeat([0., 2., 4.], len(gaze))
    all_g = np.tile(gaze, (3, 1))
    checked = model.evaluate(all_a, all_g, extension=extension, diagnostics=True)
    margins = ("denominator", "radial_tangential", "radial_radial", "brown_determinant")
    return {"probe_states": len(all_a), "A_D": [0., 2., 4.],
            "gaze_component_bound_deg": 12.5,
            "all_valid": bool(checked["valid"].all()),
            "invalid_count": int((~checked["valid"]).sum()),
            "minima": {name: float(np.min(checked[name])) for name in margins},
            "central_field_zero_max_mm": float(checked["central_field_zero_error_mm"].max()),
            "mean_B4_plus_A_effect_identity_max_mm": float(checked["mean_identity_error_mm"].max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    frozen = load_frozen_a0()
    checks = self_check(frozen)
    if args.check_only:
        print(json.dumps({"checks": checks}), flush=True)
        return
    summary_a0 = json.loads(A0_SUMMARY.read_text(encoding="utf-8"))
    if digest(SOURCE) != summary_a0["source_subset_sha256"] or digest(A0_OBS) != summary_a0["observation_artifact_sha256"]:
        raise RuntimeError("Frozen raw P4 observations changed")
    with np.load(SOURCE) as raw:
        states = raw["states"].copy()
        ids = raw["source_indices"].copy()
        grid = raw["absolute_p4_mm"].copy()
    gaze_cal = np.all(np.any(np.isclose(states[:, 1:, None], STENCIL, atol=1e-9, rtol=0), axis=2), axis=1)
    a_cal = np.any(np.isclose(states[:, 0, None], [0., 2., 4.], atol=1e-12, rtol=0), axis=1)
    cal = gaze_cal & a_cal
    if cal.sum() != 75:
        raise AssertionError("Expected exactly 75 known-A calibration observations")
    state, source_ids, observed = states[cal], ids[cal], grid[cal]
    with np.load(INDEPENDENT_CAL) as prior:
        prior_ids = prior["calibration_p4_source_indices"]
    if not np.array_equal(np.sort(source_ids[state[:, 0] == 0]), np.sort(prior_ids)):
        raise RuntimeError("A=0 P4 calibration source IDs differ from the frozen independent core")
    target = observed-observed[:, 4:5]
    model = UnifiedP4Field(frozen, np.zeros(15))

    def residual(values):
        predicted = model.evaluate(state[:, 0], state[:, 1:], extension=values)
        return ((predicted[:, OUTER]-target[:, OUTER])*1000.).ravel()

    def analytic_jacobian(values):
        checked = model.evaluate(state[:, 0], state[:, 1:], extension=values, extension_jac=True)
        return (checked["extension_jacobian"][:, OUTER]*1000.).reshape(-1, 15)

    started = time.perf_counter()
    fit = least_squares(residual, np.zeros(15), jac=analytic_jacobian,
                        x_scale="jac", ftol=1e-12, xtol=1e-12, gtol=1e-12,
                        max_nfev=300)
    wall = time.perf_counter()-started
    model.extension = fit.x.copy()
    evaluated = model.evaluate(state[:, 0], state[:, 1:], diagnostics=True)
    geometry = design_geometry(fit.jac/1000.)
    bounded = bounded_validity(model, fit.x)
    if (not fit.success or geometry["rank"] != 15 or not evaluated["valid"].all()
            or not bounded["all_valid"] or not np.isfinite(fit.x).all()):
        raise RuntimeError("15-value P4 extension failed fit/rank/regularity gate")
    with np.load(A0_OBS) as a0:
        a0_gaze = a0["gaze_deg"].copy()
    full_a0 = model.evaluate(np.zeros(len(a0_gaze)), a0_gaze)
    frozen_a0 = frozen.p4_relative_shape(a0_gaze)
    reduction_error = float(np.max(np.abs(full_a0-frozen_a0)))
    if reduction_error > 1e-12:
        raise RuntimeError("Accommodation extension changed the exact A=0 P4 shape")
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Fresh output directory required: {out}")
    out.mkdir(parents=True)
    artifact = out / "p4_full_field_27.npz"
    np.savez_compressed(artifact, extension15=fit.x, onaxis_calibration3=fit.x[:3],
                        interaction12=fit.x[3:], frozen_base12=model.base12,
                        reference_mm=frozen.p4_shape.reference_mm,
                        full_family_parameter_count=np.array(27), newly_fitted_parameter_count=np.array(15),
                        accommodation_degree=np.array(1), composition=np.array("projective_first"),
                        A_scale_d=np.array(4.), gaze_scale_deg=np.array(20.), length_mm=np.array(1.235429592627),
                        A0_PF12_source_sha256=np.array(frozen.source_hashes["P4_shape"]),
                        B4_source_sha256=np.array(frozen.source_hashes["P4_bias"]))
    np.savez_compressed(out / "calibration_diagnostics.npz", states=state, source_indices=source_ids,
                        observed_center_relative_mm=target, fitted_center_relative_mm=evaluated["prediction"],
                        signed_error_mm=evaluated["prediction"]-target,
                        Jacobian_physical_mm=fit.jac/1000.)
    summary = {"experiment": "Frozen A=0 PF12+B4 scaffold with 15 fitted accommodation field extensions",
        "known_A_calibration_poses": 75, "A_calibration_D": [0., 2., 4.],
        "frozen_base_shape_values": 12, "new_extension_values": 15,
        "full_field_family_values": 27,
        "extension_order": "s1,k1,k2,then12 A/4 x spatial interactions",
        "A0_coefficient_conversion": "frozen PF12 c_ij*(20/12.4)^(i+j)",
        "A0_reduction_max_abs_mm_all3969": reduction_error,
        "fit": {"success": bool(fit.success), "message": str(fit.message),
                "nfev": int(fit.nfev), "njev": int(fit.njev),
                "cost_um2_half": float(fit.cost), "optimality": float(fit.optimality),
                "seconds": wall, "Jacobian_physical_mm": geometry,
                "calibration_outer8": coordinate_metrics(evaluated["prediction"][:, OUTER], target[:, OUTER]),
                "calibration_all9": coordinate_metrics(evaluated["prediction"], target),
                "invalid_calibration_maps": int((~evaluated["valid"]).sum())},
        "bounded_validity": bounded,
        "artifact_path": str(artifact), "artifact_sha256": digest(artifact),
        "diagnostics_sha256": digest(out / "calibration_diagnostics.npz"),
        "raw_source_sha256": digest(SOURCE), "A0_observations_sha256": digest(A0_OBS),
        "frozen_component_hashes": frozen.source_hashes,
        "model_source_sha256": digest(ROOT / "Script/p4_unified_fullshape.py"),
        "trainer_source_sha256": digest(__file__), "checks": checks,
        "no_A0_base_refit": True, "no_G4_refit": True,
        "no_extra_A_bias_values": True}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "fit_success": bool(fit.success),
                      "rank": geometry["rank"],
                      "outer8_vertex_RMS_um": summary["fit"]["calibration_outer8"]["vertex_RMS_um"],
                      "A0_reduction_max_abs_mm": reduction_error,
                      "bounded_valid": bounded["all_valid"], "seconds": wall}, indent=2), flush=True)


if __name__ == "__main__":
    main()
