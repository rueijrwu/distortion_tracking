#!/usr/bin/env python3
"""Describe P1 S8 tangents versus gaze; no tracing or image-position inversion.

Full mode is the default; --smoke studies a small symmetric gaze subsample while
retaining all nine rays and the FULL even/even training-only normalization T.
The sole polynomial operation is the requested fixed quadratic trend-residual
diagnostic. It does not fit a camera, a gaze inverse, or a replacement model.

For later execution use the project's configured absolute Python interpreter.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/p1_raytrace/p1_raytrace_251x251_12p5.pkl"
N = 251
COMPONENTS = ("tx", "ty", "u", "v")
FIELD_XY = np.array([
    [-1., 1.], [0., 1.], [1., 1.],
    [-1., 0.], [0., 0.], [1., 0.],
    [-1., -1.], [0., -1.], [1., -1.],
])


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(array):
    """Signed range and residual magnitudes, with no acceptance threshold."""
    x = np.asarray(array, dtype=float).ravel()
    if not x.size:
        return {"count": 0}
    if not np.all(np.isfinite(x)):
        raise ValueError("Nonfinite analysis values; refusing incomplete statistics")
    return {
        "count": int(x.size), "min": float(x.min()), "max": float(x.max()),
        "range": float(np.ptp(x)), "mean": float(x.mean()),
        "std": float(x.std()), "rms": float(np.sqrt(np.mean(x * x))),
        "median": float(np.median(x)), "p95_abs": float(np.percentile(np.abs(x), 95)),
        "max_abs": float(np.max(np.abs(x))),
    }


def component_stats(array):
    return {name: stats(array[..., c]) for c, name in enumerate(COMPONENTS)}


def per_ray_stats(array):
    return {str(r): component_stats(array[:, :, r, :]) for r in range(9)}


def load_source(expected_sha256=None):
    """Validate identity/layout, not scientific hypotheses, before any output."""
    digest = sha256(INPUT)
    print(f"Input SHA-256: {digest}; check {'enabled' if expected_sha256 is not None else 'skipped (no --expected-sha256)'}", flush=True)
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        raise ValueError(f"P1 input SHA-256 mismatch: {digest}; expected {expected_sha256}")
    data = np.load(INPUT, allow_pickle=True)
    if not isinstance(data, np.ndarray) or data.dtype.names is None:
        raise ValueError("Expected the audited structured NumPy array")
    if data.shape != (N * N * 9,):
        raise ValueError(f"Expected {N}x{N}x9 flat records; got {data.shape}")
    required = (
        "ade_deg", "bde_deg", "state_index", "ray_index",
        "field_x_normalized", "field_y_normalized", "rayrsi_status",
        "l8", "m8", "n8",
    )
    missing = set(required).difference(data.dtype.names)
    if missing:
        raise ValueError(f"Missing required P1 columns: {sorted(missing)}")
    for name in required:
        if not np.all(np.isfinite(data[name])):
            raise ValueError(f"Nonfinite required input column: {name}")
    cube = data.reshape(N, N, 9)  # [BDE index, ADE index, ray index]
    axis = np.array([round(-12.5 + i * 0.1, 10) for i in range(N)])
    layout = {
        "ADE_axis": np.allclose(cube["ade_deg"], axis[None, :, None], rtol=0, atol=1e-10),
        "BDE_axis": np.allclose(cube["bde_deg"], axis[:, None, None], rtol=0, atol=1e-10),
        "state_index": np.array_equal(
            cube["state_index"], np.broadcast_to(np.arange(N * N).reshape(N, N, 1), cube.shape)),
        "ray_index": np.array_equal(
            cube["ray_index"], np.broadcast_to(np.arange(9), cube.shape)),
        "field_x": np.array_equal(
            cube["field_x_normalized"], np.broadcast_to(FIELD_XY[:, 0], cube.shape)),
        "field_y": np.array_equal(
            cube["field_y_normalized"], np.broadcast_to(FIELD_XY[:, 1], cube.shape)),
    }
    if not all(layout.values()):
        raise ValueError(f"Audited gaze/ray layout mismatch: {layout}")
    if np.any(cube["n8"] == 0):
        raise ValueError("Zero n8 makes the requested S8 tangent undefined")
    # S8 tangent convention, matching E1: tx=l8/n8 and ty=m8/n8.
    tangents = np.stack((cube["l8"] / cube["n8"], cube["m8"] / cube["n8"]), axis=-1)
    if not np.all(np.isfinite(tangents)):
        raise ValueError("Requested S8 tangents are not finite")
    active = cube["rayrsi_status"] == 0
    even = np.arange(N) % 2 == 0
    train_states = even[:, None] & even[None, :]
    train_active = train_states[:, :, None] & active
    if not np.any(train_active):
        raise ValueError("No active even/even training rays available for T")
    scale = float(np.max(np.abs(tangents[train_active])))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training-only T must be finite and positive")
    statuses, counts = np.unique(cube["rayrsi_status"], return_counts=True)
    info = {
        "path": str(INPUT), "sha256": digest, "expected_sha256": expected_sha256,
        "sha256_matches": True if expected_sha256 is not None else None,
        "sha256_check": "verified" if expected_sha256 is not None else "skipped",
        "expected_sha256_source": "--expected-sha256" if expected_sha256 is not None else None,
        "shape_BDE_ADE_ray": list(cube.shape), "columns": list(data.dtype.names),
        "layout_checks": {k: bool(v) for k, v in layout.items()},
        "status_counts": {str(float(s)): int(c) for s, c in zip(statuses, counts)},
        "n8": stats(cube["n8"]),
        "science_observations_only": {
            "all_status_zero": bool(np.all(active)),
            "all_n8_negative": bool(np.all(cube["n8"] < 0)),
            "LMN_norm_minus_one": stats(np.sqrt(
                cube["l8"] ** 2 + cube["m8"] ** 2 + cube["n8"] ** 2) - 1),
        },
    }
    normalization = {
        "T": scale, "formula": "T=max(abs(tx),abs(ty)) over active even/even training rays",
        "active_definition": "rayrsi_status == 0; all required inputs finite and n8 nonzero",
        "training_definition": "original ADE index even AND original BDE index even",
        "scope": "full input training lattice, including in smoke mode; no holdout maxima",
        "training_states": int(train_states.sum()),
        "holdout_states": int((~train_states).sum()),
        "training_rays": int(train_states.sum() * 9),
        "holdout_rays": int((~train_states).sum() * 9),
        "active_training_rays_used_for_T": int(train_active.sum()),
    }
    return axis, tangents, active, train_states, scale, info, normalization


def quadratic_residual(values, ade, bde):
    """Fixed six-term in-sample polynomial projection, solely to report residuals.

    All sampled points describe the trend; there is no holdout scoring, selection,
    camera fit, inverse model or production coefficient export in this study.
    """
    a, b = np.meshgrid(ade / np.max(np.abs(ade)), bde / np.max(np.abs(bde)))
    design = np.stack((np.ones_like(a), a, b, a * a, a * b, b * b), axis=-1).reshape(-1, 6)
    flat = values.reshape(design.shape[0], -1)
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, flat, rcond=None)
    residual = (flat - design @ coefficients).reshape(values.shape)
    return residual, {
        "purpose": "requested descriptive low-order polynomial residual only",
        "basis": ["1", "ADE/12.5", "BDE/12.5", "(ADE/12.5)^2", "ADE*BDE/12.5^2", "(BDE/12.5)^2"],
        "degree": 2, "terms_per_ray_per_component": 6,
        "scope": "in-sample fixed trend projection across all studied gaze states",
        "residual_sign": "traced value minus quadratic trend",
        "rank": int(rank), "singular_values": singular_values.tolist(),
        "coefficients_exported": False, "model_selection": False,
    }


def ray_permutation(transform):
    lookup = {tuple(xy): r for r, xy in enumerate(FIELD_XY)}
    return np.array([lookup[tuple(transform(x, y))] for x, y in FIELD_XY])


def symmetry_diagnostics(values, zero):
    """Report hypothesis residuals; never declare a symmetry from construction."""
    centered = values - zero[None, None, :, :]
    fixed = {}
    for name, axis in (("ADE", 1), ("BDE", 0)):
        fixed[name] = {
            "odd_raw_residual_f_minus_g_plus_f_g": per_ray_stats(np.flip(values, axis=axis) + values),
            "even_raw_residual_f_minus_g_minus_f_g": per_ray_stats(np.flip(values, axis=axis) - values),
            "odd_after_zero_offset_removal": per_ray_stats(np.flip(centered, axis=axis) + centered),
            "even_after_zero_offset_removal": per_ray_stats(np.flip(centered, axis=axis) - centered),
        }
    operations = (
        ("ADE_reflection", values[:, ::-1], lambda x, y: (-x, y), [-1, 1, -1, 1]),
        ("BDE_reflection", values[::-1], lambda x, y: (x, -y), [1, -1, 1, -1]),
        ("radial_inversion_180deg", values[::-1, ::-1], lambda x, y: (-x, -y), [-1, -1, -1, -1]),
        ("axis_exchange", values.swapaxes(0, 1), lambda x, y: (y, x), [1, 1, 1, 1]),
    )
    labelled = {}
    for name, grid, transform, signs in operations:
        perm = ray_permutation(transform)
        transformed = np.take(grid, perm, axis=2)
        expected = values[..., [1, 0, 3, 2]] if name == "axis_exchange" else values
        expected = expected * np.asarray(signs)
        residual = transformed - expected
        transformed_relative = transformed - transformed[:, :, 4:5, :]
        expected_relative = expected - expected[:, :, 4:5, :]
        item = {
            "transformed_ray_index_for_each_original_ray": perm.tolist(),
            "expected_component_order": ["ty", "tx", "v", "u"] if name == "axis_exchange" else list(COMPONENTS),
            "expected_component_signs": signs,
            "residual_definition": "value at transformed gaze AND transformed ray label minus signed original value",
            "raw_residual": per_ray_stats(residual),
            "central_relative_residual": per_ray_stats(transformed_relative - expected_relative),
            "nine_ray_common_mode_residual": component_stats(residual.mean(axis=2)),
            "eight_ray_mean_residual": component_stats(residual[:, :, np.arange(9) != 4, :].mean(axis=2)),
            "radial_uv_magnitude_residual": {
                str(r): stats(np.hypot(transformed[:, :, r, 2], transformed[:, :, r, 3])
                              - np.hypot(values[:, :, r, 2], values[:, :, r, 3]))
                for r in range(9)
            },
        }
        if name == "axis_exchange":
            # ADE/BDE are rotation labels; expose all sign hypotheses rather than
            # assuming their handedness corresponds to tangent component names.
            swapped = values[..., [1, 0, 3, 2]]
            item["signed_exchange_hypotheses"] = {
                f"x_sign_{sx:+d}_y_sign_{sy:+d}": per_ray_stats(
                    transformed - swapped * np.array([sx, sy, sx, sy]))
                for sx in (-1, 1) for sy in (-1, 1)
            }
        labelled[name] = item
    return {
        "fixed_ray_parity": fixed, "paired_ray_symmetry_hypotheses": labelled,
        "interpretation": "Discrete gaze/ray-label symmetry hypotheses on the square lattice; these residuals do not establish continuous rotational symmetry.",
    }


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def write_slices(path, values, relative, dade, dbde, residual, axis, indices, train, active, zero_index):
    columns = ["slice", "ADE_deg", "BDE_deg", "ADE_original_index", "BDE_original_index",
               "ray_index", "field_x_normalized", "field_y_normalized", "is_central_ray",
               "is_train_state", "is_active_ray"]
    for prefix in ("", "minus_central_", "d_dADE_", "d_dBDE_", "quadratic_residual_"):
        columns.extend(prefix + name for name in COMPONENTS)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for label in ("BDE=0", "ADE=0"):
            for i in range(len(axis)):
                bi, ai = (zero_index, i) if label == "BDE=0" else (i, zero_index)
                for r, (fx, fy) in enumerate(FIELD_XY):
                    row = [label, float(axis[ai]), float(axis[bi]), int(indices[ai]), int(indices[bi]),
                           r, float(fx), float(fy), r == 4, bool(train[bi, ai]), bool(active[bi, ai, r])]
                    for array in (values, relative, dade, dbde, residual):
                        row.extend(float(x) for x in array[bi, ai, r])
                    writer.writerow(row)


def run(smoke=False, output_dir=None, expected_sha256=None):
    axis_full, tangents, active_full, train_full, scale, source, normalization = load_source(expected_sha256)
    # This set is symmetric under i -> 250-i, contains zero gaze (125), and
    # includes both original parity classes. All nine labelled rays survive.
    indices = np.unique(np.r_[np.arange(0, N, 25), 124, 125, 126]) if smoke else np.arange(N)
    axis = axis_full[indices]
    zero_index = int(np.flatnonzero(axis == 0)[0])
    select = np.ix_(indices, indices)
    t = tangents[select]
    values = np.concatenate((t, t / scale), axis=-1)
    active = active_full[select]
    train = train_full[select]
    zero = values[zero_index, zero_index]
    relative = values - values[:, :, 4:5, :]
    relative_change = relative - relative[zero_index, zero_index][None, None, :, :]
    response = values - zero[None, None, :, :]
    common9 = values.mean(axis=2)
    common8 = values[:, :, np.arange(9) != 4, :].mean(axis=2)
    departures = response - response.mean(axis=2, keepdims=True)
    dbde, dade = np.gradient(values, axis, axis, axis=(0, 1), edge_order=2)
    d2ade = np.gradient(dade, axis, axis=1, edge_order=2)
    d2bde = np.gradient(dbde, axis, axis=0, edge_order=2)
    d2mixed = 0.5 * (np.gradient(dade, axis, axis=0, edge_order=2)
                     + np.gradient(dbde, axis, axis=1, edge_order=2))
    residual, polynomial_info = quadratic_residual(values, axis, axis)
    rays = []
    for r, (fx, fy) in enumerate(FIELD_XY):
        slope = {}
        for c, name in enumerate(COMPONENTS):
            slope[name] = {
                "d_dADE": stats(dade[:, :, r, c]), "d_dBDE": stats(dbde[:, :, r, c]),
                "d_dADE_at_zero": float(dade[zero_index, zero_index, r, c]),
                "d_dBDE_at_zero": float(dbde[zero_index, zero_index, r, c]),
            }
        rms_du_da = float(np.sqrt(np.mean(dade[:, :, r, 2] ** 2)))
        rms_du_db = float(np.sqrt(np.mean(dbde[:, :, r, 2] ** 2)))
        rms_dv_da = float(np.sqrt(np.mean(dade[:, :, r, 3] ** 2)))
        rms_dv_db = float(np.sqrt(np.mean(dbde[:, :, r, 3] ** 2)))
        rays.append({
            "ray_index": r, "field_xy_normalized": [float(fx), float(fy)],
            "central_ray_diagnostic_only": r == 4,
            "offset_at_zero_gaze": dict(zip(COMPONENTS, map(float, zero[r]))),
            "ranges": component_stats(values[:, :, r]),
            "axis_slices": {
                "BDE=0_vary_ADE": component_stats(values[zero_index, :, r]),
                "ADE=0_vary_BDE": component_stats(values[:, zero_index, r]),
                "csv": "axis_slices.csv",
            },
            "slopes_per_degree": slope,
            "curvature_per_degree_squared": {
                "d2_dADE2": component_stats(d2ade[:, :, r]),
                "d2_dBDE2": component_stats(d2bde[:, :, r]),
                "d2_dADE_dBDE": component_stats(d2mixed[:, :, r]),
            },
            "cross_axis_coupling": {
                "rms_du_dADE": rms_du_da, "rms_du_dBDE": rms_du_db,
                "rms_dv_dADE": rms_dv_da, "rms_dv_dBDE": rms_dv_db,
                "rms_du_dBDE_over_du_dADE": ratio(rms_du_db, rms_du_da),
                "rms_dv_dADE_over_dv_dBDE": ratio(rms_dv_da, rms_dv_db),
                "interpretation": "named-axis ratios only; no assumed physical main gaze axis",
            },
            "quadratic_trend_residual": component_stats(residual[:, :, r]),
            "minus_central_ray": component_stats(relative[:, :, r]),
            "minus_central_ray_change_from_zero_gaze": component_stats(relative_change[:, :, r]),
            "response_minus_nine_ray_common_response": component_stats(departures[:, :, r]),
        })
    parity_groups = {}
    ai, bi = np.meshgrid(indices, indices)
    for ade_parity in (0, 1):
        for bde_parity in (0, 1):
            mask = (ai % 2 == ade_parity) & (bi % 2 == bde_parity)
            parity_groups[f"ADE_{ade_parity}_BDE_{bde_parity}"] = {
                "states": int(mask.sum()), "rays": int(mask.sum() * 9),
                "role": "training" if ade_parity == bde_parity == 0 else "holdout",
                "ranges": component_stats(values[mask]),
            }
    summary = {
        "study": "P1 S8 tx,ty,u,v versus ADE/BDE; descriptive diagnostics only",
        "mode": "smoke" if smoke else "full", "input": source,
        "tangent_definition": {
            "surface": "S8", "column_names": {"l": "l8", "m": "m8", "n": "n8"},
            "formula": {"tx": "l8/n8", "ty": "m8/n8", "u": "tx/T", "v": "ty/T"},
            "documentation_note": "S8 tangent convention, same as E1: tx=l8/n8 and ty=m8/n8. No angle-derived tangents are used here.",
        },
        "normalization": normalization,
        "coordinates": {
            "array_axis_order": ["BDE", "ADE", "ray", "component"],
            "components": list(COMPONENTS), "gaze_unit": "degree",
            "component_units": {"tx": "dimensionless tangent", "ty": "dimensionless tangent", "u": "dimensionless normalized tangent", "v": "dimensionless normalized tangent"},
            "derivative_method": "numpy.gradient with actual degree coordinates; second-order one-sided edges",
            "smoke_derivative_note": "Smoke derivatives use the sparse nonuniform sampled grid; no science acceptance gates",
            "original_axis_indices": indices.tolist(), "ADE_deg": axis.tolist(), "BDE_deg": axis.tolist(),
            "central_ray_index": 4, "central_field_xy": [0, 0],
        },
        "accounting": {
            "studied_states": int(len(axis) ** 2), "studied_rays": int(values.size // 4),
            "active_studied_rays": int(active.sum()),
            "training_states": int(train.sum()), "holdout_states": int((~train).sum()),
            "original_index_parity_groups": parity_groups,
            "diagnostic_population": "all nine raw traced rays; nonzero statuses are reported, not silently excluded from diagnostics",
        },
        "per_ray": rays, "symmetry": symmetry_diagnostics(values, zero),
        "common_mode": {
            "nine_ray_mean": component_stats(common9), "eight_ray_mean_excluding_center": component_stats(common8),
            "nine_ray_mean_minus_central_ray": component_stats(common9 - values[:, :, 4]),
            "eight_ray_mean_minus_central_ray": component_stats(common8 - values[:, :, 4]),
            "interpretation": "Traced-ray diagnostics only; center, means and camera principal point are distinct. No measurement inputs or estimators are proposed.",
        },
        "polynomial_residual_diagnostic": polynomial_info,
        "science_gate_policy": "No scientific acceptance thresholds or science-triggered nonzero exit in either mode; input identity/usability and I/O errors remain fatal.",
        "outputs": {"summary": "summary.json", "arrays": "gaze_arrays.npz", "axis_slices": "axis_slices.csv"},
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    # No writes until input identity, layout, normalization, analysis and JSON
    # serialization succeed. In particular, no failed science hypothesis gates
    # publication or determines the process exit status.
    summary_text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    out = (Path(output_dir).resolve() if output_dir is not None else
           ROOT / "runs" / ("p1_uv_gaze_study_smoke" if smoke else "p1_uv_gaze_study"))
    if smoke and not out.name.endswith("_smoke"):
        out = out.with_name(out.name + "_smoke")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "gaze_arrays.npz", ADE_deg=axis, BDE_deg=axis,
        ADE_original_index=indices, BDE_original_index=indices,
        ray_index=np.arange(9), field_xy_normalized=FIELD_XY,
        component_names=np.array(COMPONENTS), T=np.array(scale),
        values=values, active_rays=active, training_states=train,
        offset_at_zero_gaze=zero, minus_central_ray=relative,
        minus_central_ray_change_from_zero_gaze=relative_change,
        d_dADE=dade, d_dBDE=dbde, d2_dADE2=d2ade, d2_dBDE2=d2bde,
        d2_dADE_dBDE=d2mixed, quadratic_residual=residual,
        nine_ray_mean=common9, eight_ray_mean_excluding_center=common8,
        BDE_zero_slice=values[zero_index], ADE_zero_slice=values[:, zero_index],
    )
    write_slices(out / "axis_slices.csv", values, relative, dade, dbde, residual,
                 axis, indices, train, active, zero_index)
    (out / "summary.json").write_text(summary_text, encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha256", help="Expected input SHA-256; default: skip hash check and print the hash found")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="13x13 gaze subsample; all nine rays; full training-only T")
    mode.add_argument("--full", action="store_true", help="full 251x251x9 diagnostic (default)")
    parser.add_argument("--output-dir", type=Path, help="Output folder; default: runs/p1_uv_gaze_study (smoke: runs/p1_uv_gaze_study_smoke)")
    args = parser.parse_args()
    try:
        out = run(smoke=args.smoke, output_dir=args.output_dir, expected_sha256=args.expected_sha256)
    except (OSError, ValueError, TypeError, KeyError, np.linalg.LinAlgError) as error:
        print(f"P1 study input/analysis/I/O error: {error}", file=sys.stderr)
        return 2
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
