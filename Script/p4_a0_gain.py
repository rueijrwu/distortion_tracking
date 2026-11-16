r"""A=0 gaze/gain diagnostics and fixed E2 closure; no camera fit or CODE V.

Run in PowerShell (from any directory):
& 'C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe' 'C:\Users\rueijrwu\OneDrive\Project\DistortionTracking\Script\p4_a0_gain.py'
"""
import sys

sys.dont_write_bytecode = True  # Imports must not write outside the output folder.
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from p4_a0_data import DEFAULT, EXPECTED, safe

ROOT = Path(__file__).resolve().parents[1]
F, K1, K2, TA = 425.137557657, 2.943551420, -106.482286730, 0.0228424594374
CX, CY = 0.0, 0.0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_a0(path, report):
    # Same read-only, single pickle load as p4_a0_data; minimal p4_fit checks below.
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
        report.update(input_sha256=digest.hexdigest(), input_bytes=stream.tell())
        stream.seek(0)
        data = pickle.load(stream)
    require(isinstance(data, np.ndarray) and data.shape == (21 * 101 * 101 * 9,),
            "Expected the full flat P4 structured ndarray")
    require(data.dtype.names == tuple(EXPECTED) and data.dtype.itemsize == 8 * len(EXPECTED),
            "P4 retrace schema mismatch")
    require(all(data.dtype.fields[k][0] == np.dtype("float64") for k in EXPECTED),
            "Every P4 field must be float64")
    meta = (data.dtype.metadata or {}).get("p4_retrace", {})
    require(meta.get("record_count") == len(data) and meta.get("profile") == "full",
            "Missing/inconsistent full-profile metadata")
    axes = meta.get("axes", {})
    angle = np.arange(101) * .25 - 12.5
    for key, expected in (("accommodation_d", np.arange(21) * .2),
                          ("ade_deg", angle), ("bde_deg", angle)):
        observed = np.asarray(axes.get(key, []))
        require(observed.shape == expected.shape and np.allclose(observed, expected, rtol=0, atol=1e-8),
                f"Metadata {key} axis mismatch")
    require(np.all(np.isfinite(data["accommodation_d"])), "Nonfinite accommodation values")
    selected = np.flatnonzero(data["accommodation_d"] == 0)  # Select VALUE, never filename/index.
    require(len(selected) == 101 * 101 * 9, "A=0 must contain 10,201 complete nine-ray states")
    order = np.lexsort(tuple(data[k][selected] for k in ("ray_index", "bde_index", "ade_index")))
    cube = data[selected[order]].reshape(101, 101, 9)  # Only A=0 rows are copied.
    del data
    for key in EXPECTED:
        require(np.all(np.isfinite(cube[key])), f"A=0 {key} contains nonfinite values")
    ai, bi, ri = np.arange(101)[:, None, None], np.arange(101)[None, :, None], np.arange(9)
    aindex = np.unique(cube["accommodation_index"])
    require(len(aindex) == 1 and aindex[0] == 0, "A=0 accommodation index inconsistent with metadata")
    for key, expected in (("ade_index", ai), ("bde_index", bi), ("ray_index", ri),
                          ("state_index", (aindex[0] * 101 + ai) * 101 + bi),
                          ("ade_deg", angle[:, None, None]), ("bde_deg", angle[None, :, None])):
        require(np.max(np.abs(cube[key] - expected)) <= (1e-8 if key.endswith("deg") else 0),
                f"A=0 {key} lattice/state grouping mismatch")
    fields = np.asarray(axes.get("field_xy_y_major", []), dtype=float)
    require(fields.shape == (9, 2) and len(np.unique(fields, axis=0)) == 9,
            "Expected nine distinct metadata field labels")
    for key, j in (("field_x_relative", 0), ("field_y_relative", 1)):
        require(np.max(np.abs(cube[key] - fields[:, j])) <= 1e-12, f"{key} metadata mismatch")
    for key in ("rayrsi_status", "center_rayrsi_status"):
        require(np.all(cube[key] == 0), f"A=0 {key} contains nonzero status")
    for axis in ("ade", "bde"):
        require(np.max(np.abs(cube[f"rc_{axis}_readback_deg"] - cube[f"{axis}_deg"])) <= 1e-7,
                f"{axis} readback mismatch")
    norm_error = np.max(np.abs(sum(cube[k] ** 2 for k in ("l16", "m16", "n16")) - 1))
    require(norm_error <= 2e-8 and np.all(cube["n16"] < 0), "S16 LMN norm/sign gate failed")
    require(np.max(np.abs(cube["cornea_ent_d_thi_readback_mm"])) <= 1e-8,
            "Expected zero Cornea_ENT_D thickness")
    require(max(np.max(np.abs(cube[k])) for k in ("x16", "y16", "z16")) <= 1e-6
            and np.ptp(cube["zsi"]) <= 1e-9, "Stop origin/sensor plane gate failed")
    cube.flags.writeable = False
    report["validation"] = dict(scope="Full schema/metadata; all A=0 values, states, labels, statuses and geometry",
                                A_selected_by="accommodation_d == 0", states=10201, rays=91809,
                                max_lmn_squared_norm_error=norm_error)
    return cube, fields


def metrics(error):
    e = np.asarray(error).reshape(-1, 2)
    r = np.linalg.norm(e, axis=1)
    return dict(states=len(e), signed_mean_xy=e.mean(0), rms_xy=np.sqrt(np.mean(e * e, axis=0)),
                p95_abs_xy=np.percentile(np.abs(e), 95, axis=0), max_abs_xy=np.abs(e).max(0),
                rms_norm=np.sqrt(np.mean(r * r)), p95_norm=np.percentile(r, 95), max_norm=r.max())


def scores(error, groups):
    return {name: metrics(error[mask]) for name, mask in groups.items()}


def jacobian(j):
    sv = np.linalg.svd(j, compute_uv=False)
    gain = np.trace(j) / 2
    return dict(J_tan_per_deg=j, diagonal_tan_per_deg=j.diagonal(), cross_xy_yx=j[[0, 1], [1, 0]],
                trace_half_tan_per_deg=gain, J_mm_per_deg=F * j, trace_half_mm_per_deg=F * gain,
                diagonal_anisotropy_abs_over_mean=abs(j[0, 0] - j[1, 1]) / abs(gain),
                singular_values_tan_per_deg=sv, sv_anisotropy_range_over_mean=np.ptp(sv) / sv.mean())


def analyze(cube, fields, report):
    classes = np.count_nonzero(fields, axis=1)
    center, edge, corner = classes == 0, classes == 1, classes == 2
    require((center.sum(), edge.sum(), corner.sum()) == (1, 4, 4), "Expected centre/edge/corner = 1/4/4")
    ci = int(np.flatnonzero(center)[0])
    t = np.stack((-cube["l16"] / cube["n16"], -cube["m16"] / cube["n16"]), axis=-1)
    sensor = np.stack((cube["xsi"], cube["ysi"]), axis=-1)
    w = np.stack((cube["bde_deg"][..., 0], -cube["ade_deg"][..., 0]), axis=-1)
    zero = np.all(w == 0, axis=-1)
    require(zero.sum() == 1 and zero[50, 50], "Zero gaze must be index (50,50)")
    ae, be = cube["ade_index"][..., 0] % 2 == 0, cube["bde_index"][..., 0] % 2 == 0
    train = ae & be & ~zero
    groups = dict(train_even_even=train, holdout_even_odd=ae & ~be, holdout_odd_even=~ae & be,
                  holdout_odd_odd=~ae & ~be, zero_reference=zero, all=np.ones_like(zero))
    require([int(groups[k].sum()) for k in groups] == [2600, 2550, 2550, 2500, 1, 10201],
            "Unexpected state partition counts")
    report["conventions"] = dict(w="(BDE,-ADE), degrees; indices in ADE/BDE order",
        split="Whole nine-ray states; zero is reference only; local J uses all nonzero gazes within Euclidean 2.5 deg",
        residual="prediction minus observation; RMS/P95/max norm are Euclidean; xy metrics are per component",
        sensor="Recorded xsi,ysi (S16 frame), mm; error reports in um",
        gain_mm="Equivalent linear sensor gain f*J, not a fitted sensor Jacobian",
        camera="Fixed E2: x=cx+f*u*(1+k1*r^2+k2*r^4), y=cy+f*v*(1+k1*r^2+k2*r^4); "
               "u=tx=-l16/n16, v=ty=-m16/n16, r^2=u^2+v^2; no camera parameters fitted")
    report["constants"] = dict(f_mm=F, k1=K1, k2=K2, cx_mm=CX, cy_mm=CY, recorded_T_a=TA)
    ta = np.max(np.abs(t[train]))
    report["training_tangent_check"] = dict(T_a=ta, recorded=TA, difference=ta - TA,
        max_abs_tx_ty=np.max(np.abs(t[train]), axis=(0, 1)), states=2600, rays=23400)
    p = t[50, 50]
    shift = t[..., ci, :] - p[ci]
    report["zero_gaze"] = dict(field_labels=fields, center_ray_index=ci, p_i=p, p_0=p[ci], sensor_i_mm=sensor[50, 50])
    jfd = np.column_stack(((shift[50, 52] - shift[50, 48]) / (w[50, 52, 0] - w[50, 48, 0]),
                           (shift[48, 50] - shift[52, 50]) / (w[48, 50, 1] - w[52, 50, 1])))
    local = (np.linalg.norm(w, axis=-1) <= 2.5) & ~zero
    coef, _, rank, sv = np.linalg.lstsq(w[local], shift[local], rcond=None)
    report["jacobians"] = dict(symmetric_index_50_plus_minus_2=jacobian(jfd),
        local_linear=dict(**jacobian(coef.T), states=int(local.sum()), rank=rank, singular_values=sv,
                          residual_tan=metrics(w[local] @ coef - shift[local])))
    wx, wy = w[..., 0], w[..., 1]
    pred = np.empty_like(shift)
    fits = {}
    for j, design, names in ((0, np.stack((wx, wy, wx**3, wx * wy**2), -1), ("ax", "Jxy", "sx3", "sxy")),
                             (1, np.stack((wx, wy, wy**3, wy * wx**2), -1), ("Jyx", "ay", "sy3", "syx"))):
        coef, _, rank, sv = np.linalg.lstsq(design[train], shift[..., j][train], rcond=None)
        pred[..., j] = design @ coef
        fits["xy"[j]] = dict(coefficients=dict(zip(names, coef)), rank=rank, singular_values=sv)
    report["cubic_shift_fit"] = dict(units="Linear tan/deg; cubic tan/deg^3; zero intercept", axes=fits,
        residual_tan=scores(pred - shift, groups), residual_equivalent_um=scores(F * 1000 * (pred - shift), groups))
    report["parity"] = dict(reversal_w_mapping_max_deg=np.max(np.abs(w + w[::-1, ::-1])),
        S_w_plus_S_minus_w_tan=scores(shift + shift[::-1, ::-1], groups),
        x_odd_wx_defect_tan=metrics(np.stack((shift[..., 0] + shift[:, ::-1, 0],
                                            shift[..., 1] - shift[:, ::-1, 1]), -1)),
        y_odd_wy_defect_tan=metrics(np.stack((shift[..., 0] - shift[::-1, :, 0],
                                            shift[..., 1] + shift[::-1, :, 1]), -1)))
    r2 = np.sum(t * t, axis=-1)
    predicted_sensor = np.array((CX, CY)) + F * t * (1 + K1 * r2 + K2 * r2**2)[..., None]
    closure_um = 1000 * (predicted_sensor - sensor)
    closure_scores = scores(closure_um, groups)
    for name, values in closure_scores.items():
        values["rays"] = values.pop("states")
        values["states"] = int(groups[name].sum())
    closure_norm = np.linalg.norm(closure_um, axis=-1)
    loc = np.unravel_index(np.argmax(closure_norm), closure_norm.shape)
    report["e2_forward_closure"] = dict(units="um; prediction minus traced sensor; all nine rays per state",
        per_axis_metrics="signed_mean_xy is bias; rms_xy is RMS; p95_abs_xy/max_abs_xy use absolute residuals",
        summary_um=closure_scores, max_norm_um=closure_norm[loc],
        max_norm_location=dict(ade_index=loc[0], bde_index=loc[1], ray_index=int(cube["ray_index"][loc]),
            ade_deg=cube["ade_deg"][loc], bde_deg=cube["bde_deg"][loc], field_label=fields[loc[2]],
            tangent=t[loc], predicted_sensor_mm=predicted_sensor[loc], traced_sensor_mm=sensor[loc],
            signed_residual_xy_um=closure_um[loc]))
    report["camera_term_sizes"] = {}
    for name, term in (("k1", F * K1 * t * r2[..., None] * 1000),
                       ("k2", F * K2 * t * r2[..., None]**2 * 1000)):
        norm = np.linalg.norm(term, axis=-1)
        loc = np.unravel_index(np.argmax(norm), norm.shape)
        report["camera_term_sizes"][name] = dict(units="um; signed vectors, absolute component/norm maxima",
            zero_edge_vectors=term[50, 50, edge], zero_corner_vectors=term[50, 50, corner],
            zero_edge_norms=norm[50, 50, edge], zero_corner_norms=norm[50, 50, corner],
            all_max_abs_xy=np.max(np.abs(term), axis=(0, 1, 2)), all_max_norm=norm[loc],
            max_norm_at_ade_bde_ray_index=loc, max_norm_at_tangent=t[loc])
    report["per_gaze"] = dict(shape=[101, 101], axes="ADE index, BDE index, xy component; all at A=0",
        ade_deg=cube["ade_deg"][..., 0], bde_deg=cube["bde_deg"][..., 0], w_deg=w, S_tan=shift,
        center_sensor_mm=sensor[..., ci, :])


def main():
    path, out = (ROOT / DEFAULT).resolve(), ROOT / "runs" / "p4_gain"
    targets = [out / "report.txt", out / "report.json"]
    require(not any(path == p.resolve() or (p.exists() and path.samefile(p)) for p in targets),
            "Input aliases an output; refusing to overwrite")
    report = dict(input=str(path), numpy_version=np.__version__)
    try:
        cube, fields = load_a0(path, report)
        analyze(cube, fields, report)
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
    report = safe(report)
    text = "\n\n".join(f"{k}:\n{json.dumps(v, indent=2, allow_nan=False)}"
                       for k, v in report.items() if k != "per_gaze") + "\n"
    if "per_gaze" in report:
        text += "\nPer-gaze centre-ray shifts and sensor positions are included in report.json.\n"
    out.mkdir(parents=True, exist_ok=True)
    targets[0].write_text(text, encoding="utf-8")
    targets[1].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(text, end="")
    return int(report["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
