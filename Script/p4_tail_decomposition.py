"""Read saved P4 artifacts and report worst-state gaze errors."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "p4_tail_decomposition"
SUMMARY = ROOT / "data/p4_unified_combined/calibration/full_field_profile_final/summary.json"
EVAL = ROOT / "data/p4_eval_ablation/evaluation.npz"
CORRECTED = ROOT / "data/p4_m1_pivot_fix/variant_physical_M1_corrected.npz"
CORRECTED_EVAL = ROOT / "data/p4_m1_pivot_fix/evaluation.npz"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    metric_rows = []
    def walk(obj, path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                p = f"{path}.{key}" if path else key
                if any(s in key.lower() for s in ("residual", "rms", "max", "loo")):
                    metric_rows.append((p, json.dumps(value, ensure_ascii=False, separators=(",", ":"))))
                walk(value, p)
    walk(summary)

    ev = np.load(EVAL, allow_pickle=False)
    corrected = np.load(CORRECTED, allow_pickle=False)
    corrected_eval = np.load(CORRECTED_EVAL, allow_pickle=False)
    truth = ev["truth_state_scoring_only"]
    source = ev["source_indices"]
    primary = ev["primary_mask"]
    variants = {
        "oracle_true_A": (ev["oracle_true_A_estimated_state"], ev["oracle_true_A_gaze_error_arcmin_scoring_only"]),
        "physical_M1_corrected": (corrected["estimated_state"], corrected_eval["physical_M1_corrected_gaze_error_arcmin_scoring_only"]),
    }
    report = ["# P4 tail decomposition", "", "Noise free, saved arrays only. No closed loop, CODE V, or refit was run.",
              "", "## Task 1: summary metric keys and values", "", "| Key path | Verbatim JSON value |", "|---|---|"]
    report += [f"| `{k}` | `{v}` |" for k, v in metric_rows]
    report += ["", "## Task 2: worst gaze-error states", "",
               "Error-vector components are estimated minus true ADE/BDE in arcmin; radial magnitude is the Euclidean norm in arcmin. Both requested scopes are shown: all 83,250 saved rows, then the 83,229 rows selected by `primary_mask`. The 21 excluded diagnostic rows produce the much larger all-row tail; the reported oracle maximum of 3.11 arcmin is the primary-mask maximum. Scalar ranking values are saved scoring fields.", ""]
    tables = {}
    for name, (states, saved_error) in variants.items():
        vec = (states[:, 1:3] - truth[:, 1:3]) * 60.0
        radial = saved_error
        tables[name+"_all"]=[]
        tables[name+"_primary"]=[]
        report += [f"### {name}", ""]
        for scope, mask in (("All saved rows", np.ones(len(source), dtype=bool)), ("Primary mask", primary)):
            ids = np.flatnonzero(mask)[np.argsort(radial[mask], kind="stable")[-10:][::-1]]
            report += [f"#### {scope}", "", "| Rank | source_indices | A true (D) | ADE true (deg) | BDE true (deg) | gaze error vector (ADE,BDE; arcmin) | radial gaze error (arcmin) |", "|---:|---:|---:|---:|---:|---|---:|"]
            rows=[]
            for rank, i in enumerate(ids, 1):
                row=[rank,int(source[i]),float(truth[i,0]),float(truth[i,1]),float(truth[i,2]),float(vec[i,0]),float(vec[i,1]),float(radial[i])]
                rows.append(row)
                report.append(f"| {rank} | {source[i]} | {truth[i,0]:.12g} | {truth[i,1]:.12g} | {truth[i,2]:.12g} | ({vec[i,0]:.12g}, {vec[i,1]:.12g}) | {radial[i]:.12g} |")
            tables[name+("_all" if scope=="All saved rows" else "_primary")]=np.array(rows,dtype=float)
            report.append("")
    report += ["## Component split", "", "A clean additive split of the oracle residual into B3 centre, PF12 field, and A-terms is not supported by these saved arrays. `evaluation.npz` stores end-to-end closed-loop outputs for each identity ablation; those rows change the recovered trajectory and do not store component-level gaze-vector contributions or a fixed-state additive accounting. Combining their residual differences would therefore not identify independent contributions. Stopped here without constructing a decomposition.",
               "", "## Exact array keys used", "", "- `evaluation.npz`: `truth_state_scoring_only`, `source_indices`, `primary_mask`, `oracle_true_A_estimated_state`, `oracle_true_A_gaze_error_arcmin_scoring_only`.",
               "- `variant_physical_M1_corrected.npz`: `estimated_state`; `p4_m1_pivot_fix/evaluation.npz`: `physical_M1_corrected_gaze_error_arcmin_scoring_only`.",
               "- JSON traversal: all paths whose key name contains `residual`, `rms`, `max`, or `loo` (case-insensitive).", "",
               f"Script SHA-256: `{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}`", ""]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    np.savez_compressed(OUT / "tail_tables.npz", **tables)


if __name__ == "__main__":
    main()
