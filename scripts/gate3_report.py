"""Gate 3 report: choose each method's step-size multiplier on val, report test there.

Usage: uv run python scripts/gate3_report.py <val run dir> <test run dir> [--json out.json]
(run dirs local or gs://)

Errors are averaged over the corruption streams (15 corruptions x 2 orders);
clean is reported apart. "share of Tent's gain" =
(no_adapt - method) / (no_adapt - tent), with Tent at its own chosen multiplier.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def load(run):
    run = run.rstrip("/")
    if run.startswith("gs://"):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["gcloud", "storage", "cp", f"{run}/results.json", f"{d}/r.json"],
                           check=True, capture_output=True)
            return json.loads(Path(d, "r.json").read_text())
    return json.loads(Path(run, "results.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("val")
    ap.add_argument("test")
    ap.add_argument("--json")
    args = ap.parse_args()
    val, test = load(args.val), load(args.test)
    mults = test["multipliers"]
    assert val["multipliers"] == mults

    def masks(r):
        groups = [s["group"] for s in r["streams"]]
        corrupt = np.array([g.startswith("imagenet_c/") for g in groups])
        return groups, corrupt

    _, v_corrupt = masks(val)
    t_groups, t_corrupt = masks(test)
    v_err = {m: np.array(e)[:, v_corrupt].mean(1) for m, e in val["error"].items()}      # (mults,)
    t_all = {m: np.array(e) for m, e in test["error"].items()}                           # (mults, streams)
    t_err = {m: e[:, t_corrupt].mean(1) for m, e in t_all.items()}
    ref = {k: np.array(v) for k, v in test["reference_error"].items()}
    no_adapt, bn_adapt = ref["no_adapt"][t_corrupt].mean(), ref["bn_adapt"][t_corrupt].mean()

    chosen = {m: int(np.argmin(v_err[m])) for m in v_err}
    tent_err = t_err["tent"][chosen["tent"]]
    rows = []
    for m in t_err:
        i = chosen[m]
        rows.append({"method": m, "multiplier": mults[i], "val_error": float(v_err[m][i]),
                     "test_error": float(t_err[m][i]),
                     "test_error_best_multiplier_on_test": float(t_err[m].min()),
                     "clean_error": float(t_all[m][i][~t_corrupt].mean()) if (~t_corrupt).any() else None,
                     "share_of_tent_gain": float((no_adapt - t_err[m][i]) / (no_adapt - tent_err)),
                     "skipped_updates": int(np.array(test["skipped_updates"][m])[i].sum())})

    print(f"ImageNet-C severity 5, test half of the images, mean over 15 corruptions x 2 orders")
    print(f"  no adaptation   {no_adapt:5.1f}")
    print(f"  BN-adapt        {bn_adapt:5.1f}")
    print(f"  {'method':12s} {'step x':>7s} {'test err':>9s} {'share of Tent gain':>19s} {'clean':>6s} {'(val err)':>10s}")
    for r in sorted(rows, key=lambda r: r["test_error"]):
        clean = f"{r['clean_error']:6.1f}" if r["clean_error"] is not None else "     -"
        print(f"  {r['method']:12s} {r['multiplier']:7g} {r['test_error']:9.1f} {r['share_of_tent_gain']:19.2f} "
              f"{clean} {r['val_error']:10.1f}" + (f"  [{r['skipped_updates']} skipped]" if r["skipped_updates"] else ""))
    print("  tent at the paper step size (x1):", f"{t_err['tent'][mults.index(1.0)]:.1f}")
    print("\n  test error at every multiplier (corrupt mean):")
    print(f"  {'method':12s} " + " ".join(f"{'x' + format(m, 'g'):>7s}" for m in mults))
    for m in t_err:
        print(f"  {m:12s} " + " ".join(f"{v:7.1f}" for v in t_err[m]))

    per_group = {}
    for m in t_err:
        i = chosen[m]
        per_group[m] = {g: float(np.mean([t_all[m][i][s] for s, gg in enumerate(t_groups) if gg == g]))
                        for g in sorted(set(t_groups))}
    if args.json:
        Path(args.json).write_text(json.dumps({"no_adapt": float(no_adapt), "bn_adapt": float(bn_adapt),
                                               "methods": rows, "per_group_test_error": per_group,
                                               "multipliers": mults}, indent=1))


if __name__ == "__main__":
    main()
