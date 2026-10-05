"""Pull single predictors out of saved runs into standalone files for Gate 3.

  recurrence:  a gate2_grid run's predictors.npz holds every predictor stacked
               on axis 0; take index j.
  dfa:         a gate2_predictability run's predictors.npz holds 'dfa_fit'.

Usage:
  uv run python scripts/extract_predictors.py rec <run_dir> <index> <rank> <out.npz>
  uv run python scripts/extract_predictors.py dfa <run_dir> <out.npz>
(run_dir local or gs://; out local or gs://)
"""

import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def read(path):
    if path.startswith("gs://"):
        with tempfile.TemporaryDirectory() as d:
            local = Path(d) / Path(path).name
            subprocess.run(["gcloud", "storage", "cp", path, str(local)], check=True, capture_output=True)
            return local.read_bytes()
    return Path(path).read_bytes()


def write(path, data: bytes):
    if path.startswith("gs://"):
        with tempfile.TemporaryDirectory() as d:
            local = Path(d) / Path(path).name
            local.write_bytes(data)
            subprocess.run(["gcloud", "storage", "cp", str(local), path], check=True, capture_output=True)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)


def save(path, arrays, meta):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    write(path, buf.getvalue())
    write(path.removesuffix(".npz") + ".json", json.dumps(meta, indent=1).encode())


def main():
    kind = sys.argv[1]
    run = sys.argv[2].rstrip("/")
    stacked = dict(np.load(io.BytesIO(read(f"{run}/predictors.npz"))))
    if kind == "rec":
        j, rank, out = int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
        arrays = {k: v[j] for k, v in stacked.items()}
        results = json.loads(read(f"{run}/results.json"))
        cfg = results["config"]
        meta = {"kind": "predictor", "arch": cfg.get("arch", "recurrence"), "rank": cfg.get("rank", rank),
                "lowrank_frac": cfg.get("lowrank_frac"), "source": run, "index": j,
                "config": results["predictors"][j]}
    else:
        out = sys.argv[3]
        prefix = "['dfa_fit']['"
        arrays = {k[len(prefix):-2]: v for k, v in stacked.items() if k.startswith(prefix)}
        meta = {"kind": "dfa", "source": run, "taps": len(arrays)}
        assert len(arrays) == 53, len(arrays)
    save(out, arrays, meta)
    print(f"wrote {out}: {len(arrays)} arrays; {meta}")


if __name__ == "__main__":
    main()
