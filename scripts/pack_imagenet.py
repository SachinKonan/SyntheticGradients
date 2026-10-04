"""Pack ImageNet-C tars and clean ImageNet val into shuffled ArrayRecord shards.

Runs on della (CPU only, no decoding). Output layout:

  <out>/imagenet_c/<corruption>/<severity>/shard-XXXXX-of-YYYYY.array_record
  <out>/imagenet_val/shard-XXXXX-of-YYYYY.array_record
  <out>/manifests/<source>.json

Kept by default: the 15 test corruptions at severity 5 and the 4 extra
(held-out) corruptions at all severities. Records within a group are shuffled
with a seed derived from the group name, so the output is deterministic.

Usage:
  uv run --group prep scripts/pack_imagenet.py imagenet-c raw/noise.tar --out packed
  uv run --group prep scripts/pack_imagenet.py val val.h5 --out packed
"""

import argparse
import hashlib
import json
import math
import tarfile
import zlib
from pathlib import Path

import h5py
import numpy as np
from array_record.python.array_record_module import ArrayRecordWriter

from sg.data.imagenet import EXTRA_CORRUPTIONS, TEST_CORRUPTIONS
from sg.data.records import encode, val_image_id
KEEP = {c: {5} for c in TEST_CORRUPTIONS} | {c: {1, 2, 3, 4, 5} for c in EXTRA_CORRUPTIONS}

VAL_H5 = "/scratch/gpfs/ZHUANGL/shared/imagenet-hdf5/val.h5"
RECORDS_PER_SHARD = 6250


def write_group(records: list[bytes], out_dir: Path, seed_key: str) -> dict:
    """Shuffle records deterministically and write them as ArrayRecord shards."""
    order = np.random.default_rng(zlib.crc32(seed_key.encode())).permutation(len(records))
    num_shards = math.ceil(len(records) / RECORDS_PER_SHARD)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    for s, idx in enumerate(np.array_split(order, num_shards)):
        path = out_dir / f"shard-{s:05d}-of-{num_shards:05d}.array_record"
        writer = ArrayRecordWriter(str(path), "group_size:1")
        for i in idx:
            writer.write(records[i])
        writer.close()
        shards.append({"file": path.name, "records": len(idx), "bytes": path.stat().st_size})
    return {"records": len(records), "shards": shards}


def pack_imagenet_c(tar_path: Path, out: Path, wnids: list[str]) -> dict:
    """Stream one ImageNet-C tar (corruption/severity/wnid/file) into shards.

    The tar is grouped by corruption/severity, so one group is buffered at a time.
    """
    label_of = {w: i for i, w in enumerate(wnids)}
    groups, done = {}, set()
    key, buf = None, []

    def flush():
        if key is not None and buf:
            corruption, severity = key
            groups[f"{corruption}/{severity}"] = write_group(
                buf, out / "imagenet_c" / corruption / str(severity), f"{corruption}/{severity}")
            print(f"  {corruption}/{severity}: {len(buf)} records", flush=True)

    with tarfile.open(tar_path, mode="r|*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            corruption, severity, wnid, fname = member.name.split("/")[-4:]
            severity = int(severity)
            if severity not in KEEP.get(corruption, ()):
                continue
            if (corruption, severity) != key:
                flush()
                key, buf = (corruption, severity), []
                assert key not in done, f"tar is not grouped by corruption/severity: {key}"
                done.add(key)
            buf.append(encode(label_of[wnid], val_image_id(fname), tar.extractfile(member).read()))
    flush()
    return groups


def pack_val(h5_path: Path, out: Path) -> dict:
    """Clean val: original JPEG bytes (not resized; the loader resizes/crops)."""
    with h5py.File(h5_path, "r") as h:
        images, labels, names = h["images"], h["labels"][:], h["filenames"][:]
        records = [
            encode(int(labels[i]), val_image_id(names[i].decode()), images[i].tobytes())
            for i in range(len(labels))
        ]
    return {"imagenet_val": write_group(records, out / "imagenet_val", "imagenet_val")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["imagenet-c", "val"])
    p.add_argument("source", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    with h5py.File(VAL_H5, "r") as h:
        wnids = [w.decode() if isinstance(w, bytes) else str(w) for w in h.attrs["class_names"]]
    assert len(wnids) == 1000 and wnids == sorted(wnids)

    if args.kind == "imagenet-c":
        groups = pack_imagenet_c(args.source, args.out, wnids)
    else:
        groups = pack_val(args.source, args.out)

    manifest = {
        "source": str(args.source),
        "wnids_sha256": hashlib.sha256("\n".join(wnids).encode()).hexdigest(),
        "groups": groups,
    }
    (args.out / "manifests").mkdir(parents=True, exist_ok=True)
    (args.out / "manifests" / f"{args.source.stem}.json").write_text(json.dumps(manifest, indent=1))
    print(f"done: {sum(g['records'] for g in groups.values())} records from {args.source}")


if __name__ == "__main__":
    main()
