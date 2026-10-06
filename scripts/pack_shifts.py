"""Pack natural-shift test sets into the same ArrayRecord format as ImageNet-C.

  imagenet_r       ImageNet-R (30k renditions, 200 classes), Hugging Face parquet mirror
  imagenet_sketch  ImageNet-Sketch (50,889 sketches, 1000 classes), Hugging Face zip
  imagenet_v2      ImageNetV2 matched-frequency (10k photos, 1000 classes), tar

Labels are sorted-wnid indices (torchvision order); image_id is a running index.
Images keep their original bytes and size; loaders resize (256) and center-crop (224).
ImageNet-R also gets class_subset.json: the 200 label indices to restrict logits to.

Usage: uv run --group prep --with pyarrow scripts/pack_shifts.py <r|sketch|v2> <source> --out <packed root>
"""

import argparse
import io
import json
import tarfile
import zipfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from scripts.pack_imagenet import VAL_H5, write_group
from sg.data.records import encode


def wnid_list():
    with h5py.File(VAL_H5, "r") as h:
        return [w.decode() if isinstance(w, bytes) else str(w) for w in h.attrs["class_names"]]


def from_r(source: Path, label_of):
    import pyarrow.parquet as pq
    for f in sorted(source.glob("*.parquet")):
        for batch in pq.ParquetFile(f).iter_batches(columns=["image", "wnid"], batch_size=512):
            for img, wnid in zip(batch.column("image").to_pylist(), batch.column("wnid").to_pylist()):
                yield label_of[wnid], img["bytes"]


def from_sketch(source: Path, label_of):
    with zipfile.ZipFile(source) as z:
        for name in sorted(z.namelist()):
            parts = name.split("/")
            if name.endswith("/") or len(parts) < 2 or parts[-2] not in label_of:
                continue
            yield label_of[parts[-2]], z.read(name)


def from_v2(source: Path, _):
    with tarfile.open(source, "r") as t:
        for m in t:
            if m.isfile() and not m.name.split("/")[-1].startswith("."):
                yield int(m.name.split("/")[-2]), t.extractfile(m).read()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["r", "sketch", "v2"])
    p.add_argument("source", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    wnids = wnid_list()
    label_of = {w: i for i, w in enumerate(wnids)}
    group = {"r": "imagenet_r", "sketch": "imagenet_sketch", "v2": "imagenet_v2"}[args.kind]
    reader = {"r": from_r, "sketch": from_sketch, "v2": from_v2}[args.kind]
    records, labels = [], []
    for i, (label, data) in enumerate(reader(args.source, label_of)):
        records.append(encode(label, i, data))
        labels.append(label)
    info = write_group(records, args.out / group, group)
    classes = sorted(set(labels))
    if args.kind == "r":
        assert len(classes) == 200, len(classes)
        (args.out / group / "class_subset.json").write_text(json.dumps(classes))
    else:
        assert len(classes) == 1000, len(classes)

    # Spot-check that images decode.
    rng = np.random.default_rng(0)
    for j in rng.choice(len(records), 32, replace=False):
        Image.open(io.BytesIO(records[j][8:])).convert("RGB")
    manifest = {"source": str(args.source), "classes": len(classes), "groups": {group: info}}
    (args.out / "manifests").mkdir(parents=True, exist_ok=True)
    (args.out / "manifests" / f"{group}.json").write_text(json.dumps(manifest, indent=1))
    print(f"{group}: {len(records)} images, {len(classes)} classes, {len(info['shards'])} shards")


if __name__ == "__main__":
    main()
