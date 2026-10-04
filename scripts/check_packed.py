"""Sanity-check packed shards: counts, labels, image ids, and that JPEGs decode.

Usage: uv run --group prep scripts/check_packed.py packed
"""

import io
import json
import sys
from collections import Counter
from pathlib import Path

from array_record.python.array_record_module import ArrayRecordReader
from PIL import Image

from sg.data.records import decode


def check_group(group_dir: Path, info: dict, expect_size=None, decode_n: int = 16) -> list[str]:
    errors, labels, ids = [], Counter(), set()
    for shard in info["shards"]:
        reader = ArrayRecordReader(str(group_dir / shard["file"]))
        n = reader.num_records()
        if n != shard["records"]:
            errors.append(f"{shard['file']}: {n} records, manifest says {shard['records']}")
        for j, rec in enumerate(reader.read_all()):
            label, image_id, jpeg = decode(rec)
            labels[label] += 1
            ids.add(image_id)
            if j < decode_n:
                img = Image.open(io.BytesIO(jpeg))
                img.load()
                if expect_size and img.size != expect_size:
                    errors.append(f"image {image_id} is {img.size}, expected {expect_size}")
        reader.close()
    if len(ids) != info["records"]:
        errors.append(f"{len(ids)} unique image ids for {info['records']} records")
    if min(labels) != 0 or max(labels) != 999 or len(labels) != 1000:
        errors.append(f"labels cover {len(labels)} classes")
    return errors


def main():
    out = Path(sys.argv[1])
    bad = 0
    for mf in sorted((out / "manifests").glob("*.json")):
        for name, info in json.loads(mf.read_text())["groups"].items():
            is_c = "/" in name
            group_dir = out / ("imagenet_c" if is_c else "") / name
            errors = check_group(group_dir, info, expect_size=(224, 224) if is_c else None)
            print(f"{'FAIL' if errors else 'ok  '} {name}: {info['records']} records", *errors, sep="\n  ")
            bad += bool(errors)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
