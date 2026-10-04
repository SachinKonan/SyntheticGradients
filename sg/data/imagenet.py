"""Read packed ImageNet / ImageNet-C shards (see scripts/pack_imagenet.py).

Layout under a root (local dir or mounted copy of the GCS prefix):
  imagenet_val/shard-*.array_record                      original-size JPEGs
  imagenet_c/<corruption>/<severity>/shard-*.array_record  224x224 JPEGs

Preprocessing matches torchvision eval: clean val is resized (short side 256,
bilinear) and center-cropped to 224; ImageNet-C is already 224x224. Both are
then normalized with the ImageNet mean/std. Output is float32 NHWC.
"""

import io
from pathlib import Path

import numpy as np
from array_record.python.array_record_module import ArrayRecordReader
from PIL import Image

from sg.data.records import decode
from sg.models.resnet import IMAGENET_MEAN, IMAGENET_STD


def shard_paths(root: str | Path, group: str) -> list[Path]:
    """group: 'imagenet_val' or e.g. 'imagenet_c/gaussian_noise/5'."""
    paths = sorted(Path(root, group).glob("shard-*.array_record"))
    assert paths, f"no shards under {Path(root, group)}"
    return paths


def resize_center_crop(img: Image.Image, resize=256, crop=224) -> Image.Image:
    w, h = img.size
    scale = resize / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    w, h = img.size
    left, top = round((w - crop) / 2), round((h - crop) / 2)
    return img.crop((left, top, left + crop, top + crop))


def preprocess(jpeg: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    if img.size != (224, 224):
        img = resize_center_crop(img)
    x = np.asarray(img, np.float32) / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def read_group(root, group, keep=None, limit=None):
    """Yield (image, label, image_id) in stored (pre-shuffled) order.

    keep: optional predicate on image_id, e.g. to select one half of val.
    """
    n = 0
    for path in shard_paths(root, group):
        reader = ArrayRecordReader(str(path))
        for rec in reader.read_all():
            label, image_id, jpeg = decode(rec)
            if keep is not None and not keep(image_id):
                continue
            yield preprocess(jpeg), label, image_id
            n += 1
            if limit is not None and n >= limit:
                reader.close()
                return
        reader.close()


def batches(examples, batch_size):
    """Group (image, label, image_id) tuples into stacked numpy batches; drops the tail."""
    buf = []
    for ex in examples:
        buf.append(ex)
        if len(buf) == batch_size:
            x, y, ids = zip(*buf)
            yield np.stack(x), np.array(y, np.int32), np.array(ids, np.int32)
            buf = []
