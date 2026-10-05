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

TEST_CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
)
EXTRA_CORRUPTIONS = ("speckle_noise", "gaussian_blur", "spatter", "saturate")


def shard_paths(root: str | Path, group: str) -> list[Path]:
    """group: 'imagenet_val' or e.g. 'imagenet_c/gaussian_noise/5'."""
    paths = sorted(Path(root, group).glob("shard-*.array_record"))
    assert paths, f"no shards under {Path(root, group)}"
    return paths


def resize_center_crop(img: Image.Image, resize=256, crop=224) -> Image.Image:
    """torchvision Resize(256) + CenterCrop(224) on a PIL image."""
    w, h = img.size
    if w <= h:  # short side -> resize, long side truncated as in torchvision
        img = img.resize((resize, int(resize * h / w)), Image.BILINEAR)
    else:
        img = img.resize((int(resize * w / h), resize), Image.BILINEAR)
    w, h = img.size
    left, top = round((w - crop) / 2), round((h - crop) / 2)
    return img.crop((left, top, left + crop, top + crop))


def load_uint8(jpeg: bytes, resize: bool | None = None) -> np.ndarray:
    """Decode to a (224, 224, 3) uint8 image.

    resize: apply Resize(256) + CenterCrop(224) (clean val). ImageNet-C is
    stored at 224x224 and must not be resized. None: resize unless 224x224.
    """
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    if resize or (resize is None and img.size != (224, 224)):
        img = resize_center_crop(img)
    assert img.size == (224, 224), img.size
    return np.asarray(img, np.uint8)


def normalize(x_uint8):
    """uint8 NHWC -> normalized float32. Works on numpy or jax arrays."""
    return (x_uint8 / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def needs_resize(group: str) -> bool:
    """Clean val holds original-size JPEGs; ImageNet-C groups are already 224x224."""
    return group.startswith("imagenet_val")


def preprocess(jpeg: bytes, resize: bool | None = None) -> np.ndarray:
    return normalize(load_uint8(jpeg, resize).astype(np.float32))


def read_group(root, group, keep=None, limit=None):
    """Yield (image, label, image_id) in stored (pre-shuffled) order.

    keep: optional predicate on image_id, e.g. to select one half of val.
    """
    n, resize = 0, needs_resize(group)
    for path in shard_paths(root, group):
        reader = ArrayRecordReader(str(path))
        for rec in reader.read_all():
            label, image_id, jpeg = decode(rec)
            if keep is not None and not keep(image_id):
                continue
            yield preprocess(jpeg, resize), label, image_id
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
