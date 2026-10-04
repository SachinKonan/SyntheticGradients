"""Record format shared by the packer and the loaders.

One record = 8-byte header + original JPEG bytes. Keeping the JPEG bytes as
released means ImageNet-C pixels are exactly the official ones.

  header: label (int32, sorted-wnid index = torchvision order), image_id
  (uint32, the ILSVRC2012 val number 1..50000).
"""

import struct

_HEADER = struct.Struct("<iI")


def encode(label: int, image_id: int, jpeg: bytes) -> bytes:
    return _HEADER.pack(label, image_id) + jpeg


def decode(record: bytes) -> tuple[int, int, bytes]:
    label, image_id = _HEADER.unpack_from(record)
    return label, image_id, record[_HEADER.size:]


def val_image_id(filename: str) -> int:
    """'ILSVRC2012_val_00003014.JPEG' -> 3014."""
    return int(filename.rsplit("_", 1)[1].split(".")[0])
