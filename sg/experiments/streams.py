"""Independent test-time streams, one or more per device, stepped together.

Each host loads only the streams that live on its devices. Batches are decoded
to uint8 on the host (a thread pool, prefetched) and normalized on device.
"""

import concurrent.futures as cf
import os
import queue
import subprocess
import threading
from pathlib import Path

import jax
import numpy as np
from array_record.python.array_record_module import ArrayRecordReader
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from sg.data import imagenet
from sg.data.records import decode


def fetch(root: str, rel: str, cache: Path) -> Path:
    """Local path of root/rel. A gs:// root is copied into the cache once."""
    if not root.startswith("gs://"):
        return Path(root, rel)
    dst = cache / rel
    if not (dst / ".complete").exists():
        dst.mkdir(parents=True, exist_ok=True)
        subprocess.run(["gcloud", "storage", "cp", "-r", f"{root}/{rel}/*", str(dst)], check=True)
        (dst / ".complete").touch()
    return dst


def fetch_file(url: str, cache: Path) -> str:
    """Local copy of a gs:// file, cached by its full path (different runs reuse file names)."""
    if not url.startswith("gs://"):
        return url
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / "files" / url.removeprefix("gs://")
    local.parent.mkdir(parents=True, exist_ok=True)
    if not local.exists():
        subprocess.run(["gcloud", "storage", "cp", url, str(local)], check=True)
    return str(local)


class Stream:
    """The records of one group, in a fixed order.

    order 0 keeps the stored (pre-shuffled) order; order k > 0 is a permutation
    seeded by k. keep: optional predicate on the val image id. resize: decode
    with Resize(256) + CenterCrop(224) (clean val only).
    """

    def __init__(self, group_dir: Path, order: int, batch: int, keep=None, resize=False):
        self.resize = resize
        self.records = []
        for path in imagenet.shard_paths(group_dir, ""):
            reader = ArrayRecordReader(str(path))
            recs = reader.read_all()
            reader.close()
            self.records += recs if keep is None else [r for r in recs if keep(decode(r)[1])]
        n = len(self.records)
        self.perm = np.arange(n) if order == 0 else np.random.default_rng(order).permutation(n)
        self.batch = batch
        self.num_steps = n // batch

    def batch_records(self, step):
        return [self.records[i] for i in self.perm[step * self.batch:(step + 1) * self.batch]]


def prefetch_batches(streams, num_steps, workers, depth=3):
    """Yield (x uint8 (S, B, 224, 224, 3), y int32 (S, B)) for this host's streams."""
    q = queue.Queue(maxsize=depth)
    pool = cf.ThreadPoolExecutor(workers)

    def decode_one(item):
        rec, resize = item
        label, _, jpeg = decode(rec)
        return imagenet.load_uint8(jpeg, resize), label

    def run():
        try:
            for step in range(num_steps):
                items = [(r, s.resize) for s in streams for r in s.batch_records(step)]
                out = list(pool.map(decode_one, items))
                x = np.stack([o[0] for o in out]).reshape(len(streams), -1, 224, 224, 3)
                y = np.array([o[1] for o in out], np.int32).reshape(len(streams), -1)
                q.put((x, y))
            q.put(None)
        except BaseException as exc:  # surface loader errors instead of hanging the consumer
            q.put(exc)

    threading.Thread(target=run, daemon=True).start()
    while (item := q.get()) is not None:
        if isinstance(item, BaseException):
            raise item
        yield item


class StreamMesh:
    """A device mesh over streams, and this host's share of them.

    batch_split > 1 spreads each stream's batch over that many devices (mesh axis
    "batch"), for steps too big for one device; BN statistics and losses are still
    over the full batch (XLA adds the cross-device sums)."""

    def __init__(self, num_streams: int, batch_split: int = 1):
        devices = np.array(jax.devices()).reshape(-1, batch_split)
        assert num_streams % len(devices) == 0, f"{num_streams} streams for {len(devices)} device groups"
        self.num_streams = num_streams
        self.mesh = Mesh(devices, ("streams", "batch"))
        self.shard = NamedSharding(self.mesh, P("streams"))
        self.repl = NamedSharding(self.mesh, P())
        # Each local device holds a contiguous range of streams.
        ranges = {d: range(*sl[0].indices(num_streams))
                  for d, sl in self.shard.addressable_devices_indices_map((num_streams,)).items()}
        self.local_ids = sorted({i for r in ranges.values() for i in r})
        self._pos = {s: k for k, s in enumerate(self.local_ids)}

    def put(self, local: np.ndarray, batch_axis: int | None = None):
        """Stream-sharded global array from this host's (local_streams, ...) slice; with
        batch_axis, that axis is also split over the mesh's "batch" axis."""
        shape = (self.num_streams,) + local.shape[1:]
        sharding = self.shard if batch_axis is None else NamedSharding(
            self.mesh, P("streams", *[None] * (batch_axis - 1), "batch"))
        arrays = []
        for d, idx in sharding.addressable_devices_indices_map(shape).items():
            r = range(*idx[0].indices(self.num_streams))
            arrays.append(jax.device_put(local[(slice(self._pos[r[0]], self._pos[r[-1]] + 1),) + tuple(idx[1:])], d))
        return jax.make_array_from_single_device_arrays(shape, sharding, arrays)

    def replicate(self, tree):
        return jax.tree.map(
            lambda a: jax.make_array_from_callback(np.shape(a), self.repl, lambda i: np.asarray(a)[i]), tree)

    def gather(self, x) -> np.ndarray:
        return np.asarray(multihost_utils.process_allgather(x, tiled=True))


def common_steps(streams) -> int:
    n = min(s.num_steps for s in streams)
    if jax.process_count() > 1:
        n = int(np.min(multihost_utils.process_allgather(np.array([n]))))
    return n


def add_launch_args(p):
    p.add_argument("--data-root", default="gs://sk7524-tinker-tpu-us-central2/synthgrad/data/imagenet_v1")
    p.add_argument("--weights", default="gs://sk7524-tinker-tpu-us-central2/synthgrad/weights/resnet50_tv_in1k.safetensors")
    p.add_argument("--local-cache", default=os.path.expanduser("~/synthgrad/cache"))
    p.add_argument("--out", required=True, help="results dir (local or gs://)")
    p.add_argument("--precision", default="highest", choices=["default", "high", "highest"])
    p.add_argument("--decode-workers", type=int, default=min(64, os.cpu_count()))
    p.add_argument("--coordinator", default=None)
    p.add_argument("--num-processes", type=int, default=1)
    p.add_argument("--process-id", type=int, default=0)


def init_distributed(args):
    if args.num_processes > 1:
        jax.distributed.initialize(args.coordinator, args.num_processes, args.process_id)
    jax.config.update("jax_default_matmul_precision", args.precision)


def write_output(out: str, name: str, data: bytes):
    """Write a file to a local dir or a gs:// prefix."""
    if out.startswith("gs://"):
        local = Path("/tmp/synthgrad-out") / name
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        subprocess.run(["gcloud", "storage", "cp", str(local), f"{out}/{name}"], check=True)
    else:
        Path(out, name).parent.mkdir(parents=True, exist_ok=True)
        Path(out, name).write_bytes(data)


def read_output(out: str, name: str) -> bytes | None:
    """Read a file written by write_output, or None if it does not exist."""
    if out.startswith("gs://"):
        local = Path("/tmp/synthgrad-in") / name
        local.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["gcloud", "storage", "cp", f"{out}/{name}", str(local)], capture_output=True)
        return local.read_bytes() if r.returncode == 0 else None
    path = Path(out, name)
    return path.read_bytes() if path.exists() else None


class ConcatStream:
    """Several streams back to back, `steps_each` batches from each, no reset between them
    (continual adaptation). Segment i uses its own Stream's order."""

    def __init__(self, segments, steps_each: int):
        self.segments, self.steps_each = segments, steps_each
        assert all(s.num_steps >= steps_each for s in segments)
        self.num_steps = steps_each * len(segments)
        self.resize = segments[0].resize

    def batch_records(self, step):
        """Segments repeat in order if more steps are asked for than they hold."""
        return self.segments[(step // self.steps_each) % len(self.segments)].batch_records(step % self.steps_each)
