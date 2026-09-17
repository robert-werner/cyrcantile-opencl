"""OpenCL backend: manages context, compiles kernels, and dispatches
batch operations to the GPU.

A singleton ``OpenCLBackend`` is lazily created on first use.  If no
OpenCL platform / device is available the module-level ``HAS_OPENCL``
flag stays ``False`` and the public API falls back to the pure-Python
``_cpu`` implementations; a one-time ``RuntimeWarning`` explains why.

Device selection
----------------
By default the first GPU device across all platforms is used (falling
back to any available device).  Two environment variables override
this with a case-insensitive substring match:

* ``CYRCANTILE_CL_PLATFORM`` — OpenCL platform name, e.g. ``"NVIDIA"``
* ``CYRCANTILE_CL_DEVICE``   — device name, e.g. ``"RTX"``

Buffers
-------
Device buffers are pooled and reused across calls (keyed by size) to
avoid per-call allocate/free round-trips.  Buffers are returned to the
pool only after ``queue.finish()``, so no in-flight command can still
reference them.
"""

from __future__ import annotations

import os
import pathlib
import warnings

import numpy as np

try:
    import pyopencl as cl
    HAS_OPENCL = True
except Exception:
    HAS_OPENCL = False

_KERNEL_PATH = pathlib.Path(__file__).resolve().parent / "_kernels.cl"

# Pool limits (count / total bytes of cached device buffers).
_MAX_POOLED_BUFFERS = 64
_MAX_POOLED_BYTES = 256 * 1024 * 1024


def _pick_device():
    """Select an OpenCL device honouring CYRCANTILE_CL_PLATFORM/DEVICE.

    Prefers GPU devices; returns ``None`` when no device exists at all.
    """
    if not HAS_OPENCL:  # pragma: no cover - guarded by callers
        return None
    want_p = os.environ.get("CYRCANTILE_CL_PLATFORM", "").lower()
    want_d = os.environ.get("CYRCANTILE_CL_DEVICE", "").lower()
    platforms = cl.get_platforms()
    if not platforms:
        return None
    matches = [
        d
        for p in platforms
        if not want_p or want_p in p.name.lower()
        for d in p.get_devices()
        if not want_d or want_d in d.name.lower()
    ]
    if not matches:
        if want_p or want_d:
            warnings.warn(
                "CYRCANTILE_CL_PLATFORM/DEVICE matched no OpenCL device; "
                "falling back to default selection",
                RuntimeWarning,
                stacklevel=3,
            )
        matches = [d for p in platforms for d in p.get_devices()]
        if not matches:
            return None
    gpus = [d for d in matches if d.type & cl.device_type.GPU]
    return gpus[0] if gpus else matches[0]


class _BufferPool:
    """Reuse READ_WRITE device buffers across calls, keyed by byte size."""

    def __init__(self, ctx, max_buffers=_MAX_POOLED_BUFFERS,
                 max_bytes=_MAX_POOLED_BYTES):
        self._ctx = ctx
        self._max_buffers = max_buffers
        self._max_bytes = max_bytes
        self._free = {}      # nbytes -> [buf, ...]
        self._order = []     # FIFO of pooled buffers (oldest first)
        self._sizes = {}     # id(buf) -> nbytes
        self._total = 0

    def get(self, nbytes):
        bufs = self._free.get(nbytes)
        if bufs:
            buf = bufs.pop()
            self._order.remove(buf)
            nbytes_known = self._sizes.pop(id(buf))
            self._total -= nbytes_known
            return buf
        return cl.Buffer(self._ctx, cl.mem_flags.READ_WRITE, int(nbytes))

    def put(self, buf, nbytes):
        nbytes = int(nbytes)
        if nbytes <= 0 or nbytes > self._max_bytes:
            buf.release()
            return
        while self._order and (
            self._total + nbytes > self._max_bytes
            or len(self._order) + 1 > self._max_buffers
        ):
            self._drop_oldest()
        if len(self._order) + 1 > self._max_buffers:
            buf.release()
            return
        self._free.setdefault(nbytes, []).append(buf)
        self._order.append(buf)
        self._sizes[id(buf)] = nbytes
        self._total += nbytes

    def _drop_oldest(self):
        buf = self._order.pop(0)
        nbytes = self._sizes.pop(id(buf))
        self._free[nbytes].remove(buf)
        self._total -= nbytes
        buf.release()


class OpenCLBackend:
    """Thin wrapper around an OpenCL context + compiled program."""

    _instance: OpenCLBackend | None = None

    # -- lifecycle -------------------------------------------------

    def __new__(cls):
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._init()
            cls._instance = obj
        return cls._instance

    def _init(self):
        if not HAS_OPENCL:
            raise RuntimeError("pyopencl is not available")
        dev = _pick_device()
        if dev is not None:
            self.device = dev
            self.ctx = cl.Context([dev])
        else:
            self.ctx = cl.create_some_context(interactive=False)
            self.device = self.ctx.devices[0]
        self.queue = cl.CommandQueue(self.ctx)
        source = _KERNEL_PATH.read_text(encoding="utf-8")
        self.program = cl.Program(self.ctx, source).build()
        self.mf = cl.mem_flags
        self._pool = _BufferPool(self.ctx)
        self._kernels = {}

    def _kernel(self, kernel_name):
        """Cache cl.Kernel objects — re-fetching from the program is costly."""
        kernel = self._kernels.get(kernel_name)
        if kernel is None:
            kernel = cl.Kernel(self.program, kernel_name)
            self._kernels[kernel_name] = kernel
        return kernel

    # -- generic helpers ------------------------------------------

    def _run_write(self, kernel_name, n, inputs, outputs, *scalars):
        """Run *kernel_name*; copy back every array in *outputs*.

        Device buffers come from (and return to) the buffer pool.  The
        queue is finished before buffers are released back, guaranteeing
        no pending command references them.
        """
        if int(n) <= 0:
            return
        kernel = self._kernel(kernel_name)
        cl_args = []
        bufs = []          # (buf, nbytes)
        out_bufs = []      # (buf, host_array)
        for a in inputs:
            buf = self._pool.get(a.nbytes)
            cl.enqueue_copy(self.queue, buf, np.ascontiguousarray(a))
            cl_args.append(buf)
            bufs.append((buf, a.nbytes))
        for a in outputs:
            buf = self._pool.get(a.nbytes)
            cl_args.append(buf)
            out_bufs.append((buf, a))
            bufs.append((buf, a.nbytes))
        cl_args.extend(scalars)
        kernel(self.queue, (int(n),), None, *cl_args)
        for buf, a in out_bufs:
            cl.enqueue_copy(self.queue, a, buf)
        self.queue.finish()
        for buf, nbytes in bufs:
            self._pool.put(buf, nbytes)

    # -- batch operations ------------------------------------------

    def xy(self, lng, lat, truncate=False):
        lng = np.ascontiguousarray(lng, dtype=np.float64)
        lat = np.ascontiguousarray(lat, dtype=np.float64)
        n = lng.shape[0]
        ox = np.empty(n, dtype=np.float64)
        oy = np.empty(n, dtype=np.float64)
        self._run_write("xy_batch", n, [lng, lat], [ox, oy],
                        np.int32(1 if truncate else 0), np.uint64(n))
        return ox, oy

    def lnglat(self, x, y):
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        n = x.shape[0]
        olng = np.empty(n, dtype=np.float64)
        olat = np.empty(n, dtype=np.float64)
        self._run_write("lnglat_batch", n, [x, y], [olng, olat],
                        np.uint64(n))
        return olng, olat

    def tile(self, lng, lat, zoom, truncate=False):
        lng = np.ascontiguousarray(lng, dtype=np.float64)
        lat = np.ascontiguousarray(lat, dtype=np.float64)
        n = lng.shape[0]
        ox = np.empty(n, dtype=np.int32)
        oy = np.empty(n, dtype=np.int32)
        self._run_write("tile_batch", n, [lng, lat], [ox, oy],
                        np.int32(zoom), np.int32(1 if truncate else 0),
                        np.uint64(n))
        return ox, oy

    def ul(self, tx, ty, zoom):
        tx = np.ascontiguousarray(tx, dtype=np.int32)
        ty = np.ascontiguousarray(ty, dtype=np.int32)
        n = tx.shape[0]
        olng = np.empty(n, dtype=np.float64)
        olat = np.empty(n, dtype=np.float64)
        self._run_write("ul_batch", n, [tx, ty], [olng, olat],
                        np.int32(zoom), np.uint64(n))
        return olng, olat

    def bounds(self, tx, ty, zoom):
        tx = np.ascontiguousarray(tx, dtype=np.int32)
        ty = np.ascontiguousarray(ty, dtype=np.int32)
        n = tx.shape[0]
        ow = np.empty(n, dtype=np.float64)
        os_ = np.empty(n, dtype=np.float64)
        oe = np.empty(n, dtype=np.float64)
        on = np.empty(n, dtype=np.float64)
        self._run_write("bounds_batch", n, [tx, ty], [ow, os_, oe, on],
                        np.int32(zoom), np.uint64(n))
        return ow, os_, oe, on

    def xy_bounds(self, tx, ty, zoom):
        tx = np.ascontiguousarray(tx, dtype=np.int32)
        ty = np.ascontiguousarray(ty, dtype=np.int32)
        n = tx.shape[0]
        ol = np.empty(n, dtype=np.float64)
        ob = np.empty(n, dtype=np.float64)
        or_ = np.empty(n, dtype=np.float64)
        ot = np.empty(n, dtype=np.float64)
        self._run_write("xy_bounds_batch", n, [tx, ty], [ol, ob, or_, ot],
                        np.int32(zoom), np.uint64(n))
        return ol, ob, or_, ot

    def quadkey_encode(self, tx, ty, zoom):
        tx = np.ascontiguousarray(tx, dtype=np.int32)
        ty = np.ascontiguousarray(ty, dtype=np.int32)
        n = tx.shape[0]
        oqk = np.empty(n, dtype=np.uint64)
        self._run_write("quadkey_encode_batch", n, [tx, ty], [oqk],
                        np.int32(zoom), np.uint64(n))
        return oqk

    def quadkey_decode(self, qk, zoom):
        qk = np.ascontiguousarray(qk, dtype=np.uint64)
        n = qk.shape[0]
        ox = np.empty(n, dtype=np.int32)
        oy = np.empty(n, dtype=np.int32)
        self._run_write("quadkey_decode_batch", n, [qk], [ox, oy],
                        np.int32(zoom), np.uint64(n))
        return ox, oy

    def parent(self, tx, ty, shift):
        """Ancestor of each tile at ``shift`` zoom levels up."""
        tx = np.ascontiguousarray(tx, dtype=np.int32)
        ty = np.ascontiguousarray(ty, dtype=np.int32)
        n = tx.shape[0]
        ox = np.empty(n, dtype=np.int32)
        oy = np.empty(n, dtype=np.int32)
        self._run_write("parent_batch", n, [tx, ty], [ox, oy],
                        np.int32(shift), np.uint64(n))
        return ox, oy

    def children(self, tx, ty):
        tx = np.ascontiguousarray(tx, dtype=np.int32)
        ty = np.ascontiguousarray(ty, dtype=np.int32)
        n = tx.shape[0]
        ox = np.empty(n * 4, dtype=np.int32)
        oy = np.empty(n * 4, dtype=np.int32)
        self._run_write("children_batch", n, [tx, ty], [ox, oy],
                        np.uint64(n))
        return ox, oy

    def neighbors(self, tx, ty):
        tx = np.ascontiguousarray(tx, dtype=np.int32)
        ty = np.ascontiguousarray(ty, dtype=np.int32)
        n = tx.shape[0]
        ox = np.empty(n * 8, dtype=np.int32)
        oy = np.empty(n * 8, dtype=np.int32)
        self._run_write("neighbors_batch", n, [tx, ty], [ox, oy],
                        np.uint64(n))
        return ox, oy

    def bounding_tile(self, west, north, zoom):
        """Tile of the bbox north-west corner at *zoom* (only w/n are used)."""
        west = np.ascontiguousarray(west, dtype=np.float64)
        north = np.ascontiguousarray(north, dtype=np.float64)
        n = west.shape[0]
        ox = np.empty(n, dtype=np.int32)
        oy = np.empty(n, dtype=np.int32)
        self._run_write("bounding_tile_batch", n, [west, north], [ox, oy],
                        np.int32(zoom), np.uint64(n))
        return ox, oy

    def tiles_in_bbox(self, min_x, min_y, grid_w, grid_h):
        n = grid_w * grid_h
        ox = np.empty(n, dtype=np.int32)
        oy = np.empty(n, dtype=np.int32)
        self._run_write("tiles_in_bbox", n, [], [ox, oy],
                        np.int32(min_x), np.int32(min_y),
                        np.int32(grid_w), np.uint64(n))
        return ox, oy


_warned_backend_unavailable = False


def get_backend():
    """Return the singleton ``OpenCLBackend`` or ``None`` if unavailable."""
    global _warned_backend_unavailable
    if not HAS_OPENCL:
        return None
    try:
        return OpenCLBackend()
    except Exception as e:
        if not _warned_backend_unavailable:
            warnings.warn(
                f"OpenCL backend unavailable ({e}); batch operations "
                f"fall back to the CPU implementation.",
                RuntimeWarning,
                stacklevel=2,
            )
            _warned_backend_unavailable = True
        return None
