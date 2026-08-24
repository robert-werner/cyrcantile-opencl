"""OpenCL backend: manages context, compiles kernels, and dispatches
batch operations to the GPU.

A singleton ``OpenCLBackend`` is lazily created on first use.  If no
OpenCL platform / device is available the module-level ``HAS_OPENCL``
flag stays ``False`` and the public API falls back to the pure-Python
``_cpu`` implementations.
"""

from __future__ import annotations

import pathlib
import numpy as np

try:
    import pyopencl as cl
    HAS_OPENCL = True
except Exception:
    HAS_OPENCL = False

_KERNEL_PATH = pathlib.Path(__file__).resolve().parent / "_kernels.cl"


class OpenCLBackend:
    """Thin wrapper around an OpenCL context + compiled program."""

    _instance: "OpenCLBackend | None" = None

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
        self.ctx = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.ctx)
        source = _KERNEL_PATH.read_text()
        self.program = cl.Program(self.ctx, source).build()
        self.mf = cl.mem_flags

    # -- generic helpers ------------------------------------------

    def _run_write(self, kernel_name, n, inputs, outputs, *scalars):
        """Run *kernel_name*; copy back every array in *outputs*."""
        kernel = getattr(self.program, kernel_name)
        cl_args = []
        bufs = []
        for a in inputs:
            buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                            a.nbytes, a)
            cl_args.append(buf)
            bufs.append(buf)
        out_bufs = []
        for a in outputs:
            buf = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, a.nbytes)
            cl_args.append(buf)
            out_bufs.append(buf)
            bufs.append(buf)
        cl_args.extend(scalars)
        kernel(self.queue, (int(n),), None, *cl_args)
        for a, buf in zip(outputs, out_bufs):
            cl.enqueue_copy(self.queue, a, buf)
        for buf in bufs:
            buf.release()
        self.queue.finish()

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

    def parent(self, tx, ty, zoom):
        tx = np.ascontiguousarray(tx, dtype=np.int32)
        ty = np.ascontiguousarray(ty, dtype=np.int32)
        n = tx.shape[0]
        ox = np.empty(n, dtype=np.int32)
        oy = np.empty(n, dtype=np.int32)
        self._run_write("parent_batch", n, [tx, ty], [ox, oy],
                        np.int32(zoom), np.uint64(n))
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

    def bounding_tile(self, west, south, east, north, zoom):
        west = np.ascontiguousarray(west, dtype=np.float64)
        south = np.ascontiguousarray(south, dtype=np.float64)
        east = np.ascontiguousarray(east, dtype=np.float64)
        north = np.ascontiguousarray(north, dtype=np.float64)
        n = west.shape[0]
        ox = np.empty(n, dtype=np.int32)
        oy = np.empty(n, dtype=np.int32)
        self._run_write("bounding_tile_batch", n,
                        [west, south, east, north], [ox, oy],
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


def get_backend():
    """Return the singleton ``OpenCLBackend`` or ``None`` if unavailable."""
    if not HAS_OPENCL:
        return None
    try:
        return OpenCLBackend()
    except Exception:
        return None
