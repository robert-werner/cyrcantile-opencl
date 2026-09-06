"""CUDA backend: compiles the same spherical-mercator tile operations as
CUDA device kernels via ``numba.cuda``.

The kernels mirror the OpenCL ``_kernels.cl`` (and the Vulkan ``_vulkan.py``)
implementations exactly, so results match the ``_cpu`` reference.

``numba`` is an optional dependency; ``HAS_CUDA`` is only ``True`` when
numba is installed *and* the choice of library is CUDA.  A missing library,
missing CUDA-capable GPU or driver makes ``HAS_CUDA`` stay ``False`` and the
public API falls back to the OpenCL/Vulkan/CPU backends gracefully.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import cuda
    _HAS_DRIVER = bool(cuda.is_available())
except Exception:  # pragma: no cover - no numba / no driver
    cuda = None
    _HAS_DRIVER = False

HAS_CUDA = cuda is not None and _HAS_DRIVER

# Constants (must match _cpu / _kernels.cl / _vulkan exactly).
_PI = 3.14159265358979323846
_HALF_PI = 1.57079632679489661923
_QUARTER_PI = 0.78539816339744830962
_R2D = 57.29577951308232
_D2R = 0.017453292519943295
_RE = 6378137.0
_CE = 40075016.68557849
_MAX_LAT = 85.05112878
_MAX_LNG = 180.0

_THREADS = 256


# ------------------------------------------------------------------ #
#  Device kernels
# ------------------------------------------------------------------ #

@cuda.jit(device=True)
def _clamp_lat_dev(p):
    if p > _MAX_LAT:
        return _MAX_LAT
    if p < -_MAX_LAT:
        return -_MAX_LAT
    return p


@cuda.jit(device=True)
def _clamp_lng_dev(l):
    if l > _MAX_LNG:
        return _MAX_LNG
    if l < -_MAX_LNG:
        return -_MAX_LNG
    return l


@cuda.jit
def _xy_kernel(lng, lat, ox, oy, truncate, n):
    i = cuda.grid(1)
    if i >= n:
        return
    l = lng[i]
    p = lat[i]
    if truncate:
        l = _clamp_lng_dev(l)
        p = _clamp_lat_dev(p)
    p = p * _D2R
    ox[i] = _RE * l * _D2R
    oy[i] = _RE * math.log(math.tan(_QUARTER_PI + p * 0.5))


@cuda.jit
def _lnglat_kernel(x, y, olng, olat, n):
    i = cuda.grid(1)
    if i >= n:
        return
    olng[i] = (x[i] / _RE) * _R2D
    olat[i] = (2.0 * math.atan(math.exp(y[i] / _RE)) - _HALF_PI) * _R2D


@cuda.jit
def _tile_kernel(lng, lat, ox, oy, zoom, truncate, n):
    i = cuda.grid(1)
    if i >= n:
        return
    l = lng[i]
    p = lat[i]
    if truncate:
        l = _clamp_lng_dev(l)
        p = _clamp_lat_dev(p)
    p = p * _D2R
    z2 = 2.0 ** zoom
    ox[i] = int(math.floor((l + 180.0) / 360.0 * z2))
    oy[i] = int(math.floor((1.0 - math.log(math.tan(_QUARTER_PI + p * 0.5)) / _PI)
                           * 0.5 * z2))


@cuda.jit
def _ul_kernel(tx, ty, olng, olat, zoom, n):
    i = cuda.grid(1)
    if i >= n:
        return
    z2 = 2.0 ** zoom
    olng[i] = tx[i] / z2 * 360.0 - 180.0
    olat[i] = math.atan(math.sinh(_PI * (1.0 - 2.0 * ty[i] / z2))) * _R2D


@cuda.jit
def _bounds_kernel(tx, ty, ow, os_, oe, on, zoom, n):
    i = cuda.grid(1)
    if i >= n:
        return
    z2 = 2.0 ** zoom
    ow[i] = tx[i] / z2 * 360.0 - 180.0
    oe[i] = (tx[i] + 1) / z2 * 360.0 - 180.0
    on[i] = math.atan(math.sinh(_PI * (1.0 - 2.0 * ty[i] / z2))) * _R2D
    os_[i] = math.atan(math.sinh(_PI * (1.0 - 2.0 * (ty[i] + 1) / z2))) * _R2D


@cuda.jit
def _xy_bounds_kernel(tx, ty, ol, ob, or_, ot, zoom, n):
    i = cuda.grid(1)
    if i >= n:
        return
    z2 = 2.0 ** zoom
    t_size = _CE / z2
    ol[i] = tx[i] * t_size - _CE * 0.5
    or_[i] = ol[i] + t_size
    ot[i] = _CE * 0.5 - ty[i] * t_size
    ob[i] = ot[i] - t_size


@cuda.jit
def _quadkey_encode_kernel(tx, ty, oqk, zoom, n):
    i = cuda.grid(1)
    if i >= n:
        return
    x = tx[i]
    y = ty[i]
    qk = 0
    s = 2 * (zoom - 1)
    for j in range(zoom - 1, -1, -1):
        d = ((x >> j) & 1) | (((y >> j) & 1) << 1)
        qk |= d << s
        s -= 2
    oqk[i] = qk


@cuda.jit
def _quadkey_decode_kernel(qk, ox, oy, zoom, n):
    i = cuda.grid(1)
    if i >= n:
        return
    q = qk[i]
    x = 0
    y = 0
    for j in range(zoom):
        shift = 2 * (zoom - 1 - j)
        d = (q >> shift) & 3
        mask = 1 << (zoom - 1 - j)
        if d & 1:
            x |= mask
        if d & 2:
            y |= mask
    ox[i] = x
    oy[i] = y


@cuda.jit
def _parent_kernel(tx, ty, ox, oy, zoom, n):
    i = cuda.grid(1)
    if i >= n:
        return
    if zoom == 0:
        ox[i] = tx[i]
        oy[i] = ty[i]
    else:
        ox[i] = tx[i] >> 1
        oy[i] = ty[i] >> 1


@cuda.jit
def _children_kernel(tx, ty, ox, oy, n):
    i = cuda.grid(1)
    if i >= n:
        return
    x2 = tx[i] << 1
    y2 = ty[i] << 1
    ox[i * 4 + 0] = x2
    oy[i * 4 + 0] = y2
    ox[i * 4 + 1] = x2 + 1
    oy[i * 4 + 1] = y2
    ox[i * 4 + 2] = x2
    oy[i * 4 + 2] = y2 + 1
    ox[i * 4 + 3] = x2 + 1
    oy[i * 4 + 3] = y2 + 1


@cuda.jit
def _neighbors_kernel(tx, ty, ox, oy, n):
    i = cuda.grid(1)
    if i >= n:
        return
    x = tx[i]
    y = ty[i]
    ox[i * 8 + 0] = x - 1
    oy[i * 8 + 0] = y - 1
    ox[i * 8 + 1] = x
    oy[i * 8 + 1] = y - 1
    ox[i * 8 + 2] = x + 1
    oy[i * 8 + 2] = y - 1
    ox[i * 8 + 3] = x - 1
    oy[i * 8 + 3] = y
    ox[i * 8 + 4] = x + 1
    oy[i * 8 + 4] = y
    ox[i * 8 + 5] = x - 1
    oy[i * 8 + 5] = y + 1
    ox[i * 8 + 6] = x
    oy[i * 8 + 6] = y + 1
    ox[i * 8 + 7] = x + 1
    oy[i * 8 + 7] = y + 1


@cuda.jit
def _bounding_tile_kernel(west, south, east, north, ox, oy, zoom, n):
    i = cuda.grid(1)
    if i >= n:
        return
    z2 = 2.0 ** zoom
    latn = _clamp_lat_dev(north[i]) * _D2R
    ox[i] = int(math.floor((west[i] + 180.0) / 360.0 * z2))
    oy[i] = int(math.floor((1.0 - math.log(math.tan(_QUARTER_PI + latn * 0.5)) / _PI)
                           * 0.5 * z2))


@cuda.jit
def _tiles_in_bbox_kernel(ox, oy, min_x, min_y, grid_w, n):
    i = cuda.grid(1)
    if i >= n:
        return
    ox[i] = min_x + (i % grid_w)
    oy[i] = min_y + (i // grid_w)


# ------------------------------------------------------------------ #
#  Backend class
# ------------------------------------------------------------------ #

class CudaBackend:
    """Singleton wrapper dispatching batch operations to CUDA kernels."""

    _instance: "CudaBackend | None" = None

    def __new__(cls):
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._init()
            cls._instance = obj
        return cls._instance

    def _init(self):
        if not HAS_CUDA:
            raise RuntimeError("numba.cuda is not available")
        cuda.select_device(0)
        self.has_u64 = True  # uint64 device integers always supported

    # -- helpers --------------------------------------------------

    def _blocks(self, n):
        return (n + _THREADS - 1) // _THREADS

    # -- batch operations ------------------------------------------

    def xy(self, lng, lat, truncate=False):
        lng = np.ascontiguousarray(lng, dtype=np.float64)
        lat = np.ascontiguousarray(lat, dtype=np.float64)
        n = lng.shape[0]
        d_lng = cuda.to_device(lng)
        d_lat = cuda.to_device(lat)
        d_ox = cuda.device_array(n, dtype=np.float64)
        d_oy = cuda.device_array(n, dtype=np.float64)
        _xy_kernel[self._blocks(n), _THREADS](
            d_lng, d_lat, d_ox, d_oy, np.int32(1 if truncate else 0), np.uint64(n))
        return d_ox.copy_to_host(), d_oy.copy_to_host()

    def lnglat(self, x, y):
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        n = x.shape[0]
        d_x = cuda.to_device(x)
        d_y = cuda.to_device(y)
        d_olng = cuda.device_array(n, dtype=np.float64)
        d_olat = cuda.device_array(n, dtype=np.float64)
        _lnglat_kernel[self._blocks(n), _THREADS](
            d_x, d_y, d_olng, d_olat, np.uint64(n))
        return d_olng.copy_to_host(), d_olat.copy_to_host()

    def tile(self, lng, lat, zoom, truncate=False):
        lng = np.ascontiguousarray(lng, dtype=np.float64)
        lat = np.ascontiguousarray(lat, dtype=np.float64)
        n = lng.shape[0]
        d_lng = cuda.to_device(lng)
        d_lat = cuda.to_device(lat)
        d_ox = cuda.device_array(n, dtype=np.int32)
        d_oy = cuda.device_array(n, dtype=np.int32)
        _tile_kernel[self._blocks(n), _THREADS](
            d_lng, d_lat, d_ox, d_oy, np.int32(zoom),
            np.int32(1 if truncate else 0), np.uint64(n))
        return d_ox.copy_to_host(), d_oy.copy_to_host()

    def ul(self, tx, ty, zoom):
        tx = np.ascontiguousarray(tx, dtype=np.int32)
        ty = np.ascontiguousarray(ty, dtype=np.int32)
        n = tx.shape[0]
        d_tx = cuda.to_device(tx)
        d_ty = cuda.to_device(ty)
        d_olng = cuda.device_array(n, dtype=np.float64)
        d_olat = cuda.device_array(n, dtype=np.float64)
        _ul_kernel[self._blocks(n), _THREADS](
            d_tx, d_ty, d_olng, d_olat, np.int32(zoom), np.uint64(n))
        return d_olng.copy_to_host(), d_olat.copy_to_host()

    def bounds(self, tx, ty, zoom):
        tx = np.ascontiguousarray(tx, dtype=np.int32)
        ty = np.ascontiguousarray(ty, dtype=np.int32)
        n = tx.shape[0]
        d_tx = cuda.to_device(tx)
        d_ty = cuda.to_device(ty)
        d_ow = cuda.device_array(n, dtype=np.float64)
        d_os = cuda.device_array(n, dtype=np.float64)
        d_oe = cuda.device_array(n, dtype=np.float64)
        d_on = cuda.device_array(n, dtype=np.float64)
        _bounds_kernel[self._blocks(n), _THREADS](
            d_tx, d_ty, d_ow, d_os, d_oe, d_on, np.int32(zoom), np.uint64(n))
        return (d_ow.copy_to_host(), d_os.copy_to_host(),
                d_oe.copy_to_host(), d_on.copy_to_host())

    def xy_bounds(self, tx, ty, zoom):
        tx = np.ascontiguousarray(tx, dtype=np.int32)
        ty = np.ascontiguousarray(ty, dtype=np.int32)
        n = tx.shape[0]
        d_tx = cuda.to_device(tx)
        d_ty = cuda.to_device(ty)
        d_ol = cuda.device_array(n, dtype=np.float64)
        d_ob = cuda.device_array(n, dtype=np.float64)
        d_or = cuda.device_array(n, dtype=np.float64)
        d_ot = cuda.device_array(n, dtype=np.float64)
        _xy_bounds_kernel[self._blocks(n), _THREADS](
            d_tx, d_ty, d_ol, d_ob, d_or, d_ot, np.int32(zoom), np.uint64(n))
        return (d_ol.copy_to_host(), d_ob.copy_to_host(),
                d_or.copy_to_host(), d_ot.copy_to_host())

    def quadkey_encode(self, tx, ty, zoom):
        tx = np.ascontiguousarray(tx, dtype=np.int32)
        ty = np.ascontiguousarray(ty, dtype=np.int32)
        n = tx.shape[0]
        d_tx = cuda.to_device(tx)
        d_ty = cuda.to_device(ty)
        d_oqk = cuda.device_array(n, dtype=np.uint64)
        _quadkey_encode_kernel[self._blocks(n), _THREADS](
            d_tx, d_ty, d_oqk, np.int32(zoom), np.uint64(n))
        return d_oqk.copy_to_host()

    def quadkey_decode(self, qk, zoom):
        qk = np.ascontiguousarray(qk, dtype=np.uint64)
        n = qk.shape[0]
        d_qk = cuda.to_device(qk)
        d_ox = cuda.device_array(n, dtype=np.int32)
        d_oy = cuda.device_array(n, dtype=np.int32)
        _quadkey_decode_kernel[self._blocks(n), _THREADS](
            d_qk, d_ox, d_oy, np.int32(zoom), np.uint64(n))
        return d_ox.copy_to_host(), d_oy.copy_to_host()

    def parent(self, tx, ty, zoom):
        tx = np.ascontiguousarray(tx, dtype=np.int32)
        ty = np.ascontiguousarray(ty, dtype=np.int32)
        n = tx.shape[0]
        d_tx = cuda.to_device(tx)
        d_ty = cuda.to_device(ty)
        d_ox = cuda.device_array(n, dtype=np.int32)
        d_oy = cuda.device_array(n, dtype=np.int32)
        _parent_kernel[self._blocks(n), _THREADS](
            d_tx, d_ty, d_ox, d_oy, np.int32(zoom), np.uint64(n))
        return d_ox.copy_to_host(), d_oy.copy_to_host()

    def children(self, tx, ty):
        tx = np.ascontiguousarray(tx, dtype=np.int32)
        ty = np.ascontiguousarray(ty, dtype=np.int32)
        n = tx.shape[0]
        d_tx = cuda.to_device(tx)
        d_ty = cuda.to_device(ty)
        d_ox = cuda.device_array(n * 4, dtype=np.int32)
        d_oy = cuda.device_array(n * 4, dtype=np.int32)
        _children_kernel[self._blocks(n), _THREADS](
            d_tx, d_ty, d_ox, d_oy, np.uint64(n))
        return d_ox.copy_to_host(), d_oy.copy_to_host()

    def neighbors(self, tx, ty):
        tx = np.ascontiguousarray(tx, dtype=np.int32)
        ty = np.ascontiguousarray(ty, dtype=np.int32)
        n = tx.shape[0]
        d_tx = cuda.to_device(tx)
        d_ty = cuda.to_device(ty)
        d_ox = cuda.device_array(n * 8, dtype=np.int32)
        d_oy = cuda.device_array(n * 8, dtype=np.int32)
        _neighbors_kernel[self._blocks(n), _THREADS](
            d_tx, d_ty, d_ox, d_oy, np.uint64(n))
        return d_ox.copy_to_host(), d_oy.copy_to_host()

    def bounding_tile(self, west, south, east, north, zoom):
        west = np.ascontiguousarray(west, dtype=np.float64)
        south = np.ascontiguousarray(south, dtype=np.float64)
        east = np.ascontiguousarray(east, dtype=np.float64)
        north = np.ascontiguousarray(north, dtype=np.float64)
        n = west.shape[0]
        d_west = cuda.to_device(west)
        d_south = cuda.to_device(south)
        d_east = cuda.to_device(east)
        d_north = cuda.to_device(north)
        d_ox = cuda.device_array(n, dtype=np.int32)
        d_oy = cuda.device_array(n, dtype=np.int32)
        _bounding_tile_kernel[self._blocks(n), _THREADS](
            d_west, d_south, d_east, d_north, d_ox, d_oy,
            np.int32(zoom), np.uint64(n))
        return d_ox.copy_to_host(), d_oy.copy_to_host()

    def tiles_in_bbox(self, min_x, min_y, grid_w, grid_h):
        n = grid_w * grid_h
        d_ox = cuda.device_array(n, dtype=np.int32)
        d_oy = cuda.device_array(n, dtype=np.int32)
        _tiles_in_bbox_kernel[self._blocks(n), _THREADS](
            d_ox, d_oy, np.int32(min_x), np.int32(min_y),
            np.int32(grid_w), np.uint64(n))
        return d_ox.copy_to_host(), d_oy.copy_to_host()


def get_backend():
    """Return the singleton ``CudaBackend`` or ``None`` if unavailable."""
    if not HAS_CUDA:
        return None
    try:
        return CudaBackend()
    except Exception:
        return None
