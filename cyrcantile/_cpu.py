"""Pure-Python CPU implementations — used for single-tile calls and as fallback
when no OpenCL device is available.  The math mirrors mercantile exactly."""

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ._types import Tile, LngLat, LngLatBbox, Bbox, XY

R2D = 180.0 / math.pi
D2R = math.pi / 180.0
RE = 6378137.0
CE = 2.0 * math.pi * RE
EPSILON = 1e-14
MAX_LAT = 85.05112878
MAX_LNG = 180.0


def _clamp_lat(lat: float) -> float:
    return max(-MAX_LAT, min(MAX_LAT, lat))


def _clamp_lng(lng: float) -> float:
    return max(-MAX_LNG, min(MAX_LNG, lng))


def truncate_lnglat(lng: float, lat: float) -> LngLat:
    return LngLat(_clamp_lng(lng), _clamp_lat(lat))


def radians(deg: float) -> float:
    return deg * D2R


def degrees(rad: float) -> float:
    return rad * R2D


def minmax(*xs):
    return min(xs), max(xs)


def xy(lng: float, lat: float, truncate: bool = False) -> XY:
    if truncate:
        lng, lat = _clamp_lng(lng), _clamp_lat(lat)
    lat_r = lat * D2R
    return XY(RE * lng * D2R, RE * math.log(math.tan(math.pi / 4 + lat_r / 2)))


def lnglat(x: float, y: float) -> LngLat:
    return LngLat((x / RE) * R2D,
                  (2 * math.atan(math.exp(y / RE)) - math.pi / 2) * R2D)


def tile(lng: float, lat: float, zoom: int, truncate: bool = False) -> Tile:
    if truncate:
        lng, lat = _clamp_lng(lng), _clamp_lat(lat)
    lat_r = lat * D2R
    n = 1 << zoom
    x = int(math.floor((lng + 180.0) / 360.0 * n))
    y = int(math.floor((1.0 - math.log(math.tan(math.pi / 4 + lat_r / 2)) / math.pi) * 0.5 * n))
    return Tile(x, y, zoom)


def ul(tile: Tile) -> LngLat:
    n = 1 << tile.z
    lng = tile.x / n * 360.0 - 180.0
    lat = math.atan(math.sinh(math.pi * (1 - 2 * tile.y / n))) * R2D
    return LngLat(lng, lat)


def bounds(tile: Tile) -> LngLatBbox:
    n = 1 << tile.z
    w = tile.x / n * 360.0 - 180.0
    e = (tile.x + 1) / n * 360.0 - 180.0
    north = math.atan(math.sinh(math.pi * (1 - 2 * tile.y / n))) * R2D
    south = math.atan(math.sinh(math.pi * (1 - 2 * (tile.y + 1) / n))) * R2D
    return LngLatBbox(w, south, e, north)


def xy_bounds(tile: Tile) -> Bbox:
    n = 1 << tile.z
    t_size = CE / n
    left = tile.x * t_size - CE / 2
    right = left + t_size
    top = CE / 2 - tile.y * t_size
    bottom = top - t_size
    return Bbox(left, bottom, right, top)


def quadkey(tile: Tile) -> str:
    qk = []
    for i in range(tile.z, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if tile.x & mask:
            digit += 1
        if tile.y & mask:
            digit += 2
        qk.append(str(digit))
    return "".join(qk)


def quadkey_to_tile(qk: str) -> Tile:
    x, y = 0, 0
    zoom = len(qk)
    for i, c in enumerate(qk):
        mask = 1 << (zoom - i - 1)
        d = int(c)
        if d & 1:
            x |= mask
        if d & 2:
            y |= mask
    return Tile(x, y, zoom)


def parent(tile: Tile) -> Tile:
    if tile.z == 0:
        return tile
    return Tile(tile.x >> 1, tile.y >> 1, tile.z - 1)


def children(tile: Tile):
    x2, y2 = tile.x * 2, tile.y * 2
    z1 = tile.z + 1
    return [Tile(x2, y2, z1), Tile(x2 + 1, y2, z1),
            Tile(x2, y2 + 1, z1), Tile(x2 + 1, y2 + 1, z1)]


def neighbors(tile: Tile):
    x, y, z = tile.x, tile.y, tile.z
    return [Tile(x-1, y-1, z), Tile(x, y-1, z), Tile(x+1, y-1, z),
            Tile(x-1, y, z),                   Tile(x+1, y, z),
            Tile(x-1, y+1, z), Tile(x, y+1, z), Tile(x+1, y+1, z)]


def bounding_tile(west, south, east, north, zoom=None):
    if zoom is None:
        for z in range(32):
            t = tile(west, north, z)
            b = bounds(t)
            if b.west <= west and b.east >= east and b.north >= north and b.south <= south:
                return t
        return Tile(0, 0, 0)
    return tile(west, north, zoom)


def tiles(west, south, east, north, zooms):
    if isinstance(zooms, int):
        zooms = [zooms]
    ul_tile = tile(east, north, max(zooms))
    w = max(min(west, -180.0), -180.0)
    e = min(max(east, 180.0), 180.0)
    s = max(min(south, -MAX_LAT), -MAX_LAT)
    n_ = min(max(north, MAX_LAT), MAX_LAT)
    result = []
    for z in zooms:
        tmin = tile(w, n_, z)
        tmax = tile(e, s, z)
        for x in range(tmin.x, tmax.x + 1):
            for y in range(tmin.y, tmax.y + 1):
                result.append(Tile(x, y, z))
    return result


def feature(tile: Tile, fid=None, props=None):
    b = bounds(tile)
    w, s, e, n = b.west, b.south, b.east, b.north
    return {
        "type": "Feature",
        "id": fid,
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[w, n], [e, n], [e, s], [w, s], [w, n]]],
        },
        "properties": props or {},
    }


# ------------------------------------------------------------------ #
#  Parallel CPU execution
# ------------------------------------------------------------------ #
# Two engines are supported, chosen at runtime:
#   * Numba (@njit(parallel=True)) — fastest for these arithmetic-heavy
#     kernels, with real multicore scaling.
#   * A chunked ThreadPoolExecutor over NumPy ufuncs — dependency-free
#     fallback.  NumPy's C loops release the GIL, so threads still scale
#     across cores for large arrays.
# If neither applies (small inputs) we just run the plain vectorised
# single-thread path to avoid dispatch overhead.

try:
    from numba import njit, prange
    HAS_NUMBA = True
except Exception:  # pragma: no cover
    HAS_NUMBA = False

_CPU_WORKERS = min(32, (os.cpu_count() or 1))
# Arrays smaller than this stay on the single-thread path.
_THREAD_THRESHOLD = 200_000


def _chunks(n, workers):
    size = (n + workers - 1) // workers
    return [(i * size, min((i + 1) * size, n)) for i in range((n + size - 1) // size)]


def _maybe_parallel(arrays, fn, n):
    """Run ``fn(start, stop)`` for each slice over a shared thread pool.

    ``fn`` receives ``(arrays_subset, start, stop)`` and returns one result
    per call; the list of per-chunk results is returned.  For inputs whose
    needs exceed the bounded pool it degrades gracefully.
    """
    if n < _THREAD_THRESHOLD or _CPU_WORKERS <= 1:
        return [fn(arrays, 0, n)]
    nchunks = min(_CPU_WORKERS, (n + 63) // 64)
    chunks = [c for c in _chunks(n, nchunks) if c[1] > c[0]]
    if len(chunks) <= 1:
        return [fn(arrays, 0, n)]
    with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
        return list(ex.map(lambda c: fn(arrays, c[0], c[1]), chunks))


@np.errstate(over="ignore", invalid="ignore", divide="ignore")
def _tile_chunk(lng, lat, zoom, truncate, start, stop):
    l = lng[start:stop]
    p = lat[start:stop]
    if truncate:
        l = np.clip(l, -MAX_LNG, MAX_LNG)
        p = np.clip(p, -MAX_LAT, MAX_LAT)
    lat_r = p * D2R
    n = 1 << zoom
    x = np.floor((l + 180.0) / 360.0 * n).astype(np.int32)
    y = np.floor((1.0 - np.log(np.tan(np.pi / 4 + lat_r / 2)) / np.pi) * 0.5 * n).astype(np.int32)
    return x, y


@np.errstate(over="ignore", invalid="ignore", divide="ignore")
def _xy_chunk(lng, lat, truncate, start, stop):
    l = lng[start:stop]
    p = lat[start:stop]
    if truncate:
        l = np.clip(l, -MAX_LNG, MAX_LNG)
        p = np.clip(p, -MAX_LAT, MAX_LAT)
    lat_r = p * D2R
    return RE * l * D2R, RE * np.log(np.tan(np.pi / 4 + lat_r / 2))


@np.errstate(over="ignore", invalid="ignore", divide="ignore")
def _lnglat_chunk(x, y, start, stop):
    x = x[start:stop]
    y = y[start:stop]
    return (x / RE) * R2D, (2 * np.arctan(np.exp(y / RE)) - np.pi / 2) * R2D


@np.errstate(over="ignore", invalid="ignore", divide="ignore")
def _ul_chunk(tx, zoom, start, stop):
    tx = tx[start:stop]
    n = float(1 << zoom)
    lng = tx / n * 360.0 - 180.0
    lat = np.arctan(np.sinh(np.pi * (1 - 2 * tx / n))) * R2D
    return lng, lat


@np.errstate(over="ignore", invalid="ignore", divide="ignore")
def _bounds_chunk(tx, zoom, start, stop):
    tx = tx[start:stop]
    n = float(1 << zoom)
    w = tx / n * 360.0 - 180.0
    e = (tx + 1) / n * 360.0 - 180.0
    north = np.arctan(np.sinh(np.pi * (1 - 2 * tx / n))) * R2D
    south = np.arctan(np.sinh(np.pi * (1 - 2 * (tx + 1) / n))) * R2D
    return w, south, e, north


def _xy_bounds_chunk(tx, zoom, start, stop):
    tx = tx[start:stop]
    n = float(1 << zoom)
    t_size = CE / n
    left = tx * t_size - CE / 2
    right = left + t_size
    top = CE / 2 - tx * t_size
    bottom = top - t_size
    return left, bottom, right, top


def _normalise_vec(*arrays):
    """Coerce a list of flat array-likes to a common-size, contiguous, f64/i64
    ndarray.  Returns (arrays, n)."""
    outs = []
    n = None
    for a in arrays:
        a = np.ascontiguousarray(a)
        outs.append(a)
        if n is None:
            n = a.shape[0]
    return outs, n


# Numba JIT kernels (parallel = multicore).  Lazily compiled on first use.
if HAS_NUMBA:
    @njit(parallel=True, cache=True, fastmath=True, nogil=True)
    def _tile_jit(lng, lat, zoom, truncate):
        n = lng.shape[0]
        res_x = np.empty(n, dtype=np.int32)
        res_y = np.empty(n, dtype=np.int32)
        nn = 1 << zoom
        pi4 = math.pi / 4
        for i in prange(n):
            l = lng[i]
            p = lat[i]
            if truncate:
                l = min(max(l, -MAX_LNG), MAX_LNG)
                p = min(max(p, -MAX_LAT), MAX_LAT)
            lat_r = p * D2R
            res_x[i] = int(math.floor((l + 180.0) / 360.0 * nn))
            res_y[i] = int(math.floor((1.0 - math.log(math.tan(pi4 + lat_r / 2)) / math.pi) * 0.5 * nn))
        return res_x, res_y

    @njit(parallel=True, cache=True, fastmath=True, nogil=True)
    def _xy_jit(lng, lat, truncate):
        n = lng.shape[0]
        res_x = np.empty(n)
        res_y = np.empty(n)
        pi4 = math.pi / 4
        for i in prange(n):
            l = lng[i]
            p = lat[i]
            if truncate:
                l = min(max(l, -MAX_LNG), MAX_LNG)
                p = min(max(p, -MAX_LAT), MAX_LAT)
            lat_r = p * D2R
            res_x[i] = RE * l * D2R
            res_y[i] = RE * math.log(math.tan(pi4 + lat_r / 2))
        return res_x, res_y

    @njit(parallel=True, cache=True, fastmath=True, nogil=True)
    def _lnglat_jit(x, y):
        n = x.shape[0]
        res_x = np.empty(n)
        res_y = np.empty(n)
        for i in prange(n):
            res_x[i] = (x[i] / RE) * R2D
            res_y[i] = (2 * math.atan(math.exp(y[i] / RE)) - math.pi / 2) * R2D
        return res_x, res_y

    @njit(parallel=True, cache=True, fastmath=True, nogil=True)
    def _ul_jit(tx, zoom):
        n = tx.shape[0]
        res_x = np.empty(n)
        res_y = np.empty(n)
        nn = float(1 << zoom)
        for i in prange(n):
            res_x[i] = tx[i] / nn * 360.0 - 180.0
            res_y[i] = math.atan(math.sinh(math.pi * (1 - 2 * tx[i] / nn))) * R2D
        return res_x, res_y

    @njit(parallel=True, cache=True, fastmath=True, nogil=True)
    def _bounds_jit(tx, zoom):
        n = tx.shape[0]
        w = np.empty(n)
        s = np.empty(n)
        e = np.empty(n)
        no = np.empty(n)
        nn = float(1 << zoom)
        for i in prange(n):
            t = tx[i]
            w[i] = t / nn * 360.0 - 180.0
            e[i] = (t + 1) / nn * 360.0 - 180.0
            no[i] = math.atan(math.sinh(math.pi * (1 - 2 * t / nn))) * R2D
            s[i] = math.atan(math.sinh(math.pi * (1 - 2 * (t + 1) / nn))) * R2D
        return w, s, e, no

    @njit(parallel=True, cache=True, fastmath=True, nogil=True)
    def _xy_bounds_jit(tx, zoom):
        n = tx.shape[0]
        left = np.empty(n)
        top = np.empty(n)
        nn = float(1 << zoom)
        t_size = CE / nn
        hce = CE * 0.5
        for i in prange(n):
            t = tx[i]
            left[i] = t * t_size - hce
            top[i] = hce - t * t_size
        return left, top - t_size, left + t_size, top


# ------------------------------------------------------------------ #
#  Vectorised CPU fallbacks (numpy) — used when OpenCL is unavailable
# ------------------------------------------------------------------ #

def _choose_engine(n):
    """Return ('numba', f) | ('thread', f) | ('single', f) for an array of n."""
    if HAS_NUMBA and n >= _THREAD_THRESHOLD:
        return "numba"
    if _CPU_WORKERS > 1 and n >= _THREAD_THRESHOLD:
        return "thread"
    return "single"


def xy_vec(lng, lat, truncate=False):
    (lng, lat), n = _normalise_vec(lng, lat)
    engine = _choose_engine(n)
    if engine == "numba":
        return _xy_jit(lng, lat, truncate)
    if engine == "thread":
        parts = _maybe_parallel((lng, lat), lambda a, s, e: _xy_chunk(a[0], a[1], truncate, s, e), n)
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    return _xy_chunk(lng, lat, truncate, 0, n)


def lnglat_vec(x, y):
    (x, y), n = _normalise_vec(x, y)
    engine = _choose_engine(n)
    if engine == "numba":
        return _lnglat_jit(x, y)
    if engine == "thread":
        parts = _maybe_parallel((x, y), lambda a, s, e: _lnglat_chunk(a[0], a[1], s, e), n)
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    return _lnglat_chunk(x, y, 0, n)


def tile_vec(lng, lat, zoom, truncate=False):
    (lng, lat), n = _normalise_vec(lng, lat)
    engine = _choose_engine(n)
    if engine == "numba":
        return _tile_jit(lng, lat, zoom, truncate)
    if engine == "thread":
        parts = _maybe_parallel((lng, lat), lambda a, s, e: _tile_chunk(a[0], a[1], zoom, truncate, s, e), n)
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    return _tile_chunk(lng, lat, zoom, truncate, 0, n)


def ul_vec(tx, zoom):
    (tx,), n = _normalise_vec(tx)
    engine = _choose_engine(n)
    if engine == "numba":
        return _ul_jit(tx, zoom)
    if engine == "thread":
        parts = _maybe_parallel((tx,), lambda a, s, e: _ul_chunk(a[0], zoom, s, e), n)
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    return _ul_chunk(tx, zoom, 0, n)


def bounds_vec(tx, zoom):
    (tx,), n = _normalise_vec(tx)
    engine = _choose_engine(n)
    if engine == "numba":
        return _bounds_jit(tx, zoom)
    if engine == "thread":
        parts = _maybe_parallel((tx,), lambda a, s, e: _bounds_chunk(a[0], zoom, s, e), n)
        return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))
    return _bounds_chunk(tx, zoom, 0, n)


def xy_bounds_vec(tx, zoom):
    (tx,), n = _normalise_vec(tx)
    engine = _choose_engine(n)
    if engine == "numba":
        return _xy_bounds_jit(tx, zoom)
    if engine == "thread":
        parts = _maybe_parallel((tx,), lambda a, s, e: _xy_bounds_chunk(a[0], zoom, s, e), n)
        return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))
    return _xy_bounds_chunk(tx, zoom, 0, n)
