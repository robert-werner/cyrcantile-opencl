"""Pure-Python / NumPy CPU implementations.

Used for single-tile calls and as fallback when no GPU backend is
available.  The scalar math mirrors ``mercantile`` exactly; the
vectorised batch functions mirror the GPU kernels in ``_kernels.cl``,
``_vulkan.py`` and ``_cuda.py``.
"""

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ._types import XY, Bbox, LngLat, LngLatBbox, Tile

R2D = 180.0 / math.pi
D2R = math.pi / 180.0
RE = 6378137.0
CE = 2.0 * math.pi * RE
EPSILON = 1e-14
MAX_LAT = 85.05112878
MAX_LNG = 180.0
# mercantile's longitude/latitude epsilon used by bounding_tile.
LL_EPSILON = 1e-11


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
    x = RE * lng * D2R
    if lat <= -90.0:
        y = float("-inf")
    elif lat >= 90.0:
        y = float("inf")
    else:
        y = RE * math.log(math.tan(math.pi / 4 + lat * D2R / 2))
    return XY(x, y)


def lnglat(x: float, y: float) -> LngLat:
    return LngLat((x / RE) * R2D,
                  (2 * math.atan(math.exp(y / RE)) - math.pi / 2) * R2D)


def tile(lng: float, lat: float, zoom: int, truncate: bool = False) -> Tile:
    """Get the tile containing a longitude and latitude.

    Matches mercantile 1.2.x: tile indices are clamped into the valid
    ``[0, 2**zoom - 1]`` range and an ``EPSILON`` nudge pulls points on
    the right edge of a tile into the next tile over.  Raises
    ``ValueError`` for ``|lat| == 90`` where y is undefined.
    """
    if truncate:
        lng, lat = _clamp_lng(lng), _clamp_lat(lat)
    x = lng / 360.0 + 0.5
    sinlat = math.sin(lat * D2R)
    try:
        y = 0.5 - 0.25 * math.log((1.0 + sinlat) / (1.0 - sinlat)) / math.pi
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"Y can not be computed: lat={lat!r}") from None
    z2 = 1 << zoom
    if x <= 0.0:
        xt = 0
    elif x >= 1.0:
        xt = z2 - 1
    else:
        xt = int(math.floor((x + EPSILON) * z2))
    if y <= 0.0:
        yt = 0
    elif y >= 1.0:
        yt = z2 - 1
    else:
        yt = int(math.floor((y + EPSILON) * z2))
    return Tile(xt, yt, zoom)


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
        if c not in "0123":
            raise ValueError(f"invalid quadkey digit {c!r} (must be 0, 1, 2 or 3)")
        mask = 1 << (zoom - i - 1)
        d = int(c)
        if d & 1:
            x |= mask
        if d & 2:
            y |= mask
    return Tile(x, y, zoom)


def parent(tile: Tile, zoom: int | None = None) -> Tile | None:
    """Return the ancestor of ``tile`` at zoom ``zoom`` (default ``tile.z - 1``).

    Matches mercantile: returns ``None`` for a z=0 tile and raises
    ``ValueError`` when ``zoom`` is not an integer below ``tile.z``.
    """
    x, y, z = tile
    if z == 0:
        return None
    if zoom is not None and (z <= zoom or zoom != int(zoom)):
        raise ValueError("zoom must be an integer and less than that of the input tile")
    target = z - 1 if zoom is None else int(zoom)
    shift = z - target
    return Tile(x >> shift, y >> shift, target)


def children(tile: Tile, zoom: int | None = None):
    """Get the children of a tile.

    The immediate children are ordered top-left, top-right,
    bottom-right, bottom-left (mercantile order).  With *zoom* given,
    all descendants at that zoom are returned in depth-first clockwise
    winding order.
    """
    x, y, z = tile
    if zoom is not None and (z >= zoom or zoom != int(zoom)):
        raise ValueError("zoom must be an integer greater than that of the input tile")
    target = zoom if zoom is not None else z + 1
    tiles = [tile]
    while tiles[0][2] < target:
        xt, yt, zt = tiles.pop(0)
        tiles += [
            Tile(xt * 2, yt * 2, zt + 1),
            Tile(xt * 2 + 1, yt * 2, zt + 1),
            Tile(xt * 2 + 1, yt * 2 + 1, zt + 1),
            Tile(xt * 2, yt * 2 + 1, zt + 1),
        ]
    return tiles


def neighbors(tile: Tile):
    """Up to eight neighbours of a tile.

    Neighbours outside the zoom's valid grid (e.g. ``Tile(-1, -1, z)``)
    are omitted, matching mercantile.  Order is x-major: (x-1, y-1),
    (x-1, y), (x-1, y+1), (x, y-1), (x, y+1), (x+1, y-1), (x+1, y),
    (x+1, y+1).
    """
    x, y, z = tile
    hi = (1 << z) - 1
    out = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            if i == 0 and j == 0:
                continue
            nx, ny = x + i, y + j
            if 0 <= nx <= hi and 0 <= ny <= hi:
                out.append(Tile(nx, ny, z))
    return out


def _rshift(val: int, n: int) -> int:
    return (val % 0x100000000) >> n


def _bbox_zoom(x0: int, y0: int, x1: int, y1: int) -> int:
    # Algorithm from mercantile.
    MAX_ZOOM = 28
    for z in range(MAX_ZOOM):
        mask = 1 << (32 - (z + 1))
        if (x0 & mask) != (x1 & mask) or (y0 & mask) != (y1 & mask):
            return z
    return MAX_ZOOM


def bounding_tile(west, south, east, north, zoom=None, truncate=False) -> Tile:
    """Get the smallest tile containing a geographic bounding box.

    With ``zoom`` given, the tile of the north-west corner at that zoom
    is returned (mercantile semantics).  Without ``zoom`` the deepest
    single tile covering the whole bbox is searched, following
    mercantile's algorithm (bbox cells at z=32 plus a zoom scan).
    """
    if zoom is not None:
        return tile(west, north, zoom, truncate=truncate)
    if truncate:
        west, south = truncate_lnglat(west, south)
        east, north = truncate_lnglat(east, north)
    e = east - LL_EPSILON
    s = south + LL_EPSILON
    try:
        tmin = tile(west, north, 32)
        tmax = tile(e, s, 32)
    except ValueError:
        # |lat| beyond the mercator domain (mercantile returns the world tile).
        return Tile(0, 0, 0)
    z = _bbox_zoom(tmin.x, tmin.y, tmax.x, tmax.y)
    if z == 0:
        return Tile(0, 0, 0)
    x = _rshift(tmin.x, 32 - z)
    y = _rshift(tmin.y, 32 - z)
    return Tile(x, y, z)


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
#  Packed-quadkey <-> string conversion (vectorised, no Python loops)
# ------------------------------------------------------------------ #

def quadkey_ints_to_strings(qk, zoom: int) -> list[str]:
    """Convert packed uint64 quadkeys (2 bits per digit) into a list of
    quadkey digit strings of length ``zoom``."""
    qk = np.asarray(qk, dtype=np.uint64)
    n = qk.shape[0]
    if n == 0:
        return []
    if zoom <= 0:
        return [""] * n
    if zoom > 32:
        raise ValueError("quadkey zoom must be <= 32")
    # digits[row] = digit for bit j = zoom-1-row, i.e. string order.
    digits = np.empty((zoom, n), dtype=np.uint32)
    for row in range(zoom):
        j = zoom - 1 - row
        digits[row] = ((qk >> np.uint64(2 * j)) & np.uint64(3)).astype(np.uint32)
    chars = np.ascontiguousarray((digits + np.uint32(48)).T)  # (n, zoom) '0'+digit
    return chars.view(f"U{zoom}").ravel().tolist()


def quadkey_strings_to_ints(keys) -> np.ndarray:
    """Convert a sequence of equal-length quadkey digit strings into a
    packed uint64 array.  Validates digit range and uniform length."""
    keys = list(keys)
    n = len(keys)
    if n == 0:
        return np.empty(0, dtype=np.uint64)
    if not all(isinstance(k, str) for k in keys):
        raise TypeError("quadkeys must be strings")
    zoom = len(keys[0])
    if any(len(k) != zoom for k in keys):
        raise ValueError("all quadkeys must have the same length (zoom)")
    if zoom > 32:
        raise ValueError("quadkey zoom must be <= 32")
    if zoom == 0:
        return np.zeros(n, dtype=np.uint64)
    a = np.ascontiguousarray(np.array(keys, dtype=f"U{zoom}"))
    u = a.view(np.uint32).reshape(n, zoom)
    digits = (u - np.uint32(48)).astype(np.uint64)
    if (digits > 3).any():
        raise ValueError("quadkey digits must be 0, 1, 2 or 3")
    shifts = (2 * (zoom - 1 - np.arange(zoom))).astype(np.uint64)
    return np.bitwise_or.reduce(digits << shifts, axis=1)


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

_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def _get_executor() -> ThreadPoolExecutor:
    """Lazily create one shared thread pool for all chunked operations."""
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(
                    max_workers=_CPU_WORKERS, thread_name_prefix="cyrcantile")
    return _executor


def _chunks(n, workers):
    size = (n + workers - 1) // workers
    return [(i * size, min((i + 1) * size, n)) for i in range((n + size - 1) // size)]


def _maybe_parallel(arrays, fn, n):
    """Run ``fn(start, stop)`` for each slice over the shared thread pool.

    ``fn`` receives ``(arrays_subset, start, stop)`` and returns one result
    per call; the list of per-chunk results is returned.
    """
    if n < _THREAD_THRESHOLD or _CPU_WORKERS <= 1:
        return [fn(arrays, 0, n)]
    nchunks = min(_CPU_WORKERS, (n + 63) // 64)
    chunks = [c for c in _chunks(n, nchunks) if c[1] > c[0]]
    if len(chunks) <= 1:
        return [fn(arrays, 0, n)]
    ex = _get_executor()
    return list(ex.map(lambda c: fn(arrays, c[0], c[1]), chunks))


@np.errstate(over="ignore", invalid="ignore", divide="ignore")
def _tile_chunk(lng, lat, zoom, truncate, start, stop):
    """Mercantile-compatible: clamp + EPSILON nudge (see cpu.tile)."""
    lo = lng[start:stop]
    p = lat[start:stop]
    if truncate:
        lo = np.clip(lo, -MAX_LNG, MAX_LNG)
        p = np.clip(p, -MAX_LAT, MAX_LAT)
    x = lo / 360.0 + 0.5
    sinlat = np.sin(p * D2R)
    y = 0.5 - 0.25 * np.log((1.0 + sinlat) / (1.0 - sinlat)) / np.pi
    z2 = float(1 << zoom)
    xt = np.where(x <= 0.0, 0.0,
                  np.where(x >= 1.0, z2 - 1.0, np.floor((x + EPSILON) * z2)))
    yt = np.where(y <= 0.0, 0.0,
                  np.where(y >= 1.0, z2 - 1.0, np.floor((y + EPSILON) * z2)))
    return xt.astype(np.int32), yt.astype(np.int32)


@np.errstate(over="ignore", invalid="ignore", divide="ignore")
def _xy_chunk(lng, lat, truncate, start, stop):
    lo = lng[start:stop]
    p = lat[start:stop]
    if truncate:
        lo = np.clip(lo, -MAX_LNG, MAX_LNG)
        p = np.clip(p, -MAX_LAT, MAX_LAT)
    lat_r = p * D2R
    x = RE * lo * D2R
    y = RE * np.log(np.tan(np.pi / 4 + lat_r / 2))
    y = np.where(p <= -90.0, -np.inf, np.where(p >= 90.0, np.inf, y))
    return x, y


@np.errstate(over="ignore", invalid="ignore", divide="ignore")
def _lnglat_chunk(x, y, start, stop):
    x = x[start:stop]
    y = y[start:stop]
    return (x / RE) * R2D, (2 * np.arctan(np.exp(y / RE)) - np.pi / 2) * R2D


@np.errstate(over="ignore", invalid="ignore", divide="ignore")
def _ul_chunk(tx, ty, zoom, start, stop):
    tx = tx[start:stop]
    ty = ty[start:stop]
    n = float(1 << zoom)
    lng = tx / n * 360.0 - 180.0
    lat = np.arctan(np.sinh(np.pi * (1 - 2 * ty / n))) * R2D
    return lng, lat


@np.errstate(over="ignore", invalid="ignore", divide="ignore")
def _bounds_chunk(tx, ty, zoom, start, stop):
    tx = tx[start:stop]
    ty = ty[start:stop]
    n = float(1 << zoom)
    w = tx / n * 360.0 - 180.0
    e = (tx + 1) / n * 360.0 - 180.0
    north = np.arctan(np.sinh(np.pi * (1 - 2 * ty / n))) * R2D
    south = np.arctan(np.sinh(np.pi * (1 - 2 * (ty + 1) / n))) * R2D
    return w, south, e, north


def _xy_bounds_chunk(tx, ty, zoom, start, stop):
    tx = tx[start:stop]
    ty = ty[start:stop]
    n = float(1 << zoom)
    t_size = CE / n
    left = tx * t_size - CE / 2
    right = left + t_size
    top = CE / 2 - ty * t_size
    bottom = top - t_size
    return left, bottom, right, top


def _normalise_vec(*arrays):
    """Coerce flat array-likes to contiguous ndarrays.  Returns (arrays, n)."""
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
        z2 = float(1 << zoom)
        pi = math.pi
        for i in prange(n):
            lo = lng[i]
            p = lat[i]
            if truncate:
                lo = min(max(lo, -MAX_LNG), MAX_LNG)
                p = min(max(p, -MAX_LAT), MAX_LAT)
            x = lo / 360.0 + 0.5
            sinlat = math.sin(p * D2R)
            y = 0.5 - 0.25 * math.log((1.0 + sinlat) / (1.0 - sinlat)) / pi
            if x <= 0.0:
                res_x[i] = 0
            elif x >= 1.0:
                res_x[i] = int(z2) - 1
            else:
                res_x[i] = int(math.floor((x + EPSILON) * z2))
            if y <= 0.0:
                res_y[i] = 0
            elif y >= 1.0:
                res_y[i] = int(z2) - 1
            else:
                res_y[i] = int(math.floor((y + EPSILON) * z2))
        return res_x, res_y

    @njit(parallel=True, cache=True, fastmath=True, nogil=True)
    def _xy_jit(lng, lat, truncate):
        n = lng.shape[0]
        res_x = np.empty(n)
        res_y = np.empty(n)
        pi4 = math.pi / 4
        for i in prange(n):
            lo = lng[i]
            p = lat[i]
            if truncate:
                lo = min(max(lo, -MAX_LNG), MAX_LNG)
                p = min(max(p, -MAX_LAT), MAX_LAT)
            lat_r = p * D2R
            res_x[i] = RE * lo * D2R
            if p <= -90.0:
                res_y[i] = -math.inf
            elif p >= 90.0:
                res_y[i] = math.inf
            else:
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
    def _ul_jit(tx, ty, zoom):
        n = tx.shape[0]
        res_x = np.empty(n)
        res_y = np.empty(n)
        nn = float(1 << zoom)
        for i in prange(n):
            res_x[i] = tx[i] / nn * 360.0 - 180.0
            res_y[i] = math.atan(math.sinh(math.pi * (1 - 2 * ty[i] / nn))) * R2D
        return res_x, res_y

    @njit(parallel=True, cache=True, fastmath=True, nogil=True)
    def _bounds_jit(tx, ty, zoom):
        n = tx.shape[0]
        w = np.empty(n)
        s = np.empty(n)
        e = np.empty(n)
        no = np.empty(n)
        nn = float(1 << zoom)
        for i in prange(n):
            t = tx[i]
            u = ty[i]
            w[i] = t / nn * 360.0 - 180.0
            e[i] = (t + 1) / nn * 360.0 - 180.0
            no[i] = math.atan(math.sinh(math.pi * (1 - 2 * u / nn))) * R2D
            s[i] = math.atan(math.sinh(math.pi * (1 - 2 * (u + 1) / nn))) * R2D
        return w, s, e, no

    @njit(parallel=True, cache=True, fastmath=True, nogil=True)
    def _xy_bounds_jit(tx, ty, zoom):
        n = tx.shape[0]
        left = np.empty(n)
        top = np.empty(n)
        nn = float(1 << zoom)
        t_size = CE / nn
        hce = CE * 0.5
        for i in prange(n):
            t = tx[i]
            u = ty[i]
            left[i] = t * t_size - hce
            top[i] = hce - u * t_size
        return left, top - t_size, left + t_size, top


# ------------------------------------------------------------------ #
#  Vectorised CPU fallbacks (numpy) — used when no GPU backend is available
# ------------------------------------------------------------------ #

def _choose_engine(n):
    """Return 'numba' | 'thread' | 'single' for an array of n elements."""
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
        parts = _maybe_parallel(
            (lng, lat), lambda a, s, e: _xy_chunk(a[0], a[1], truncate, s, e), n)
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    return _xy_chunk(lng, lat, truncate, 0, n)


def lnglat_vec(x, y):
    (x, y), n = _normalise_vec(x, y)
    engine = _choose_engine(n)
    if engine == "numba":
        return _lnglat_jit(x, y)
    if engine == "thread":
        parts = _maybe_parallel(
            (x, y), lambda a, s, e: _lnglat_chunk(a[0], a[1], s, e), n)
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    return _lnglat_chunk(x, y, 0, n)


def tile_vec(lng, lat, zoom, truncate=False):
    (lng, lat), n = _normalise_vec(lng, lat)
    engine = _choose_engine(n)
    if engine == "numba":
        return _tile_jit(lng, lat, zoom, truncate)
    if engine == "thread":
        parts = _maybe_parallel(
            (lng, lat), lambda a, s, e: _tile_chunk(a[0], a[1], zoom, truncate, s, e), n)
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    return _tile_chunk(lng, lat, zoom, truncate, 0, n)


def ul_vec(tx, ty, zoom):
    (tx, ty), n = _normalise_vec(tx, ty)
    engine = _choose_engine(n)
    if engine == "numba":
        return _ul_jit(tx, ty, zoom)
    if engine == "thread":
        parts = _maybe_parallel(
            (tx, ty), lambda a, s, e: _ul_chunk(a[0], a[1], zoom, s, e), n)
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    return _ul_chunk(tx, ty, zoom, 0, n)


def bounds_vec(tx, ty, zoom):
    (tx, ty), n = _normalise_vec(tx, ty)
    engine = _choose_engine(n)
    if engine == "numba":
        return _bounds_jit(tx, ty, zoom)
    if engine == "thread":
        parts = _maybe_parallel(
            (tx, ty), lambda a, s, e: _bounds_chunk(a[0], a[1], zoom, s, e), n)
        return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))
    return _bounds_chunk(tx, ty, zoom, 0, n)


def xy_bounds_vec(tx, ty, zoom):
    (tx, ty), n = _normalise_vec(tx, ty)
    engine = _choose_engine(n)
    if engine == "numba":
        return _xy_bounds_jit(tx, ty, zoom)
    if engine == "thread":
        parts = _maybe_parallel(
            (tx, ty), lambda a, s, e: _xy_bounds_chunk(a[0], a[1], zoom, s, e), n)
        return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))
    return _xy_bounds_chunk(tx, ty, zoom, 0, n)


# ------------------------------------------------------------------ #
#  Vectorised tile algebra (bit twiddling — cheap even in plain NumPy)
# ------------------------------------------------------------------ #

# Children order matches mercantile: top-left, top-right, bottom-right,
# bottom-left.  Neighbour order is x-major (see cpu.neighbors).
_CHILD_DX = np.array([0, 1, 1, 0], dtype=np.int32)
_CHILD_DY = np.array([0, 0, 1, 1], dtype=np.int32)
_NEIGH_DX = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
_NEIGH_DY = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)


def parent_vec(tx, ty, shift: int):
    """Ancestor at ``shift`` zoom levels up (shift >= 0; 0 = identity)."""
    tx = np.ascontiguousarray(tx, dtype=np.int32)
    ty = np.ascontiguousarray(ty, dtype=np.int32)
    if shift < 0:
        raise ValueError("shift must be >= 0")
    return tx >> np.int32(shift), ty >> np.int32(shift)


def children_vec(tx, ty):
    """4 children per input tile; flat arrays of length 4n."""
    tx = np.ascontiguousarray(tx, dtype=np.int32)
    ty = np.ascontiguousarray(ty, dtype=np.int32)
    ox = (np.repeat(tx * 2, 4).reshape(-1, 4) + _CHILD_DX).ravel()
    oy = (np.repeat(ty * 2, 4).reshape(-1, 4) + _CHILD_DY).ravel()
    return ox, oy


def neighbors_vec(tx, ty):
    """8 neighbours per input tile (NW,N,NE,W,E,SW,S,SE); flat arrays of 8n."""
    tx = np.ascontiguousarray(tx, dtype=np.int32)
    ty = np.ascontiguousarray(ty, dtype=np.int32)
    ox = (np.repeat(tx, 8).reshape(-1, 8) + _NEIGH_DX).ravel()
    oy = (np.repeat(ty, 8).reshape(-1, 8) + _NEIGH_DY).ravel()
    return ox, oy


def quadkey_encode_vec(tx, ty, zoom: int):
    """Pack tile x/y bits into uint64 quadkeys (2 bits per digit)."""
    x = np.ascontiguousarray(tx, dtype=np.uint64)
    y = np.ascontiguousarray(ty, dtype=np.uint64)
    q = np.zeros(x.shape[0], dtype=np.uint64)
    for j in range(zoom):
        dx = (x >> np.uint64(j)) & np.uint64(1)
        dy = (y >> np.uint64(j)) & np.uint64(1)
        q |= (dx | (dy << np.uint64(1))) << np.uint64(2 * j)
    return q


def quadkey_decode_vec(qk, zoom: int):
    """Unpack uint64 quadkeys into int32 (x, y) arrays."""
    q = np.ascontiguousarray(qk, dtype=np.uint64)
    x = np.zeros(q.shape[0], dtype=np.int32)
    y = np.zeros(q.shape[0], dtype=np.int32)
    for k in range(zoom):
        x |= ((q >> np.uint64(2 * k)) & np.uint64(1)).astype(np.int32) << np.int32(k)
        y |= ((q >> np.uint64(2 * k + 1)) & np.uint64(1)).astype(np.int32) << np.int32(k)
    return x, y
