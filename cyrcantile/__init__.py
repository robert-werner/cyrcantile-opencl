"""cyrcantile — OpenCL-accelerated spherical-mercator tile utilities.

A drop-in replacement for the ``mercantile`` / ``cyrcantile`` API backed
by OpenCL kernels for batch GPU acceleration.  Single-value calls use
the pure-Python CPU path; array inputs are dispatched to OpenCL
automatically.

Example
-------

::

    import cyrcantile as ct

    # single value — CPU path
    t = ct.tile(-9.14, 53.12, 7)          # Tile(x=62, y=60, z=7)
    b = ct.bounds(t)                      # LngLatBbox(...)

    # batch — GPU path (requires pyopencl + OpenCL device)
    lons = [-9.14, -0.12, 13.38]
    lats = [ 53.12,  51.50, 52.52]
    xs, ys = ct.tile(lons, lats, 7)       # numpy arrays
"""

from __future__ import annotations

import math
import numpy as np

from ._types import Tile, LngLat, LngLatBbox, Bbox, XY, MinMax
from . import _cpu as cpu
from ._backend import get_backend, HAS_OPENCL
from ._cpu import HAS_NUMBA

__version__ = "0.2.0"
__all__ = [
    "Tile", "LngLat", "LngLatBbox", "Bbox", "XY", "MinMax",
    "xy", "lnglat", "tile", "ul", "bounds", "xy_bounds",
    "quadkey", "quadkey_to_tile", "parent", "children",
    "neighbors", "bounding_tile", "tiles", "feature",
    "radians", "degrees", "minmax", "truncate_lnglat",
    "HAS_OPENCL", "HAS_NUMBA",
]

# Re-export constants
R2D = cpu.R2D
D2R = cpu.D2R
RE = cpu.RE
CE = cpu.CE
EPSILON = cpu.EPSILON
MAX_LAT = cpu.MAX_LAT

_backend = get_backend()

# ------------------------------------------------------------------ #
#  Utility functions (always CPU — trivial scalar math)
# ------------------------------------------------------------------ #

radians = cpu.radians
degrees = cpu.degrees
minmax = cpu.minmax
truncate_lnglat = cpu.truncate_lnglat


def _is_array(v) -> bool:
    return isinstance(v, (list, tuple, np.ndarray)) and len(v) > 1


# ------------------------------------------------------------------ #
#  xy — geographic → Web Mercator
# ------------------------------------------------------------------ #

def xy(lng, lat, truncate=False):
    if _is_array(lng):
        if _backend:
            ox, oy = _backend.xy(lng, lat, truncate=truncate)
            return ox, oy
        lng = np.asarray(lng, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        return cpu.xy_vec(lng, lat, truncate)
    r = cpu.xy(lng, lat, truncate)
    return r.x, r.y


# ------------------------------------------------------------------ #
#  lnglat — Web Mercator → geographic
# ------------------------------------------------------------------ #

def lnglat(x, y):
    if _is_array(x):
        if _backend:
            olng, olat = _backend.lnglat(x, y)
            return olng, olat
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return cpu.lnglat_vec(x, y)
    r = cpu.lnglat(x, y)
    return r.lng, r.lat


# ------------------------------------------------------------------ #
#  tile — geographic → Tile
# ------------------------------------------------------------------ #

def tile(lng, lat, zoom, truncate=False):
    if _is_array(lng):
        if _backend:
            ox, oy = _backend.tile(lng, lat, zoom, truncate=truncate)
            return ox, oy
        lng = np.asarray(lng, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        return cpu.tile_vec(lng, lat, zoom, truncate)
    return cpu.tile(lng, lat, zoom, truncate)


# ------------------------------------------------------------------ #
#  ul — Tile → upper-left corner
# ------------------------------------------------------------------ #

def ul(tile_input, zoom=None):
    if isinstance(tile_input, Tile):
        return cpu.ul(tile_input)
    tx = np.asarray(tile_input, dtype=np.int32)
    if zoom is None:
        raise ValueError("zoom required for batch ul")
    if _backend:
        olng, olat = _backend.ul(tx, tx, zoom)
        return olng, olat
    return cpu.ul_vec(tx, zoom)


# ------------------------------------------------------------------ #
#  bounds — Tile → geographic bbox
# ------------------------------------------------------------------ #

def bounds(tile_input, zoom=None):
    if isinstance(tile_input, Tile):
        return cpu.bounds(tile_input)
    tx = np.asarray(tile_input, dtype=np.int32)
    if zoom is None:
        raise ValueError("zoom required for batch bounds")
    if _backend:
        ow, os_, oe, on = _backend.bounds(tx, tx, zoom)
        return ow, os_, oe, on
    return cpu.bounds_vec(tx, zoom)


# ------------------------------------------------------------------ #
#  xy_bounds — Tile → Web-Mercator bbox
# ------------------------------------------------------------------ #

def xy_bounds(tile_input, zoom=None):
    if isinstance(tile_input, Tile):
        return cpu.xy_bounds(tile_input)
    tx = np.asarray(tile_input, dtype=np.int32)
    if zoom is None:
        raise ValueError("zoom required for batch xy_bounds")
    if _backend:
        ol, ob, or_, ot = _backend.xy_bounds(tx, tx, zoom)
        return ol, ob, or_, ot
    return cpu.xy_bounds_vec(tx, zoom)


# ------------------------------------------------------------------ #
#  quadkey — Tile ↔ quadkey string
# ------------------------------------------------------------------ #

def quadkey(tile_input, zoom=None):
    if isinstance(tile_input, Tile):
        return cpu.quadkey(tile_input)
    tx = np.asarray(tile_input, dtype=np.int32)
    if zoom is None:
        raise ValueError("zoom required for batch quadkey")
    if _backend:
        qk_int = _backend.quadkey_encode(tx, tx, zoom)
        return [_int_to_quadkey(q, zoom) for q in qk_int]
    return [cpu.quadkey(Tile(int(x), int(x), zoom)) for x in tx]


def quadkey_to_tile(qk):
    if isinstance(qk, str):
        return cpu.quadkey_to_tile(qk)
    zoom = len(qk[0]) if qk else 0
    qk_int = np.array([_quadkey_to_int(k) for k in qk], dtype=np.uint64)
    if _backend:
        ox, oy = _backend.quadkey_decode(qk_int, zoom)
        return [Tile(int(x), int(y), zoom) for x, y in zip(ox, oy)]
    return [cpu.quadkey_to_tile(k) for k in qk]


def _int_to_quadkey(q: int, zoom: int) -> str:
    digits = []
    for _ in range(zoom):
        digits.append(str(q & 3))
        q >>= 2
    return "".join(reversed(digits))


def _quadkey_to_int(qk: str) -> int:
    result = 0
    for c in qk:
        result = (result << 2) | int(c)
    return result


# ------------------------------------------------------------------ #
#  parent / children / neighbors
# ------------------------------------------------------------------ #

def parent(tile_input, zoom=None):
    if isinstance(tile_input, Tile):
        return cpu.parent(tile_input)
    tx = np.asarray(tile_input, dtype=np.int32)
    if zoom is None:
        raise ValueError("zoom required for batch parent")
    if _backend:
        ox, oy = _backend.parent(tx, tx, zoom)
        return [Tile(int(x), int(y), zoom - 1) for x, y in zip(ox, oy)]
    return [cpu.parent(Tile(int(x), int(x), zoom)) for x in tx]


def children(tile_input, zoom=None):
    if isinstance(tile_input, Tile):
        return cpu.children(tile_input)
    tx = np.asarray(tile_input, dtype=np.int32)
    if zoom is None:
        raise ValueError("zoom required for batch children")
    if _backend:
        ox, oy = _backend.children(tx, tx)
        z1 = zoom + 1
        return [Tile(int(ox[i]), int(oy[i]), z1) for i in range(len(ox))]
    return [c for x in tx for c in cpu.children(Tile(int(x), int(x), zoom))]


def neighbors(tile_input, zoom=None):
    if isinstance(tile_input, Tile):
        return cpu.neighbors(tile_input)
    tx = np.asarray(tile_input, dtype=np.int32)
    if zoom is None:
        raise ValueError("zoom required for batch neighbors")
    if _backend:
        ox, oy = _backend.neighbors(tx, tx)
        return [Tile(int(ox[i]), int(oy[i]), zoom) for i in range(len(ox))]
    return [n for x in tx for n in cpu.neighbors(Tile(int(x), int(x), zoom))]


# ------------------------------------------------------------------ #
#  bounding_tile / tiles / feature
# ------------------------------------------------------------------ #

def bounding_tile(west, south, east, north, zoom=None):
    if _is_array(west):
        if zoom is None:
            raise ValueError("zoom required for batch bounding_tile")
        if _backend:
            ox, oy = _backend.bounding_tile(west, south, east, north, zoom)
            return [Tile(int(x), int(y), zoom) for x, y in zip(ox, oy)]
        return [cpu.bounding_tile(w, s, e, n, zoom)
                for w, s, e, n in zip(west, south, east, north)]
    return cpu.bounding_tile(west, south, east, north, zoom)


def tiles(west, south, east, north, zooms):
    if isinstance(zooms, int):
        zooms = [zooms]
    w = max(west, -180.0)
    e = min(east, 180.0)
    s = max(south, -MAX_LAT)
    n = min(north, MAX_LAT)
    result = []
    for z in zooms:
        tmin = cpu.tile(w, n, z)
        tmax = cpu.tile(e, s, z)
        gx = tmax.x - tmin.x + 1
        gy = tmax.y - tmin.y + 1
        if _backend and gx * gy > 256:
            ox, oy = _backend.tiles_in_bbox(tmin.x, tmin.y, gx, gy)
            result.extend(Tile(int(ox[i]), int(oy[i]), z) for i in range(len(ox)))
        else:
            for x in range(tmin.x, tmax.x + 1):
                for y in range(tmin.y, tmax.y + 1):
                    result.append(Tile(x, y, z))
    return result


def feature(tile: Tile, fid=None, props=None):
    return cpu.feature(tile, fid, props)
