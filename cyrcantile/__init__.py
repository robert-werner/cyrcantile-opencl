"""cyrcantile вЂ” GPU-accelerated spherical-mercator tile utilities.

A drop-in replacement for the ``mercantile`` / ``cyrcantile`` API backed
by OpenCL, Vulkan or CUDA kernels for batch GPU acceleration.  Single
value calls use the pure-Python CPU path; array inputs are dispatched
to the GPU automatically.

Batch (array) conventions
-------------------------

* Coordinate inputs may be lists, tuples or 1-D NumPy arrays; outputs
  are NumPy arrays.
* Tile-coordinate operations accept either two arrays plus a zoom
  (``ct.ul(xs, ys, 12)``) or a list of ``Tile`` objects
  (``ct.ul(tiles)`` вЂ” mixed zoom levels are supported).
* Outputs are plain arrays: ``tile``/``parent``/``quadkey_to_tile`` в†’
  ``(xs, ys)``, ``ul``/``lnglat``/``xy`` в†’ ``(a, b)``, ``bounds`` в†’
  ``(w, s, e, n)``, ``xy_bounds`` в†’ ``(l, b, r, t)``,
  ``children`` в†’ shape ``(n, 4)``, ``neighbors`` в†’ shape ``(n, 8)``.

Example
-------

::

    import cyrcantile as ct

    # single value вЂ” CPU path
    t = ct.tile(-9.14, 53.12, 7)          # Tile(x=60, y=41, z=7)
    b = ct.bounds(t)                      # LngLatBbox(...)

    # batch вЂ” GPU path (requires a GPU backend)
    lons = [-9.14, -0.12, 13.38]
    lats = [ 53.12,  51.50, 52.52]
    xs, ys = ct.tile(lons, lats, 7)       # numpy int32 arrays
"""

from __future__ import annotations

import operator

import numpy as np

from . import _cpu as cpu
from ._backend import HAS_OPENCL, get_backend
from ._cpu import HAS_NUMBA
from ._types import XY, Bbox, LngLat, LngLatBbox, MinMax, Tile

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("cyrcantile")
except Exception:  # pragma: no cover - running from source without install
    __version__ = "0.3.0"

__all__ = [
    "Tile", "LngLat", "LngLatBbox", "Bbox", "XY", "MinMax",
    "xy", "lnglat", "tile", "ul", "bounds", "xy_bounds",
    "quadkey", "quadkey_to_tile", "parent", "children",
    "neighbors", "bounding_tile", "tiles", "feature",
    "radians", "degrees", "minmax", "truncate_lnglat",
    "HAS_OPENCL", "HAS_NUMBA", "ACTIVE_BACKEND", "MAX_BATCH_ZOOM",
]

# Re-export constants
R2D = cpu.R2D
D2R = cpu.D2R
RE = cpu.RE
CE = cpu.CE
EPSILON = cpu.EPSILON
MAX_LAT = cpu.MAX_LAT
MAX_LNG = cpu.MAX_LNG
LL_EPSILON = cpu.LL_EPSILON

#: Maximum zoom for batch (array) operations вЂ” tile indices are int32,
#: which caps valid indices below 2**31 (x == 2**zoom must fit).
MAX_BATCH_ZOOM = 30

_backend = get_backend()

#: Which backend serves automatic batch dispatch: "opencl" or "cpu".
#: The Vulkan and CUDA backends are opt-in (see ``cyrcantile._vulkan`` /
#: ``cyrcantile._cuda``).
ACTIVE_BACKEND = "opencl" if _backend is not None else "cpu"

# ------------------------------------------------------------------ #
#  Utility functions (always CPU вЂ” trivial scalar math)
# ------------------------------------------------------------------ #

radians = cpu.radians
degrees = cpu.degrees
minmax = cpu.minmax
truncate_lnglat = cpu.truncate_lnglat
feature = cpu.feature


# ------------------------------------------------------------------ #
#  Input normalisation helpers
# ------------------------------------------------------------------ #

def _is_arraylike(v) -> bool:
    """True for lists/tuples (excluding named tuples) and non-0d arrays."""
    if isinstance(v, np.ndarray):
        return v.ndim > 0
    return isinstance(v, (list, tuple)) and not hasattr(v, "_fields")


def _is_tile_list(v) -> bool:
    """True for a non-empty list/tuple whose items are all Tiles."""
    return (
        isinstance(v, (list, tuple))
        and not hasattr(v, "_fields")
        and len(v) > 0
        and all(isinstance(t, Tile) for t in v)
    )


def _is_empty_batch(v) -> bool:
    if isinstance(v, np.ndarray):
        return v.size == 0
    return isinstance(v, (list, tuple)) and not hasattr(v, "_fields") and len(v) == 0


def _as_flat(v, dtype, name):
    """Coerce to a flat contiguous ndarray of *dtype*."""
    a = np.asarray(v)
    if a.ndim == 0:
        a = a.reshape(1)
    try:
        return np.ascontiguousarray(a.ravel(), dtype=dtype)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"{name} must be a flat sequence of numbers or a list of Tile, "
            f"got {type(v).__name__}"
        ) from e


def _batch_pair(a, b, name_a, name_b):
    aa = _as_flat(a, np.float64, name_a)
    ab = _as_flat(b, np.float64, name_b)
    if aa.shape[0] != ab.shape[0]:
        raise ValueError(
            f"{name_a} and {name_b} must have the same length "
            f"({aa.shape[0]} != {ab.shape[0]})"
        )
    return aa, ab


def _validate_batch_zoom(zoom) -> int:
    try:
        zoom = operator.index(zoom)
    except TypeError:
        raise TypeError(
            f"zoom must be an integer, got {type(zoom).__name__}"
        ) from None
    if not 0 <= zoom <= MAX_BATCH_ZOOM:
        raise ValueError(
            f"zoom must be in [0, {MAX_BATCH_ZOOM}] for batch operations, got {zoom}"
        )
    return zoom


def _tile_args(x, y, zoom, require_zoom=True):
    """Normalise (x, y[, zoom]) tile-coordinate batch inputs."""
    if y is None:
        raise ValueError(
            "batch tile operations require separate x and y arrays "
            "(or a list of Tile)"
        )
    if require_zoom and zoom is None:
        raise ValueError("zoom is required for batch tile operations")
    if zoom is not None:
        zoom = _validate_batch_zoom(zoom)
    xx = _as_flat(x, np.int32, "x")
    yy = _as_flat(y, np.int32, "y")
    if xx.shape[0] != yy.shape[0]:
        raise ValueError(
            f"x and y must have the same length ({xx.shape[0]} != {yy.shape[0]})"
        )
    return xx, yy, zoom


def _tiles_arrays(tiles):
    """Extract (x, y, z) int arrays from a list of Tile."""
    n = len(tiles)
    x = np.fromiter((t.x for t in tiles), dtype=np.int32, count=n)
    y = np.fromiter((t.y for t in tiles), dtype=np.int32, count=n)
    z = np.fromiter((t.z for t in tiles), dtype=np.int32, count=n)
    return x, y, z


def _run_grouped(tiles, fn):
    """Run ``fn(x, y, zoom)`` per zoom group; scatter outputs back in order.

    ``fn`` returns a tuple of 1-D arrays with one element per input tile.
    Returns the same tuple with elements in the original input order.
    """
    x, y, z = _tiles_arrays(tiles)
    n = len(tiles)
    uz = np.unique(z)
    if len(uz) == 1:
        return fn(x, y, int(uz[0]))
    outs = None
    for z0 in uz:
        idx = np.flatnonzero(z == z0)
        parts = fn(x[idx], y[idx], int(z0))
        if outs is None:
            outs = [np.zeros(n, dtype=p.dtype) for p in parts]
        for o, p in zip(outs, parts):
            o[idx] = p
    return tuple(outs)


def _empty_pair(dtype=np.float64):
    return np.empty(0, dtype=dtype), np.empty(0, dtype=dtype)


def _empty_quad(dtype=np.float64):
    return tuple(np.empty(0, dtype=dtype) for _ in range(4))


# ------------------------------------------------------------------ #
#  xy вЂ” geographic в†’ Web Mercator
# ------------------------------------------------------------------ #

def xy(lng, lat, truncate=False):
    """Convert longitude/latitude to Web Mercator x/y (metres).

    Scalar call returns a ``(x, y)`` tuple; array inputs return a pair
    of float64 arrays (GPU batch path when available).
    """
    if _is_arraylike(lng) or _is_arraylike(lat):
        a, b = _batch_pair(lng, lat, "lng", "lat")
        if _backend is not None:
            return _backend.xy(a, b, truncate=truncate)
        return cpu.xy_vec(a, b, truncate)
    r = cpu.xy(lng, lat, truncate)
    return r.x, r.y


# ------------------------------------------------------------------ #
#  lnglat вЂ” Web Mercator в†’ geographic
# ------------------------------------------------------------------ #

def lnglat(x, y):
    """Convert Web Mercator x/y to longitude/latitude (degrees)."""
    if _is_arraylike(x) or _is_arraylike(y):
        a, b = _batch_pair(x, y, "x", "y")
        if _backend is not None:
            return _backend.lnglat(a, b)
        return cpu.lnglat_vec(a, b)
    r = cpu.lnglat(x, y)
    return r.lng, r.lat


# ------------------------------------------------------------------ #
#  tile вЂ” geographic в†’ Tile
# ------------------------------------------------------------------ #

def tile(lng, lat, zoom, truncate=False):
    """Get the tile containing longitude/latitude at *zoom*.

    Scalar call returns a ``Tile``; array inputs return a pair of int32
    arrays ``(xs, ys)``.
    """
    if _is_arraylike(lng) or _is_arraylike(lat):
        a, b = _batch_pair(lng, lat, "lng", "lat")
        zoom = _validate_batch_zoom(zoom)
        if _backend is not None:
            return _backend.tile(a, b, zoom, truncate=truncate)
        return cpu.tile_vec(a, b, zoom, truncate)
    return cpu.tile(lng, lat, zoom, truncate)


# ------------------------------------------------------------------ #
#  Tile в†’ corner / bounds (batch forms)
# ------------------------------------------------------------------ #

def _reject_tilelist_extras(allowed=("to_zoom",)):
    if allowed:
        extra = f"only {', '.join(allowed)} may be given"
    else:
        extra = "no extra arguments are accepted"
    raise ValueError(
        f"when passing a list of Tile, {extra}; "
        "use (x, y, zoom) arrays otherwise"
    )


def _ul_batch(xx, yy, z):
    if _backend is not None:
        return _backend.ul(xx, yy, z)
    return cpu.ul_vec(xx, yy, z)


def _bounds_batch(xx, yy, z):
    if _backend is not None:
        return _backend.bounds(xx, yy, z)
    return cpu.bounds_vec(xx, yy, z)


def _xy_bounds_batch(xx, yy, z):
    if _backend is not None:
        return _backend.xy_bounds(xx, yy, z)
    return cpu.xy_bounds_vec(xx, yy, z)


def ul(x, y=None, zoom=None):
    """Upper-left (lng, lat) corner of a tile.

    Accepts a ``Tile``, a list of ``Tile`` (mixed zooms allowed) or
    ``(x, y, zoom)`` arrays; the batch forms return a pair of float64
    arrays ``(lngs, lats)``.
    """
    if isinstance(x, Tile):
        return cpu.ul(x)
    if _is_tile_list(x):
        if y is not None or zoom is not None:
            _reject_tilelist_extras()
        return _run_grouped(x, _ul_batch)
    if _is_empty_batch(x):
        return _empty_pair()
    xx, yy, z = _tile_args(x, y, zoom)
    return _ul_batch(xx, yy, z)


def bounds(x, y=None, zoom=None):
    """Geographic bounding box (w, s, e, n) of a tile.

    Accepts a ``Tile``, a list of ``Tile`` (mixed zooms allowed) or
    ``(x, y, zoom)`` arrays; the batch forms return a 4-tuple of float64
    arrays ``(wests, souths, easts, norths)``.
    """
    if isinstance(x, Tile):
        return cpu.bounds(x)
    if _is_tile_list(x):
        if y is not None or zoom is not None:
            _reject_tilelist_extras()
        return _run_grouped(x, _bounds_batch)
    if _is_empty_batch(x):
        return _empty_quad()
    xx, yy, z = _tile_args(x, y, zoom)
    return _bounds_batch(xx, yy, z)


def xy_bounds(x, y=None, zoom=None):
    """Web-Mercator bounding box (l, b, r, t) of a tile, in metres.

    Accepts a ``Tile``, a list of ``Tile`` (mixed zooms allowed) or
    ``(x, y, zoom)`` arrays; the batch forms return a 4-tuple of float64
    arrays ``(lefts, bottoms, rights, tops)``.
    """
    if isinstance(x, Tile):
        return cpu.xy_bounds(x)
    if _is_tile_list(x):
        if y is not None or zoom is not None:
            _reject_tilelist_extras()
        return _run_grouped(x, _xy_bounds_batch)
    if _is_empty_batch(x):
        return _empty_quad()
    xx, yy, z = _tile_args(x, y, zoom)
    return _xy_bounds_batch(xx, yy, z)


# ------------------------------------------------------------------ #
#  quadkey вЂ” Tile в†” quadkey string
# ------------------------------------------------------------------ #

def _quadkey_ints(xx, yy, zoom):
    if _backend is not None:
        return _backend.quadkey_encode(xx, yy, zoom)
    return cpu.quadkey_encode_vec(xx, yy, zoom)


def quadkey(x, y=None, zoom=None):
    """Quadkey string for a tile.

    Accepts a ``Tile``, a list of ``Tile`` (mixed zooms allowed) or
    ``(x, y, zoom)`` arrays; the batch forms return a list of strings.
    """
    if isinstance(x, Tile):
        return cpu.quadkey(x)
    if _is_tile_list(x):
        if y is not None or zoom is not None:
            _reject_tilelist_extras()
        xs, ys, zs = _tiles_arrays(x)
        out = [""] * len(x)
        for z0 in np.unique(zs):
            z0 = int(z0)
            idx = np.flatnonzero(zs == z0)
            ints = _quadkey_ints(xs[idx], ys[idx], z0)
            for i, s in zip(idx.tolist(), cpu.quadkey_ints_to_strings(ints, z0)):
                out[i] = s
        return out
    if _is_empty_batch(x):
        return []
    xx, yy, zoom = _tile_args(x, y, zoom)
    return cpu.quadkey_ints_to_strings(_quadkey_ints(xx, yy, zoom), zoom)


def quadkey_to_tile(qk, zoom=None):
    """Inverse of :func:`quadkey`.

    A single string returns a ``Tile``; a list of equal-length strings
    returns a list of ``Tile``; a packed uint64 array (plus *zoom*)
    returns a pair of int32 arrays ``(xs, ys)``.
    """
    if isinstance(qk, str):
        return cpu.quadkey_to_tile(qk)
    if isinstance(qk, np.ndarray) and qk.dtype == np.uint64:
        if zoom is None:
            raise ValueError("zoom is required for packed uint64 quadkeys")
        zoom = _validate_batch_zoom(zoom)
        if qk.size == 0:
            return _empty_pair(np.int32)
        if _backend is not None:
            return _backend.quadkey_decode(np.ascontiguousarray(qk.ravel()), zoom)
        return cpu.quadkey_decode_vec(qk.ravel(), zoom)
    keys = list(qk)
    if not keys:
        return []
    ints = cpu.quadkey_strings_to_ints(keys)  # validates uniform length + digits
    zoom = len(keys[0])
    if _backend is not None:
        ox, oy = _backend.quadkey_decode(ints, zoom)
    else:
        ox, oy = cpu.quadkey_decode_vec(ints, zoom)
    return [Tile(int(a), int(b), zoom) for a, b in zip(ox, oy)]


# ------------------------------------------------------------------ #
#  parent / children / neighbors
# ------------------------------------------------------------------ #

def _parent_batch(xx, yy, shift):
    if _backend is not None:
        return _backend.parent(xx, yy, shift)
    return cpu.parent_vec(xx, yy, shift)


def parent(x, y=None, zoom=None, to_zoom=None):
    """Ancestor tile of each input tile.

    Scalar form (mercantile-compatible): ``parent(tile)`` returns the
    tile one zoom lower; ``parent(tile, zoom)`` returns the ancestor at
    *target* zoom.  ``parent(Tile(0, 0, 0))`` returns ``None``.

    Batch form: ``parent(xs, ys, zoom, to_zoom=None)`` or a list of
    ``Tile``.  Here *zoom* is the source zoom of the input tiles and
    *to_zoom* the ancestor level (default ``zoom - 1``).  Returns a
    pair of int32 arrays ``(xs, ys)``.
    """
    if isinstance(x, Tile):
        # Scalar form: the target zoom may be given as the 2nd positional
        # argument (legacy parent(tile, zoom)), as ``zoom=`` (mercantile
        # keyword) or as ``to_zoom=``.
        target = to_zoom
        if target is None and zoom is not None:
            target = zoom
        if target is None and isinstance(y, int) and not isinstance(y, bool):
            target = y
        return cpu.parent(x, target)
    if _is_tile_list(x):
        if y is not None or zoom is not None:
            _reject_tilelist_extras(allowed=("to_zoom",))
        if to_zoom is None:
            xs, ys, _ = _tiles_arrays(x)
            return _parent_batch(xs, ys, 1)
        to_zoom = operator.index(to_zoom)
        xs, ys, zs = _tiles_arrays(x)
        if np.any(zs <= to_zoom):
            raise ValueError("to_zoom must be less than the zoom of every input tile")
        return _run_grouped(x, lambda xx, yy, zz: _parent_batch(xx, yy, zz - to_zoom))
    if _is_empty_batch(x):
        return _empty_pair(np.int32)
    xx, yy, zoom = _tile_args(x, y, zoom)
    if to_zoom is None:
        shift = 1
    else:
        to_zoom = operator.index(to_zoom)
        if not 0 <= to_zoom < zoom:
            raise ValueError(f"to_zoom must be in [0, {zoom - 1}]")
        shift = zoom - to_zoom
    return _parent_batch(xx, yy, shift)


def _children_batch(xx, yy):
    if _backend is not None:
        ox, oy = _backend.children(xx, yy)
    else:
        ox, oy = cpu.children_vec(xx, yy)
    return ox.reshape(-1, 4), oy.reshape(-1, 4)


def children(x, y=None, zoom=None):
    """Four children of a tile.

    Scalar form (``Tile`` input) returns the immediate children in
    mercantile order — top-left, top-right, bottom-right, bottom-left —
    or, with ``zoom`` given, all descendants at that zoom.

    Batch forms (a list of ``Tile`` or ``(x, y)`` arrays — *zoom* is
    accepted but unused) return a pair of int32 arrays of shape
    ``(n, 4)``, one child per column, in the same mercantile order.
    """
    if isinstance(x, Tile):
        # Scalar form: the target zoom may be given as the 2nd positional
        # argument (legacy children(tile, zoom)) or as ``zoom=``.
        target = zoom
        if target is None and isinstance(y, int) and not isinstance(y, bool):
            target = y
        return cpu.children(x, target)
    if _is_tile_list(x):
        if y is not None or zoom is not None:
            _reject_tilelist_extras(allowed=())
        xs, ys, _ = _tiles_arrays(x)
        return _children_batch(xs, ys)
    if _is_empty_batch(x):
        return (np.empty((0, 4), dtype=np.int32), np.empty((0, 4), dtype=np.int32))
    xx, yy, _ = _tile_args(x, y, zoom, require_zoom=False)
    return _children_batch(xx, yy)


def _neighbors_batch(xx, yy):
    if _backend is not None:
        ox, oy = _backend.neighbors(xx, yy)
    else:
        ox, oy = cpu.neighbors_vec(xx, yy)
    return ox.reshape(-1, 8), oy.reshape(-1, 8)


def neighbors(x, y=None, zoom=None):
    """Eight neighbours of a tile.

    Scalar form (``Tile`` input) returns up to eight ``Tile`` with
    invalid out-of-grid neighbours omitted (mercantile semantics); order
    is x-major.

    Batch forms (a list of ``Tile`` or ``(x, y)`` arrays — *zoom* is
    accepted but unused) return a pair of int32 arrays of shape
    ``(n, 8)`` in x-major order **without** filtering — apply a mask if
    you need only valid neighbours.
    """
    if isinstance(x, Tile):
        return cpu.neighbors(x)
    if _is_tile_list(x):
        if y is not None or zoom is not None:
            _reject_tilelist_extras(allowed=())
        xs, ys, _ = _tiles_arrays(x)
        return _neighbors_batch(xs, ys)
    if _is_empty_batch(x):
        return (np.empty((0, 8), dtype=np.int32), np.empty((0, 8), dtype=np.int32))
    xx, yy, _ = _tile_args(x, y, zoom, require_zoom=False)
    return _neighbors_batch(xx, yy)


# ------------------------------------------------------------------ #
#  bounding_tile / tiles
# ------------------------------------------------------------------ #

def bounding_tile(west, south, east, north, zoom=None, truncate=False):
    """Get the smallest tile containing a geographic bounding box.

    Without *zoom*, the deepest single covering tile is searched
    (mercantile algorithm).  With *zoom*, the tile of the north-west
    corner is returned.  Array inputs require *zoom* and return a pair
    of int32 arrays ``(xs, ys)``.
    """
    if (
        _is_arraylike(west) or _is_arraylike(south)
        or _is_arraylike(east) or _is_arraylike(north)
    ):
        if zoom is None:
            raise ValueError("zoom is required for batch bounding_tile")
        zoom = _validate_batch_zoom(zoom)
        w, s, e, n = _batch_quad(west, south, east, north)
        if w.shape[0] == 0:
            return _empty_pair(np.int32)
        if _backend is not None:
            return _backend.bounding_tile(w, n, zoom)
        return cpu.tile_vec(w, n, zoom, truncate)
    return cpu.bounding_tile(west, south, east, north, zoom, truncate)


def _batch_quad(w, s, e, n):
    ww = _as_flat(w, np.float64, "west")
    ss = _as_flat(s, np.float64, "south")
    ee = _as_flat(e, np.float64, "east")
    nn = _as_flat(n, np.float64, "north")
    lengths = {a.shape[0] for a in (ww, ss, ee, nn)}
    if len(lengths) > 1:
        raise ValueError(
            f"west, south, east and north must have the same length ({sorted(lengths)})"
        )
    return ww, ss, ee, nn


def tiles(west, south, east, north, zooms):
    """Enumerate all tiles covering a geographic bbox at the given zooms.

    Matches mercantile: inputs are clamped to the Web-Mercator limits, a
    small epsilon is applied to the south/east edges (so the bounds of a
    single tile yield exactly that tile) and antimeridian-crossing boxes
    (``west > east``) are split into two boxes.  Returns a list of
    ``Tile``.  Large grids (> 256 tiles per zoom) are grid-filled on the
    GPU.
    """
    if isinstance(zooms, int):
        zooms = [zooms]
    if west > east:
        bboxes = [(-180.0, south, east, north), (west, south, 180.0, north)]
    else:
        bboxes = [(west, south, east, north)]
    result = []
    for w, s, e, n in bboxes:
        # mercantile clamps to 85.051129 (a hair beyond MAX_LAT) so that
        # the poles map onto the edge tiles.
        w = max(-180.0, w)
        s = max(-85.051129, s)
        e = min(180.0, e)
        n = min(85.051129, n)
        for z in zooms:
            z = operator.index(z)
            tmin = cpu.tile(w, n, z)
            tmax = cpu.tile(e - LL_EPSILON, s + LL_EPSILON, z)
            gx = tmax.x - tmin.x + 1
            gy = tmax.y - tmin.y + 1
            if gx <= 0 or gy <= 0:
                continue
            if _backend is not None and gx * gy > 256:
                ox, oy = _backend.tiles_in_bbox(tmin.x, tmin.y, gx, gy)
                result.extend(Tile(xv, yv, z) for xv, yv in zip(ox.tolist(), oy.tolist()))
            else:
                for xv in range(tmin.x, tmax.x + 1):
                    for yv in range(tmin.y, tmax.y + 1):
                        result.append(Tile(xv, yv, z))
    return result
