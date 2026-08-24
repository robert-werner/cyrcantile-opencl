"""Pure-Python CPU implementations — used for single-tile calls and as fallback
when no OpenCL device is available.  The math mirrors mercantile exactly."""

import math

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
#  Vectorised CPU fallbacks (numpy) — used when OpenCL is unavailable
# ------------------------------------------------------------------ #

def xy_vec(lng, lat, truncate=False):
    import numpy as _np
    if truncate:
        lng = _np.clip(lng, -MAX_LNG, MAX_LNG)
        lat = _np.clip(lat, -MAX_LAT, MAX_LAT)
    lat_r = lat * D2R
    return RE * lng * D2R, RE * _np.log(_np.tan(_np.pi / 4 + lat_r / 2))


def lnglat_vec(x, y):
    import numpy as _np
    return (x / RE) * R2D, (2 * _np.arctan(_np.exp(y / RE)) - _np.pi / 2) * R2D


def tile_vec(lng, lat, zoom, truncate=False):
    import numpy as _np
    if truncate:
        lng = _np.clip(lng, -MAX_LNG, MAX_LNG)
        lat = _np.clip(lat, -MAX_LAT, MAX_LAT)
    lat_r = lat * D2R
    n = 1 << zoom
    x = _np.floor((lng + 180.0) / 360.0 * n).astype(_np.int32)
    y = _np.floor((1.0 - _np.log(_np.tan(_np.pi / 4 + lat_r / 2)) / _np.pi) * 0.5 * n).astype(_np.int32)
    return x, y


def ul_vec(tx, zoom):
    import numpy as _np
    n = float(1 << zoom)
    lng = tx / n * 360.0 - 180.0
    lat = _np.arctan(_np.sinh(_np.pi * (1 - 2 * tx / n))) * R2D
    return lng, lat


def bounds_vec(tx, zoom):
    import numpy as _np
    n = float(1 << zoom)
    w = tx / n * 360.0 - 180.0
    e = (tx + 1) / n * 360.0 - 180.0
    north = _np.arctan(_np.sinh(_np.pi * (1 - 2 * tx / n))) * R2D
    south = _np.arctan(_np.sinh(_np.pi * (1 - 2 * (tx + 1) / n))) * R2D
    return w, south, e, north


def xy_bounds_vec(tx, zoom):
    import numpy as _np
    n = float(1 << zoom)
    t_size = CE / n
    left = tx * t_size - CE / 2
    right = left + t_size
    top = CE / 2 - tx * t_size
    bottom = top - t_size
    return left, bottom, right, top
