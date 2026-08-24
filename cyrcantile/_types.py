"""Named-tuple data types mirroring the original cyrcantile Cython structs."""

from typing import NamedTuple


class Tile(NamedTuple):
    """An XYZ web-mercator tile."""
    x: int
    y: int
    z: int


class LngLat(NamedTuple):
    """A longitude / latitude pair in degrees (WGS-84)."""
    lng: float
    lat: float


class LngLatBbox(NamedTuple):
    """A geographic bounding box in degrees."""
    west: float
    south: float
    east: float
    north: float


class Bbox(NamedTuple):
    """A Web-Mercator (EPSG:3857) bounding box in metres."""
    left: float
    bottom: float
    right: float
    top: float


class XY(NamedTuple):
    """A Web-Mercator (EPSG:3857) coordinate in metres."""
    x: float
    y: float


class MinMax(NamedTuple):
    """Inclusive integer range."""
    min: int
    max: int
