"""Scalar API — hardcoded expected values (mercantile-compatible)."""

from __future__ import annotations

import math

import pytest

import cyrcantile as ct
from cyrcantile._types import Bbox, LngLat, LngLatBbox, Tile

T = Tile(60, 41, 7)


def test_tile():
    assert ct.tile(-9.14, 53.12, 7) == Tile(60, 41, 7)
    assert ct.tile(-9.14, 53.12, 7, truncate=True) == Tile(60, 41, 7)


def test_xy():
    x, y = ct.xy(-9.14, 53.12)
    assert x == pytest.approx(-1017460.1458505205)
    assert y == pytest.approx(7005225.592415733)


def test_xy_poles_match_mercantile():
    assert ct.xy(0.0, 90.0)[1] == math.inf
    assert ct.xy(0.0, -90.0)[1] == -math.inf


def test_xy_truncate():
    assert ct.xy(200.0, 100.0, truncate=True) == ct.xy(180.0, 85.05112878)
    assert ct.xy(-200.0, -100.0, truncate=True) == ct.xy(-180.0, -85.05112878)


def test_lnglat_roundtrip():
    x, y = ct.xy(-9.14, 53.12)
    lng, lat = ct.lnglat(x, y)
    assert lng == pytest.approx(-9.14, abs=1e-9)
    assert lat == pytest.approx(53.12, abs=1e-9)


def test_ul():
    assert ct.ul(T) == LngLat(lng=-11.25, lat=54.1624339680678)


def test_bounds():
    b = ct.bounds(T)
    assert b == LngLatBbox(
        west=-11.25, south=52.48278022207821, east=-8.4375, north=54.1624339680678
    )


def test_xy_bounds():
    b = ct.xy_bounds(T)
    assert b == Bbox(
        left=-1252344.271424327, bottom=6887893.492833803,
        right=-939258.203568245, top=7200979.560689885,
    )
    assert b.right - b.left == pytest.approx(40075016.68557849 / 128)


def test_quadkey():
    assert ct.quadkey(T) == "0313102"
    assert ct.quadkey(Tile(0, 0, 0)) == ""


def test_quadkey_to_tile():
    assert ct.quadkey_to_tile("0313102") == T
    assert ct.quadkey_to_tile("") == Tile(0, 0, 0)


def test_parent():
    assert ct.parent(T) == Tile(30, 20, 6)
    # legacy positional target zoom
    assert ct.parent(T, 4) == Tile(7, 5, 4)
    # mercantile keyword
    assert ct.parent(T, zoom=4) == Tile(7, 5, 4)
    assert ct.parent(T, to_zoom=4) == Tile(7, 5, 4)
    # mercantile: parent of a z=0 tile is None
    assert ct.parent(Tile(0, 0, 0)) is None


def test_parent_validation():
    with pytest.raises(ValueError):
        ct.parent(T, zoom=7)  # target must be below the tile's zoom
    with pytest.raises(ValueError):
        ct.parent(T, zoom=2.5)


def test_children():
    # mercantile order: top-left, top-right, bottom-right, bottom-left
    ch = ct.children(T)
    assert ch == [Tile(120, 82, 8), Tile(121, 82, 8),
                  Tile(121, 83, 8), Tile(120, 83, 8)]


def test_children_descend_zoom():
    assert ct.children(T, 8) == ct.children(T)  # one level down
    two = ct.children(T, 9)
    assert len(two) == 16
    expected = [c for t in ct.children(T) for c in ct.children(t)]
    assert two == expected
    with pytest.raises(ValueError):
        ct.children(T, 7)  # target zoom must be greater than the tile's zoom


def test_neighbors():
    nb = ct.neighbors(T)
    assert len(nb) == 8
    assert Tile(59, 40, 7) in nb
    assert Tile(61, 42, 7) in nb
    assert T not in nb


def test_bounding_tile_with_zoom():
    # tile of the NW corner
    assert ct.bounding_tile(-9.5, 53.0, -9.0, 53.3, zoom=8) == ct.tile(-9.5, 53.0, 8)


def test_bounding_tile_search():
    # matches mercantile: deepest single covering tile
    assert ct.bounding_tile(-9.5, 53.0, -9.0, 53.3) == Tile(121, 83, 8)
    # world bbox -> z0
    assert ct.bounding_tile(-180.0, -85.0, 180.0, 85.0) == Tile(0, 0, 0)


def test_tiles():
    result = ct.tiles(-9.5, 53.0, -9.0, 53.3, [14, 15, 16])
    assert len(result) == 11202
    assert result[0] == Tile(7759, 5314, 14)
    # world at z1 = 4 tiles
    assert len(ct.tiles(-180, -85, 180, 85, 1)) == 4
    # world at z0 = exactly one tile
    assert ct.tiles(-180, -85.05112878, 180, 85.05112878, 0) == [Tile(0, 0, 0)]
    # zooms as int
    assert ct.tiles(-9.5, 53.0, -9.0, 53.3, 14) == ct.tiles(-9.5, 53.0, -9.0, 53.3, [14])


def test_tiles_single_tile_bounds():
    # mercantile property: the bounds of a tile yield exactly that tile
    t = Tile(8056, 5455, 14)
    b = ct.bounds(t)
    assert ct.tiles(b.west, b.south, b.east, b.north, [14]) == [t]


def test_tiles_antimeridian():
    got = ct.tiles(170.0, -10.0, -170.0, 10.0, [2])
    assert len(got) == 4
    xs = sorted(t.x for t in got)
    assert xs[0] <= 1 and xs[-1] >= 2  # tiles on both sides of the dateline


def test_tiles_clamps_inputs():
    inner = ct.tiles(-9.5, 53.0, -9.0, 53.3, [12])
    outer = ct.tiles(-200.0, -100.0, 200.0, 100.0, [12])
    assert len(inner) < len(outer)


def test_truncate_lnglat():
    assert ct.truncate_lnglat(200.0, 100.0) == (180.0, 85.05112878)
    assert ct.truncate_lnglat(-200.0, -100.0) == (-180.0, -85.05112878)


def test_radians_degrees_minmax():
    assert ct.radians(180.0) == pytest.approx(math.pi)
    assert ct.degrees(math.pi) == pytest.approx(180.0)
    assert ct.minmax(3, 1, 2) == (1, 3)


def test_feature():
    feat = ct.feature(T, fid=1, props={"layer": "base"})
    assert feat["type"] == "Feature"
    assert feat["id"] == 1
    assert feat["properties"] == {"layer": "base"}
    geom = feat["geometry"]
    assert geom["type"] == "Polygon"
    ring = geom["coordinates"][0]
    assert ring[0] == ring[-1]
    assert len(ring) == 5
    b = ct.bounds(T)
    assert ring[0] == [b.west, b.north]


def test_exports():
    for name in ct.__all__:
        assert hasattr(ct, name)
    assert ct.ACTIVE_BACKEND in ("opencl", "cpu")
