"""Oracle tests against mercantile (skipped when mercantile is absent)."""

from __future__ import annotations

import pytest

mercantile = pytest.importorskip("mercantile")

import cyrcantile as ct
from cyrcantile._types import Tile


def _tiles(rng, n=200):
    out = []
    for _ in range(n):
        z = int(rng.randint(0, 13))
        out.append(Tile(int(rng.randint(0, 1 << z)), int(rng.randint(0, 1 << z)), z))
    return out


def test_tile_matches_mercantile(points):
    lngs, lats = points
    for lng, lat in zip(lngs[:200], lats[:200]):
        assert ct.tile(lng, lat, 9) == mercantile.tile(lng, lat, 9)


def test_tile_truncate_matches_mercantile(rng):
    # mercantile 1.2 clamps truncate() to ±90 and then errors; cyrcantile
    # clamps to the web-mercator limit (±85.05112878) instead, so the
    # comparison stays below ±90 where both implementations agree.
    lngs = rng.uniform(-200, 200, 100)
    lats = rng.uniform(-89.9, 89.9, 100)
    for lng, lat in zip(lngs, lats):
        assert ct.tile(lng, lat, 9, truncate=True) == mercantile.tile(lng, lat, 9, truncate=True)


def test_xy_matches_mercantile(rng):
    lngs = rng.uniform(-180, 180, 100)
    lats = rng.uniform(-89.9, 89.9, 100)
    for lng, lat in zip(lngs, lats):
        mx, my = mercantile.xy(lng, lat)
        cx, cy = ct.xy(lng, lat)
        assert cx == pytest.approx(mx, rel=1e-12)
        assert cy == pytest.approx(my, rel=1e-12)


def test_lnglat_matches_mercantile(merc):
    xs, ys = merc
    for x, y in zip(xs[:100], ys[:100]):
        mlng, mlat = mercantile.lnglat(x, y)
        clng, clat = ct.lnglat(x, y)
        assert clng == pytest.approx(mlng, rel=1e-12)
        assert clat == pytest.approx(mlat, rel=1e-12)


def test_ul_bounds_match_mercantile(rng):
    tiles = _tiles(rng)
    for t in tiles:
        mul = mercantile.ul(t.x, t.y, t.z)
        cul = ct.ul(t)
        assert cul.lng == pytest.approx(mul.lng, rel=1e-12, abs=1e-12)
        assert cul.lat == pytest.approx(mul.lat, rel=1e-12, abs=1e-12)

        mb = mercantile.bounds(t)
        cb = ct.bounds(t)
        assert cb.west == pytest.approx(mb.west, rel=1e-12, abs=1e-12)
        assert cb.east == pytest.approx(mb.east, rel=1e-12, abs=1e-12)
        assert cb.south == pytest.approx(mb.south, rel=1e-12, abs=1e-12)
        assert cb.north == pytest.approx(mb.north, rel=1e-12, abs=1e-12)


def test_xy_bounds_matches_mercantile(rng):
    tiles = _tiles(rng)
    for t in tiles:
        mb = mercantile.xy_bounds(t)
        cb = ct.xy_bounds(t)
        for field in ("left", "bottom", "right", "top"):
            assert getattr(cb, field) == pytest.approx(
                getattr(mb, field), rel=1e-12, abs=1e-9
            )


def test_quadkey_matches_mercantile(rng):
    tiles = _tiles(rng)
    for t in tiles:
        assert ct.quadkey(t) == mercantile.quadkey(t)


def test_parent_matches_mercantile(rng):
    tiles = _tiles(rng)
    for t in tiles:
        mp = mercantile.parent(t)
        cp = ct.parent(t)
        assert cp == mp
        if t.z >= 1:
            for target in range(0, t.z):
                assert ct.parent(t, target) == mercantile.parent(t, zoom=target)
                assert ct.parent(t, zoom=target) == mercantile.parent(t, zoom=target)


def test_children_matches_mercantile(rng):
    tiles = _tiles(rng)
    for t in tiles:
        assert ct.children(t) == mercantile.children(t)


def test_neighbors_matches_mercantile(rng):
    tiles = _tiles(rng)
    for t in tiles:
        ours = set(ct.neighbors(t))
        theirs = set(mercantile.neighbors(t))
        assert ours == theirs


def test_bounding_tile_matches_mercantile(rng):
    # note: mercantile's bounding_tile has no zoom argument; the zoomed
    # variant is a cyrcantile extension (tile of the NW corner)
    for _ in range(100):
        w = rng.uniform(-180, 175)
        e = w + rng.uniform(0.01, 180 - w)
        s = rng.uniform(-85, 80)
        n = s + rng.uniform(0.01, 85 - s)
        assert ct.bounding_tile(w, s, e, n) == mercantile.bounding_tile(w, s, e, n)


def test_tiles_matches_mercantile():
    ours = ct.tiles(-9.5, 53.0, -9.0, 53.3, [14, 15, 16])
    theirs = list(mercantile.tiles(-9.5, 53.0, -9.0, 53.3, [14, 15, 16]))
    assert set(ours) == set(theirs)
    assert len(ours) == len(theirs)


def test_tiles_world_matches_mercantile():
    for z in range(0, 6):
        ours = ct.tiles(-180, -85.05112878, 180, 85.05112878, [z])
        theirs = list(mercantile.tiles(-180, -85.05112878, 180, 85.05112878, [z]))
        assert set(ours) == set(theirs)
        assert len(ours) == len(theirs)


def test_tiles_antimeridian_matches_mercantile():
    ours = ct.tiles(170.0, -10.0, -170.0, 10.0, [2, 3])
    theirs = list(mercantile.tiles(170.0, -10.0, -170.0, 10.0, [2, 3]))
    assert set(ours) == set(theirs)
    assert len(ours) == len(theirs)
