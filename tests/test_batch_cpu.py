"""Batch (vectorised) CPU paths — every batch result must equal the scalar
loop over the same inputs, on every engine (single / numba / threads)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import random_tiles

import cyrcantile as ct
from cyrcantile import _cpu
from cyrcantile._types import Tile

# ------------------------------------------------------------------ #
#  geographic batches
# ------------------------------------------------------------------ #

def test_tile_batch(points):
    lngs, lats = points
    xs, ys = ct.tile(lngs, lats, 9)
    exp_x = np.array([ct.tile(lo, p, 9).x for lo, p in zip(lngs, lats)], dtype=np.int32)
    exp_y = np.array([ct.tile(lo, p, 9).y for lo, p in zip(lngs, lats)], dtype=np.int32)
    np.testing.assert_array_equal(xs, exp_x)
    np.testing.assert_array_equal(ys, exp_y)
    assert xs.dtype == np.int32


def test_tile_batch_truncate(rng):
    lngs = rng.uniform(-200, 200, 500)
    lats = rng.uniform(-100, 100, 500)
    xs, ys = ct.tile(lngs, lats, 9, truncate=True)
    exp = [ct.tile(lo, p, 9, truncate=True) for lo, p in zip(lngs, lats)]
    np.testing.assert_array_equal(xs, [t.x for t in exp])
    np.testing.assert_array_equal(ys, [t.y for t in exp])


def test_tile_batch_single_element():
    xs, ys = ct.tile([-9.14], [53.12], 7)
    assert xs.shape == (1,)
    np.testing.assert_array_equal(xs, [60])
    np.testing.assert_array_equal(ys, [41])


def test_xy_batch(points):
    lngs, lats = points
    xs, ys = ct.xy(lngs, lats)
    exp = [ct.xy(lo, p) for lo, p in zip(lngs, lats)]
    np.testing.assert_allclose(xs, [e[0] for e in exp], rtol=1e-12)
    np.testing.assert_allclose(ys, [e[1] for e in exp], rtol=1e-12)


def test_lnglat_batch(merc):
    xs, ys = merc
    lngs, lats = ct.lnglat(xs, ys)
    exp = [ct.lnglat(x, y) for x, y in zip(xs, ys)]
    np.testing.assert_allclose(lngs, [e[0] for e in exp], rtol=1e-12)
    np.testing.assert_allclose(lats, [e[1] for e in exp], rtol=1e-12)


def test_roundtrip_batch(points):
    lngs, lats = points
    x, y = ct.xy(lngs, lats)
    lng2, lat2 = ct.lnglat(x, y)
    np.testing.assert_allclose(lng2, lngs, rtol=1e-12)
    np.testing.assert_allclose(lat2, lats, rtol=1e-12)


# ------------------------------------------------------------------ #
#  tile batches: (x, y, zoom) arrays and tile lists
# ------------------------------------------------------------------ #

def test_ul_batch_arrays():
    xs, ys, tiles = random_tiles(300, 10)
    lngs, lats = ct.ul(xs, ys, 10)
    exp = [ct.ul(t) for t in tiles]
    np.testing.assert_allclose(lngs, [e.lng for e in exp], rtol=1e-12)
    np.testing.assert_allclose(lats, [e.lat for e in exp], rtol=1e-12)


def test_ul_batch_tile_list_mixed_zoom():
    tiles = [Tile(60, 41, 7), Tile(30, 20, 6), Tile(121, 83, 8),
             Tile(2000, 1300, 12)]
    lngs, lats = ct.ul(tiles)
    exp = [ct.ul(t) for t in tiles]
    np.testing.assert_allclose(lngs, [e.lng for e in exp], rtol=1e-12)
    np.testing.assert_allclose(lats, [e.lat for e in exp], rtol=1e-12)


def test_bounds_batch():
    xs, ys, tiles = random_tiles(300, 10)
    w, s, e, n = ct.bounds(xs, ys, 10)
    exp = [ct.bounds(t) for t in tiles]
    np.testing.assert_allclose(w, [b.west for b in exp], rtol=1e-12)
    np.testing.assert_allclose(s, [b.south for b in exp], rtol=1e-12)
    np.testing.assert_allclose(e, [b.east for b in exp], rtol=1e-12)
    np.testing.assert_allclose(n, [b.north for b in exp], rtol=1e-12)


def test_bounds_batch_tile_list_mixed_zoom():
    tiles = [Tile(60, 41, 7), Tile(30, 20, 6), Tile(121, 83, 8)]
    w, s, e, n = ct.bounds(tiles)
    exp = [ct.bounds(t) for t in tiles]
    np.testing.assert_allclose(w, [b.west for b in exp], rtol=1e-12)
    np.testing.assert_allclose(n, [b.north for b in exp], rtol=1e-12)


def test_xy_bounds_batch():
    xs, ys, tiles = random_tiles(300, 10)
    lf, b, r, t = ct.xy_bounds(xs, ys, 10)
    exp = [ct.xy_bounds(tt) for tt in tiles]
    np.testing.assert_allclose(lf, [bb.left for bb in exp], rtol=1e-12)
    np.testing.assert_allclose(b, [bb.bottom for bb in exp], rtol=1e-12)
    np.testing.assert_allclose(r, [bb.right for bb in exp], rtol=1e-12)
    np.testing.assert_allclose(t, [bb.top for bb in exp], rtol=1e-12)


def test_parent_batch():
    xs, ys, tiles = random_tiles(300, 10)
    px, py = ct.parent(xs, ys, 10)
    np.testing.assert_array_equal(px, xs >> 1)
    np.testing.assert_array_equal(py, ys >> 1)


def test_parent_batch_target_zoom():
    xs, ys, tiles = random_tiles(300, 10)
    px, py = ct.parent(xs, ys, 10, to_zoom=4)
    exp = [ct.parent(t, 4) for t in tiles]
    np.testing.assert_array_equal(px, [t.x for t in exp])
    np.testing.assert_array_equal(py, [t.y for t in exp])


def test_parent_batch_tile_list_mixed_zoom():
    tiles = [Tile(60, 41, 7), Tile(30, 20, 6), Tile(121, 83, 8)]
    px, py = ct.parent(tiles)
    np.testing.assert_array_equal(px, [t.x >> 1 for t in tiles])
    np.testing.assert_array_equal(py, [t.y >> 1 for t in tiles])
    px, py = ct.parent(tiles, to_zoom=4)
    exp = [ct.parent(t, 4) for t in tiles]
    np.testing.assert_array_equal(px, [t.x for t in exp])
    np.testing.assert_array_equal(py, [t.y for t in exp])


def test_children_batch():
    xs, ys, tiles = random_tiles(200, 10)
    cx, cy = ct.children(xs, ys, 10)
    assert cx.shape == (200, 4)
    assert cy.shape == (200, 4)
    for i, t in enumerate(tiles):
        exp = ct.children(t)
        np.testing.assert_array_equal(cx[i], [c.x for c in exp])
        np.testing.assert_array_equal(cy[i], [c.y for c in exp])


def test_children_batch_tile_list():
    tiles = [Tile(60, 41, 7), Tile(30, 20, 6)]
    cx, cy = ct.children(tiles)
    assert cx.shape == (2, 4)
    exp = [c for t in tiles for c in ct.children(t)]
    np.testing.assert_array_equal(cx.ravel(), [c.x for c in exp])
    np.testing.assert_array_equal(cy.ravel(), [c.y for c in exp])


def test_neighbors_batch():
    xs, ys, tiles = random_tiles(200, 10)
    nx, ny = ct.neighbors(xs, ys, 10)
    assert nx.shape == (200, 8)
    # batch neighbours are unfiltered and x-major ordered
    from cyrcantile._cpu import _NEIGH_DX, _NEIGH_DY
    np.testing.assert_array_equal(nx, xs[:, None] + _NEIGH_DX)
    np.testing.assert_array_equal(ny, ys[:, None] + _NEIGH_DY)


def test_neighbors_batch_matches_scalar_for_interior_tiles():
    # interior tiles: no neighbour gets filtered by the scalar API
    r = np.random.RandomState(5)
    xs = r.randint(1, (1 << 10) - 2, 100).astype(np.int32)
    ys = r.randint(1, (1 << 10) - 2, 100).astype(np.int32)
    nx, ny = ct.neighbors(xs, ys, 10)
    for i in range(len(xs)):
        exp = ct.neighbors(Tile(int(xs[i]), int(ys[i]), 10))
        np.testing.assert_array_equal(nx[i], [nb.x for nb in exp])
        np.testing.assert_array_equal(ny[i], [nb.y for nb in exp])


def test_bounding_tile_batch():
    r = np.random.RandomState(7)
    wests = r.uniform(-180, 170, 300)
    norths = r.uniform(-84, 84, 300)
    easts = wests + 0.1
    souths = norths - 0.1
    xs, ys = ct.bounding_tile(wests, souths, easts, norths, zoom=8)
    exp = [ct.bounding_tile(w, s, e, n, zoom=8) for w, s, e, n in zip(wests, souths, easts, norths)]
    np.testing.assert_array_equal(xs, [t.x for t in exp])
    np.testing.assert_array_equal(ys, [t.y for t in exp])


# ------------------------------------------------------------------ #
#  quadkey batches
# ------------------------------------------------------------------ #

def test_quadkey_batch():
    xs, ys, tiles = random_tiles(300, 10)
    qks = ct.quadkey(xs, ys, 10)
    assert qks == [ct.quadkey(t) for t in tiles]


def test_quadkey_batch_tile_list_mixed_zoom():
    tiles = [Tile(60, 41, 7), Tile(30, 20, 6), Tile(121, 83, 8)]
    assert ct.quadkey(tiles) == [ct.quadkey(t) for t in tiles]


def test_quadkey_to_tile_batch():
    xs, ys, tiles = random_tiles(300, 10)
    qks = [ct.quadkey(t) for t in tiles]
    back = ct.quadkey_to_tile(qks)
    assert back == tiles


def test_quadkey_to_tile_packed():
    xs, ys, tiles = random_tiles(300, 10)
    ints = _cpu.quadkey_encode_vec(xs, ys, 10)
    ox, oy = ct.quadkey_to_tile(ints, zoom=10)
    np.testing.assert_array_equal(ox, xs)
    np.testing.assert_array_equal(oy, ys)


# ------------------------------------------------------------------ #
#  empty inputs
# ------------------------------------------------------------------ #

def test_empty_batches():
    assert ct.xy([], [])[0].shape == (0,)
    assert ct.lnglat([], [])[0].shape == (0,)
    xs, ys = ct.tile([], [], 7)
    assert xs.shape == (0,) and ys.shape == (0,)
    lngs, lats = ct.ul([], [], 7)
    assert lngs.shape == (0,) and lats.shape == (0,)
    assert len(ct.bounds([], [], 7)) == 4
    assert len(ct.xy_bounds([], [], 7)) == 4
    assert ct.quadkey([]) == []
    assert ct.quadkey_to_tile([]) == []
    px, py = ct.parent([], [], 7)
    assert px.shape == (0,)
    cx, cy = ct.children([], [])
    assert cx.shape == (0, 4)
    nx, ny = ct.neighbors([], [])
    assert nx.shape == (0, 8)
    bx, by = ct.bounding_tile([], [], [], [], zoom=7)
    assert bx.shape == (0,)
    assert ct.ul([])[0].shape == (0,)
    assert ct.children([])[0].shape == (0, 4)


# ------------------------------------------------------------------ #
#  engine coverage (single / numba / threads)
# ------------------------------------------------------------------ #

BIG = 250_001  # above _THREAD_THRESHOLD


def test_engine_choice():
    assert _cpu._choose_engine(10) == "single"
    if _cpu.HAS_NUMBA:
        assert _cpu._choose_engine(BIG) == "numba"
    elif _cpu._CPU_WORKERS > 1:
        assert _cpu._choose_engine(BIG) == "thread"


@pytest.mark.skipif(not _cpu.HAS_NUMBA, reason="numba not installed")
def test_tile_vec_numba_engine():
    r = np.random.RandomState(3)
    lngs = r.uniform(-180, 180, BIG)
    lats = r.uniform(-85, 85, BIG)
    assert _cpu._choose_engine(BIG) == "numba"
    xs, ys = _cpu.tile_vec(lngs, lats, 9)
    ex, ey = _cpu._tile_chunk(lngs, lats, 9, False, 0, BIG)
    np.testing.assert_array_equal(xs, ex)
    np.testing.assert_array_equal(ys, ey)


def test_tile_vec_thread_engine(monkeypatch):
    if _cpu._CPU_WORKERS <= 1:
        pytest.skip("single CPU worker")
    monkeypatch.setattr(_cpu, "HAS_NUMBA", False)
    assert _cpu._choose_engine(BIG) == "thread"
    r = np.random.RandomState(3)
    lngs = r.uniform(-180, 180, BIG)
    lats = r.uniform(-85, 85, BIG)
    xs, ys = _cpu.tile_vec(lngs, lats, 9)
    ex, ey = _cpu._tile_chunk(lngs, lats, 9, False, 0, BIG)
    np.testing.assert_array_equal(xs, ex)
    np.testing.assert_array_equal(ys, ey)


def test_xy_vec_thread_engine_ul_bounds(monkeypatch):
    """ul_vec / bounds_vec / xy_bounds_vec through the thread engine."""
    if _cpu._CPU_WORKERS <= 1:
        pytest.skip("single CPU worker")
    monkeypatch.setattr(_cpu, "HAS_NUMBA", False)
    n = BIG
    tx = np.arange(n, dtype=np.int32)
    ty = np.arange(n, dtype=np.int32) * 2
    lngs, lats = _cpu.ul_vec(tx, ty, 9)
    elng, elat = _cpu._ul_chunk(tx, ty, 9, 0, n)
    np.testing.assert_allclose(lngs, elng, rtol=1e-12)
    np.testing.assert_allclose(lats, elat, rtol=1e-12)

    w, s, e, no = _cpu.bounds_vec(tx, ty, 9)
    ew, es, ee, en = _cpu._bounds_chunk(tx, ty, 9, 0, n)
    np.testing.assert_allclose(w, ew, rtol=1e-12)
    np.testing.assert_allclose(no, en, rtol=1e-12)

    lf, b, r, t = _cpu.xy_bounds_vec(tx, ty, 9)
    el, eb, er, et = _cpu._xy_bounds_chunk(tx, ty, 9, 0, n)
    np.testing.assert_allclose(lf, el, rtol=1e-12)
    np.testing.assert_allclose(t, et, rtol=1e-12)


# ------------------------------------------------------------------ #
#  tiles() grid fill
# ------------------------------------------------------------------ #

def test_tiles_gpu_grid_equals_cpu_loop():
    """With or without the GPU grid-fill path the enumeration must agree."""
    zooms = [10, 12]
    got = ct.tiles(-9.5, 53.0, -9.0, 53.3, zooms)
    expected = []
    for z in zooms:
        tmin = ct.tile(-9.5, 53.3, z)
        tmax = ct.tile(-9.0 - 1e-11, 53.0 + 1e-11, z)
        expected += [
            Tile(x, y, z)
            for x in range(tmin.x, tmax.x + 1)
            for y in range(tmin.y, tmax.y + 1)
        ]
    assert got == expected


def test_tiles_small_bbox_stays_on_cpu_path():
    got = ct.tiles(-9.5, 53.0, -9.0, 53.3, [12])
    assert len(got) > 0
    assert all(isinstance(t, Tile) for t in got)
