"""GPU backends (OpenCL / Vulkan / CUDA) must reproduce the CPU results.

OpenCL and CUDA compute in f64 — results are required to match the CPU
closely.  Vulkan computes in f32, so float outputs use loose tolerances
and tile indices may differ by at most 1 at floor boundaries.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from conftest import random_tiles

from cyrcantile import _cpu


def _available_backends():
    out = []
    for name, module in (
        ("opencl", "cyrcantile._backend"),
        ("vulkan", "cyrcantile._vulkan"),
        ("cuda", "cyrcantile._cuda"),
    ):
        try:
            mod = importlib.import_module(module)
            backend = mod.get_backend()
        except Exception:
            backend = None
        if backend is not None:
            out.append((name, backend))
    return out


BACKENDS = _available_backends()
IDS = [name for name, _ in BACKENDS]


def _tol(name):
    if name == "vulkan":
        return dict(rtol=2e-4, atol=2.0)  # f32 storage
    return dict(rtol=1e-9, atol=1e-9)  # f64


def _assert_ints_close(name, actual, desired, max_off=0):
    if name == "vulkan":
        np.testing.assert_array_less(np.abs(actual - desired), max_off + 1)
    else:
        np.testing.assert_array_equal(actual, desired)


@pytest.mark.skipif(not BACKENDS, reason="no GPU backend available")
@pytest.mark.parametrize("name,backend", BACKENDS, ids=IDS)
class TestBackends:
    N = 2000

    def test_tile(self, name, backend):
        rng = np.random.RandomState(11)
        lngs = rng.uniform(-179.9, 179.9, self.N)
        lats = rng.uniform(-84.9, 84.9, self.N)
        xs, ys = backend.tile(lngs, lats, 12)
        ex, ey = _cpu.tile_vec(lngs, lats, 12)
        _assert_ints_close(name, xs, ex, max_off=1)
        _assert_ints_close(name, ys, ey, max_off=1)

    def test_tile_truncate(self, name, backend):
        rng = np.random.RandomState(12)
        lngs = rng.uniform(-200, 200, self.N)
        lats = rng.uniform(-100, 100, self.N)
        xs, ys = backend.tile(lngs, lats, 9, truncate=True)
        ex, ey = _cpu.tile_vec(lngs, lats, 9, truncate=True)
        _assert_ints_close(name, xs, ex, max_off=1)
        _assert_ints_close(name, ys, ey, max_off=1)

    def test_xy(self, name, backend):
        rng = np.random.RandomState(13)
        lngs = rng.uniform(-179.9, 179.9, self.N)
        lats = rng.uniform(-84.9, 84.9, self.N)
        xs, ys = backend.xy(lngs, lats)
        ex, ey = _cpu.xy_vec(lngs, lats)
        np.testing.assert_allclose(xs, ex, **_tol(name))
        np.testing.assert_allclose(ys, ey, **_tol(name))

    def test_lnglat(self, name, backend):
        rng = np.random.RandomState(14)
        xs = rng.uniform(-2e7, 2e7, self.N)
        ys = rng.uniform(-2e7, 2e7, self.N)
        lngs, lats = backend.lnglat(xs, ys)
        elngs, elats = _cpu.lnglat_vec(xs, ys)
        np.testing.assert_allclose(lngs, elngs, **_tol(name))
        np.testing.assert_allclose(lats, elats, **_tol(name))

    def test_ul(self, name, backend):
        xs, ys, _ = random_tiles(self.N, 10, seed=15)
        lngs, lats = backend.ul(xs, ys, 10)
        elngs, elats = _cpu.ul_vec(xs, ys, 10)
        np.testing.assert_allclose(lngs, elngs, **_tol(name))
        np.testing.assert_allclose(lats, elats, **_tol(name))

    def test_bounds(self, name, backend):
        xs, ys, _ = random_tiles(self.N, 10, seed=16)
        w, s, e, n = backend.bounds(xs, ys, 10)
        ew, es, ee, en = _cpu.bounds_vec(xs, ys, 10)
        for got, exp in zip((w, s, e, n), (ew, es, ee, en)):
            np.testing.assert_allclose(got, exp, **_tol(name))

    def test_xy_bounds(self, name, backend):
        xs, ys, _ = random_tiles(self.N, 10, seed=17)
        lf, b, r, t = backend.xy_bounds(xs, ys, 10)
        el, eb, er, et = _cpu.xy_bounds_vec(xs, ys, 10)
        for got, exp in zip((lf, b, r, t), (el, eb, er, et)):
            np.testing.assert_allclose(got, exp, **_tol(name))

    def test_quadkey_roundtrip(self, name, backend):
        xs, ys, tiles = random_tiles(self.N, 10, seed=18)
        ints = backend.quadkey_encode(xs, ys, 10)
        exp_ints = _cpu.quadkey_encode_vec(xs, ys, 10)
        np.testing.assert_array_equal(ints, exp_ints)
        ox, oy = backend.quadkey_decode(ints, 10)
        np.testing.assert_array_equal(ox, xs)
        np.testing.assert_array_equal(oy, ys)

    def test_parent(self, name, backend):
        xs, ys, _ = random_tiles(self.N, 10, seed=19)
        for shift in (1, 3):
            px, py = backend.parent(xs, ys, shift)
            np.testing.assert_array_equal(px, xs >> shift)
            np.testing.assert_array_equal(py, ys >> shift)
        # shift 0 is the identity
        px, py = backend.parent(xs, ys, 0)
        np.testing.assert_array_equal(px, xs)
        np.testing.assert_array_equal(py, ys)

    def test_children(self, name, backend):
        xs, ys, tiles = random_tiles(500, 10, seed=20)
        ox, oy = backend.children(xs, ys)
        assert ox.shape == (2000,)
        exp_x = [c.x for t in tiles for c in _tile_children(t)]
        exp_y = [c.y for t in tiles for c in _tile_children(t)]
        np.testing.assert_array_equal(ox, exp_x)
        np.testing.assert_array_equal(oy, exp_y)

    def test_neighbors(self, name, backend):
        xs, ys, _ = random_tiles(500, 10, seed=21)
        ox, oy = backend.neighbors(xs, ys)
        assert ox.shape == (4000,)
        # batch neighbours are unfiltered and x-major ordered
        from cyrcantile._cpu import _NEIGH_DX, _NEIGH_DY
        np.testing.assert_array_equal(ox, (np.repeat(xs, 8).reshape(-1, 8) + _NEIGH_DX).ravel())
        np.testing.assert_array_equal(oy, (np.repeat(ys, 8).reshape(-1, 8) + _NEIGH_DY).ravel())

    def test_bounding_tile(self, name, backend):
        rng = np.random.RandomState(22)
        wests = rng.uniform(-180, 170, self.N)
        norths = rng.uniform(-84, 84, self.N)
        xs, ys = backend.bounding_tile(wests, norths, 8)
        ex, ey = _cpu.tile_vec(wests, norths, 8)
        _assert_ints_close(name, xs, ex, max_off=1)
        _assert_ints_close(name, ys, ey, max_off=1)

    def test_tiles_in_bbox(self, name, backend):
        ox, oy = backend.tiles_in_bbox(8056, 5455, 14, 6)
        assert ox.shape == (84,)
        for i, (x, y) in enumerate(zip(ox, oy)):
            assert x == 8056 + i % 14
            assert y == 5455 + i // 14

    def test_repeated_calls_consistent(self, name, backend):
        """Catches buffer-pool / reuse bugs: same input twice, same result."""
        rng = np.random.RandomState(23)
        lngs = rng.uniform(-180, 180, 1234)
        lats = rng.uniform(-85, 85, 1234)
        a = backend.tile(lngs, lats, 7)
        b = backend.tile(lngs, lats, 7)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        c = backend.xy(lngs, lats)  # same-size buffers reused from the pool
        d = backend.xy(lngs, lats)
        np.testing.assert_array_equal(c[0], d[0])
        np.testing.assert_array_equal(c[1], d[1])


def _tile_children(t):
    import cyrcantile as ct
    return ct.children(t)


@pytest.mark.skipif(not BACKENDS, reason="no GPU backend available")
def test_public_api_dispatches_to_backend():
    """The auto-dispatch path must produce the same numbers as the CPU."""
    import cyrcantile as ct

    rng = np.random.RandomState(24)
    lngs = rng.uniform(-180, 180, 1500)
    lats = rng.uniform(-85, 85, 1500)
    xs, ys = ct.tile(lngs, lats, 10)
    ex, ey = _cpu.tile_vec(lngs, lats, 10)
    name = ct.ACTIVE_BACKEND
    _assert_ints_close(name, xs, ex, max_off=1)
    _assert_ints_close(name, ys, ey, max_off=1)
    assert ct.ACTIVE_BACKEND == ("opencl" if ct._backend is not None else "cpu")
