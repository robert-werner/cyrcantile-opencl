"""CUDA kernel parity via the numba CUDA simulator (cudasim).

The simulator executes the ``@cuda.jit`` kernels on the CPU, so the
CUDA backend can be verified on machines without an NVIDIA GPU — this
file would have caught both historical CUDA bugs (a stale ``_blocks``
reference and a missing degrees-to-radians conversion in ``xy``).

The env var must be set before numba is imported, so the check runs in
a subprocess.  When a real CUDA device exists, the same script
exercises the real backend instead of the simulator.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_SCRIPT = r"""
import numpy as np
from cyrcantile import _cpu, _cuda

bk = _cuda.get_backend()
if bk is None:
    # neither the simulator nor a real CUDA device is engaged
    print("CUDA-BACKEND-UNAVAILABLE")
    raise SystemExit(0)

rng = np.random.RandomState(7)
n = 512
zoom = 12
lng = rng.uniform(-179.9, 179.9, n)
lat = rng.uniform(-84.9, 84.9, n)
lim = 1 << zoom
tx = rng.randint(0, lim, n).astype(np.int32)
ty = rng.randint(0, lim, n).astype(np.int32)


def eq(actual, expected, what):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert np.array_equal(actual, expected), (
        f"{what}: mismatch, max diff "
        f"{np.abs(actual.astype(np.int64) - expected.astype(np.int64)).max()}")


def close(actual, expected, what):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    diff = np.abs(actual - expected)
    assert np.allclose(actual, expected, rtol=1e-9, atol=1e-6), (
        f"{what}: mismatch, max diff {diff.max()}")


xs, ys = bk.tile(lng, lat, zoom)
exs, eys = _cpu.tile_vec(lng, lat, zoom)
eq(xs, exs, "tile.x"); eq(ys, eys, "tile.y")

xs, ys = bk.tile(lng, lat, zoom, truncate=True)
exs, eys = _cpu.tile_vec(lng, lat, zoom, truncate=True)
eq(xs, exs, "tile(truncate).x"); eq(ys, eys, "tile(truncate).y")

gx, gy = bk.xy(lng, lat)
ex, ey = _cpu.xy_vec(lng, lat)
close(gx, ex, "xy.x"); close(gy, ey, "xy.y")

glm, gla = bk.lnglat(ex, ey)
elm, ela = _cpu.lnglat_vec(ex, ey)
close(glm, elm, "lnglat.lng"); close(gla, ela, "lnglat.lat")

gul, gut = bk.ul(tx, ty, zoom)
eul, eut = _cpu.ul_vec(tx, ty, zoom)
close(gul, eul, "ul.lng"); close(gut, eut, "ul.lat")

g = bk.bounds(tx, ty, zoom)
e = _cpu.bounds_vec(tx, ty, zoom)
for i, name in enumerate("wsen"):
    close(g[i], e[i], f"bounds.{name}")

g = bk.xy_bounds(tx, ty, zoom)
e = _cpu.xy_bounds_vec(tx, ty, zoom)
for i, name in enumerate("lbrt"):
    close(g[i], e[i], f"xy_bounds.{name}")

qk = bk.quadkey_encode(tx, ty, zoom)
eq(qk, _cpu.quadkey_encode_vec(tx, ty, zoom), "quadkey_encode")
dx, dy = bk.quadkey_decode(qk, zoom)
eq(dx, tx, "quadkey_decode.x"); eq(dy, ty, "quadkey_decode.y")

for shift in (0, 1, 3):
    px, py = bk.parent(tx, ty, shift)
    eq(px, tx >> shift, f"parent({shift}).x")
    eq(py, ty >> shift, f"parent({shift}).y")

cx, cy = bk.children(tx, ty)
ecx, ecy = _cpu.children_vec(tx, ty)
eq(cx, ecx, "children.x"); eq(cy, ecy, "children.y")

nx, ny = bk.neighbors(tx, ty)
enx, eny = _cpu.neighbors_vec(tx, ty)
eq(nx, enx, "neighbors.x"); eq(ny, eny, "neighbors.y")

west = rng.uniform(-179.0, 170.0, n)
north = rng.uniform(-84.0, 84.0, n)
bx, by = bk.bounding_tile(west, north, zoom)
ebx, eby = _cpu.tile_vec(west, north, zoom)
eq(bx, ebx, "bounding_tile.x"); eq(by, eby, "bounding_tile.y")

ox, oy = bk.tiles_in_bbox(100, 200, 17, 9)
idx = np.arange(17 * 9)
eq(ox, 100 + idx % 17, "tiles_in_bbox.x")
eq(oy, 200 + idx // 17, "tiles_in_bbox.y")

print("CUDASIM-OK")
"""


def _has_numba() -> bool:
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _has_numba(), reason="numba not installed")
def test_cuda_kernel_parity_via_simulator():
    env = dict(os.environ)
    env["NUMBA_ENABLE_CUDASIM"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = result.stdout + result.stderr
    if "CUDA-BACKEND-UNAVAILABLE" in result.stdout:
        pytest.skip("cudasim not supported by this numba and no CUDA device")
    assert result.returncode == 0, output
    assert "CUDASIM-OK" in result.stdout, output
