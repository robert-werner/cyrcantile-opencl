"""Demo & micro-benchmark for cyrcantile backends.

Usage:
    python main.py
    python main.py --size 100_000_000 --zoom 12
    python main.py --backends opencl,vulkan,cuda --backends-cpu false
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import cyrcantile as ct
from cyrcantile import _cpu


def banner(msg):
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def demo_single():
    banner("SINGLE-VALUE API (CPU path)")
    t = ct.tile(-9.14, 53.12, 7)
    print(f"  tile(-9.14, 53.12, 7)        = {t}")

    b = ct.bounds(t)
    print(f"  bounds(t)                      = {b}")

    xy = ct.xy(-9.14, 53.12)
    print(f"  xy(-9.14, 53.12)               = {xy}")

    ll = ct.lnglat(*xy)
    print(f"  lnglat({xy[0]:.1f}, {xy[1]:.1f})   = {ll}")

    qk = ct.quadkey(t)
    print(f"  quadkey(t)                     = {qk}")

    t2 = ct.quadkey_to_tile(qk)
    print(f"  quadkey_to_tile(qk)            = {t2}")

    p = ct.parent(t)
    print(f"  parent(t)                      = {p}")

    ch = ct.children(t)
    print(f"  children(t)                    = {ch}")

    nb = ct.neighbors(t)
    print(f"  neighbors(t)                   = {len(nb)} tiles")

    bt = ct.bounding_tile(-9.5, 53.0, -9.0, 53.3)
    print(f"  bounding_tile(bbox)            = {bt}")


def make_data(n, zoom):
    np.random.seed(42)
    lons = np.random.uniform(-180, 180, n)
    lats = np.random.uniform(-85, 85, n)
    return lons, lats, zoom


def bench_backend(name, run, lons, lats, zoom):
    """Run one GPU batch benchmark; returns (ms, xs, ys) or (None, None, None)."""
    if run is None:
        print(f"  {name} available: False - skipping")
        return None, None, None
    try:
        # warmup (also compiles kernels / pipelines)
        run(lons[:4096], lats[:4096], zoom)
        t0 = time.perf_counter()
        xs, ys = run(lons, lats, zoom)
        gpu_ms = (time.perf_counter() - t0) * 1000
        n = lons.shape[0]
        print(f"  {name} tile({n:,} pts, z={zoom}) = {gpu_ms:.1f} ms")
        return gpu_ms, xs, ys
    except Exception as e:
        print(f"  ({name} benchmark failed: {e})")
        return None, None, None


def cpu_reference(lons, lats, zoom, label="CPU"):
    # warm up the numba JIT (needs >= _THREAD_THRESHOLD elements to
    # select the JIT engine) so the timed call excludes compilation
    nwarm = min(len(lons), 250_000)
    _cpu.tile_vec(lons[:nwarm], lats[:nwarm], zoom)
    t0 = time.perf_counter()
    xs_c, ys_c = _cpu.tile_vec(lons, lats, zoom)
    cpu_ms = (time.perf_counter() - t0) * 1000
    print(f"  {label} tile_vec({lons.shape[0]:,} pts, z={zoom})  = {cpu_ms:.1f} ms")
    return xs_c, ys_c


def demo_batch(lons, lats, zoom, backends):
    banner("BATCH API (auto-dispatch)")
    print(f"  active backend: {ct.ACTIVE_BACKEND}   OpenCL available: {ct.HAS_OPENCL}")

    gpu_ms = xs = ys = None
    if "opencl" in backends and ct._backend is not None:
        gpu_ms, xs, ys = bench_backend("OpenCL", ct._backend.tile, lons, lats, zoom)

    xs_c, ys_c = cpu_reference(lons, lats, zoom)
    if xs is not None:
        match = np.array_equal(xs, xs_c) and np.array_equal(ys, ys_c)
        print(f"  OpenCL == CPU results: {match}")


def demo_batch_vulkan(lons, lats, zoom, backends):
    if "vulkan" not in backends:
        return
    try:
        from cyrcantile._vulkan import HAS_VULKAN
        from cyrcantile._vulkan import get_backend as vk_get_backend
    except Exception:
        HAS_VULKAN, vk_get_backend = False, None

    banner("BATCH API - Vulkan backend")
    print(f"  Vulkan available: {HAS_VULKAN}")
    backend = vk_get_backend() if (HAS_VULKAN and vk_get_backend) else None
    run = backend.tile if backend is not None else None
    gpu_ms, xs, ys = bench_backend("Vulkan", run, lons, lats, zoom)

    xs_c, ys_c = cpu_reference(lons, lats, zoom)
    if xs is not None:
        # Vulkan computes in f32: tile indices may flip at floor boundaries.
        mism = int(np.count_nonzero(xs != xs_c) + np.count_nonzero(ys != ys_c))
        print(f"  Vulkan vs CPU: {mism}/{2 * len(xs)} indices differ (f32 boundaries)")


def demo_batch_cuda(lons, lats, zoom, backends):
    if "cuda" not in backends:
        return
    try:
        from cyrcantile._cuda import HAS_CUDA
        from cyrcantile._cuda import get_backend as cuda_get_backend
    except Exception:
        HAS_CUDA, cuda_get_backend = False, None

    banner("BATCH API - CUDA backend")
    print(f"  CUDA available: {HAS_CUDA}")
    backend = cuda_get_backend() if (HAS_CUDA and cuda_get_backend) else None
    run = backend.tile if backend is not None else None
    gpu_ms, xs, ys = bench_backend("CUDA", run, lons, lats, zoom)

    xs_c, ys_c = cpu_reference(lons, lats, zoom)
    if xs is not None:
        match = np.array_equal(xs, xs_c) and np.array_equal(ys, ys_c)
        print(f"  CUDA == CPU results: {match}")


def demo_cpu_parallel(lons, lats, zoom, enabled):
    if not enabled:
        return
    banner("CPU PARALLEL BENCHMARK")
    print(f"  Numba available: {ct.HAS_NUMBA}   CPU workers: {_cpu._CPU_WORKERS}")

    # warmup (compiles numba kernels / first pass)
    _cpu.tile_vec(lons[:4096], lats[:4096], zoom)

    engine = _cpu._choose_engine(lons.shape[0])
    t0 = time.perf_counter()
    xs_c, ys_c = _cpu.tile_vec(lons, lats, zoom)
    cpu_ms = (time.perf_counter() - t0) * 1000
    print(f"  CPU tile_vec({lons.shape[0]:,} pts, z={zoom})  = {cpu_ms:.1f} ms  (engine: {engine})")

    # single-thread reference for the speedup ratio
    t0 = time.perf_counter()
    _cpu._tile_chunk(lons, lats, zoom, False, 0, lons.shape[0])
    single_ms = (time.perf_counter() - t0) * 1000
    print(f"  CPU single-thread              = {single_ms:.1f} ms")
    print(f"  parallel speedup               = {single_ms / cpu_ms:.2f}x")


def demo_tiles_in_bbox():
    banner("TILES IN BBOX")
    west, south, east, north = -9.5, 53.0, -9.0, 53.3
    result = ct.tiles(west, south, east, north, [14, 15, 16])
    print(f"  bbox({west}, {south}, {east}, {north})")
    print(f"  zooms [14, 15, 16] -> {len(result)} tiles")
    if result:
        print(f"  first: {result[0]}  last: {result[-1]}")


def demo_feature():
    banner("GEOJSON FEATURE")
    t = ct.tile(-9.14, 53.12, 10)
    feat = ct.feature(t, fid=1, props={"name": "demo"})
    print(json.dumps(feat, indent=2)[:400] + " ...")


def main():
    parser = argparse.ArgumentParser(description="cyrcantile demo & benchmark")
    parser.add_argument("--size", type=int, default=8_000_000,
                        help="points per batch benchmark (default 8,000,000)")
    parser.add_argument("--zoom", type=int, default=12, help="zoom level (default 12)")
    parser.add_argument("--backends", default="opencl,vulkan,cuda",
                        help="comma-separated GPU backends to benchmark")
    parser.add_argument("--cpu-parallel", dest="cpu_parallel",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="also run the CPU parallel benchmark")
    args = parser.parse_args()

    demo_single()

    lons, lats, zoom = make_data(args.size, args.zoom)
    backends = [b.strip().lower() for b in args.backends.split(",") if b.strip()]

    demo_batch(lons, lats, zoom, backends)
    demo_batch_vulkan(lons, lats, zoom, backends)
    demo_batch_cuda(lons, lats, zoom, backends)
    demo_cpu_parallel(lons, lats, zoom, args.cpu_parallel)
    demo_tiles_in_bbox()
    demo_feature()
    print("\nDone.")


if __name__ == "__main__":
    main()
