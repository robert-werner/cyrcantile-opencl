"""Demo & micro-benchmark for cyrcantile OpenCL backend."""

import time
import numpy as np
import cyrcantile as ct


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


def demo_batch():
    banner("BATCH API (GPU path)")
    print(f"  OpenCL available: {ct.HAS_OPENCL}")

    np.random.seed(42)
    n = 100_000
    lons = np.random.uniform(-180, 180, n)
    lats = np.random.uniform(-85, 85, n)
    zoom = 12

    if ct.HAS_OPENCL:
        t0 = time.perf_counter()
        xs, ys = ct.tile(lons, lats, zoom)
        gpu_ms = (time.perf_counter() - t0) * 1000
        print(f"  GPU tile_batch({n:,} pts, z={zoom})  = {gpu_ms:.1f} ms")
    else:
        print("  (OpenCL not available — skipping GPU benchmark)")

    from cyrcantile import _cpu
    t0 = time.perf_counter()
    xs_c, ys_c = _cpu.tile_vec(lons, lats, zoom)
    cpu_ms = (time.perf_counter() - t0) * 1000
    print(f"  CPU tile_vec({n:,} pts, z={zoom})  = {cpu_ms:.1f} ms")

    if ct.HAS_OPENCL:
        match = np.array_equal(xs, xs_c) and np.array_equal(ys, ys_c)
        print(f"  GPU == CPU results: {match}")


def demo_batch_vulkan():
    banner("BATCH API — Vulkan backend")
    try:
        from cyrcantile._vulkan import get_backend as vk_get_backend, HAS_VULKAN
    except Exception:
        HAS_VULKAN = False
        vk_get_backend = None
    print(f"  Vulkan available: {HAS_VULKAN}")

    np.random.seed(42)
    # 8M points: large enough to be a meaningful batch benchmark while each
    # f64 input buffer (64 MB) stays under the typical per-binding limit of
    # 128 MB (max_storage_buffer_binding_size) seen on llvmpipe.
    n = 8_000_000
    lons = np.random.uniform(-180, 180, n)
    lats = np.random.uniform(-85, 85, n)
    zoom = 12

    if HAS_VULKAN and vk_get_backend is not None:
        try:
            vk = vk_get_backend()
            # warmup (also compiles the pipeline)
            vk.tile(lons[:4096], lats[:4096], zoom)
            t0 = time.perf_counter()
            xs, ys = vk.tile(lons, lats, zoom)
            gpu_ms = (time.perf_counter() - t0) * 1000
            print(f"  Vulkan tile({n:,} pts, z={zoom}) = {gpu_ms:.1f} ms")
        except Exception as e:
            print(f"  (Vulkan benchmark failed: {e})")
            gpu_ms = xs = ys = None
    else:
        print("  (Vulkan not available — skipping Vulkan benchmark)")
        gpu_ms = xs = ys = None

    from cyrcantile import _cpu
    t0 = time.perf_counter()
    xs_c, ys_c = _cpu.tile_vec(lons, lats, zoom)
    cpu_ms = (time.perf_counter() - t0) * 1000
    print(f"  CPU tile_vec({n:,} pts, z={zoom})  = {cpu_ms:.1f} ms")

    if xs is not None:
        match = np.array_equal(xs, xs_c) and np.array_equal(ys, ys_c)
        print(f"  Vulkan == CPU results: {match}")


def demo_cpu_parallel():
    banner("CPU PARALLEL BENCHMARK")
    from cyrcantile import _cpu
    print(f"  Numba available: {ct.HAS_NUMBA}   CPU workers: {_cpu._CPU_WORKERS}")

    np.random.seed(42)
    n = 8_000_000
    lons = np.random.uniform(-180, 180, n)
    lats = np.random.uniform(-85, 85, n)
    zoom = 12

    # warmup (compiles numba kernels / first pass)
    _cpu.tile_vec(lons[:4096], lats[:4096], zoom)

    engine = _cpu._choose_engine(n)
    t0 = time.perf_counter()
    xs_c, ys_c = _cpu.tile_vec(lons, lats, zoom)
    cpu_ms = (time.perf_counter() - t0) * 1000
    print(f"  CPU tile_vec({n:,} pts, z={zoom})  = {cpu_ms:.1f} ms  (engine: {engine})")

    # single-thread reference for the speedup ratio
    t0 = time.perf_counter()
    _cpu._tile_chunk(lons, lats, zoom, False, 0, n)
    single_ms = (time.perf_counter() - t0) * 1000
    print(f"  CPU single-thread              = {single_ms:.1f} ms")
    print(f"  parallel speedup               = {single_ms / cpu_ms:.2f}x")


def demo_tiles_in_bbox():
    banner("TILES IN BBOX")
    west, south, east, north = -9.5, 53.0, -9.0, 53.3
    result = ct.tiles(west, south, east, north, [14, 15, 16])
    print(f"  bbox({west}, {south}, {east}, {north})")
    print(f"  zooms [14, 15, 16] → {len(result)} tiles")
    if result:
        print(f"  first: {result[0]}  last: {result[-1]}")


def demo_feature():
    banner("GEOJSON FEATURE")
    t = ct.tile(-9.14, 53.12, 10)
    feat = ct.feature(t, fid=1, props={"name": "demo"})
    import json
    print(json.dumps(feat, indent=2)[:400] + " ...")


if __name__ == "__main__":
    demo_single()
    demo_batch()
    demo_batch_vulkan()
    demo_cpu_parallel()
    demo_tiles_in_bbox()
    demo_feature()
    print("\nDone.")
