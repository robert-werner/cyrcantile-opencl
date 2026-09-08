"""Head-to-head benchmark: cyrcantile vs mercantile vs supermercado.

Compares the three libraries on identical workloads:
  - xy        : lng/lat → Web Mercator
  - tile      : lng/lat → tile x/y
  - lnglat    : Web Mercator → lng/lat  (cyrcantile only — mercantile lacks it)
  - edge      : edge detection on a boolean grid

Run from repo root:
    python benchmarks/compare.py
    python benchmarks/compare.py --size 2000000 --repeat 5
    python benchmarks/compare.py --operation xy
"""

from __future__ import annotations

import argparse
import time

import numpy as np

# ── constants ──────────────────────────────────────────────────────

PI = np.pi
RE = 6378137.0
R2D = 180.0 / PI
D2R = PI / 180.0
HALF_PI = PI / 2.0
QPI = PI / 4.0


# ── helpers ────────────────────────────────────────────────────────

def banner(msg: str) -> None:
    print(f"\n{'─' * 64}")
    print(f"  {msg}")
    print(f"{'─' * 64}")


def timer(fn, repeat: int = 3, warmup: int = 1) -> float:
    """Return average time in ms."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t0) / repeat * 1000


# ── test data ──────────────────────────────────────────────────────

def make_points(n: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    lngs = rng.uniform(-180, 180, n).astype(np.float64)
    lats = rng.uniform(-85, 85, n).astype(np.float64)
    return lngs, lats


def make_merc(n: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    xs = rng.uniform(-20037508, 20037508, n).astype(np.float64)
    ys = rng.uniform(-20037508, 20037508, n).astype(np.float64)
    return xs, ys


def make_edge_grid(grid_side: int = 1024, seed: int = 42):
    rng = np.random.RandomState(seed)
    inner = rng.random((grid_side, grid_side)) < 0.3
    burn = np.zeros((grid_side + 2, grid_side + 2), dtype=np.uint8)
    burn[1:-1, 1:-1] = inner
    return burn, -512, -512, 12


# ── pure-NumPy baselines ──────────────────────────────────────────

def np_xy(lngs, lats):
    x = RE * lngs * D2R
    y = RE * np.log(np.tan(QPI + lats * D2R * 0.5))
    return x, y


def np_lnglat(xs, ys):
    lng = (xs / RE) * R2D
    lat = (2.0 * np.arctan(np.exp(ys / RE)) - HALF_PI) * R2D
    return lng, lat


def np_tile(lngs, lats, zoom):
    z2 = 2.0 ** zoom
    sinlat = np.clip(np.sin(lats * D2R), -0.999999999, 0.999999999)
    y01 = 0.5 - 0.25 * np.log((1.0 + sinlat) / (1.0 - sinlat)) / PI
    x01 = lngs / 360.0 + 0.5
    eps = 1e-14
    ox = np.clip(np.floor((x01 + eps) * z2), 0, int(z2) - 1).astype(np.int32)
    oy = np.clip(np.floor((y01 + eps) * z2), 0, int(z2) - 1).astype(np.int32)
    return ox, oy


def np_edge(burn, xmin, ymin, zoom):
    idxs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    rolled = [np.roll(np.roll(burn, d[0], 0), d[1], 1) for d in idxs]
    edge = np.min(np.dstack(rolled), axis=2) ^ burn
    edge[burn == 0] = False
    coords = np.dstack(np.where(edge))[0]
    if len(coords) == 0:
        return np.empty((0, 3), dtype=np.int32)
    coords[:, 0] += xmin - 1
    coords[:, 1] += ymin - 1
    return np.column_stack((coords, np.full(len(coords), zoom, dtype=np.uint8)))


# ── benchmarks ─────────────────────────────────────────────────────

def bench_xy(lngs, lats, repeat: int):
    import cyrcantile as ct
    import mercantile

    n = len(lngs)
    banner(f"xy  ({n:,} points)")

    results = {}

    # mercantile — scalar loop
    def _merc():
        for i in range(n):
            mercantile.xy(lngs[i], lats[i])
    t = timer(_merc, repeat)
    results["mercantile"] = t
    print(f"  mercantile     {t:10.2f} ms  (scalar loop)")

    # numpy vectorized
    t = timer(lambda: np_xy(lngs, lats), repeat)
    results["NumPy"] = t
    print(f"  NumPy          {t:10.2f} ms")

    # cyrcantile scalar loop
    def _ct_scalar():
        for i in range(n):
            ct.xy(lngs[i], lats[i])
    t = timer(_ct_scalar, repeat)
    results["cyrcantile(1)"] = t
    print(f"  cyrcantile(1)  {t:10.2f} ms  (scalar loop)")

    # cyrcantile batch (CPU or GPU)
    t = timer(lambda: ct.xy(lngs, lats), repeat)
    results["cyrcantile"] = t
    print(f"  cyrcantile     {t:10.2f} ms  (batch)")

    return results


def bench_tile(lngs, lats, zoom: int, repeat: int):
    import cyrcantile as ct
    import mercantile

    n = len(lngs)
    banner(f"tile  ({n:,} points, z={zoom})")

    results = {}

    # mercantile — scalar loop
    def _merc():
        for i in range(n):
            mercantile.tile(lngs[i], lats[i], zoom)
    t = timer(_merc, repeat)
    results["mercantile"] = t
    print(f"  mercantile     {t:10.2f} ms  (scalar loop)")

    # numpy vectorized
    t = timer(lambda: np_tile(lngs, lats, zoom), repeat)
    results["NumPy"] = t
    print(f"  NumPy          {t:10.2f} ms")

    # cyrcantile scalar loop
    def _ct_scalar():
        for i in range(n):
            ct.tile(lngs[i], lats[i], zoom)
    t = timer(_ct_scalar, repeat)
    results["cyrcantile(1)"] = t
    print(f"  cyrcantile(1)  {t:10.2f} ms  (scalar loop)")

    # cyrcantile batch
    t = timer(lambda: ct.tile(lngs, lats, zoom), repeat)
    results["cyrcantile"] = t
    print(f"  cyrcantile     {t:10.2f} ms  (batch)")

    return results


def bench_lnglat(xs, ys, repeat: int):
    import cyrcantile as ct

    n = len(xs)
    banner(f"lnglat  ({n:,} points)")

    results = {}

    # numpy vectorized (mercantile has no lnglat at all)
    t = timer(lambda: np_lnglat(xs, ys), repeat)
    results["NumPy"] = t
    print(f"  NumPy          {t:10.2f} ms")

    # cyrcantile scalar
    def _ct_scalar():
        for i in range(n):
            ct.lnglat(xs[i], ys[i])
    t = timer(_ct_scalar, repeat)
    results["cyrcantile(1)"] = t
    print(f"  cyrcantile(1)  {t:10.2f} ms  (scalar loop)")

    # cyrcantile batch
    t = timer(lambda: ct.lnglat(xs, ys), repeat)
    results["cyrcantile"] = t
    print(f"  cyrcantile     {t:10.2f} ms  (batch)")

    print("  mercantile     N/A  (no lnglat API)")

    return results


def bench_edge(burn, xmin, ymin, zoom, repeat: int):
    import supermercado.edge_finder as ef

    grid = burn.shape[0]
    banner(f"edge_stencil  ({grid}×{grid} grid)")

    results = {}

    # numpy 8-roll baseline
    t = timer(lambda: np_edge(burn, xmin, ymin, zoom), repeat)
    results["NumPy"] = t
    print(f"  NumPy          {t:10.2f} ms")

    # supermercado (original) — needs JSON string tiles
    tile_strings = []
    for r in range(burn.shape[0]):
        for c in range(burn.shape[1]):
            if burn[r, c]:
                tile_strings.append(f"[{r + xmin}, {c + ymin}, {zoom}]")
    try:
        t = timer(lambda: ef.findedges(tile_strings, False), repeat)
        results["supermercado"] = t
        print(f"  supermercado   {t:10.2f} ms")
    except Exception as e:
        print(f"  supermercado   (failed: {e})")

    # cyrcantile (Numba accelerated)
    try:
        from gigamercado._accel import edge_detect

        # warm up JIT
        edge_detect(burn, xmin, ymin, zoom)
        t = timer(lambda: edge_detect(burn, xmin, ymin, zoom), repeat)
        results["cyrcantile"] = t
        print(f"  cyrcantile     {t:10.2f} ms  (Numba parallel)")
    except Exception as e:
        print(f"  cyrcantile     (failed: {e})")

    return results


# ── summary ────────────────────────────────────────────────────────

def print_summary(all_results: dict):
    banner("SPEEDUP SUMMARY")
    backends = ["mercantile", "NumPy", "cyrcantile(1)", "cyrcantile", "supermercado"]
    header = f"{'Operation':<16}" + "".join(f"{b:>15}" for b in backends)
    print(header)
    print("─" * len(header))

    for op, res in all_results.items():
        # Use mercantile as baseline when present, otherwise NumPy
        base = res.get("mercantile", res.get("NumPy", 1.0))
        cols = ""
        for b in backends:
            if b in res:
                if b == "mercantile" and "mercantile" in res:
                    cols += f"{'1.00':>15}"
                else:
                    speedup = base / res[b]
                    cols += f"{speedup:>14.1f}x"
            else:
                cols += f"{'—':>15}"
        print(f"{op:<16}{cols}")

    banner("ABSOLUTE TIMES (ms)")
    header = f"{'Operation':<16}" + "".join(f"{b:>15}" for b in backends)
    print(header)
    print("─" * len(header))

    for op, res in all_results.items():
        cols = ""
        for b in backends:
            if b in res:
                cols += f"{res[b]:>14.1f}"
            else:
                cols += f"{'—':>15}"
        print(f"{op:<16}{cols}")


# ── main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="cyrcantile vs mercantile vs supermercado benchmark"
    )
    parser.add_argument("--size", type=int, default=500_000,
                        help="Points per benchmark (default 500 000)")
    parser.add_argument("--grid", type=int, default=1024,
                        help="Edge grid side (default 1024)")
    parser.add_argument("--repeat", type=int, default=3,
                        help="Repetitions (default 3)")
    parser.add_argument("--operation",
                        choices=["xy", "tile", "lnglat", "edge", "all"],
                        default="all")
    args = parser.parse_args()

    n, grid, repeat = args.size, args.grid, args.repeat

    banner("benchmark: cyrcantile vs mercantile vs supermercado")
    print(f"  points = {n:,}   grid = {grid}×{grid}   repeat = {repeat}")

    lngs, lats = make_points(n)
    xs, ys = make_merc(n)
    burn, xmin, ymin, zoom = make_edge_grid(grid)

    # Warm up cyrcantile JIT / GPU dispatch
    import cyrcantile as ct
    ct.xy(lngs[:4096], lats[:4096])
    ct.tile(lngs[:4096], lats[:4096], 12)
    ct.lnglat(xs[:4096], ys[:4096])

    ops = {
        "xy":     lambda: bench_xy(lngs, lats, repeat),
        "tile":   lambda: bench_tile(lngs, lats, 12, repeat),
        "lnglat": lambda: bench_lnglat(xs, ys, repeat),
        "edge":   lambda: bench_edge(burn, xmin, ymin, zoom, repeat),
    }

    all_results = {}
    if args.operation == "all":
        for name, fn in ops.items():
            all_results[name] = fn()
        print_summary(all_results)
    else:
        ops[args.operation]()

    print(f"\n  Done ✓\n")


if __name__ == "__main__":
    main()
