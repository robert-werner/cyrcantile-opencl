"""Batch benchmark: millions of coordinates, millions of tiles.

Times every batch operation on all available backends (OpenCL, Vulkan,
CUDA, multicore CPU, single-thread CPU), verifies the GPU results
against the CPU reference, and finishes with end-to-end public-API
timings (quadkey strings, string decoding, tile enumeration).

Usage:
    python benchmarks/batch.py
    python benchmarks/batch.py --coords 10_000_000 --tiles 10_000_000
    python benchmarks/batch.py --backends opencl,cpu --repeat 5
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import math
import time
import warnings

import numpy as np

import cyrcantile as ct
from cyrcantile import _cpu

# AMD/POCL OpenCL compilers emit (benign) remarks during kernel build;
# the build itself is verified below, so keep the output clean.
try:
    import pyopencl

    warnings.filterwarnings("ignore", category=pyopencl.CompilerWarning)
except Exception:  # pragma: no cover - pyopencl optional
    pass

# Relative tolerances for result verification, per storage precision.
# The absolute tolerance is scaled by each output array's magnitude:
# rounding differences (FMA contraction, f32 storage) are bounded by the
# ulp of the *intermediates*, which can be large even when the output
# value itself is near zero (e.g. bottom = top - tile_size).
_F64_RTOL = 1e-9
_F32_RTOL = 2e-4
_F64_ATOL_SCALE = 1e-9   # ~4 ulps at f64 for the array's magnitude
_F32_ATOL_SCALE = 2e-6   # ~16 ulps at f32 (plus transcendental error)


# ------------------------------------------------------------------ #
#  Backends
# ------------------------------------------------------------------ #

@contextlib.contextmanager
def _cpu_single():
    """Force the vectorised CPU paths onto a single thread."""
    saved = (_cpu.HAS_NUMBA, _cpu._CPU_WORKERS)
    _cpu.HAS_NUMBA = False
    _cpu._CPU_WORKERS = 1
    try:
        yield
    finally:
        _cpu.HAS_NUMBA, _cpu._CPU_WORKERS = saved


@contextlib.contextmanager
def _no_autodispatch():
    """Temporarily disable GPU auto-dispatch in the public API."""
    saved = ct._backend
    ct._backend = None
    try:
        yield
    finally:
        ct._backend = saved


class CpuAdapter:
    """Expose the CPU vector functions behind the GPU backend API."""

    def __init__(self, single=False):
        self.single = single
        self.precision = "f64"
        self.name = "cpu-1" if single else "cpu"

    def _call(self, fn, *args):
        if self.single:
            with _cpu_single():
                return fn(*args)
        return fn(*args)

    def tile(self, lng, lat, zoom):
        return self._call(_cpu.tile_vec, lng, lat, zoom, False)

    def xy(self, lng, lat):
        return self._call(_cpu.xy_vec, lng, lat, False)

    def lnglat(self, x, y):
        return self._call(_cpu.lnglat_vec, x, y)

    def ul(self, tx, ty, zoom):
        return self._call(_cpu.ul_vec, tx, ty, zoom)

    def bounds(self, tx, ty, zoom):
        return self._call(_cpu.bounds_vec, tx, ty, zoom)

    def xy_bounds(self, tx, ty, zoom):
        return self._call(_cpu.xy_bounds_vec, tx, ty, zoom)

    def quadkey_encode(self, tx, ty, zoom):
        return self._call(_cpu.quadkey_encode_vec, tx, ty, zoom)

    def quadkey_decode(self, qk, zoom):
        return self._call(_cpu.quadkey_decode_vec, qk, zoom)

    def parent(self, tx, ty, shift):
        return self._call(_cpu.parent_vec, tx, ty, shift)

    def children(self, tx, ty):
        return self._call(_cpu.children_vec, tx, ty)

    def neighbors(self, tx, ty):
        return self._call(_cpu.neighbors_vec, tx, ty)

    def bounding_tile(self, west, north, zoom):
        return self._call(_cpu.tile_vec, west, north, zoom)


def collect_backends(wanted):
    """Build the (name, backend, precision) list for the benchmark."""
    backends = []
    for name, module in (
        ("opencl", "cyrcantile._backend"),
        ("vulkan", "cyrcantile._vulkan"),
        ("cuda", "cyrcantile._cuda"),
    ):
        if name not in wanted:
            continue
        try:
            mod = importlib.import_module(module)
            backend = mod.get_backend()
        except Exception:
            backend = None
        if backend is not None:
            backends.append((name, backend, "f32" if name == "vulkan" else "f64"))
        else:
            print(f"# {name}: unavailable - skipped")
    if "cpu" in wanted:
        cpu = CpuAdapter()
        backends.append((cpu.name, cpu, "f64"))
    if "cpu-1" in wanted:
        cpu1 = CpuAdapter(single=True)
        backends.append((cpu1.name, cpu1, "f64"))
    return backends


# ------------------------------------------------------------------ #
#  Timing & verification
# ------------------------------------------------------------------ #

def best_of(fn, repeat):
    """Warm up once, then return (best ms, last output)."""
    fn()
    best = math.inf
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0, out


def _as_tuple(x):
    return x if isinstance(x, tuple) else (x,)


def _float_tol(precision, expected):
    """(rtol, atol) for an output array, with a magnitude-scaled atol."""
    e = np.asarray(expected)
    scale = max(1.0, float(np.abs(e).max(initial=0.0)))
    if precision == "f32":
        return _F32_RTOL, _F32_ATOL_SCALE * scale
    return _F64_RTOL, _F64_ATOL_SCALE * scale


def check_match(precision, got, expected):
    """Return None when *got* matches *expected*, else a description."""
    for g, e in zip(_as_tuple(got), _as_tuple(expected)):
        if g is e:
            continue  # same object (CPU reference itself)
        g = np.asarray(g)
        e = np.asarray(e)
        if np.issubdtype(g.dtype, np.floating):
            rtol, atol = _float_tol(precision, e)
            if not np.allclose(g, e, rtol=rtol, atol=atol):
                diff = np.abs(g - e)
                return (f"float mismatch: max |diff| {diff.max():.3g} "
                        f"vs rtol {rtol:g} / atol {atol:.3g} (scale-scaled)")
        elif precision == "f32":
            # tile indices may flip by one at f32 floor boundaries
            diff = np.abs(g.astype(np.int64) - e.astype(np.int64))
            if diff.max(initial=0) > 1:
                return f"index differs by {int(diff.max())} (allowed: 1)"
        elif not np.array_equal(g, e):
            bad = int(np.count_nonzero(g != e))
            return f"int mismatch: {bad} of {g.size} elements differ"
    return None


def fmt(ms):
    if ms is None:
        return "-"
    if ms >= 1000.0:
        return f"{ms / 1000.0:.2f} s"
    return f"{ms:.1f} ms"


# ------------------------------------------------------------------ #
#  Data & operations
# ------------------------------------------------------------------ #

class Data:
    pass


def make_data(ncoords, ntiles, zoom):
    rng = np.random.RandomState(42)
    d = Data()
    d.zoom = zoom
    d.lng = rng.uniform(-179.9, 179.9, ncoords)
    d.lat = rng.uniform(-84.9, 84.9, ncoords)
    d.xm = rng.uniform(-2.0e7, 2.0e7, ncoords)
    d.ym = rng.uniform(-2.0e7, 2.0e7, ncoords)
    lim = 1 << zoom
    d.tx = rng.randint(0, lim, ntiles).astype(np.int32)
    d.ty = rng.randint(0, lim, ntiles).astype(np.int32)
    d.qk = _cpu.quadkey_encode_vec(d.tx, d.ty, zoom)
    d.west = rng.uniform(-179.0, 170.0, ncoords)
    d.north = rng.uniform(-84.0, 84.0, ncoords)
    return d


def build_ops(d):
    """(label, group, callable(backend)) for every batch operation."""
    zoom = d.zoom
    return [
        ("tile", "coords", lambda be: be.tile(d.lng, d.lat, zoom)),
        ("xy", "coords", lambda be: be.xy(d.lng, d.lat)),
        ("lnglat", "coords", lambda be: be.lnglat(d.xm, d.ym)),
        ("bounding_tile", "coords", lambda be: be.bounding_tile(d.west, d.north, zoom)),
        ("ul", "tiles", lambda be: be.ul(d.tx, d.ty, zoom)),
        ("bounds", "tiles", lambda be: be.bounds(d.tx, d.ty, zoom)),
        ("xy_bounds", "tiles", lambda be: be.xy_bounds(d.tx, d.ty, zoom)),
        ("quadkey_encode", "tiles", lambda be: be.quadkey_encode(d.tx, d.ty, zoom)),
        ("quadkey_decode", "tiles", lambda be: be.quadkey_decode(d.qk, zoom)),
        ("parent", "tiles", lambda be: be.parent(d.tx, d.ty, 3)),
        ("children", "tiles", lambda be: be.children(d.tx, d.ty)),
        ("neighbors", "tiles", lambda be: be.neighbors(d.tx, d.ty)),
    ]


# ------------------------------------------------------------------ #
#  Main matrix
# ------------------------------------------------------------------ #

def run_matrix(ops, backends, repeat):
    results = {}
    failures = []
    notes = []
    reference = CpuAdapter()  # multicore CPU, not part of the timing columns
    for label, group, fn in ops:
        expected = fn(reference)
        row = {}
        for name, backend, precision in backends:
            try:
                ms, out = best_of(lambda fn=fn, backend=backend: fn(backend), repeat)
                row[name] = ms
                msg = check_match(precision, out, expected)
                if msg:
                    failures.append(f"{label} / {name}: {msg}")
                del out
            except NotImplementedError as e:
                row[name] = None
                notes.append(f"{name} / {label}: {e}")
            except Exception as e:  # noqa: BLE001 - report and continue
                row[name] = None
                failures.append(f"{label} / {name}: {type(e).__name__}: {e}")
        results[label] = (group, row)
        del expected
    return results, failures, notes


def print_matrix(results, names):
    header = f"{'operation':<16}" + "".join(f"{name:>12}" for name in names)
    print("\n" + "-" * len(header))
    print("  BATCH OPERATIONS (best of N runs)")
    print("-" * len(header))
    print(header)
    current_group = None
    for label, (group, row) in results.items():
        if group != current_group:
            current_group = group
            title = "coordinate batches" if group == "coords" else "tile batches"
            print(f"-- {title} " + "-" * (len(header) - 17 - len(title)))
        cells = "".join(f"{fmt(row.get(name)):>12}" for name in names)
        print(f"{label:<16}{cells}")
    print("-" * len(header))


def print_speedups(results, names, repeat_note):
    gpu_names = [n for n in names if n not in ("cpu", "cpu-1")]
    base_name = "cpu-1" if "cpu-1" in names else ("cpu" if "cpu" in names else None)
    if not gpu_names or base_name is None:
        return
    print("\n  GPU vs single-thread CPU")
    print(f"  {'operation':<16}{'best GPU':<24}{'cpu-1':<12}{'speedup':>10}")
    for label, (_, row) in results.items():
        gpu = {n: row[n] for n in gpu_names if row.get(n) is not None}
        base = row.get(base_name)
        if not gpu or base is None:
            continue
        best_name = min(gpu, key=gpu.get)
        speedup = base / gpu[best_name]
        print(f"  {label:<16}{best_name + ' ' + fmt(gpu[best_name]):<24}"
              f"{fmt(base):<12}{speedup:>9.1f}x")


# ------------------------------------------------------------------ #
#  End-to-end public API
# ------------------------------------------------------------------ #

def bbox_for_tiles(m, zoom=12):
    """A geographic bbox covered by roughly *m* tiles at *zoom*."""
    world = 1 << (2 * zoom)  # tiles covering the whole world
    side = min(1.0, math.sqrt(m / world))
    w, s = -170.0, -80.0
    return w, s, w + side * 339.0, s + side * 159.0


def run_public_api(d, n_strings, n_tiles_enum, repeat):
    zoom = d.zoom
    cap = min(len(d.tx), n_strings)
    tx, ty = d.tx[:cap], d.ty[:cap]
    print("\n  END-TO-END PUBLIC API (auto-dispatched batch + Python glue)")
    print("-" * 70)

    # quadkey: GPU-packed encode + vectorised string conversion
    gpu_ms, _ = best_of(lambda: ct.quadkey(tx, ty, zoom), repeat)
    cpu_ms, keys = best_of(
        lambda: _cpu.quadkey_ints_to_strings(
            _cpu.quadkey_encode_vec(tx, ty, zoom), zoom), repeat)
    print(f"  quadkey -> strings ({cap:,} tiles)  "
          f"gpu {fmt(gpu_ms):>10}   cpu {fmt(cpu_ms):>10}")

    # quadkey string decoding (GPU decode + Tile list construction)
    dec_ms, tiles = best_of(lambda: ct.quadkey_to_tile(keys), repeat)
    print(f"  strings -> tiles  ({cap:,} keys)   "
          f"gpu {fmt(dec_ms):>10}   (Tile list dominates)")
    del tiles

    # tiles(): GPU grid fill vs CPU nested loops
    w, s, e, n = bbox_for_tiles(n_tiles_enum)
    gpu_ms, tiles = best_of(lambda: ct.tiles(w, s, e, n, [12]), repeat)
    with _no_autodispatch():
        cpu_ms, tiles_cpu = best_of(lambda: ct.tiles(w, s, e, n, [12]), 1)
    ok = len(tiles) == len(tiles_cpu) and tiles[0] == tiles_cpu[0] \
        and tiles[-1] == tiles_cpu[-1]
    del tiles, tiles_cpu
    print(f"  tiles() enumerate ({len(ct.tiles(w, s, e, n, [12])):,} tiles)  "
          f"gpu {fmt(gpu_ms):>10}   cpu loops {fmt(cpu_ms):>10}   match: {ok}")


# ------------------------------------------------------------------ #
#  Entry point
# ------------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser(
        description="cyrcantile batch benchmark (millions of coords & tiles)")
    parser.add_argument("--coords", type=int, default=2_000_000,
                        help="coordinate batch size (default 2,000,000)")
    parser.add_argument("--tiles", type=int, default=2_000_000,
                        help="tile batch size (default 2,000,000)")
    parser.add_argument("--zoom", type=int, default=14,
                        help="zoom level for the batches (default 14)")
    parser.add_argument("--repeat", type=int, default=3,
                        help="timed repetitions, best is kept (default 3)")
    parser.add_argument("--backends", default="opencl,vulkan,cuda,cpu,cpu-1",
                        help="comma-separated backends to benchmark")
    parser.add_argument("--no-public-api", action="store_true",
                        help="skip the end-to-end public API section")
    return parser.parse_args()


REQUIRED_PUBLIC_API = ("quadkey", "quadkey_to_tile", "parent", "children",
                        "neighbors", "bounding_tile", "tiles", "feature")


def _describe_module(mod):
    """Human-readable import location, also for namespace packages."""
    origin = getattr(mod, "__file__", None)
    if origin:
        return origin
    spec = getattr(mod, "__spec__", None)
    locs = list(getattr(spec, "submodule_search_locations", None) or [])
    if locs:
        return f"namespace package at {locs[0]!r} (no __init__.py!)"
    return "unknown location (namespace package, no __init__.py)"


def _check_environment():
    """Fail fast when cyrcantile resolves to a stale or partial import."""
    version = getattr(ct, "__version__", "unknown")
    location = _describe_module(ct)
    print(f"  cyrcantile {version} from {location}")
    missing = [name for name in REQUIRED_PUBLIC_API if not hasattr(ct, name)]
    if missing:
        raise SystemExit(
            f"error: the imported cyrcantile is incomplete:\n"
            f"  location: {location}\n"
            f"  version:  {version}\n"
            f"  missing public API: {', '.join(missing)}\n"
            "\n"
            "A namespace package (no __init__.py) or a stale/partially\n"
            "synced checkout produces exactly this: the batch backends\n"
            "still import, but the top-level API does not exist.\n"
            "\n"
            "Fix: re-sync the repository (cyrcantile/__init__.py in\n"
            "particular) and run:  pip install -e .\n"
            'Check: python -c "import cyrcantile; '
            'print(cyrcantile.__version__, cyrcantile.__file__)"')


def main():
    args = parse_args()
    if args.coords <= 0 or args.tiles <= 0:
        raise SystemExit("--coords and --tiles must be positive")
    wanted = {b.strip().lower() for b in args.backends.split(",") if b.strip()}

    print("=" * 70)
    print("  cyrcantile batch benchmark")
    _check_environment()
    print(f"  coords: {args.coords:,}   tiles: {args.tiles:,}   "
          f"zoom: {args.zoom}   repeat: {args.repeat}")
    print("=" * 70)

    backends = collect_backends(wanted)
    if not backends:
        raise SystemExit("no backends selected/available")
    names = [name for name, _, _ in backends]

    data = make_data(args.coords, args.tiles, args.zoom)
    ops = build_ops(data)

    results, failures, notes = run_matrix(ops, backends, args.repeat)
    print_matrix(results, names)
    print_speedups(results, names, args.repeat)

    for note in notes:
        print(f"  note: {note}")
    if failures:
        print("\n  VERIFICATION FAILURES:")
        for failure in failures:
            print(f"  X {failure}")
        raise SystemExit(1)
    print("\n  verification: all backends match the CPU reference "
          "(f64 exact; f32 within tolerance, tile indices may differ by 1)")

    if not args.no_public_api:
        run_public_api(data, n_strings=min(args.tiles, 1_000_000),
                       n_tiles_enum=min(args.tiles, 4_000_000), repeat=args.repeat)

    print("\nDone.")


if __name__ == "__main__":
    main()
