# cyrcantile — OpenCL / Vulkan / CUDA

GPU-accelerated spherical-mercator tile utilities — an OpenCL, Vulkan and
CUDA kernel rewrite of the original [cyrcantile](https://github.com/robert-werner/cyrcantile)
Cython project.

## What it does

Provides the same API as [mercantile](https://github.com/mapbox/mercantile)
1.2.x for converting between geographic coordinates (WGS-84), Web Mercator
(EPSG:3857), and XYZ tile indices — but with the compute-heavy batch
operations offloaded to GPU kernels (OpenCL, Vulkan, or CUDA).

## Installation

```bash
pip install .                    # core (NumPy only, CPU fallback)
pip install .[opencl]            # + OpenCL   (pyopencl; Linux may need ocl-icd-opencl-dev)
pip install .[vulkan]            # + Vulkan   (wgpu)
pip install .[cuda]              # + CUDA     (numba + NVIDIA GPU/driver)
pip install .[cpu-speed]         # + multicore CPU (numba)
pip install .[dev]               # tests + lint + mercantile oracle
```

`requirements.txt` holds only the dev/test requirements; pick the backend
extras you need.

## Architecture

```
cyrcantile/
├── __init__.py       # Public API — auto-dispatches CPU / GPU
├── _kernels.cl       # OpenCL C kernels (13 batch operations)
├── _backend.py       # PyOpenCL context manager, kernel & buffer pool
├── _vulkan.py        # wgpu/Vulkan backend with WGSL compute kernels
├── _cuda.py          # numba.cuda backend with device kernels
├── _cpu.py           # Pure-Python + NumPy (+ optional Numba) fallback
└── _types.py         # NamedTuple types (Tile, LngLat, Bbox, …)
```

### Dispatch logic

- **Single value** → pure-Python CPU path (minimal overhead, no GPU round-trip)
- **Array input** → GPU kernel (OpenCL via PyOpenCL) — dispatched
  automatically by the public API
- **No GPU device** → fast multicore CPU fallback (with a one-time
  `RuntimeWarning` explaining why)
- **Vulkan / CUDA** → opt-in, used directly via their `get_backend()`
  exports (see `main.py`), so they never interfere with automatic dispatch

`ct.ACTIVE_BACKEND` reports what serves automatic batch dispatch
(`"opencl"` or `"cpu"`). `ct.HAS_OPENCL`, `ct.HAS_NUMBA`, and
`cyrcantile._vulkan.HAS_VULKAN` / `cyrcantile._cuda.HAS_CUDA` report what
is available.

### OpenCL device selection

By default the first GPU device across all platforms is used (falling back
to any device). Two environment variables override this with a
case-insensitive substring match:

| Variable | Example | Effect |
|---|---|---|
| `CYRCANTILE_CL_PLATFORM` | `"NVIDIA"` | restrict platform by name |
| `CYRCANTILE_CL_DEVICE` | `"RTX"` | restrict device by name |

### CPU parallelism

The NumPy vectorised fallback runs on **all cores** for large batches:

| Engine | When | Notes |
|--------|------|-------|
| **Numba** (`@njit(parallel=True)`) | `numba` installed | Fastest; true multicore scaling (`fastmath`) |
| **Thread pool** (chunked NumPy) | no `numba`; ≥ 200k elements | Dependency-free; scales across cores |
| **Single-thread NumPy** | < 200k elements | Avoids dispatch overhead on small inputs |

`ct.HAS_NUMBA` reports whether the JIT path is active.

## Usage

```python
import cyrcantile as ct

# Single value — CPU path
t = ct.tile(-9.14, 53.12, 7)       # Tile(x=60, y=41, z=7)
b = ct.bounds(t)                    # LngLatBbox(west=..., ...)
qk = ct.quadkey(t)                  # "0313102"
p  = ct.parent(t)                   # Tile(x=30, y=20, z=6)
xs, ys = ct.xy(-9.14, 53.12)        # Web-Mercator metres

# Batch — GPU path
import numpy as np
lons = np.array([-9.14, -0.12, 13.38])
lats = np.array([ 53.12,  51.50, 52.52])
xs, ys = ct.tile(lons, lats, 7)     # → numpy int32 arrays

# Tile-coordinate batches: two arrays + zoom, or a list of Tile
lngs, lats = ct.ul(xs, ys, 7)
lngs, lats = ct.ul([ct.Tile(60, 41, 7), ct.Tile(121, 83, 8)])  # mixed zooms OK
w, s, e, n = ct.bounds(xs, ys, 7)
cx, cy = ct.children(xs, ys, 7)     # shape (n, 4), mercantile order
nx, ny = ct.neighbors(xs, ys, 7)    # shape (n, 8), x-major, unfiltered
px, py = ct.parent(xs, ys, 7)       # one zoom up
px, py = ct.parent(xs, ys, 7, to_zoom=4)

# Enumerate tiles in a bounding box
tiles = ct.tiles(-9.5, 53.0, -9.0, 53.3, [14, 15, 16])

# GeoJSON feature
feat = ct.feature(t, fid=1, props={"layer": "base"})
```

### Batch conventions

| Operation | Batch inputs | Batch output |
|---|---|---|
| `xy`, `lnglat` | `lng[], lat[]` / `x[], y[]` | `(a, b)` float64 arrays |
| `tile`, `bounding_tile` | `lng[], lat[], zoom` / `w,s,e,n[], zoom` | `(xs, ys)` int32 arrays |
| `ul` | `x[], y[], zoom` or `Tile[]` | `(lngs, lats)` float64 |
| `bounds`, `xy_bounds` | `x[], y[], zoom` or `Tile[]` | 4× float64 arrays `(w,s,e,n)` / `(l,b,r,t)` |
| `quadkey` | `x[], y[], zoom` or `Tile[]` | list of strings |
| `quadkey_to_tile` | `str[]` (or packed `uint64[]` + zoom) | list of `Tile` (or `(xs, ys)`) |
| `parent` | `x[], y[], zoom[, to_zoom]` or `Tile[]` | `(xs, ys)` int32 |
| `children` | `x[], y[]` or `Tile[]` | `(xs, ys)` int32, shape `(n, 4)` |
| `neighbors` | `x[], y[]` or `Tile[]` | `(xs, ys)` int32, shape `(n, 8)` |

Batch zoom is capped at `ct.MAX_BATCH_ZOOM` (30) because tile indices are
int32. Batch `neighbors` are **not** filtered for grid validity (the
scalar API omits invalid neighbours, mercantile-style) — mask the output
if you need only valid ones.

## Mercantile compatibility

The scalar API matches mercantile 1.2.x exactly: `tile()` clamps indices
into the valid range and applies its EPSILON nudge, `children()` uses the
TL/TR/BR/BL order (with zoom descend), `neighbors()` omits out-of-grid
tiles, `bounding_tile()` uses the same search algorithm, and `tiles()`
handles antimeridian-crossing boxes. This parity is pinned by the test
suite against mercantile as an oracle.

Known deliberate divergences:

- `truncate=True` clamps latitude to ±85.05112878 (the web-mercator limit)
  instead of mercantile 1.2's ±90 — mercantile then *fails* to compute a
  tile for the clamped pole, which defeats the purpose of truncation.
- `bounding_tile(..., zoom=z)` is a cyrcantile extension (mercantile has
  no zoom argument); it returns the tile of the NW corner.
- mercantile-specific exception classes (`InvalidLatitudeError`, …) are
  plain `ValueError`/`TypeError` here.

## Command line interface

`pip install .[cli]` installs a mercantile-style `cyrcantile` command:

```console
$ cyrcantile info
cyrcantile 0.3.0
active batch backend: opencl
OpenCL device: Baffin [AMD Accelerated Parallel Processing]
numba CPU engine: True

$ cyrcantile tile -9.14 53.12 7
[60, 41, 7]

$ cyrcantile tiles -9.5 53.0 -9.0 53.3 12 | head -3
[1939, 1328, 12]
[1939, 1329, 12]
[1939, 1330, 12]

$ echo "[486, 332, 10]" | cyrcantile parent
[243, 166, 9]

$ echo "[486, 332, 10]" | cyrcantile quadkey
0313102310

$ echo "0313102310" | cyrcantile quadkey
[486, 332, 10]

$ echo "[-105.05, 39.95, -105, 40]" | cyrcantile bounding-tile
[426, 775, 11]

$ cyrcantile bench --size 8_000_000
```

`parent`, `children`, `neighbors`, `quadkey`, `bounds` and `shapes` read
newline-delimited JSON `[x, y, z]` tiles from stdin (or a file/literal
string) and process them with the **batch** API — large inputs run on
the GPU automatically.  `shapes --collect` emits a GeoJSON
FeatureCollection.  See `cyrcantile COMMAND --help` for all options.

## Benchmark

```bash
python main.py                       # 8M points by default
python main.py --size 100_000_000    # the big guns
python benchmarks/compare.py         # vs mercantile / supermercado / NumPy
python benchmarks/batch.py           # millions of coords & millions of tiles
```

`benchmarks/batch.py` times **all 12 batch operations** (4 coordinate + 8
tile) on every available backend (OpenCL, Vulkan, CUDA, multicore CPU,
single-thread CPU), verifies the GPU results against the CPU reference and
finishes with end-to-end public-API timings (quadkey strings, string
decoding, tile enumeration):

```bash
python benchmarks/batch.py --coords 10_000_000 --tiles 10_000_000 --repeat 5
python benchmarks/batch.py --backends opencl,cpu   # subset of backends
```

Typical output (8M random points, zoom 12; AMD Radeon RX 560 + 12-core CPU):

| Backend | Time |
|---------|------|
| OpenCL (GPU, f64) | ~75 ms |
| Vulkan (GPU, f32) | ~90 ms |
| CPU single-thread | ~500 ms |
| CPU parallel (numba, 12 cores) | ~27 ms |

Note: on modest GPUs with f64 kernels a multicore numba CPU can be
competitive; the GPU backends shine on bigger batches (and on machines
whose CPUs are busy).  Vulkan's f32 may flip ~0.02% of tile indices at
floor boundaries.  Bit-twiddling ops (`parent`) and pure arithmetic
(`xy_bounds`) are faster on the CPU than on the GPU — the dispatch
thresholds favour the GPU for transcendental-heavy work (`tile`,
`bounds`, `quadkey` at 30–50x).

## Notes & caveats

- **Vulkan precision:** WGSL kernels use f32 storage buffers (f64 needs
  `shader-f64`, which many devices lack). Float outputs are accurate to
  ~1e-5 relative; `tile` indices may differ by 1 at floor boundaries.
  OpenCL and CUDA compute in f64 and match the CPU exactly.
- **Vulkan quadkeys** require the `shader-int64` device feature; the
  backend raises `NotImplementedError` when it is missing.
- On `llvmpipe` (software) Vulkan devices the f32 transcendentals
  (`exp`/`log`/`tan`/`atan`/`sin`/`sinh`) compute incorrectly, so
  validate with `sqrt`/`floor`-based kernels (parent/children/neighbors)
  there or use a real GPU.
- GPU backends read/write flat 1-D C-contiguous arrays; multi-dimensional
  inputs are ravelled.

## Migration from Cython cyrcantile / v0.2

| Original | 0.3 |
|---|---|
| `_base.pxd` (struct decls) | `_types.py` (NamedTuples) |
| `_base.pyx` (Cython impl) | `_kernels.cl` + `_backend.py` / `_vulkan.py` / `_cuda.py` |
| `setup.py` (cythonize) | `pyproject.toml` |
| `numba` dependency | optional extras: `opencl` / `vulkan` / `cuda` / `cpu-speed` |

Breaking changes in 0.3:

- Batch tile operations (`ul`, `bounds`, `xy_bounds`, `quadkey`,
  `parent`, `children`, `neighbors`) now take **separate `x` and `y`
  arrays** (or a list of `Tile`). Previously the single input array was
  used for both coordinates — the old results were wrong.
- Batch `children`/`neighbors` return shaped arrays `(n, 4)` / `(n, 8)`
  instead of flat lists of `Tile`.
- Batch `parent` returns `(xs, ys)` arrays; the scalar form now honours
  the target-zoom argument (mercantile semantics) and returns `None` for
  a z=0 tile.
- `children()` order is mercantile's TL, TR, BR, BL; scalar `neighbors()`
  omits invalid neighbours; `tile()` clamps indices; `tiles()` handles
  the antimeridian — all matching mercantile 1.2.
- `bounding_tile()` without zoom now performs the real mercantile search
  (previously it always returned `Tile(0, 0, 0)`).
- Lists of a single element now follow the batch path (array in → array
  out); inputs are validated (mismatched lengths, zoom bounds, quadkey
  digits raise `ValueError`/`TypeError`).

## Development

```bash
pip install -e .[dev]
pytest              # CPU tests + GPU backends when available
ruff check .        # lint
```

GPU backend tests run automatically for every backend that initialises
(OpenCL, Vulkan, CUDA); without a GPU they are skipped.  The CUDA
kernels are additionally verified on any machine through numba's CUDA
simulator (`NUMBA_ENABLE_CUDASIM=1` — see `tests/test_cudasim.py`), which
is how the historical `xy` degrees/radians bug was caught.

## License

MIT — see [LICENSE](LICENSE).
