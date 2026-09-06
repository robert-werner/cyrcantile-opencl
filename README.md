# cyrcantile — OpenCL / Vulkan / CUDA

GPU-accelerated spherical-mercator tile utilities — an OpenCL, Vulkan and
CUDA kernel rewrite of the original [cyrcantile](https://github.com/robert-werner/cyrcantile)
Cython project.

## What it does

Provides the same API as [mercantile](https://github.com/mapbox/mercantile)
for converting between geographic coordinates (WGS-84), Web Mercator
(EPSG:3857), and XYZ tile indices — but with the compute-heavy batch
operations offloaded to GPU kernels (OpenCL, Vulkan, or CUDA).

## Architecture

```
cyrcantile/
├── __init__.py       # Public API — auto-dispatches CPU / GPU
├── _kernels.cl       # OpenCL C kernels (13 batch operations)
├── _backend.py       # PyOpenCL context manager & kernel dispatch
├── _vulkan.py        # wgpu/Vulkan backend with WGSL compute kernels
├── _cuda.py          # numba.cuda backend with device kernels
├── _cpu.py           # Pure-Python + NumPy fallback
└── _types.py         # NamedTuple types (Tile, LngLat, Bbox, …)
```

### OpenCL kernels

| Kernel | Description |
|--------|-------------|
| `xy_batch` | Geographic → Web Mercator (batch) |
| `lnglat_batch` | Web Mercator → Geographic (batch) |
| `tile_batch` | Geographic → Tile x,y at zoom (batch) |
| `ul_batch` | Tile → upper-left corner lng/lat (batch) |
| `bounds_batch` | Tile → geographic bbox w/s/e/n (batch) |
| `xy_bounds_batch` | Tile → Web-Mercator bbox l/b/r/t (batch) |
| `quadkey_encode_batch` | Tile → packed quadkey (batch) |
| `quadkey_decode_batch` | Packed quadkey → Tile (batch) |
| `parent_batch` | Tile → parent tile (batch) |
| `children_batch` | Tile → 4 children (batch) |
| `neighbors_batch` | Tile → 8 neighbours (batch) |
| `bounding_tile_batch` | bbox → bounding tile NW corner (batch) |
| `tiles_in_bbox` | Grid-fill enumerate tiles in bbox |

### Dispatch logic

- **Single value** → pure-Python CPU path (minimal overhead, no GPU round-trip)
- **Array input** → GPU kernel (OpenCL via PyOpenCL, Vulkan via wgpu, or CUDA via numba)
- **No GPU device** → fast multicore CPU fallback

The OpenCL path is dispatched automatically by the public API. The Vulkan
and CUDA backends are used directly via their `get_backend()` /
`HAS_VULKAN` / `HAS_CUDA` exports (see `main.py`), so they never interfere
with automatic dispatch or each other.

### CPU parallelism

The NumPy vectorised fallback runs on **all cores** for large batches. Two
engines are chosen at runtime:

| Engine | When | Notes |
|--------|------|-------|
| **Numba** (`@njit(parallel=True)`) | `numba` installed | Fastest; true multicore scaling (`fastmath`) |
| **Thread pool** (chunked NumPy) | no `numba`; ≥ 200k elements | Dependency-free; scales across cores |
| **Single-thread NumPy** | < 200k elements | Avoids dispatch overhead on small inputs |

`ct.HAS_NUMBA` reports whether the JIT path is active.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

> Requires a GPU backend. Install `wgpu` for Vulkan, `pyopencl` for OpenCL
> (Linux may need `ocl-icd-opencl-dev`), or run `pip install numba` and have
> a CUDA-capable GPU + driver for the CUDA backend.

> **CUDA support:** the CUDA backend (`_cuda.py`) JIT-compiles the same
> kernels with `numba.cuda`. `HAS_CUDA` is `True` only when numba is
> installed *and* an NVIDIA GPU is reachable. Results match the OpenCL
> kernels exactly.

> **Note:** on `llvmpipe` (software) Vulkan devices the f64 transcendentals
> (`exp`/`log`/`tan`/`atan`/`sin`/`cos`/`sinh`/`exp2`) compute incorrectly, so
> validate with `sqrt`/`floor`-based kernels (parent/children/neighbors) there
> or use a real GPU.

## Usage

```python
import cyrcantile as ct

# Single value — CPU path
t = ct.tile(-9.14, 53.12, 7)       # Tile(x=62, y=60, z=7)
b = ct.bounds(t)                    # LngLatBbox(west=..., ...)
qk = ct.quadkey(t)                  # "0211333"
p  = ct.parent(t)                   # Tile(x=31, y=30, z=6)

# Batch — GPU path
import numpy as np
lons = np.array([-9.14, -0.12, 13.38])
lats = np.array([ 53.12, 51.50, 52.52])
xs, ys = ct.tile(lons, lats, 7)     # → numpy int arrays

# Enumerate tiles in a bounding box
tiles = ct.tiles(-9.5, 53.0, -9.0, 53.3, [14, 15, 16])

# GeoJSON feature
feat = ct.feature(t, fid=1, props={"layer": "base"})
```

## Benchmark

```bash
python main.py
```

Typical output (8M random points, zoom 12, 4 cores):

| Backend | Time |
|---------|------|
| Vulkan (llvmpipe) | ~550 ms |
| CUDA (numba) | device-dependent |
| CPU single-thread | ~527 ms |
| CPU parallel (numba) | ~57 ms |

## Migration from Cython

| Original (Cython) | OpenCL rewrite |
|---|---|
| `_base.pxd` (struct decls) | `_types.py` (NamedTuples) |
| `_base.pyx` (Cython impl) | `_kernels.cl` + `_backend.py` / `_vulkan.py` / `_cuda.py` |
| `setup.py` (cythonize) | `setup.py` (package_data) |
| `numba` dependency | `wgpu` / `pyopencl` / `numba` (CUDA) dependency |

## License

MIT
