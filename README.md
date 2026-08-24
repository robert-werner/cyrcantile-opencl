# cyrcantile — OpenCL

GPU-accelerated spherical-mercator tile utilities — an OpenCL kernel
rewrite of the original [cyrcantile](https://github.com/robert-werner/cyrcantile)
Cython project.

## What it does

Provides the same API as [mercantile](https://github.com/mapbox/mercantile)
for converting between geographic coordinates (WGS-84), Web Mercator
(EPSG:3857), and XYZ tile indices — but with the compute-heavy batch
operations offloaded to OpenCL kernels running on the GPU.

## Architecture

```
cyrcantile/
├── __init__.py       # Public API — auto-dispatches CPU / GPU
├── _kernels.cl       # OpenCL C kernels (13 batch operations)
├── _backend.py       # PyOpenCL context manager & kernel dispatch
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
- **Array input** → OpenCL kernel (parallel GPU computation)
- **No OpenCL device** → automatic NumPy vectorised fallback

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

> Requires an OpenCL-capable GPU and platform drivers. Install `pyopencl`
> with `pip install pyopencl` (Linux may need `ocl-icd-opencl-dev`).

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

Typical output (100K random points, zoom 12):

| Backend | Time |
|---------|------|
| OpenCL (GPU) | ~2 ms |
| NumPy (CPU) | ~45 ms |

## Migration from Cython

| Original (Cython) | OpenCL rewrite |
|---|---|
| `_base.pxd` (struct decls) | `_types.py` (NamedTuples) |
| `_base.pyx` (Cython impl) | `_kernels.cl` + `_backend.py` |
| `setup.py` (cythonize) | `setup.py` (package_data) |
| `numba` dependency | `pyopencl` dependency |

## License

MIT
