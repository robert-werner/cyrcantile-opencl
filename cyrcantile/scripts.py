"""cyrcantile command line interface.

A mercantile-style CLI.  Tile-based commands read newline-delimited JSON
``[x, y, z]`` arrays (or ``{"tile": [x, y, z], ...}`` objects) from a
file, stdin, or a literal string argument, and write one JSON array per
output tile — exactly like mercantile.  Where possible the commands use
the *batch* API, so large inputs are computed on the GPU automatically.

Examples
--------

    cyrcantile tile -9.14 53.12 7
    echo "[486, 332, 10]" | cyrcantile parent
    echo "[486, 332, 10]" | cyrcantile quadkey
    cyrcantile tiles -9.5 53.0 -9.0 53.3 12 13
    cyrcantile info
    cyrcantile bench --size 1_000_000
"""

import json
import logging
import sys

import click

import cyrcantile as ct
from cyrcantile._types import Tile

__all__ = ["cli"]

logger = logging.getLogger(__name__)

RS = "\x1e"


def configure_logging(verbosity):
    log_level = max(10, 30 - 10 * verbosity)
    logging.basicConfig(stream=sys.stderr, level=log_level)


def normalize_input(input):
    """Normalize file, stdin, or literal-string input."""
    try:
        src = click.open_file(input).readlines()
    except OSError:
        src = [input]
    return src


def iter_lines(lines):
    """Iterate over lines of input, stripping and skipping."""
    for line in lines:
        line = line.strip()
        if line:
            yield line


def parse_tile_line(line):
    """Parse a JSON line into a Tile ([x, y, z] array or tile object)."""
    obj = json.loads(line)
    if isinstance(obj, dict):
        obj = obj["tile"]
    if not isinstance(obj, (list, tuple)) or len(obj) < 3:
        raise click.BadParameter(f"not a tile: {line!r}", param_hint="input")
    x, y, z = obj[:3]
    if z < 0 or x < 0 or y < 0:
        raise click.BadParameter(f"not a valid tile: {line!r}", param_hint="input")
    return Tile(int(x), int(y), int(z))


def read_tiles(input):
    """Read all input lines as a list of Tile."""
    return [parse_tile_line(line) for line in iter_lines(normalize_input(input))]


def echo_tile(t, seq=False):
    if seq:
        click.echo(RS)
    click.echo(json.dumps([t.x, t.y, t.z]))


# The CLI command group.
@click.group(help="Command line interface for the cyrcantile package.")
@click.option("--verbose", "-v", count=True, help="Increase verbosity.")
@click.option("--quiet", "-q", count=True, help="Decrease verbosity.")
@click.version_option(version=ct.__version__, message="%(version)s")
@click.pass_context
def cli(ctx, verbose, quiet):
    """Execute the main cyrcantile command."""
    configure_logging(verbose - quiet)
    ctx.obj = {"verbosity": verbose - quiet}


# Commands are below.

@cli.command(
    short_help="Get the tile of a lng/lat point.",
    # allow negative lng/lat values like "-9.14" (click would otherwise
    # mistake them for options) while real options still parse
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
)
# LNG/LAT are strings (not type=float) so that negative values like
# "-9.14" are not mistaken for options by click.
@click.argument("lng")
@click.argument("lat")
@click.argument("zoom")
@click.option("--truncate", is_flag=True, default=False,
              help="Truncate inputs to web-mercator limits.")
def tile(lng, lat, zoom, truncate):
    """Print the Web Mercator tile containing LNG LAT at ZOOM.

    \b
    cyrcantile tile -9.14 53.12 7
    [60, 41, 7]
    """
    try:
        lng_f = float(lng)
        lat_f = float(lat)
        zoom_i = int(zoom)
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint="arguments") from e
    t = ct.tile(lng_f, lat_f, zoom_i, truncate=truncate)
    echo_tile(t)


@cli.command(
    short_help="Print tiles that cover a bounding box.",
    # allow negative west/south values (see the tile command)
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
)
# String arguments again, to support negative west/south values.
@click.argument("args", nargs=-1, required=True)
@click.option("--seq", is_flag=True, default=False,
              help="Write a RS-delimited JSON sequence (default is LF).")
def tiles(args, seq):
    """Print all Web Mercator tiles covering the bounding box at ZOOMS.

    \b
    cyrcantile tiles WEST SOUTH EAST NORTH ZOOM [ZOOM ...]

    Large grids are enumerated on the GPU.  Inputs are clamped to the
    web-mercator limits and antimeridian-crossing boxes are handled.

    \b
    cyrcantile tiles -9.5 53.0 -9.0 53.3 12
    [1939, 1328, 12]
    ...
    """
    if len(args) < 5:
        raise click.UsageError(
            "usage: cyrcantile tiles WEST SOUTH EAST NORTH ZOOM [ZOOM ...]")
    try:
        west, south, east, north = (float(a) for a in args[:4])
        zooms = [int(a) for a in args[4:]]
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint="arguments") from e
    for t in ct.tiles(west, south, east, north, zooms):
        echo_tile(t, seq)


@cli.command("bounding-tile",
             short_help="Print the bounding tile of a bbox, point or GeoJSON.")
@click.argument("input", default="-", required=False)
@click.option("--zoom", type=int, default=None,
              help="Tile of the NW corner at this zoom (default: search).")
def bounding_tile(input, zoom):
    """Print the smallest Web Mercator tile bounding [w, s, e, n] boxes
    read from stdin.  With --zoom, the tile of the NW corner is printed.

    \b
    echo "[-105.05, 39.95, -105, 40]" | cyrcantile bounding-tile
    [426, 775, 11]
    """
    for line in iter_lines(normalize_input(input)):
        obj = json.loads(line)
        if isinstance(obj, dict):
            obj = obj.get("bbox")
            if obj is None:
                raise click.BadParameter(
                    f"no bbox in object: {line!r}", param_hint="input")
        if len(obj) == 2:
            obj = list(obj) + list(obj)
        if len(obj) != 4:
            raise click.BadParameter(f"not a bbox: {line!r}", param_hint="input")
        w, s, e, n = obj
        t = ct.bounding_tile(w, s, e, n, zoom)
        if t is None:
            continue
        echo_tile(t)


@cli.command(short_help="Print the parent tiles.")
@click.argument("input", default="-", required=False)
@click.option("--depth", type=int, default=1,
              help="Number of zoom levels to traverse (default is 1).")
def parent(input, depth):
    """Takes [x, y, z] tiles as input and writes parents to stdout.

    All input tiles are converted in one batch call per target zoom,
    so large inputs run on the GPU.

    \b
    echo "[486, 332, 10]" | cyrcantile parent
    [243, 166, 9]
    """
    if depth < 1:
        raise click.UsageError("depth must be >= 1")
    tiles_in = read_tiles(input)
    if not tiles_in:
        return
    results = [None] * len(tiles_in)
    groups = {}
    for i, t in enumerate(tiles_in):
        target = t.z - depth
        if target < 0:
            raise click.UsageError(f"Invalid parent level: {target}")
        groups.setdefault(target, []).append(i)
    for target, idxs in groups.items():
        xs, ys = ct.parent([tiles_in[i] for i in idxs], to_zoom=target)
        for i, x, y in zip(idxs, xs.tolist(), ys.tolist()):
            results[i] = (x, y, target)
    for x, y, z in results:
        click.echo(json.dumps([x, y, z]))


@cli.command(short_help="Print the children of the tiles.")
@click.argument("input", default="-", required=False)
@click.option("--depth", type=int, default=1,
              help="Number of zoom levels to traverse (default is 1).")
def children(input, depth):
    """Takes [x, y, z] tiles as input and writes children to stdout
    in mercantile order (top-left, top-right, bottom-right, bottom-left).

    Each traversal level is one batch call, so large inputs run on the GPU.

    \b
    echo "[486, 332, 10]" | cyrcantile children
    [972, 664, 11]
    [973, 664, 11]
    [973, 665, 11]
    [972, 665, 11]
    """
    if depth < 1:
        raise click.UsageError("depth must be >= 1")
    cur = read_tiles(input)
    for _ in range(depth):
        if not cur:
            return
        # children ignores the source zoom; only bookkeeping needs it
        zs = [t.z for t in cur]
        xs, ys = ct.children(cur)  # shape (n, 4), batch (GPU when large)
        cur = [
            Tile(int(xs[i, c]), int(ys[i, c]), zs[i] + 1)
            for i in range(len(cur))
            for c in range(4)
        ]
    for t in cur:
        click.echo(json.dumps([t.x, t.y, t.z]))


@cli.command(short_help="Print the neighbors of the tiles.")
@click.argument("input", default="-", required=False)
def neighbors(input):
    """Takes [x, y, z] tiles as input and writes adjacent tiles on the
    same zoom level to stdout.  Out-of-grid neighbors are omitted.

    Neighbors are computed with one batch call (GPU when large).

    \b
    echo "[486, 332, 10]" | cyrcantile neighbors
    [485, 331, 10]
    ...
    """
    import numpy as np

    tiles_in = read_tiles(input)
    if not tiles_in:
        return
    xs, ys = ct.neighbors(tiles_in)  # shape (n, 8), batch (GPU when large)
    zs = np.array([t.z for t in tiles_in], dtype=np.int64)
    hi = (np.left_shift(1, zs) - 1)[:, None]
    valid = (xs >= 0) & (xs <= hi) & (ys >= 0) & (ys <= hi)
    for i, t in enumerate(tiles_in):
        for c in range(8):
            if valid[i, c]:
                click.echo(json.dumps([int(xs[i, c]), int(ys[i, c]), t.z]))


@cli.command(short_help="Convert to/from quadkeys.")
@click.argument("input", default="-", required=False)
def quadkey(input):
    """Takes [x, y, z] tiles or quadkeys as input and writes quadkeys
    or [x, y, z] tiles to stdout, respectively.

    Tile inputs are converted with one batch call (GPU when large).

    \b
    echo "[486, 332, 10]" | cyrcantile quadkey
    0313102310

    \b
    echo "0313102310" | cyrcantile quadkey
    [486, 332, 10]
    """
    lines = list(iter_lines(normalize_input(input)))
    tile_lines = [line for line in lines if line[0] == "["]
    keys = iter(ct.quadkey([parse_tile_line(line) for line in tile_lines]))
    try:
        for line in lines:
            if line[0] == "[":
                click.echo(next(keys))
            else:
                t = ct.quadkey_to_tile(line)
                click.echo(json.dumps([t.x, t.y, t.z]))
    except (ValueError, TypeError) as e:
        raise click.BadParameter(str(e), param_hint="input") from e


@cli.command(short_help="Print the geographic bounds of tiles.")
@click.argument("input", default="-", required=False)
@click.option("--extents/--no-extents", default=False,
              help="Write bounds as ws-separated strings (default is JSON).")
@click.option("--seq", is_flag=True, default=False,
              help="Write a RS-delimited JSON sequence (default is LF).")
def bounds(input, extents, seq):
    """Takes [x, y, z] tiles as input and writes their geographic
    bounding boxes [w, s, e, n] to stdout.

    All bounds are computed with one batch call (GPU when large).

    \b
    echo "[486, 332, 10]" | cyrcantile bounds
    [-9.140625, 53.120405283106564, -8.7890625, 53.33087298301705]
    """
    tiles_in = read_tiles(input)
    if not tiles_in:
        return
    w, s, e, n = ct.bounds(tiles_in)  # batch (GPU when large)
    for row in zip(w.tolist(), s.tolist(), e.tolist(), n.tolist()):
        if seq:
            click.echo(RS)
        if extents:
            click.echo(" ".join(map(str, row)))
        else:
            click.echo(json.dumps(row))


@cli.command(short_help="Print the shapes of tiles as GeoJSON.")
@click.argument("input", default="-", required=False)
@click.option("--collect", is_flag=True, default=False,
              help="Output as a GeoJSON FeatureCollection (default: features).")
@click.option("--indent", type=int, default=None,
              help="Indentation level for JSON output.")
@click.option("--seq", is_flag=True, default=False,
              help="Write a RS-delimited JSON sequence (default is LF).")
def shapes(input, collect, indent, seq):
    """Takes [x, y, z] tiles as input and writes GeoJSON features to
    stdout, one per tile.  Objects of the form
    ``{"tile": [x, y, z], "properties": {...}, "id": ...}`` pass their
    properties/id through to the feature.

    Bounds for the collection bbox are computed with one batch call.

    \b
    echo "[486, 332, 10]" | cyrcantile shapes --collect
    {"type": "FeatureCollection", ...}
    """
    items = []
    for line in iter_lines(normalize_input(input)):
        obj = json.loads(line)
        props, fid = {}, None
        if isinstance(obj, dict):
            obj, props, fid = obj["tile"], obj.get("properties"), obj.get("id")
        if not isinstance(obj, (list, tuple)) or len(obj) < 3:
            raise click.BadParameter(f"not a tile: {line!r}", param_hint="input")
        items.append((Tile(int(obj[0]), int(obj[1]), int(obj[2])), fid, props))

    features = []
    col_xs, col_ys = [], []
    if items:
        w, s, e, n = ct.bounds([t for t, _, _ in items])  # batch (GPU when large)
        for (t, fid, props), bw, bs, be, bn in zip(items, w.tolist(), s.tolist(),
                                                   e.tolist(), n.tolist()):
            feature = ct.feature(t, fid=fid, props=props)
            if collect:
                features.append(feature)
                col_xs.extend([bw, be])
                col_ys.extend([bs, bn])
            else:
                if seq:
                    click.echo(RS)
                click.echo(json.dumps(feature, sort_keys=True, indent=indent))

    if collect and features:
        bbox = [min(col_xs), min(col_ys), max(col_xs), max(col_ys)]
        click.echo(json.dumps(
            {"type": "FeatureCollection", "bbox": bbox, "features": features},
            sort_keys=True, indent=indent))


@cli.command(short_help="Show backend and runtime info.")
def info():
    """Print the package version, active batch backend and device."""
    click.echo(f"cyrcantile {ct.__version__}")
    click.echo(f"active batch backend: {ct.ACTIVE_BACKEND}")
    if ct._backend is not None:
        dev = ct._backend.device
        platform = dev.platform.name.strip()
        click.echo(f"OpenCL device: {dev.name.strip()} [{platform}]")
    click.echo(f"numba CPU engine: {ct.HAS_NUMBA}")
    try:
        from cyrcantile._vulkan import HAS_VULKAN
        click.echo(f"Vulkan available: {HAS_VULKAN}")
    except Exception:
        click.echo("Vulkan available: False")
    try:
        from cyrcantile._cuda import HAS_CUDA
        click.echo(f"CUDA available: {HAS_CUDA}")
    except Exception:
        click.echo("CUDA available: False")


@cli.command(short_help="Run a tile-batch benchmark.")
@click.option("--size", type=int, default=8_000_000,
              help="Number of random points (default 8,000,000).")
@click.option("--zoom", type=int, default=12, help="Zoom level (default 12).")
def bench(size, zoom):
    """Benchmark the batch tile computation on SIZE random points:
    GPU (auto-dispatched OpenCL) versus the vectorised CPU path."""
    import time

    import numpy as np

    if size <= 0:
        raise click.UsageError("size must be positive")
    np.random.seed(42)
    lons = np.random.uniform(-180, 180, size)
    lats = np.random.uniform(-85, 85, size)

    if ct._backend is not None:
        ct._backend.tile(lons[:4096], lats[:4096], zoom)  # warmup
        t0 = time.perf_counter()
        gx, gy = ct._backend.tile(lons, lats, zoom)
        gpu_ms = (time.perf_counter() - t0) * 1000
        click.echo(f"OpenCL tile({size:,} pts, z={zoom}) = {gpu_ms:.1f} ms")
    else:
        gx = gy = None
        click.echo("OpenCL unavailable - GPU benchmark skipped")

    from cyrcantile import _cpu

    nwarm = min(size, 250_000)
    _cpu.tile_vec(lons[:nwarm], lats[:nwarm], zoom)  # warm up numba
    t0 = time.perf_counter()
    cx, cy = _cpu.tile_vec(lons, lats, zoom)
    cpu_ms = (time.perf_counter() - t0) * 1000
    click.echo(f"CPU tile_vec({size:,} pts, z={zoom}) = {cpu_ms:.1f} ms")

    if gx is not None:
        match = bool(np.array_equal(gx, cx) and np.array_equal(gy, cy))
        if gpu_ms:
            click.echo(f"GPU/CPU speedup: {cpu_ms / gpu_ms:.1f}x")
        click.echo(f"OpenCL == CPU results: {match}")


if __name__ == "__main__":
    cli()
