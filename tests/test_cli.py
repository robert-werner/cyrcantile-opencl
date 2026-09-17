"""Tests for the mercantile-style CLI (cyrcantile.scripts)."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from cyrcantile.scripts import cli

runner = CliRunner()


def run(args, input=None):
    result = runner.invoke(cli, args, input=input, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result.output


def lines_of(output):
    return [json.loads(line) for line in output.splitlines() if line]


# ------------------------------------------------------------------ #
#  tile
# ------------------------------------------------------------------ #

def test_tile():
    out = run(["tile", "-9.14", "53.12", "7"])
    assert json.loads(out) == [60, 41, 7]


def test_tile_truncate():
    out = run(["tile", "200", "100", "5", "--truncate"])
    t = json.loads(out)
    assert t[2] == 5
    assert 0 <= t[0] < 32 and 0 <= t[1] < 32


# ------------------------------------------------------------------ #
#  tiles / bounding-tile
# ------------------------------------------------------------------ #

def test_tiles():
    out = run(["tiles", "-9.5", "53.0", "-9.0", "53.3", "12"])
    tiles = lines_of(out)
    assert len(tiles) == 49
    assert all(t[2] == 12 for t in tiles)
    assert tiles[0] == [1939, 1328, 12]
    # unique
    assert len({tuple(t) for t in tiles}) == 49


def test_tiles_multiple_zooms():
    out = run(["tiles", "-9.5", "53.0", "-9.0", "53.3", "11", "12"])
    tiles = lines_of(out)
    assert {t[2] for t in tiles} == {11, 12}


def test_tiles_antimeridian():
    out = run(["tiles", "170.0", "-10.0", "-170.0", "10.0", "2"])
    tiles = lines_of(out)
    assert len(tiles) == 4


def test_tiles_seq():
    out = run(["tiles", "-9.5", "53.0", "-9.0", "53.3", "11", "--seq"])
    assert out.startswith("\x1e\n")
    assert len(lines_of(out)) == 16


def test_bounding_tile():
    out = run(["bounding-tile"], input="[-105.05, 39.95, -105, 40]\n")
    assert json.loads(out) == [426, 775, 11]


def test_bounding_tile_point():
    # a 2-element bbox is a point
    out = run(["bounding-tile"], input="[-9.14, 53.12]\n")
    assert len(lines_of(out)) == 1


def test_bounding_tile_with_zoom():
    out = run(["bounding-tile", "--zoom", "8"], input="[-9.5, 53.0, -9.0, 53.3]\n")
    assert json.loads(out) == [121, 83, 8]


def test_bounding_tile_string_input():
    out = run(["bounding-tile", "[-105.05, 39.95, -105, 40]"])
    assert json.loads(out) == [426, 775, 11]


# ------------------------------------------------------------------ #
#  parent / children / neighbors
# ------------------------------------------------------------------ #

def test_parent():
    out = run(["parent"], input="[486, 332, 10]\n")
    assert json.loads(out) == [243, 166, 9]


def test_parent_depth():
    out = run(["parent", "--depth", "2"], input="[486, 332, 10]\n")
    assert json.loads(out) == [121, 83, 8]


def test_parent_multiple_tiles_preserves_order():
    inp = "[486, 332, 10]\n[100, 200, 8]\n"
    out = run(["parent"], input=inp)
    tiles = lines_of(out)
    assert tiles == [[243, 166, 9], [50, 100, 7]]


def test_parent_invalid_level():
    result = runner.invoke(cli, ["parent"], input="[1, 1, 0]\n")
    assert result.exit_code != 0


def test_children():
    out = run(["children"], input="[486, 332, 10]\n")
    assert lines_of(out) == [
        [972, 664, 11], [973, 664, 11], [973, 665, 11], [972, 665, 11],
    ]


def test_children_depth():
    out = run(["children", "--depth", "2"], input="[486, 332, 10]\n")
    tiles = lines_of(out)
    assert len(tiles) == 16
    assert all(t[2] == 12 for t in tiles)


def test_neighbors():
    out = run(["neighbors"], input="[486, 332, 10]\n")
    tiles = lines_of(out)
    assert len(tiles) == 8
    assert sorted(map(tuple, tiles)) == sorted([
        (485, 331, 10), (485, 332, 10), (485, 333, 10),
        (486, 331, 10), (486, 333, 10),
        (487, 331, 10), (487, 332, 10), (487, 333, 10),
    ])


def test_neighbors_filters_invalid():
    # corner tile at z1: only 3 neighbors are on the grid
    out = run(["neighbors"], input="[0, 0, 1]\n")
    tiles = lines_of(out)
    assert sorted(map(tuple, tiles)) == [(0, 1, 1), (1, 0, 1), (1, 1, 1)]


# ------------------------------------------------------------------ #
#  quadkey
# ------------------------------------------------------------------ #

def test_quadkey_encode():
    out = run(["quadkey"], input="[486, 332, 10]\n")
    assert out.splitlines() == ["0313102310"]


def test_quadkey_decode():
    out = run(["quadkey"], input="0313102310\n")
    assert json.loads(out) == [486, 332, 10]


def test_quadkey_batch_preserves_order():
    inp = "[486, 332, 10]\n[60, 41, 7]\n"
    out = run(["quadkey"], input=inp)
    assert out.splitlines() == ["0313102310", "0313102"]


def test_quadkey_mixed():
    inp = "[486, 332, 10]\n0313102\n"
    out = run(["quadkey"], input=inp)
    assert out.splitlines() == ["0313102310", "[60, 41, 7]"]


def test_quadkey_bad_digit():
    result = runner.invoke(cli, ["quadkey"], input="9313102\n")
    assert result.exit_code != 0
    assert "0, 1, 2 or 3" in result.output


# ------------------------------------------------------------------ #
#  bounds / shapes
# ------------------------------------------------------------------ #

def test_bounds():
    out = run(["bounds"], input="[486, 332, 10]\n")
    w, s, e, n = json.loads(out)
    assert w == pytest.approx(-9.140625)
    assert e == pytest.approx(-8.7890625)
    assert s == pytest.approx(53.120405283106564)
    assert n == pytest.approx(53.33087298301705)


def test_bounds_extents():
    out = run(["bounds", "--extents"], input="[486, 332, 10]\n")
    parts = out.split()
    assert len(parts) == 4


def test_shapes_feature():
    out = run(["shapes"], input="[486, 332, 10]\n")
    feature = json.loads(out)
    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Polygon"


def test_shapes_properties_passthrough():
    inp = '{"tile": [486, 332, 10], "id": 7, "properties": {"name": "foo"}}\n'
    out = run(["shapes"], input=inp)
    feature = json.loads(out)
    assert feature["id"] == 7
    assert feature["properties"] == {"name": "foo"}


def test_shapes_collect():
    inp = "[486, 332, 10]\n[486, 333, 10]\n"
    out = run(["shapes", "--collect"], input=inp)
    fc = json.loads(out)
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 2
    assert len(fc["bbox"]) == 4
    assert fc["bbox"][0] < fc["bbox"][2]


def test_shapes_bad_input():
    result = runner.invoke(cli, ["shapes"], input="not json\n")
    assert result.exit_code != 0


# ------------------------------------------------------------------ #
#  info / bench / general
# ------------------------------------------------------------------ #

def test_info():
    out = run(["info"])
    assert "cyrcantile" in out
    assert "active batch backend" in out


def test_bench_small():
    out = run(["bench", "--size", "2000"])
    assert "CPU tile_vec" in out


def test_help_lists_commands():
    out = run(["--help"])
    for cmd in ("tile", "tiles", "bounding-tile", "parent", "children",
                "neighbors", "quadkey", "bounds", "shapes", "info", "bench"):
        assert cmd in out


def test_version():
    out = run(["--version"])
    assert out.strip().isdigit() or "." in out


def test_parse_tile_rejects_negative():
    result = runner.invoke(cli, ["parent"], input="[-1, 2, 5]\n")
    assert result.exit_code != 0
