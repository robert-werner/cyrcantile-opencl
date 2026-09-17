"""Packed-quadkey vectorised conversion and input validation."""

from __future__ import annotations

import numpy as np
import pytest

import cyrcantile as ct
from cyrcantile import _cpu
from cyrcantile._types import Tile


def test_ints_to_strings_roundtrip():
    zoom = 12
    xs = np.arange(300, dtype=np.int32)
    ys = np.arange(300, dtype=np.int32) * 7
    ints = _cpu.quadkey_encode_vec(xs, ys, zoom)
    strs = _cpu.quadkey_ints_to_strings(ints, zoom)
    assert len(strs) == 300
    assert all(len(s) == zoom for s in strs)
    # every string decodes back to the original tile
    back = _cpu.quadkey_decode_vec(_cpu.quadkey_strings_to_ints(strs), zoom)
    np.testing.assert_array_equal(back[0], xs)
    np.testing.assert_array_equal(back[1], ys)


def test_ints_to_strings_matches_scalar():
    zoom = 7
    xs = np.array([60, 30, 121, 0], dtype=np.int32)
    ys = np.array([41, 20, 83, 0], dtype=np.int32)
    ints = _cpu.quadkey_encode_vec(xs, ys, zoom)
    strs = _cpu.quadkey_ints_to_strings(ints, zoom)
    assert strs == [
        ct.quadkey(Tile(60, 41, 7)),
        ct.quadkey(Tile(30, 20, 7)),
        ct.quadkey(Tile(121, 83, 7)),
        ct.quadkey(Tile(0, 0, 7)),
    ]


def test_strings_to_ints_matches_scalar():
    ints = _cpu.quadkey_strings_to_ints(["0313102"])
    assert ints[0] == int(_cpu.quadkey_encode_vec(np.array([60]), np.array([41]), 7)[0])


def test_zoom_zero_quadkey():
    ints = _cpu.quadkey_encode_vec(np.array([0]), np.array([0]), 0)
    assert _cpu.quadkey_ints_to_strings(ints, 0) == [""]
    assert _cpu.quadkey_strings_to_ints([""]) [0] == 0


def test_empty_inputs():
    assert _cpu.quadkey_ints_to_strings(np.array([], dtype=np.uint64), 7) == []
    assert _cpu.quadkey_strings_to_ints([]).shape == (0,)


def test_mixed_length_keys_rejected():
    with pytest.raises(ValueError, match="same length"):
        _cpu.quadkey_strings_to_ints(["0", "00"])
    with pytest.raises(ValueError, match="same length"):
        ct.quadkey_to_tile(["0", "00"])


def test_bad_digit_rejected():
    with pytest.raises(ValueError, match="0, 1, 2 or 3"):
        _cpu.quadkey_strings_to_ints(["9"])
    with pytest.raises(ValueError, match="0, 1, 2 or 3"):
        ct.quadkey_to_tile(["012x"])
    # scalar form validates too
    with pytest.raises(ValueError, match="0, 1, 2 or 3"):
        ct.quadkey_to_tile("9")


def test_non_string_keys_rejected():
    with pytest.raises(TypeError):
        _cpu.quadkey_strings_to_ints([1, 2])


def test_batch_zoom_over_32_rejected():
    with pytest.raises(ValueError):
        _cpu.quadkey_strings_to_ints(["0" * 33])


# ------------------------------------------------------------------ #
#  public API validation
# ------------------------------------------------------------------ #

def test_xy_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        ct.xy([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="same length"):
        ct.lnglat([1.0, 2.0, 3.0], [1.0, 2.0])


def test_tile_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        ct.tile([1.0, 2.0], [1.0], 7)


def test_tile_batch_zoom_bounds():
    with pytest.raises(ValueError, match="zoom"):
        ct.tile([1.0], [1.0], 31)
    with pytest.raises(ValueError, match="zoom"):
        ct.tile([1.0], [1.0], -1)
    with pytest.raises(TypeError):
        ct.tile([1.0], [1.0], "7")


def test_tile_pair_requires_y_and_zoom():
    with pytest.raises(ValueError, match="x and y"):
        ct.ul([1, 2, 3], zoom=7)
    with pytest.raises(ValueError, match="zoom is required"):
        ct.ul([1, 2], [3, 4])
    with pytest.raises(ValueError, match="zoom is required"):
        ct.bounds([1, 2], [3, 4])
    with pytest.raises(ValueError, match="x and y"):
        ct.bounds([1, 2, 3], zoom=7)


def test_xy_length_mismatch_tile_pair():
    with pytest.raises(ValueError, match="same length"):
        ct.ul([1, 2, 3], [1, 2], 7)
    with pytest.raises(ValueError, match="same length"):
        ct.parent([1, 2, 3], [1, 2], 7)


def test_bounding_tile_batch_requires_zoom():
    with pytest.raises(ValueError, match="zoom"):
        ct.bounding_tile([1.0], [2.0], [3.0], [4.0])


def test_bounding_tile_batch_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        ct.bounding_tile([1.0, 2.0], [1.0], [1.0, 2.0], [1.0], zoom=8)


def test_parent_target_zoom_validation():
    with pytest.raises(ValueError, match="to_zoom"):
        ct.parent([1], [2], 7, to_zoom=7)
    with pytest.raises(ValueError, match="to_zoom"):
        ct.parent([1], [2], 7, to_zoom=8)


def test_parent_tile_list_target_zoom_validation():
    tiles = [Tile(1, 1, 5), Tile(2, 2, 6)]
    with pytest.raises(ValueError, match="every input tile"):
        ct.parent(tiles, to_zoom=6)


def test_tile_list_rejects_extra_args():
    tiles = [Tile(60, 41, 7)]
    with pytest.raises(ValueError, match="list of Tile"):
        ct.ul(tiles, [41])
    with pytest.raises(ValueError, match="list of Tile"):
        ct.bounds(tiles, 7)
    with pytest.raises(ValueError, match="list of Tile"):
        ct.quadkey(tiles, zoom=7)
    with pytest.raises(ValueError, match="list of Tile"):
        ct.children(tiles, [1])
    with pytest.raises(ValueError, match="list of Tile"):
        ct.neighbors(tiles, zoom=7)


def test_non_numeric_input_rejected():
    with pytest.raises(TypeError):
        ct.ul(["a", "b"], [1, 2], 7)
    with pytest.raises((TypeError, ValueError)):
        # mixed Tile + int list is not a tile list, and not numeric either
        ct.ul([Tile(1, 1, 7), 5], [1, 2], 7)


def test_packed_quadkey_requires_zoom():
    ints = np.array([5], dtype=np.uint64)
    with pytest.raises(ValueError, match="zoom is required"):
        ct.quadkey_to_tile(ints)
