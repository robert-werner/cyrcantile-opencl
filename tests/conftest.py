"""Shared test fixtures & data helpers."""

from __future__ import annotations

import numpy as np
import pytest

from cyrcantile._types import Tile


@pytest.fixture(scope="session")
def rng():
    return np.random.RandomState(42)


@pytest.fixture(scope="session")
def points(rng):
    """5,000 in-range geographic points (deterministic)."""
    lngs = rng.uniform(-179.9, 179.9, 5000)
    lats = rng.uniform(-84.9, 84.9, 5000)
    return lngs, lats


@pytest.fixture(scope="session")
def merc(rng):
    """5,000 in-range Web-Mercator points (deterministic)."""
    xs = rng.uniform(-2.0e7, 2.0e7, 5000)
    ys = rng.uniform(-2.0e7, 2.0e7, 5000)
    return xs, ys


def random_tiles(n, zoom, seed=42):
    """Deterministic random valid tiles at *zoom*."""
    r = np.random.RandomState(seed)
    lim = 1 << zoom
    xs = r.randint(0, lim, n)
    ys = r.randint(0, lim, n)
    return (xs.astype(np.int32), ys.astype(np.int32),
            [Tile(int(x), int(y), zoom) for x, y in zip(xs, ys)])
