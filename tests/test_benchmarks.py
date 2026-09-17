"""Smoke test: benchmarks/batch.py must run cleanly on small sizes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "batch.py"


def test_batch_benchmark_smoke():
    result = subprocess.run(
        [
            sys.executable, str(BENCH),
            "--coords", "20000", "--tiles", "20000",
            "--repeat", "1", "--no-public-api", "--backends", "cpu,cpu-1",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # every batch operation is timed and verified
    for op in ("tile", "xy", "lnglat", "bounding_tile", "ul", "bounds",
               "xy_bounds", "quadkey_encode", "quadkey_decode", "parent",
               "children", "neighbors"):
        assert op in result.stdout
    assert "verification: all backends match" in result.stdout
