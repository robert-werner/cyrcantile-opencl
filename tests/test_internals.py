"""Structural guards that don't need GPU hardware.

Catches the "backend method references a deleted helper" bug class
(e.g. CudaBackend._launch calling the removed self._blocks) on machines
where the backend cannot even initialise, plus API-skew between the
OpenCL / Vulkan / CUDA backends.
"""

from __future__ import annotations

import importlib
import inspect
import re

BACKEND_CLASSES = [
    ("_backend", "OpenCLBackend"),
    ("_cuda", "CudaBackend"),
    ("_vulkan", "VulkanBackend"),
]

BACKEND_METHODS = [
    "xy", "lnglat", "tile", "ul", "bounds", "xy_bounds",
    "quadkey_encode", "quadkey_decode", "parent", "children",
    "neighbors", "bounding_tile", "tiles_in_bbox",
]


def _get_class(module_name, class_name):
    mod = importlib.import_module(f"cyrcantile.{module_name}")
    return getattr(mod, class_name)


def test_backend_methods_reference_defined_members():
    for module_name, class_name in BACKEND_CLASSES:
        cls = _get_class(module_name, class_name)
        src = inspect.getsource(cls)
        defined = set(re.findall(r"def (\w+)", src))
        referenced = set(re.findall(r"self\.(\w+)\(", src))
        missing = sorted(r for r in referenced if r not in defined)
        assert not missing, f"{class_name} calls undefined members: {missing}"


def test_backends_expose_common_api():
    for module_name, class_name in BACKEND_CLASSES:
        cls = _get_class(module_name, class_name)
        for method in BACKEND_METHODS:
            assert callable(getattr(cls, method, None)), (
                f"{class_name}.{method} is missing — the backends must "
                f"stay API-compatible"
            )
