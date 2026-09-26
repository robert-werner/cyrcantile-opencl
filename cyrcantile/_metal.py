"""Metal backend for Apple GPUs, implemented through ``wgpu``.

The compute implementation is shared with :mod:`cyrcantile._vulkan`:
both APIs consume the same WGSL shaders and the same WebGPU buffer /
pipeline code.  ``wgpu`` selects its native Metal backend on macOS, so no
Vulkan ICD, Metal shader compiler invocation, or platform-specific binary
is needed.

This backend is opt-in, like Vulkan and CUDA.  Obtain it with
``get_backend()`` and invoke its batch methods directly.  It intentionally
does not replace the public API's OpenCL/CPU auto-dispatch policy.

WGSL storage buffers use ``f32``.  Consequently, floating outputs have
lower precision than the OpenCL/CUDA ``f64`` backends, and tile indices can
differ by one exactly at a floor boundary.
"""

from __future__ import annotations

from . import _vulkan as _wgpu_impl


def _pick_metal_adapter():
    """Return the first native Metal adapter, or ``None`` when unavailable."""
    if not _wgpu_impl.HAS_WGPU:
        return None
    try:
        adapters = _wgpu_impl.wgpu.gpu.enumerate_adapters_sync()
        return next(
            (adapter for adapter in adapters if adapter.info["backend_type"] == "Metal"),
            None,
        )
    except Exception:
        return None


_adapter = _pick_metal_adapter()
HAS_METAL = _adapter is not None


def _require_metal_u64(method_name):
    device = _adapter.info["device"] if _adapter is not None else ""
    raise NotImplementedError(
        f"Metal device {device!r} lacks the shader-int64 feature required by "
        f"{method_name}(); use the OpenCL/CUDA backend or the CPU fallback"
    )


class MetalBackend(_wgpu_impl.VulkanBackend):
    """Lazily initialised singleton for the native macOS Metal adapter."""

    _instance: MetalBackend | None = None

    def _init(self):
        if not HAS_METAL or _adapter is None:
            raise RuntimeError("No Metal (wgpu) adapter available")
        self.has_u64 = "shader-int64" in set(_adapter.features)
        required = {"shader-int64"} if self.has_u64 else set()
        self.device = _adapter.request_device_sync(
            required_features=required,
            label="cyrcantile-metal",
        )
        self._pipes = {}

    def quadkey_encode(self, tx, ty, zoom):
        if not self.has_u64:
            _require_metal_u64("quadkey_encode")
        return super().quadkey_encode(tx, ty, zoom)

    def quadkey_decode(self, qk, zoom):
        if not self.has_u64:
            _require_metal_u64("quadkey_decode")
        return super().quadkey_decode(qk, zoom)


def get_backend():
    """Return the singleton ``MetalBackend`` or ``None`` if unavailable."""
    if not HAS_METAL:
        return None
    try:
        return MetalBackend()
    except Exception:
        return None
