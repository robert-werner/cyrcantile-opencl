"""Vulkan (via wgpu) backend.

Provides the same batch operations as the OpenCL backend but computes
them with Vulkan compute shaders written in WGSL.  The shaders are
translated to SPIR-V by wgpu's bundled Naga compiler, so no external
glslc / glslang toolchain is required.

Unlike the OpenCL backend this module is imported lazily: ``wgpu`` and
the Vulkan ICD are only loaded when a Vulkan device is actually
requested, so a missing ``wgpu`` installation or missing Vulkan driver
simply means ``HAS_VULKAN`` stays ``False`` and callers fall back
gracefully.

All WGSL shaders use f32 storage buffers (f64 requires the shader-f64
feature which many devices lack).  The precision is sufficient for tile
operations through zoom 20; see the README for the llvmpipe caveat.
"""

from __future__ import annotations

import numpy as np

try:
    import wgpu
    try:
        import wgpu.backends.auto  # noqa: F401 - old wgpu needs it; new one does not
    except ImportError:
        pass  # modern wgpu selects the native backend automatically
    HAS_WGPU = True
except Exception:
    HAS_WGPU = False

try:
    HAS_VULKAN = False
    if HAS_WGPU:
        _adapters = wgpu.gpu.enumerate_adapters_sync()
        _vk = [a for a in _adapters if a.info["backend_type"] == "Vulkan"]
        _adapter = _vk[0] if _vk else None
        if _adapter is not None:
            HAS_VULKAN = True
except Exception:
    _adapter = None
    HAS_VULKAN = False

# Scalar parameters are packed into a single uniform buffer bound at 20.
_UNIFORM_BINDING = 20

_UNIFORM_PREAMBLE = (
    f"@group(0) @binding({_UNIFORM_BINDING}) var<uniform> u: array<vec4<i32>, 2>;\n"
)

# The compute grid is 2D: each row covers this many threads (a non-zero
# multiple of the 256-invocation workgroup size).  Keeping `x` well under
# the 65535-workgroup-per-dimension limit means even 100M+ element batches
# can be dispatched as several rows of fixed width instead of one huge row.
_ROW_THREADS = 1 << 20  # 1,048,576 threads == 4096 workgroups of 256
_ROW_WORKGROUPS = _ROW_THREADS // 256

# Baked into every kernel so the flat index is `gid.y * _ROW_THREADS + gid.x`.
_FLAT_IDX_SRC = (
    f"const FLAT_STRIDE: u32 = {_ROW_THREADS}u;\n"
    "fn flat_id(gid: vec3<u32>) -> i32 { return i32(gid.x + gid.y * FLAT_STRIDE); }\n"
)

_PI = 3.14159265358979323846
_HALF_PI = 1.57079632679489661923
_QUARTER_PI = 0.78539816339744830962
_R2D = 57.29577951308232
_D2R = 0.017453292519943295
_RE = 6378137.0
_CE = 40075016.68557849
_MAX_LAT = 85.05112878
_MAX_LNG = 180.0
# NOTE: at f32 precision x + EPSILON == x, so the mercantile nudge is a
# no-op here; kept only for structural parity with the other backends.
_EPSILON = 1e-14


def _clamp_src():
    return (
        f"fn clamp_lat(p: f32) -> f32 {{ if p > f32({_MAX_LAT}) {{ return f32({_MAX_LAT}); }} "
        f"if p < f32(-{_MAX_LAT}) {{ return f32(-{_MAX_LAT}); }} return p; }}\n"
        f"fn clamp_lng(l: f32) -> f32 {{ if l > f32({_MAX_LNG}) {{ return f32({_MAX_LNG}); }} "
        f"if l < f32(-{_MAX_LNG}) {{ return f32(-{_MAX_LNG}); }} return l; }}\n"
    )


def _tile_idx_src() -> str:
    """Mercantile-compatible tile indices: clamp + EPSILON nudge."""
    return (
        "fn tile_idx_x(l: f32, z2: f32) -> i32 {\n"
        "    let x = l / 360.0 + 0.5;\n"
        "    if (x <= 0.0) { return 0; }\n"
        "    if (x >= 1.0) { return i32(z2) - 1; }\n"
        "    return i32(floor((x + __EPS__) * z2));\n"
        "}\n"
        "fn tile_idx_y(p: f32, z2: f32) -> i32 {\n"
        "    let sinlat = sin(p * __D2R__);\n"
        "    let y = 0.5 - 0.25 * log((1.0 + sinlat) / (1.0 - sinlat)) / __PI__;\n"
        "    if (y <= 0.0) { return 0; }\n"
        "    if (y >= 1.0) { return i32(z2) - 1; }\n"
        "    return i32(floor((y + __EPS__) * z2));\n"
        "}\n"
    ).replace("__D2R__", str(_D2R)).replace("__PI__", str(_PI)).replace("__EPS__", str(_EPSILON))


# Per-kernel WGSL.  Uniform slots (u[i]) are documented per kernel.
_WGSL = {
    "xy": (
        _UNIFORM_PREAMBLE
        + _FLAT_IDX_SRC
        + _clamp_src()
        + """
@group(0) @binding(0) var<storage, read> lng: array<f32>;
@group(0) @binding(1) var<storage, read> lat: array<f32>;
@group(0) @binding(2) var<storage, read_write> ox: array<f32>;
@group(0) @binding(3) var<storage, read_write> oy: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = flat_id(gid);
    if (i >= u[0].y) { return; }
    var l = lng[i];
    var p = lat[i];
    if (u[0].x != 0) { l = clamp_lng(l); p = clamp_lat(p); }
    p = p * __D2R__;
    ox[i] = __RE__ * l * __D2R__;
    oy[i] = __RE__ * log(tan(__QUARTER_PI__ + p * 0.5));
}
""".replace("__D2R__", str(_D2R)).replace("__RE__", str(_RE)).replace("__QUARTER_PI__", str(_QUARTER_PI))
    ),
    "lnglat": (
        _UNIFORM_PREAMBLE
        + _FLAT_IDX_SRC
        + """
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> olng: array<f32>;
@group(0) @binding(3) var<storage, read_write> olat: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = flat_id(gid);
    if (i >= u[0].x) { return; }
    olng[i] = (x[i] / __RE__) * __R2D__;
    olat[i] = (2.0 * atan(exp(y[i] / __RE__)) - __HALF_PI__) * __R2D__;
}
""".replace("__RE__", str(_RE)).replace("__R2D__", str(_R2D)).replace("__HALF_PI__", str(_HALF_PI))
    ),
    "tile": (
        _UNIFORM_PREAMBLE
        + _FLAT_IDX_SRC
        + _clamp_src()
        + _tile_idx_src()
        + """
@group(0) @binding(0) var<storage, read> lng: array<f32>;
@group(0) @binding(1) var<storage, read> lat: array<f32>;
@group(0) @binding(2) var<storage, read_write> ox: array<i32>;
@group(0) @binding(3) var<storage, read_write> oy: array<i32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = flat_id(gid);
    if (i >= u[0].z) { return; }
    var l = lng[i];
    var p = lat[i];
    if (u[0].y != 0) { l = clamp_lng(l); p = clamp_lat(p); }
    let z2 = exp2(f32(u[0].x));
    ox[i] = tile_idx_x(l, z2);
    oy[i] = tile_idx_y(p, z2);
}
"""
    ),
    "ul": (
        _UNIFORM_PREAMBLE
        + _FLAT_IDX_SRC
        + """
@group(0) @binding(0) var<storage, read> tx: array<i32>;
@group(0) @binding(1) var<storage, read> ty: array<i32>;
@group(0) @binding(2) var<storage, read_write> olng: array<f32>;
@group(0) @binding(3) var<storage, read_write> olat: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = flat_id(gid);
    if (i >= u[0].y) { return; }
    let z2 = exp2(f32(u[0].x));
    olng[i] = f32(tx[i]) / z2 * 360.0 - 180.0;
    olat[i] = atan(sinh(__PI__ * (1.0 - 2.0 * f32(ty[i]) / z2))) * __R2D__;
}
""".replace("__PI__", str(_PI)).replace("__R2D__", str(_R2D))
    ),
    "bounds": (
        _UNIFORM_PREAMBLE
        + _FLAT_IDX_SRC
        + """
@group(0) @binding(0) var<storage, read> tx: array<i32>;
@group(0) @binding(1) var<storage, read> ty: array<i32>;
@group(0) @binding(2) var<storage, read_write> ow: array<f32>;
@group(0) @binding(3) var<storage, read_write> os: array<f32>;
@group(0) @binding(4) var<storage, read_write> oe: array<f32>;
@group(0) @binding(5) var<storage, read_write> on: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = flat_id(gid);
    if (i >= u[0].y) { return; }
    let z2 = exp2(f32(u[0].x));
    ow[i] = f32(tx[i]) / z2 * 360.0 - 180.0;
    oe[i] = f32(tx[i] + 1) / z2 * 360.0 - 180.0;
    on[i] = atan(sinh(__PI__ * (1.0 - 2.0 * f32(ty[i]) / z2))) * __R2D__;
    os[i] = atan(sinh(__PI__ * (1.0 - 2.0 * (f32(ty[i]) + 1.0) / z2))) * __R2D__;
}
""".replace("__PI__", str(_PI)).replace("__R2D__", str(_R2D))
    ),
    "xy_bounds": (
        _UNIFORM_PREAMBLE
        + _FLAT_IDX_SRC
        + """
@group(0) @binding(0) var<storage, read> tx: array<i32>;
@group(0) @binding(1) var<storage, read> ty: array<i32>;
@group(0) @binding(2) var<storage, read_write> ol: array<f32>;
@group(0) @binding(3) var<storage, read_write> ob: array<f32>;
@group(0) @binding(4) var<storage, read_write> orr: array<f32>;
@group(0) @binding(5) var<storage, read_write> ot: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = flat_id(gid);
    if (i >= u[0].y) { return; }
    let z2 = exp2(f32(u[0].x));
    let ts = __CE__ / z2;
    ol[i] = f32(tx[i]) * ts - __CE__ * 0.5;
    orr[i] = ol[i] + ts;
    ot[i] = __CE__ * 0.5 - f32(ty[i]) * ts;
    ob[i] = ot[i] - ts;
}
""".replace("__CE__", str(_CE))
    ),
    # NOTE: quadkey kernels require the shader-int64 device feature; the
    # backend raises NotImplementedError instead of dispatching when the
    # feature is missing.  Abstract-int literals ("0", "1", "3") are used
    # deliberately so they adopt the u64 type of the other operand.
    "quadkey_encode": (
        _UNIFORM_PREAMBLE
        + _FLAT_IDX_SRC
        + """
@group(0) @binding(0) var<storage, read> tx: array<i32>;
@group(0) @binding(1) var<storage, read> ty: array<i32>;
@group(0) @binding(2) var<storage, read_write> qk: array<u64>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = flat_id(gid);
    if (i >= u[0].y) { return; }
    let zoom = u[0].x;
    var x: u64 = u64(tx[i]);
    var y: u64 = u64(ty[i]);
    var q: u64 = 0;
    var s: i32 = 2 * (zoom - 1);
    var j: i32 = zoom - 1;
    while (j >= 0) {
        let d = ((x >> u32(j)) & 1) | (((y >> u32(j)) & 1) << 1);
        q = q | (d << u32(s));
        s = s - 2;
        j = j - 1;
    }
    qk[i] = q;
}
"""
    ),
    "quadkey_decode": (
        _UNIFORM_PREAMBLE
        + _FLAT_IDX_SRC
        + """
@group(0) @binding(0) var<storage, read> qk: array<u64>;
@group(0) @binding(1) var<storage, read_write> ox: array<i32>;
@group(0) @binding(2) var<storage, read_write> oy: array<i32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = flat_id(gid);
    if (i >= u[0].y) { return; }
    let zoom = u[0].x;
    let q = qk[i];
    var x: u64 = 0;
    var y: u64 = 0;
    var j: i32 = 0;
    while (j < zoom) {
        let k = zoom - 1 - j;
        let sh = u32(2 * k);
        let bk = u32(k);
        if (((q >> sh) & 1) != 0) { x = x | (u64(1) << bk); }
        if (((q >> sh) & 2) != 0) { y = y | (u64(1) << bk); }
        j = j + 1;
    }
    ox[i] = i32(x);
    oy[i] = i32(y);
}
"""
    ),
    # u[0].x = shift (0 = identity, 1 = immediate parent, ...)
    "parent": (
        _UNIFORM_PREAMBLE
        + _FLAT_IDX_SRC
        + """
@group(0) @binding(0) var<storage, read> tx: array<i32>;
@group(0) @binding(1) var<storage, read> ty: array<i32>;
@group(0) @binding(2) var<storage, read_write> ox: array<i32>;
@group(0) @binding(3) var<storage, read_write> oy: array<i32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = flat_id(gid);
    if (i >= u[0].y) { return; }
    let s = u32(u[0].x);
    if (s == 0u) {
        ox[i] = tx[i]; oy[i] = ty[i];
    } else {
        ox[i] = tx[i] >> s; oy[i] = ty[i] >> s;
    }
}
"""
    ),
    "children": (
        _UNIFORM_PREAMBLE
        + _FLAT_IDX_SRC
        + """
@group(0) @binding(0) var<storage, read> tx: array<i32>;
@group(0) @binding(1) var<storage, read> ty: array<i32>;
@group(0) @binding(2) var<storage, read_write> ox: array<i32>;
@group(0) @binding(3) var<storage, read_write> oy: array<i32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = flat_id(gid);
    if (i >= u[0].x) { return; }
    let x2 = tx[i] << 1;
    let y2 = ty[i] << 1;
    let k = i * 4;
    // mercantile order: top-left, top-right, bottom-right, bottom-left
    ox[k]     = x2;     oy[k]     = y2;
    ox[k + 1] = x2 + 1; oy[k + 1] = y2;
    ox[k + 2] = x2 + 1; oy[k + 2] = y2 + 1;
    ox[k + 3] = x2;     oy[k + 3] = y2 + 1;
}
"""
    ),
    # x-major order, unfiltered (the scalar API filters invalid neighbours)
    "neighbors": (
        _UNIFORM_PREAMBLE
        + _FLAT_IDX_SRC
        + """
@group(0) @binding(0) var<storage, read> tx: array<i32>;
@group(0) @binding(1) var<storage, read> ty: array<i32>;
@group(0) @binding(2) var<storage, read_write> ox: array<i32>;
@group(0) @binding(3) var<storage, read_write> oy: array<i32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = flat_id(gid);
    if (i >= u[0].x) { return; }
    let x = tx[i];
    let y = ty[i];
    let k = i * 8;
    ox[k]     = x - 1; oy[k]     = y - 1;
    ox[k + 1] = x - 1; oy[k + 1] = y;
    ox[k + 2] = x - 1; oy[k + 2] = y + 1;
    ox[k + 3] = x;     oy[k + 3] = y - 1;
    ox[k + 4] = x;     oy[k + 4] = y + 1;
    ox[k + 5] = x + 1; oy[k + 5] = y - 1;
    ox[k + 6] = x + 1; oy[k + 6] = y;
    ox[k + 7] = x + 1; oy[k + 7] = y + 1;
}
"""
    ),
    # Only west/north are needed: the bounding tile at a given zoom is the
    # tile of the bbox north-west corner (mercantile semantics).
    "bounding_tile": (
        _UNIFORM_PREAMBLE
        + _FLAT_IDX_SRC
        + _clamp_src()
        + _tile_idx_src()
        + """
@group(0) @binding(0) var<storage, read> west: array<f32>;
@group(0) @binding(1) var<storage, read> north: array<f32>;
@group(0) @binding(2) var<storage, read_write> ox: array<i32>;
@group(0) @binding(3) var<storage, read_write> oy: array<i32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = flat_id(gid);
    if (i >= u[0].y) { return; }
    let z2 = exp2(f32(u[0].x));
    ox[i] = tile_idx_x(west[i], z2);
    oy[i] = tile_idx_y(clamp_lat(north[i]), z2);
}
"""
    ),
    "tiles_in_bbox": (
        _UNIFORM_PREAMBLE
        + _FLAT_IDX_SRC
        + """
@group(0) @binding(0) var<storage, read_write> ox: array<i32>;
@group(0) @binding(1) var<storage, read_write> oy: array<i32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = flat_id(gid);
    if (i >= u[0].w) { return; }
    ox[i] = u[0].x + i % u[0].z;
    oy[i] = u[0].y + i / u[0].z;
}
"""
    ),
}


def _require_u64(method_name):
    raise NotImplementedError(
        f"Vulkan device {_adapter.info['device'] if _adapter else ''!r} "
        f"lacks the shader-int64 feature required by {method_name}(); "
        f"use the OpenCL/CUDA backend or the CPU fallback"
    )


class VulkanBackend:
    """Lazily-initialised singleton wrapping a Vulkan device + cached pipelines."""

    _instance: VulkanBackend | None = None

    def __new__(cls):
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._init()
            cls._instance = obj
        return cls._instance

    def _init(self):
        if not HAS_VULKAN or _adapter is None:
            raise RuntimeError("No Vulkan (wgpu) adapter available")
        features = set(_adapter.features)
        self.has_u64 = "shader-int64" in features
        required = set()
        if self.has_u64:
            required.add("shader-int64")
        self.device = _adapter.request_device_sync(
            required_features=required,
            label="cyrcantile-vulkan",
        )
        self._pipes = {}

    # -- helpers ---------------------------------------------------

    def _pipe(self, key):
        pipe = self._pipes.get(key)
        if pipe is not None:
            return pipe
        wgsl = _WGSL[key]
        shader = self.device.create_shader_module(code=wgsl)
        pipe = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": shader, "entry_point": "main"},
        )
        self._pipes[key] = pipe
        return pipe

    def _dispatch(self, key, buffers, n):
        """buffers: list of 3-tuples (binding, kind, gpubuffer)."""
        pipe = self._pipe(key)
        bindings = []
        for binding, _kind, buf in buffers:
            resource = {"buffer": buf, "offset": 0, "size": buf.size}
            bindings.append({"binding": binding, "resource": resource})
        bind_group = self.device.create_bind_group(
            layout=pipe.get_bind_group_layout(0), entries=bindings
        )
        # Cover `n` threads with a 2D grid of fixed-width rows.  Each row is
        # _ROW_WORKGROUPS (4096) workgroups of 256 threads (= _ROW_THREADS
        # invocations); extra rows add 1 to the y dimension, which stays well
        # within the per-dimension 65535 workgroup limit for realistic sizes.
        rows = (n + _ROW_THREADS - 1) // _ROW_THREADS
        encoder = self.device.create_command_encoder()
        pass_enc = encoder.begin_compute_pass()
        pass_enc.set_pipeline(pipe)
        pass_enc.set_bind_group(0, bind_group)
        pass_enc.dispatch_workgroups(_ROW_WORKGROUPS, rows, 1)
        pass_enc.end()
        self.device.queue.submit([encoder.finish()])

    def _upload(self, arr):
        return self.device.create_buffer_with_data(
            data=arr, usage=wgpu.BufferUsage.STORAGE
        )

    def _out(self, nbytes):
        return self.device.create_buffer(
            size=nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )

    def _uni(self, values):
        data = np.zeros(8, dtype=np.int32)
        data[: len(values)] = values
        return self.device.create_buffer_with_data(
            data=data, usage=wgpu.BufferUsage.UNIFORM
        )

    def _read(self, buf, dtype, count):
        m = self.device.queue.read_buffer(buf)
        return np.frombuffer(m, dtype=dtype, count=count).copy()

    # -- batch operations ------------------------------------------

    def xy(self, lng, lat, truncate=False):
        lng = np.ascontiguousarray(lng, np.float32)
        lat = np.ascontiguousarray(lat, np.float32)
        n = lng.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        bl = self._upload(lng)
        ba = self._upload(lat)
        bo = self._out(n * 4)
        by = self._out(n * 4)
        bu = self._uni([1 if truncate else 0, n])
        self._dispatch("xy", [(0, 0, bl), (1, 0, ba), (2, 0, bo), (3, 0, by),
                              (20, 0, bu)], n)
        return (self._read(bo, np.float32, n).astype(np.float64),
                self._read(by, np.float32, n).astype(np.float64))

    def lnglat(self, x, y):
        x = np.ascontiguousarray(x, np.float32)
        y = np.ascontiguousarray(y, np.float32)
        n = x.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        bx = self._upload(x)
        by = self._upload(y)
        bo = self._out(n * 4)
        bl = self._out(n * 4)
        bu = self._uni([n])
        self._dispatch("lnglat", [(0, 0, bx), (1, 0, by), (2, 0, bo), (3, 0, bl),
                                  (20, 0, bu)], n)
        return (self._read(bo, np.float32, n).astype(np.float64),
                self._read(bl, np.float32, n).astype(np.float64))

    def tile(self, lng, lat, zoom, truncate=False):
        lng = np.ascontiguousarray(lng, np.float32)
        lat = np.ascontiguousarray(lat, np.float32)
        n = lng.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        bl = self._upload(lng)
        ba = self._upload(lat)
        bo = self._out(n * 4)
        by = self._out(n * 4)
        bu = self._uni([zoom, 1 if truncate else 0, n])
        self._dispatch("tile", [(0, 0, bl), (1, 0, ba), (2, 0, bo), (3, 0, by),
                                (20, 0, bu)], n)
        return self._read(bo, np.int32, n), self._read(by, np.int32, n)

    def ul(self, tx, ty, zoom):
        tx = np.ascontiguousarray(tx, np.int32)
        ty = np.ascontiguousarray(ty, np.int32)
        n = tx.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        bx = self._upload(tx)
        by = self._upload(ty)
        bo = self._out(n * 4)
        bl = self._out(n * 4)
        bu = self._uni([zoom, n])
        self._dispatch("ul", [(0, 0, bx), (1, 0, by), (2, 0, bo), (3, 0, bl),
                              (20, 0, bu)], n)
        return (self._read(bo, np.float32, n).astype(np.float64),
                self._read(bl, np.float32, n).astype(np.float64))

    def bounds(self, tx, ty, zoom):
        tx = np.ascontiguousarray(tx, np.int32)
        ty = np.ascontiguousarray(ty, np.int32)
        n = tx.shape[0]
        if n == 0:
            return tuple(np.empty(0, dtype=np.float64) for _ in range(4))
        bx = self._upload(tx)
        by = self._upload(ty)
        bw = self._out(n * 4)
        bs = self._out(n * 4)
        be = self._out(n * 4)
        bn = self._out(n * 4)
        bu = self._uni([zoom, n])
        self._dispatch("bounds", [(0, 0, bx), (1, 0, by), (2, 0, bw),
                                  (3, 0, bs), (4, 0, be), (5, 0, bn),
                                  (20, 0, bu)], n)
        return tuple(self._read(b, np.float32, n).astype(np.float64)
                     for b in (bw, bs, be, bn))

    def xy_bounds(self, tx, ty, zoom):
        tx = np.ascontiguousarray(tx, np.int32)
        ty = np.ascontiguousarray(ty, np.int32)
        n = tx.shape[0]
        if n == 0:
            return tuple(np.empty(0, dtype=np.float64) for _ in range(4))
        bx = self._upload(tx)
        by = self._upload(ty)
        bl = self._out(n * 4)
        bb = self._out(n * 4)
        br = self._out(n * 4)
        bt = self._out(n * 4)
        bu = self._uni([zoom, n])
        self._dispatch("xy_bounds", [(0, 0, bx), (1, 0, by), (2, 0, bl),
                                     (3, 0, bb), (4, 0, br), (5, 0, bt),
                                     (20, 0, bu)], n)
        return tuple(self._read(b, np.float32, n).astype(np.float64)
                     for b in (bl, bb, br, bt))

    def quadkey_encode(self, tx, ty, zoom):
        if not self.has_u64:
            _require_u64("quadkey_encode")
        tx = np.ascontiguousarray(tx, np.int32)
        ty = np.ascontiguousarray(ty, np.int32)
        n = tx.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.uint64)
        bx = self._upload(tx)
        by = self._upload(ty)
        bq = self._out(n * 8)
        bu = self._uni([zoom, n])
        self._dispatch("quadkey_encode", [(0, 0, bx), (1, 0, by), (2, 0, bq),
                                          (20, 0, bu)], n)
        return self._read(bq, np.uint64, n)

    def quadkey_decode(self, qk, zoom):
        if not self.has_u64:
            _require_u64("quadkey_decode")
        qk = np.ascontiguousarray(qk, np.uint64)
        n = qk.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        bq = self._upload(qk)
        bo = self._out(n * 4)
        by = self._out(n * 4)
        bu = self._uni([zoom, n])
        self._dispatch("quadkey_decode", [(0, 0, bq), (1, 0, bo), (2, 0, by),
                                          (20, 0, bu)], n)
        return self._read(bo, np.int32, n), self._read(by, np.int32, n)

    def parent(self, tx, ty, shift):
        """Ancestor of each tile at ``shift`` zoom levels up."""
        tx = np.ascontiguousarray(tx, np.int32)
        ty = np.ascontiguousarray(ty, np.int32)
        n = tx.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        bx = self._upload(tx)
        by = self._upload(ty)
        bo = self._out(n * 4)
        byy = self._out(n * 4)
        bu = self._uni([shift, n])
        self._dispatch("parent", [(0, 0, bx), (1, 0, by), (2, 0, bo), (3, 0, byy),
                                  (20, 0, bu)], n)
        return self._read(bo, np.int32, n), self._read(byy, np.int32, n)

    def children(self, tx, ty):
        tx = np.ascontiguousarray(tx, np.int32)
        ty = np.ascontiguousarray(ty, np.int32)
        n = tx.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        bx = self._upload(tx)
        by = self._upload(ty)
        bo = self._out(n * 4 * 4)
        byy = self._out(n * 4 * 4)
        bu = self._uni([n])
        self._dispatch("children", [(0, 0, bx), (1, 0, by), (2, 0, bo), (3, 0, byy),
                                    (20, 0, bu)], n)
        return self._read(bo, np.int32, n * 4), self._read(byy, np.int32, n * 4)

    def neighbors(self, tx, ty):
        tx = np.ascontiguousarray(tx, np.int32)
        ty = np.ascontiguousarray(ty, np.int32)
        n = tx.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        bx = self._upload(tx)
        by = self._upload(ty)
        bo = self._out(n * 8 * 4)
        byy = self._out(n * 8 * 4)
        bu = self._uni([n])
        self._dispatch("neighbors", [(0, 0, bx), (1, 0, by), (2, 0, bo), (3, 0, byy),
                                     (20, 0, bu)], n)
        return self._read(bo, np.int32, n * 8), self._read(byy, np.int32, n * 8)

    def bounding_tile(self, west, north, zoom):
        """Tile of the bbox north-west corner at *zoom* (only w/n are used)."""
        west = np.ascontiguousarray(west, np.float32)
        north = np.ascontiguousarray(north, np.float32)
        n = west.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        bw = self._upload(west)
        bn = self._upload(north)
        bo = self._out(n * 4)
        by = self._out(n * 4)
        bu = self._uni([zoom, n])
        self._dispatch("bounding_tile", [(0, 0, bw), (1, 0, bn),
                                         (2, 0, bo), (3, 0, by),
                                         (20, 0, bu)], n)
        return self._read(bo, np.int32, n), self._read(by, np.int32, n)

    def tiles_in_bbox(self, min_x, min_y, grid_w, grid_h):
        n = grid_w * grid_h
        bo = self._out(n * 4)
        by = self._out(n * 4)
        bu = self._uni([min_x, min_y, grid_w, n])
        self._dispatch("tiles_in_bbox", [(0, 0, bo), (1, 0, by), (20, 0, bu)], n)
        return self._read(bo, np.int32, n), self._read(by, np.int32, n)


def get_backend():
    """Return the singleton ``VulkanBackend`` or ``None`` if unavailable."""
    if not HAS_VULKAN:
        return None
    try:
        return VulkanBackend()
    except Exception:
        return None
