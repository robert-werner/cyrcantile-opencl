"""Vulkan (via wgpu) backend.

Provides the same batch operations as the OpenCL backend but computes
them with Vulkan compute shaders written in WGSL.  The shaders are
translated to SPIR-V by wgpu's bundled Naga compiler, so no external
glslc / glslang toolchain is required.

Unlike the OpenCL backend this module is imported lazily: ``wgpu`` and the
Vulkan ICD are only loaded when a Vulkan device is actually requested, so a
missing ``wgpu`` installation or missing AMD/Mesa Vulkan driver simply means
``HAS_VULKAN`` stays ``False`` and the public API falls back gracefully.

AMD GPUs are exposed through Mesa's ``RADV`` (``libvulkan_radeon``) on
Linux; any other Vulkan ICD is equally supported.
"""

from __future__ import annotations

import numpy as np

try:
    import wgpu
    import wgpu.backends.auto  # noqa: F401 - loads the native backend
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
    "@group(0) @binding(%d) var<uniform> u: array<vec4<i32>, 2>;\n" % _UNIFORM_BINDING
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


def _clamp_src():
    return (
        f"fn clamp_lat(p: f64) -> f64 {{ if p > {_MAX_LAT} {{ return {_MAX_LAT}; }} "
        f"if p < -{_MAX_LAT} {{ return -{_MAX_LAT}; }} return p; }}\n"
        f"fn clamp_lng(l: f64) -> f64 {{ if l > {_MAX_LNG} {{ return {_MAX_LNG}; }} "
        f"if l < -{_MAX_LNG} {{ return -{_MAX_LNG}; }} return l; }}\n"
    )


def _tile_y(p: str) -> str:
    return f"(1.0 - log(tan({_QUARTER_PI} + {p} * 0.5)) / {_PI}) * 0.5"


# Per-kernel WGSL.  Uniform slots (u[i]) are documented per kernel.
_WGSL = {
    "xy": (
        _UNIFORM_PREAMBLE
        + _clamp_src()
        + """
@group(0) @binding(0) var<storage, read> lng: array<f64>;
@group(0) @binding(1) var<storage, read> lat: array<f64>;
@group(0) @binding(2) var<storage, read_write> ox: array<f64>;
@group(0) @binding(3) var<storage, read_write> oy: array<f64>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
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
        + """
@group(0) @binding(0) var<storage, read> x: array<f64>;
@group(0) @binding(1) var<storage, read> y: array<f64>;
@group(0) @binding(2) var<storage, read_write> olng: array<f64>;
@group(0) @binding(3) var<storage, read_write> olat: array<f64>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    if (i >= u[0].x) { return; }
    olng[i] = (x[i] / __RE__) * __R2D__;
    olat[i] = (2.0 * atan(exp(y[i] / __RE__)) - __HALF_PI__) * __R2D__;
}
""".replace("__RE__", str(_RE)).replace("__R2D__", str(_R2D)).replace("__HALF_PI__", str(_HALF_PI))
    ),
    "tile": (
        _UNIFORM_PREAMBLE
        + _clamp_src()
        + """
@group(0) @binding(0) var<storage, read> lng: array<f64>;
@group(0) @binding(1) var<storage, read> lat: array<f64>;
@group(0) @binding(2) var<storage, read_write> ox: array<i32>;
@group(0) @binding(3) var<storage, read_write> oy: array<i32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    if (i >= u[0].z) { return; }
    var l = lng[i];
    var p = lat[i];
    if (u[0].y != 0) { l = clamp_lng(l); p = clamp_lat(p); }
    p = p * __D2R__;
    let z2 = exp2(f64(u[0].x));
    ox[i] = i32(floor((l + 180.0) / 360.0 * z2));
    oy[i] = i32(floor(__TY__ * z2));
}
""".replace("__D2R__", str(_D2R)).replace("__TY__", _tile_y("p"))
    ),
    "ul": (
        _UNIFORM_PREAMBLE
        + """
@group(0) @binding(0) var<storage, read> tx: array<i32>;
@group(0) @binding(1) var<storage, read> ty: array<i32>;
@group(0) @binding(2) var<storage, read_write> olng: array<f64>;
@group(0) @binding(3) var<storage, read_write> olat: array<f64>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    if (i >= u[0].y) { return; }
    let z2 = exp2(f64(u[0].x));
    olng[i] = f64(tx[i]) / z2 * 360.0 - 180.0;
    olat[i] = atan(sinh(__PI__ * (1.0 - 2.0 * f64(ty[i]) / z2))) * __R2D__;
}
""".replace("__PI__", str(_PI)).replace("__R2D__", str(_R2D))
    ),
    "bounds": (
        _UNIFORM_PREAMBLE
        + """
@group(0) @binding(0) var<storage, read> tx: array<i32>;
@group(0) @binding(1) var<storage, read> ty: array<i32>;
@group(0) @binding(2) var<storage, read_write> ow: array<f64>;
@group(0) @binding(3) var<storage, read_write> os: array<f64>;
@group(0) @binding(4) var<storage, read_write> oe: array<f64>;
@group(0) @binding(5) var<storage, read_write> on: array<f64>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    if (i >= u[0].y) { return; }
    let z2 = exp2(f64(u[0].x));
    ow[i] = f64(tx[i]) / z2 * 360.0 - 180.0;
    oe[i] = f64(tx[i] + 1) / z2 * 360.0 - 180.0;
    on[i] = atan(sinh(__PI__ * (1.0 - 2.0 * f64(ty[i]) / z2))) * __R2D__;
    os[i] = atan(sinh(__PI__ * (1.0 - 2.0 * (f64(ty[i]) + 1.0) / z2))) * __R2D__;
}
""".replace("__PI__", str(_PI)).replace("__R2D__", str(_R2D))
    ),
    "xy_bounds": (
        _UNIFORM_PREAMBLE
        + """
@group(0) @binding(0) var<storage, read> tx: array<i32>;
@group(0) @binding(1) var<storage, read> ty: array<i32>;
@group(0) @binding(2) var<storage, read_write> ol: array<f64>;
@group(0) @binding(3) var<storage, read_write> ob: array<f64>;
@group(0) @binding(4) var<storage, read_write> orr: array<f64>;
@group(0) @binding(5) var<storage, read_write> ot: array<f64>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    if (i >= u[0].y) { return; }
    let z2 = exp2(f64(u[0].x));
    let ts = __CE__ / z2;
    ol[i] = f64(tx[i]) * ts - __CE__ * 0.5;
    orr[i] = ol[i] + ts;
    ot[i] = __CE__ * 0.5 - f64(ty[i]) * ts;
    ob[i] = ot[i] - ts;
}
""".replace("__CE__", str(_CE))
    ),
    "quadkey_encode": (
        _UNIFORM_PREAMBLE
        + """
@group(0) @binding(0) var<storage, read> tx: array<i32>;
@group(0) @binding(1) var<storage, read> ty: array<i32>;
@group(0) @binding(2) var<storage, read_write> qk: array<u64>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    if (i >= u[0].y) { return; }
    let zoom = u[0].x;
    var x: u64 = u64(tx[i]);
    var y: u64 = u64(ty[i]);
    var q: u64 = 0u;
    var s: i32 = 2 * (zoom - 1);
    var j: i32 = zoom - 1;
    while (j >= 0) {
        let d = ((x >> u32(j)) & 1u) | (((y >> u32(j)) & 1u) << 1u);
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
        + """
@group(0) @binding(0) var<storage, read> qk: array<u64>;
@group(0) @binding(1) var<storage, read_write> ox: array<i32>;
@group(0) @binding(2) var<storage, read_write> oy: array<i32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    if (i >= u[0].y) { return; }
    let zoom = u[0].x;
    let q = qk[i];
    var x: i32 = 0;
    var y: i32 = 0;
    var j: i32 = 0;
    while (j < zoom) {
        let shift = 2 * (zoom - 1 - j);
        let digit = i32((q >> u32(shift)) & 3u);
        let mask = 1 << (zoom - 1 - j);
        if (digit & 1) != 0 { x = x | mask; }
        if (digit & 2) != 0 { y = y | mask; }
        j = j + 1;
    }
    ox[i] = x;
    oy[i] = y;
}
"""
    ),
    "parent": (
        _UNIFORM_PREAMBLE
        + """
@group(0) @binding(0) var<storage, read> tx: array<i32>;
@group(0) @binding(1) var<storage, read> ty: array<i32>;
@group(0) @binding(2) var<storage, read_write> ox: array<i32>;
@group(0) @binding(3) var<storage, read_write> oy: array<i32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    if (i >= u[0].y) { return; }
    if (u[0].x == 0) {
        ox[i] = tx[i]; oy[i] = ty[i];
    } else {
        ox[i] = tx[i] >> 1; oy[i] = ty[i] >> 1;
    }
}
"""
    ),
    "children": (
        _UNIFORM_PREAMBLE
        + """
@group(0) @binding(0) var<storage, read> tx: array<i32>;
@group(0) @binding(1) var<storage, read> ty: array<i32>;
@group(0) @binding(2) var<storage, read_write> ox: array<i32>;
@group(0) @binding(3) var<storage, read_write> oy: array<i32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    if (i >= u[0].x) { return; }
    let x2 = tx[i] << 1;
    let y2 = ty[i] << 1;
    let k = i * 4;
    ox[k]     = x2;     oy[k]     = y2;
    ox[k + 1] = x2 + 1; oy[k + 1] = y2;
    ox[k + 2] = x2;     oy[k + 2] = y2 + 1;
    ox[k + 3] = x2 + 1; oy[k + 3] = y2 + 1;
}
"""
    ),
    "neighbors": (
        _UNIFORM_PREAMBLE
        + """
@group(0) @binding(0) var<storage, read> tx: array<i32>;
@group(0) @binding(1) var<storage, read> ty: array<i32>;
@group(0) @binding(2) var<storage, read_write> ox: array<i32>;
@group(0) @binding(3) var<storage, read_write> oy: array<i32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    if (i >= u[0].x) { return; }
    let x = tx[i];
    let y = ty[i];
    let k = i * 8;
    ox[k]     = x - 1; oy[k]     = y - 1;
    ox[k + 1] = x;     oy[k + 1] = y - 1;
    ox[k + 2] = x + 1; oy[k + 2] = y - 1;
    ox[k + 3] = x - 1; oy[k + 3] = y;
    ox[k + 4] = x + 1; oy[k + 4] = y;
    ox[k + 5] = x - 1; oy[k + 5] = y + 1;
    ox[k + 6] = x;     oy[k + 6] = y + 1;
    ox[k + 7] = x + 1; oy[k + 7] = y + 1;
}
"""
    ),
    "bounding_tile": (
        _UNIFORM_PREAMBLE
        + _clamp_src()
        + """
@group(0) @binding(0) var<storage, read> west: array<f64>;
@group(0) @binding(1) var<storage, read> south: array<f64>;
@group(0) @binding(2) var<storage, read> east: array<f64>;
@group(0) @binding(3) var<storage, read> north: array<f64>;
@group(0) @binding(4) var<storage, read_write> ox: array<i32>;
@group(0) @binding(5) var<storage, read_write> oy: array<i32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    if (i >= u[0].y) { return; }
    let z2 = exp2(f64(u[0].x));
    let latn = clamp_lat(north[i]) * __D2R__;
    ox[i] = i32(floor((west[i] + 180.0) / 360.0 * z2));
    oy[i] = i32(floor(__TY__ * z2));
}
""".replace("__D2R__", str(_D2R)).replace("__TY__", _tile_y("latn"))
    ),
    "tiles_in_bbox": (
        _UNIFORM_PREAMBLE
        + """
@group(0) @binding(0) var<storage, read_write> ox: array<i32>;
@group(0) @binding(1) var<storage, read_write> oy: array<i32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    if (i >= u[0].w) { return; }
    ox[i] = u[0].x + i % u[0].z;
    oy[i] = u[0].y + i / u[0].z;
}
"""
    ),
}


class VulkanBackend:
    """Lazily-initialised singleton wrapping a Vulkan device + cached pipelines."""

    _instance: "VulkanBackend | None" = None

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
        required = {"shader-f64"}
        self.has_u64 = "shader-int64" in features
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
        for binding, kind, buf in buffers:
            resource = {"buffer": buf, "offset": 0, "size": buf.size}
            bindings.append({"binding": binding, "resource": resource})
        bind_group = self.device.create_bind_group(
            layout=pipe.get_bind_group_layout(0), entries=bindings
        )
        encoder = self.device.create_command_encoder()
        pass_enc = encoder.begin_compute_pass()
        pass_enc.set_pipeline(pipe)
        pass_enc.set_bind_group(0, bind_group)
        pass_enc.dispatch_workgroups((n + 255) // 256)
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

    def _common_io(self, key, inputs, dtype, out_n, uni):
        in_bufs = [self._upload(np.ascontiguousarray(a, dtype))
                   for a in inputs]
        n = out_n
        return in_bufs, n

    def xy(self, lng, lat, truncate=False):
        lng = np.ascontiguousarray(lng, np.float64)
        lat = np.ascontiguousarray(lat, np.float64)
        n = lng.shape[0]
        bl = self._upload(lng)
        ba = self._upload(lat)
        bo = self._out(n * 8)
        by = self._out(n * 8)
        bu = self._uni([1 if truncate else 0, n])
        self._dispatch("xy", [(0, 0, bl), (1, 0, ba), (2, 0, bo), (3, 0, by),
                              (20, 0, bu)], n)
        return self._read(bo, np.float64, n), self._read(by, np.float64, n)

    def lnglat(self, x, y):
        x = np.ascontiguousarray(x, np.float64)
        y = np.ascontiguousarray(y, np.float64)
        n = x.shape[0]
        bx = self._upload(x)
        by = self._upload(y)
        bo = self._out(n * 8)
        bl = self._out(n * 8)
        bu = self._uni([n])
        self._dispatch("lnglat", [(0, 0, bx), (1, 0, by), (2, 0, bo), (3, 0, bl),
                                  (20, 0, bu)], n)
        return self._read(bo, np.float64, n), self._read(bl, np.float64, n)

    def tile(self, lng, lat, zoom, truncate=False):
        lng = np.ascontiguousarray(lng, np.float64)
        lat = np.ascontiguousarray(lat, np.float64)
        n = lng.shape[0]
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
        bx = self._upload(tx)
        by = self._upload(ty)
        bo = self._out(n * 8)
        bl = self._out(n * 8)
        bu = self._uni([zoom, n])
        self._dispatch("ul", [(0, 0, bx), (1, 0, by), (2, 0, bo), (3, 0, bl),
                              (20, 0, bu)], n)
        return self._read(bo, np.float64, n), self._read(bl, np.float64, n)

    def bounds(self, tx, ty, zoom):
        tx = np.ascontiguousarray(tx, np.int32)
        ty = np.ascontiguousarray(ty, np.int32)
        n = tx.shape[0]
        bx = self._upload(tx)
        by = self._upload(ty)
        bw = self._out(n * 8)
        bs = self._out(n * 8)
        be = self._out(n * 8)
        bn = self._out(n * 8)
        bu = self._uni([zoom, n])
        self._dispatch("bounds", [(0, 0, bx), (1, 0, by), (2, 0, bw),
                                  (3, 0, bs), (4, 0, be), (5, 0, bn),
                                  (20, 0, bu)], n)
        return (self._read(bw, np.float64, n), self._read(bs, np.float64, n),
                self._read(be, np.float64, n), self._read(bn, np.float64, n))

    def xy_bounds(self, tx, ty, zoom):
        tx = np.ascontiguousarray(tx, np.int32)
        ty = np.ascontiguousarray(ty, np.int32)
        n = tx.shape[0]
        bx = self._upload(tx)
        by = self._upload(ty)
        bl = self._out(n * 8)
        bb = self._out(n * 8)
        br = self._out(n * 8)
        bt = self._out(n * 8)
        bu = self._uni([zoom, n])
        self._dispatch("xy_bounds", [(0, 0, bx), (1, 0, by), (2, 0, bl),
                                     (3, 0, bb), (4, 0, br), (5, 0, bt),
                                     (20, 0, bu)], n)
        return (self._read(bl, np.float64, n), self._read(bb, np.float64, n),
                self._read(br, np.float64, n), self._read(bt, np.float64, n))

    def quadkey_encode(self, tx, ty, zoom):
        if not self.has_u64:
            return None
        tx = np.ascontiguousarray(tx, np.int32)
        ty = np.ascontiguousarray(ty, np.int32)
        n = tx.shape[0]
        bx = self._upload(tx)
        by = self._upload(ty)
        bq = self._out(n * 8)
        bu = self._uni([zoom, n])
        self._dispatch("quadkey_encode", [(0, 0, bx), (1, 0, by), (2, 0, bq),
                                          (20, 0, bu)], n)
        return self._read(bq, np.uint64, n)

    def quadkey_decode(self, qk, zoom):
        if not self.has_u64:
            return None
        qk = np.ascontiguousarray(qk, np.uint64)
        n = qk.shape[0]
        bq = self._upload(qk)
        bo = self._out(n * 4)
        by = self._out(n * 4)
        bu = self._uni([zoom, n])
        self._dispatch("quadkey_decode", [(0, 0, bq), (1, 0, bo), (2, 0, by),
                                          (20, 0, bu)], n)
        return self._read(bo, np.int32, n), self._read(by, np.int32, n)

    def parent(self, tx, ty, zoom):
        tx = np.ascontiguousarray(tx, np.int32)
        ty = np.ascontiguousarray(ty, np.int32)
        n = tx.shape[0]
        bx = self._upload(tx)
        by = self._upload(ty)
        bo = self._out(n * 4)
        byy = self._out(n * 4)
        bu = self._uni([zoom, n])
        self._dispatch("parent", [(0, 0, bx), (1, 0, by), (2, 0, bo), (3, 0, byy),
                                  (20, 0, bu)], n)
        return self._read(bo, np.int32, n), self._read(byy, np.int32, n)

    def children(self, tx, ty):
        tx = np.ascontiguousarray(tx, np.int32)
        ty = np.ascontiguousarray(ty, np.int32)
        n = tx.shape[0]
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
        bx = self._upload(tx)
        by = self._upload(ty)
        bo = self._out(n * 8 * 4)
        byy = self._out(n * 8 * 4)
        bu = self._uni([n])
        self._dispatch("neighbors", [(0, 0, bx), (1, 0, by), (2, 0, bo), (3, 0, byy),
                                     (20, 0, bu)], n)
        return self._read(bo, np.int32, n * 8), self._read(byy, np.int32, n * 8)

    def bounding_tile(self, west, south, east, north, zoom):
        west = np.ascontiguousarray(west, np.float64)
        south = np.ascontiguousarray(south, np.float64)
        east = np.ascontiguousarray(east, np.float64)
        north = np.ascontiguousarray(north, np.float64)
        n = west.shape[0]
        bw = self._upload(west)
        bs = self._upload(south)
        be = self._upload(east)
        bn = self._upload(north)
        bo = self._out(n * 4)
        by = self._out(n * 4)
        bu = self._uni([zoom, n])
        self._dispatch("bounding_tile", [(0, 0, bw), (1, 0, bs), (2, 0, be),
                                         (3, 0, bn), (4, 0, bo), (5, 0, by),
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
