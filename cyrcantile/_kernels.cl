# cyrcantile/_kernels.cl
# OpenCL kernels for spherical-mercator tile computations.
# Ported from the Cython _base.pyx to GPU-parallel batch kernels.

#define PI       3.14159265358979323846
#define HALF_PI  1.57079632679489661923
#define R2D      57.29577951308232
#define D2R      0.017453292519943295
#define RE       6378137.0
#define CE       40075016.68557849
#define MAX_LAT  85.05112878
#define MAX_LNG  180.0

/* ---- helpers -------------------------------------------------- */

double _clamp_lat(double lat) {
    return fmax(-MAX_LAT, fmin(MAX_LAT, lat));
}

double _clamp_lng(double lng) {
    if (lng > MAX_LNG)  return MAX_LNG;
    if (lng < -MAX_LNG) return -MAX_LNG;
    return lng;
}

/* ---- 1. Geographic (WGS-84) -> Web Mercator (EPSG:3857) ------ */

__kernel void xy_batch(
    __global const double *lng,
    __global const double *lat,
    __global double *ox,
    __global double *oy,
    const int  truncate,
    const ulong n
) {
    size_t i = get_global_id(0);
    if (i >= n) return;

    double l = truncate ? _clamp_lng(lng[i]) : lng[i];
    double p = _clamp_lat(lat[i]) * D2R;

    ox[i] = RE * l * D2R;
    oy[i] = RE * log(tan(HALF_PI + p * 0.5));
}

/* ---- 2. Web Mercator (EPSG:3857) -> Geographic (WGS-84) ----- */

__kernel void lnglat_batch(
    __global const double *x,
    __global const double *y,
    __global double *olng,
    __global double *olat,
    const ulong n
) {
    size_t i = get_global_id(0);
    if (i >= n) return;

    olng[i] = (x[i] / RE) * R2D;
    olat[i] = (2.0 * atan(exp(y[i] / RE)) - HALF_PI) * R2D;
}

/* ---- 3. Geographic -> Tile (x, y) at *zoom* ----------------- */

__kernel void tile_batch(
    __global const double *lng,
    __global const double *lat,
    __global int *ox,
    __global int *oy,
    const int  zoom,
    const int  truncate,
    const ulong n
) {
    size_t i = get_global_id(0);
    if (i >= n) return;

    double l = truncate ? _clamp_lng(lng[i]) : lng[i];
    double p = _clamp_lat(lat[i]) * D2R;
    double z2 = exp2((double)zoom);

    ox[i] = (int)floor((l + 180.0) / 360.0 * z2);
    oy[i] = (int)floor((1.0 - log(tan(HALF_PI + p * 0.5)) / PI) * 0.5 * z2);
}

/* ---- 4. Tile -> Upper-Left corner (lng, lat) --------------- */

__kernel void ul_batch(
    __global const int *tx,
    __global const int *ty,
    __global double *olng,
    __global double *olat,
    const int  zoom,
    const ulong n
) {
    size_t i = get_global_id(0);
    if (i >= n) return;

    double z2 = exp2((double)zoom);

    olng[i] = (double)tx[i] / z2 * 360.0 - 180.0;
    olat[i] = atan(sinh(PI * (1.0 - 2.0 * (double)ty[i] / z2))) * R2D;
}

/* ---- 5. Tile -> Geographic bounds (w, s, e, n) ------------- */

__kernel void bounds_batch(
    __global const int  *tx,
    __global const int  *ty,
    __global double *ow,
    __global double *os,
    __global double *oe,
    __global double *on,
    const int  zoom,
    const ulong n
) {
    size_t i = get_global_id(0);
    if (i >= n) return;

    double z2 = exp2((double)zoom);

    ow[i] = (double)tx[i]       / z2 * 360.0 - 180.0;
    oe[i] = (double)(tx[i] + 1) / z2 * 360.0 - 180.0;
    on[i] = atan(sinh(PI * (1.0 - 2.0 *  (double)ty[i]       / z2))) * R2D;
    os[i] = atan(sinh(PI * (1.0 - 2.0 * ((double)ty[i] + 1.0) / z2))) * R2D;
}

/* ---- 6. Tile -> Web-Mercator bounds (l, b, r, t) ----------- */

__kernel void xy_bounds_batch(
    __global const int  *tx,
    __global const int  *ty,
    __global double *ol,
    __global double *ob,
    __global double *or,
    __global double *ot,
    const int  zoom,
    const ulong n
) {
    size_t i = get_global_id(0);
    if (i >= n) return;

    double z2    = exp2((double)zoom);
    double t_size = CE / z2;

    ol[i] = (double)tx[i] * t_size - CE * 0.5;
    or[i] = ol[i] + t_size;
    ot[i] = CE * 0.5 - (double)ty[i] * t_size;
    ob[i] = ot[i] - t_size;
}

/* ---- 7. Tile -> Quadkey (packed ulong, 2 bits/digit) ------- */

__kernel void quadkey_encode_batch(
    __global const int *tx,
    __global const int *ty,
    __global ulong *oqk,
    const int  zoom,
    const ulong n
) {
    size_t i = get_global_id(0);
    if (i >= n) return;

    ulong qk = 0;
    for (int j = zoom; j > 0; j--) {
        int mask  = 1 << (j - 1);
        int digit = 0;
        if (tx[i] & mask) digit += 1;
        if (ty[i] & mask) digit += 2;
        qk = (qk << 2) | (ulong)digit;
    }
    oqk[i] = qk;
}

/* ---- 8. Quadkey (packed ulong) -> Tile (x, y) -------------- */

__kernel void quadkey_decode_batch(
    __global const ulong *qk,
    __global int *ox,
    __global int *oy,
    const int  zoom,
    const ulong n
) {
    size_t i = get_global_id(0);
    if (i >= n) return;

    int x = 0, y = 0;
    ulong q = qk[i];
    for (int j = 0; j < zoom; j++) {
        int digit = (int)((q >> (2 * (zoom - 1 - j))) & 3UL);
        int mask  = 1 << (zoom - 1 - j);
        if (digit & 1) x |= mask;
        if (digit & 2) y |= mask;
    }
    ox[i] = x;
    oy[i] = y;
}

/* ---- 9. Tile -> Parent tile ------------------------------- */

__kernel void parent_batch(
    __global const int *tx,
    __global const int *ty,
    __global int *ox,
    __global int *oy,
    const int  zoom,
    const ulong n
) {
    size_t i = get_global_id(0);
    if (i >= n) return;

    if (zoom == 0) {
        ox[i] = tx[i];
        oy[i] = ty[i];
    } else {
        ox[i] = tx[i] >> 1;
        oy[i] = ty[i] >> 1;
    }
}

/* ---- 10. Tile -> 4 Children -------------------------------- */

__kernel void children_batch(
    __global const int *tx,
    __global const int *ty,
    __global int *ox,
    __global int *oy,
    const ulong n
) {
    size_t i = get_global_id(0);
    if (i >= n) return;

    int x2 = tx[i] << 1;
    int y2 = ty[i] << 1;

    ox[i*4+0] = x2;     oy[i*4+0] = y2;
    ox[i*4+1] = x2 + 1; oy[i*4+1] = y2;
    ox[i*4+2] = x2;     oy[i*4+2] = y2 + 1;
    ox[i*4+3] = x2 + 1; oy[i*4+3] = y2 + 1;
}

/* ---- 11. Tile -> 8 Neighbours (NW,N,NE,W,E,SW,S,SE) ------- */

__kernel void neighbors_batch(
    __global const int *tx,
    __global const int *ty,
    __global int *ox,
    __global int *oy,
    const ulong n
) {
    size_t i = get_global_id(0);
    if (i >= n) return;

    int x = tx[i], y = ty[i];

    ox[i*8+0] = x-1; oy[i*8+0] = y-1;
    ox[i*8+1] = x;   oy[i*8+1] = y-1;
    ox[i*8+2] = x+1; oy[i*8+2] = y-1;
    ox[i*8+3] = x-1; oy[i*8+3] = y;
    ox[i*8+4] = x+1; oy[i*8+4] = y;
    ox[i*8+5] = x-1; oy[i*8+5] = y+1;
    ox[i*8+6] = x;   oy[i*8+6] = y+1;
    ox[i*8+7] = x+1; oy[i*8+7] = y+1;
}

/* ---- 12. Bounding-tile NW corner for a geographic bbox ----- */

__kernel void bounding_tile_batch(
    __global const double *west,
    __global const double *south,
    __global const double *east,
    __global const double *north,
    __global int *ox,
    __global int *oy,
    const int  zoom,
    const ulong n
) {
    size_t i = get_global_id(0);
    if (i >= n) return;

    double z2   = exp2((double)zoom);
    double latn = _clamp_lat(north[i]) * D2R;

    ox[i] = (int)floor((west[i] + 180.0) / 360.0 * z2);
    oy[i] = (int)floor((1.0 - log(tan(HALF_PI + latn * 0.5)) / PI) * 0.5 * z2);
}

/* ---- 13. Grid-fill: enumerate tiles in a bbox ------------- */

__kernel void tiles_in_bbox(
    __global int *ox,
    __global int *oy,
    const int  min_x,
    const int  min_y,
    const int  grid_w,
    const ulong n
) {
    size_t i = get_global_id(0);
    if (i >= n) return;

    ox[i] = min_x + (int)(i % (ulong)grid_w);
    oy[i] = min_y + (int)(i / (ulong)grid_w);
}
