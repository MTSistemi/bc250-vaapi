/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_pred_template.c - intra prediction, Rec. ITU-T H.265 clause 8.4.4.2.
 *
 * Included once per bit depth by hevc_pred.c. See hevc_pixel.h: `pixel`
 * is a byte at eight bits and a sixteen-bit word above it, and every
 * stride here is counted in samples.
 */

static inline int FUNC(clip_pixel)(int v)
{
    return v < 0 ? 0 : (v > PIXEL_MAX ? PIXEL_MAX : v);
}

/* The reference array, laid out around the corner: index 0 is the corner
 * sample p[-1][-1], 1..2N is the row above, and -1..-2N the column to the
 * left. One array instead of two, so the angular modes that reach across
 * the corner can walk straight through it. */
#define RIF(r, i) ((r)[64 + (i)])

/* Clause 6.4.1 through the z-scan order: whether the block at (x, y) has
 * already been decoded.
 *
 * ⚠️ Not "is it above or to the left". A coding tree unit is walked as a
 * quadtree, so the block above right of a transform block may or may not
 * have been decoded depending on where both sit in the tree. The z-scan
 * address is what answers that, and comparing coordinates instead gets the
 * top-right reference samples wrong for exactly the blocks where they
 * matter. */
static bool FUNC(already_decoded)(const hevcd_t *d, int x, int y, int x_cur, int y_cur)
{
    const hevc_sps_t *sps = d->sps;
    if (x < 0 || y < 0 || x >= sps->width || y >= sps->height)
        return false;
    /* ⚠️ Same wall as in already_done(): an earlier tile has lower
     * addresses and would otherwise look like a legitimate neighbour. */
    if (hevcd_tile_at(d, x, y) != d->tile_now) return false;
    if (hevcd_slice_at(d, x, y) != d->slice_now) return false;
    const int stride = sps->width >> sps->log2_min_tb;
    const int a = d->min_tb_addr_zs[(y >> sps->log2_min_tb) * stride
                                    + (x >> sps->log2_min_tb)];
    const int b = d->min_tb_addr_zs[(y_cur >> sps->log2_min_tb) * stride
                                    + (x_cur >> sps->log2_min_tb)];
    if (a >= b) return false;
    /* ⚠️ 8.4.4.2.2: under constrained intra prediction a sample from a
     * block that was not intra coded does not exist either, as far as
     * intra prediction is concerned. The substitution below fills it from
     * the intra samples around it, exactly as it fills the picture's
     * edge. The motion field is cleared for every picture, so an intra
     * block reads as no prediction flag at all. */
    if (d->pps->constrained_intra_pred
        && d->mvf[(y >> 2) * d->min_pu_width + (x >> 2)].pred_flag)
        return false;
    return true;
}

/* 8.4.4.2.2: gather, then substitute. */
static void FUNC(references)(const hevcd_t *d, int c_idx, int x0, int y0, int n,
                        pixel *r)
{
    const pixel *plane = (const pixel *)d->plane[c_idx];
    const int stride = d->stride[c_idx];
    const int scale_of = c_idx ? 1 : 0;          /* chroma is half resolution */
    const int lx = x0 << scale_of, ly = y0 << scale_of;   /* in luma coordinates */
    const int unit = 1 << scale_of;             /* luma samples per sample */

    bool c_e[4 * 64 + 1];
    memset(c_e, 0, sizeof(c_e));
    bool something = false;

    /* The column to the left, from the bottom up, then the corner, then
     * the row above from left to right: the order the substitution walks. */
    for (int i = 0; i < 2 * n; i++) {
        const int y = y0 + 2 * n - 1 - i;
        const int ok = FUNC(already_decoded)(d, lx - unit, ly + ((2 * n - 1 - i) << scale_of),
                                        lx, ly);
        if (ok && y < (d->sps->height >> scale_of)) {
            RIF(r, -(2 * n - i)) = plane[y * stride + x0 - 1];
            c_e[64 - (2 * n - i)] = true;
            something = true;
        }
    }
    {
        const int ok = FUNC(already_decoded)(d, lx - unit, ly - unit, lx, ly);
        if (ok) {
            RIF(r, 0) = plane[(y0 - 1) * stride + x0 - 1];
            c_e[64] = true;
            something = true;
        }
    }
    for (int i = 0; i < 2 * n; i++) {
        const int x = x0 + i;
        const int ok = FUNC(already_decoded)(d, lx + (i << scale_of), ly - unit, lx, ly);
        if (ok && x < (d->sps->width >> scale_of)) {
            RIF(r, i + 1) = plane[(y0 - 1) * stride + x];
            c_e[64 + i + 1] = true;
            something = true;
        }
    }

    if (!something) {
        const int bd = c_idx ? d->sps->bit_depth_chroma
                             : d->sps->bit_depth_luma;
        /* ⚠️ Not memset: it writes bytes, and above eight bits the
         * value truncates to zero and the count covers half the array.
         * 8.4.4.2.2. */
        for (int i = 0; i < 4 * 64 + 1; i++) r[i] = 1 << (bd - 1);
        return;
    }

    /* Walk from the bottom left, anticlockwise: each hole takes the value
     * of the one before it. The first hole, if it is at the very start,
     * takes the first sample that does exist. */
    if (!c_e[64 - 2 * n]) {
        int k = 64 - 2 * n;
        while (k <= 64 + 2 * n && !c_e[k]) k++;
        r[64 - 2 * n] = r[k];
        c_e[64 - 2 * n] = true;
    }
    for (int k = 64 - 2 * n + 1; k <= 64 + 2 * n; k++)
        if (!c_e[k]) { r[k] = r[k - 1]; c_e[k] = true; }
}

/* 8.4.4.2.3: whether to smooth the references, and how much. */
static void FUNC(filter_edge)(const hevcd_t *d, int mode, int n, int c_idx, pixel *r)
{
    if (c_idx != 0 || n == 4 || mode == HEVCD_INTRA_DC)
        return;

    /* Table 8-3, by log2 of the block side: 8 has a threshold of seven,
     * 16 of one, 32 of none. A four is handled above, and there is no
     * entry for it here - which is what the two zeros are holding.
     *
     * ⚠️ Indexed by the logarithm, not by the size. The first version of
     * this was off by one place and read past the end for a 32, which is
     * a wrong smoothing decision on exactly the blocks where smoothing
     * matters most. */
    static const int threshold[6] = { 0, 0, 0, 7, 1, 0 };
    int lg = 0;
    while ((1 << lg) < n) lg++;
    /* For planar this comes out as ten, which is what the clause intends:
     * the mode is as far from horizontal and vertical as anything gets. */
    const int dv = abs(mode - 26), dh = abs(mode - 10);
    const int dist = dv < dh ? dv : dh;
    if (dist <= threshold[lg])
        return;

    pixel f[4 * 64 + 1];
    memcpy(f, r, sizeof(f));

    /* The strong smoothing of a 32x32 block, when both edges are close
     * enough to a straight line that a ramp will do: a gradient with no
     * steps at all, which is what a flat sky needs. */
    /* ⚠️ 1 << (BitDepth - 5), which is the eight that used to be written
     * here. Table 8-3 above does not move with the depth - that one is a
     * distance between mode numbers. */
    const int flat = 1 << (d->sps->bit_depth_luma - 5);
    if (d->sps->strong_intra_smoothing && n == 32
        && abs(RIF(r, 0) + RIF(r, 2 * n) - 2 * RIF(r, n)) < flat
        && abs(RIF(r, 0) + RIF(r, -2 * n) - 2 * RIF(r, -n)) < flat) {
        for (int i = 1; i < 2 * n; i++) {
            RIF(f, i) = (pixel)(((64 - i) * RIF(r, 0)
                                   + i * RIF(r, 2 * n) + 32) >> 6);
            RIF(f, -i) = (pixel)(((64 - i) * RIF(r, 0)
                                    + i * RIF(r, -2 * n) + 32) >> 6);
        }
    } else {
        RIF(f, 0) = (pixel)((RIF(r, -1) + 2 * RIF(r, 0) + RIF(r, 1) + 2) >> 2);
        for (int i = 1; i < 2 * n; i++) {
            RIF(f, i) = (pixel)((RIF(r, i - 1) + 2 * RIF(r, i)
                                   + RIF(r, i + 1) + 2) >> 2);
            RIF(f, -i) = (pixel)((RIF(r, -(i - 1)) + 2 * RIF(r, -i)
                                    + RIF(r, -(i + 1)) + 2) >> 2);
        }
    }
    memcpy(r, f, sizeof(f));
}

/* 8.4.4.2.5, planar: a bilinear ramp between the four edges. */
static void FUNC(planar)(const pixel *r, int n, int lg, pixel *dst, int stride)
{
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++)
            dst[y * stride + x] = (pixel)
                (((n - 1 - x) * RIF(r, -(y + 1)) + (x + 1) * RIF(r, n + 1)
                  + (n - 1 - y) * RIF(r, x + 1) + (y + 1) * RIF(r, -(n + 1))
                  + n) >> (lg + 1));
}

/* 8.4.4.2.5, DC: the average, with the two edges smoothed into it on small
 * luma blocks so the join does not show. */
static void FUNC(smoothed)(const pixel *r, int n, int lg, int c_idx,
                     pixel *dst, int stride)
{
    int sum = n;
    for (int i = 0; i < n; i++)
        sum += RIF(r, i + 1) + RIF(r, -(i + 1));
    const int dc = sum >> (lg + 1);

    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++)
            dst[y * stride + x] = (pixel)dc;

    if (c_idx != 0 || n >= 32)
        return;
    dst[0] = (pixel)((RIF(r, -1) + 2 * dc + RIF(r, 1) + 2) >> 2);
    for (int x = 1; x < n; x++)
        dst[x] = (pixel)((RIF(r, x + 1) + 3 * dc + 2) >> 2);
    for (int y = 1; y < n; y++)
        dst[y * stride] = (pixel)((RIF(r, -(y + 1)) + 3 * dc + 2) >> 2);
}

/* 8.4.4.2.6, the thirty-three angles.
 *
 * ⚠️ A mode whose angle is negative reaches past the corner and needs the
 * other edge projected onto its own reference line - which is what
 * invAngle is for. Without it the samples past the corner are whatever was
 * left there, and the error is confined to one triangle of the block. */
static void FUNC(angular)(const pixel *r, int mode, int n, int c_idx,
                     pixel *dst, int stride)
{
    const int ang = hevcd_intra_angle[mode - 2];
    const bool vertical = mode >= 18;

    /* One reference line, built in the direction the mode walks. */
    int16_t ref_pic[3 * 64 + 1];
    int16_t *base = ref_pic + 64;
    const int sign = vertical ? 1 : -1;

    (void)sign;
    /* Index 0 is the corner either way; after that the row above for a
     * vertical mode and the column to the left for a horizontal one. */
    base[0] = RIF(r, 0);
    for (int x = 1; x <= n; x++)
        base[x] = vertical ? RIF(r, x) : RIF(r, -x);

    if (ang < 0) {
        const int until = (n * ang) >> 5;
        if (until < -1) {
            const int inv = hevcd_inv_angle[mode - 11];
            for (int x = -1; x >= until; x--) {
                const int k = ((x * inv + 128) >> 8);
                base[x] = vertical ? RIF(r, -k) : RIF(r, k);
            }
        }
    } else {
        for (int x = n + 1; x <= 2 * n; x++)
            base[x] = vertical ? RIF(r, x) : RIF(r, -x);
    }

    for (int y = 0; y < n; y++) {
        const int idx = ((y + 1) * ang) >> 5;
        const int fatt = ((y + 1) * ang) & 31;
        for (int x = 0; x < n; x++) {
            int v;
            if (fatt)
                v = ((32 - fatt) * base[x + idx + 1]
                     + fatt * base[x + idx + 2] + 16) >> 5;
            else
                v = base[x + idx + 1];
            if (vertical) dst[y * stride + x] = (pixel)v;
            else           dst[x * stride + y] = (pixel)v;
        }
    }

    /* The exactly vertical and exactly horizontal modes smooth their first
     * line against the other edge, on small luma blocks. */
    if (c_idx == 0 && n < 32) {
        if (mode == HEVCD_INTRA_ANGULAR_26) {
            for (int y = 0; y < n; y++)
                dst[y * stride] = (pixel)FUNC(clip_pixel)(RIF(r, 1)
                            + ((RIF(r, -(y + 1)) - RIF(r, 0)) >> 1));
        } else if (mode == HEVCD_INTRA_ANGULAR_10) {
            for (int x = 0; x < n; x++)
                dst[x] = (pixel)FUNC(clip_pixel)(RIF(r, -1)
                            + ((RIF(r, x + 1) - RIF(r, 0)) >> 1));
        }
    }
}

static void FUNC(predict_intra)(hevcd_t *d, int c_idx, int x0, int y0,
                                int log2_size, int mode)
{
    const int n = 1 << log2_size;
    const int bd = c_idx ? d->sps->bit_depth_chroma : d->sps->bit_depth_luma;
    pixel r[4 * 64 + 1];
    for (int i = 0; i < 4 * 64 + 1; i++) r[i] = 1 << (bd - 1);

    FUNC(references)(d, c_idx, x0, y0, n, r);
    FUNC(filter_edge)(d, mode, n, c_idx, r);

    /* ⚠️ Convert first, index after: the stride is in samples. */
    pixel *dst = (pixel *)d->plane[c_idx]
               + (size_t)y0 * d->stride[c_idx] + x0;
    const int stride = d->stride[c_idx];

    if (mode == HEVCD_INTRA_PLANAR)      FUNC(planar)(r, n, log2_size, dst, stride);
    else if (mode == HEVCD_INTRA_DC)     FUNC(smoothed)(r, n, log2_size, c_idx, dst, stride);
    else                                 FUNC(angular)(r, mode, n, c_idx, dst, stride);
}
