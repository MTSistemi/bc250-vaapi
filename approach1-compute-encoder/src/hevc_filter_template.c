/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_filter_template.c - deblocking (8.7.2) and SAO (8.7.3).
 *
 * Included once per bit depth by hevc_filter.c. See hevc_pixel.h: every
 * stride in here is counted in samples.
 */

static inline int FUNC(clip)(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline int FUNC(clip_pixel)(int v)
{
    return v < 0 ? 0 : (v > PIXEL_MAX ? PIXEL_MAX : v);
}

/* The luma parameter of the coding unit covering a sample. */
static int FUNC(qp_di)(const hevcd_t *d, int x, int y)
{
    const int stride = d->sps->min_cb_width;
    const int l = d->sps->log2_min_cb;
    return d->qp_y_map[(y >> l) * stride + (x >> l)];
}

/* A coding unit coded losslessly keeps its samples exactly as they are:
 * filtering them would be the one thing that made the stream lossy. */
static bool FUNC(untouchable)(const hevcd_t *d, int x, int y)
{
    if (!d->no_filter) return false;
    const int stride = d->sps->min_cb_width;
    const int l = d->sps->log2_min_cb;
    return d->no_filter[(y >> l) * stride + (x >> l)] != 0;
}

/* One four-line segment of a luma FUNC(edge).
 *
 * `base` points at q0 of the first line, `forward` steps from p to q and
 * `giu` steps from one line to the next: for a vertical FUNC(edge) those are one
 * sample and one row, for a horizontal FUNC(edge) the other way round. Writing
 * it once in these two steps is what keeps the two directions from
 * drifting apart, which is where a hand-unrolled deblocking filter usually
 * goes wrong.
 */
static void FUNC(filter_luma)(pixel *base, int forward, int giu,
                        int beta, int tc, bool keep_p, bool keep_q, int bd)
{
#define P(k, i) ((int)base[(i) * giu - ((k) + 1) * forward])
#define Q(k, i) ((int)base[(i) * giu + (k) * forward])
#define WRITE_P(k, i, v) \
    do { if (!keep_p) base[(i) * giu - ((k) + 1) * forward] = (v); } while (0)
#define WRITE_Q(k, i, v) \
    do { if (!keep_q) base[(i) * giu + (k) * forward] = (v); } while (0)

    /* 8.7.2.5.3. The decision looks at the first and the last line of the
     * four and at nothing in between: four lines of an eight-sample block
     * FUNC(edge) are alike enough that two of them decide for all four, and
     * halving the work of the decision was worth it to the committee. */
    const int dp0 = abs(P(2, 0) - 2 * P(1, 0) + P(0, 0));
    const int dp3 = abs(P(2, 3) - 2 * P(1, 3) + P(0, 3));
    const int dq0 = abs(Q(2, 0) - 2 * Q(1, 0) + Q(0, 0));
    const int dq3 = abs(Q(2, 3) - 2 * Q(1, 3) + Q(0, 3));
    const int dpq0 = dp0 + dq0, dpq3 = dp3 + dq3;

    if (dpq0 + dpq3 >= beta) return;

    const int dp = dp0 + dp3, dq = dq0 + dq3;
    const int threshold = (5 * tc + 1) >> 1;

    /* 8.7.2.5.6: strong only when the step really is a step - flat on both
     * sides, and the jump across small enough to be an artefact rather
     * than an FUNC(edge) that belongs to the picture. */
    bool forte = true;
    for (int i = 0; i < 4 && forte; i += 3) {
        const int dpq = 2 * (i == 0 ? dpq0 : dpq3);
        forte = dpq < (beta >> 2)
             && abs(P(3, i) - P(0, i)) + abs(Q(0, i) - Q(3, i)) < (beta >> 3)
             && abs(P(0, i) - Q(0, i)) < threshold;
    }

    if (forte) {
        /* 8.7.2.5.7. Three samples each side, every one of them held
         * within two tC of where it started: the filter may smooth a step
         * but it may not invent one. */
        for (int i = 0; i < 4; i++) {
            const int p0 = P(0, i), p1 = P(1, i), p2 = P(2, i), p3 = P(3, i);
            const int q0 = Q(0, i), q1 = Q(1, i), q2 = Q(2, i), q3 = Q(3, i);
            WRITE_P(0, i, (pixel)FUNC(clip)((p2 + 2 * p1 + 2 * p0 + 2 * q0
                                              + q1 + 4) >> 3,
                                             p0 - 2 * tc, p0 + 2 * tc));
            WRITE_P(1, i, (pixel)FUNC(clip)((p2 + p1 + p0 + q0 + 2) >> 2,
                                             p1 - 2 * tc, p1 + 2 * tc));
            WRITE_P(2, i, (pixel)FUNC(clip)((2 * p3 + 3 * p2 + p1 + p0 + q0
                                              + 4) >> 3,
                                             p2 - 2 * tc, p2 + 2 * tc));
            WRITE_Q(0, i, (pixel)FUNC(clip)((p1 + 2 * p0 + 2 * q0 + 2 * q1
                                              + q2 + 4) >> 3,
                                             q0 - 2 * tc, q0 + 2 * tc));
            WRITE_Q(1, i, (pixel)FUNC(clip)((p0 + q0 + q1 + q2 + 2) >> 2,
                                             q1 - 2 * tc, q1 + 2 * tc));
            WRITE_Q(2, i, (pixel)FUNC(clip)((p0 + q0 + q1 + 3 * q2 + 2 * q3
                                              + 4) >> 3,
                                             q2 - 2 * tc, q2 + 2 * tc));
        }
        return;
    }

    /* 8.7.2.5.7, the weak filter. One sample each side always, a second
     * one only on whichever side was flat enough to deserve it. */
    const bool touch_p1 = dp < ((beta + (beta >> 1)) >> 3);
    const bool touch_q1 = dq < ((beta + (beta >> 1)) >> 3);

    for (int i = 0; i < 4; i++) {
        const int p0 = P(0, i), p1 = P(1, i), p2 = P(2, i);
        const int q0 = Q(0, i), q1 = Q(1, i), q2 = Q(2, i);
        int delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
        if (abs(delta) >= tc * 10) continue;
        delta = FUNC(clip)(delta, -tc, tc);
        WRITE_P(0, i, (pixel)FUNC(clip_pixel)(p0 + delta));
        WRITE_Q(0, i, (pixel)FUNC(clip_pixel)(q0 - delta));
        if (touch_p1) {
            const int dp1 = FUNC(clip)((((p2 + p0 + 1) >> 1) - p1 + delta) >> 1,
                                     -(tc >> 1), tc >> 1);
            WRITE_P(1, i, (pixel)FUNC(clip_pixel)(p1 + dp1));
        }
        if (touch_q1) {
            const int dq1 = FUNC(clip)((((q2 + q0 + 1) >> 1) - q1 - delta) >> 1,
                                     -(tc >> 1), tc >> 1);
            WRITE_Q(1, i, (pixel)FUNC(clip_pixel)(q1 + dq1));
        }
    }
#undef P
#undef Q
#undef WRITE_P
#undef WRITE_Q
}

/* 8.7.2.5.5. Chroma gets one sample each side and no decision at all: it
 * is filtered where the boundary FUNC(strength) is two and nowhere else, which
 * in an intra picture means every FUNC(edge). */
static void FUNC(filter_chroma)(pixel *base, int forward, int giu, int tc, int bd,
                         bool keep_p, bool keep_q)
{
    for (int i = 0; i < 4; i++) {
        pixel *p1 = base + i * giu - 2 * forward;
        pixel *p0 = base + i * giu - forward;
        pixel *q0 = base + i * giu;
        pixel *q1 = base + i * giu + forward;
        const int delta = FUNC(clip)(((((int)*q0 - *p0) * 4) + *p1 - *q1 + 4) >> 3,
                                   -tc, tc);
        if (!keep_p) *p0 = (pixel)FUNC(clip_pixel)(*p0 + delta);
        if (!keep_q) *q0 = (pixel)FUNC(clip_pixel)(*q0 - delta);
    }
}

/* Table 8-10 again, as 8.7.2.5.5 asks for it. */
static int FUNC(qp_chroma)(int qp_i)
{
    if (qp_i < 30) return qp_i < 0 ? 0 : qp_i;
    if (qp_i > 43) return qp_i - 6;
    return hevcd_qp_c[qp_i - 30];
}

/* beta and tC for one FUNC(edge), 8.7.2.5.3. The boundary FUNC(strength) only ever
 * reaches the tables through tC, and only by two quantiser steps. */
static int FUNC(beta_di)(const hevcd_t *d, int qp)
{
    const int q = FUNC(clip)(qp + d->slice->beta_offset, 0, 51);
    return hevcd_beta[q] << (d->sps->bit_depth_luma - 8);
}

static int FUNC(tc_di)(const hevcd_t *d, int qp, int bs)
{
    const int q = FUNC(clip)(qp + 2 * (bs - 1) + d->slice->tc_offset, 0, 53);
    return hevcd_tc[q] << (d->sps->bit_depth_luma - 8);
}

/* Is the FUNC(edge) on this side of an 8x8 cell one the filter may cross. */
static bool FUNC(edge)(const hevcd_t *d, int x, int y, int which)
{
    return (d->edges[(y >> 3) * d->edges_stride + (x >> 3)] & which) != 0;
}

/* 8.7.2.4. Two, one, or nothing at all.
 *
 * Two means an intra block is involved and the step across the FUNC(edge) is
 * whatever the prediction could not reach; that is worth the strong
 * filter and it is the only case where chroma is touched at all. One
 * means two inter blocks that disagree - a coded residual at a transform
 * FUNC(edge), different reference pictures, or motion a quarter sample apart.
 * Nothing means two blocks that were predicted the same way from the same
 * place, where any step across the FUNC(edge) would be something the filter
 * invented.
 */
static int FUNC(strength)(const hevcd_t *d, int xp, int yp, int xq, int yq,
                 bool transform_edge)
{
    const int stride = d->min_pu_width;
    const hevcd_mvf_t *p = &d->mvf[(yp >> 2) * stride + (xp >> 2)];
    const hevcd_mvf_t *q = &d->mvf[(yq >> 2) * stride + (xq >> 2)];

    if (!p->pred_flag || !q->pred_flag) return 2;

    if (transform_edge && d->cbf_map
        && (d->cbf_map[(yp >> 2) * stride + (xp >> 2)]
            || d->cbf_map[(yq >> 2) * stride + (xq >> 2)]))
        return 1;

    const int np = (p->pred_flag == HEVCD_PF_BI) ? 2 : 1;
    const int nq = (q->pred_flag == HEVCD_PF_BI) ? 2 : 1;
    if (np != nq) return 1;

#define DIFFER(a, la, b, lb)                                   \
    (abs((a)->mv[la][0] - (b)->mv[lb][0]) >= 4                  \
     || abs((a)->mv[la][1] - (b)->mv[lb][1]) >= 4)

    if (np == 1) {
        const int lp = (p->pred_flag & HEVCD_PF_L0) ? 0 : 1;
        const int lq = (q->pred_flag & HEVCD_PF_L0) ? 0 : 1;
        if (p->ref_poc[lp] != q->ref_poc[lq]) return 1;
        return DIFFER(p, lp, q, lq) ? 1 : 0;
    }

    /* Both predict from two places. They agree only if the two pairs of
     * pictures are the same pair - in either order - and the vectors that
     * go with them are close enough. */
    const bool same_ones = (p->ref_poc[0] == q->ref_poc[0]
                         && p->ref_poc[1] == q->ref_poc[1]);
    const bool crossed = (p->ref_poc[0] == q->ref_poc[1]
                             && p->ref_poc[1] == q->ref_poc[0]);
    if (!same_ones && !crossed) return 1;

    if (p->ref_poc[0] != p->ref_poc[1]) {
        /* ⚠️ Two different pictures: there is only one way to pair them
         * up, and it is whichever way makes the pictures match. */
        if (same_ones)
            return (DIFFER(p, 0, q, 0) || DIFFER(p, 1, q, 1)) ? 1 : 0;
        return (DIFFER(p, 0, q, 1) || DIFFER(p, 1, q, 0)) ? 1 : 0;
    }

    /* The same picture twice: either pairing will do, and the FUNC(edge) is
     * quiet if either one is close enough. */
    const bool dritte = !(DIFFER(p, 0, q, 0) || DIFFER(p, 1, q, 1));
    const bool crossed_refs = !(DIFFER(p, 0, q, 1) || DIFFER(p, 1, q, 0));
    return (dritte || crossed_refs) ? 0 : 1;
#undef DIFFER
}

/* One direction over the whole picture. `vertical` says which edges are
 * looked at, not which way the filter reads: a vertical FUNC(edge) is filtered
 * along x and stepped along y. */
static void FUNC(one_direction)(hevcd_t *d, bool vertical)
{
    const hevc_sps_t *sps = d->sps;
    const int forward_l = vertical ? 1 : d->stride[0];
    const int giu_l = vertical ? d->stride[0] : 1;
    const int which = vertical ? 1 : 2;

    for (int y = 0; y < sps->height; y += vertical ? 4 : 8)
        for (int x = 0; x < sps->width; x += vertical ? 8 : 4) {
            /* The picture's own border is never an FUNC(edge), and neither is a
             * position the coding tree never put a block boundary at. */
            if (vertical ? x == 0 : y == 0) continue;
            if (!FUNC(edge)(d, x, y, which)) continue;

            const int xp = vertical ? x - 1 : x;
            const int yp = vertical ? y : y - 1;
            const int bs = d->mvf
                ? FUNC(strength)(d, xp, yp, x, y,
                        (d->edges[(y >> 3) * d->edges_stride + (x >> 3)]
                         & (vertical ? 4 : 8)) != 0)
                : 2;
            if (!bs) continue;
            const int qp = (FUNC(qp_di)(d, x, y) + FUNC(qp_di)(d, xp, yp) + 1) >> 1;
            const bool keep_p = FUNC(untouchable)(d, xp, yp);
            const bool keep_q = FUNC(untouchable)(d, x, y);
            if (keep_p && keep_q) continue;

            FUNC(filter_luma)((pixel *)d->plane[0]
                        + (size_t)y * d->stride[0] + x,
                        forward_l, giu_l, FUNC(beta_di)(d, qp), FUNC(tc_di)(d, qp, bs),
                        keep_p, keep_q, d->sps->bit_depth_luma);

            /* ⚠️ Chroma is filtered on its own grid, which is eight chroma
             * samples and therefore sixteen luma ones. Filtering it
             * wherever luma is filtered doubles the edges it touches and
             * softens the picture in a way no reference decoder does. */
            if (bs != 2) continue;
            if (vertical ? (x & 15) : (y & 15)) continue;
            if (vertical ? (y & 7) : (x & 7)) continue;

            for (int c = 1; c < 3; c++) {
                const int off = c == 1 ? d->pps->cb_qp_offset
                                       : d->pps->cr_qp_offset;
                const int tc = hevcd_tc[FUNC(clip)(FUNC(qp_chroma)(FUNC(clip)(qp + off,
                                                                   0, 57))
                                                 + 2 + d->slice->tc_offset,
                                                 0, 53)]
                               << (d->sps->bit_depth_chroma - 8);
                if (!tc) continue;
                const int forward_c = vertical ? 1 : d->stride[c];
                const int giu_c = vertical ? d->stride[c] : 1;
                FUNC(filter_chroma)((pixel *)d->plane[c]
                             + (size_t)(y / 2) * d->stride[c]
                             + x / 2, forward_c, giu_c, tc,
                             d->sps->bit_depth_chroma, keep_p, keep_q);
            }
        }
}

/* 8.7.2 over the finished picture.
 *
 * ⚠️ The offsets come from the slice, and one picture may carry several
 * slices with different ones. With a single slice per picture - which is
 * what the harness feeds it - this is exact; with more it would need the
 * offsets kept per coding tree block, the same way the quantisation
 * parameter already is.
 */
static void FUNC(deblock)(hevcd_t *d)
{
    if (!d->edges || d->slice->deblocking_filter_disabled) return;
    FUNC(one_direction)(d, true);
    FUNC(one_direction)(d, false);
}

/* ------------------------------------------------- sample adaptive offset */

/* 8.7.3. The last thing in the decoding loop, and the only part of it the
 * encoder steers directly: four offsets per coding tree block per plane,
 * chosen by measuring the error that everything upstream left behind.
 *
 * ⚠️ It reads the picture the deblocking filter produced and writes a
 * different one. An implementation that reads and writes the same plane
 * gets the band offset right and the FUNC(edge) offset wrong, in a way that is
 * worth a handful of sample values and shows up only where the offsets
 * are large - which is exactly where nobody looks first.
 */
static int FUNC(sign)(int v)
{
    return v > 0 ? 1 : (v < 0 ? -1 : 0);
}

/* Which two neighbours each FUNC(edge) class compares against: horizontal,
 * vertical, and the two diagonals. */
static const int8_t FUNC(sao_dx)[4][2] = { { -1, 1 }, { 0, 0 }, { -1, 1 }, { 1, -1 } };
static const int8_t FUNC(sao_dy)[4][2] = { { 0, 0 }, { -1, 1 }, { -1, 1 }, { -1, 1 } };

static void FUNC(sao_block)(hevcd_t *d, int c, int rx, int ry,
                       const hevcd_sao_t *s)
{
    if (!s->kind[c]) return;

    const int giu = c ? 1 : 0;
    const int log2 = d->sps->log2_ctb - giu;
    const int w = d->sps->width >> giu, h = d->sps->height >> giu;
    const int x0 = rx << log2, y0 = ry << log2;
    const int side = 1 << log2;
    const int x1 = x0 + side < w ? x0 + side : w;
    const int y1 = y0 + side < h ? y0 + side : h;
    const int bd = c ? d->sps->bit_depth_chroma : d->sps->bit_depth_luma;
    const int stride = d->stride[c];
    pixel *plane = (pixel *)d->plane[c];
    const pixel *before = (const pixel *)d->copy_of[c];

    if (s->kind[c] == 1) {
        /* By band: the range of a sample is cut into thirty-two bands and
         * four consecutive ones get an offset each. An encoder reaches for
         * this where the error is a shift of level rather than a step -
         * a flat area that came out slightly too dark, say. */
        for (int y = y0; y < y1; y++)
            for (int x = x0; x < x1; x++) {
                if (FUNC(untouchable)(d, x << giu, y << giu)) continue;
                pixel *p = plane + (size_t)y * stride + x;
                /* Thirty-two bands over the whole range, so the index
                 * is the sample shifted down by BitDepth - 5. */
                const int k = ((*p >> (bd - 5)) - s->position[c]) & 31;
                if (k < 4) *p = (pixel)FUNC(clip_pixel)(*p + s->off[c][k]);
            }
        return;
    }

    /* By FUNC(edge): each sample is compared with two neighbours along one of
     * four directions, which sorts it into a valley, a step, or a peak,
     * and each of those gets its own offset. */
    const int cl = s->category[c];
    const int ax = FUNC(sao_dx)[cl][0], ay = FUNC(sao_dy)[cl][0];
    const int bx = FUNC(sao_dx)[cl][1], by = FUNC(sao_dy)[cl][1];

    for (int y = y0; y < y1; y++)
        for (int x = x0; x < x1; x++) {
            /* A sample whose neighbour would be outside the picture is
             * left alone: there is nothing to compare it with. */
            if (x + ax < 0 || x + ax >= w || x + bx < 0 || x + bx >= w) continue;
            if (y + ay < 0 || y + ay >= h || y + by < 0 || y + by >= h) continue;
            if (FUNC(untouchable)(d, x << giu, y << giu)) continue;

            const int v = before[(size_t)y * stride + x];
            int idx = 2 + FUNC(sign)(v - before[(size_t)(y + ay) * stride + x + ax])
                        + FUNC(sign)(v - before[(size_t)(y + by) * stride + x + bx]);
            /* Table 8-x: the five cases are renumbered so that "neither up
             * nor down" lands on zero, which is the one with no offset. */
            if (idx <= 2) idx = (idx == 2) ? 0 : idx + 1;
            if (!idx) continue;
            plane[(size_t)y * stride + x] = (pixel)
                FUNC(clip_pixel)(v + s->off[c][idx - 1]);
        }
}

static void FUNC(sao)(hevcd_t *d)
{
    if (!d->sao || !d->plane[0]) return;

    bool serve = false;
    for (int i = 0; i < d->sps->ctb_count && !serve; i++)
        serve = d->sao[i].kind[0] || d->sao[i].kind[1] || d->sao[i].kind[2];
    if (!serve) return;

    const size_t measure[3] = { d->n_planes, d->n_planes / 4, d->n_planes / 4 };
    for (int c = 0; c < 3; c++) {
        if (!d->copy_of[c] || d->n_copy < d->n_planes) {
            free(d->copy_of[c]);
            d->copy_of[c] = malloc(measure[c]);
            if (!d->copy_of[c]) return;
        }
        memcpy(d->copy_of[c], d->plane[c], measure[c]);
    }
    d->n_copy = d->n_planes;

    for (int ry = 0; ry < d->sps->ctb_height; ry++)
        for (int rx = 0; rx < d->sps->ctb_width; rx++) {
            const hevcd_sao_t *s = &d->sao[ry * d->sps->ctb_width + rx];
            for (int c = 0; c < 3; c++) FUNC(sao_block)(d, c, rx, ry, s);
        }
}
