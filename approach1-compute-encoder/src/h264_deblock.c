/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * h264_deblock.c - the deblocking filter, Rec. ITU-T H.264 clause 8.7.
 */
#include "h264_deblock.h"

#include <stdlib.h>

#include "h264_dec_tables.h"

static inline uint8_t clip_uint8(int v)
{
    return (uint8_t)(v & ~255 ? (-v) >> 31 : v);
}

static inline int clip3(int lo, int hi, int v)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/* ------------------------------------------------------------- the filters */

/* Clause 8.7.2.3, one line across a luma edge with bS below 4.
 *
 * `q` points at q0; `d` is the step from one sample to the next across the
 * edge, so p0 is q[-d], p1 is q[-2d] and so on. A vertical edge has d = 1, a
 * horizontal one d = stride.
 */
static inline void luma_plain(uint8_t *q, int d, int alpha, int beta, int tc0)
{
    const int p0 = q[-d], p1 = q[-2 * d], p2 = q[-3 * d];
    const int q0 = q[0],  q1 = q[d],     q2 = q[2 * d];

    if (abs(p0 - q0) >= alpha || abs(p1 - p0) >= beta || abs(q1 - q0) >= beta)
        return;

    const int ap = abs(p2 - p0);
    const int aq = abs(q2 - q0);
    /* ⚠️ tc for the p0/q0 pair grows by one for each side that is flat
     * enough to also have its p1 filtered, but the clip on p1 and q1
     * themselves stays at the unincremented tc0. */
    const int tc = tc0 + (ap < beta) + (aq < beta);

    const int delta = clip3(-tc, tc, ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3));
    if (ap < beta)
        q[-2 * d] = (uint8_t)(p1 + clip3(-tc0, tc0,
                              (p2 + ((p0 + q0 + 1) >> 1) - (p1 << 1)) >> 1));
    if (aq < beta)
        q[d] = (uint8_t)(q1 + clip3(-tc0, tc0,
                         (q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1));
    q[-d] = clip_uint8(p0 + delta);
    q[0]  = clip_uint8(q0 - delta);
}

/* Clause 8.7.2.4, one line across a luma edge with bS 4. */
static inline void luma_forte(uint8_t *q, int d, int alpha, int beta)
{
    const int p0 = q[-d], p1 = q[-2 * d], p2 = q[-3 * d], p3 = q[-4 * d];
    const int q0 = q[0],  q1 = q[d],     q2 = q[2 * d],  q3 = q[3 * d];

    if (abs(p0 - q0) >= alpha || abs(p1 - p0) >= beta || abs(q1 - q0) >= beta)
        return;

    const int narrow = abs(p0 - q0) < ((alpha >> 2) + 2);

    if (abs(p2 - p0) < beta && narrow) {
        q[-d]     = (uint8_t)((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
        q[-2 * d] = (uint8_t)((p2 + p1 + p0 + q0 + 2) >> 2);
        q[-3 * d] = (uint8_t)((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3);
    } else {
        q[-d] = (uint8_t)((2 * p1 + p0 + q1 + 2) >> 2);
    }

    if (abs(q2 - q0) < beta && narrow) {
        q[0]     = (uint8_t)((q2 + 2 * q1 + 2 * q0 + 2 * p0 + p1 + 4) >> 3);
        q[d]     = (uint8_t)((q2 + q1 + q0 + p0 + 2) >> 2);
        q[2 * d] = (uint8_t)((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3);
    } else {
        q[0] = (uint8_t)((2 * q1 + q0 + p1 + 2) >> 2);
    }
}

/* Clause 8.7.2.3 and 8.7.2.4 for chroma: p1 and q1 are never written, and
 * the normal filter's tc is one higher because there is no ap/aq term. */
static inline void chroma_plain(uint8_t *q, int d, int alpha, int beta, int tc0)
{
    const int p0 = q[-d], p1 = q[-2 * d];
    const int q0 = q[0],  q1 = q[d];

    if (abs(p0 - q0) >= alpha || abs(p1 - p0) >= beta || abs(q1 - q0) >= beta)
        return;

    const int tc = tc0 + 1;
    const int delta = clip3(-tc, tc, ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3));
    q[-d] = clip_uint8(p0 + delta);
    q[0]  = clip_uint8(q0 - delta);
}

static inline void chroma_forte(uint8_t *q, int d, int alpha, int beta)
{
    const int p0 = q[-d], p1 = q[-2 * d];
    const int q0 = q[0],  q1 = q[d];

    if (abs(p0 - q0) >= alpha || abs(p1 - p0) >= beta || abs(q1 - q0) >= beta)
        return;

    q[-d] = (uint8_t)((2 * p1 + p0 + q1 + 2) >> 2);
    q[0]  = (uint8_t)((2 * q1 + q0 + p1 + 2) >> 2);
}

/* ------------------------------------------------------- strength, 8.7.2.1 */

/* The 4x4 block index inside a macroblock, from its position in samples. */
static inline int block(int x4, int y4) { return y4 * 4 + x4; }

/* Whether the block containing 4x4 position `b` of macroblock `m` carries
 * any coefficient, clause 8.7.2.1.
 *
 * ⚠️ "The block" is the 8x8 one when the macroblock uses the 8x8
 * transform, so all four of its 4x4 positions have to answer together. The
 * OR is done here rather than left to the entropy decoders, because the two
 * of them store different things: CABAC has no per-4x4 count for an 8x8
 * block and writes a flag into all four, while CAVLC reads four real counts
 * and needs to keep them - they are the context of the next block's
 * coeff_token. */
static inline bool has_coefficients(const h264d_mb_t *m, int b)
{
    if (!m->transform8x8)
        return m->nnz[0][b] != 0;
    /* The 8x8 block's corner: clear the low bit of the row and of the
     * column, which are bits 2 and 0 of the raster index. */
    const int angle = b & 10;
    return m->nnz[0][angle] || m->nnz[0][angle + 1]
        || m->nnz[0][angle + 4] || m->nnz[0][angle + 5];
}

static int strength(const h264d_mb_t *p, int bp, const h264d_mb_t *q, int bq,
                 bool macroblock_edge)
{
    if (p->intra || q->intra)
        return macroblock_edge ? 4 : 3;

    if (has_coefficients(p, bp) || has_coefficients(q, bq))
        return 2;

    /* Clause 8.7.2.1, the motion test. Two blocks predicted from different
     * pictures, or from a different number of them, or whose vectors differ
     * by a whole sample in any component, get a strength of 1.
     *
     * ⚠️ The comparison is between reference PICTURES, not between reference
     * indices. The same picture can sit at different indices in the two
     * lists, and two blocks pointing at it through different indices are
     * pointing at the same thing. Comparing indices puts an edge where the
     * encoder put none, which is visible as faint grid lines on smooth
     * motion.
     */
    /* The 8x8 partition a 4x4 block belongs to. With b = y4*4 + x4, that is
     * bit 1 of y4 for the row and bit 1 of x4 for the column, i.e. bits 3
     * and 1 of b - not bit 0, which is the low bit of x4. */
    const int pp = ((bp >> 2) & 2) | ((bp >> 1) & 1);
    const int pq = ((bq >> 2) & 2) | ((bq >> 1) & 1);
    const int8_t rp0 = p->ref[0][pp];
    const int8_t rp1 = p->ref[1][pp];
    const int8_t rq0 = q->ref[0][pq];
    const int8_t rq1 = q->ref[1][pq];

    const int np = (rp0 >= 0) + (rp1 >= 0);
    const int nq = (rq0 >= 0) + (rq1 >= 0);
    if (np != nq)
        return 1;

    const int16_t *mp0 = p->mv[0][bp], *mp1 = p->mv[1][bp];
    const int16_t *mq0 = q->mv[0][bq], *mq1 = q->mv[1][bq];

    if (np == 1) {
        const int8_t rp = rp0 >= 0 ? rp0 : rp1;
        const int8_t rq = rq0 >= 0 ? rq0 : rq1;
        if (rp != rq)
            return 1;
        const int16_t *mp = rp0 >= 0 ? mp0 : mp1;
        const int16_t *mq = rq0 >= 0 ? mq0 : mq1;
        return (abs(mp[0] - mq[0]) >= 4 || abs(mp[1] - mq[1]) >= 4) ? 1 : 0;
    }

    /* Two predictions each. They may be listed the other way round, and when
     * both point at the same picture either pairing counts as a match. */
    if (rp0 == rq0 && rp1 == rq1) {
        if (rp0 == rp1) {
            /* both lists on the same picture: either pairing will do */
            bool straight = abs(mp0[0] - mq0[0]) < 4 && abs(mp0[1] - mq0[1]) < 4
                       && abs(mp1[0] - mq1[0]) < 4 && abs(mp1[1] - mq1[1]) < 4;
            bool crossed_one = abs(mp0[0] - mq1[0]) < 4 && abs(mp0[1] - mq1[1]) < 4
                           && abs(mp1[0] - mq0[0]) < 4 && abs(mp1[1] - mq0[1]) < 4;
            return (straight || crossed_one) ? 0 : 1;
        }
        return (abs(mp0[0] - mq0[0]) < 4 && abs(mp0[1] - mq0[1]) < 4
             && abs(mp1[0] - mq1[0]) < 4 && abs(mp1[1] - mq1[1]) < 4) ? 0 : 1;
    }
    if (rp0 == rq1 && rp1 == rq0) {
        return (abs(mp0[0] - mq1[0]) < 4 && abs(mp0[1] - mq1[1]) < 4
             && abs(mp1[0] - mq0[0]) < 4 && abs(mp1[1] - mq0[1]) < 4) ? 0 : 1;
    }
    return 1;
}

/* ------------------------------------------------------------ the picture */

/* Clause 8.5.8. At eight bits QpBdOffsetC is zero, so qPI never goes
 * negative and the table covers the whole range. */
static inline int qp_chroma(int qpy, int offset)
{
    return h264d_chroma_qp[clip3(0, 51, qpy + offset)];
}

/* One edge of four lines, luma. */
static void luma_edge(uint8_t *q, int d, int stride, int bs,
                       int qp_p, int qp_q, int off_a, int off_b)
{
    if (!bs) return;
    const int qpav = (qp_p + qp_q + 1) >> 1;
    const int ia = clip3(0, 51, qpav + off_a);
    const int ib = clip3(0, 51, qpav + off_b);
    const int alpha = h264d_alpha[ia];
    const int beta = h264d_beta[ib];
    if (!alpha || !beta) return;

    if (bs == 4)
        for (int i = 0; i < 4; i++)
            luma_forte(q + (size_t)i * stride, d, alpha, beta);
    else {
        const int tc0 = h264d_tc0[ia][bs - 1];
        for (int i = 0; i < 4; i++)
            luma_plain(q + (size_t)i * stride, d, alpha, beta, tc0);
    }
}

/* One edge of two lines, chroma. */
static void chroma_edge(uint8_t *q, int d, int stride, int bs,
                        int qp_p, int qp_q, int off_a, int off_b)
{
    if (!bs) return;
    const int qpav = (qp_p + qp_q + 1) >> 1;
    const int ia = clip3(0, 51, qpav + off_a);
    const int ib = clip3(0, 51, qpav + off_b);
    const int alpha = h264d_alpha[ia];
    const int beta = h264d_beta[ib];
    if (!alpha || !beta) return;

    if (bs == 4)
        for (int i = 0; i < 2; i++)
            chroma_forte(q + (size_t)i * stride, d, alpha, beta);
    else {
        const int tc0 = h264d_tc0[ia][bs - 1];
        for (int i = 0; i < 2; i++)
            chroma_plain(q + (size_t)i * stride, d, alpha, beta, tc0);
    }
}

/* One macroblock's edges: the vertical ones left to right, then the
 * horizontal ones top to bottom, which is the order clause 8.7 fixes. */
void h264d_deblock_mb(const h264d_deblock_pic_t *p, int mx, int my)
{
    uint8_t *y = p->y, *cb = p->cb, *cr = p->cr;
    const int sy = p->stride_y, sc = p->stride_c;
    const int mb_w = p->mb_w;
    const h264d_mb_t *mbs = p->mbs;
    const uint8_t *slice_of_mb = p->slice_of_mb;
    const h264d_deblock_params_t *params = p->params;
    const int cqp_off = p->cqp_off, cqp_off2 = p->cqp_off2;
    const int idx = my * mb_w + mx;
    const h264d_mb_t *m = &mbs[idx];
    const h264d_deblock_params_t *pr = &params[slice_of_mb[idx]];
    if (pr->disable_idc == 1)
        return;

    const int oa = pr->alpha_offset, ob = pr->beta_offset;
    uint8_t *py = y + (size_t)my * 16 * sy + mx * 16;
    uint8_t *pcb = cb + (size_t)my * 8 * sc + mx * 8;
    uint8_t *pcr = cr + (size_t)my * 8 * sc + mx * 8;

    const bool skip_left = mx == 0
        || (pr->disable_idc == 2 && slice_of_mb[idx - 1] != slice_of_mb[idx]);
    const bool skip_above = my == 0
        || (pr->disable_idc == 2 && slice_of_mb[idx - mb_w] != slice_of_mb[idx]);

    const int qpc  = qp_chroma(m->qpy, cqp_off);
    const int qpc2 = qp_chroma(m->qpy, cqp_off2);

    /* --- vertical edges, left to right --------------------------- */
    for (int e = 0; e < 4; e++) {
        if (e == 0 && skip_left) continue;
        if (e && m->transform8x8 && (e & 1)) continue;

        const h264d_mb_t *neighbour = e ? m : &mbs[idx - 1];
        const int qp_p = neighbour->qpy;
        const int qpc_p  = qp_chroma(qp_p, cqp_off);
        const int qpc2_p = qp_chroma(qp_p, cqp_off2);

        /* Chroma has an edge only where luma has one every eight
         * samples, so only edges 0 and 2 - but the strength is the
         * same answer, derived once here for both. */
        const bool con_chroma = (e == 0 || e == 2);
        for (int r = 0; r < 4; r++) {
            const int bq = block(e, r);
            const int bp = e ? block(e - 1, r) : block(3, r);
            const int bs = strength(neighbour, bp, m, bq, e == 0);
            if (!bs) continue;              /* nothing is filtered */
            luma_edge(py + (size_t)r * 4 * sy + e * 4, 1, sy, bs,
                       qp_p, m->qpy, oa, ob);
            if (con_chroma) {
                chroma_edge(pcb + (size_t)r * 2 * sc + e * 2, 1, sc, bs,
                            qpc_p, qpc, oa, ob);
                chroma_edge(pcr + (size_t)r * 2 * sc + e * 2, 1, sc, bs,
                            qpc2_p, qpc2, oa, ob);
            }
        }
    }

    /* --- horizontal edges, top to bottom ------------------------- */
    for (int e = 0; e < 4; e++) {
        if (e == 0 && skip_above) continue;
        if (e && m->transform8x8 && (e & 1)) continue;

        const h264d_mb_t *neighbour = e ? m : &mbs[idx - mb_w];
        const int qp_p = neighbour->qpy;
        const int qpc_p  = qp_chroma(qp_p, cqp_off);
        const int qpc2_p = qp_chroma(qp_p, cqp_off2);

        const bool con_chroma = (e == 0 || e == 2);
        for (int c = 0; c < 4; c++) {
            const int bq = block(c, e);
            const int bp = e ? block(c, e - 1) : block(c, 3);
            const int bs = strength(neighbour, bp, m, bq, e == 0);
            if (!bs) continue;
            luma_edge(py + (size_t)e * 4 * sy + c * 4, sy, 1, bs,
                       qp_p, m->qpy, oa, ob);
            if (con_chroma) {
                chroma_edge(pcb + (size_t)e * 2 * sc + c * 2, sc, 1, bs,
                            qpc_p, qpc, oa, ob);
                chroma_edge(pcr + (size_t)e * 2 * sc + c * 2, sc, 1, bs,
                            qpc2_p, qpc2, oa, ob);
            }
        }
    }
}

void h264d_deblock_picture(uint8_t *y, int sy,
                           uint8_t *cb, uint8_t *cr, int sc,
                           int mb_w, int mb_h,
                           const h264d_mb_t *mbs,
                           const uint8_t *slice_of_mb,
                           const h264d_deblock_params_t *params,
                           int cqp_off, int cqp_off2)
{
    const h264d_deblock_pic_t p = {
        y, cb, cr, sy, sc, mb_w, mb_h, mbs, slice_of_mb, params,
        cqp_off, cqp_off2
    };
    for (int my = 0; my < mb_h; my++)
        for (int mx = 0; mx < mb_w; mx++)
            h264d_deblock_mb(&p, mx, my);
}
