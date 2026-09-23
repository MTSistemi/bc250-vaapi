/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * h264_mb_cabac.c - the macroblock layer read with CABAC, Rec. ITU-T H.264
 * clause 7.3.5 with the context derivations of 9.3.3.1.
 *
 * Every syntax element here is a handful of bins, and almost every one of
 * them picks its context from the macroblocks above and to the left. Getting
 * a context wrong does not desynchronise the bitstream straight away - CABAC
 * keeps decoding, just with the wrong probabilities - so the damage shows up
 * a few hundred bins later as a macroblock with an impossible type. That is
 * why the neighbour lookups are written once, at the top, rather than
 * open-coded at each use.
 *
 * ⚠️ A neighbour in a different slice is not available. A slice has to
 * decode on its own, which is what slices are for, and h264d_mb_left() and
 * friends return NULL across a slice boundary. The deblocking filter is the
 * only part that may look across, and it decides for itself.
 */
#include "h264_dec_internal.h"

#include <stdlib.h>
#include <string.h>

#include "h264_dec_tables.h"
#include "h264_parts.h"
#include "h264_pred.h"

/* Decoding order of the sixteen 4x4 luma blocks, as raster positions inside
 * the macroblock. The standard walks them in the Z order of 6.4.3; the
 * decoder stores everything in raster order, because every neighbour lookup
 * and the deblocking filter want raster. */
static const uint8_t zscan[16] = {
    0, 1, 4, 5,  2, 3, 6, 7,  8, 9, 12, 13,  10, 11, 14, 15
};

/* ---------------------------------------------------------- neighbours */

/* The 4x4 block to the left of raster block `b`, and the macroblock it is
 * in. Returns NULL when there is none. */
static const h264d_mb_t *left4(const h264_decoder_t *d, int b, int *out)
{
    *out = 0;                       /* set even when there is no neighbour */
    if (b & 3) { *out = b - 1; return &d->mbs[d->mb_idx]; }
    const h264d_mb_t *m = h264d_mb_left(d);
    if (!m) return NULL;
    *out = b + 3;
    return m;
}

static const h264d_mb_t *above4(const h264_decoder_t *d, int b, int *out)
{
    *out = 0;
    if (b >= 4) { *out = b - 4; return &d->mbs[d->mb_idx]; }
    const h264d_mb_t *m = h264d_mb_top(d);
    if (!m) return NULL;
    *out = b + 12;
    return m;
}

/* The same for a chroma 4x4 block, which is one of a 2x2 grid. */
static const h264d_mb_t *left_chroma(const h264_decoder_t *d, int b, int *out)
{
    *out = 0;                       /* set even when there is no neighbour */
    if (b & 1) { *out = b - 1; return &d->mbs[d->mb_idx]; }
    const h264d_mb_t *m = h264d_mb_left(d);
    if (!m) return NULL;
    *out = b + 1;
    return m;
}

static const h264d_mb_t *above_chroma(const h264_decoder_t *d, int b, int *out)
{
    *out = 0;
    if (b >= 2) { *out = b - 2; return &d->mbs[d->mb_idx]; }
    const h264d_mb_t *m = h264d_mb_top(d);
    if (!m) return NULL;
    *out = b + 2;
    return m;
}

/* ------------------------------------------------------ syntax elements */

static inline int decide(h264_decoder_t *d, int ctx)
{
    return h264d_cabac_decision(&d->cabac, ctx);
}

/* 9.3.3.1.1.1 */
static int read_mb_skip(h264_decoder_t *d, bool bslice)
{
    const h264d_mb_t *a = h264d_mb_left(d);
    const h264d_mb_t *b = h264d_mb_top(d);
    int inc = 0;
    if (a && a->type != H264D_MB_P_SKIP && a->type != H264D_MB_B_SKIP) inc++;
    if (b && b->type != H264D_MB_P_SKIP && b->type != H264D_MB_B_SKIP) inc++;
    return decide(d, (bslice ? CTX_MB_SKIP_B : CTX_MB_SKIP_P) + inc);
}

/* 9.3.2.5 and Table 9-36: the intra part of mb_type, shared by the I-slice
 * form and the intra escape of a P or B slice. Returns 0 for I_NxN, 25 for
 * I_PCM, 1..24 for the Intra_16x16 variants. */
static int read_mb_type_intra(h264_decoder_t *d, int base, bool in_i_slice)
{
    int row_state = base;
    if (in_i_slice) {
        const h264d_mb_t *a = h264d_mb_left(d);
        const h264d_mb_t *b = h264d_mb_top(d);
        int inc = 0;
        if (a && a->type != H264D_MB_I_NxN) inc++;
        if (b && b->type != H264D_MB_I_NxN) inc++;
        if (!decide(d, base + inc))
            return 0;
        row_state = base + 2;
    } else {
        if (!decide(d, base))
            return 0;
    }

    /* ⚠️ The second bin is the terminate decision, not an ordinary one. It
     * is the only place outside end_of_slice_flag that uses it, and reading
     * it as a normal bin leaves codIRange two too high from here on. */
    if (h264d_cabac_terminate(&d->cabac))
        return 25;                      /* I_PCM */

    int t = 1;
    t += 12 * decide(d, row_state + 1);     /* cbp_luma: 0 or 15 */
    if (decide(d, row_state + 2))           /* cbp_chroma */
        t += 4 + 4 * decide(d, row_state + 2 + (in_i_slice ? 1 : 0));
    t += 2 * decide(d, row_state + 3 + (in_i_slice ? 1 : 0));
    t += 1 * decide(d, row_state + 3 + (in_i_slice ? 2 : 0));
    return t;
}

/* Table 9-37, P slices. Returns the standard's mb_type, with the intra
 * types offset by 5 as the standard numbers them. */
static int read_mb_type_p(h264_decoder_t *d)
{
    if (!decide(d, CTX_MB_TYPE_P)) {
        if (!decide(d, CTX_MB_TYPE_P + 1))
            return 3 * decide(d, CTX_MB_TYPE_P + 2);      /* 16x16 or 8x8 */
        return 2 - decide(d, CTX_MB_TYPE_P + 3);          /* 16x8 or 8x16 */
    }
    return read_mb_type_intra(d, CTX_MB_TYPE_P + 3, false) + 5;
}

/* Table 9-37, B slices. */
static int read_mb_type_b(h264_decoder_t *d)
{
    const h264d_mb_t *a = h264d_mb_left(d);
    const h264d_mb_t *b = h264d_mb_top(d);
    int inc = 0;
    if (a && a->type != H264D_MB_B_SKIP && a->type != H264D_MB_B_DIRECT) inc++;
    if (b && b->type != H264D_MB_B_SKIP && b->type != H264D_MB_B_DIRECT) inc++;

    if (!decide(d, CTX_MB_TYPE_B + inc))
        return 0;                                   /* B_Direct_16x16 */
    if (!decide(d, CTX_MB_TYPE_B + 3))
        return 1 + decide(d, CTX_MB_TYPE_B + 5);    /* B_L0_16x16, B_L1_16x16 */

    int bits = decide(d, CTX_MB_TYPE_B + 4) << 3;
    bits |= decide(d, CTX_MB_TYPE_B + 5) << 2;
    bits |= decide(d, CTX_MB_TYPE_B + 5) << 1;
    bits |= decide(d, CTX_MB_TYPE_B + 5);
    if (bits < 8)
        return bits + 3;
    if (bits == 13)
        return read_mb_type_intra(d, CTX_MB_TYPE_B + 5, false) + 23;
    if (bits == 14)
        return 11;                                  /* B_L1_L0_8x16 */
    if (bits == 15)
        return 22;                                  /* B_8x8 */
    bits = (bits << 1) | decide(d, CTX_MB_TYPE_B + 5);
    return bits - 4;
}

static int read_sub_mb_type_p(h264_decoder_t *d)
{
    if (decide(d, CTX_SUB_MB_TYPE_P))     return 0;   /* 8x8 */
    if (!decide(d, CTX_SUB_MB_TYPE_P + 1)) return 1;  /* 8x4 */
    if (decide(d, CTX_SUB_MB_TYPE_P + 2)) return 2;   /* 4x8 */
    return 3;                                         /* 4x4 */
}

static int read_sub_mb_type_b(h264_decoder_t *d)
{
    if (!decide(d, CTX_SUB_MB_TYPE_B))
        return 0;                                     /* B_Direct_8x8 */
    if (!decide(d, CTX_SUB_MB_TYPE_B + 1))
        return 1 + decide(d, CTX_SUB_MB_TYPE_B + 3);
    int t = 3;
    if (decide(d, CTX_SUB_MB_TYPE_B + 2)) {
        if (decide(d, CTX_SUB_MB_TYPE_B + 3))
            return 11 + decide(d, CTX_SUB_MB_TYPE_B + 3);
        t += 4;
    }
    t += 2 * decide(d, CTX_SUB_MB_TYPE_B + 3);
    t += decide(d, CTX_SUB_MB_TYPE_B + 3);
    return t;
}

/* 9.3.3.1.1.6 */
static int read_ref_idx(h264_decoder_t *d, int list_idx, int blk8, int cap)
{
    if (cap <= 1) return 0;

    const h264d_mb_t *mb = &d->mbs[d->mb_idx];
    int inc = 0;
    /* The neighbouring 8x8 partitions: to the left and above this one. */
    const int bx = blk8 & 1, by = blk8 >> 1;
    const h264d_mb_t *a;
    int pa;
    if (bx) { a = mb; pa = blk8 - 1; }
    else    { a = h264d_mb_left(d); pa = blk8 + 1; }
    const h264d_mb_t *b;
    int pb;
    if (by) { b = mb; pb = blk8 - 2; }
    else    { b = h264d_mb_top(d); pb = blk8 + 2; }

    /* ⚠️ The INDEX the slice signalled, not the frame store slot. Almost
     * every reference sits in a slot above zero, so asking the slot instead
     * made this context read as "the neighbour used a high reference" nearly
     * always. */
    /* ⚠️ A direct neighbour does not count, whatever index the derivation
     * gave it: it signalled nothing, so there is nothing to have been
     * large. */
    if (a && !a->intra && !((a->direct >> pa) & 1) && a->ref_idx[list_idx][pa] > 0)
        inc += 1;
    if (b && !b->intra && !((b->direct >> pb) & 1) && b->ref_idx[list_idx][pb] > 0)
        inc += 2;

    if (!decide(d, CTX_REF_IDX + inc))
        return 0;
    if (!decide(d, CTX_REF_IDX + 4))
        return 1;

    /* ⚠️ Table 9-34 gives ref_idx an UNBOUNDED unary binarization, so even
     * the largest legal index is followed by a terminating zero. Stopping at
     * the top of the range without consuming that zero leaves the engine one
     * bin behind for the rest of the slice. The limit below is only there so
     * a corrupt stream cannot spin. */
    int v = 2;
    while (v < 32 && decide(d, CTX_REF_IDX + 5))
        v++;
    if (v > cap - 1)
        v = cap - 1;
    return v;
}

/* 9.3.3.1.1.7, then the UEG3 binarization of 9.3.2.3. */
static int read_mvd(h264_decoder_t *d, int comp, int sum_neighbours)
{
    const int base = comp ? CTX_MVD_Y : CTX_MVD_X;
    int inc = sum_neighbours < 3 ? 0 : (sum_neighbours > 32 ? 2 : 1);

    if (!decide(d, base + inc))
        return 0;

    /* ⚠️ Table 9-39: after the first bin the context increments run 3, 4, 5,
     * 6 and then stay at 6. Counting them from the wrong end gave 4, 5, 6, 8
     * - and a wrong context does not fail loudly, it just decodes the wrong
     * bin, so the unary prefix closed early and the vector came out short by
     * two quarter-samples. */
    int v = 1;
    while (v < 9 && decide(d, base + (v + 2 < 6 ? v + 2 : 6)))
        v++;

    /* UEG3 with uCoff = 9: only a full run of nine ones has a suffix. */
    if (v == 9)
        v += (int)h264d_cabac_eg_bypass(&d->cabac, 3);

    return h264d_cabac_bypass(&d->cabac) ? -v : v;
}

/* 9.3.3.1.1.4. ⚠️ The condition is inverted with respect to every other
 * neighbour test here: the context counts 8x8 blocks that are NOT coded. */
static int read_cbp(h264_decoder_t *d)
{
    const h264d_mb_t *mb = &d->mbs[d->mb_idx];
    int cbp = 0;

    for (int i = 0; i < 4; i++) {
        const int bx = i & 1, by = i >> 1;
        int va, vb;
        if (bx) va = (cbp >> (i - 1)) & 1;
        else {
            const h264d_mb_t *a = h264d_mb_left(d);
            va = a ? (a->cbp >> (i + 1)) & 1 : 1;
        }
        if (by) vb = (cbp >> (i - 2)) & 1;
        else {
            const h264d_mb_t *b = h264d_mb_top(d);
            vb = b ? (b->cbp >> (i + 2)) & 1 : 1;
        }
        const int inc = (va ? 0 : 1) + (vb ? 0 : 2);
        cbp |= decide(d, CTX_CBP_LUMA + inc) << i;
    }
    (void)mb;

    /* 9.3.3.1.1.4 for chroma: the first bin asks whether either neighbour
     * has any chroma coefficients, the second whether either has AC ones. */
    const h264d_mb_t *a = h264d_mb_left(d);
    const h264d_mb_t *b = h264d_mb_top(d);
    int ca = a ? (a->cbp >> 4) : 0;
    int cb = b ? (b->cbp >> 4) : 0;
    int inc = (ca ? 1 : 0) + (cb ? 2 : 0);
    if (decide(d, CTX_CBP_CHROMA + inc)) {
        inc = 4 + (ca == 2 ? 1 : 0) + (cb == 2 ? 2 : 0);
        cbp |= (decide(d, CTX_CBP_CHROMA + inc) ? 2 : 1) << 4;
    }
    return cbp;
}

/* 9.3.3.1.1.5, then the mapping of 9.3.2.7. */
static int read_qp_delta(h264_decoder_t *d)
{
    int inc = d->last_qp_delta_nonzero ? 1 : 0;
    if (!decide(d, CTX_MB_QP_DELTA + inc)) {
        d->last_qp_delta_nonzero = 0;
        return 0;
    }
    int k = 1;
    if (decide(d, CTX_MB_QP_DELTA + 2)) {
        k = 2;
        while (k < 96 && decide(d, CTX_MB_QP_DELTA + 3))
            k++;
    }
    d->last_qp_delta_nonzero = 1;
    /* value k maps to (-1)^(k+1) * Ceil(k / 2) */
    return (k & 1) ? (k + 1) / 2 : -(k / 2);
}

/* 9.3.3.1.1.8 */
static int read_chroma_pred_mode(h264_decoder_t *d)
{
    const h264d_mb_t *a = h264d_mb_left(d);
    const h264d_mb_t *b = h264d_mb_top(d);
    int inc = 0;
    if (a && a->intra && a->type != H264D_MB_I_PCM && a->chroma_pred_mode != 0) inc++;
    if (b && b->intra && b->type != H264D_MB_I_PCM && b->chroma_pred_mode != 0) inc++;

    if (!decide(d, CTX_INTRA_CHROMA_PRED + inc))
        return 0;
    if (!decide(d, CTX_INTRA_CHROMA_PRED + 3))
        return 1;
    return decide(d, CTX_INTRA_CHROMA_PRED + 3) ? 3 : 2;
}

static int read_transform8x8(h264_decoder_t *d)
{
    const h264d_mb_t *a = h264d_mb_left(d);
    const h264d_mb_t *b = h264d_mb_top(d);
    int inc = (a && a->transform8x8) + (b && b->transform8x8);
    return decide(d, CTX_TRANSFORM_8X8 + inc);
}

/* ------------------------------------------------------------- residual */

/* 9.3.3.1.1.9, condTermFlagN for coded_block_flag. */
static int cbf_neighbour(const h264_decoder_t *d, const h264d_mb_t *n, int block,
                      int plane, bool dc, bool current_intra)
{
    if (!n)
        return current_intra ? 1 : 0;
    if (n->type == H264D_MB_I_PCM)
        return 1;
    if (n->type == H264D_MB_P_SKIP || n->type == H264D_MB_B_SKIP)
        return 0;
    if (dc)
        return n->cbf_dc[plane] ? 1 : 0;
    return n->nnz[plane][block] ? 1 : 0;
    (void)d;
}

/* Clause 7.3.5.3.3 with the contexts of 9.3.3.1.3.
 *
 * `cat` is the block category of Table 9-42, `n` the number of coefficients
 * the block can hold, and `out` receives them in the scan order the caller
 * asked for - the caller un-scans them, because the 4x4 and 8x8 scans differ
 * and the DC blocks use a third one.
 *
 * Returns how many coefficients are non-zero, which is what the CAVLC
 * context of the next macroblock and the deblocking filter both want.
 */
static int read_residual(h264_decoder_t *d, int cat, int n, int16_t *out,
                         int cbf_inc, bool read_cbf)
{
    memset(out, 0, (size_t)n * sizeof(int16_t));

    if (read_cbf) {
        const int ctx = CTX_CBF + 4 * cat + cbf_inc;
        if (!decide(d, ctx))
            return 0;
    }

    const int field = 0;    /* progressive only */
    const int base_sig  = h264d_sig_coeff_offset[field][cat];
    const int base_last = h264d_last_coeff_offset[field][cat];
    const int base_abs  = h264d_abs_level_offset[cat];
    const bool otto = (cat == CAT_LUMA_8X8);

    uint8_t significant[64];
    memset(significant, 0, sizeof(significant));

    int last = n - 1;
    int i = 0;
    while (i < n - 1) {
        int inc_sig, inc_last;
        if (otto) {
            inc_sig = h264d_sig_coeff_offset_8x8[field][i];
            inc_last = h264d_last_coeff_offset_8x8[i];
        } else if (cat == CAT_CHROMA_DC) {
            /* NumC8x8 is 1 at 4:2:0, so the index is just i capped at 2. */
            inc_sig = inc_last = i < 2 ? i : 2;
        } else {
            inc_sig = inc_last = i;
        }
        if (decide(d, base_sig + inc_sig)) {
            significant[i] = 1;
            if (decide(d, base_last + inc_last)) {
                last = i;
                break;
            }
        }
        i++;
    }
    if (i == n - 1)
        significant[i] = 1;      /* ran to the end: the last one must be */
    else
        significant[last] = 1;

    /* Levels, read backwards from the last significant coefficient. */
    int n_ones = 0, n_greater = 0, nonzero = 0;
    for (int k = last; k >= 0; k--) {
        if (!significant[k])
            continue;

        int inc = n_greater ? 0
                                  : (1 + n_ones < 4 ? 1 + n_ones : 4);
        int level = 1;
        if (decide(d, base_abs + inc)) {
            /* The tail uses a second context group, capped one lower for
             * chroma DC because that block has fewer coefficients. */
            const int ctx_cap = (cat == CAT_CHROMA_DC) ? 3 : 4;

            /* ⚠️ Fourteen, not thirteen. coeff_abs_level_minus1 is UEG0 with
             * uCoff = 14 (9.3.2.3): the prefix is a run of up to fourteen
             * ones, and only a full run of fourteen is followed by the
             * Exp-Golomb suffix. A run shorter than that ends with a zero
             * bin, which the loop below consumes on the way out. */
            int v = 1;
            while (v < 14) {
                inc = 5 + (n_greater < ctx_cap ? n_greater : ctx_cap);
                if (!decide(d, base_abs + inc))
                    break;
                v++;
            }
            if (v == 14)
                v += (int)h264d_cabac_eg_bypass(&d->cabac, 0);
            level = v + 1;
            n_greater++;
        } else {
            n_ones++;
        }

        out[k] = (int16_t)(h264d_cabac_bypass(&d->cabac) ? -level : level);
        nonzero++;
    }
    return nonzero;
}


/* ------------------------------------------------------------- motion */

/* Write one partition's vector and reference into every 4x4 block it
 * covers. */
static void place(h264d_mb_t *m, int list_idx, int blk, int w4, int h4,
                 int ref, int slot, int16_t mvx, int16_t mvy,
                 int16_t dx, int16_t dy)
{
    const int x4 = blk & 3, y4 = blk >> 2;
    for (int y = 0; y < h4; y++) {
        for (int x = 0; x < w4; x++) {
            const int b = (y4 + y) * 4 + x4 + x;
            m->mv[list_idx][b][0] = mvx;
            m->mv[list_idx][b][1] = mvy;
            m->mvd[list_idx][b][0] = dx;
            m->mvd[list_idx][b][1] = dy;
        }
    }
    /* The reference is kept per 8x8, so each covered 4x4 stamps the 8x8 it
     * belongs to. A partition either sits inside one 8x8 or covers whole
     * ones, so this never writes two different references into the same
     * slot. */
    for (int y = 0; y < h4; y++)
        for (int x = 0; x < w4; x++) {
            const int p = h264d_part8(((y4 + y) * 4) + x4 + x);
            m->ref[list_idx][p] = (int8_t)slot;
            m->ref_idx[list_idx][p] = (int8_t)ref;
        }
}

/* The sum of the neighbours' coded differences, which picks the mvd context
 * (9.3.3.1.1.7). Both neighbours are looked up in the macroblock store, so a
 * partition of this same macroblock that has already been read counts too. */
static int add_mvd(const h264_decoder_t *d, int list_idx, int blk, int comp)
{
    int s = 0;
    int pa, pb;
    const h264d_mb_t *a = left4(d, blk, &pa);
    const h264d_mb_t *b = above4(d, blk, &pb);
    if (a && !a->intra) s += abs(a->mvd[list_idx][pa][comp]);
    if (b && !b->intra) s += abs(b->mvd[list_idx][pb][comp]);
    return s;
}

static inline bool usa(int pred, int list_idx)
{
    if (pred == H264D_PRED_BI) return true;
    return pred == (list_idx ? H264D_PRED_L1 : H264D_PRED_L0);
}

/* Clause 7.3.5.1 and 7.3.5.2, and the derivation of 8.4.1.
 *
 * ⚠️ Two nested levels read at different granularities. A macroblock has
 * one, two or four PARTITIONS; each has one, two or four SUB-partitions,
 * and only a P_8x8 or B_8x8 has more than one. A reference index is
 * signalled per partition, a motion vector difference per sub-partition.
 *
 * ⚠️ And the syntax comes in passes, not in one walk: every ref_idx of list
 * 0, then every ref_idx of list 1, then every difference of list 0, then of
 * list 1. That is the order the bitstream holds them in.
 *
 * ⚠️ The DERIVATION is a third order again: partition by partition, because
 * each one's vector is predicted from the ones before it - and a direct
 * partition is part of that sequence, not a tidying step afterwards.
 */
static int read_motion(h264_decoder_t *d, h264d_mb_t *m, bool bslice, int t)
{
    struct {
        uint8_t pred;
        uint8_t n;                  /* sub-partitions */
        uint8_t blk[4];             /* raster 4x4 index of each */
        uint8_t w4, h4;             /* size of a sub-partition */
        int8_t  ref[2];
        int16_t dx[2][4], dy[2][4]; /* the differences as read */
    } part[4];
    int np = 0;
    bool below_8x8 = false;

    if (m->type == H264D_MB_P_8x8 || m->type == H264D_MB_B_8x8) {
        h264d_sub_t sub[4];
        for (int i = 0; i < 4; i++) {
            const int s = bslice ? read_sub_mb_type_b(d) : read_sub_mb_type_p(d);
            m->sub_type[i] = (int8_t)s;
            sub[i] = bslice ? h264d_sub_b[s >= 0 && s < 13 ? s : 12]
                            : h264d_sub_p[s >= 0 && s < 4 ? s : 3];
            /* noSubMbPartSizeLessThan8x8Flag, clause 7.3.5. A
             * sub-macroblock cut smaller than 8x8 clears it - but a
             * B_Direct_8x8 does not, unless direct_8x8_inference_flag is
             * off.
             *
             * ⚠️ B_Direct_8x8 is modelled here as four 4x4 pieces, because
             * that is how its vectors are derived, so the size test catches
             * it and has to be told not to. Otherwise a B_8x8 with one
             * direct sub-macroblock never reads transform_size_8x8_flag and
             * the slice comes apart from there. */
            if (bslice && s == 0) {
                if (!d->pic.direct_8x8_inference)
                    below_8x8 = true;
            } else if (sub[i].w4 < 2 || sub[i].h4 < 2) {
                below_8x8 = true;
            }
        }
        for (int i = 0; i < 4; i++) {
            part[i].pred = sub[i].pred;
            part[i].n = sub[i].n;
            part[i].w4 = sub[i].w4;
            part[i].h4 = sub[i].h4;
            for (int k = 0; k < sub[i].n; k++)
                part[i].blk[k] = (uint8_t)(h264d_blk8[i]
                                            + h264d_sub_offset(&sub[i], k));
        }
        np = 4;
    } else if (m->type == H264D_MB_P_16x16 || m->type == H264D_MB_B_16x16) {
        part[0].pred = (uint8_t)(bslice ? (t == 1 ? H264D_PRED_L0
                                          : t == 2 ? H264D_PRED_L1
                                                   : H264D_PRED_BI)
                                         : H264D_PRED_L0);
        part[0].n = 1; part[0].blk[0] = 0; part[0].w4 = 4; part[0].h4 = 4;
        np = 1;
    } else if (m->type == H264D_MB_B_DIRECT) {
        part[0].pred = H264D_PRED_DIRECT;
        part[0].n = 1; part[0].blk[0] = 0; part[0].w4 = 4; part[0].h4 = 4;
        np = 1;
    } else {
        const bool vertical = (m->type == H264D_MB_P_8x16
                             || m->type == H264D_MB_B_8x16);
        const int w4 = vertical ? 2 : 4, h4 = vertical ? 4 : 2;
        const int second = vertical ? 2 : 8;
        uint8_t p0 = H264D_PRED_L0, p1 = H264D_PRED_L0;
        if (bslice) {
            int pair = (t - 4) / 2;
            if (pair < 0) pair = 0;
            if (pair > 8) pair = 8;
            p0 = h264d_b_pair[pair][0];
            p1 = h264d_b_pair[pair][1];
        }
        part[0].pred = p0; part[0].n = 1; part[0].blk[0] = 0;
        part[0].w4 = (uint8_t)w4; part[0].h4 = (uint8_t)h4;
        part[1].pred = p1; part[1].n = 1; part[1].blk[0] = (uint8_t)second;
        part[1].w4 = (uint8_t)w4; part[1].h4 = (uint8_t)h4;
        np = 2;
    }

    m->sub_8x8 = below_8x8 ? 1 : 0;

    /* Which 8x8s are direct. The ref_idx context of 9.3.3.1.1.6 does not
     * count a direct neighbour, whatever index the derivation gives it. */
    m->direct = 0;
    for (int i = 0; i < np; i++)
        if (part[i].pred == H264D_PRED_DIRECT)
            m->direct |= (uint8_t)(np == 1 ? 0xf : (1 << i));

    for (int i = 0; i < np; i++) {
        part[i].ref[0] = part[i].ref[1] = -1;
        memset(part[i].dx, 0, sizeof(part[i].dx));
        memset(part[i].dy, 0, sizeof(part[i].dy));
    }

    /* ---- reference indices, one per partition, list 0 then list 1 ---- */
    for (int list_idx = 0; list_idx < (bslice ? 2 : 1); list_idx++) {
        const int cap = d->slice.num_ref_idx[list_idx];
        for (int i = 0; i < np; i++) {
            if (part[i].pred == H264D_PRED_DIRECT || !usa(part[i].pred, list_idx))
                continue;
            const int r = read_ref_idx(d, list_idx, h264d_part8(part[i].blk[0]), cap);
            part[i].ref[list_idx] = (int8_t)r;
            /* In the store straight away: the next partition's ref_idx
             * context asks what this one used. */
            for (int k = 0; k < part[i].n; k++)
                place(m, list_idx, part[i].blk[k], part[i].w4, part[i].h4,
                     r, d->slice.ref_list[list_idx][r], 0, 0, 0, 0);
        }
    }

    /* ---- differences, one per sub-partition ---- */
    for (int list_idx = 0; list_idx < (bslice ? 2 : 1); list_idx++) {
        for (int i = 0; i < np; i++) {
            if (part[i].pred == H264D_PRED_DIRECT || !usa(part[i].pred, list_idx))
                continue;
            for (int k = 0; k < part[i].n; k++) {
                const int b = part[i].blk[k];
                const int dx = read_mvd(d, 0, add_mvd(d, list_idx, b, 0));
                const int dy = read_mvd(d, 1, add_mvd(d, list_idx, b, 1));
                part[i].dx[list_idx][k] = (int16_t)dx;
                part[i].dy[list_idx][k] = (int16_t)dy;
                /* The difference goes in now, because the next one's
                 * context is the sum of its neighbours' differences. The
                 * vector itself waits for the derivation pass. */
                const int x4 = b & 3, y4 = b >> 2;
                for (int yy = 0; yy < part[i].h4; yy++)
                    for (int xx = 0; xx < part[i].w4; xx++) {
                        const int bb = (y4 + yy) * 4 + x4 + xx;
                        m->mvd[list_idx][bb][0] = (int16_t)dx;
                        m->mvd[list_idx][bb][1] = (int16_t)dy;
                    }
            }
        }
    }

    /* ---- the derivation, partition by partition ---- */
    for (int i = 0; i < np; i++) {
        if (part[i].pred == H264D_PRED_DIRECT) {
            const int mask = (np == 1) ? 0xf : (1 << i);
            if (h264d_direct(d, m, mask) != 0)
                return 1;
            continue;
        }
        for (int list_idx = 0; list_idx < (bslice ? 2 : 1); list_idx++) {
            if (!usa(part[i].pred, list_idx))
                continue;
            const int r = part[i].ref[list_idx];
            for (int k = 0; k < part[i].n; k++) {
                const int b = part[i].blk[k];
                int16_t pmv[2];
                h264d_predict_mv(d, list_idx, b, part[i].w4, part[i].h4, r, pmv);
                place(m, list_idx, b, part[i].w4, part[i].h4,
                     r, d->slice.ref_list[list_idx][r],
                     (int16_t)(pmv[0] + part[i].dx[list_idx][k]),
                     (int16_t)(pmv[1] + part[i].dy[list_idx][k]),
                     part[i].dx[list_idx][k], part[i].dy[list_idx][k]);
            }
        }
    }

    return 0;
}

/* --------------------------------------------------------- the macroblock */

static void azzera_mb(h264d_mb_t *m)
{
    memset(m, 0, sizeof(*m));
    /* ⚠️ Both of them. A partition that does not use a list has to say so
     * through the INDEX as well as the slot: the vector prediction of
     * 8.4.1.3 compares indices, so an index left at zero makes a neighbour
     * that predicts from nothing claim it used reference zero. */
    memset(m->ref, -1, sizeof(m->ref));
    memset(m->ref_idx, -1, sizeof(m->ref_idx));
    memset(m->sub_type, -1, sizeof(m->sub_type));
}

/* The residual of one macroblock, clause 7.3.5.3. */
static void read_residual_mb(h264_decoder_t *d, h264d_mb_t *m, bool i16)
{
    const bool intra = m->intra;

    if (i16) {
        const h264d_mb_t *a; int ba;
        const h264d_mb_t *b; int bb;
        a = h264d_mb_left(d); ba = 0;
        b = h264d_mb_top(d);  bb = 0;
        int inc = cbf_neighbour(d, a, ba, 0, true, intra)
                + 2 * cbf_neighbour(d, b, bb, 0, true, intra);
        int16_t tmp[16];
        int nz = read_residual(d, CAT_I16_DC, 16, tmp, inc, true);
        for (int k = 0; k < 16; k++)
            d->res->dc_luma[h264d_zigzag4[k]] = tmp[k];
        m->cbf_dc[0] = nz ? 1 : 0;
    }

    /* Luma. */
    if (m->transform8x8) {
        for (int b8 = 0; b8 < 4; b8++) {
            if (!((m->cbp >> b8) & 1)) {
                memset(d->res->coeff8[b8], 0, sizeof(d->res->coeff8[b8]));
                continue;
            }
            /* No coded_block_flag for an 8x8 luma block at 4:2:0: the
             * coded block pattern has already said it is there. */
            int16_t tmp8[64];
            int nz = read_residual(d, CAT_LUMA_8X8, 64, tmp8, 0, false);
            memset(d->res->coeff8[b8], 0, sizeof(d->res->coeff8[b8]));
            for (int k = 0; k < 64; k++)
                d->res->coeff8[b8][h264d_zigzag8[k]] = tmp8[k];
            const int bx = (b8 & 1) * 2, by = (b8 >> 1) * 2;
            for (int y = 0; y < 2; y++)
                for (int x = 0; x < 2; x++)
                    m->nnz[0][(by + y) * 4 + bx + x] = (uint8_t)(nz ? 1 : 0);
        }
    } else {
        for (int k = 0; k < 16; k++) {
            const int b = zscan[k];
            const int b8 = ((b >> 3) << 1) | ((b >> 1) & 1);
            if (!((m->cbp >> b8) & 1)) {
                memset(d->res->luma[b], 0, sizeof(d->res->luma[b]));
                m->nnz[0][b] = 0;
                continue;
            }
            int pa, pb;
            const h264d_mb_t *a = left4(d, b, &pa);
            const h264d_mb_t *bmb = above4(d, b, &pb);
            int inc = cbf_neighbour(d, a, pa, 0, false, intra)
                    + 2 * cbf_neighbour(d, bmb, pb, 0, false, intra);
            const int cat = i16 ? CAT_I16_AC : CAT_LUMA_4X4;
            const int n = i16 ? 15 : 16;
            int16_t tmp[16];
            int nz = read_residual(d, cat, n, tmp, inc, true);
            /* An Intra_16x16 block codes only its fifteen AC coefficients;
             * the DC comes from the separate block above. */
            memset(d->res->luma[b], 0, sizeof(d->res->luma[b]));
            if (i16) {
                /* The AC coefficients start at scan position 1: position 0
                 * is the DC, which came from its own block. */
                for (int k = 0; k < 15; k++)
                    d->res->luma[b][h264d_zigzag4[k + 1]] = tmp[k];
            } else {
                for (int k = 0; k < 16; k++)
                    d->res->luma[b][h264d_zigzag4[k]] = tmp[k];
            }
            m->nnz[0][b] = (uint8_t)nz;
        }
    }

    /* Chroma DC, then chroma AC. */
    const int cbp_c = m->cbp >> 4;
    for (int p = 0; p < 2; p++) {
        if (cbp_c) {
            const h264d_mb_t *a = h264d_mb_left(d);
            const h264d_mb_t *b = h264d_mb_top(d);
            int inc = cbf_neighbour(d, a, 0, p + 1, true, intra)
                    + 2 * cbf_neighbour(d, b, 0, p + 1, true, intra);
            /* ⚠️ No un-scan here. At 4:2:0 the chroma DC block is 2x2 and
             * its scan (h264d_chroma_dc_scan) is the identity, so the four
             * coefficients come out already in raster order. */
            int nz = read_residual(d, CAT_CHROMA_DC, 4, d->res->dc_chroma[p], inc, true);
            m->cbf_dc[p + 1] = nz ? 1 : 0;
        } else {
            memset(d->res->dc_chroma[p], 0, sizeof(d->res->dc_chroma[p]));
            m->cbf_dc[p + 1] = 0;
        }
    }
    for (int p = 0; p < 2; p++) {
        for (int b = 0; b < 4; b++) {
            if (cbp_c != 2) {
                memset(d->res->chroma[p][b], 0, sizeof(d->res->chroma[p][b]));
                m->nnz[p + 1][b] = 0;
                continue;
            }
            int pa, pb;
            const h264d_mb_t *a = left_chroma(d, b, &pa);
            const h264d_mb_t *bmb = above_chroma(d, b, &pb);
            int inc = cbf_neighbour(d, a, pa, p + 1, false, intra)
                    + 2 * cbf_neighbour(d, bmb, pb, p + 1, false, intra);
            int16_t tmp[16];
            int nz = read_residual(d, CAT_CHROMA_AC, 15, tmp, inc, true);
            memset(d->res->chroma[p][b], 0, sizeof(d->res->chroma[p][b]));
            for (int k = 0; k < 15; k++)
                d->res->chroma[p][b][h264d_zigzag4[k + 1]] = tmp[k];
            m->nnz[p + 1][b] = (uint8_t)nz;
        }
    }
}

/* The Intra_16x16 mb_type encodes the prediction mode, the luma coded block
 * pattern and the chroma one in a single number, clause 7.4.5 Table 7-11. */
static void spacchetta_i16(int t, int *mode, int *cbp)
{
    const int k = t - 1;                 /* 0..23 */
    *mode = k & 3;                       /* four prediction modes */
    /* ⚠️ The chroma pattern cycles through THREE values, not four, so it is
     * (k / 4) modulo 3 and not the low two bits of k / 4. The two agree
     * while k is under 12 - that is, while the luma pattern is zero - and
     * part company exactly where the luma residual appears, which is what
     * made this look like a bug in the residual rather than in the type. */
    *cbp = ((k / 4) % 3) << 4;
    if (k >= 12) *cbp |= 15;             /* luma is all or nothing */
}

int h264d_decode_mb_cabac(h264_decoder_t *d)
{
    h264d_residual_at(d);
    h264d_mb_t *m = &d->mbs[d->mb_idx];
    azzera_mb(m);

    const bool bslice = d->slice.type == 1;
    const bool islice = d->slice.type == 2;

    if (!islice) {
        if (read_mb_skip(d, bslice)) {
            m->type = (uint8_t)(bslice ? H264D_MB_B_SKIP : H264D_MB_P_SKIP);
            m->qpy = (int8_t)d->qpy;
            d->last_qp_delta_nonzero = 0;

            /* ⚠️ Nothing to clear, and it used to matter: a skipped
             * macroblock never goes through the residual reader, so its
             * slot still holds whatever used it before. The chroma DCs were
             * once read out of it unconditionally and the macroblock before
             * tinted this one. Reconstruction now asks coded_block_pattern
             * first, which is the right place for the question. */
            if (bslice) {
                /* B_Skip is B_Direct_16x16 with nothing coded at all.
                 *
                 * ⚠️ Marked direct, and that matters beyond bookkeeping: the
                 * ref_idx context of 9.3.3.1.1.6 does not count a skipped or
                 * direct neighbour. Leaving the flag clear let a run of
                 * B_Skip macroblocks whose derivation happened to pick
                 * reference 1 tip the context of the next coded one, which
                 * desynchronised the slice twenty macroblocks from the end. */
                m->direct = 0xf;
                if (h264d_direct(d, m, 0xf) != 0)
                    return 1;
                /* ⚠️ Nothing to zero: a macroblock with no coefficients
                 * has its coded_block_pattern at zero, and reconstruction
                 * reads that before it reads any residual. */
                return 0;
            }
            /* Clause 8.4.1.1: one 16x16 partition on reference 0, with the
             * vector predicted as usual except that it collapses to zero at
             * the picture edge or beside a neighbour that is itself still. */
            int16_t mv[2];
            h264d_skip_mv_p(d, mv);
            place(m, 0, 0, 4, 4, 0, d->slice.ref_list[0][0], mv[0], mv[1], 0, 0);
            return 0;
        }
    }

    int t = islice ? read_mb_type_intra(d, CTX_MB_TYPE_I, true)
                   : (bslice ? read_mb_type_b(d) : read_mb_type_p(d));

    /* Resolve the slice-relative numbering into our own. */
    int i16_mode = -1;
    if (islice || (!bslice && t >= 5) || (bslice && t >= 23)) {
        const int ti = islice ? t : (bslice ? t - 23 : t - 5);
        if (ti == 25) {
            m->type = H264D_MB_I_PCM;
            m->intra = 1;
        } else if (ti == 0) {
            m->type = H264D_MB_I_NxN;
            m->intra = 1;
        } else {
            m->type = H264D_MB_I_16x16;
            m->intra = 1;
            int cbp;
            spacchetta_i16(ti, &i16_mode, &cbp);
            m->cbp = (uint8_t)cbp;
        }
    } else if (bslice) {
        static const uint8_t map_b[23] = {
            H264D_MB_B_DIRECT, H264D_MB_B_16x16, H264D_MB_B_16x16,
            H264D_MB_B_16x16,
            H264D_MB_B_16x8, H264D_MB_B_8x16, H264D_MB_B_16x8, H264D_MB_B_8x16,
            H264D_MB_B_16x8, H264D_MB_B_8x16, H264D_MB_B_16x8, H264D_MB_B_8x16,
            H264D_MB_B_16x8, H264D_MB_B_8x16, H264D_MB_B_16x8, H264D_MB_B_8x16,
            H264D_MB_B_16x8, H264D_MB_B_8x16, H264D_MB_B_16x8, H264D_MB_B_8x16,
            H264D_MB_B_16x8, H264D_MB_B_8x16, H264D_MB_B_8x8,
        };
        m->type = map_b[t < 23 ? t : 22];
    } else {
        static const uint8_t map_p[4] = {
            H264D_MB_P_16x16, H264D_MB_P_16x8, H264D_MB_P_8x16, H264D_MB_P_8x8
        };
        m->type = map_p[t < 4 ? t : 3];
    }

    if (m->type == H264D_MB_I_PCM) {
        /* Clause 7.3.5: the engine is re-initialised after the raw samples,
         * which the caller does because it owns the byte pointer. */
        m->qpy = (int8_t)d->qpy;
        memset(m->nnz, 16, sizeof(m->nnz));
        return 2;   /* not supported yet: tell the caller to drop the slice */
    }

    /* --- intra modes ------------------------------------------------- */
    if (m->intra) {
        if (m->type == H264D_MB_I_NxN) {
            if (d->pic.transform_8x8_mode)
                m->transform8x8 = (uint8_t)read_transform8x8(d);

            const int stride = m->transform8x8 ? 4 : 1;
            for (int k = 0; k < 16; k += stride) {
                const int b = zscan[k];
                /* 8.3.1.1: the predicted mode is the smaller of the two
                 * neighbours' modes, and a neighbour that is not Intra_NxN
                 * counts as DC unless constrained intra prediction says it
                 * is unavailable altogether. */
                int pa, pb;
                const h264d_mb_t *a = left4(d, b, &pa);
                const h264d_mb_t *bm = above4(d, b, &pb);
                int ma = (a && a->type == H264D_MB_I_NxN) ? a->ipred[pa]
                       : (a ? 2 : -1);
                int mb2 = (bm && bm->type == H264D_MB_I_NxN) ? bm->ipred[pb]
                        : (bm ? 2 : -1);
                int predicted = (ma < 0 || mb2 < 0) ? 2
                             : (ma < mb2 ? ma : mb2);

                int mode;
                if (decide(d, CTX_PREV_INTRA4X4_FLAG)) {
                    mode = predicted;
                } else {
                    int r = decide(d, CTX_REM_INTRA4X4);
                    r |= decide(d, CTX_REM_INTRA4X4) << 1;
                    r |= decide(d, CTX_REM_INTRA4X4) << 2;
                    mode = r < predicted ? r : r + 1;
                }
                if (stride == 1) {
                    m->ipred[b] = (int8_t)mode;
                } else {
                    /* An Intra_8x8 mode covers all four of its 4x4 slots, so
                     * that the neighbour derivation above finds it whichever
                     * one it asks about. */
                    const int bx = b & 3, by = b >> 2;
                    for (int y = 0; y < 2; y++)
                        for (int x = 0; x < 2; x++)
                            m->ipred[(by + y) * 4 + bx + x] = (int8_t)mode;
                }
            }
        }
        m->chroma_pred_mode = (int8_t)read_chroma_pred_mode(d);
    } else {
        /* --- motion ---------------------------------------------------- */
        if (read_motion(d, m, bslice, t) != 0)
            return 1;
    }

    /* --- coded block pattern and the residual ------------------------ */
    const bool i16 = (m->type == H264D_MB_I_16x16);
    if (!i16) {
        m->cbp = (uint8_t)read_cbp(d);
        /* Clause 7.3.5. The 8x8 transform is only offered when there is luma
         * residual to transform, when no sub-partition is smaller than 8x8,
         * and - for a direct macroblock - when direct_8x8_inference_flag
         * says its vectors are uniform over each 8x8. A macroblock carved
         * into 4x4 pieces cannot use an 8x8 transform, which is what
         * sub_8x8 records while the partitions are read. */
        if ((m->cbp & 15) && d->pic.transform_8x8_mode
            && m->type != H264D_MB_I_NxN && !m->sub_8x8
            && (m->type != H264D_MB_B_DIRECT || d->pic.direct_8x8_inference))
            m->transform8x8 = (uint8_t)read_transform8x8(d);
    }

    if (m->cbp || i16) {
        d->qpy += read_qp_delta(d);
        d->qpy = ((d->qpy + 52) % 52 + 52) % 52;
    } else {
        d->last_qp_delta_nonzero = 0;
    }
    m->qpy = (int8_t)d->qpy;

    read_residual_mb(d, m, i16);
    if (i16)
        m->ipred[0] = (int8_t)i16_mode;

    return 0;
}
