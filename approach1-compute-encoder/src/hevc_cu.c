/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_cu.c - the coding tree, Rec. ITU-T H.265 clauses 7.3.8.2 to 7.3.8.10.
 *
 * Where H.264 has a macroblock of a fixed sixteen samples, H.265 has a
 * coding tree unit of up to sixty-four that splits down a quadtree as far
 * as the picture needs, and then splits again - separately - for the
 * transform. So three trees are read here, nested: the coding quadtree,
 * the prediction units hanging off each of its leaves, and the transform
 * tree inside those.
 *
 * ⚠️ A syntax element that is not sent is not absent: it is inferred, and
 * the inferred value is usually not zero. split_transform_flag is one when
 * the block is bigger than the largest transform, cbf_cb is inherited from
 * the parent at the smallest chroma size, and an intra coding unit's
 * rqt_root_cbf is one without ever being coded. Reading those as zero
 * costs nothing at the time and desynchronises the slice later.
 */
#include "bitreader.h"
#include "hevc_dec_internal.h"

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------- neighbours */

/* Whether the block at these coordinates has already been decoded and
 * may be used from here. 6.4.1, still simplified to one slice.
 *
 * ⚠️ A tile boundary is a wall, and this function decides CABAC
 * contexts: the most probable intra modes and the split flag's context
 * both ask it. A neighbour wrongly called available does not make a
 * slightly wrong picture, it makes the arithmetic decoder disagree with
 * the encoder about which context to use, and everything after that is
 * noise. */
static bool available(const hevcd_t *d, int x, int y)
{
    if (x < 0 || y < 0 || x >= d->sps->width || y >= d->sps->height)
        return false;
    if (hevcd_tile_at(d, x, y) != d->tile_now) return false;
    /* ⚠️ And the same slice. A unit in a slice that has not been decoded
     * yet reads back as minus one and is refused by the same comparison,
     * which is what makes this also answer "has it happened yet". */
    return hevcd_slice_at(d, x, y) == d->slice_now;
}

/* ------------------------------------------------------------------- SAO */

/* Clause 7.3.8.3. Read, not yet applied: the offsets it carries are a
 * filter over the finished picture, and nothing is finished yet. What
 * matters here is that every bin is consumed. */
static void read_sao(hevcd_t *d, int rx, int ry)
{
    hevcd_cabac_t *c = &d->cabac;
    const hevc_slice_t *sl = d->slice;

    const int ctb_stride = d->sps->ctb_width;
    hevcd_sao_t *mine = &d->sao[ry * ctb_stride + rx];
    memset(mine, 0, sizeof *mine);

    /* Merging is how the encoder says "the same as next door" in two
     * bins instead of thirty.
     *
     * ⚠️ Next door has to be in the same tile. The flag is not even sent
     * when it is not, so reading it there consumes a bin the encoder
     * never wrote - which desynchronises everything after it. */
    const int lg = d->sps->log2_ctb;
    const bool can_left = rx > 0
        && hevcd_tile_at(d, (rx - 1) << lg, ry << lg) == d->tile_now
        && hevcd_slice_at(d, (rx - 1) << lg, ry << lg) == d->slice_now;
    const bool can_up = ry > 0
        && hevcd_tile_at(d, rx << lg, (ry - 1) << lg) == d->tile_now
        && hevcd_slice_at(d, rx << lg, (ry - 1) << lg) == d->slice_now;

    if (can_left && hevcd_bin(c, HEVCD_CTX_SAO_MERGE_FLAG)) {
        *mine = d->sao[ry * ctb_stride + rx - 1];
        return;
    }
    if (can_up && hevcd_bin(c, HEVCD_CTX_SAO_MERGE_FLAG)) {
        *mine = d->sao[(ry - 1) * ctb_stride + rx];
        return;
    }

    /* ⚠️ One type for luma and one for chroma, and the chroma one governs
     * both planes. The offsets are per plane all the same, and the edge
     * class is sent only with the plane that carried the type. */
    int chroma_type = 0;
    for (int plane = 0; plane < 3; plane++) {
        if (plane == 0 && !sl->sao_luma) continue;
        if (plane > 0 && !sl->sao_chroma) continue;

        int kind;
        if (plane == 2) {
            kind = chroma_type;
        } else {
            /* Truncated Rice, cMax 2: the first bin has a context, the
             * second does not. */
            kind = 0;
            if (hevcd_bin(c, HEVCD_CTX_SAO_TYPE_IDX))
                kind = hevcd_bypass(c) ? 2 : 1;
            if (plane == 1) chroma_type = kind;
        }
        if (kind == 0) continue;

        /* ⚠️ Truncated Rice in bypass, and cMax MOVES with the depth:
         * (1 << (Min(BitDepth, 10) - 5)) - 1, so seven at eight bits and
         * thirty-one at ten. Stopping at seven in a ten-bit stream ends
         * the codeword in the wrong place and everything after it is
         * noise - it showed up as a terminating bin that was not one. */
        const int depth = plane ? d->sps->bit_depth_chroma
                                : d->sps->bit_depth_luma;
        const int cmax = (1 << ((depth < 10 ? depth : 10) - 5)) - 1;

        int absolute[4];
        for (int i = 0; i < 4; i++) {
            int v = 0;
            while (v < cmax && hevcd_bypass(c)) v++;
            absolute[i] = v;
        }
        mine->kind[plane] = (uint8_t)kind;
        if (kind == 1) {
            for (int i = 0; i < 4; i++) {
                const int sign = (absolute[i] && hevcd_bypass(c)) ? -1 : 1;
                mine->off[plane][i] = (int8_t)(sign * absolute[i]);
            }
            mine->position[plane] = (uint8_t)hevcd_bypass_n(c, 5);
        } else {
            /* ⚠️ An edge offset carries no signs. The first two are
             * defined to be positive and the last two negative, because
             * the four cases they answer to are a valley, a step up, a
             * step down and a peak - and the filter only ever pushes a
             * sample back towards its neighbours. */
            for (int i = 0; i < 4; i++)
                mine->off[plane][i] = (int8_t)(i < 2 ? absolute[i]
                                                    : -absolute[i]);
            if (plane != 2) mine->category[plane] = (uint8_t)hevcd_bypass_n(c, 2);
            else mine->category[2] = mine->category[1];
        }
    }
}

/* ------------------------------------------------- intra prediction modes */

/* Clause 8.4.2: the three most probable modes, from the units to the left
 * and above.
 *
 * ⚠️ The unit above counts only when it is inside this coding tree unit.
 * One row up and outside it, the mode is taken as DC however it was
 * actually coded - the standard will not let a prediction depend on a row
 * of the picture that a wavefront may not have finished. */
static void probable_modes(const hevcd_t *d, int x, int y, int cand[3])
{
    const int stride = d->min_pu_width;
    int a, b;

    if (!available(d, x - 1, y))
        a = HEVCD_INTRA_DC;
    else
        a = d->intra_mode[(y >> 2) * stride + ((x - 1) >> 2)];

    const int y_above = y - 1;
    if (!available(d, x, y_above)
        || ((y_above >> d->sps->log2_ctb) != (y >> d->sps->log2_ctb)))
        b = HEVCD_INTRA_DC;
    else
        b = d->intra_mode[(y_above >> 2) * stride + (x >> 2)];

    if (a == b) {
        if (a < 2) {
            cand[0] = HEVCD_INTRA_PLANAR;
            cand[1] = HEVCD_INTRA_DC;
            cand[2] = HEVCD_INTRA_ANGULAR_26;
        } else {
            cand[0] = a;
            cand[1] = 2 + ((a + 29) % 32);
            cand[2] = 2 + ((a - 2 + 1) % 32);
        }
    } else {
        cand[0] = a;
        cand[1] = b;
        if (a != HEVCD_INTRA_PLANAR && b != HEVCD_INTRA_PLANAR)
            cand[2] = HEVCD_INTRA_PLANAR;
        else if (a != HEVCD_INTRA_DC && b != HEVCD_INTRA_DC)
            cand[2] = HEVCD_INTRA_DC;
        else
            cand[2] = HEVCD_INTRA_ANGULAR_26;
    }
}

static void write_mode(hevcd_t *d, int x, int y, int side, int mode)
{
    const int stride = d->min_pu_width;
    for (int j = 0; j < side; j += 4)
        for (int i = 0; i < side; i += 4) {
            const int px = (x + i) >> 2, py = (y + j) >> 2;
            if (px < d->min_pu_width && py < d->min_pu_height)
                d->intra_mode[py * stride + px] = (uint8_t)mode;
        }
}

/* 7.3.8.7: a PCM coding unit. The samples are sent as they are, at
 * their own bit depth, byte aligned, after the arithmetic codeword that
 * carried pcm_flag - so they start where that codeword really ends, which
 * is the same question the end of every wavefront substream asks, and
 * h264d_cabac_byte_pos() already answers it. The arithmetic decoder then
 * starts again on the first byte after them (9.3.2.5).
 *
 * ⚠️ Returns false when the samples would run past the end of the slice:
 * the caller ends the slice there rather than reading garbage. */
static bool read_pcm(hevcd_t *d, int x0, int y0, int log2_size)
{
    hevcd_cabac_t *c = &d->cabac;
    const hevc_sps_t *sps = d->sps;
    const int side = 1 << log2_size;
    const int bl = sps->pcm_bit_depth_luma, bc = sps->pcm_bit_depth_chroma;
    const int dl = sps->bit_depth_luma, dc = sps->bit_depth_chroma;

    const uint8_t *p = c->start + h264d_cabac_byte_pos(c);
    if (p > c->end) return false;
    const size_t left = (size_t)(c->end - p);
    const size_t bits = (size_t)side * side * bl
                        + 2 * (size_t)(side / 2) * (side / 2) * bc;
    const size_t bytes = (bits + 7) / 8;
    if (bytes > left) return false;

    br_t br;
    br_init(&br, p, bytes);
    for (int y = 0; y < side; y++)
        for (int x = 0; x < side; x++) {
            const int v = (int)br_read(&br, bl) << (dl - bl);
            const size_t at = (size_t)(y0 + y) * d->stride[0] + x0 + x;
            if (dl > 8) ((uint16_t *)d->plane[0])[at] = (uint16_t)v;
            else d->plane[0][at] = (uint8_t)v;
        }
    for (int k = 1; k < 3; k++)
        for (int y = 0; y < side / 2; y++)
            for (int x = 0; x < side / 2; x++) {
                const int v = (int)br_read(&br, bc) << (dc - bc);
                const size_t at = (size_t)(y0 / 2 + y) * d->stride[k]
                                  + x0 / 2 + x;
                if (dc > 8) ((uint16_t *)d->plane[k])[at] = (uint16_t)v;
                else d->plane[k][at] = (uint8_t)v;
            }

    h264d_cabac_init_engine(c, p + bytes, left - bytes);

    /* 8.4.2: a neighbour that is PCM offers DC as its candidate mode. */
    write_mode(d, x0, y0, side, HEVCD_INTRA_DC);

    /* 8.7.2 and 8.7.3: with pcm_loop_filter_disabled_flag the samples
     * stay exactly as sent, the same treatment as a lossless unit. */
    if (sps->pcm_loop_filter_disabled && d->no_filter) {
        const int n = side >> sps->log2_min_cb;
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++) {
                const int px = (x0 >> sps->log2_min_cb) + i;
                const int py = (y0 >> sps->log2_min_cb) + j;
                if (px < sps->min_cb_width && py < sps->min_cb_height)
                    d->no_filter[py * sps->min_cb_width + px] = 1;
            }
    }
    return true;
}

/* Table 8-3: the chroma mode, from an index and the luma mode beside it. */
static int chroma_mode(int idx, int luma)
{
    static const int base[4] = { HEVCD_INTRA_PLANAR, HEVCD_INTRA_ANGULAR_26,
                                 HEVCD_INTRA_ANGULAR_10, HEVCD_INTRA_DC };
    if (idx == 4) return luma;
    return (base[idx] == luma) ? 34 : base[idx];
}

/* ⚠️ Rows are decoded at the same time under wavefront parallelism, and
 * a block's bottom edge lands in a cell the row below is also writing to.
 * One byte, two writers, and a plain read-modify-write loses whichever
 * bit arrived first. */
#define MARK_EDGE(i, bit) \
    __atomic_fetch_or(&d->edges[i], (uint8_t)(bit), __ATOMIC_RELAXED)

/* 8.7.2.2: this block's left and top edges are block boundaries, and the
 * deblocking filter is allowed to cross them.
 *
 * ⚠️ Only edges that fall on the eight-sample grid count. A transform
 * block boundary four samples in is a real boundary and the filter still
 * ignores it: filtering on a four grid would leave no unfiltered sample
 * anywhere, since the filter reaches four samples each way.
 */
static void mark_edges(hevcd_t *d, int x0, int y0, int log2_size,
                        bool transform_of)
{
    if (!d->edges) return;
    const int side = 1 << log2_size;
    const int stride = d->edges_stride;
    /* Bits two and three say the same edge is also a transform block's,
     * which is a different question from whether it may be filtered: a
     * coded residual on either side of a transform edge is worth
     * smoothing, the same residual in the middle of one is not. */
    const int v = transform_of ? 1 | 4 : 1;
    const int o = transform_of ? 2 | 8 : 2;

    if ((x0 & 7) == 0)
        for (int j = 0; j < side; j += 8)
            MARK_EDGE(((y0 + j) >> 3) * stride + (x0 >> 3), v);
    if ((y0 & 7) == 0)
        for (int i = 0; i < side; i += 8)
            MARK_EDGE((y0 >> 3) * stride + ((x0 + i) >> 3), o);

    /* ⚠️ And the far side too. A block's right edge is its neighbour's
     * left one and the neighbour marks it - unless the neighbour has no
     * transform blocks of its own, which a skipped coding unit does not.
     * Marking only the near edges loses every transform boundary that
     * happens to have a skipped unit on the other side of it. */
    if (!transform_of) return;
    const int xf = x0 + side, yf = y0 + side;
    if ((xf & 7) == 0 && xf < d->sps->width)
        for (int j = 0; j < side; j += 8)
            MARK_EDGE(((y0 + j) >> 3) * stride + (xf >> 3), v);
    if ((yf & 7) == 0 && yf < d->sps->height)
        for (int i = 0; i < side; i += 8)
            MARK_EDGE((yf >> 3) * stride + ((x0 + i) >> 3), o);
}

/* The same, for a rectangle: a prediction unit's own edges. 8.7.2.2
 * filters transform block edges and prediction block edges alike, and an
 * asymmetric partition puts one where no transform ever will.
 *
 * ⚠️ Not a transform boundary, so a coded residual either side of it does
 * not raise the strength. Only the motion does. */
static void mark_edges_rect(hevcd_t *d, int x0, int y0, int w, int h)
{
    if (!d->edges) return;
    const int stride = d->edges_stride;
    if ((x0 & 7) == 0)
        for (int j = 0; j < h; j += 8)
            MARK_EDGE(((y0 + j) >> 3) * stride + (x0 >> 3), 1);
    if ((y0 & 7) == 0)
        for (int i = 0; i < w; i += 8)
            MARK_EDGE((y0 >> 3) * stride + ((x0 + i) >> 3), 2);
}

/* Which smallest transform blocks carry a luma residual. */
static void mark_cbf(hevcd_t *d, int x0, int y0, int log2_size)
{
    if (!d->cbf_map) return;
    const int side = 1 << log2_size;
    for (int j = 0; j < side; j += 4)
        for (int i = 0; i < side; i += 4) {
            const int px = (x0 + i) >> 2, py = (y0 + j) >> 2;
            if (px < d->min_pu_width && py < d->min_pu_height)
                d->cbf_map[py * d->min_pu_width + px] = 1;
        }
}


/* ------------------------------------------------ inter prediction units */

/* Where the prediction units of each partition mode sit, Table 7-10 and
 * figure 7-2. Written as a table of quarters so the asymmetric modes are
 * not four more special cases. */
int hevcd_pu_count(int part_mode)
{
    if (part_mode == HEVCD_PART_2Nx2N) return 1;
    if (part_mode == HEVCD_PART_NxN) return 4;
    return 2;
}

void hevcd_pu_rect(int part_mode, int k, int side,
                         int *x, int *y, int *w, int *h)
{
    const int middle = side / 2, quarto = side / 4;

    switch (part_mode) {
    case HEVCD_PART_2NxN:
        *x = 0; *y = k * middle; *w = side; *h = middle; return;
    case HEVCD_PART_Nx2N:
        *x = k * middle; *y = 0; *w = middle; *h = side; return;
    case HEVCD_PART_NxN:
        *x = (k & 1) * middle; *y = (k >> 1) * middle;
        *w = middle; *h = middle; return;
    case HEVCD_PART_2NxnU:
        *x = 0; *y = k ? quarto : 0; *w = side;
        *h = k ? side - quarto : quarto; return;
    case HEVCD_PART_2NxnD:
        *x = 0; *y = k ? side - quarto : 0; *w = side;
        *h = k ? quarto : side - quarto; return;
    case HEVCD_PART_nLx2N:
        *x = k ? quarto : 0; *y = 0;
        *w = k ? side - quarto : quarto; *h = side; return;
    case HEVCD_PART_nRx2N:
        *x = k ? side - quarto : 0; *y = 0;
        *w = k ? quarto : side - quarto; *h = side; return;
    default:
        *x = 0; *y = 0; *w = side; *h = side; return;
    }
}

/* What this prediction unit's motion was, written over every smallest
 * block it covers. The neighbours that come after read it there. */
static void write_field(hevcd_t *d, int x0, int y0, int w, int h,
                         const hevcd_mvf_t *m)
{
    if (!d->mvf) return;
    for (int j = 0; j < h; j += 4)
        for (int i = 0; i < w; i += 4) {
            const int px = (x0 + i) >> 2, py = (y0 + j) >> 2;
            if (px < d->min_pu_width && py < d->min_pu_height)
                d->mvf[py * d->min_pu_width + px] = *m;
        }
}

/* An intra coding unit has no motion, and saying so is not the same as
 * leaving whatever the last picture put there: the neighbours ask. */
static void intra_field(hevcd_t *d, int x0, int y0, int side)
{
    hevcd_mvf_t empty;
    memset(&empty, 0, sizeof empty);
    empty.ref_idx[0] = empty.ref_idx[1] = -1;
    write_field(d, x0, y0, side, side, &empty);
}

/* 9.3.3.3: exp-Golomb of order k, entirely in bypass. */
static unsigned golomb(hevcd_cabac_t *c, int k)
{
    int q = 0;
    while (q < 24 && hevcd_bypass(c)) q++;
    unsigned v = q ? (((1u << q) - 1) << k) : 0;
    const int count = q + k;
    if (count) v += hevcd_bypass_n(c, count);
    return v;
}

/* 9.3.3.7. Four different binarisations wearing one name: which one
 * applies depends on the block size and on whether the sequence allows
 * asymmetric partitions.
 *
 * ⚠️ The context of the third bin is not the same in the two cases. At
 * the smallest coding block it is index two and it is deciding between
 * Nx2N and NxN; above it, index three, deciding between a symmetric
 * partition and an asymmetric one. */
static int read_part_mode(hevcd_t *d, int log2_size)
{
    hevcd_cabac_t *c = &d->cabac;
    const hevc_sps_t *sps = d->sps;

    if (hevcd_bin(c, HEVCD_CTX_PART_MODE + 0)) return HEVCD_PART_2Nx2N;
    const bool horizontal = hevcd_bin(c, HEVCD_CTX_PART_MODE + 1) != 0;

    if (log2_size > sps->log2_min_cb) {
        if (!sps->amp_enabled
            || hevcd_bin(c, HEVCD_CTX_PART_MODE + 3))
            return horizontal ? HEVCD_PART_2NxN : HEVCD_PART_Nx2N;
        const bool second_one = hevcd_bypass(c) != 0;
        if (horizontal)
            return second_one ? HEVCD_PART_2NxnD : HEVCD_PART_2NxnU;
        return second_one ? HEVCD_PART_nRx2N : HEVCD_PART_nLx2N;
    }

    if (horizontal) return HEVCD_PART_2NxN;
    /* ⚠️ An inter 4x4 does not exist, so at an eight-sample coding block
     * the vertical branch has nowhere left to go and the bin is not
     * sent. */
    if (log2_size == 3) return HEVCD_PART_Nx2N;
    return hevcd_bin(c, HEVCD_CTX_PART_MODE + 2) ? HEVCD_PART_Nx2N
                                                 : HEVCD_PART_NxN;
}

/* Truncated Rice with the first two bins in context and the rest in
 * bypass, for both lists. */
static int read_ref_idx(hevcd_t *d, int how_many)
{
    hevcd_cabac_t *c = &d->cabac;
    const int maximum = how_many - 1;
    const int with_context = maximum < 2 ? maximum : 2;

    int i = 0;
    while (i < with_context && hevcd_bin(c, HEVCD_CTX_REF_IDX_L0 + i)) i++;
    if (i == 2)
        while (i < maximum && hevcd_bypass(c)) i++;
    return i;
}

static int read_merge_idx(hevcd_t *d, int count)
{
    hevcd_cabac_t *c = &d->cabac;
    if (count <= 1) return 0;
    if (!hevcd_bin(c, HEVCD_CTX_MERGE_IDX)) return 0;
    int i = 1;
    while (i < count - 1 && hevcd_bypass(c)) i++;
    return i;
}

/* 7.3.8.9. The two components are interleaved rather than sent one after
 * the other, which lets the two greater-than-zero flags share a context
 * without either of them waiting for the other's remainder. */
static void read_mvd(hevcd_t *d, int16_t mvd[2])
{
    hevcd_cabac_t *c = &d->cabac;
    bool above_zero[2], above_one[2] = { false, false };

    for (int i = 0; i < 2; i++)
        above_zero[i] = hevcd_bin(c, HEVCD_CTX_ABS_MVD_GREATER0_FLAG) != 0;
    for (int i = 0; i < 2; i++)
        if (above_zero[i])
            above_one[i] = hevcd_bin(c, HEVCD_CTX_ABS_MVD_GREATER1_FLAG + 1) != 0;

    for (int i = 0; i < 2; i++) {
        mvd[i] = 0;
        if (!above_zero[i]) continue;
        int v = 1;
        if (above_one[i]) v = 2 + (int)golomb(c, 1);
        mvd[i] = (int16_t)(hevcd_bypass(c) ? -v : v);
    }
}

/* 7.3.8.6. Everything a prediction unit says about where it copies from:
 * which lists, which pictures in them, and how far off the prediction
 * the true motion was. */
static bool read_pu(hevcd_t *d, int x0, int y0, int w, int h,
                     int part_idx, bool skip)
{
    hevcd_cabac_t *c = &d->cabac;
    const hevc_slice_t *sl = d->slice;
    const int n_merge = 5 - sl->five_minus_max_num_merge_cand;
    hevcd_mvf_t m;

    memset(&m, 0, sizeof m);
    m.ref_idx[0] = m.ref_idx[1] = -1;

    if (skip || hevcd_bin(c, HEVCD_CTX_MERGE_FLAG)) {
        const int idx = read_merge_idx(d, n_merge);
        hevcd_merge(d, x0, y0, w, h, part_idx, idx, &m);
        /* ⚠️ A merged unit copies its neighbour's index, and the
         * neighbour's index was resolved against the same lists, so it
         * still means the same picture. Its recorded picture may not be:
         * a temporal candidate has no index of its own. */
        for (int l = 0; l < 2; l++)
            if ((m.pred_flag & (1 << l)) && m.ref_idx[l] >= 0
                && m.ref_idx[l] < d->n_refs[l])
                m.ref_poc[l] = d->ref_pic[l][m.ref_idx[l]]->poc;
        write_field(d, x0, y0, w, h, &m);
        hevcd_predict_inter(d, x0, y0, w, h, &m);
        return true;
    }

    /* Which lists this unit predicts from. In a P slice there is only
     * one and nothing is sent. */
    int lists = 1;                          /* bit 0 list 0, bit 1 list 1 */
    if (sl->type == 0) {
        /* ⚠️ An 8x4 or a 4x8 may not be bi-predicted: two of those in a
         * row would fetch more samples than the level allows, so the
         * first bin is not even sent for them. */
        if (w + h != 12
            && hevcd_bin(c, HEVCD_CTX_INTER_PRED_IDC + d->cu.depth))
            lists = 3;
        else
            lists = hevcd_bin(c, HEVCD_CTX_INTER_PRED_IDC + 4) ? 2 : 1;
    }

    m.pred_flag = (uint8_t)lists;
    for (int l = 0; l < 2; l++) {
        if (!(lists & (1 << l))) continue;
        m.ref_idx[l] = (int8_t)(sl->num_ref_idx[l] > 1
                                ? read_ref_idx(d, sl->num_ref_idx[l]) : 0);
        if (m.ref_idx[l] < d->n_refs[l])
            m.ref_poc[l] = d->ref_pic[l][m.ref_idx[l]]->poc;
        int16_t mvd[2] = { 0, 0 };
        if (l == 1 && sl->mvd_l1_zero && lists == 3) {
            /* Sent as nothing at all: the slice header promised it. */
        } else {
            read_mvd(d, mvd);
        }
        const int mvp = hevcd_bin(c, HEVCD_CTX_MVP_LX_FLAG);
        hevcd_amvp(d, x0, y0, w, h, l, mvp, &m);
        /* ⚠️ The difference is added after the predictor is derived and
         * wraps at sixteen bits: the standard says the sum is taken
         * modulo 2^16, so a vector near the limit comes back round rather
         * than being clipped. */
        m.mv[l][0] = (int16_t)(m.mv[l][0] + mvd[0]);
        m.mv[l][1] = (int16_t)(m.mv[l][1] + mvd[1]);
    }
    write_field(d, x0, y0, w, h, &m);
    hevcd_predict_inter(d, x0, y0, w, h, &m);
    return false;
}

/* cu_skip_flag's context asks how many of the two neighbours were skipped
 * themselves, which is the cheapest possible guess at whether this part
 * of the picture is standing still. */
static int skip_context(const hevcd_t *d, int x0, int y0)
{
    const hevc_sps_t *sps = d->sps;
    const int stride = sps->min_cb_width;
    const int l = sps->log2_min_cb;
    int n = 0;

    /* ⚠️ available(), not "inside the picture". The two agree until a
     * tile boundary runs between here and the neighbour. */
    if (available(d, x0 - 1, y0)
        && d->skip[(y0 >> l) * stride + ((x0 - 1) >> l)]) n++;
    if (available(d, x0, y0 - 1)
        && d->skip[((y0 - 1) >> l) * stride + (x0 >> l)]) n++;
    return n;
}

static void mark_skip(hevcd_t *d, int x0, int y0, int log2_size, bool skip)
{
    const hevc_sps_t *sps = d->sps;
    const int stride = sps->min_cb_width;
    const int n = 1 << (log2_size - sps->log2_min_cb);

    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) {
            const int px = (x0 >> sps->log2_min_cb) + i;
            const int py = (y0 >> sps->log2_min_cb) + j;
            if (px < sps->min_cb_width && py < sps->min_cb_height)
                d->skip[py * stride + px] = skip ? 1 : 0;
        }
}

/* ------------------------------------------------------- transform units */

static void read_qp_delta(hevcd_t *d)
{
    hevcd_cabac_t *c = &d->cabac;
    if (!d->pps->cu_qp_delta_enabled || d->cu_qp_delta_coded) return;
    d->cu_qp_delta_coded = true;

    /* 9.3.3.10: a truncated Rice prefix of at most five, then an
     * exp-Golomb suffix in bypass. */
    int prefix = 0;
    while (prefix < 5
           && hevcd_bin(c, HEVCD_CTX_CU_QP_DELTA + (prefix ? 1 : 0)))
        prefix++;

    int value = prefix;
    if (prefix == 5) {
        int k = 0;
        while (k < 32 && hevcd_bypass(c)) k++;
        value = 5 + (int)((1u << k) - 1) + (int)(k ? hevcd_bypass_n(c, k) : 0);
    }
    if (value && hevcd_bypass(c))
        value = -value;
    d->cu_qp_delta = value;
    /* ⚠️ The wrap is over 52 + QpBdOffsetY, not 52, and it lands on
     * -QpBdOffsetY rather than zero. Same shape at eight bits, where the
     * offset is nothing, and different at ten. */
    const int qp_bd_offset = 6 * (d->sps->bit_depth_luma - 8);
    d->qp_y = ((d->qp_y_pred + value + 52 + 2 * qp_bd_offset)
               % (52 + qp_bd_offset)) - qp_bd_offset;
}

/* 8.6.1: the luma quantisation parameter's prediction, worked out once
 * per quantisation group.
 *
 * The prediction is the average of the group to the left and the group
 * above, and neither of them counts unless it sits inside the same coding
 * tree block. A neighbour one block over has certainly been decoded, and
 * the standard refuses it anyway, so that a decoder never has to keep a
 * whole row of parameters alive to decode the next one. What is left when
 * both are refused is qPY_PREV: the parameter of the last coding unit
 * decoded before this group.
 *
 * ⚠️ This runs for every group and not only for the ones that carry a
 * delta. A group with no delta still takes the prediction as its
 * parameter, so a decoder that simply lets QpY stand is right exactly as
 * long as the previous group happened to be the left neighbour - which at
 * the start of a row it is not.
 */
static void start_qg(hevcd_t *d, int x0, int y0)
{
    const int log2_ctb = d->sps->log2_ctb;
    const int log2_cb = d->sps->log2_min_cb;
    const int stride = d->sps->min_cb_width;

    d->qg_x = x0;
    d->qg_y = y0;

    /* ⚠️ Not cleared here. This runs once per quadtree node down to the
     * group's own size, and only the last of those calls is the group's;
     * clearing on the first one hands the group the previous group's
     * parameter at the exact point where the standard says not to. It is
     * cleared once a coding unit has been read. */
    const int before = d->qg_restarts ? d->slice->qp : d->qp_y_prev;

    int a = before, b = before;
    if (x0 > 0 && ((x0 - 1) >> log2_ctb) == (x0 >> log2_ctb))
        a = d->qp_y_map[(y0 >> log2_cb) * stride + ((x0 - 1) >> log2_cb)];
    if (y0 > 0 && ((y0 - 1) >> log2_ctb) == (y0 >> log2_ctb))
        b = d->qp_y_map[((y0 - 1) >> log2_cb) * stride + (x0 >> log2_cb)];

    d->qp_y_pred = (a + b + 1) >> 1;
    d->qp_y = d->qp_y_pred;
}

/* Table 8-10: the chroma quantisation parameter, which is not the luma one
 * even before the offsets. Above 29 it stops following, because chroma
 * tolerates coarser quantisation than luma does and the standard says so
 * in a table rather than a formula. */
static int qp_chroma(int qp_i)
{
    /* ⚠️ No clamp at zero. Table 8-10 says qPCb = qPiCb for anything
     * under thirty, and above eight bits qPiCb is allowed to be negative
     * - down to -QpBdOffsetC. The caller has already clipped it there;
     * flooring it again here would quietly raise the quantiser. */
    if (qp_i < 30) return qp_i;
    if (qp_i > 43) return qp_i - 6;
    return hevcd_qp_c[qp_i - 30];
}

/* 8.6.1, for the coding unit being reconstructed. */
/* ⚠️ What the dequantiser wants is QP', not QP: QP'Y = QpY + QpBdOffsetY
 * and QpBdOffsetY = 6 * (BitDepth - 8), which is nothing at eight bits
 * and twelve at ten. Leave it out and every coefficient comes back a
 * factor of four thousand too small, which is a grey picture rather than
 * a wrong one - no crash, no complaint. The chroma side clips in QP
 * space first and adds the offset after, as 8.6.1 spells out. */
static int block_qp(const hevcd_t *d, int c_idx)
{
    const int bd = c_idx ? d->sps->bit_depth_chroma : d->sps->bit_depth_luma;
    const int qp_bd_offset = 6 * (bd - 8);

    if (c_idx == 0) return d->qp_y + qp_bd_offset;
    const int off = (c_idx == 1)
        ? d->pps->cb_qp_offset + d->slice->cb_qp_offset
        : d->pps->cr_qp_offset + d->slice->cr_qp_offset;
    int q = d->qp_y + off;
    if (q < -qp_bd_offset) q = -qp_bd_offset;
    if (q > 57) q = 57;
    return qp_chroma(q) + qp_bd_offset;
}

/* One transform block: predict it, then add whatever residual it has.
 *
 * ⚠️ In that order and one block at a time. The next block predicts from
 * this one's reconstructed samples, so a version that read all the
 * residuals first and reconstructed afterwards would predict from
 * whatever was in the picture before. */
static void reconstruct_tb(hevcd_t *d, int c_idx, int x, int y,
                            int log2_size, bool has_residual)
{
    /* The luma mode is per prediction block; chroma has one per coding
     * unit. For a chroma block the coordinates are halved, so the mode is
     * looked up at the luma position it covers. */
    const int lx = c_idx ? x * 2 : x, ly = c_idx ? y * 2 : y;
    const int mode = (c_idx == 0)
        ? d->intra_mode[(ly >> 2) * d->min_pu_width + (lx >> 2)]
        : d->cu.intra_mode_c;

    /* ⚠️ Only an intra block predicts from the samples beside it. An
     * inter one predicts from another picture, which is not written yet:
     * until it is, the residual lands on whatever is there. */
    if (d->cu.pred_mode == HEVCD_MODE_INTRA)
        hevcd_predict_intra(d, c_idx, x, y, log2_size, mode);
    if (!has_residual) return;

    /* 8.6.2: with the bypass the coefficients are the residual already.
     * Everything below this point - the scaling, the two transform stages,
     * the rounding - exists to undo a quantisation that never happened. */
    /* ⚠️ Luma and chroma carry their own depth. They are equal in every
     * profile we accept, and reading the wrong one would be invisible
     * until the day they are not. */
    const int bd = c_idx ? d->sps->bit_depth_chroma : d->sps->bit_depth_luma;

    if (d->cu.transquant_bypass) {
        hevcd_add(d->plane[c_idx], d->stride[c_idx], x, y,
                  d->coeff, log2_size, bd);
        return;
    }

    /* 8.6.4.2: the matrix for the block's size, prediction mode and
     * component. ⚠️ Not for a transform-skipped block above 4x4, as the
     * later editions of the standard and every reference decoder have it.
     * The 32x32 lists exist for luma only. */
    const uint8_t *m = NULL;
    if (d->scaling_on && !(d->transform_skip && log2_size > 2)) {
        const int mat = (d->cu.pred_mode == HEVCD_MODE_INTRA ? 0 : 3)
                        + (log2_size == 5 ? 0 : c_idx);
        m = log2_size == 2 ? d->sf4[mat]
          : log2_size == 3 ? d->sf8[mat]
          : log2_size == 4 ? d->sf16[mat] : d->sf32[mat];
    }
    if (m)
        hevcd_dequantize_scaled(d->coeff, log2_size, block_qp(d, c_idx),
                                bd, m);
    else
        hevcd_dequantize(d->coeff, log2_size, block_qp(d, c_idx), bd);
    if (d->transform_skip)
        hevcd_skip_transform(d->coeff, log2_size, bd);
    else
        hevcd_transform(d->coeff, log2_size,
                        c_idx == 0 && log2_size == 2
                        && d->cu.pred_mode == HEVCD_MODE_INTRA, bd);
    hevcd_add(d->plane[c_idx], d->stride[c_idx], x, y,
              d->coeff, log2_size, bd);
}

static void read_tu(hevcd_t *d, int x0, int y0, int x_base, int y_base,
                     int log2_size, int depth, int blk,
                     bool cbf_luma, bool cbf_cb, bool cbf_cr)
{
    if (cbf_luma || cbf_cb || cbf_cr)
        read_qp_delta(d);

    if (cbf_luma)
        hevcd_read_residual(d, x0, y0, log2_size, 0);
    reconstruct_tb(d, 0, x0, y0, log2_size, cbf_luma);

    if (log2_size > 2) {
        const int cx = x0 >> 1, cy = y0 >> 1;
        if (cbf_cb) hevcd_read_residual(d, x0, y0, log2_size - 1, 1);
        reconstruct_tb(d, 1, cx, cy, log2_size - 1, cbf_cb);
        if (cbf_cr) hevcd_read_residual(d, x0, y0, log2_size - 1, 2);
        reconstruct_tb(d, 2, cx, cy, log2_size - 1, cbf_cr);
    } else if (blk == 3) {
        /* ⚠️ At the smallest luma size the four 4x4 blocks share one 4x4
         * chroma block, which is read with the last of them and lives at
         * the parent's corner. */
        const int cx = x_base >> 1, cy = y_base >> 1;
        if (cbf_cb) hevcd_read_residual(d, x_base, y_base, 2, 1);
        reconstruct_tb(d, 1, cx, cy, 2, cbf_cb);
        if (cbf_cr) hevcd_read_residual(d, x_base, y_base, 2, 2);
        reconstruct_tb(d, 2, cx, cy, 2, cbf_cr);
    }
    (void)depth;
}

static void read_transform_tree(hevcd_t *d, int x0, int y0,
                                     int x_base, int y_base, int log2_size,
                                     int depth, int blk,
                                     bool cbf_cb_padre, bool cbf_cr_padre)
{
    hevcd_cabac_t *c = &d->cabac;
    const hevc_sps_t *sps = d->sps;

    const int max_depth = d->cu.pred_mode == HEVCD_MODE_INTRA
        ? sps->max_transform_hierarchy_depth_intra + (d->cu.intra_split ? 1 : 0)
        : sps->max_transform_hierarchy_depth_inter;

    bool split;
    if (log2_size <= sps->log2_max_tb && log2_size > sps->log2_min_tb
        && depth < max_depth && !(d->cu.intra_split && depth == 0)) {
        split = hevcd_bin(c, HEVCD_CTX_SPLIT_TRANSFORM_FLAG + 5 - log2_size) != 0;
    } else {
        /* Inferred, and rarely zero: too big for the largest transform, an
         * intra unit split into four prediction blocks, or an inter unit
         * whose partitions the transform is not allowed to straddle.
         *
         * ⚠️ That last one only bites when the sequence allows no
         * transform depth at all for inter: with a depth to spare the
         * flag is read instead, and the condition above already says so. */
        const bool inter_split = sps->max_transform_hierarchy_depth_inter == 0
            && d->cu.pred_mode == HEVCD_MODE_INTER
            && d->cu.part_mode != HEVCD_PART_2Nx2N && depth == 0;
        split = (log2_size > sps->log2_max_tb)
              || (d->cu.intra_split && depth == 0) || inter_split;
    }

    bool cbf_cb = false, cbf_cr = false;
    if (log2_size > 2) {
        if (depth == 0 || cbf_cb_padre)
            cbf_cb = hevcd_bin(c, HEVCD_CTX_CBF_CB_CR + depth) != 0;
        if (depth == 0 || cbf_cr_padre)
            cbf_cr = hevcd_bin(c, HEVCD_CTX_CBF_CB_CR + depth) != 0;
    } else {
        /* At 4x4 luma there is no chroma of its own: it belongs to the
         * parent and is inherited unread. */
        cbf_cb = cbf_cb_padre;
        cbf_cr = cbf_cr_padre;
    }

    if (split) {
        const int middle = 1 << (log2_size - 1);
        read_transform_tree(d, x0, y0, x0, y0, log2_size - 1,
                                 depth + 1, 0, cbf_cb, cbf_cr);
        read_transform_tree(d, x0 + middle, y0, x0, y0, log2_size - 1,
                                 depth + 1, 1, cbf_cb, cbf_cr);
        read_transform_tree(d, x0, y0 + middle, x0, y0, log2_size - 1,
                                 depth + 1, 2, cbf_cb, cbf_cr);
        read_transform_tree(d, x0 + middle, y0 + middle, x0, y0,
                                 log2_size - 1, depth + 1, 3, cbf_cb, cbf_cr);
        return;
    }

    bool cbf_luma = true;
    if (d->cu.pred_mode == HEVCD_MODE_INTRA || depth != 0 || cbf_cb || cbf_cr)
        cbf_luma = hevcd_bin(c, HEVCD_CTX_CBF_LUMA + (depth == 0 ? 1 : 0)) != 0;

    mark_edges(d, x0, y0, log2_size, true);
    if (cbf_luma) mark_cbf(d, x0, y0, log2_size);
    read_tu(d, x0, y0, x_base, y_base, log2_size, depth, blk,
             cbf_luma, cbf_cb, cbf_cr);
}

/* ---------------------------------------------------------- coding units */

static void read_cu(hevcd_t *d, int x0, int y0, int log2_size)
{
    hevcd_cabac_t *c = &d->cabac;
    const hevc_sps_t *sps = d->sps;
    const int side = 1 << log2_size;

    d->cu.x = x0;
    d->cu.y = y0;
    d->cu.log2_size = log2_size;
    d->cu.pred_mode = HEVCD_MODE_INTRA;
    d->cu.part_mode = HEVCD_PART_2Nx2N;
    d->cu.transquant_bypass = false;
    d->cu.intra_split = false;

    if (d->pps->transquant_bypass_enabled)
        d->cu.transquant_bypass = hevcd_bin(c, HEVCD_CTX_CU_TRANSQUANT_BYPASS_FLAG) != 0;

    /* An I slice has neither cu_skip_flag nor pred_mode_flag: everything
     * in it is intra by definition. */
    d->cu.skip = false;
    if (d->slice->type != 2) {
        d->cu.skip = hevcd_bin(c, HEVCD_CTX_SKIP_FLAG
                               + skip_context(d, x0, y0)) != 0;
        mark_skip(d, x0, y0, log2_size, d->cu.skip);
        if (d->cu.skip) {
            /* A skipped unit is one merge index and nothing else: no
             * partition, no residual, not even a prediction mode. */
            d->cu.pred_mode = HEVCD_MODE_INTER;
            read_pu(d, x0, y0, side, side, 0, true);
            return;
        }
        if (!hevcd_bin(c, HEVCD_CTX_PRED_MODE_FLAG))
            d->cu.pred_mode = HEVCD_MODE_INTER;
    } else {
        mark_skip(d, x0, y0, log2_size, false);
    }
    intra_field(d, x0, y0, side);

    if (d->cu.pred_mode == HEVCD_MODE_INTER) {
        d->cu.part_mode = read_part_mode(d, log2_size);

        bool merged = false;
        const int how_many = hevcd_pu_count(d->cu.part_mode);
        for (int k = 0; k < how_many; k++) {
            int px, py, pw, ph;
            hevcd_pu_rect(d->cu.part_mode, k, side, &px, &py, &pw, &ph);
            mark_edges_rect(d, x0 + px, y0 + py, pw, ph);
            merged = read_pu(d, x0 + px, y0 + py, pw, ph, k, false);
        }

        /* 7.3.8.5: a whole-block merge says nothing about its residual,
         * because a skip would have been sent instead if there were
         * none. Every other inter unit says whether it has one. */
        bool has_residual = true;
        if (!(d->cu.part_mode == HEVCD_PART_2Nx2N && merged))
            has_residual = hevcd_bin(c, HEVCD_CTX_NO_RESIDUAL_DATA_FLAG) != 0;
        if (has_residual)
            read_transform_tree(d, x0, y0, x0, y0, log2_size, 0, 0,
                                     false, false);
        return;
    }

    if (log2_size == sps->log2_min_cb) {
        if (!hevcd_bin(c, HEVCD_CTX_PART_MODE)) {
            d->cu.part_mode = HEVCD_PART_NxN;
            d->cu.intra_split = true;
        }
    }

    if (sps->pcm_enabled && d->cu.part_mode == HEVCD_PART_2Nx2N
        && log2_size >= sps->log2_min_pcm_cb
        && log2_size <= sps->log2_max_pcm_cb) {
        /* pcm_flag, coded as a terminating bin. */
        if (hevcd_terminate(c)) {
            if (!read_pcm(d, x0, y0, log2_size)) d->slice_end = true;
            return;
        }
    }

    const int pb = (d->cu.part_mode == HEVCD_PART_NxN) ? side / 2 : side;
    const int count = (d->cu.part_mode == HEVCD_PART_NxN) ? 2 : 1;

    bool from_probable[2][2];
    for (int j = 0; j < count; j++)
        for (int i = 0; i < count; i++)
            from_probable[j][i] =
                hevcd_bin(c, HEVCD_CTX_PREV_INTRA_LUMA_PRED_FLAG) != 0;

    int luma_mode_0 = HEVCD_INTRA_DC;
    for (int j = 0; j < count; j++) {
        for (int i = 0; i < count; i++) {
            const int x = x0 + i * pb, y = y0 + j * pb;
            int cand[3];
            probable_modes(d, x, y, cand);

            int mode;
            if (from_probable[j][i]) {
                /* mpm_idx: truncated Rice, cMax 2, all bypass. */
                int idx = 0;
                if (hevcd_bypass(c)) idx = hevcd_bypass(c) ? 2 : 1;
                mode = cand[idx];
            } else {
                /* The three candidates are taken out of the range, so what
                 * is sent is five bits into what is left. */
                if (cand[0] > cand[1]) { const int t = cand[0]; cand[0] = cand[1]; cand[1] = t; }
                if (cand[0] > cand[2]) { const int t = cand[0]; cand[0] = cand[2]; cand[2] = t; }
                if (cand[1] > cand[2]) { const int t = cand[1]; cand[1] = cand[2]; cand[2] = t; }
                mode = (int)hevcd_bypass_n(c, 5);
                for (int k = 0; k < 3; k++)
                    if (mode >= cand[k]) mode++;
            }
            write_mode(d, x, y, pb, mode);
            if (i == 0 && j == 0) luma_mode_0 = mode;
        }
    }

    /* intra_chroma_pred_mode: one context-coded bin, then two in bypass
     * when it is set. At 4:2:0 there is one for the whole coding unit. */
    int idx_c = 4;
    if (hevcd_bin(c, HEVCD_CTX_INTRA_CHROMA_PRED_MODE))
        idx_c = (int)hevcd_bypass_n(c, 2);
    d->cu.intra_mode_c = chroma_mode(idx_c, luma_mode_0);

    intra_field(d, x0, y0, side);

    /* An intra coding unit's rqt_root_cbf is not sent: it is one. */
    read_transform_tree(d, x0, y0, x0, y0, log2_size, 0, 0, false, false);
}

/* ------------------------------------------------------ the coding quadtree */

static void read_quadtree(hevcd_t *d, int x0, int y0, int log2_size, int depth)
{
    hevcd_cabac_t *c = &d->cabac;
    const hevc_sps_t *sps = d->sps;
    const int side = 1 << log2_size;

    bool split;
    if (x0 + side <= sps->width && y0 + side <= sps->height
        && log2_size > sps->log2_min_cb) {
        /* 9.3.4.2.2: how deep the neighbours were split. */
        int ctx = 0;
        const int stride = sps->min_cb_width;
        if (available(d, x0 - 1, y0)
            && d->ct_depth[(y0 >> sps->log2_min_cb) * stride
                           + ((x0 - 1) >> sps->log2_min_cb)] > depth)
            ctx++;
        if (available(d, x0, y0 - 1)
            && d->ct_depth[((y0 - 1) >> sps->log2_min_cb) * stride
                           + (x0 >> sps->log2_min_cb)] > depth)
            ctx++;
        split = hevcd_bin(c, HEVCD_CTX_SPLIT_CODING_UNIT_FLAG + ctx) != 0;
    } else {
        /* A block that runs off the picture has to be split until it does
         * not, and one already at the smallest size cannot be. */
        split = log2_size > sps->log2_min_cb;
    }

    if (d->pps->cu_qp_delta_enabled
        && log2_size >= sps->log2_ctb - d->pps->diff_cu_qp_delta_depth) {
        d->cu_qp_delta_coded = false;
        d->cu_qp_delta = 0;
        start_qg(d, x0, y0);
    }

    if (split) {
        const int middle = side >> 1;
        read_quadtree(d, x0, y0, log2_size - 1, depth + 1);
        if (x0 + middle < sps->width)
            read_quadtree(d, x0 + middle, y0, log2_size - 1, depth + 1);
        if (y0 + middle < sps->height)
            read_quadtree(d, x0, y0 + middle, log2_size - 1, depth + 1);
        if (x0 + middle < sps->width && y0 + middle < sps->height)
            read_quadtree(d, x0 + middle, y0 + middle, log2_size - 1, depth + 1);
        return;
    }

    /* Remember how deep this went, for the neighbours that come after. */
    const int stride = sps->min_cb_width;
    const int n = side >> sps->log2_min_cb;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) {
            const int px = (x0 >> sps->log2_min_cb) + i;
            const int py = (y0 >> sps->log2_min_cb) + j;
            if (px < sps->min_cb_width && py < sps->min_cb_height)
                d->ct_depth[py * stride + px] = (uint8_t)depth;
        }

    mark_edges(d, x0, y0, log2_size, false);
    d->cu.depth = depth;
    read_cu(d, x0, y0, log2_size);

    if (d->no_filter && d->cu.transquant_bypass)
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++) {
                const int px = (x0 >> sps->log2_min_cb) + i;
                const int py = (y0 >> sps->log2_min_cb) + j;
                if (px < sps->min_cb_width && py < sps->min_cb_height)
                    d->no_filter[py * stride + px] = 1;
            }

    /* And what parameter it ended up with, for the groups that come after
     * and, later, for the deblocking filter. */
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) {
            const int px = (x0 >> sps->log2_min_cb) + i;
            const int py = (y0 >> sps->log2_min_cb) + j;
            if (px < sps->min_cb_width && py < sps->min_cb_height)
                d->qp_y_map[py * stride + px] = (int8_t)d->qp_y;
        }
    d->qp_y_prev = d->qp_y;
    d->qg_restarts = false;
}


/* 6.5.2: the z-scan address of every smallest transform block, so that
 * "has this neighbour been decoded yet" can be answered by comparing two
 * numbers. With one tile and one slice the coding tree units are in raster
 * order, and inside each of them the address is the interleaving of the
 * bits of x and y - which is what a quadtree walk is. */
int hevcd_prepare_zscan(hevcd_t *d)
{
    const hevc_sps_t *sps = d->sps;
    const int w = sps->width >> sps->log2_min_tb;
    const int h = sps->height >> sps->log2_min_tb;
    const size_t serve = (size_t)w * h;

    if (!d->min_tb_addr_zs || d->n_zs < serve) {
        free(d->min_tb_addr_zs);
        d->min_tb_addr_zs = calloc(serve, sizeof(int32_t));
        d->n_zs = serve;
        if (!d->min_tb_addr_zs) return -1;
    }

    const int diff = sps->log2_ctb - sps->log2_min_tb;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            const int cx = (x << sps->log2_min_tb) >> sps->log2_ctb;
            const int cy = (y << sps->log2_min_tb) >> sps->log2_ctb;
            /* ⚠️ Tile scan. Raster is right only while there is one
             * tile, and wrong in a way that looks like a corrupt
             * picture rather than a wrong address. */
            const int rs = sps->ctb_width * cy + cx;
            const int ts = d->rs_to_ts ? d->rs_to_ts[rs] : rs;
            int32_t a = (int32_t)ts << (diff * 2);
            for (int i = 0; i < diff; i++) {
                const int m = 1 << i;
                a += ((m & x) ? m * m : 0) + ((m & y) ? 2 * m * m : 0);
            }
            d->min_tb_addr_zs[y * w + x] = a;
        }
    }
    return 0;
}

int hevcd_read_ctu(hevcd_t *d, int x0, int y0)
{
    const hevc_sps_t *sps = d->sps;
    const hevc_slice_t *sl = d->slice;

    /* Which tile everything inside this unit belongs to. Worked out once
     * here rather than in each availability test, which is handed a
     * neighbour and has no idea where "here" is. */
    d->tile_now = hevcd_tile_at(d, x0, y0);
    if (d->slice_of_ctb) {
        const int rs = (y0 >> sps->log2_ctb) * sps->ctb_width
                       + (x0 >> sps->log2_ctb);
        if (rs >= 0 && rs < sps->ctb_count) d->slice_of_ctb[rs] = d->slice_now;
    }

    if (sl->sao_luma || sl->sao_chroma)
        read_sao(d, x0 >> sps->log2_ctb, y0 >> sps->log2_ctb);

    read_quadtree(d, x0, y0, sps->log2_ctb, 0);
    return d->slice_end ? 1 : 0;
}
