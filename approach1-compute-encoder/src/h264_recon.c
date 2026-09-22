/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * h264_recon.c - dequantisation and the inverse transforms.
 *
 * All of clause 8.5 that does not depend on neighbouring pixels: the scaling
 * of 8.5.12.1, the 4x4 and 8x8 inverse transforms of 8.5.12.2 and 8.5.13.2,
 * and the DC transforms of 8.5.10 and 8.5.11.
 *
 * The transforms are exact integer arithmetic. There is no "close enough"
 * here: two decoders that disagree by one in one sample drift apart over the
 * following inter-predicted pictures until the difference is a visible
 * smear, which is why H.264 specifies them to the shift.
 */
#include "h264_recon.h"

#include <string.h>

#include "h264_dec_tables.h"

static inline uint8_t clip_uint8(int v)
{
    return (uint8_t)(v & ~255 ? (-v) >> 31 : v);
}

/* ⚠️ The three columns of h264d_dequant4_init are NOT the standard's
 * v[m][0..2] in order. The standard picks by coordinate parity - both even,
 * both odd, mixed - and this table is indexed by how MANY of the two
 * coordinates are odd, so its middle column is the standard's "mixed" and
 * its last column is the standard's "both odd". Reading it in the standard's
 * order swaps two of the three classes, which is a small enough error to
 * look like mild ringing rather than a broken picture.
 */
static inline int category4(int n)
{
    return (n & 1) + ((n >> 2) & 1);        /* column parity + row parity */
}

/* The 8x8 classes repeat every four samples in both directions, so the class
 * of a raster position depends only on (row mod 4, column mod 4). */
static inline int category8(int n)
{
    int row = (n >> 3) & 3;
    int column = n & 3;
    return h264d_dequant8_init_scan[row * 4 + column];
}

void h264d_dequant_build(h264d_dequant_t *dq, int qp,
                         const uint8_t scaling4[6][16],
                         const uint8_t scaling8[6][64])
{
    /* The shift of 8.5.12.1 is folded in here for qp >= 24, where the scaling
     * is an exact left shift; below that the transform has to round, so the
     * factor is left unshifted and the rounding happens per coefficient. */
    const int q6 = qp % 6;
    for (int list_idx = 0; list_idx < 6; list_idx++) {
        for (int n = 0; n < 16; n++)
            dq->d4[list_idx][n] = (int32_t)h264d_dequant4_init[q6][category4(n)]
                             * (int32_t)scaling4[list_idx][n];
        for (int n = 0; n < 64; n++)
            dq->d8[list_idx][n] = (int32_t)h264d_dequant8_init[q6][category8(n)]
                             * (int32_t)scaling8[list_idx][n];
    }
    dq->qp = qp;
    dq->valid = 1;
}

void h264d_dequant_build_all(h264d_dequant_set_t *set,
                             const uint8_t scaling4[6][16],
                             const uint8_t scaling8[6][64])
{
    for (int r = 0; r < 6; r++)
        h264d_dequant_build(&set->per_rest[r], r, scaling4, scaling8);
    set->valid = 1;
}

/* 8.5.12.1. Returns the scaled coefficient. */
static inline int scale4(int c, int32_t factor, int qp)
{
    int per = qp / 6;
    if (per >= 4)
        return (c * factor) * (1 << (per - 4));
    return (c * factor + (1 << (3 - per))) >> (4 - per);
}

static inline int scale8(int c, int32_t factor, int qp)
{
    int per = qp / 6;
    if (per >= 6)
        return (c * factor) * (1 << (per - 6));
    return (c * factor + (1 << (5 - per))) >> (6 - per);
}

/* 8.5.12.2, the 4x4 inverse transform. One dimension at a time; the
 * butterflies are the standard's, written out rather than looped so the
 * shifts stay visible. */
static inline void idct4_rows(int32_t t[16])
{
    for (int i = 0; i < 4; i++) {
        int32_t *d = t + i * 4;
        int32_t e0 = d[0] + d[2];
        int32_t e1 = d[0] - d[2];
        int32_t e2 = (d[1] >> 1) - d[3];
        int32_t e3 = d[1] + (d[3] >> 1);
        d[0] = e0 + e3;
        d[1] = e1 + e2;
        d[2] = e1 - e2;
        d[3] = e0 - e3;
    }
}

void h264d_idct4_add(uint8_t *dst, int stride, int16_t block[16],
                     const int32_t dequant[16], int qp, bool dc_pronto)
{
    int32_t t[16];

    t[0] = dc_pronto ? block[0]
                     : (block[0] ? scale4(block[0], dequant[0], qp) : 0);
    for (int n = 1; n < 16; n++)
        t[n] = block[n] ? scale4(block[n], dequant[n], qp) : 0;

    idct4_rows(t);

    for (int i = 0; i < 4; i++) {
        int32_t d0 = t[i], d1 = t[i + 4], d2 = t[i + 8], d3 = t[i + 12];
        int32_t e0 = d0 + d2;
        int32_t e1 = d0 - d2;
        int32_t e2 = (d1 >> 1) - d3;
        int32_t e3 = d1 + (d3 >> 1);
        t[i]      = e0 + e3;
        t[i + 4]  = e1 + e2;
        t[i + 8]  = e1 - e2;
        t[i + 12] = e0 - e3;
    }

    for (int y = 0; y < 4; y++) {
        uint8_t *r = dst + y * stride;
        for (int x = 0; x < 4; x++)
            r[x] = clip_uint8(r[x] + ((t[y * 4 + x] + 32) >> 6));
    }
    memset(block, 0, 16 * sizeof(int16_t));
}

void h264d_idct4_dc_add(uint8_t *dst, int stride, int dc)
{
    int v = (dc + 32) >> 6;
    for (int y = 0; y < 4; y++) {
        uint8_t *r = dst + y * stride;
        r[0] = clip_uint8(r[0] + v);
        r[1] = clip_uint8(r[1] + v);
        r[2] = clip_uint8(r[2] + v);
        r[3] = clip_uint8(r[3] + v);
    }
}

/* 8.5.13.2, the 8x8 inverse transform. */
static inline void idct8_one_dimension(int32_t d[8], int stride)
{
#define D(i) d[(i) * stride]
    int32_t a0 = D(0) + D(4);
    int32_t a2 = D(0) - D(4);
    int32_t a4 = (D(2) >> 1) - D(6);
    int32_t a6 = (D(6) >> 1) + D(2);

    int32_t b0 = a0 + a6;
    int32_t b2 = a2 + a4;
    int32_t b4 = a2 - a4;
    int32_t b6 = a0 - a6;

    int32_t a1 = -D(3) + D(5) - D(7) - (D(7) >> 1);
    int32_t a3 =  D(1) + D(7) - D(3) - (D(3) >> 1);
    int32_t a5 = -D(1) + D(7) + D(5) + (D(5) >> 1);
    int32_t a7 =  D(3) + D(5) + D(1) + (D(1) >> 1);

    int32_t b1 = (a7 >> 2) + a1;
    int32_t b3 =  a3 + (a5 >> 2);
    int32_t b5 = (a3 >> 2) - a5;
    int32_t b7 =  a7 - (a1 >> 2);

    D(0) = b0 + b7;
    D(7) = b0 - b7;
    D(1) = b2 + b5;
    D(6) = b2 - b5;
    D(2) = b4 + b3;
    D(5) = b4 - b3;
    D(3) = b6 + b1;
    D(4) = b6 - b1;
#undef D
}

void h264d_idct8_add(uint8_t *dst, int stride, int16_t block[64],
                     const int32_t dequant[64], int qp)
{
    int32_t t[64];
    for (int n = 0; n < 64; n++)
        t[n] = block[n] ? scale8(block[n], dequant[n], qp) : 0;

    for (int i = 0; i < 8; i++)
        idct8_one_dimension(t + i * 8, 1);
    for (int i = 0; i < 8; i++)
        idct8_one_dimension(t + i, 8);

    for (int y = 0; y < 8; y++) {
        uint8_t *r = dst + y * stride;
        for (int x = 0; x < 8; x++)
            r[x] = clip_uint8(r[x] + ((t[y * 8 + x] + 32) >> 6));
    }
    memset(block, 0, 64 * sizeof(int16_t));
}

void h264d_idct8_dc_add(uint8_t *dst, int stride, int dc)
{
    int v = (dc + 32) >> 6;
    for (int y = 0; y < 8; y++) {
        uint8_t *r = dst + y * stride;
        for (int x = 0; x < 8; x++)
            r[x] = clip_uint8(r[x] + v);
    }
}

/* 8.5.10. The Hadamard transform of the luma DC coefficients, then their
 * scaling - in that order, unlike every other block, where the scaling comes
 * first. */
void h264d_luma_dc_transform(int16_t dc[16], int qp, int32_t dequant0)
{
    int32_t t[16];
    for (int n = 0; n < 16; n++)
        t[n] = dc[n];

    for (int i = 0; i < 4; i++) {
        int32_t *d = t + i * 4;
        int32_t a = d[0] + d[2], b = d[0] - d[2];
        int32_t c = d[1] - d[3], e = d[1] + d[3];
        d[0] = a + e; d[1] = b + c; d[2] = b - c; d[3] = a - e;
    }
    for (int i = 0; i < 4; i++) {
        int32_t d0 = t[i], d1 = t[i + 4], d2 = t[i + 8], d3 = t[i + 12];
        int32_t a = d0 + d2, b = d0 - d2;
        int32_t c = d1 - d3, e = d1 + d3;
        t[i] = a + e; t[i + 4] = b + c; t[i + 8] = b - c; t[i + 12] = a - e;
    }

    const int per = qp / 6;
    for (int n = 0; n < 16; n++) {
        int64_t v;
        if (per >= 6)
            v = ((int64_t)t[n] * dequant0) * (1 << (per - 6));
        else
            v = (((int64_t)t[n] * dequant0) + (1 << (5 - per))) >> (6 - per);
        dc[n] = (int16_t)v;
    }
}

/* 8.5.11, the 2x2 Hadamard of the chroma DC coefficients of a 4:2:0 block. */
void h264d_chroma_dc_transform(int16_t dc[4], int qp, int32_t dequant0)
{
    int32_t a = dc[0] + dc[1];
    int32_t b = dc[0] - dc[1];
    int32_t c = dc[2] + dc[3];
    int32_t d = dc[2] - dc[3];

    int32_t t[4] = { a + c, b + d, a - c, b - d };
    const int per = qp / 6;
    for (int n = 0; n < 4; n++)
        dc[n] = (int16_t)(((int64_t)t[n] * dequant0) << per >> 5);
}
