/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * h264_recon.h - dequantisation and the inverse transforms, Rec. ITU-T H.264
 * clause 8.5.
 */
#ifndef BC250_H264_RECON_H
#define BC250_H264_RECON_H

#include <stdbool.h>
#include <stdint.h>

/* The dequantisation factors for one quantisation parameter, laid out in
 * raster order so the transform can walk them alongside the coefficients.
 * Built once per (scaling list, QP) pair rather than per block: a picture
 * uses at most a handful of QPs and rebuilding these 64 multiplies per
 * macroblock was measurable. */
typedef struct {
    int32_t d4[6][16];      /* [scaling list][position] */
    int32_t d8[6][64];
    int     qp;
    int     valid;
} h264d_dequant_t;

/* All six at once, indexed by qp % 6.
 *
 * ⚠️ The factors depend on the quantisation parameter only through qp % 6.
 * The rest of it is a shift, and the transforms apply that themselves from
 * the qp they are given. So this is rebuilt when the scaling lists change
 * and at no other time. */
typedef struct {
    h264d_dequant_t per_rest[6];
    int valid;
} h264d_dequant_set_t;

void h264d_dequant_build(h264d_dequant_t *dq, int qp,
                         const uint8_t scaling4[6][16],
                         const uint8_t scaling8[6][64]);

void h264d_dequant_build_all(h264d_dequant_set_t *set,
                             const uint8_t scaling4[6][16],
                             const uint8_t scaling8[6][64]);

/* 8.5.12.1 scaling, then 8.5.12.2 transformation, then the add and the clip.
 * `block` holds the coefficients in raster order and is destroyed. */
/* `dc_pronto` says the DC has already been through its own transform and
 * scaling - an Intra_16x16 or a chroma block - so it is used as it stands.
 * ⚠️ This used to be signalled by passing a null dequant array, which meant
 * the one case that had to be handled specially was also the one that
 * dereferenced a null pointer if anything slipped through. */
void h264d_idct4_add(uint8_t *dst, int stride, int16_t block[16],
                     const int32_t dequant[16], int qp, bool dc_pronto);
void h264d_idct8_add(uint8_t *dst, int stride, int16_t block[64],
                     const int32_t dequant[64], int qp);

/* The DC-only case: every coefficient but the first is zero, which is common
 * enough to be worth not running the full transform for. */
void h264d_idct4_dc_add(uint8_t *dst, int stride, int dc);
void h264d_idct8_dc_add(uint8_t *dst, int stride, int dc);

/* 8.5.10: the Hadamard transform of the sixteen luma DC coefficients of an
 * Intra_16x16 macroblock, and their scaling. In place. */
void h264d_luma_dc_transform(int16_t dc[16], int qp, int32_t dequant0);

/* 8.5.11: the same for the four chroma DC coefficients of a 4:2:0 block. */
void h264d_chroma_dc_transform(int16_t dc[4], int qp, int32_t dequant0);

#endif /* BC250_H264_RECON_H */
