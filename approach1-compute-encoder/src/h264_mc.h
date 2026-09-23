/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * h264_mc.h - inter prediction, Rec. ITU-T H.264 clause 8.4.2.
 *
 * Luma is interpolated to quarter-sample accuracy with the standard's
 * six-tap filter, chroma to eighth-sample accuracy with a bilinear one.
 *
 * The source pointer must already sit inside a padded block: reading a
 * six-tap filter at the edge of a picture reaches two samples before and
 * four after, and H.264 lets a motion vector point outside the picture
 * entirely, so the caller copies the reference through
 * h264d_mc_fetch_luma()/h264d_mc_fetch_chroma() first. Clamping inside the
 * filter instead would put a branch on every tap.
 */
#ifndef BC250_H264_MC_H
#define BC250_H264_MC_H

#include <stdint.h>

/* How much padding h264d_mc_luma() reads around its block. */
#define H264D_MC_PAD_BEFORE 2
#define H264D_MC_PAD_AFTER  4

/* Make a w x h block of a reference plane readable with
 * H264D_MC_PAD_BEFORE samples before it and H264D_MC_PAD_AFTER after, in
 * both directions. Returns the integer-sample origin and writes the stride
 * to read it at into *out_stride.
 *
 * ⚠️ The result may point into the plane itself rather than into `dst`.
 * A block whose padded region is entirely inside the picture needs no copy
 * at all - the samples are already there - and at 1080p that is almost
 * every block. `dst` must still be (w + 6) * (h + 6), for the ones that
 * do reach an edge, where the edge sample is replicated outwards. */
const uint8_t *h264d_mc_fetch_luma(uint8_t *dst, int *out_stride,
                                   const uint8_t *plane, int stride,
                                   int plane_w, int plane_h,
                                   int x, int y, int w, int h);

/* The same for chroma, which the bilinear filter reads one sample past in
 * each direction. `dst` is (w + 1) * (h + 1). */
const uint8_t *h264d_mc_fetch_chroma(uint8_t *dst, int *out_stride,
                                     const uint8_t *plane, int stride,
                                     int plane_w, int plane_h,
                                     int x, int y, int w, int h);

/* Clause 8.4.2.2.1. `src` points at the integer sample inside a padded
 * block; `xfrac` and `yfrac` are the quarter-sample parts of the motion
 * vector, 0..3. */
void h264d_mc_luma(uint8_t *dst, int dst_stride,
                   const uint8_t *src, int src_stride,
                   int w, int h, int xfrac, int yfrac);

/* Clause 8.4.2.2.2. `xfrac` and `yfrac` are eighth-sample, 0..7. */
void h264d_mc_chroma(uint8_t *dst, int dst_stride,
                     const uint8_t *src, int src_stride,
                     int w, int h, int xfrac, int yfrac);

/* Clause 8.4.2.3.1, the default average of two predictions. */
void h264d_mc_average(uint8_t *dst, int dst_stride,
                      const uint8_t *a, int a_stride,
                      const uint8_t *b, int b_stride, int w, int h);

/* Clause 8.4.2.3.2, explicit weighted prediction from one list. */
void h264d_mc_weight(uint8_t *dst, int dst_stride,
                     const uint8_t *src, int src_stride,
                     int w, int h, int log2_denom, int weight, int offset);

/* Clause 8.4.2.3.2, explicit weighted prediction from both lists. */
void h264d_mc_weight_bi(uint8_t *dst, int dst_stride,
                        const uint8_t *a, int a_stride,
                        const uint8_t *b, int b_stride, int w, int h,
                        int log2_denom, int w0, int o0, int w1, int o1);

#endif /* BC250_H264_MC_H */
