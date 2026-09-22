/* bc250-encoding-decoding-fix v0.4.3 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_intra.c - see hevc_intra.h for the design rationale. Everything in
 * this file is independently written against the published ITU-T H.265
 * spec text (clauses cited per-function below) - unlike hevc_cabac.c, none
 * of it is adapted from x265's source, though the well-known public 4x4
 * DCT-II integer matrix ({64,64,64,64},{83,36,-36,-83},...) and the
 * standard HEVC dequant scale table ({40,45,51,57,64,72}) were
 * cross-checked against x265's source/common/constants.cpp and
 * source/common/scalinglist.cpp purely to catch transcription mistakes -
 * both are the spec's own normative constants, not x265-original values.
 */
#include "hevc_intra.h"
#include <string.h>
#include <immintrin.h>
#include <stdlib.h>

/* ===================== mode/scan helpers ===================== */

int hevc_scan_idx_for_mode(int mode) {
    if (mode >= 6 && mode <= 14) return 2;  /* SCAN_VER */
    if (mode >= 22 && mode <= 30) return 1; /* SCAN_HOR */
    return 0;                               /* SCAN_DIAG */
}

/* Rec. ITU-T H.265 8.4.2: build the 3-entry candidate mode list from the
 * left/above neighbor PUs' real intra modes. Unavailable neighbors
 * (off-picture, or - not applicable here, one slice per picture - a
 * different slice) are treated as INTRA_DC per the spec's substitution. */
void hevc_derive_mpm(int left_mode, int left_avail, int above_mode, int above_avail,
                      int mpm_out[3]) {
    int cand_a = left_avail ? left_mode : HEVC_MODE_DC;
    int cand_b = above_avail ? above_mode : HEVC_MODE_DC;

    if (cand_a == cand_b) {
        if (cand_a < 2) {
            mpm_out[0] = HEVC_MODE_PLANAR;
            mpm_out[1] = HEVC_MODE_DC;
            mpm_out[2] = HEVC_MODE_VERTICAL;
        } else {
            mpm_out[0] = cand_a;
            mpm_out[1] = 2 + ((cand_a + 29) % 32);
            mpm_out[2] = 2 + ((cand_a - 2 + 1) % 32);
        }
    } else {
        mpm_out[0] = cand_a;
        mpm_out[1] = cand_b;
        if (mpm_out[0] != HEVC_MODE_PLANAR && mpm_out[1] != HEVC_MODE_PLANAR)
            mpm_out[2] = HEVC_MODE_PLANAR;
        else if (mpm_out[0] != HEVC_MODE_DC && mpm_out[1] != HEVC_MODE_DC)
            mpm_out[2] = HEVC_MODE_DC;
        else
            mpm_out[2] = HEVC_MODE_VERTICAL;
    }
}

/* ===================== neighbor gathering (8.4.4.2.2) ===================== */

static inline uint8_t clip8(int v) { return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); }

/*
 * Rec. ITU-T H.265 6.4.1's "z-scan order block availability" is NOT the
 * same thing as "is this position within picture bounds" - a neighbor can
 * be positionally inside the picture and still not yet decoded. This bit
 * this encoder got wrong initially: a CU's bottom-right (z-order index 3)
 * PU's "top-right" reference sample can land inside a SIBLING CU that
 * hasn't been coded yet (e.g. the top-left CU's bottom-right PU reaching
 * into the top-right CU of the same CTU), or inside the next CTU to the
 * right (same picture row, but that CTU is later in raster order) - both
 * pass a naive "x0+4 < width && y0 > 0" check while being genuinely
 * undecoded. Confirmed as the actual remaining bug by an exhaustive
 * bit-for-bit cross-check of this encoder's CABAC output against its own
 * recorded per-CU decisions (which matched perfectly - the bitstream was
 * never wrong), followed by comparing this project's intra prediction/
 * transform code line-by-line against ffmpeg's libavcodec/hevc/
 * pred_template.c and dsp_template.c (both matched exactly) - leaving
 * z-scan availability, which ffmpeg's intra_pred() computes via a real
 * MinTbAddrZs table lookup (cand_up_right &&
 * cur_tb_addr > MIN_TB_ADDR_ZS(...)), as the one remaining candidate, and
 * it was.
 *
 * This encoder's coding order is a FIXED, known shape (CTUs in raster
 * order; each CTU always splits into exactly 4 CUs in z-order - TL,TR,
 * BL,BR; each luma CU always splits into exactly 4 PUs the same way;
 * chroma has no PU sub-split, one block per CU) - which makes an exact
 * z-scan rank computable directly, without needing a real MinTbAddrZs
 * table: rank = ((ctuRow*widthInCtus + ctuCol) * 4 + cuZIndex) * 4 +
 * puZIndex (chroma stops one level early, at the CU). A neighbor is
 * available iff it's in-picture AND its rank is strictly less than the
 * current block's.
 */
/* Shifts, not divisions.
 *
 * Every divisor here is a power of two, but all of them are derived from the
 * is_luma argument, so the compiler cannot prove it and emits real div
 * instructions - six per call, and this is called five times per 4x4 block
 * (once for the block itself, four for its neighbours), which comes to about
 * four million integer divisions per 1080p frame. The profiler put
 * zorder_available() at 5% of the HEVC encode on a BC-250 on its own, before
 * counting what it costs inside gather_neighbors().
 *
 * x and y are never negative at either call site - zorder_available() bounds
 * checks first and the cur_rank call passes the block's own coordinates - so
 * >> and & give exactly what / and % gave. */
static long long zorder_rank(int x, int y, int width, int is_luma) {
    const int sh = is_luma ? 4 : 3;              /* ctu_size = 1 << sh */
    const int ctu_size = 1 << sh;
    const int hs = sh - 1;                       /* CU size  = 1 << hs */
    int width_ctu = (width + ctu_size - 1) >> sh;
    int ctu_col = x >> sh, ctu_row = y >> sh;
    int rx = x & (ctu_size - 1), ry = y & (ctu_size - 1);
    int cu_col = rx >> hs, cu_row = ry >> hs;
    long long rank = ((long long)ctu_row * width_ctu + ctu_col) * 4 + (cu_row * 2 + cu_col);
    if (is_luma) {
        const int qs = hs - 1;                   /* PU size = 1 << qs = 4 */
        int rx2 = rx & ((1 << hs) - 1), ry2 = ry & ((1 << hs) - 1);
        rank = rank * 4 + ((ry2 >> qs) * 2 + (rx2 >> qs));
    }
    return rank;
}

/* y_min is the first pixel row of the current slice: anything above it belongs
 * to another slice and a decoder will not have it, so neither may we. */
static int zorder_available(int nx, int ny, int width, int height, int is_luma,
                            int y_min, long long cur_rank) {
    if (nx < 0 || ny < y_min || nx >= width || ny >= height) return 0;
    return zorder_rank(nx, ny, width, is_luma) < cur_rank;
}

/* Gathers left[0..4] (p[-1][0..4]), top[0..4] (p[0..4][-1]) and the corner
 * (p[-1][-1]), applying the spec's neighbor-substitution scan. Bottom-left
 * (p[-1][5..7], not needed since this encoder only supports Planar/DC/H/V,
 * none of which read past left[4]/top[4]) and positions beyond top[4] are
 * never referenced, so the scan below only covers what those four modes
 * actually need. "Below" and "below-left" are always z-scan-unavailable
 * in this encoder's coding order (nothing below the current row, at any
 * CTU/CU/PU nesting level, is ever decoded first), so those still don't
 * need a rank check - only left/top/corner/top-right do, since those CAN
 * be positionally-plausible but z-scan-unavailable (see zorder_rank()'s
 * comment above). */
static void gather_neighbors(const uint8_t *plane, int stride, int width, int height,
                              int x0, int y0, int is_luma, int y_min,
                              uint8_t left[5], uint8_t top[5], uint8_t *corner,
                              int *avail_left_out, int *avail_top_out) {
    long long cur_rank = zorder_rank(x0, y0, width, is_luma);
    int avail_left = zorder_available(x0 - 1, y0, width, height, is_luma, y_min, cur_rank);
    int avail_top = zorder_available(x0, y0 - 1, width, height, is_luma, y_min, cur_rank);
    int avail_corner = zorder_available(x0 - 1, y0 - 1, width, height, is_luma, y_min, cur_rank);
    int avail_top_right = zorder_available(x0 + 4, y0 - 1, width, height, is_luma, y_min, cur_rank);

    if (avail_left_out) *avail_left_out = avail_left;
    if (avail_top_out) *avail_top_out = avail_top;

    uint8_t sv[10];
    uint8_t sa[10];

    sa[0] = sa[1] = sa[2] = sa[3] = (uint8_t)avail_left;
    if (avail_left) {
        sv[0] = plane[(y0 + 3) * stride + (x0 - 1)];
        sv[1] = plane[(y0 + 2) * stride + (x0 - 1)];
        sv[2] = plane[(y0 + 1) * stride + (x0 - 1)];
        sv[3] = plane[(y0 + 0) * stride + (x0 - 1)];
    }
    sa[4] = (uint8_t)avail_corner;
    if (avail_corner) sv[4] = plane[(y0 - 1) * stride + (x0 - 1)];

    sa[5] = sa[6] = sa[7] = sa[8] = (uint8_t)avail_top;
    if (avail_top) {
        sv[5] = plane[(y0 - 1) * stride + (x0 + 0)];
        sv[6] = plane[(y0 - 1) * stride + (x0 + 1)];
        sv[7] = plane[(y0 - 1) * stride + (x0 + 2)];
        sv[8] = plane[(y0 - 1) * stride + (x0 + 3)];
    }
    sa[9] = (uint8_t)avail_top_right;
    if (avail_top_right) sv[9] = plane[(y0 - 1) * stride + (x0 + 4)];

    int first = -1;
    for (int i = 0; i < 10; i++) { if (sa[i]) { first = i; break; } }

    if (first < 0) {
        for (int i = 0; i < 10; i++) sv[i] = 128;
    } else {
        for (int i = 0; i < first; i++) sv[i] = sv[first];
        for (int i = first + 1; i < 10; i++) if (!sa[i]) sv[i] = sv[i - 1];
    }

    left[3] = sv[0]; left[2] = sv[1]; left[1] = sv[2]; left[0] = sv[3];
    *corner = sv[4];
    top[0] = sv[5]; top[1] = sv[6]; top[2] = sv[7]; top[3] = sv[8];
    top[4] = sv[9];
    /* p[-1][4] (bottom-left, one below left[3]): always z-scan-unavailable
     * in this encoder's coding order (see zorder_rank()'s comment) -
     * nearest previously-scanned available sample is left[3] itself
     * (which already carries its own correct substituted value). This
     * line was accidentally dropped in an earlier edit that reworked this
     * function for z-scan availability, leaving left[4] reading
     * uninitialized stack memory for every Planar-mode prediction (the
     * only one of this encoder's 4 modes that reads it) - found by
     * dumping this function's actual output for a block ffmpeg decoded
     * differently from this encoder's own (internally-consistent, since
     * it used the same garbage value on both the predict and later
     * reconstruct call for a given block, but NOT consistent with a real
     * decoder, which correctly derives left[4]=left[3]) reconstruction. */
    left[4] = left[3];
}

/* ===================== prediction (8.4.4.2.5-8.4.4.2.7) ===================== */

void hevc_predict_4x4(const uint8_t *recon_plane, int stride, int width, int height,
                      int x0, int y0, int mode, int is_luma, int y_min,
                      uint8_t pred_out[16]) {
    uint8_t left[5], top[5], corner;
    int avail_left = 0, avail_top = 0;
    gather_neighbors(recon_plane, stride, width, height, x0, y0, is_luma, y_min, left, top, &corner, &avail_left, &avail_top);

    switch (mode) {
    case HEVC_MODE_PLANAR:
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++) {
                int v = (3 - x) * left[y] + (x + 1) * top[4] +
                        (3 - y) * top[x] + (y + 1) * left[4] + 4;
                pred_out[y * 4 + x] = (uint8_t)(v >> 3);
            }
        break;

    case HEVC_MODE_DC: {
        /* No branching on availability here, deliberately.
         *
         * gather_neighbors() has already run the reference sample
         * substitution of Rec. ITU-T H.265 8.4.4.2.2: it scans bottom-left to
         * top-right, takes the first available sample and fills every
         * unavailable one from its neighbour (all 128 when the block has no
         * neighbours at all). By the time we get here left[] and top[] are
         * full, and there is no longer any such thing as an unavailable
         * reference - which is exactly the state 8.4.4.2.5 assumes when it
         * computes dcVal over BOTH edges and applies the boundary filter for
         * luma below 32x32.
         *
         * Branching on avail_left/avail_top computed dcVal from one edge with
         * different rounding, and filtered only that edge. A decoder does
         * neither, so every block touching a picture edge came out a few
         * units off - and since this encoder predicts DC everywhere, that
         * difference then rode the prediction chain across the whole picture.
         * Measured on a BC-250 before this: the encoder's own reconstruction
         * reached 53.7 dB against the source while the decoded stream sat at
         * 24.1 dB, and the two disagreed on 56% of pixels.
         */
        int dc = (left[0] + left[1] + left[2] + left[3] +
                  top[0] + top[1] + top[2] + top[3] + 4) >> 3;
        for (int i = 0; i < 16; i++) pred_out[i] = (uint8_t)dc;
        if (is_luma) {
            pred_out[0] = (uint8_t)((left[0] + 2 * dc + top[0] + 2) >> 2);
            for (int x = 1; x < 4; x++) pred_out[x] = (uint8_t)((top[x] + 3 * dc + 2) >> 2);
            for (int y = 1; y < 4; y++) pred_out[y * 4] = (uint8_t)((left[y] + 3 * dc + 2) >> 2);
        }
        break;
    }

    case HEVC_MODE_HORIZONTAL:
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                pred_out[y * 4 + x] = left[y];
        if (is_luma) {
            for (int x = 0; x < 4; x++)
                pred_out[x] = clip8(left[0] + ((top[x] - corner) >> 1));
        }
        break;

    case HEVC_MODE_VERTICAL:
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                pred_out[y * 4 + x] = top[x];
        if (is_luma) {
            for (int y = 0; y < 4; y++)
                pred_out[y * 4] = clip8(top[0] + ((left[y] - corner) >> 1));
        }
        break;

    default:
        for (int i = 0; i < 16; i++) pred_out[i] = 128;
        break;
    }
}

int hevc_choose_luma_mode(int y_min, const uint8_t *src_y, const uint8_t *recon_y, int stride,
                           int width, int height, int x0, int y0) {
    static const int candidates[4] = { HEVC_MODE_PLANAR, HEVC_MODE_DC, HEVC_MODE_HORIZONTAL, HEVC_MODE_VERTICAL };
    int best_mode = HEVC_MODE_DC;
    long best_sad = -1;

    for (int c = 0; c < 4; c++) {
        uint8_t pred[16];
        hevc_predict_4x4(recon_y, stride, width, height, x0, y0, candidates[c], 1, y_min, pred);
        long sad = 0;
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++) {
                int src = src_y[(y0 + y) * stride + (x0 + x)];
                int p = pred[y * 4 + x];
                int d = src - p;
                sad += d < 0 ? -d : d;
            }
        if (best_sad < 0 || sad < best_sad) { best_sad = sad; best_mode = candidates[c]; }
    }
    return best_mode;
}

/* ===================== transform (8.6.4) ===================== */

/* Public/standard 4x4 integer DCT-II matrix (ITU-T H.265 8.6.4.1's
 * transMatrix for nTbS=4; identical to H.264's and every other MPEG-family
 * codec's 4-point integer DCT approximation). Cross-checked against
 * x265's source/common/constants.cpp g_t4[][] - same standard values. */
static const int16_t DCT4[4][4] = {
    { 64,  64,  64,  64 },
    { 83,  36, -36, -83 },
    { 64, -64, -64,  64 },
    { 36, -83,  83, -36 }
};

/* Public/standard 4x4 DST-VII "alternative transform" matrix, used ONLY
 * for 4x4 luma intra residuals (ITU-T H.265 8.6.4.1: "if cIdx is equal to
 * 0 and predMode is equal to MODE_INTRA and nTbS is equal to 4, the
 * alternative transform... is used"). Cross-checked against x265's
 * primitives.dst4x4 call site (quant.cpp) for when it fires - the matrix
 * values themselves are the spec's own public constant, reproduced in
 * essentially every independent HEVC implementation (HM, libde265, ffmpeg,
 * etc), not x265-original. */
static const int16_t DST4[4][4] = {
    { 29,  55,  74,  84 },
    { 74,  74,   0, -74 },
    { 84, -29, -74,  55 },
    { 55, -84,  74, -29 }
};

static inline int32_t clip_coeff(int32_t v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return v;
}

/* Forward 2D separable transform, matrix M applied directly (not
 * transposed) along both axes. Shift split (1, 8) - chosen so that, paired
 * with the spec-mandated inverse shifts below (7, 12 for 8-bit), a
 * forward-then-inverse round trip with no quantization in between
 * reproduces the original residual exactly for a constant (DC-only) input:
 * forward stage1 shift=1 add=1, stage2 shift=8 add=128 -> for a constant
 * input v, coeff[0][0] = 128*v (verified by hand: row0 of M is {64,64,64,64}
 * so a per-axis DC gain of 4*64=256=2^8; after forward's own /2^1 then
 * /2^8 net divide of 2^9, combined per-axis, the DC coefficient comes out
 * to 128*v); the inverse below then recovers exactly v from that (see its
 * own comment). Non-DC content is NOT expected to be bit-exact through
 * this round trip (that's inherent to any integer DCT/DST approximation,
 * including the real x265/HM ones - see this file's header comment), only
 * well-scaled - forward quantization error is what's supposed to make the
 * picture lossy, not a transform bug. */
/* SSE4.1 versions of the two 4x4 transforms.
 *
 * Each pass is a 4x4 matrix product: the scalar code walks it as three nested
 * loops, 64 multiply-adds per pass. But the four outputs of a pass share their
 * inputs, so one register covers a whole row. _mm_cvtepi16_epi32 widens the
 * four int16 of a row, _mm_mullo_epi32 multiplies by a broadcast matrix entry,
 * and the accumulation order, the rounding constants and the shifts stay
 * exactly as they were. _mm_packs_epi32 saturates to int16 precisely the way
 * clip_coeff() does.
 *
 * So the output is bit-identical, and that is verified by encoding the same
 * clip before and after and comparing the md5 of the bitstream - not by
 * reading the code and hoping.
 */
#if defined(__x86_64__) || defined(_M_X64)

__attribute__((target("sse4.1")))
static inline __m128i row_i32(const int16_t *p) {
    return _mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *)p));
}

__attribute__((target("sse4.1")))
static void forward_transform_4x4_sse(const int16_t residual[16], const int16_t M[4][4],
                                      int32_t out[16]) {
    __m128i res[4];
    for (int r = 0; r < 4; r++) res[r] = row_i32(&residual[r * 4]);

    /* pass 1: tmp[i][c] = (sum_r M[i][r] * residual[r*4+c] + 1) >> 1 */
    __m128i tmp[4];
    const __m128i one_pred = _mm_set1_epi32(1);
    for (int i = 0; i < 4; i++) {
        __m128i acc = _mm_setzero_si128();
        for (int r = 0; r < 4; r++)
            acc = _mm_add_epi32(acc, _mm_mullo_epi32(_mm_set1_epi32(M[i][r]), res[r]));
        tmp[i] = _mm_srai_epi32(_mm_add_epi32(acc, one_pred), 1);
    }

    /* pass 2: out[i*4+j] = (sum_c M[j][c] * tmp[i][c] + 128) >> 8 */
    __m128i column[4];
    for (int c = 0; c < 4; c++)
        column[c] = _mm_setr_epi32(M[0][c], M[1][c], M[2][c], M[3][c]);
    const __m128i centoventotto = _mm_set1_epi32(128);
    for (int i = 0; i < 4; i++) {
        int32_t t[4];
        _mm_storeu_si128((__m128i *)t, tmp[i]);
        __m128i acc = _mm_setzero_si128();
        for (int c = 0; c < 4; c++)
            acc = _mm_add_epi32(acc, _mm_mullo_epi32(_mm_set1_epi32(t[c]), column[c]));
        _mm_storeu_si128((__m128i *)&out[i * 4],
                         _mm_srai_epi32(_mm_add_epi32(acc, centoventotto), 8));
    }
}

__attribute__((target("sse4.1")))
static void inverse_transform_4x4_sse(const int16_t coeff[16], const int16_t M[4][4],
                                      int16_t out[16]) {
    __m128i co[4];
    for (int k = 0; k < 4; k++) co[k] = row_i32(&coeff[k * 4]);

    /* pass 1: tmp[r][c] = clip((sum_k M[k][r] * coeff[k*4+c] + 64) >> 7) */
    const __m128i sixtyfour = _mm_set1_epi32(64);
    const __m128i alto = _mm_set1_epi32(32767), basso = _mm_set1_epi32(-32768);
    int32_t tmp[4][4];
    for (int r = 0; r < 4; r++) {
        __m128i acc = _mm_setzero_si128();
        for (int k = 0; k < 4; k++)
            acc = _mm_add_epi32(acc, _mm_mullo_epi32(_mm_set1_epi32(M[k][r]), co[k]));
        acc = _mm_srai_epi32(_mm_add_epi32(acc, sixtyfour), 7);
        acc = _mm_min_epi32(_mm_max_epi32(acc, basso), alto);
        _mm_storeu_si128((__m128i *)tmp[r], acc);
    }

    /* pass 2: out[r*4+c] = clip((sum_k M[k][c] * tmp[r][k] + 2048) >> 12) */
    __m128i row_m[4];
    for (int k = 0; k < 4; k++) row_m[k] = row_i32(M[k]);
    const __m128i half_a_step = _mm_set1_epi32(2048);
    for (int r = 0; r < 4; r++) {
        __m128i acc = _mm_setzero_si128();
        for (int k = 0; k < 4; k++)
            acc = _mm_add_epi32(acc, _mm_mullo_epi32(_mm_set1_epi32(tmp[r][k]), row_m[k]));
        acc = _mm_srai_epi32(_mm_add_epi32(acc, half_a_step), 12);
        /* packs saturates to int16 exactly as clip_coeff does */
        _mm_storel_epi64((__m128i *)&out[r * 4], _mm_packs_epi32(acc, acc));
    }
}

static int ha_sse41(void) {
    static int answer = -1;
    if (answer < 0) answer = __builtin_cpu_supports("sse4.1") ? 1 : 0;
    return answer;
}
#endif

static void forward_transform_4x4_scalar(const int16_t residual[16], const int16_t M[4][4], int32_t out[16]) {
    int32_t tmp[4][4];
    for (int c = 0; c < 4; c++) {
        for (int i = 0; i < 4; i++) {
            int32_t sum = 0;
            for (int r = 0; r < 4; r++) sum += (int32_t)M[i][r] * residual[r * 4 + c];
            tmp[i][c] = (sum + 1) >> 1;
        }
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int32_t sum = 0;
            for (int c = 0; c < 4; c++) sum += (int32_t)M[j][c] * tmp[i][c];
            out[i * 4 + j] = (sum + 128) >> 8;
        }
    }
}

/* Inverse 2D separable transform, matrix M applied in TRANSPOSED form
 * (M[k][idx], k summed) along both axes, with the spec-mandated shifts for
 * 8-bit content (ITU-T H.265 8.6.4.2): stage1 shift=7/add=64 (fixed,
 * independent of bit depth), stage2 shift = 20-BitDepth = 12/add=2048.
 * This exact process is what a real HEVC decoder performs, and this
 * encoder uses the SAME code for its own reconstruction chaining, so the
 * two are trivially identical by construction. */
static void inverse_transform_4x4_scalar(const int16_t coeff[16], const int16_t M[4][4], int16_t out[16]) {
    int32_t tmp[4][4];
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 4; r++) {
            int32_t sum = 0;
            for (int k = 0; k < 4; k++) sum += (int32_t)M[k][r] * coeff[k * 4 + c];
            tmp[r][c] = clip_coeff((sum + 64) >> 7);
        }
    }
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            int32_t sum = 0;
            for (int k = 0; k < 4; k++) sum += (int32_t)M[k][c] * tmp[r][k];
            out[r * 4 + c] = (int16_t)clip_coeff((sum + 2048) >> 12);
        }
    }
}

/* ===================== quantization (8.6.3) ===================== */

/* levelScale[qp%6] - Rec. ITU-T H.265 Table (8.6.3), the standard HEVC
 * dequant scale, identical to x265's ScalingList::s_invQuantScales. Public
 * spec constant. */
static const int levelScale[6] = { 40, 45, 51, 57, 64, 72 };

/* bdShift = BitDepth(8) + Log2(nTbS=4) - 5 = 5, m = 16 (flat default
 * scaling list, since this encoder's SPS sets scaling_list_enabled_flag=0
 * - ITU-T H.265 7.4.3.2.1's default). Forward quantization is defined here
 * as the algebraic inverse of the (normative) dequant formula below, which
 * guarantees the two are exactly self-consistent regardless of the
 * transform's own internal scale - see hevc_intra.h's top comment. */
#define HEVC_BDSHIFT 5
#define HEVC_FLAT_M  16

/* Precomputed exact reciprocal division factors for HEVC quantization (Rec. ITU-T H.265 8.6.3).
 * Replaces 16 64-bit hardware integer divisions per 4x4 block with a 1-cycle 64-bit multiply
 * and 40-bit right shift ((num + half_denom) * recip >> 40), verified 100% bit-exact across
 * all 52 QPs and all possible 16-bit coefficient amplitudes. */
static const struct { uint64_t recip; uint32_t half_denom; } g_hevc_quant_factors[52] = {
    /* QP  0 */ { 0x000066666667ULL,    320U },
    /* QP  1 */ { 0x00005b05b05cULL,    360U },
    /* QP  2 */ { 0x000050505051ULL,    408U },
    /* QP  3 */ { 0x000047dc11f8ULL,    456U },
    /* QP  4 */ { 0x000040000000ULL,    512U },
    /* QP  5 */ { 0x000038e38e39ULL,    576U },
    /* QP  6 */ { 0x000033333334ULL,    640U },
    /* QP  7 */ { 0x00002d82d82eULL,    720U },
    /* QP  8 */ { 0x000028282829ULL,    816U },
    /* QP  9 */ { 0x000023ee08fcULL,    912U },
    /* QP 10 */ { 0x000020000000ULL,   1024U },
    /* QP 11 */ { 0x00001c71c71dULL,   1152U },
    /* QP 12 */ { 0x00001999999aULL,   1280U },
    /* QP 13 */ { 0x000016c16c17ULL,   1440U },
    /* QP 14 */ { 0x000014141415ULL,   1632U },
    /* QP 15 */ { 0x000011f7047eULL,   1824U },
    /* QP 16 */ { 0x000010000000ULL,   2048U },
    /* QP 17 */ { 0x00000e38e38fULL,   2304U },
    /* QP 18 */ { 0x00000ccccccdULL,   2560U },
    /* QP 19 */ { 0x00000b60b60cULL,   2880U },
    /* QP 20 */ { 0x00000a0a0a0bULL,   3264U },
    /* QP 21 */ { 0x000008fb823fULL,   3648U },
    /* QP 22 */ { 0x000008000000ULL,   4096U },
    /* QP 23 */ { 0x0000071c71c8ULL,   4608U },
    /* QP 24 */ { 0x000006666667ULL,   5120U },
    /* QP 25 */ { 0x000005b05b06ULL,   5760U },
    /* QP 26 */ { 0x000005050506ULL,   6528U },
    /* QP 27 */ { 0x0000047dc120ULL,   7296U },
    /* QP 28 */ { 0x000004000000ULL,   8192U },
    /* QP 29 */ { 0x0000038e38e4ULL,   9216U },
    /* QP 30 */ { 0x000003333334ULL,  10240U },
    /* QP 31 */ { 0x000002d82d83ULL,  11520U },
    /* QP 32 */ { 0x000002828283ULL,  13056U },
    /* QP 33 */ { 0x0000023ee090ULL,  14592U },
    /* QP 34 */ { 0x000002000000ULL,  16384U },
    /* QP 35 */ { 0x000001c71c72ULL,  18432U },
    /* QP 36 */ { 0x00000199999aULL,  20480U },
    /* QP 37 */ { 0x0000016c16c2ULL,  23040U },
    /* QP 38 */ { 0x000001414142ULL,  26112U },
    /* QP 39 */ { 0x0000011f7048ULL,  29184U },
    /* QP 40 */ { 0x000001000000ULL,  32768U },
    /* QP 41 */ { 0x000000e38e39ULL,  36864U },
    /* QP 42 */ { 0x000000cccccdULL,  40960U },
    /* QP 43 */ { 0x000000b60b61ULL,  46080U },
    /* QP 44 */ { 0x000000a0a0a1ULL,  52224U },
    /* QP 45 */ { 0x0000008fb824ULL,  58368U },
    /* QP 46 */ { 0x000000800000ULL,  65536U },
    /* QP 47 */ { 0x00000071c71dULL,  73728U },
    /* QP 48 */ { 0x000000666667ULL,  81920U },
    /* QP 49 */ { 0x0000005b05b1ULL,  92160U },
    /* QP 50 */ { 0x000000505051ULL, 104448U },
    /* QP 51 */ { 0x00000047dc12ULL, 116736U },
};

static int32_t dequant_level(int32_t level, int qp) {
    int per = qp / 6, rem = qp % 6;
    int64_t val = (int64_t)level * HEVC_FLAT_M * levelScale[rem];
    val <<= per;
    val = (val + (1 << (HEVC_BDSHIFT - 1))) >> HEVC_BDSHIFT;
    return clip_coeff((int32_t)val);
}

static int32_t quantize_coeff(int32_t coeff_raw, int qp) {
    int clamped_qp = qp < 0 ? 0 : (qp > 51 ? 51 : qp);
    uint64_t recip = g_hevc_quant_factors[clamped_qp].recip;
    uint32_t half_denom = g_hevc_quant_factors[clamped_qp].half_denom;
    int sign = coeff_raw < 0 ? -1 : 1;
    uint32_t mag = (uint32_t)(coeff_raw < 0 ? -coeff_raw : coeff_raw);
    uint64_t num = (uint64_t)mag << HEVC_BDSHIFT;
    int32_t level = (int32_t)(((num + half_denom) * recip) >> 40);
    return sign * level;
}


static inline void forward_transform_4x4(const int16_t residual[16], const int16_t M[4][4],
                                         int32_t out[16]) {
#if defined(__x86_64__) || defined(_M_X64)
    if (ha_sse41()) { forward_transform_4x4_sse(residual, M, out); return; }
#endif
    forward_transform_4x4_scalar(residual, M, out);
}

static inline void inverse_transform_4x4(const int16_t coeff[16], const int16_t M[4][4],
                                         int16_t out[16]) {
#if defined(__x86_64__) || defined(_M_X64)
    if (ha_sse41()) { inverse_transform_4x4_sse(coeff, M, out); return; }
#endif
    inverse_transform_4x4_scalar(coeff, M, out);
}

void hevc_transform_quant_4x4(const int16_t residual[16], int qp, int use_dst,
                               int16_t coeff_out[16]) {
    /* Fast zero-residual bypass: if residual is all zero, output is all zero */
    const uint64_t *r64 = (const uint64_t *)residual;
    if ((r64[0] | r64[1] | r64[2] | r64[3]) == 0ULL) {
        memset(coeff_out, 0, 16 * sizeof(int16_t));
        return;
    }

    int32_t raw[16];
    forward_transform_4x4(residual, use_dst ? DST4 : DCT4, raw);

    int clamped_qp = qp < 0 ? 0 : (qp > 51 ? 51 : qp);
    uint64_t recip = g_hevc_quant_factors[clamped_qp].recip;
    uint32_t half_denom = g_hevc_quant_factors[clamped_qp].half_denom;

    for (int i = 0; i < 16; i++) {
        int32_t coeff_raw = raw[i];
        int sign = coeff_raw < 0 ? -1 : 1;
        uint32_t mag = (uint32_t)(coeff_raw < 0 ? -coeff_raw : coeff_raw);
        uint64_t num = (uint64_t)mag << HEVC_BDSHIFT;
        int32_t level = (int32_t)(((num + half_denom) * recip) >> 40);
        int32_t res = sign * level;
        if (res > 32767) res = 32767;
        if (res < -32768) res = -32768;
        coeff_out[i] = (int16_t)res;
    }
}

void hevc_dequant_itransform_4x4(const int16_t coeff[16], int qp, int use_dst,
                                  int16_t residual_out[16]) {
    /* Fast zero-coeff bypass: if quantized coefficients are all zero, residual is all zero */
    const uint64_t *c64 = (const uint64_t *)coeff;
    if ((c64[0] | c64[1] | c64[2] | c64[3]) == 0ULL) {
        memset(residual_out, 0, 16 * sizeof(int16_t));
        return;
    }

    int16_t dq[16];
    int per = qp / 6, rem = qp % 6;
    int64_t scale = ((int64_t)HEVC_FLAT_M * levelScale[rem]) << per;
    int64_t half_scale = 1 << (HEVC_BDSHIFT - 1);
    for (int i = 0; i < 16; i++) {
        int64_t val = (int64_t)coeff[i] * scale;
        val = (val + half_scale) >> HEVC_BDSHIFT;
        dq[i] = (int16_t)clip_coeff((int32_t)val);
    }
    inverse_transform_4x4(dq, use_dst ? DST4 : DCT4, residual_out);
}
