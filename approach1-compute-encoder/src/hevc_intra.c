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

/* The prediction and the mode decision, once per bit depth. See
 * hevc_intra_template.c. */
#define BIT_DEPTH 8
#include "hevc_pixel.h"
#include "hevc_intra_template.c"
#undef BIT_DEPTH

#define BIT_DEPTH 10
#include "hevc_pixel.h"
#include "hevc_intra_template.c"
#undef BIT_DEPTH

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

int hevc_chroma_qp_from_luma(int qp_luma) {
    /* qPiCb = Clip3(-QpBdOffsetC, 57, QpY + pps_cb_qp_offset +
     * slice_cb_qp_offset). Both PPS chroma offsets are written as 0 and
     * pps_slice_chroma_qp_offsets_present_flag is 0 (see write_pps()), and
     * QpBdOffsetC is 0 at 8-bit, so qPi is just QpY clamped - and Cb and Cr
     * therefore share one value. */
    static const int qpc_30_43[14] = { 29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37 };
    int qpi = qp_luma < 0 ? 0 : (qp_luma > 57 ? 57 : qp_luma);
    if (qpi < 30) return qpi;
    if (qpi > 43) return qpi - 6;
    return qpc_30_43[qpi - 30];
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

/* ===================== ten bits ===================== */

/* The same transforms and quantizer at ten bits.
 *
 * Three numbers move with the bit depth, and nothing else does:
 *
 *   - the inverse transform's second stage shifts by 20 - BitDepth, so 10
 *     instead of 12 (8.6.4.2);
 *   - the dequantizer's bdShift is BitDepth + Log2(nTbS) - 5, so 7 instead
 *     of 5 (8.6.2 and 8.6.3);
 *   - the QP the scaling uses is Qp'Y = QpY + QpBdOffsetY, twelve more at
 *     ten bits, and the same for chroma. The caller adds that: `qp` here is
 *     the primed value, 0..63.
 *
 * The forward transform is ours to choose, and its first stage shifts by two
 * more than at eight bits so that a ten-bit residual, four times the
 * eight-bit one, comes out of it at the same scale. With the dequantizer's
 * two extra bits and QpBdOffset's factor of four, a given QP then produces
 * the same levels at either depth - which is what makes the rate control and
 * the QP the caller asked for mean the same thing at ten bits.
 *
 * Plain C: the eight-bit vector code has its shifts built in, and this path
 * has to be right before it is fast.
 */
#define HEVC_BDSHIFT_10 7

static void forward_transform_4x4_10(const int16_t residual[16], const int16_t M[4][4],
                                     int32_t out[16]) {
    int32_t tmp[4][4];
    for (int c = 0; c < 4; c++) {
        for (int i = 0; i < 4; i++) {
            int32_t sum = 0;
            for (int r = 0; r < 4; r++) sum += (int32_t)M[i][r] * residual[r * 4 + c];
            tmp[i][c] = (sum + 4) >> 3;
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

static void inverse_transform_4x4_10(const int16_t coeff[16], const int16_t M[4][4],
                                     int16_t out[16]) {
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
            out[r * 4 + c] = (int16_t)clip_coeff((sum + 512) >> 10);
        }
    }
}

void hevc_transform_quant_4x4_10(const int16_t residual[16], int qp, int use_dst,
                                  int16_t coeff_out[16]) {
    const uint64_t *r64 = (const uint64_t *)residual;
    if ((r64[0] | r64[1] | r64[2] | r64[3]) == 0ULL) {
        memset(coeff_out, 0, 16 * sizeof(int16_t));
        return;
    }

    int32_t raw[16];
    forward_transform_4x4_10(residual, use_dst ? DST4 : DCT4, raw);

    const int q = qp < 0 ? 0 : (qp > 63 ? 63 : qp);
    /* The algebraic inverse of the dequantizer below, rounded to nearest -
     * the eight-bit path's reciprocal table does the same division. */
    const uint64_t denom = (uint64_t)(HEVC_FLAT_M * levelScale[q % 6]) << (q / 6);
    for (int i = 0; i < 16; i++) {
        const int32_t v = raw[i];
        const uint64_t mag = (uint64_t)(v < 0 ? -(int64_t)v : v);
        int64_t level = (int64_t)(((mag << HEVC_BDSHIFT_10) + denom / 2) / denom);
        if (level > 32767) level = 32767;
        coeff_out[i] = (int16_t)(v < 0 ? -level : level);
    }
}

void hevc_dequant_itransform_4x4_10(const int16_t coeff[16], int qp, int use_dst,
                                     int16_t residual_out[16]) {
    const uint64_t *c64 = (const uint64_t *)coeff;
    if ((c64[0] | c64[1] | c64[2] | c64[3]) == 0ULL) {
        memset(residual_out, 0, 16 * sizeof(int16_t));
        return;
    }

    const int q = qp < 0 ? 0 : (qp > 63 ? 63 : qp);
    const int64_t scale = ((int64_t)HEVC_FLAT_M * levelScale[q % 6]) << (q / 6);
    int16_t dq[16];
    for (int i = 0; i < 16; i++) {
        int64_t val = (int64_t)coeff[i] * scale;
        val = (val + (1 << (HEVC_BDSHIFT_10 - 1))) >> HEVC_BDSHIFT_10;
        dq[i] = (int16_t)(val > 32767 ? 32767 : (val < -32768 ? -32768 : val));
    }
    inverse_transform_4x4_10(dq, use_dst ? DST4 : DCT4, residual_out);
}
