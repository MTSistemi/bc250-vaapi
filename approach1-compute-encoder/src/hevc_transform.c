/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_transform.c - dequantisation and the inverse transforms,
 * Rec. ITU-T H.265 clauses 8.6.2 to 8.6.4.
 *
 * H.265 has one transform matrix, not four. The 32-point DCT is written
 * out in the standard and the 16-point one is its even rows, the 8-point
 * the even rows of those, and the 4-point the even rows of those again -
 * which is why every size can be read out of the same table with a stride.
 *
 * The one exception is a 4x4 intra luma block, which uses a DST instead.
 * Intra residuals grow away from the reference samples rather than being
 * flat, and the DST's first basis function does the same; at four samples
 * that is worth its own transform, and above four it stops being.
 */
#include "hevc_dec_internal.h"

#include <string.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

static inline int clip16(int v)
{
    return v < -32768 ? -32768 : (v > 32767 ? 32767 : v);
}

/* ⚠️ Clip3(0, (1 << BitDepth) - 1, v), not Clip3(0, 255, v). The two are
 * the same thing at eight bits and nowhere else. */
static inline int clip_pixel(int v, int bd)
{
    const int max = (1 << bd) - 1;
    return v < 0 ? 0 : (v > max ? max : v);
}

/* 8.6.3. m[x][y] is 16 throughout: a stream with its own quantisation
 * matrices is refused at the parameter set. */
void hevcd_dequantize(int16_t *coeff, int log2_size, int qp, int bd)
{
    const int shift = bd + log2_size - 5;      /* BitDepth + Log2(nTbS) - 5 */
    const int add = 1 << (shift - 1);
    const int64_t scale_of = (int64_t)hevcd_level_scale[qp % 6] << (qp / 6);
    const int count = 1 << (2 * log2_size);

    for (int i = 0; i < count; i++) {
        if (!coeff[i]) continue;
        const int64_t v = ((int64_t)coeff[i] * scale_of * 16 + add) >> shift;
        coeff[i] = (int16_t)clip16((int)(v < -32768 ? -32768
                                             : (v > 32767 ? 32767 : v)));
    }
}

/* The same with a quantisation matrix: m holds the factor of every
 * position, in the block's raster order. A matrix of sixteens is the flat
 * case above, sample for sample. */
void hevcd_dequantize_scaled(int16_t *coeff, int log2_size, int qp, int bd,
                             const uint8_t *m)
{
    const int shift = bd + log2_size - 5;
    const int add = 1 << (shift - 1);
    const int64_t scale_of = (int64_t)hevcd_level_scale[qp % 6] << (qp / 6);
    const int count = 1 << (2 * log2_size);

    for (int i = 0; i < count; i++) {
        if (!coeff[i]) continue;
        const int64_t v = ((int64_t)coeff[i] * scale_of * m[i] + add) >> shift;
        coeff[i] = (int16_t)clip16((int)(v < -32768 ? -32768
                                             : (v > 32767 ? 32767 : v)));
    }
}

/* 8.6.4.3: one line of the transform, over `n` points.
 *
 * ⚠️ The matrix row used is k * (32 / n), which is what "the even rows of
 * the even rows" means in practice. Reading hevcd_dct[k] instead would be
 * a 32-point transform truncated, which is a different function. */
static void line_transform(const int16_t *src, int stride, int32_t *out,
                           int n, int n_in)
{
    const int step = 32 / n;

    /* Which inputs are not zero, and which matrix row each one reaches
     * for. Everything else contributes nothing to any output - and past
     * n_in the caller already knows every input is zero. */
    const int8_t *row_m[32];
    int val[32];
    int count = 0;
    for (int k = 0; k < n_in; k++) {
        const int c = src[k * stride];
        if (!c) continue;
        val[count] = c;
        row_m[count] = hevcd_dct[k * step];
        count++;
    }
    if (!count) {
        memset(out, 0, (size_t)n * sizeof *out);
        return;
    }

#if defined(__x86_64__) || defined(_M_X64)
    /* Eight outputs at a time. The matrix is eight-bit and the input
     * sixteen, so each entry widens to sixteen and _mm_madd_epi16 carries
     * the product in thirty-two - the same pairing with a zero that the
     * weighted prediction uses, and for the same reason. */
    if (n >= 8) {
        const __m128i zero = _mm_setzero_si128();
        for (int i = 0; i < n; i += 8) {
            __m128i lo = zero, alto = zero;
            for (int j = 0; j < count; j++) {
                const __m128i cv =
                    _mm_set1_epi32((int32_t)(uint32_t)(uint16_t)val[j]);
                const __m128i m8 =
                    _mm_loadl_epi64((const __m128i *)(row_m[j] + i));
                const __m128i m16 =
                    _mm_srai_epi16(_mm_unpacklo_epi8(m8, m8), 8);
                lo = _mm_add_epi32(lo, _mm_madd_epi16(
                        _mm_unpacklo_epi16(m16, zero), cv));
                alto = _mm_add_epi32(alto, _mm_madd_epi16(
                        _mm_unpackhi_epi16(m16, zero), cv));
            }
            _mm_storeu_si128((__m128i *)(out + i), lo);
            _mm_storeu_si128((__m128i *)(out + i + 4), alto);
        }
        return;
    }
#endif

    for (int i = 0; i < n; i++) {
        int32_t s = 0;
        for (int j = 0; j < count; j++)
            s += (int32_t)row_m[j][i] * val[j];
        out[i] = s;
    }
}

/* The DST of 8.6.4.2, for a 4x4 intra luma block. Written as the standard
 * factors it rather than as a matrix product: four multiplications instead
 * of sixteen, and the same numbers. */
static void dst4(const int16_t *src, int stride, int32_t *out)
{
    const int c0 = src[0] + src[2 * stride];
    const int c1 = src[2 * stride] + src[3 * stride];
    const int c2 = src[0] - src[3 * stride];
    const int c3 = 74 * src[stride];

    out[0] = 29 * c0 + 55 * c1 + c3;
    out[1] = 55 * c2 - 29 * c1 + c3;
    /* ⚠️ x0 - x2 + x3, and none of x1. The factored form makes it easy to
     * write x1 in here by mistake, and the result is right in three of the
     * four outputs - which is exactly wrong enough to look like something
     * else's fault. */
    out[2] = 74 * (src[0] - src[2 * stride] + src[3 * stride]);
    out[3] = 55 * c0 + 29 * c2 - c3;
}

/* n values of one line, rounded, shifted and clipped to sixteen bits.
 * _mm_packs_epi32 saturates to exactly the range clip16() clips to. */
static void store_line(int16_t *o, const int32_t *v, int n, int add, int shift)
{
#if defined(__x86_64__) || defined(_M_X64)
    if (n >= 8) {
        const __m128i a = _mm_set1_epi32(add);
        const __m128i s = _mm_cvtsi32_si128(shift);
        for (int i = 0; i < n; i += 8) {
            const __m128i lo = _mm_sra_epi32(_mm_add_epi32(
                _mm_loadu_si128((const __m128i *)(v + i)), a), s);
            const __m128i hi = _mm_sra_epi32(_mm_add_epi32(
                _mm_loadu_si128((const __m128i *)(v + i + 4)), a), s);
            _mm_storeu_si128((__m128i *)(o + i), _mm_packs_epi32(lo, hi));
        }
        return;
    }
#endif
    for (int i = 0; i < n; i++) o[i] = (int16_t)clip16((v[i] + add) >> shift);
}

/* 8.6.4.2: columns first with a shift of seven, then rows with what is
 * left. Both stages clip to sixteen bits, which the standard says and
 * which matters: the intermediate really can leave the range.
 *
 * ⚠️ Most blocks hold a handful of coefficients in one corner. Looking for
 * them one line at a time cost more than the arithmetic: a 32x32 block
 * with three coefficients was 2048 positions checked and 2048 values
 * clipped. So the rectangle that holds coefficients is found first. The
 * columns outside it transform to zero and are not computed, and the
 * second stage reads only the columns inside it. The intermediate is kept
 * transposed, one column per row, so the first stage writes it in order. */
void hevcd_transform(int16_t *coeff, int log2_size, bool dst, int bd)
{
    const int n = 1 << log2_size;
    int16_t tmp[32 * 32];
    int32_t row[32];

    /* ⚠️ Only the second stage moves with the depth. The seven of the
     * first is seven at ten bits too: the standard fixes it, and making it
     * look symmetrical would be wrong. */
    const int shift = 20 - bd;
    const int add = 1 << (shift - 1);

    /* The DST exists at 4x4 only, and dst4() makes four values. Said
     * here in so many words, so that neither the loops nor the compiler
     * have to take n on trust. */
    if (dst && n == 4) {
        for (int x = 0; x < 4; x++) {
            dst4(coeff + x, 4, row);
            for (int y = 0; y < 4; y++)
                tmp[y * 4 + x] = (int16_t)clip16((row[y] + 64) >> 7);
        }
        for (int y = 0; y < 4; y++) {
            dst4(tmp + y * 4, 1, row);
            for (int x = 0; x < 4; x++)
                coeff[y * 4 + x] = (int16_t)clip16((row[x] + add) >> shift);
        }
        return;
    }

    int max_x = -1, max_y = -1;
    for (int y = 0; y < n; y++) {
        const int16_t *r = coeff + y * n;
        int x = n - 1;
        while (x >= 0 && !r[x]) x--;
        if (x >= 0) {
            max_y = y;
            if (x > max_x) max_x = x;
        }
    }
    if (max_x < 0) {                    /* nothing at all: zero out */
        memset(coeff, 0, (size_t)n * n * sizeof *coeff);
        return;
    }

    /* tmp[x * n + y] is the first stage's output for column x, row y. */
    for (int x = 0; x <= max_x; x++) {
        line_transform(coeff + x, n, row, n, max_y + 1);
        store_line(tmp + x * n, row, n, 64, 7);
    }
    for (int y = 0; y < n; y++) {
        line_transform(tmp + y, n, row, n, max_x + 1);
        store_line(coeff + y * n, row, n, add, shift);
    }
}

/* 8.6.2: a block whose transform was skipped is scaled and nothing else.
 * The seven and the final shift are the two stages the transform would
 * have done, with the transform taken out from between them. */
void hevcd_skip_transform(int16_t *coeff, int log2_size, int bd)
{
    const int count = 1 << (2 * log2_size);
    const int shift = 20 - bd;
    const int add = 1 << (shift - 1);
    for (int i = 0; i < count; i++)
        coeff[i] = (int16_t)clip16((((int)coeff[i] << 7) + add) >> shift);
}

/* One copy per depth, then the one line that chooses.
 *
 * ⚠️ `stride` is in samples. At eight bits that is also the number of
 * bytes, which is why a mistake here would stay hidden. */
#define BIT_DEPTH 8
#include "hevc_pixel.h"
#include "hevc_add_template.c"
#undef BIT_DEPTH

#define BIT_DEPTH 10
#include "hevc_pixel.h"
#include "hevc_add_template.c"
#undef BIT_DEPTH

/* ⚠️ The plane and the coordinates, not a pointer the caller has already
 * moved. `stride` counts samples: at eight bits that is the same number
 * of bytes, at ten it is half of them, and the only place that knows
 * which is here. */
void hevcd_add(uint8_t *plane, int stride, int x, int y,
               const int16_t *res, int log2_size, int bd)
{
    const size_t off = (size_t)y * stride + x;
    if (bd > 8) add_residual_10((uint16_t *)plane + off, stride,
                                res, log2_size);
    else        add_residual_8(plane + off, stride, res, log2_size);
}
