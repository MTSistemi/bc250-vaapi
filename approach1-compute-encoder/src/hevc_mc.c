/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_mc.c - fetching the samples a motion vector points at,
 * Rec. ITU-T H.265 clause 8.5.3.3.
 *
 * A motion vector points a quarter of a sample at a time for luma and an
 * eighth for chroma, so most of the time it points between samples and the
 * ones in between have to be made up. H.265 makes them with an eight-tap
 * filter where H.264 used six, which is most of the reason its motion
 * compensation is sharper and most of the reason it is slower.
 *
 * ⚠️ Everything in here works at fourteen bits and clips to eight only at
 * the very end. Two predictions averaged after each has been rounded to
 * eight bits are wrong by half a level per sample, everywhere, for ever.
 */
#include "hevc_dec_internal.h"

#include <string.h>

#define MAX_SIDE 64

/* ⚠️ Clip3(0, (1 << BitDepth) - 1, v). The same thing as a clip to 255
 * at eight bits and nowhere else. */
static inline int clip_pixel(int v, int bd)
{
    const int max = (1 << bd) - 1;
    return v < 0 ? 0 : (v > max ? max : v);
}

/* ⚠️ A motion vector may point off the edge of the reference picture, and
 * legitimately: an object entering the frame was not there before. The
 * edge sample is repeated outwards rather than the fetch being refused. */
/* ------------------------------------------------- the vector paths */

/* SSE2 is part of the x86-64 ABI, so the two stages that take fourteen
 * bits back down to eight need no runtime check. The filter itself wants
 * _mm_maddubs_epi16, which is SSSE3, and asks first.
 *
 * ⚠️ Nothing here is allowed to disagree with the scalar twin beside it by
 * so much as a level. Both are exercised by the same suites, which compare
 * whole sequences with ffmpeg byte for byte, so a vector path that gets an
 * order or a shift wrong fails loudly instead of quietly softening the
 * picture. That is what makes these safe to write. */
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

static int ha_ssse3(void)
{
    static int answer = -1;
    if (answer < 0) answer = __builtin_cpu_supports("ssse3") ? 1 : 0;
    return answer;
}

/* Two consecutive taps, broadcast: the shape _mm_maddubs_epi16 wants,
 * which multiplies unsigned samples by signed taps and adds each adjacent
 * pair into sixteen bits. */
__attribute__((target("ssse3")))
static inline __m128i pair(const int8_t *f, int k)
{
    const uint16_t two_bytes = (uint16_t)((uint8_t)f[k])
                            | (uint16_t)((uint8_t)f[k + 1] << 8);
    return _mm_set1_epi16((int16_t)two_bytes);
}

/* The same pair as two sixteen-bit lanes, for _mm_madd_epi16 when the
 * samples coming in are already fourteen-bit. */
__attribute__((target("ssse3")))
static inline __m128i pair32(const int8_t *f, int k)
{
    const uint32_t two_shorts = (uint32_t)(uint16_t)(int16_t)f[k]
                             | ((uint32_t)(uint16_t)(int16_t)f[k + 1] << 16);
    return _mm_set1_epi32((int32_t)two_shorts);
}

/* ⚠️ Eight outputs of an eight tap filter need fifteen bytes and the load
 * takes sixteen. At the end of a row that sixteenth byte can be one past
 * the end of the reference picture, and a picture whose last row ends on a
 * page boundary would fault on a read the filter never uses. */
__attribute__((target("ssse3")))
static inline __m128i load16(const uint8_t *p, int n_available)
{
    if (n_available >= 16) return _mm_loadu_si128((const __m128i *)p);
    uint8_t t[16];
    memset(t, 0, sizeof t);
    memcpy(t, p, (size_t)n_available);
    return _mm_loadu_si128((const __m128i *)t);
}

/* Along a row, eight or four taps, eight outputs at a time.
 *
 * The taps of one output overlap the taps of the next, so one load covers
 * all eight: _mm_shuffle_epi8 lays out the pair each output needs for tap
 * k and k+1, and four (or two) maddubs and three (or one) adds finish it.
 *
 * ⚠️ maddubs saturates. It cannot bite here - the largest H.265 luma
 * filter sums to 112, so a pair reaches at most 75 * 255 and the whole
 * sum 112 * 255, both inside sixteen bits - and that is why the adds may
 * be plain wrapping adds that match the scalar truncation exactly. */
#define HORIZONTAL_V(nome, N, SCALAR)                                       \
__attribute__((target("ssse3")))                                              \
static void nome(const uint8_t *src, int sp, int w, int h,                    \
                 const int8_t *f, int16_t *out, int pf)                     \
{                                                                             \
    const __m128i c0 = pair(f, 0), c2 = pair(f, 2);                       \
    const __m128i c4 = (N) == 8 ? pair(f, 4) : _mm_setzero_si128();         \
    const __m128i c6 = (N) == 8 ? pair(f, 6) : _mm_setzero_si128();         \
    const __m128i m0 = _mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4,                  \
                                     4, 5, 5, 6, 6, 7, 7, 8);                 \
    const __m128i m2 = _mm_setr_epi8(2, 3, 3, 4, 4, 5, 5, 6,                  \
                                     6, 7, 7, 8, 8, 9, 9, 10);                \
    const __m128i m4 = _mm_setr_epi8(4, 5, 5, 6, 6, 7, 7, 8,                  \
                                     8, 9, 9, 10, 10, 11, 11, 12);            \
    const __m128i m6 = _mm_setr_epi8(6, 7, 7, 8, 8, 9, 9, 10,                 \
                                     10, 11, 11, 12, 12, 13, 13, 14);         \
    for (int r = 0; r < h; r++) {                                             \
        const uint8_t *s = src + (size_t)r * sp;                              \
        int16_t *o = out + (size_t)r * pf;                                  \
        int c = 0;                                                            \
        for (; c + 8 <= w; c += 8) {                                          \
            const __m128i v = load16(s + c, w + (N) - 1 - c);                 \
            __m128i a = _mm_maddubs_epi16(_mm_shuffle_epi8(v, m0), c0);       \
            a = _mm_add_epi16(a, _mm_maddubs_epi16(                           \
                    _mm_shuffle_epi8(v, m2), c2));                            \
            if ((N) == 8) {                                                   \
                a = _mm_add_epi16(a, _mm_maddubs_epi16(                       \
                        _mm_shuffle_epi8(v, m4), c4));                        \
                a = _mm_add_epi16(a, _mm_maddubs_epi16(                       \
                        _mm_shuffle_epi8(v, m6), c6));                        \
            }                                                                 \
            _mm_storeu_si128((__m128i *)(o + c), a);                          \
        }                                                                     \
        if (c < w) SCALAR(s, o, c, w, f);                                    \
    }                                                                         \
}

/* Down a column, from whole samples. One load per tap row, and
 * _mm_unpacklo_epi8 puts tap k and tap k+1 of the same column side by side
 * where maddubs expects them. */
#define VERTICAL_V(nome, N, SCALAR)                                         \
__attribute__((target("ssse3")))                                              \
static void nome(const uint8_t *src, int sp, int w, int h,                    \
                 const int8_t *f, int16_t *out, int pf)                     \
{                                                                             \
    const __m128i c0 = pair(f, 0), c2 = pair(f, 2);                       \
    const __m128i c4 = (N) == 8 ? pair(f, 4) : _mm_setzero_si128();         \
    const __m128i c6 = (N) == 8 ? pair(f, 6) : _mm_setzero_si128();         \
    for (int r = 0; r < h; r++) {                                             \
        const uint8_t *s = src + (size_t)r * sp;                              \
        int16_t *o = out + (size_t)r * pf;                                  \
        int c = 0;                                                            \
        for (; c + 8 <= w; c += 8) {                                          \
            __m128i l[8];                                                     \
            for (int k = 0; k < (N); k++)                                     \
                l[k] = _mm_loadl_epi64(                                       \
                    (const __m128i *)(s + (size_t)k * sp + c));               \
            __m128i a = _mm_maddubs_epi16(_mm_unpacklo_epi8(l[0], l[1]), c0); \
            a = _mm_add_epi16(a, _mm_maddubs_epi16(                           \
                    _mm_unpacklo_epi8(l[2], l[3]), c2));                      \
            if ((N) == 8) {                                                   \
                a = _mm_add_epi16(a, _mm_maddubs_epi16(                       \
                        _mm_unpacklo_epi8(l[4], l[5]), c4));                  \
                a = _mm_add_epi16(a, _mm_maddubs_epi16(                       \
                        _mm_unpacklo_epi8(l[6], l[7]), c6));                  \
            }                                                                 \
            _mm_storeu_si128((__m128i *)(o + c), a);                          \
        }                                                                     \
        if (c < w) SCALAR(s, sp, o, c, w, f);                                \
    }                                                                         \
}

/* Down a column, from the fourteen-bit output of a horizontal pass. These
 * no longer fit in sixteen bits once multiplied, so _mm_madd_epi16 carries
 * them in thirty-two.
 *
 * ⚠️ And the way back down is a shuffle, not a pack. _mm_packs_epi32
 * saturates; the scalar path truncates. They agree on every stream that
 * conforms and part company on one that does not, which is the kind of
 * difference that surfaces years later in a crash report. */
#define VERTICAL16_V(nome, N, SCALAR)                                       \
__attribute__((target("ssse3")))                                              \
static void nome(const int16_t *src, int sp, int w, int h,                    \
                 const int8_t *f, int16_t *out, int pf)                     \
{                                                                             \
    const __m128i c0 = pair32(f, 0), c2 = pair32(f, 2);                   \
    const __m128i c4 = (N) == 8 ? pair32(f, 4) : _mm_setzero_si128();       \
    const __m128i c6 = (N) == 8 ? pair32(f, 6) : _mm_setzero_si128();       \
    const __m128i giu = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13,               \
                                      -1, -1, -1, -1, -1, -1, -1, -1);        \
    for (int r = 0; r < h; r++) {                                             \
        const int16_t *s = src + (size_t)r * sp;                              \
        int16_t *o = out + (size_t)r * pf;                                  \
        int c = 0;                                                            \
        for (; c + 8 <= w; c += 8) {                                          \
            __m128i v[8];                                                     \
            for (int k = 0; k < (N); k++)                                     \
                v[k] = _mm_loadu_si128(                                       \
                    (const __m128i *)(s + (size_t)k * sp + c));               \
            __m128i lo = _mm_madd_epi16(_mm_unpacklo_epi16(v[0], v[1]), c0);  \
            __m128i hi = _mm_madd_epi16(_mm_unpackhi_epi16(v[0], v[1]), c0);  \
            lo = _mm_add_epi32(lo, _mm_madd_epi16(                            \
                    _mm_unpacklo_epi16(v[2], v[3]), c2));                     \
            hi = _mm_add_epi32(hi, _mm_madd_epi16(                            \
                    _mm_unpackhi_epi16(v[2], v[3]), c2));                     \
            if ((N) == 8) {                                                   \
                lo = _mm_add_epi32(lo, _mm_madd_epi16(                        \
                        _mm_unpacklo_epi16(v[4], v[5]), c4));                 \
                hi = _mm_add_epi32(hi, _mm_madd_epi16(                        \
                        _mm_unpackhi_epi16(v[4], v[5]), c4));                 \
                lo = _mm_add_epi32(lo, _mm_madd_epi16(                        \
                        _mm_unpacklo_epi16(v[6], v[7]), c6));                 \
                hi = _mm_add_epi32(hi, _mm_madd_epi16(                        \
                        _mm_unpackhi_epi16(v[6], v[7]), c6));                 \
            }                                                                 \
            lo = _mm_srai_epi32(lo, 6);                                       \
            hi = _mm_srai_epi32(hi, 6);                                       \
            _mm_storeu_si128((__m128i *)(o + c),                              \
                _mm_unpacklo_epi64(_mm_shuffle_epi8(lo, giu),                 \
                                   _mm_shuffle_epi8(hi, giu)));               \
        }                                                                     \
        if (c < w) SCALAR(s, sp, o, c, w, f);                                \
    }                                                                         \
}

/* The tails, for the columns at the right edge that do not fill a
 * register. The same arithmetic, one sample at a time. */
#define TAIL_HORIZ(N)                                                          \
static inline void tail_horiz##N(const uint8_t *s, int16_t *o, int c, int w,   \
                                const int8_t *f)                              \
{                                                                             \
    for (; c < w; c++) {                                                      \
        int v = 0;                                                            \
        for (int k = 0; k < (N); k++) v += f[k] * s[c + k];                   \
        o[c] = (int16_t)v;                                                    \
    }                                                                         \
}

#define TAIL_VERT(N, TYPE, DOWN, suffix)                                     \
static inline void tail_vert##suffix(const TYPE *s, int sp, int16_t *o,     \
                                       int c, int w, const int8_t *f)         \
{                                                                             \
    for (; c < w; c++) {                                                      \
        int v = 0;                                                            \
        for (int k = 0; k < (N); k++) v += f[k] * s[(size_t)k * sp + c];      \
        o[c] = (int16_t)(v >> (DOWN));                                         \
    }                                                                         \
}

TAIL_HORIZ(8)
TAIL_HORIZ(4)
TAIL_VERT(8, uint8_t, 0, 8)
TAIL_VERT(4, uint8_t, 0, 4)
TAIL_VERT(8, int16_t, 6, 8_16)
TAIL_VERT(4, int16_t, 6, 4_16)

HORIZONTAL_V(horiz8_v, 8, tail_horiz8)
HORIZONTAL_V(horiz4_v, 4, tail_horiz4)
VERTICAL_V(vert8_v, 8, tail_vert8)
VERTICAL_V(vert4_v, 4, tail_vert4)
VERTICAL16_V(vert8_16_v, 8, tail_vert8_16)
VERTICAL16_V(vert4_16_v, 4, tail_vert4_16)

/* A motion vector that lands on a whole sample: nothing to filter, just
 * the samples moved up into fourteen bits. */
static void copy14_v(const uint8_t *src, int sp, int w, int h,
                      int16_t *out, int pf)
{
    const __m128i zero = _mm_setzero_si128();
    for (int r = 0; r < h; r++) {
        const uint8_t *s = src + (size_t)r * sp;
        int16_t *o = out + (size_t)r * pf;
        int c = 0;
        for (; c + 8 <= w; c += 8) {
            const __m128i v = _mm_loadl_epi64((const __m128i *)(s + c));
            _mm_storeu_si128((__m128i *)(o + c),
                _mm_slli_epi16(_mm_unpacklo_epi8(v, zero), 6));
        }
        for (; c < w; c++) o[c] = (int16_t)(s[c] << 6);
    }
}

/* One prediction down to eight bits. ⚠️ Adding 32 in sixteen bits is safe
 * only because the largest fourteen-bit intermediate is 112 * 255. */
static void one_pred_v(uint8_t *dst, int stride, int w, int h,
                  const int16_t *a, int stride_a)
{
    const __m128i trentadue = _mm_set1_epi16(32);
    for (int r = 0; r < h; r++) {
        const int16_t *s = a + (size_t)r * stride_a;
        uint8_t *o = dst + (size_t)r * stride;
        int c = 0;
        for (; c + 8 <= w; c += 8) {
            __m128i v = _mm_loadu_si128((const __m128i *)(s + c));
            v = _mm_srai_epi16(_mm_add_epi16(v, trentadue), 6);
            _mm_storel_epi64((__m128i *)(o + c), _mm_packus_epi16(v, v));
        }
        for (; c < w; c++) o[c] = (uint8_t)clip_pixel((s[c] + 32) >> 6, 8);
    }
}

/* Two averaged. ⚠️ Two fourteen-bit values added do not fit in sixteen,
 * so this one widens first - which is also why it is the slower of the
 * two and worth having in vectors at all. */
static void two_pred_v(uint8_t *dst, int stride, int w, int h,
                  const int16_t *a, const int16_t *b, int stride_p)
{
    const __m128i sixtyfour = _mm_set1_epi32(64);
    for (int r = 0; r < h; r++) {
        const int16_t *sa = a + (size_t)r * stride_p;
        const int16_t *sb = b + (size_t)r * stride_p;
        uint8_t *o = dst + (size_t)r * stride;
        int c = 0;
        for (; c + 8 <= w; c += 8) {
            const __m128i va = _mm_loadu_si128((const __m128i *)(sa + c));
            const __m128i vb = _mm_loadu_si128((const __m128i *)(sb + c));
            __m128i lo = _mm_add_epi32(
                _mm_srai_epi32(_mm_unpacklo_epi16(va, va), 16),
                _mm_srai_epi32(_mm_unpacklo_epi16(vb, vb), 16));
            __m128i hi = _mm_add_epi32(
                _mm_srai_epi32(_mm_unpackhi_epi16(va, va), 16),
                _mm_srai_epi32(_mm_unpackhi_epi16(vb, vb), 16));
            lo = _mm_srai_epi32(_mm_add_epi32(lo, sixtyfour), 7);
            hi = _mm_srai_epi32(_mm_add_epi32(hi, sixtyfour), 7);
            const __m128i sixteen = _mm_packs_epi32(lo, hi);
            _mm_storel_epi64((__m128i *)(o + c),
                             _mm_packus_epi16(sixteen, sixteen));
        }
        for (; c < w; c++) o[c] = (uint8_t)clip_pixel((sa[c] + sb[c] + 64) >> 7, 8);
    }
}

/* 8.5.3.3.4.3 in vectors. The product of a fourteen-bit sample and a
 * weight does not fit in sixteen bits, so _mm_madd_epi16 carries it in
 * thirty-two: pairing each sample with a zero and each weight with a zero
 * turns one multiply-add into exactly the multiply we want, sign and all,
 * with no widening step of its own. */
static void one_weighted_v(uint8_t *dst, int stride, int w, int h,
                         const int16_t *a, int stride_a,
                         int weight, int off, int den)
{
    const int log2wd = den + 6;
    const __m128i zero = _mm_setzero_si128();
    const __m128i pv = _mm_set1_epi32((int32_t)(uint32_t)(uint16_t)weight);
    const __m128i rounding = _mm_set1_epi32(1 << (log2wd - 1));
    const __m128i ov = _mm_set1_epi32(off);
    const __m128i giu = _mm_cvtsi32_si128(log2wd);

    for (int r = 0; r < h; r++) {
        const int16_t *s = a + (size_t)r * stride_a;
        uint8_t *o = dst + (size_t)r * stride;
        int c = 0;
        for (; c + 8 <= w; c += 8) {
            const __m128i v = _mm_loadu_si128((const __m128i *)(s + c));
            __m128i lo = _mm_madd_epi16(_mm_unpacklo_epi16(v, zero), pv);
            __m128i hi = _mm_madd_epi16(_mm_unpackhi_epi16(v, zero), pv);
            lo = _mm_add_epi32(_mm_sra_epi32(_mm_add_epi32(lo, rounding), giu), ov);
            hi = _mm_add_epi32(_mm_sra_epi32(_mm_add_epi32(hi, rounding), giu), ov);
            const __m128i sixteen = _mm_packs_epi32(lo, hi);
            _mm_storel_epi64((__m128i *)(o + c),
                             _mm_packus_epi16(sixteen, sixteen));
        }
        for (; c < w; c++) {
            const int v = s[c];
            o[c] = (uint8_t)clip_pixel(((v * weight + (1 << (log2wd - 1))) >> log2wd) + off, 8);
        }
    }
}

static void two_weighted_v(uint8_t *dst, int stride, int w, int h,
                         const int16_t *a, const int16_t *b, int stride_p,
                         int pa, int pb, int oa, int ob, int den)
{
    const int log2wd = den + 6;
    const __m128i zero = _mm_setzero_si128();
    const __m128i pav = _mm_set1_epi32((int32_t)(uint32_t)(uint16_t)pa);
    const __m128i pbv = _mm_set1_epi32((int32_t)(uint32_t)(uint16_t)pb);
    const __m128i rounding = _mm_set1_epi32((oa + ob + 1) << log2wd);
    const __m128i giu = _mm_cvtsi32_si128(log2wd + 1);

    for (int r = 0; r < h; r++) {
        const int16_t *sa = a + (size_t)r * stride_p;
        const int16_t *sb = b + (size_t)r * stride_p;
        uint8_t *o = dst + (size_t)r * stride;
        int c = 0;
        for (; c + 8 <= w; c += 8) {
            const __m128i va = _mm_loadu_si128((const __m128i *)(sa + c));
            const __m128i vb = _mm_loadu_si128((const __m128i *)(sb + c));
            __m128i lo = _mm_add_epi32(
                _mm_madd_epi16(_mm_unpacklo_epi16(va, zero), pav),
                _mm_madd_epi16(_mm_unpacklo_epi16(vb, zero), pbv));
            __m128i hi = _mm_add_epi32(
                _mm_madd_epi16(_mm_unpackhi_epi16(va, zero), pav),
                _mm_madd_epi16(_mm_unpackhi_epi16(vb, zero), pbv));
            lo = _mm_sra_epi32(_mm_add_epi32(lo, rounding), giu);
            hi = _mm_sra_epi32(_mm_add_epi32(hi, rounding), giu);
            const __m128i sixteen = _mm_packs_epi32(lo, hi);
            _mm_storel_epi64((__m128i *)(o + c),
                             _mm_packus_epi16(sixteen, sixteen));
        }
        for (; c < w; c++)
            o[c] = (uint8_t)clip_pixel((sa[c] * pa + sb[c] * pb
                              + ((oa + ob + 1) << log2wd)) >> (log2wd + 1), 8);
    }
}

#endif /* x86-64 */

/* ----------------------------------------------- and the scalar twins */

/* One pass of the filter across a rectangle, reading whole samples along
 * a row.
 *
 * ⚠️ The tap count is a compile time constant in each instance. With it
 * in a variable the compiler keeps neither the eight coefficients in
 * registers nor any chance of doing several samples at once, and this is
 * three quarters of the decoder's time. */
#define HORIZONTAL(nome, N)                                               \
static void nome(const uint8_t *src, int sp, int w, int h,                 \
                 const int8_t *f, int16_t *out, int pf, int down)          \
{                                                                          \
    for (int r = 0; r < h; r++) {                                          \
        const uint8_t *s = src + (size_t)r * sp;                           \
        int16_t *o = out + (size_t)r * pf;                               \
        for (int c = 0; c < w; c++) {                                      \
            int v = 0;                                                     \
            for (int k = 0; k < N; k++) v += f[k] * s[c + k];              \
            o[c] = (int16_t)(v >> down);                                   \
        }                                                                  \
    }                                                                      \
}

/* The same down a column. `TYPE` is whole samples on the way in and the
 * fourteen-bit output of a horizontal pass on the way back, `DOWN` the
 * shift that takes the second pass back to fourteen bits. */
#define VERTICAL(nome, N, TYPE)                                            \
static void nome(const TYPE *src, int sp, int w, int h,                    \
                 const int8_t *f, int16_t *out, int pf, int down)          \
{                                                                          \
    for (int r = 0; r < h; r++) {                                          \
        const TYPE *s = src + (size_t)r * sp;                              \
        int16_t *o = out + (size_t)r * pf;                               \
        for (int c = 0; c < w; c++) {                                      \
            int v = 0;                                                     \
            for (int k = 0; k < N; k++) v += f[k] * s[(size_t)k * sp + c]; \
            o[c] = (int16_t)(v >> down);                                   \
        }                                                                  \
    }                                                                      \
}

HORIZONTAL(horiz8, 8)
HORIZONTAL(horiz4, 4)
VERTICAL(vert8, 8, uint8_t)
VERTICAL(vert4, 4, uint8_t)
VERTICAL(vert8_16, 8, int16_t)
VERTICAL(vert4_16, 4, int16_t)

/* ------------------------------------------------ and which one to use */

/* ⚠️ The vector paths are eight bit only. They pack to unsigned bytes,
 * which at ten bits would saturate everything above 255 and hand back a
 * picture that looks decoded. So the depth decides, not only the
 * processor - and BC250_HEVC_NOSIMD forces the scalar path, so the suites
 * can be run through the code that carries every other depth. */
static bool use_vectors(int bd)
{
#if defined(__x86_64__) || defined(_M_X64)
    static int allowed = -1;
    if (allowed < 0) allowed = getenv("BC250_HEVC_NOSIMD") ? 0 : 1;
    return allowed && ha_ssse3() && bd == 8;
#else
    (void)bd;
    return false;
#endif
}

#define BOTH_WAYS(nome, TYPE)                                             \
static void nome##_any(const TYPE *s, int sp, int w, int h,                \
                       const int8_t *f, int16_t *o, int po,                \
                       int down, bool vec)                                 \
{                                                                          \
    IF_VECTOR(nome)                                                       \
    nome(s, sp, w, h, f, o, po, down);                                     \
}

#if defined(__x86_64__) || defined(_M_X64)
#define IF_VECTOR(nome) if (vec) { nome##_v(s, sp, w, h, f, o, po); return; }
#else
#define IF_VECTOR(nome) (void)vec;
#endif

BOTH_WAYS(horiz8, uint8_t)
BOTH_WAYS(horiz4, uint8_t)
BOTH_WAYS(vert8, uint8_t)
BOTH_WAYS(vert4, uint8_t)
BOTH_WAYS(vert8_16, int16_t)
BOTH_WAYS(vert4_16, int16_t)

/* One rectangle of one plane, at a fractional position, into fourteen-bit
 * intermediate values.
 *
 * `before` is how far back the filter reaches: three samples for the eight
 * taps of luma, one for the four of chroma. */
static void interpolate(const uint8_t *ref_pic, int stride, int w_pic, int h_pic,
                      int x, int y, int w, int h, int fx, int fy,
                      const int8_t *filter_kind, int count, int before,
                      int16_t *out, int out_stride, int bd)
{
    const int8_t *fh = filter_kind + (size_t)fx * count;
    const int8_t *fv = filter_kind + (size_t)fy * count;

    /* 8.5.3.3.3.2: shift1 after the horizontal pass, shift2 after the
     * vertical one. ⚠️ Only the first moves with the depth; the six is
     * six at every depth the standard defines. */
    const int down = bd - 8;
    const bool vec = use_vectors(bd);

    /* The rectangle of whole samples the passes will read: as far back as
     * the filter reaches, and only in the directions it actually filters. */
    const int px = fx ? before : 0, tx = fx ? count : 1;
    const int py = fy ? before : 0, ty = fy ? count : 1;
    const int bx = x - px, by = y - py;
    const int bw = w + tx - 1, bh = h + ty - 1;

    /* ⚠️ A motion vector may point off the edge of the reference picture,
     * and legitimately: an object entering the frame was not there before.
     * The edge sample is repeated outwards rather than the fetch being
     * refused - but deciding that once per tap, eight times per sample,
     * is what made this the slowest thing in the decoder. Decide it once
     * per block instead: either the whole window is inside the picture and
     * the filter reads it where it lies, or the window is copied out once
     * with its edges repeated and the filter reads the copy. */
    const uint8_t *src;
    int sp;
    uint8_t border[(MAX_SIDE + 7) * (MAX_SIDE + 7)];
    if (bx >= 0 && by >= 0 && bx + bw <= w_pic && by + bh <= h_pic) {
        src = ref_pic + (size_t)by * stride + bx;
        sp = stride;
    } else {
        /* The window hangs over an edge. Clamping every sample by itself
         * is how it was written first and it costs more than the filter
         * that reads the result: the row is the same for a whole span of
         * columns, so each output row is a repeat of one sample, a copy
         * of the middle, and a repeat of the last. */
        sp = MAX_SIDE + 7;
        const int left = bx < 0 ? (-bx > bw ? bw : -bx) : 0;
        const int inside_end = bx + bw > w_pic ? w_pic - bx : bw;
        const int right = inside_end < left ? left : inside_end;
        for (int r = 0; r < bh; r++) {
            int sy = by + r;
            sy = sy < 0 ? 0 : (sy >= h_pic ? h_pic - 1 : sy);
            const uint8_t *ref_row = ref_pic + (size_t)sy * stride;
            uint8_t *o = border + (size_t)r * sp;
            if (left > 0) memset(o, ref_row[0], (size_t)left);
            if (right > left)
                memcpy(o + left, ref_row + bx + left,
                       (size_t)(right - left));
            if (bw > right)
                memset(o + right, ref_row[w_pic - 1], (size_t)(bw - right));
        }
        src = border;
    }

    if (!fx && !fy) {
#if defined(__x86_64__) || defined(_M_X64)
        if (vec) { copy14_v(src, sp, w, h, out, out_stride); return; }
#endif
        /* shift3 = 14 - BitDepth: the samples go up to fourteen bits, they
         * do not come down. */
        const int up = 14 - bd;
        for (int r = 0; r < h; r++)
            for (int c = 0; c < w; c++)
                out[r * out_stride + c] =
                    (int16_t)(src[(size_t)r * sp + c] << up);
        return;
    }

    if (!fy) {
        if (count == 8) horiz8_any(src, sp, w, h, fh, out, out_stride, down, vec);
        else            horiz4_any(src, sp, w, h, fh, out, out_stride, down, vec);
        return;
    }

    if (!fx) {
        if (count == 8) vert8_any(src, sp, w, h, fv, out, out_stride, down, vec);
        else            vert4_any(src, sp, w, h, fv, out, out_stride, down, vec);
        return;
    }

    /* Both: horizontally first, over enough extra rows above and below for
     * the vertical pass to have something to stand on. */
    int16_t middle[(MAX_SIDE + 7) * MAX_SIDE];
    const int tall = h + count - 1;
    if (count == 8) {
        horiz8_any(src, sp, w, tall, fh, middle, MAX_SIDE, down, vec);
        vert8_16_any(middle, MAX_SIDE, w, h, fv, out, out_stride, 6, vec);
    } else {
        horiz4_any(src, sp, w, tall, fh, middle, MAX_SIDE, down, vec);
        vert4_16_any(middle, MAX_SIDE, w, h, fv, out, out_stride, 6, vec);
    }
}

/* ------------------------------------- fourteen bits back down to eight */

static void one_pred(uint8_t *dst, int stride, int w, int h,
                const int16_t *a, int stride_a, int bd)
{
#if defined(__x86_64__) || defined(_M_X64)
    if (use_vectors(bd)) { one_pred_v(dst, stride, w, h, a, stride_a); return; }
#endif
    /* 8.5.3.3.4.2: shift1 = 14 - BitDepth, offset1 = 1 << (shift1 - 1) */
    const int sh = 14 - bd, add = 1 << (13 - bd);
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++)
            dst[r * stride + c] = (uint8_t)
                clip_pixel((a[r * stride_a + c] + add) >> sh, bd);
}

static void two_pred(uint8_t *dst, int stride, int w, int h,
                const int16_t *a, const int16_t *b, int stride_p, int bd)
{
#if defined(__x86_64__) || defined(_M_X64)
    if (use_vectors(bd)) {
        two_pred_v(dst, stride, w, h, a, b, stride_p);
        return;
    }
#endif
    const int sh = 15 - bd, add = 1 << (14 - bd);
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++)
            dst[r * stride + c] = (uint8_t)clip_pixel(
                (a[r * stride_p + c] + b[r * stride_p + c] + add) >> sh, bd);
}

/* 8.5.3.3.4.3. ⚠️ Used whenever the slice carries a weight table, even
 * where the weight happens to be neutral: for a neutral weight this is
 * the same arithmetic as the plain path, so there is nothing to gain by
 * deciding per block and something to lose by getting the decision
 * wrong. */
static void one_weighted(uint8_t *dst, int stride, int w, int h,
                       const int16_t *a, int stride_a,
                       int weight, int off, int den, int bd)
{
    const int log2wd = den + 14 - bd;
    const int o = off * (1 << (bd - 8));
#if defined(__x86_64__) || defined(_M_X64)
    /* log2wd is den + 6 at eight bits and den is never negative, so the
     * other branch is unreachable on any stream the parser accepts. It
     * stays anyway. */
    if (use_vectors(bd) && log2wd >= 1) {
        one_weighted_v(dst, stride, w, h, a, stride_a, weight, off, den);
        return;
    }
#endif
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++) {
            const int v = a[r * stride_a + c];
            dst[r * stride + c] = (uint8_t)clip_pixel(
                log2wd >= 1 ? (((v * weight + (1 << (log2wd - 1))) >> log2wd) + o)
                            : (v * weight + o), bd);
        }
}

static void two_weighted(uint8_t *dst, int stride, int w, int h,
                       const int16_t *a, const int16_t *b, int stride_p,
                       int pa, int pb, int oa, int ob, int den, int bd)
{
    const int log2wd = den + 14 - bd;
    const int sa = oa * (1 << (bd - 8)), sb = ob * (1 << (bd - 8));
#if defined(__x86_64__) || defined(_M_X64)
    if (use_vectors(bd)) {
        two_weighted_v(dst, stride, w, h, a, b, stride_p, pa, pb, oa, ob, den);
        return;
    }
#endif
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++)
            dst[r * stride + c] = (uint8_t)clip_pixel(
                (a[r * stride_p + c] * pa + b[r * stride_p + c] * pb
                 + ((sa + sb + 1) << log2wd)) >> (log2wd + 1), bd);
}

/* Does this slice carry a weight table at all. */
static bool has_weights(const hevcd_t *d)
{
    return (d->pps->weighted_pred && d->slice->type == 1)
        || (d->pps->weighted_bipred && d->slice->type == 0);
}

void hevcd_predict_inter(hevcd_t *d, int x0, int y0, int w, int h,
                         const hevcd_mvf_t *m)
{
    const hevc_sps_t *sps = d->sps;
    /* ⚠️ On the stack, not static. Sixteen kilobytes is nothing and one
     * shared buffer would be one prediction handed to whichever row asked
     * last. */
    int16_t p[2][MAX_SIDE * MAX_SIDE];
    const bool usa[2] = { (m->pred_flag & HEVCD_PF_L0) != 0,
                          (m->pred_flag & HEVCD_PF_L1) != 0 };
    const bool weights = has_weights(d);

    for (int plane = 0; plane < 3; plane++) {
        /* ⚠️ Luma and chroma carry their own depth. Equal in every profile
         * we accept, and reading the wrong one would be invisible until
         * the day they are not. */
        const int bd = plane ? sps->bit_depth_chroma : sps->bit_depth_luma;
        const int giu = plane ? 1 : 0;
        const int pw = w >> giu, ph = h >> giu;
        const int px = x0 >> giu, py = y0 >> giu;
        const int w_pic = sps->width >> giu, h_pic = sps->height >> giu;

        for (int l = 0; l < 2; l++) {
            if (!usa[l]) continue;
            const int i = m->ref_idx[l];
            if (i < 0 || i >= d->n_refs[l] || !d->ref_pic[l][i]) return;
            const hevcd_img_t *r = d->ref_pic[l][i];
            const int mvx = m->mv[l][0], mvy = m->mv[l][1];
            /* ⚠️ Luma counts quarters and chroma eighths. At 4:2:0 the
             * chroma plane is half the size, so the same vector lands on a
             * finer grid there, not a coarser one. */
            const int steps = plane ? 3 : 2;
            interpolate(r->plane[plane], r->stride[plane], w_pic, h_pic,
                      px + (mvx >> steps), py + (mvy >> steps), pw, ph,
                      mvx & ((1 << steps) - 1), mvy & ((1 << steps) - 1),
                      plane ? &hevcd_epel[0][0] : &hevcd_qpel[0][0],
                      plane ? 4 : 8, plane ? 1 : 3,
                      p[l], MAX_SIDE, bd);
        }

        uint8_t *dst = d->plane[plane] + (size_t)py * d->stride[plane] + px;

        if (!weights) {
            if (usa[0] && usa[1])
                two_pred(dst, d->stride[plane], pw, ph, p[0], p[1], MAX_SIDE, bd);
            else
                one_pred(dst, d->stride[plane], pw, ph, p[usa[0] ? 0 : 1],
                         MAX_SIDE, bd);
            continue;
        }

        const int den = plane ? d->slice->chroma_log2_weight_denom
                              : d->slice->luma_log2_weight_denom;
        int weight[2] = { 1 << den, 1 << den }, off[2] = { 0, 0 };
        for (int l = 0; l < 2; l++) {
            if (!usa[l]) continue;
            const int i = m->ref_idx[l];
            weight[l] = plane ? d->slice->chroma_weight[l][i][plane - 1]
                              : d->slice->luma_weight[l][i];
            off[l] = plane ? d->slice->chroma_offset[l][i][plane - 1]
                           : d->slice->luma_offset[l][i];
        }

        if (usa[0] && usa[1])
            two_weighted(dst, d->stride[plane], pw, ph, p[0], p[1], MAX_SIDE,
                       weight[0], weight[1], off[0], off[1], den, bd);
        else {
            const int l = usa[0] ? 0 : 1;
            one_weighted(dst, d->stride[plane], pw, ph, p[l], MAX_SIDE,
                       weight[l], off[l], den, bd);
        }
    }
}
