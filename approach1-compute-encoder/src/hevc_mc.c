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

#include <stdatomic.h>
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

/* ⚠️ Atomic: the first inter block of a stream is usually decoded by
 * several wavefront rows at once, and each of them asks. */
static int ha_ssse3(void)
{
    static _Atomic int answer = -1;
    int a = atomic_load_explicit(&answer, memory_order_relaxed);
    if (a < 0) {
        a = __builtin_cpu_supports("ssse3") ? 1 : 0;
        atomic_store_explicit(&answer, a, memory_order_relaxed);
    }
    return a;
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

/* Two whole-sample predictions averaged. _mm_avg_epu8 is (a + b + 1) >> 1,
 * which is exactly what 8.5.3.3.4.2 gives two samples taken up to fourteen
 * bits, added, and brought back down with its rounding. */
static void average_v(uint8_t *dst, int stride, const uint8_t *a, int sa,
                      const uint8_t *b, int sb, int w, int h)
{
    for (int r = 0; r < h; r++) {
        const uint8_t *pa = a + (size_t)r * sa;
        const uint8_t *pb = b + (size_t)r * sb;
        uint8_t *o = dst + (size_t)r * stride;
        int c = 0;
        for (; c + 16 <= w; c += 16)
            _mm_storeu_si128((__m128i *)(o + c),
                _mm_avg_epu8(_mm_loadu_si128((const __m128i *)(pa + c)),
                             _mm_loadu_si128((const __m128i *)(pb + c))));
        for (; c + 8 <= w; c += 8)
            _mm_storel_epi64((__m128i *)(o + c),
                _mm_avg_epu8(_mm_loadl_epi64((const __m128i *)(pa + c)),
                             _mm_loadl_epi64((const __m128i *)(pb + c))));
        for (; c < w; c++) o[c] = (uint8_t)((pa[c] + pb[c] + 1) >> 1);
    }
}

#endif /* x86-64 */

/* ⚠️ The vector paths are eight bit only. They pack to unsigned bytes,
 * which at ten bits would saturate everything above 255 and hand back a
 * picture that looks decoded. So the depth decides, not only the
 * processor - and BC250_HEVC_NOSIMD forces the scalar path, so the suites
 * can be run through the code that carries every other depth. */
static bool use_vectors(int bd)
{
#if defined(__x86_64__) || defined(_M_X64)
    static _Atomic int allowed = -1;
    int a = atomic_load_explicit(&allowed, memory_order_relaxed);
    if (a < 0) {
        a = getenv("BC250_HEVC_NOSIMD") ? 0 : 1;
        atomic_store_explicit(&allowed, a, memory_order_relaxed);
    }
    return a && ha_ssse3() && bd == 8;
#else
    (void)bd;
    return false;
#endif
}


#define BIT_DEPTH 8
#include "hevc_pixel.h"
#include "hevc_mc_template.c"
#undef BIT_DEPTH

#define BIT_DEPTH 10
#include "hevc_pixel.h"
#include "hevc_mc_template.c"
#undef BIT_DEPTH

void hevcd_predict_inter(hevcd_t *d, int x0, int y0, int w, int h,
                         const hevcd_mvf_t *m)
{
    if (d->sps->bit_depth_luma > 8) predict_inter_10(d, x0, y0, w, h, m);
    else                            predict_inter_8(d, x0, y0, w, h, m);
}
