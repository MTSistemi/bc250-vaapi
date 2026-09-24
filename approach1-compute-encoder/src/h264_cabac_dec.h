/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * h264_cabac_dec.h - the CABAC arithmetic decoding engine, Rec. ITU-T H.264
 * clause 9.3.3.2.
 *
 * The encoder's engine lives in cabac.c and the two share the normative
 * tables (rangeTabLPS and the packed transition table) but nothing else: an
 * arithmetic coder and its decoder are not symmetric enough for shared code
 * to be worth the indirection in either hot loop.
 *
 * State packing follows the encoder's, which follows x264's: one byte per
 * context holding (pStateIdx << 1) | valMPS, where pStateIdx is the
 * standard's index counted backwards (x264 index = 63 - spec index), so that
 * the most skewed state sits at 0. cabac_context_init() in cabac.c produces
 * exactly this packing.
 *
 * ⚠️ Renormalisation here uses a leading-zero count, not the encoder's
 * cabac_renorm_shift[] lookup. That table is indexed by range>>3 and its
 * entry 0 covers ranges 2..7 with a single shift of 6, which is only correct
 * from 4 upwards - range 2 would renormalise to 128 and the next decision's
 * `(range >> 6) - 4` would index the LPS table at -2. It is dead code in
 * practice, because x264 state 0 (the only one with rangeLPS = 2) is
 * unreachable: the transition table's rows 0 and 1 are absorbing, no other
 * row points into them, and context initialisation produces pStateIdx in
 * 1..63. A decoder eats attacker-shaped bytes, though, so it does not get to
 * rely on that; clz is exact for every range and costs one instruction.
 *
 * ⚠️ How the offset is held. The standard's codIOffset is nine bits, and
 * every renormalisation shifts the next bits of the stream into it. Here it
 * sits at the top of a 64-bit `value`, with the stream bits already read
 * ahead below it:
 *
 *     value = codIOffset << cache_bits | the next cache_bits bits
 *
 * so a renormalisation by sh is `cache_bits -= sh` and nothing else - the
 * next bits are already in place - and every comparison with the offset is
 * a comparison with the range shifted up by cache_bits, which is exact
 * because the bits below are smaller than one step of it. The engine this
 * replaced kept the offset in nine bits and pulled every bit into it one
 * renormalisation at a time; that, and a branch per bin, was most of what
 * reading a coefficient cost.
 */
#ifndef BC250_H264_CABAC_DEC_H
#define BC250_H264_CABAC_DEC_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "cabac.h"            /* cabac_range_lps, cabac_transition */
#include <string.h>

#include "h264_dec_tables.h"  /* h264d_cabac_init_I, h264d_cabac_init_PB */

#define H264D_CABAC_CTX 1024

typedef struct {
    uint32_t range;          /* codIRange, 256..510 between bins */
    /* codIOffset << cache_bits, with the next cache_bits bits of the
     * stream below it. codIOffset is below range, so value stays below
     * 510 << cache_bits and cache_bits may reach 55. */
    uint64_t value;
    int      cache_bits;
    const uint8_t *start;
    /* The next byte to read. ⚠️ It keeps advancing past `end` while zero
     * bytes are fed in its place (see refill), so that it always counts
     * every bit read - which is what overrun() and byte_pos() ask. */
    const uint8_t *ptr;
    const uint8_t *end;

    uint8_t  state[H264D_CABAC_CTX];
} h264d_cabac_t;

/* Tops the read-ahead up to between 48 and 55 bits. Past the end we feed
 * zero bytes rather than stopping. A slice that reads beyond its own data
 * is corrupt and its macroblocks are discarded, but the decode loop has to
 * be able to keep running until the slice ends rather than unwinding out of
 * the middle of a macroblock.
 *
 * ⚠️ The pointer advances over those zeros too, without ever reading them
 * from memory. The engine this replaced left it at the end, so the zeros
 * counted as bits not yet read, and overrun() could not see a slice that
 * had run past its data - the one thing it is for. */
static inline void h264d_cabac_refill(h264d_cabac_t *c)
{
    if (c->end - c->ptr >= 8) {
        /* Most of the time: one unaligned load, as many whole bytes of it
         * as fit. cache_bits is at most 47 on the way in, so that is at
         * least one byte. */
        uint64_t w;
        memcpy(&w, c->ptr, 8);
        w = __builtin_bswap64(w);
        const int n = (55 - c->cache_bits) >> 3;
        c->value = (c->value << (8 * n)) | (w >> (64 - 8 * n));
        c->ptr += n;
        c->cache_bits += 8 * n;
        return;
    }
    while (c->cache_bits <= 47) {
        const uint8_t b = (c->ptr < c->end) ? *c->ptr : 0;
        c->ptr++;
        c->value = (c->value << 8) | (uint64_t)b;
        c->cache_bits += 8;
    }
}

/* Whether the slice has actually been decoded past its own end.
 *
 * ⚠️ Not the same question as "has a byte past the end been touched". The
 * engine keeps up to seven bytes in its cache, so it reads ahead as a
 * matter of course and reaches the last byte of a perfectly good slice long
 * before it has consumed it. What counts is how many bits have been handed
 * out, which is what this computes. Checking the pointer instead flags
 * every healthy slice.
 */
static inline bool h264d_cabac_overrun(const h264d_cabac_t *c)
{
    ptrdiff_t n_read = (c->ptr - c->start) * 8 - c->cache_bits;
    return n_read > (ptrdiff_t)(c->end - c->start) * 8;
}

/* Where the next substream begins: how many bytes this one has actually
 * consumed, rounded up to the byte alignment that follows it.
 *
 * ⚠️ Not where the read-ahead pointer is. The engine keeps a cache and
 * reaches ahead as a matter of course; what is wanted is how many bits
 * have been handed out, which is a different number. */
static inline size_t h264d_cabac_byte_pos(const h264d_cabac_t *c)
{
    const ptrdiff_t bit = (c->ptr - c->start) * 8 - c->cache_bits;
    /* ⚠️ Negative until the engine has handed out as many bits as it read
     * ahead when it filled its cache, which on a substream shorter than
     * the cache is most of the substream. Rounding that up and returning
     * it as a size_t gives a number near the top of the range, and every
     * comparison against it then means the opposite of what it says. */
    return bit <= 0 ? 0 : (size_t)((bit + 7) / 8);
}

/* Clause 9.3.1.1, context variable initialisation.
 *
 * The encoder's cabac_context_init() does the same arithmetic but only over
 * the 276 contexts it can write and only with cabac_init_idc 0, because an
 * encoder picks its own idc. A decoder is told which one to use and has to
 * have all 1024 contexts of all four tables.
 */
static inline void h264d_cabac_ctx_init(uint8_t *state, bool intra_slice,
                                        int cabac_init_idc, int qp)
{
    if (qp < 0) qp = 0;
    if (qp > 51) qp = 51;
    if (cabac_init_idc < 0) cabac_init_idc = 0;
    if (cabac_init_idc > 2) cabac_init_idc = 2;

    const int8_t (*t)[2] = intra_slice ? h264d_cabac_init_I
                                       : h264d_cabac_init_PB[cabac_init_idc];
    for (int j = 0; j < H264D_CABAC_CTX; j++) {
        int s = ((t[j][0] * qp) >> 4) + t[j][1];
        s = s < 1 ? 1 : (s > 126 ? 126 : s);
        int p = s < 127 - s ? s : 127 - s;   /* x264's index: 63 - spec's */
        state[j] = (uint8_t)((p << 1) | (s >= 64 ? 1 : 0));
    }
}

/* Clause 9.3.1.2: range starts at 510 and offset is the next nine bits.
 * The caller has already skipped the alignment bits, so `data` starts on a
 * byte boundary. Shared with H.265, which specifies the same machine - see
 * hevc_cabac_dec.h. */
static inline void h264d_cabac_init_engine(h264d_cabac_t *c,
                                           const uint8_t *data, size_t size)
{
    c->start = data;
    c->ptr = data;
    c->end = data + size;
    c->value = 0;
    c->cache_bits = 0;
    c->range = 510;
    h264d_cabac_refill(c);          /* at least 48 bits */
    c->cache_bits -= 9;             /* the top nine are codIOffset */
}

static inline void h264d_cabac_init(h264d_cabac_t *c,
                                    const uint8_t *data, size_t size,
                                    bool intra_slice, int cabac_init_idc, int qp)
{
    h264d_cabac_init_engine(c, data, size);
    h264d_cabac_ctx_init(c->state, intra_slice, cabac_init_idc, qp);
}

/* Brings range back to 256..510. The offset takes the same number of new
 * bits, and they are already sitting below it. */
static inline void h264d_cabac_renorm(h264d_cabac_t *c)
{
    /* range is at least 2 here, so clz is defined. 31 - clz(range) is the
     * index of its top bit; the shift brings that bit to position 8. */
    const int sh = (int)__builtin_clz(c->range) - 23;
    c->range <<= sh;
    c->cache_bits -= sh;
    if (c->cache_bits < 16) h264d_cabac_refill(c);
}

/* Clause 9.3.3.2.1, DecodeDecision. Without a branch on the outcome: which
 * way a bin goes is exactly what the processor cannot guess, and a
 * mispredicted branch per bin cost more than the arithmetic. */
static inline int h264d_cabac_decision(h264d_cabac_t *c, int ctx)
{
    const unsigned s = c->state[ctx];
    /* range is 256..510, so (range >> 6) - 4 is (range >> 6) & 3. */
    const uint32_t rlps = cabac_range_lps[s >> 1][(c->range >> 6) & 3];
    const uint32_t rmps = c->range - rlps;
    const uint64_t scaled = (uint64_t)rmps << c->cache_bits;
    /* All ones when the offset lands in the least probable interval. */
    const uint64_t lps = (uint64_t)0 - (uint64_t)(c->value >= scaled);
    c->value -= scaled & lps;
    c->range = rmps ^ ((rmps ^ rlps) & (uint32_t)lps);
    const int bin = (int)((s ^ (unsigned)lps) & 1u);
    c->state[ctx] = cabac_transition[s][bin];
    h264d_cabac_renorm(c);
    return bin;
}

/* Clause 9.3.3.2.3, DecodeBypass. No context, no renormalisation of range:
 * the offset takes one more bit, which is already below it. */
static inline int h264d_cabac_bypass(h264d_cabac_t *c)
{
    c->cache_bits--;
    const uint64_t scaled = (uint64_t)c->range << c->cache_bits;
    const uint64_t one = (uint64_t)0 - (uint64_t)(c->value >= scaled);
    c->value -= scaled & one;
    if (c->cache_bits < 16) h264d_cabac_refill(c);
    return (int)(one & 1u);
}

/* Several bypass bins at once, most significant first, which is what the
 * sign and suffix bits of a coefficient level need.
 *
 * Bypass decoding is long division: each bin doubles the offset, takes the
 * next bit in, and subtracts the range if it fits. n of them in a row are
 * therefore the quotient of the offset with the next n bits appended,
 * divided by the range - and the remainder is the offset afterwards. One
 * division instead of n bins, each of which a processor has no way to
 * predict, because bypass bins are by construction incompressible. */
static inline uint32_t h264d_cabac_bypass_n(h264d_cabac_t *c, int n)
{
    if (n <= 0) return 0;
    if (n < 4) {
        uint32_t v = 0;
        while (n-- > 0) v = (v << 1) | (uint32_t)h264d_cabac_bypass(c);
        return v;
    }
    uint32_t v = 0;
    if (n > 32) {                   /* no caller asks for this many */
        v = h264d_cabac_bypass_n(c, n - 32);
        n = 32;
    }
    if (c->cache_bits < n) h264d_cabac_refill(c);
    c->cache_bits -= n;
    /* The offset with n more bits: below range << n, so under 2^41. */
    const uint64_t wide = c->value >> c->cache_bits;
    uint64_t q;
    if (n <= 23) q = (uint32_t)wide / c->range;
    else         q = wide / c->range;
    c->value -= (q * c->range) << c->cache_bits;
    if (c->cache_bits < 16) h264d_cabac_refill(c);
    return (v << n) | (uint32_t)q;
}

/* Clause 9.3.3.2.4, DecodeTerminate. Returns 1 when the slice ends here, in
 * which case the engine must not be used again. */
static inline int h264d_cabac_terminate(h264d_cabac_t *c)
{
    c->range -= 2;
    if (c->value >= ((uint64_t)c->range << c->cache_bits))
        return 1;
    h264d_cabac_renorm(c);
    return 0;
}

/* Unary with a cap, the shape most of the mb-layer syntax elements take. */
static inline int h264d_cabac_unary(h264d_cabac_t *c, const int *ctx, int n_ctx, int cap)
{
    int v = 0;
    while (v < cap && h264d_cabac_decision(c, ctx[v < n_ctx ? v : n_ctx - 1]))
        v++;
    return v;
}

/* Exp-Golomb suffix in bypass mode, clause 9.3.2.3. Used by the coefficient
 * levels and the motion vector differences once their unary prefix has run
 * out. */
static inline uint32_t h264d_cabac_eg_bypass(h264d_cabac_t *c, int k)
{
    uint32_t v = 0;
    while (h264d_cabac_bypass(c)) {
        v += 1u << k;
        k++;
        if (k > 30) break;          /* a corrupt stream must not spin here */
    }
    return v + h264d_cabac_bypass_n(c, k);
}

#endif /* BC250_H264_CABAC_DEC_H */
