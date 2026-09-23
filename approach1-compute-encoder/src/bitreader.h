/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * bitreader.h - RBSP bit reader for the decoders.
 *
 * The write side lives in bitstream.h; this is the read side, and the two
 * deliberately do not share a struct. A reader has to cope with running off
 * the end of a corrupt stream on every single call, which a writer never
 * does, so mixing them would put a bounds check in the encoder's hot loop
 * for nothing.
 *
 * Everything here works on RBSP, i.e. emulation prevention bytes already
 * removed by br_extract_rbsp(). CABAC in particular cannot be fed EBSP: it
 * reads the bitstream two bytes at a time with no syntax to tell it where a
 * 0x03 would have been, so an un-stripped 0x000003 silently shifts every
 * subsequent bin.
 */
#ifndef BC250_BITREADER_H
#define BC250_BITREADER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    const uint8_t *data;
    size_t size;      /* bytes in data */
    size_t bitpos;    /* next bit to be read, counted from data[0] bit 7 */
} br_t;

static inline void br_init(br_t *br, const uint8_t *data, size_t size)
{
    br->data = data;
    br->size = size;
    br->bitpos = 0;
}

static inline size_t br_bits_left(const br_t *br)
{
    size_t total = br->size * 8;
    return br->bitpos >= total ? 0 : total - br->bitpos;
}

/* Reading past the end returns zero bits rather than failing. A decoder that
 * has run off the end is already producing nonsense for this slice; the
 * caller checks br_overrun() at slice granularity and drops the slice, which
 * is both simpler and safer than threading an error return through every
 * syntax element. */
static inline bool br_overrun(const br_t *br)
{
    return br->bitpos > br->size * 8;
}

/* Up to 24 bits, which is all any H.264 or HEVC syntax element needs in one
 * go (the longest are the 32-bit-capable Exp-Golomb prefixes, and those are
 * assembled from two reads). */
static inline uint32_t br_peek(const br_t *br, int n)
{
    if (n <= 0) return 0;
    size_t byte = br->bitpos >> 3;
    int off = (int)(br->bitpos & 7);
    uint32_t v = 0;
    /* Four bytes always cover 24 bits at any bit offset. */
    for (int i = 0; i < 4; i++) {
        v <<= 8;
        if (byte + (size_t)i < br->size) v |= br->data[byte + i];
    }
    v <<= off;                  /* drop the bits already consumed */
    v >>= (32 - n);
    return v & (n >= 32 ? 0xFFFFFFFFu : ((1u << n) - 1u));
}

static inline void br_skip(br_t *br, int n)
{
    br->bitpos += (size_t)n;
}

static inline uint32_t br_read(br_t *br, int n)
{
    uint32_t v = br_peek(br, n);
    br->bitpos += (size_t)n;
    return v;
}

static inline uint32_t br_read1(br_t *br)
{
    size_t byte = br->bitpos >> 3;
    int off = (int)(br->bitpos & 7);
    br->bitpos++;
    if (byte >= br->size) return 0;
    return (br->data[byte] >> (7 - off)) & 1u;
}

/* ue(v), Rec. ITU-T H.264 9.1. The leading-zero count is bounded at 32 so a
 * run of zero bytes in a damaged stream cannot spin here. */
static inline uint32_t br_read_ue(br_t *br)
{
    int lz = 0;
    while (lz < 32 && br_read1(br) == 0) {
        if (br->bitpos > br->size * 8) return 0;
        lz++;
    }
    if (lz == 0) return 0;
    if (lz >= 32) return 0;
    return (1u << lz) - 1u + br_read(br, lz);
}

/* se(v), Rec. ITU-T H.264 9.1.1 */
static inline int32_t br_read_se(br_t *br)
{
    uint32_t k = br_read_ue(br);
    int32_t v = (int32_t)((k + 1u) >> 1);
    return (k & 1u) ? v : -v;
}

/* Rec. ITU-T H.264 7.2 more_rbsp_data(): true while anything other than the
 * rbsp_stop_one_bit and its zero padding remains. */
static inline bool br_more_rbsp_data(const br_t *br)
{
    size_t total = br->size * 8;
    if (br->bitpos >= total) return false;
    /* Find the last set bit in the buffer - that is the stop bit. */
    size_t last = total;
    while (last > br->bitpos) {
        size_t b = last - 1;
        if ((br->data[b >> 3] >> (7 - (b & 7))) & 1u) {
            last = b;
            break;
        }
        last = b;
    }
    return br->bitpos < last;
}

/* Strip emulation prevention bytes: 0x00 0x00 0x03 -> 0x00 0x00, where the
 * 0x03 is only removed when it is followed by 0x00..0x03 (H.264 7.4.1.1).
 * Returns the RBSP length written into dst, which is never larger than src.
 *
 * ⚠️ The check on the byte AFTER the 0x03 matters: a 0x03 that is not an
 * emulation prevention byte is ordinary payload, and dropping it corrupts
 * the slice. Several toy decoders skip every 0x000003 and mostly get away
 * with it, because the pattern is rare in real coefficients.
 */
static inline size_t br_extract_rbsp(uint8_t *dst, size_t dst_cap,
                                     const uint8_t *src, size_t src_len)
{
    size_t o = 0;
    size_t zeros = 0;
    for (size_t i = 0; i < src_len && o < dst_cap; i++) {
        uint8_t c = src[i];
        if (zeros >= 2 && c == 0x03) {
            if (i + 1 < src_len && src[i + 1] > 0x03) {
                /* not an emulation prevention byte after all */
                dst[o++] = c;
                zeros = 0;
                continue;
            }
            zeros = 0;
            continue;   /* drop it */
        }
        dst[o++] = c;
        zeros = (c == 0x00) ? zeros + 1 : 0;
    }
    return o;
}

/* As br_extract_rbsp, and it also moves a bit offset counted in `src` into
 * the coordinates of `dst`.
 *
 * ⚠️ An offset can only be moved if the bytes removed before it are
 * counted while they are removed. Doing it afterwards means scanning the
 * source twice and getting the straddling cases wrong; doing it here costs
 * one comparison per byte. */
static inline size_t br_extract_rbsp_map(uint8_t *dst, size_t dst_cap,
                                         const uint8_t *src, size_t src_len,
                                         size_t *bit_offset)
{
    const size_t limit = bit_offset ? (*bit_offset >> 3) : src_len;
    size_t removed = 0;
    size_t o = 0;
    size_t zeros = 0;
    for (size_t i = 0; i < src_len && o < dst_cap; i++) {
        uint8_t c = src[i];
        if (zeros >= 2 && c == 0x03) {
            if (i + 1 < src_len && src[i + 1] > 0x03) {
                dst[o++] = c;
                zeros = 0;
                continue;
            }
            zeros = 0;
            if (i < limit) removed++;
            continue;
        }
        dst[o++] = c;
        zeros = (c == 0x00) ? zeros + 1 : 0;
    }
    if (bit_offset) *bit_offset -= removed * 8;
    return o;
}

/* The inverse of br_extract_rbsp_map's offset move: a bit position in the
 * un-escaped buffer, back into the raw one it came from.
 *
 * Only a caller that un-escaped a NAL for its own use needs this, to say
 * where it got to in terms the raw buffer understands. */
static inline size_t br_raw_offset(const uint8_t *src, size_t src_len,
                                   size_t rbsp_bit)
{
    const size_t target = rbsp_bit >> 3;
    size_t o = 0;
    size_t zeros = 0;
    for (size_t i = 0; i < src_len; i++) {
        if (o == target)
            return (i << 3) | (rbsp_bit & 7);
        uint8_t c = src[i];
        if (zeros >= 2 && c == 0x03) {
            if (i + 1 < src_len && src[i + 1] > 0x03) {
                o++;
                zeros = 0;
                continue;
            }
            zeros = 0;
            continue;
        }
        o++;
        zeros = (c == 0x00) ? zeros + 1 : 0;
    }
    return rbsp_bit;
}

#endif /* BC250_BITREADER_H */
