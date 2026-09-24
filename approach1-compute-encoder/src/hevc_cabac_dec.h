/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_cabac_dec.h - CABAC decoding for H.265.
 *
 * ⚠️ The arithmetic decoder itself is the H.264 one, unchanged, because
 * H.265 specifies the same one. Clause 9.3.4.3 of H.265 and clause 9.3.3.2
 * of H.264 describe the same machine: the same 9-bit offset read after
 * alignment, the same 510 initial range, the same rangeTabLps and the same
 * state transitions. What H.265 changed is everything around it - which
 * contexts exist, how they are initialised, and what the bins mean.
 *
 * So this file is the part that differs, and nothing else. Sharing the
 * engine is not a shortcut: writing it twice would mean two places for the
 * same renormalisation bug to hide.
 */
#ifndef BC250_HEVC_CABAC_DEC_H
#define BC250_HEVC_CABAC_DEC_H

#include "h264_cabac_dec.h"
#include "hevc_dec_tables.h"

typedef h264d_cabac_t hevcd_cabac_t;

/* Clause 9.3.2.2. initType is 0 for an I slice; for P and B it is 1 or 2
 * and cabac_init_flag swaps them, which is what lets a P slice start from
 * a B slice's statistics and the other way round. */
static inline int hevcd_init_type(int slice_type, bool cabac_init_flag)
{
    /* slice_type: 0 B, 1 P, 2 I - the H.265 numbering, which is not the
     * H.264 one. */
    if (slice_type == 2) return 0;
    if (slice_type == 1) return cabac_init_flag ? 2 : 1;
    return cabac_init_flag ? 1 : 2;
}

static inline void hevcd_cabac_ctx_init(uint8_t *state, int init_type, int qp)
{
    if (qp < 0) qp = 0;
    if (qp > 51) qp = 51;
    if (init_type < 0) init_type = 0;
    if (init_type > 2) init_type = 2;
    /* The states are pre-computed for every initType and every QP, so a
     * slice header costs a memcpy rather than 179 multiplications. */
    memcpy(state, hevcd_ctx_init[init_type * 52 + qp], HEVCD_CTX);
}

static inline void hevcd_cabac_init(hevcd_cabac_t *c,
                                    const uint8_t *data, size_t size,
                                    int slice_type, bool cabac_init_flag, int qp)
{
    h264d_cabac_init_engine(c, data, size);
    hevcd_cabac_ctx_init(c->state, hevcd_init_type(slice_type, cabac_init_flag),
                         qp);
}

/* One bin against a context, one in bypass, and the end-of-slice bin. */
static inline int hevcd_bin(hevcd_cabac_t *c, int ctx)
{
    return h264d_cabac_decision(c, ctx);
}

static inline int hevcd_bypass(hevcd_cabac_t *c)
{
    return h264d_cabac_bypass(c);
}

/* Several bypass bins, most significant first - in one division, see
 * h264d_cabac_bypass_n(). */
static inline uint32_t hevcd_bypass_n(hevcd_cabac_t *c, int n)
{
    return h264d_cabac_bypass_n(c, n);
}

static inline int hevcd_terminate(hevcd_cabac_t *c)
{
    return h264d_cabac_terminate(c);
}

static inline size_t hevcd_overrun(const hevcd_cabac_t *c)
{
    return h264d_cabac_overrun(c);
}

#endif /* BC250_HEVC_CABAC_DEC_H */
