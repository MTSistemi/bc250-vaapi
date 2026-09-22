/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_filter.c - the deblocking filter, Rec. ITU-T H.265 clause 8.7.2.
 *
 * Block transforms leave steps at the block edges, and the eye finds a
 * straight edge that is not in the picture far more readily than it finds
 * the error that produced it. So the standard smooths them - and does it
 * normatively, in the decoding loop, because the filtered picture is what
 * the next one predicts from. A decoder that skips it does not merely look
 * worse: it drifts.
 *
 * ⚠️ Every vertical edge in the picture is filtered before any horizontal
 * one, and the horizontal pass reads what the vertical pass wrote. Doing
 * it edge by edge, both directions at once, gives a different picture.
 * Two passes over the whole picture is the simplest way to be sure, and
 * the passes are independent inside themselves: the edges of one direction
 * are eight samples apart and reach four, so no two of them touch.
 */
#include "hevc_dec_internal.h"

#include <stdlib.h>
#include <string.h>

#define BIT_DEPTH 8
#include "hevc_pixel.h"
#include "hevc_filter_template.c"
#undef BIT_DEPTH

#define BIT_DEPTH 10
#include "hevc_pixel.h"
#include "hevc_filter_template.c"
#undef BIT_DEPTH

void hevcd_deblock(hevcd_t *d)
{
    if (d->sps->bit_depth_luma > 8) deblock_10(d);
    else                            deblock_8(d);
}

void hevcd_sao(hevcd_t *d)
{
    if (d->sps->bit_depth_luma > 8) sao_10(d);
    else                            sao_8(d);
}

void hevcd_free_filters(hevcd_t *d)
{
    free(d->sao);
    d->sao = NULL;
    d->n_sao = 0;
    for (int c = 0; c < 3; c++) { free(d->copy_of[c]); d->copy_of[c] = NULL; }
    d->n_copy = 0;
}
