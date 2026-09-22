/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_pred.c - intra prediction, one copy per bit depth.
 *
 * The work is in hevc_pred_template.c. This file exists to include it
 * twice and to pick between the two.
 */
#include "hevc_dec_internal.h"

#include <stdlib.h>
#include <string.h>

#define BIT_DEPTH 8
#include "hevc_pixel.h"
#include "hevc_pred_template.c"
#undef BIT_DEPTH

#define BIT_DEPTH 10
#include "hevc_pixel.h"
#include "hevc_pred_template.c"
#undef BIT_DEPTH

void hevcd_predict_intra(hevcd_t *d, int c_idx, int x0, int y0, int log2_size,
                         int mode)
{
    const int bd = c_idx ? d->sps->bit_depth_chroma : d->sps->bit_depth_luma;
    if (bd > 8) predict_intra_10(d, c_idx, x0, y0, log2_size, mode);
    else        predict_intra_8(d, c_idx, x0, y0, log2_size, mode);
}
