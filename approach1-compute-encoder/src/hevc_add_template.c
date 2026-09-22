/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_add_template.c - the residual onto the prediction.
 *
 * Included once per bit depth by hevc_transform.c. See hevc_pixel.h.
 */

/* 8.6.5: the residual onto the prediction, clipped back into the
 * picture's own depth. */
static void FUNC(add_residual)(pixel *dst, int stride, const int16_t *res,
                               int log2_size)
{
    const int n = 1 << log2_size;
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++) {
            const int v = dst[y * stride + x] + res[y * n + x];
            dst[y * stride + x] = (pixel)(v < 0 ? 0 : (v > PIXEL_MAX
                                                       ? PIXEL_MAX : v));
        }
}
