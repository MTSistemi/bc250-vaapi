/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_intra_template.c - the encoder's 4x4 intra prediction and mode
 * decision, once per bit depth.
 *
 * Included twice by hevc_intra.c, the way the decoder includes its own
 * templates: see hevc_pixel.h. `pixel` is a byte at eight bits and a
 * sixteen-bit word at ten, and every stride is counted in samples. The
 * eight-bit functions keep the names they always had (hevc_intra.h maps
 * the _8 names onto them); the ten-bit ones end in _10.
 */

static inline pixel FUNC(clip_pixel)(int v)
{
    return (pixel)(v < 0 ? 0 : (v > PIXEL_MAX ? PIXEL_MAX : v));
}

/* Gathers left[0..4] (p[-1][0..4]), top[0..4] (p[0..4][-1]) and the corner
 * (p[-1][-1]), applying the spec's neighbor-substitution scan. p[-1][5..7]
 * and positions beyond top[4] are never referenced by the four modes this
 * encoder uses (Planar/DC/H/V), so the scan below only covers what those
 * modes actually need. Every sample it does read gets a rank check, since
 * each can be positionally plausible and still z-scan-unavailable (see
 * zorder_rank()'s comment in hevc_intra.c) - p[-1][4] included, see the
 * end of the function. */
static void FUNC(gather_neighbors)(const pixel *plane, int stride, int width, int height,
                                   int x0, int y0, int is_luma, int y_min,
                                   pixel left[5], pixel top[5], pixel *corner,
                                   int *avail_left_out, int *avail_top_out) {
    long long cur_rank = zorder_rank(x0, y0, width, is_luma);
    int avail_left = zorder_available(x0 - 1, y0, width, height, is_luma, y_min, cur_rank);
    int avail_top = zorder_available(x0, y0 - 1, width, height, is_luma, y_min, cur_rank);
    int avail_corner = zorder_available(x0 - 1, y0 - 1, width, height, is_luma, y_min, cur_rank);
    int avail_top_right = zorder_available(x0 + 4, y0 - 1, width, height, is_luma, y_min, cur_rank);

    if (avail_left_out) *avail_left_out = avail_left;
    if (avail_top_out) *avail_top_out = avail_top;

    pixel sv[10];
    uint8_t sa[10];

    sa[0] = sa[1] = sa[2] = sa[3] = (uint8_t)avail_left;
    if (avail_left) {
        sv[0] = plane[(y0 + 3) * stride + (x0 - 1)];
        sv[1] = plane[(y0 + 2) * stride + (x0 - 1)];
        sv[2] = plane[(y0 + 1) * stride + (x0 - 1)];
        sv[3] = plane[(y0 + 0) * stride + (x0 - 1)];
    }
    sa[4] = (uint8_t)avail_corner;
    if (avail_corner) sv[4] = plane[(y0 - 1) * stride + (x0 - 1)];

    sa[5] = sa[6] = sa[7] = sa[8] = (uint8_t)avail_top;
    if (avail_top) {
        sv[5] = plane[(y0 - 1) * stride + (x0 + 0)];
        sv[6] = plane[(y0 - 1) * stride + (x0 + 1)];
        sv[7] = plane[(y0 - 1) * stride + (x0 + 2)];
        sv[8] = plane[(y0 - 1) * stride + (x0 + 3)];
    }
    sa[9] = (uint8_t)avail_top_right;
    if (avail_top_right) sv[9] = plane[(y0 - 1) * stride + (x0 + 4)];

    int first = -1;
    for (int i = 0; i < 10; i++) { if (sa[i]) { first = i; break; } }

    if (first < 0) {
        /* 8.4.4.2.2: with no neighbour at all every reference sample is
         * 1 << (BitDepth - 1) - 128 at eight bits, 512 at ten. */
        for (int i = 0; i < 10; i++) sv[i] = (pixel)(1 << (BIT_DEPTH - 1));
    } else {
        for (int i = 0; i < first; i++) sv[i] = sv[first];
        for (int i = first + 1; i < 10; i++) if (!sa[i]) sv[i] = sv[i - 1];
    }

    left[3] = sv[0]; left[2] = sv[1]; left[1] = sv[2]; left[0] = sv[3];
    *corner = sv[4];
    top[0] = sv[5]; top[1] = sv[6]; top[2] = sv[7]; top[3] = sv[8];
    top[4] = sv[9];
    /* p[-1][4], the first sample below the left column, which Planar reads
     * and nothing else here does.
     *
     * ⚠️ It is NOT always unavailable. This said it was, and set it to
     * left[3] unconditionally - which is right only when the block below
     * left has not been coded yet. For a block on the left edge of a CTU
     * that block sits in the CTU to the left, coded long before; for the
     * first PU of a right-hand CU it is the last PU of the CU to its left.
     * A decoder reads the real sample there, this encoder used another, and
     * every Planar block in those places was reconstructed differently on
     * the two sides - and everything predicted from it after that. Measured
     * at 1280x720 through hevc_encoder_encode_raw(): the encoder's own
     * reconstruction at
     * 38-46 dB against the source, ffmpeg's decode of the same stream at
     * 27-29 dB. Chroma, which only ever uses DC, was exact.
     *
     * Same rank test as the other neighbours. When it is not there,
     * 8.4.4.2.2's substitution walks up from the bottom and fills it from
     * the nearest sample above, which is left[3] - substituted itself if it
     * had to be. */
    if (zorder_available(x0 - 1, y0 + 4, width, height, is_luma, y_min, cur_rank))
        left[4] = plane[(y0 + 4) * stride + (x0 - 1)];
    else
        left[4] = left[3];
}

/* ===================== prediction (8.4.4.2.5-8.4.4.2.7) ===================== */

static inline void FUNC(predict_from_refs)(const pixel left[5], const pixel top[5], pixel corner,
                                           int mode, int is_luma, pixel pred_out[16]) {
    switch (mode) {
    case HEVC_MODE_PLANAR:
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++) {
                int v = (3 - x) * left[y] + (x + 1) * top[4] +
                        (3 - y) * top[x] + (y + 1) * left[4] + 4;
                pred_out[y * 4 + x] = (pixel)(v >> 3);
            }
        break;

    case HEVC_MODE_DC: {
        int dc = (left[0] + left[1] + left[2] + left[3] +
                  top[0] + top[1] + top[2] + top[3] + 4) >> 3;
        for (int i = 0; i < 16; i++) pred_out[i] = (pixel)dc;
        if (is_luma) {
            pred_out[0] = (pixel)((left[0] + 2 * dc + top[0] + 2) >> 2);
            for (int x = 1; x < 4; x++) pred_out[x] = (pixel)((top[x] + 3 * dc + 2) >> 2);
            for (int y = 1; y < 4; y++) pred_out[y * 4] = (pixel)((left[y] + 3 * dc + 2) >> 2);
        }
        break;
    }

    case HEVC_MODE_HORIZONTAL:
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                pred_out[y * 4 + x] = left[y];
        if (is_luma) {
            for (int x = 0; x < 4; x++)
                pred_out[x] = FUNC(clip_pixel)(left[0] + ((top[x] - corner) >> 1));
        }
        break;

    case HEVC_MODE_VERTICAL:
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                pred_out[y * 4 + x] = top[x];
        if (is_luma) {
            for (int y = 0; y < 4; y++)
                pred_out[y * 4] = FUNC(clip_pixel)(top[0] + ((left[y] - corner) >> 1));
        }
        break;

    default:
        for (int i = 0; i < 16; i++) pred_out[i] = (pixel)(1 << (BIT_DEPTH - 1));
        break;
    }
}

void FUNC(hevc_predict_4x4)(const pixel *recon_plane, int stride, int width, int height,
                            int x0, int y0, int mode, int is_luma, int y_min,
                            pixel pred_out[16]) {
    pixel left[5], top[5], corner;
    int avail_left = 0, avail_top = 0;
    FUNC(gather_neighbors)(recon_plane, stride, width, height, x0, y0, is_luma, y_min,
                           left, top, &corner, &avail_left, &avail_top);
    FUNC(predict_from_refs)(left, top, corner, mode, is_luma, pred_out);
}

static inline int FUNC(sad_4x4)(const pixel a[16], const pixel b[16]) {
    int sad = 0;
    for (int i = 0; i < 16; i++) {
        int d = (int)a[i] - (int)b[i];
        sad += d < 0 ? -d : d;
    }
    return sad;
}

int FUNC(hevc_choose_luma_mode)(int y_min, const pixel *src_y, const pixel *recon_y, int stride,
                                int width, int height, int x0, int y0, pixel pred_out[16]) {
    /* Gather ONCE for all candidates instead of 4 separate calls */
    pixel left[5], top[5], corner;
    FUNC(gather_neighbors)(recon_y, stride, width, height, x0, y0, 1, y_min,
                           left, top, &corner, NULL, NULL);

    /* Hoist 4x4 source block into contiguous 16 samples for vectorized SAD */
    pixel src16[16];
    const pixel *srow = src_y + (size_t)y0 * (size_t)stride + (size_t)x0;
    memcpy(src16 + 0,  srow,                      4 * sizeof(pixel));
    memcpy(src16 + 4,  srow + stride,             4 * sizeof(pixel));
    memcpy(src16 + 8,  srow + 2 * (size_t)stride, 4 * sizeof(pixel));
    memcpy(src16 + 12, srow + 3 * (size_t)stride, 4 * sizeof(pixel));

    FUNC(predict_from_refs)(left, top, corner, HEVC_MODE_PLANAR, 1, pred_out);
    int best_mode = HEVC_MODE_PLANAR;
    int best_sad = FUNC(sad_4x4)(src16, pred_out);

#define HEVC_TRY_MODE(M) do {                                              \
        pixel pred_[16];                                                   \
        FUNC(predict_from_refs)(left, top, corner, (M), 1, pred_);         \
        int sad_ = FUNC(sad_4x4)(src16, pred_);                            \
        if (sad_ < best_sad) {                                             \
            best_sad = sad_;                                               \
            best_mode = (M);                                               \
            memcpy(pred_out, pred_, sizeof(pred_));                        \
        }                                                                  \
    } while (0)

    HEVC_TRY_MODE(HEVC_MODE_DC);
    HEVC_TRY_MODE(HEVC_MODE_HORIZONTAL);
    HEVC_TRY_MODE(HEVC_MODE_VERTICAL);
#undef HEVC_TRY_MODE

    return best_mode;
}
