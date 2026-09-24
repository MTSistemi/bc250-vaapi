/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_enc_template.c - the part of the HEVC encoder that touches samples,
 * once per bit depth.
 *
 * Included twice by encoder_h265.c, the way the decoder includes its own
 * templates: see hevc_pixel.h. `pixel` is a byte at eight bits and a
 * sixteen-bit word at ten, and every stride is counted in samples.
 *
 * Only three things change with the depth. The QP the transforms see is
 * Qp' = QP + 6 * (BitDepth - 8) (7.4.3.2.1), the samples clip at
 * (1 << BitDepth) - 1, and a sum of absolute differences is four times
 * larger at ten bits, so it is brought back to eight-bit units before it
 * meets a threshold or the rate control - both were tuned at eight bits and
 * should mean the same thing at ten.
 */

#undef QP_BD_OFFSET
#define QP_BD_OFFSET (6 * (BIT_DEPTH - 8))

static inline pixel FUNC(clip_sample)(int v) {
    return (pixel)(v < 0 ? 0 : (v > PIXEL_MAX ? PIXEL_MAX : v));
}

/* Replicate-pad a downloaded plane (real w x h) into a coded_w x coded_h
 * working buffer - only the bottom/right margin (if any) needs padding,
 * since coded dims are always >= real dims by construction. */
static void FUNC(pad_replicate)(pixel *dst, uint32_t dst_w, uint32_t dst_h,
                                const pixel *src, uint32_t src_stride,
                                uint32_t src_w, uint32_t src_h) {
    for (uint32_t y = 0; y < dst_h; y++) {
        uint32_t sy = y < src_h ? y : src_h - 1;
        const pixel *srow = src + (size_t)sy * src_stride;
        pixel *drow = dst + (size_t)y * dst_w;
        for (uint32_t x = 0; x < dst_w; x++) {
            uint32_t sx = x < src_w ? x : src_w - 1;
            drow[x] = srow[sx];
        }
    }
}

#if BIT_DEPTH == 8

static inline uint32_t FUNC(compute_sad_8x8_luma)(const pixel *src_y,
                                                  const pixel *ref_y,
                                                  uint32_t stride,
                                                  int cu_x, int cu_y,
                                                  int dx, int dy)
{
    const uint8_t *s = &src_y[cu_y * stride + cu_x];
    const uint8_t *r = &ref_y[(cu_y + dy) * stride + (cu_x + dx)];
#if defined(__SSE2__) || defined(__x86_64__) || defined(_M_X64)
    __m128i acc = _mm_setzero_si128();
    for (int y = 0; y < 8; y++) {
        __m128i s_row = _mm_loadl_epi64((const __m128i *)s);
        __m128i r_row = _mm_loadl_epi64((const __m128i *)r);
        acc = _mm_add_epi32(acc, _mm_sad_epu8(s_row, r_row));
        s += stride;
        r += stride;
    }
    return (uint32_t)_mm_cvtsi128_si32(acc);
#else
    uint32_t sad = 0;
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            int diff = (int)s[x] - (int)r[x];
            sad += (diff < 0) ? -diff : diff;
        }
        s += stride;
        r += stride;
    }
    return sad;
#endif
}

static inline uint32_t FUNC(compute_sad_4x4_chroma)(const pixel *src_cb,
                                                    const pixel *src_cr,
                                                    const pixel *ref_cb,
                                                    const pixel *ref_cr,
                                                    uint32_t cstride,
                                                    int cx, int cy,
                                                    int cdx, int cdy)
{
    const uint8_t *scb = &src_cb[cy * cstride + cx];
    const uint8_t *scr = &src_cr[cy * cstride + cx];
    const uint8_t *rcb = &ref_cb[(cy + cdy) * cstride + (cx + cdx)];
    const uint8_t *rcr = &ref_cr[(cy + cdy) * cstride + (cx + cdx)];
#if defined(__SSE2__) || defined(__x86_64__) || defined(_M_X64)
    __m128i acc = _mm_setzero_si128();
    for (int y = 0; y < 4; y++) {
        uint32_t scb_4, scr_4, rcb_4, rcr_4;
        memcpy(&scb_4, scb, 4);
        memcpy(&scr_4, scr, 4);
        memcpy(&rcb_4, rcb, 4);
        memcpy(&rcr_4, rcr, 4);
        uint64_t s_both = ((uint64_t)scr_4 << 32) | scb_4;
        uint64_t r_both = ((uint64_t)rcr_4 << 32) | rcb_4;
        __m128i s_vec = _mm_loadl_epi64((const __m128i *)&s_both);
        __m128i r_vec = _mm_loadl_epi64((const __m128i *)&r_both);
        acc = _mm_add_epi32(acc, _mm_sad_epu8(s_vec, r_vec));
        scb += cstride;
        scr += cstride;
        rcb += cstride;
        rcr += cstride;
    }
    return (uint32_t)_mm_cvtsi128_si32(acc);
#else
    uint32_t sad = 0;
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            int dcb = (int)scb[x] - (int)rcb[x];
            int dcr = (int)scr[x] - (int)rcr[x];
            sad += (dcb < 0 ? -dcb : dcb) + (dcr < 0 ? -dcr : dcr);
        }
        scb += cstride;
        scr += cstride;
        rcb += cstride;
        rcr += cstride;
    }
    return sad;
#endif
}

#else /* BIT_DEPTH > 8 */

/* PSADBW only takes bytes. With sixteen-bit samples the absolute
 * difference is the two saturating subtractions ORed together - one of
 * them is always zero - and PMADDWD by one adds neighbouring pairs into
 * 32 bits, where eight rows of ten-bit differences cannot overflow. */
static inline uint32_t FUNC(compute_sad_8x8_luma)(const pixel *src_y,
                                                  const pixel *ref_y,
                                                  uint32_t stride,
                                                  int cu_x, int cu_y,
                                                  int dx, int dy)
{
    const pixel *s = &src_y[cu_y * stride + cu_x];
    const pixel *r = &ref_y[(cu_y + dy) * stride + (cu_x + dx)];
#if defined(__SSE2__) || defined(__x86_64__) || defined(_M_X64)
    const __m128i one = _mm_set1_epi16(1);
    __m128i acc = _mm_setzero_si128();
    for (int y = 0; y < 8; y++) {
        __m128i a = _mm_loadu_si128((const __m128i *)s);
        __m128i b = _mm_loadu_si128((const __m128i *)r);
        __m128i d = _mm_or_si128(_mm_subs_epu16(a, b), _mm_subs_epu16(b, a));
        acc = _mm_add_epi32(acc, _mm_madd_epi16(d, one));
        s += stride;
        r += stride;
    }
    acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, _MM_SHUFFLE(1, 0, 3, 2)));
    acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, _MM_SHUFFLE(2, 3, 0, 1)));
    return (uint32_t)_mm_cvtsi128_si32(acc);
#else
    uint32_t sad = 0;
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            int diff = (int)s[x] - (int)r[x];
            sad += (diff < 0) ? -diff : diff;
        }
        s += stride;
        r += stride;
    }
    return sad;
#endif
}

static inline uint32_t FUNC(compute_sad_4x4_chroma)(const pixel *src_cb,
                                                    const pixel *src_cr,
                                                    const pixel *ref_cb,
                                                    const pixel *ref_cr,
                                                    uint32_t cstride,
                                                    int cx, int cy,
                                                    int cdx, int cdy)
{
    const pixel *scb = &src_cb[cy * cstride + cx];
    const pixel *scr = &src_cr[cy * cstride + cx];
    const pixel *rcb = &ref_cb[(cy + cdy) * cstride + (cx + cdx)];
    const pixel *rcr = &ref_cr[(cy + cdy) * cstride + (cx + cdx)];
    uint32_t sad = 0;
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            int dcb = (int)scb[x] - (int)rcb[x];
            int dcr = (int)scr[x] - (int)rcr[x];
            sad += (dcb < 0 ? -dcb : dcb) + (dcr < 0 ? -dcr : dcr);
        }
        scb += cstride;
        scr += cstride;
        rcb += cstride;
        rcr += cstride;
    }
    return sad;
}

#endif /* BIT_DEPTH */

static void FUNC(encode_cu)(hevc_encoder_t *enc, hevc_cabac_t *cab, int cu_x, int cu_y, bool is_idr, int y_min, uint32_t *sad_out) {
    int qp = enc->qp;
    uint32_t cw = enc->coded_width, ch = enc->coded_height;
    uint32_t ccw = cw / 2, cch = ch / 2;
    int cux = cu_x / HEVC_CU_SIZE;
    int cuy = cu_y / HEVC_CU_SIZE;
    uint32_t cu_stride = enc->width_ctu * 2;
    uint32_t cu_idx = (uint32_t)cuy * cu_stride + (uint32_t)cux;

    const pixel *src_y = enc->src_y, *src_cb = enc->src_cb, *src_cr = enc->src_cr;
    pixel *recon_y = enc->recon_y, *recon_cb = enc->recon_cb, *recon_cr = enc->recon_cr;
    const pixel *prev_y = enc->prev_recon_y;
    const pixel *prev_cb = enc->prev_recon_cb, *prev_cr = enc->prev_recon_cr;

    int cond_l = (cux > 0 && enc->cu_skip_map[cu_idx - 1]) ? 1 : 0;
    int cond_a = (cu_y > y_min && enc->cu_skip_map[cu_idx - cu_stride]) ? 1 : 0;
    int skip_ctx_inc = cond_l + cond_a;

    bool is_skip = false;
    int chosen_merge_idx = 0;
    int chosen_dx = 0, chosen_dy = 0;

    if (!is_idr && enc->has_ref) {
        hevc_mv_t cand_mvs[5];
        derive_merge_candidates(y_min / HEVC_CU_SIZE, enc, cux, cuy, cand_mvs);

        int best_cand_idx = -1;
        uint32_t best_cand_sad = UINT32_MAX;
        int cx = cu_x / 2, cy = cu_y / 2;

        /* GPU Motion Vector guidance: if GPU confirmed low-distortion stationary block, fast-track candidate (0,0) */
        uint32_t ctu_idx = ((uint32_t)cuy / 2) * enc->width_ctu + ((uint32_t)cux / 2);
        bool gpu_says_static = false;
        if (enc->num_gpu_mvs > 0 && ctu_idx < enc->num_gpu_mvs) {
            const gpu_mv_t *gm = &enc->gpu_mvs[ctu_idx];
            if (gm->mvx == 0 && gm->mvy == 0 && gm->sad <= 384) {
                gpu_says_static = true;
            }
        }

        /* Evaluate ITU-T standard merge candidates in order */
        for (int i = 0; i < 5; i++) {
            int c_dx = cand_mvs[i].x / 4;
            int c_dy = cand_mvs[i].y / 4;

            /* Check frame bounds */
            if (cu_x + c_dx < 0 || cu_x + c_dx + 8 > (int)cw ||
                cu_y + c_dy < 0 || cu_y + c_dy + 8 > (int)ch) continue;

            /* Skip redundant evaluations */
            bool dup = false;
            for (int p = 0; p < i; p++) {
                if (cand_mvs[p].x == cand_mvs[i].x && cand_mvs[p].y == cand_mvs[i].y) {
                    dup = true;
                    break;
                }
            }
            if (dup) continue;

            int cdx = (c_dx >= 0) ? (c_dx >> 1) : ((c_dx - 1) >> 1);
            int cdy = (c_dy >= 0) ? (c_dy >> 1) : ((c_dy - 1) >> 1);
            uint32_t c_sad = FUNC(compute_sad_8x8_luma)(src_y, prev_y, cw, cu_x, cu_y, c_dx, c_dy) +
                             FUNC(compute_sad_4x4_chroma)(src_cb, src_cr, prev_cb, prev_cr,
                                                          ccw, cx, cy, cdx, cdy);
            c_sad >>= BIT_DEPTH - 8;
            if (c_sad < best_cand_sad) {
                best_cand_sad = c_sad;
                best_cand_idx = i;
                /* Fast path: stationary (0,0) or perfect match exits immediately */
                if (c_dx == 0 && c_dy == 0 && (best_cand_sad <= 32 || (gpu_says_static && best_cand_sad <= 96))) {
                    break;
                }
            }
        }

        /* Skip threshold scaled with QP and quality level.
         * For quality levels 1..3 (high-quality archival / transcode), keep threshold strict
         * so subtle textures, grain, and fine motion are preserved with full residual coding.
         * Quality levels 4..5 use balanced skipping, and 6..7 use speed skipping. */
        uint32_t threshold;
        if (enc->quality_level <= 3) {
            threshold = 48 * (1 + (enc->qp / 16));
        } else if (enc->quality_level <= 5) {
            threshold = 64 * (1 + (enc->qp / 12));
        } else {
            threshold = 96 * (1 + (enc->qp / 8));
        }
        if (enc->skip_override >= 0) {
            threshold = (uint32_t)enc->skip_override;
        }

        if (best_cand_idx >= 0 && best_cand_sad <= threshold) {
            is_skip = true;
            chosen_merge_idx = best_cand_idx;
            chosen_dx = cand_mvs[best_cand_idx].x / 4;
            chosen_dy = cand_mvs[best_cand_idx].y / 4;
            *sad_out += best_cand_sad;
        } else {
            *sad_out += (best_cand_sad != UINT32_MAX ? best_cand_sad : (threshold * 2));
        }
    }

    if (is_skip) {
        enc->cu_skip_map[cu_idx] = 1;
        enc->cu_is_inter[cu_idx] = 1;
        enc->mv_x_map[cu_idx] = (int16_t)(chosen_dx * 4);
        enc->mv_y_map[cu_idx] = (int16_t)(chosen_dy * 4);
        hevc_cabac_code_cu_skip_flag(cab, 1, skip_ctx_inc);
        hevc_cabac_code_merge_idx(cab, chosen_merge_idx);

        for (int y = 0; y < HEVC_CU_SIZE; y++) {
            memcpy(&recon_y[(cu_y + y) * cw + cu_x],
                   &prev_y[(cu_y + chosen_dy + y) * cw + (cu_x + chosen_dx)],
                   HEVC_CU_SIZE * sizeof(pixel));
        }
        int cx = cu_x / 2, cy = cu_y / 2;
        int cdx = (chosen_dx >= 0) ? (chosen_dx >> 1) : ((chosen_dx - 1) >> 1);
        int cdy = (chosen_dy >= 0) ? (chosen_dy >> 1) : ((chosen_dy - 1) >> 1);
        for (int y = 0; y < HEVC_PU_SIZE; y++) {
            memcpy(&recon_cb[(cy + y) * ccw + cx],
                   &prev_cb[(cy + cdy + y) * ccw + (cx + cdx)],
                   HEVC_PU_SIZE * sizeof(pixel));
            memcpy(&recon_cr[(cy + y) * ccw + cx],
                   &prev_cr[(cy + cdy + y) * ccw + (cx + cdx)],
                   HEVC_PU_SIZE * sizeof(pixel));
        }
        for (int pu = 0; pu < 4; pu++) {
            int px = cu_x + pu_off_x[pu], py = cu_y + pu_off_y[pu];
            enc->luma_mode_map[(py / 4) * enc->mode_map_stride + (px / 4)] = HEVC_MODE_DC;
        }
        return;
    }

    enc->cu_skip_map[cu_idx] = 0;
    enc->cu_is_inter[cu_idx] = 0;
    enc->mv_x_map[cu_idx] = 0;
    enc->mv_y_map[cu_idx] = 0;
    if (!is_idr) {
        hevc_cabac_code_cu_skip_flag(cab, 0, skip_ctx_inc);
        hevc_cabac_code_pred_mode_flag(cab, 1 /* MODE_INTRA */);
    }

    hevc_cabac_code_part_mode_intra(cab, 0 /* PART_NxN */);

    int pu_modes[4];
    int16_t luma_coeff[4][16];
    int cbf_luma[4];

    /* Step 1: decide + reconstruct all 4 luma PUs in z-order */
    for (int pu = 0; pu < 4; pu++) {
        int px = cu_x + pu_off_x[pu], py = cu_y + pu_off_y[pu];
        pixel pred[16];
        int mode;
        /* Quality levels 1..6 use full directional intra prediction (Planar, DC, Horizontal, Vertical)
         * to preserve edges and textures. Level 7 (ultra-fast speed preset) uses DC fallback. */
        if (enc->quality_level >= 7) {
            mode = HEVC_MODE_DC;
            FUNC(hevc_predict_4x4)(recon_y, cw, cw, ch, px, py, mode, 1, y_min, pred);
        } else {
            mode = FUNC(hevc_choose_luma_mode)(y_min, src_y, recon_y, (int)cw, (int)cw, (int)ch, px, py, pred);
        }
        pu_modes[pu] = mode;

        int16_t residual[16];
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                residual[y * 4 + x] = (int16_t)(src_y[(py + y) * cw + (px + x)] - pred[y * 4 + x]);

        int16_t coeff[16];
        FUNC(hevc_transform_quant_4x4)(residual, qp + QP_BD_OFFSET, 1 /* DST for 4x4 luma intra */, coeff);
        memcpy(luma_coeff[pu], coeff, sizeof(coeff));
        cbf_luma[pu] = any_nonzero16(coeff);

        if (cbf_luma[pu]) {
            int16_t recon_residual[16];
            FUNC(hevc_dequant_itransform_4x4)(coeff, qp + QP_BD_OFFSET, 1, recon_residual);
            for (int y = 0; y < 4; y++)
                for (int x = 0; x < 4; x++)
                    recon_y[(py + y) * cw + (px + x)] = FUNC(clip_sample)(pred[y * 4 + x] + recon_residual[y * 4 + x]);
        } else {
            for (int y = 0; y < 4; y++)
                memcpy(&recon_y[(py + y) * cw + px], &pred[y * 4], 4 * sizeof(pixel));
        }

        enc->luma_mode_map[(py / 4) * enc->mode_map_stride + (px / 4)] = (int8_t)mode;
    }

    /* Chroma: one 4x4 Cb + one 4x4 Cr per CU, DC prediction only */
    int cx = cu_x / 2, cy = cu_y / 2;
    pixel pred_cb[16], pred_cr[16];
    FUNC(hevc_predict_4x4)(recon_cb, ccw, ccw, cch, cx, cy, HEVC_MODE_DC, 0, y_min / 2, pred_cb);
    FUNC(hevc_predict_4x4)(recon_cr, ccw, ccw, cch, cx, cy, HEVC_MODE_DC, 0, y_min / 2, pred_cr);

    int16_t res_cb[16], res_cr[16];
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) {
            res_cb[y * 4 + x] = (int16_t)(src_cb[(cy + y) * ccw + (cx + x)] - pred_cb[y * 4 + x]);
            res_cr[y * 4 + x] = (int16_t)(src_cr[(cy + y) * ccw + (cx + x)] - pred_cr[y * 4 + x]);
        }

    /* Chroma quantizes at QpC, not QpY - Rec. ITU-T H.265 Table 8-10.
     * Passing luma QP directly causes divergence from the standard when QP >= 30. */
    int cqp = hevc_chroma_qp_from_luma(qp) + QP_BD_OFFSET;

    int16_t coeff_cb[16], coeff_cr[16];
    FUNC(hevc_transform_quant_4x4)(res_cb, cqp, 0, coeff_cb);
    FUNC(hevc_transform_quant_4x4)(res_cr, cqp, 0, coeff_cr);
    int cbf_cb = any_nonzero16(coeff_cb);
    int cbf_cr = any_nonzero16(coeff_cr);

    if (cbf_cb) {
        int16_t rres_cb[16];
        FUNC(hevc_dequant_itransform_4x4)(coeff_cb, cqp, 0, rres_cb);
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                recon_cb[(cy + y) * ccw + (cx + x)] = FUNC(clip_sample)(pred_cb[y * 4 + x] + rres_cb[y * 4 + x]);
    } else {
        for (int y = 0; y < 4; y++)
            memcpy(&recon_cb[(cy + y) * ccw + cx], &pred_cb[y * 4], 4 * sizeof(pixel));
    }

    if (cbf_cr) {
        int16_t rres_cr[16];
        FUNC(hevc_dequant_itransform_4x4)(coeff_cr, cqp, 0, rres_cr);
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                recon_cr[(cy + y) * ccw + (cx + x)] = FUNC(clip_sample)(pred_cr[y * 4 + x] + rres_cr[y * 4 + x]);
    } else {
        for (int y = 0; y < 4; y++)
            memcpy(&recon_cr[(cy + y) * ccw + cx], &pred_cr[y * 4], 4 * sizeof(pixel));
    }

    /* Step 2: emit the 4 PUs' real intra_luma_pred_mode syntax. ITU-T
     * H.265 7.3.8.5's coding_unit() codes this as TWO separate passes over
     * all 4 PUs - every prev_intra_luma_pred_flag first, THEN every
     * mpm_idx/rem_intra_luma_pred_mode - not interleaved per PU (see
     * hevc_cabac_code_intra_luma_flag()/_data()'s comment; getting this
     * order wrong was this encoder's first real bug, caught by comparing
     * this encoder's own reconstruction - which matched the source fine -
     * against ffmpeg's actual decode of the resulting bitstream, which
     * didn't: a CABAC bit-order mistake still produces a structurally
     * valid, crash-free bitstream, just one that decodes to noise from
     * that point on). */
    int mpm[4][3];
    int pred_idx[4];
    for (int pu = 0; pu < 4; pu++) {
        int px = cu_x + pu_off_x[pu], py = cu_y + pu_off_y[pu];
        int mx = px / 4, my = py / 4;
        int left_avail = px > 0;
        /* Per ITU-T H.265 8.4.2, candIntraPredModeB is forced to INTRA_DC
         * whenever yCb-1 crosses into the CTU row above the current one.
         * Marking above_avail false across CTU boundaries ensures bit-exact
         * MPM candidate list synchronization with all standard decoders. */
        int above_avail = (py > 0) && ((py % HEVC_CTU_SIZE) != 0);
        int left_mode = left_avail ? enc->luma_mode_map[my * enc->mode_map_stride + (mx - 1)] : 0;
        int above_mode = above_avail ? enc->luma_mode_map[(my - 1) * enc->mode_map_stride + mx] : 0;
        hevc_derive_mpm(left_mode, left_avail, above_mode, above_avail, mpm[pu]);
        pred_idx[pu] = hevc_cabac_code_intra_luma_flag(cab, pu_modes[pu], mpm[pu]);
    }
    for (int pu = 0; pu < 4; pu++)
        hevc_cabac_code_intra_luma_data(cab, pu_modes[pu], pred_idx[pu], mpm[pu]);

    /* Step 3: chroma mode (always DC; luma_mode_pu0 decides whether that's
     * signaled as index-3-of-candidate-list or as the derived/DM mode -
     * see hevc_cabac_code_intra_chroma_pred_mode()'s comment). */
    hevc_cabac_code_intra_chroma_pred_mode(cab, pu_modes[0]);

    /* Step 4: transform_tree - chroma cbf BITS first (trafoDepth=0, this
     * CU's root), then the 4 luma leaves' cbf+residual, then finally the
     * chroma RESIDUAL DATA (coded once per CU, after all 4 luma leaves -
     * this specific ordering, bits-before-luma but data-after-luma, is
     * exactly what ITU-T H.265's transform_tree()/transform_unit()
     * recursion produces for a CU whose chroma has already hit the 4x4
     * floor - see this file's top comment and x265's own
     * Entropy::encodeTransform(), which this encoder's fixed two-level
     * structure is a manually-unrolled special case of). */
    hevc_cabac_code_cbf_chroma(cab, cbf_cb, 0);
    hevc_cabac_code_cbf_chroma(cab, cbf_cr, 0);

    for (int pu = 0; pu < 4; pu++) {
        hevc_cabac_code_cbf_luma(cab, cbf_luma[pu], 1);
        if (cbf_luma[pu]) {
            int scan_idx = hevc_scan_idx_for_mode(pu_modes[pu]);
            hevc_cabac_code_residual_4x4(cab, luma_coeff[pu], 1, scan_idx);
        }
    }
    if (cbf_cb) hevc_cabac_code_residual_4x4(cab, coeff_cb, 0, 0 /* chroma always diagonal in 4:2:0 */);
    if (cbf_cr) hevc_cabac_code_residual_4x4(cab, coeff_cr, 0, 0);
}

static void FUNC(encode_ctu)(hevc_encoder_t *enc, hevc_cabac_t *cab, int ctu_col, int ctu_row, bool is_idr, int y_min, uint32_t *sad_out) {
    int ctu_x = ctu_col * HEVC_CTU_SIZE, ctu_y = ctu_row * HEVC_CTU_SIZE;
    int cond_l = ctu_col > 0 ? 1 : 0;
    int cond_a = (ctu_row * HEVC_CTU_SIZE > y_min) ? 1 : 0;
    hevc_cabac_code_split_cu_flag(cab, 1, cond_l + cond_a);

    static const int cu_off_x[4] = { 0, 8, 0, 8 };
    static const int cu_off_y[4] = { 0, 0, 8, 8 };
    for (int i = 0; i < 4; i++)
        FUNC(encode_cu)(enc, cab, ctu_x + cu_off_x[i], ctu_y + cu_off_y[i], is_idr, y_min, sad_out);
}

/* The downloaded picture (dl_y / dl_uv, real width x height, chroma
 * interleaved) into the separate, padded source planes the encoder reads.
 * At ten bits the download is P010: each sample sits in the top ten bits
 * of its word, so it comes down by six on the way. */
static void FUNC(load_source)(hevc_encoder_t *encoder) {
    pixel *src_y = encoder->src_y, *src_cb = encoder->src_cb, *src_cr = encoder->src_cr;
    uint32_t cw2 = encoder->width / 2, ch2 = encoder->height / 2;
    uint32_t ccw = encoder->coded_width / 2, cch = encoder->coded_height / 2;
#if BIT_DEPTH == 8
    FUNC(pad_replicate)(src_y, encoder->coded_width, encoder->coded_height,
                        encoder->dl_y, encoder->width, encoder->width, encoder->height);

    for (uint32_t y = 0; y < ch2; y++) {
        const uint8_t *uvrow = encoder->dl_uv + (size_t)y * encoder->width;
        deinterleava_uv(&src_cb[y * ccw], &src_cr[y * ccw], uvrow, cw2);
    }
#else
    const uint16_t *dl_y = (const uint16_t *)encoder->dl_y;
    const uint16_t *dl_uv = (const uint16_t *)encoder->dl_uv;
    for (uint32_t y = 0; y < encoder->height; y++) {
        const uint16_t *s = dl_y + (size_t)y * encoder->width;
        pixel *d = src_y + (size_t)y * encoder->coded_width;
        for (uint32_t x = 0; x < encoder->width; x++) d[x] = (pixel)(s[x] >> 6);
    }
    FUNC(pad_replicate)(src_y, encoder->coded_width, encoder->coded_height,
                        src_y, encoder->coded_width, encoder->width, encoder->height);

    for (uint32_t y = 0; y < ch2; y++) {
        const uint16_t *s = dl_uv + (size_t)y * encoder->width;
        pixel *cb = src_cb + (size_t)y * ccw, *cr = src_cr + (size_t)y * ccw;
        for (uint32_t x = 0; x < cw2; x++) {
            cb[x] = (pixel)(s[2 * x] >> 6);
            cr[x] = (pixel)(s[2 * x + 1] >> 6);
        }
    }
#endif
    FUNC(pad_replicate)(src_cb, ccw, cch, src_cb, ccw, cw2, ch2);
    FUNC(pad_replicate)(src_cr, ccw, cch, src_cr, ccw, cw2, ch2);
}
