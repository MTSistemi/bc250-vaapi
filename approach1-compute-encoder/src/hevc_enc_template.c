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

/* ------------------------------------------------ search-only half planes */

/* The previous picture's luma at whole samples and at the three half-sample
 * phases (right, below, both), each with a margin of HPEL_MARGIN samples of
 * repeated edge, rebuilt once per P picture.
 *
 * ⚠️ For the motion SEARCH only. A quarter sample is approximated as the
 * average of the two nearest points of this half-sample grid, which is how
 * H.264 defines it and not how HEVC does - HEVC filters each quarter phase
 * with its own taps. That is fine for choosing a vector: the prediction that
 * goes into the picture is always built by hevc_mc_uni(), the decoder's own
 * interpolation, from the vector chosen. Interpolating every candidate the
 * exact way was a quarter of the encoder's time. */
static void FUNC(build_hpel)(hevc_encoder_t *enc)
{
    const int w = (int)enc->coded_width, h = (int)enc->coded_height;
    const int M = HPEL_MARGIN, ps = enc->hpel_stride;
    const pixel *ref = enc->prev_recon_y;
    pixel *I = (pixel *)enc->hpel[0], *H = (pixel *)enc->hpel[1];
    pixel *V = (pixel *)enc->hpel[2], *HV = (pixel *)enc->hpel[3];
    int32_t *T = enc->hpel_tmp;   /* horizontal sums, picture rows -M-3 .. h+M+3 */
    const int rows = h + 2 * M, cols = w + 2 * M;

#define CLAMPX(xx) ((xx) < 0 ? 0 : ((xx) >= w ? w - 1 : (xx)))
#define CLAMPY(yy) ((yy) < 0 ? 0 : ((yy) >= h ? h - 1 : (yy)))
/* The half-sample filter, 8.5.3.3.3.1, taps -1 4 -11 40 40 -11 4 -1. */
#define TAPS(p, s) (-(p)[0] + 4 * (p)[(s)] - 11 * (p)[2 * (s)] + 40 * (p)[3 * (s)] \
                    + 40 * (p)[4 * (s)] - 11 * (p)[5 * (s)] + 4 * (p)[6 * (s)] - (p)[7 * (s)])

    /* Pass 1: every source row, edge-extended by M + 4 samples on each side,
     * then the horizontal filter over it - one clean loop the compiler can
     * vectorize, instead of a clamp per tap. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows + 7; r++) {
        const pixel *row = ref + (size_t)CLAMPY(r - M - 3) * w;
        pixel ext[w + 2 * M + 8];
        for (int i = 0; i < M + 3; i++) ext[i] = row[0];
        memcpy(ext + M + 3, row, (size_t)w * sizeof(pixel));
        for (int i = M + 3 + w; i < w + 2 * M + 8; i++) ext[i] = row[w - 1];
        int32_t *o = T + (size_t)r * ps;
        /* ext[c] is picture column c - M - 3: the taps for output column c
         * start there. */
        for (int c = 0; c < cols; c++) o[c] = TAPS(ext + c, 1);
    }

    /* Pass 2: whole, right half, lower half, both. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; r++) {
        const int y = r - M;
        const int32_t *t8 = T + (size_t)r * ps;   /* row r of T is picture row y - 3 */
        pixel *Ir = I + (size_t)r * ps, *Hr = H + (size_t)r * ps;
        pixel *Vr = V + (size_t)r * ps, *HVr = HV + (size_t)r * ps;
        /* The eight source rows the vertical filter reads, edge-extended the
         * same way. */
        pixel ext[8][w + 2 * M];
        for (int k = 0; k < 8; k++) {
            const pixel *row = ref + (size_t)CLAMPY(y - 3 + k) * w;
            for (int i = 0; i < M; i++) ext[k][i] = row[0];
            memcpy(ext[k] + M, row, (size_t)w * sizeof(pixel));
            for (int i = M + w; i < w + 2 * M; i++) ext[k][i] = row[w - 1];
        }
        for (int c = 0; c < cols; c++) {
            Ir[c] = ext[3][c];
            Hr[c] = FUNC(clip_sample)((t8[(size_t)3 * ps + c] + 32) >> 6);
            const int v = -ext[0][c] + 4 * ext[1][c] - 11 * ext[2][c] + 40 * ext[3][c]
                        + 40 * ext[4][c] - 11 * ext[5][c] + 4 * ext[6][c] - ext[7][c];
            Vr[c] = FUNC(clip_sample)((v + 32) >> 6);
            const int hv = TAPS(t8 + c, ps);
            HVr[c] = FUNC(clip_sample)((hv + 2048) >> 12);
        }
    }
#undef TAPS
#undef CLAMPX
#undef CLAMPY
}

/* 8x8 SAD between the source and a block of a half plane, and between the
 * source and the average of two such blocks - both in the plane's own
 * sample units. */
static inline uint32_t FUNC(sad8_ptr)(const pixel *a, int sa, const pixel *b, int sb)
{
#if BIT_DEPTH == 8 && (defined(__SSE2__) || defined(__x86_64__) || defined(_M_X64))
    __m128i acc = _mm_setzero_si128();
    for (int y = 0; y < 8; y += 2) {
        __m128i x0 = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)(a + y * sa)),
                                        _mm_loadl_epi64((const __m128i *)(a + (y + 1) * sa)));
        __m128i y0 = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)(b + y * sb)),
                                        _mm_loadl_epi64((const __m128i *)(b + (y + 1) * sb)));
        acc = _mm_add_epi32(acc, _mm_sad_epu8(x0, y0));
    }
    return (uint32_t)(_mm_cvtsi128_si32(acc) + _mm_extract_epi16(acc, 4));
#else
    uint32_t s = 0;
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++) {
            int d = (int)a[y * sa + x] - (int)b[y * sb + x];
            s += (uint32_t)(d < 0 ? -d : d);
        }
    return s;
#endif
}

static inline uint32_t FUNC(sad8_avg)(const pixel *a, int sa, const pixel *b, const pixel *c, int sbc)
{
#if BIT_DEPTH == 8 && (defined(__SSE2__) || defined(__x86_64__) || defined(_M_X64))
    __m128i acc = _mm_setzero_si128();
    for (int y = 0; y < 8; y += 2) {
        __m128i x0 = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)(a + y * sa)),
                                        _mm_loadl_epi64((const __m128i *)(a + (y + 1) * sa)));
        __m128i b0 = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)(b + y * sbc)),
                                        _mm_loadl_epi64((const __m128i *)(b + (y + 1) * sbc)));
        __m128i c0 = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)(c + y * sbc)),
                                        _mm_loadl_epi64((const __m128i *)(c + (y + 1) * sbc)));
        acc = _mm_add_epi32(acc, _mm_sad_epu8(x0, _mm_avg_epu8(b0, c0)));
    }
    return (uint32_t)(_mm_cvtsi128_si32(acc) + _mm_extract_epi16(acc, 4));
#else
    uint32_t s = 0;
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++) {
            int p = ((int)b[y * sbc + x] + (int)c[y * sbc + x] + 1) >> 1;
            int d = (int)a[y * sa + x] - p;
            s += (uint32_t)(d < 0 ? -d : d);
        }
    return s;
#endif
}

/* The half-grid block whose top-left sample sits at quarter position
 * (qx, qy) with both fractions even: plane by phase, offset by margin. */
static inline const pixel *FUNC(hpel_block)(const hevc_encoder_t *enc, int qx, int qy)
{
    const int phase = ((qx & 2) ? 1 : 0) + ((qy & 2) ? 2 : 0);
    const int x = (qx >> 2) + HPEL_MARGIN, y = (qy >> 2) + HPEL_MARGIN;
    return (const pixel *)enc->hpel[phase] + (size_t)y * enc->hpel_stride + x;
}

/* The approximate search SAD of the CU at quarter position (qx, qy) of the
 * previous picture, eight-bit units. Positions beyond the margin are
 * clamped: the search never needs them, a far merge candidate might. */
static uint32_t FUNC(search_sad)(const hevc_encoder_t *enc, const pixel *src, int qx, int qy)
{
    const int lo = -(HPEL_MARGIN - 2) * 4;
    const int hx = ((int)enc->coded_width - 8 + HPEL_MARGIN - 2) * 4;
    const int hy = ((int)enc->coded_height - 8 + HPEL_MARGIN - 2) * 4;
    qx = qx < lo ? lo : (qx > hx ? hx : qx);
    qy = qy < lo ? lo : (qy > hy ? hy : qy);
    const int ss = (int)enc->coded_width, ps = enc->hpel_stride;
    uint32_t s;
    if (!(qx & 1) && !(qy & 1)) {
        s = FUNC(sad8_ptr)(src, ss, FUNC(hpel_block)(enc, qx, qy), ps);
    } else if ((qx & 1) && (qy & 1)) {
        s = FUNC(sad8_avg)(src, ss, FUNC(hpel_block)(enc, qx - 1, qy - 1),
                           FUNC(hpel_block)(enc, qx + 1, qy + 1), ps);
    } else if (qx & 1) {
        s = FUNC(sad8_avg)(src, ss, FUNC(hpel_block)(enc, qx - 1, qy),
                           FUNC(hpel_block)(enc, qx + 1, qy), ps);
    } else {
        s = FUNC(sad8_avg)(src, ss, FUNC(hpel_block)(enc, qx, qy - 1),
                           FUNC(hpel_block)(enc, qx, qy + 1), ps);
    }
    return s >> (BIT_DEPTH - 8);
}

/* ---------------------------------------------------------------- intra */

/* An intra CU as it was decided and reconstructed: the four luma modes, the
 * quantized blocks and their flags. Its samples are already in the frame's
 * reconstruction. */
typedef struct {
    int pu_modes[4];
    int16_t luma_coeff[4][16];
    int cbf_luma[4];
    int16_t coeff_cb[16], coeff_cr[16];
    int cbf_cb, cbf_cr;
} FUNC(intra_cu_t);

/* Decide the four 4x4 luma modes and chroma, and reconstruct the CU into
 * the frame - PU by PU, since each one predicts from the one before. */
static void FUNC(intra_trial)(hevc_encoder_t *enc, int cu_x, int cu_y, int y_min,
                              FUNC(intra_cu_t) *r)
{
    const int qp = enc->qp;
    const uint32_t cw = enc->coded_width, ch = enc->coded_height;
    const uint32_t ccw = cw / 2, cch = ch / 2;
    const pixel *src_y = enc->src_y, *src_cb = enc->src_cb, *src_cr = enc->src_cr;
    pixel *recon_y = enc->recon_y, *recon_cb = enc->recon_cb, *recon_cr = enc->recon_cr;

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
        r->pu_modes[pu] = mode;

        int16_t residual[16];
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                residual[y * 4 + x] = (int16_t)(src_y[(py + y) * cw + (px + x)] - pred[y * 4 + x]);

        int16_t coeff[16];
        FUNC(hevc_transform_quant_4x4)(residual, qp + QP_BD_OFFSET, 1 /* DST for 4x4 luma intra */, coeff);
        memcpy(r->luma_coeff[pu], coeff, sizeof(coeff));
        r->cbf_luma[pu] = any_nonzero16(coeff);

        if (r->cbf_luma[pu]) {
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

    FUNC(hevc_transform_quant_4x4)(res_cb, cqp, 0, r->coeff_cb);
    FUNC(hevc_transform_quant_4x4)(res_cr, cqp, 0, r->coeff_cr);
    r->cbf_cb = any_nonzero16(r->coeff_cb);
    r->cbf_cr = any_nonzero16(r->coeff_cr);

    if (r->cbf_cb) {
        int16_t rres_cb[16];
        FUNC(hevc_dequant_itransform_4x4)(r->coeff_cb, cqp, 0, rres_cb);
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                recon_cb[(cy + y) * ccw + (cx + x)] = FUNC(clip_sample)(pred_cb[y * 4 + x] + rres_cb[y * 4 + x]);
    } else {
        for (int y = 0; y < 4; y++)
            memcpy(&recon_cb[(cy + y) * ccw + cx], &pred_cb[y * 4], 4 * sizeof(pixel));
    }

    if (r->cbf_cr) {
        int16_t rres_cr[16];
        FUNC(hevc_dequant_itransform_4x4)(r->coeff_cr, cqp, 0, rres_cr);
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                recon_cr[(cy + y) * ccw + (cx + x)] = FUNC(clip_sample)(pred_cr[y * 4 + x] + rres_cr[y * 4 + x]);
    } else {
        for (int y = 0; y < 4; y++)
            memcpy(&recon_cr[(cy + y) * ccw + cx], &pred_cr[y * 4], 4 * sizeof(pixel));
    }
}

/* The syntax of an intra CU whose decision intra_trial() made. */
static void FUNC(emit_intra)(hevc_encoder_t *enc, hevc_cabac_t *cab, int cu_x, int cu_y,
                             bool is_idr, int skip_ctx_inc, const FUNC(intra_cu_t) *r)
{
    if (!is_idr) {
        hevc_cabac_code_cu_skip_flag(cab, 0, skip_ctx_inc);
        hevc_cabac_code_pred_mode_flag(cab, 1 /* MODE_INTRA */);
    }

    hevc_cabac_code_part_mode_intra(cab, 0 /* PART_NxN */);

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
        pred_idx[pu] = hevc_cabac_code_intra_luma_flag(cab, r->pu_modes[pu], mpm[pu]);
    }
    for (int pu = 0; pu < 4; pu++)
        hevc_cabac_code_intra_luma_data(cab, r->pu_modes[pu], pred_idx[pu], mpm[pu]);

    /* Step 3: chroma mode (always DC; luma_mode_pu0 decides whether that's
     * signaled as index-3-of-candidate-list or as the derived/DM mode -
     * see hevc_cabac_code_intra_chroma_pred_mode()'s comment). */
    hevc_cabac_code_intra_chroma_pred_mode(cab, r->pu_modes[0]);

    /* Step 4: transform_tree - chroma cbf BITS first (trafoDepth=0, this
     * CU's root), then the 4 luma leaves' cbf+residual, then finally the
     * chroma RESIDUAL DATA (coded once per CU, after all 4 luma leaves -
     * this specific ordering, bits-before-luma but data-after-luma, is
     * exactly what ITU-T H.265's transform_tree()/transform_unit()
     * recursion produces for a CU whose chroma has already hit the 4x4
     * floor - see this file's top comment and x265's own
     * Entropy::encodeTransform(), which this encoder's fixed two-level
     * structure is a manually-unrolled special case of). */
    hevc_cabac_code_cbf_chroma(cab, r->cbf_cb, 0);
    hevc_cabac_code_cbf_chroma(cab, r->cbf_cr, 0);

    for (int pu = 0; pu < 4; pu++) {
        hevc_cabac_code_cbf_luma(cab, r->cbf_luma[pu], 1);
        if (r->cbf_luma[pu]) {
            int scan_idx = hevc_scan_idx_for_mode(r->pu_modes[pu]);
            hevc_cabac_code_residual_4x4(cab, r->luma_coeff[pu], 1, scan_idx);
        }
    }
    if (r->cbf_cb) hevc_cabac_code_residual_4x4(cab, r->coeff_cb, 0, 0 /* chroma always diagonal in 4:2:0 */);
    if (r->cbf_cr) hevc_cabac_code_residual_4x4(cab, r->coeff_cr, 0, 0);
}

/* ---------------------------------------------------------------- inter */

/* The prediction of the CU at (cu_x, cu_y) from the previous picture, moved
 * by `mv` (quarter samples; chroma takes the same vector in eighths of its
 * own samples, 8.5.3.2.10). Chroma only when asked for. */
static void FUNC(predict_cu)(const hevc_encoder_t *enc, int cu_x, int cu_y, hevc_mv_t mv,
                             pixel py[64], pixel pcb[16], pixel pcr[16])
{
    const int cw = (int)enc->coded_width, ch = (int)enc->coded_height;
    FUNC(hevc_mc_uni)(enc->prev_recon_y, cw, cw, ch, cu_x, cu_y, 8, 8, mv.x, mv.y, 0, py, 8);
    if (pcb) {
        FUNC(hevc_mc_uni)(enc->prev_recon_cb, cw / 2, cw / 2, ch / 2, cu_x / 2, cu_y / 2,
                          4, 4, mv.x, mv.y, 1, pcb, 4);
        FUNC(hevc_mc_uni)(enc->prev_recon_cr, cw / 2, cw / 2, ch / 2, cu_x / 2, cu_y / 2,
                          4, 4, mv.x, mv.y, 1, pcr, 4);
    }
}

static inline int64_t FUNC(sse)(const pixel *a, uint32_t sa, const pixel *b, uint32_t sb,
                                int w, int h)
{
    int64_t s = 0;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            int d = (int)a[y * sa + x] - (int)b[y * sb + x];
            s += d * d;
        }
    return s >> (2 * (BIT_DEPTH - 8));
}

/* An inter CU's residual, quantized, and what it reconstructs to. */
typedef struct {
    int tu8;             /* one 8x8 luma transform instead of four 4x4 */
    int16_t coeff_y[4][16];
    int16_t coeff_y8[64];
    int cbf_y8;
    int16_t coeff_cb[16], coeff_cr[16];
    int cbf_y[4], cbf_cb, cbf_cr;
    pixel rec_y[64], rec_cb[16], rec_cr[16];
    int64_t dist;        /* squared error against the source, eight-bit units */
    int bits;            /* coefficients, roughly */
} FUNC(inter_res_t);

static void FUNC(inter_residual)(const hevc_encoder_t *enc, int cu_x, int cu_y,
                                 const pixel py[64], const pixel pcb[16], const pixel pcr[16],
                                 int tu8, FUNC(inter_res_t) *r)
{
    const int qp = enc->qp;
    const uint32_t cw = enc->coded_width, ccw = cw / 2;
    const pixel *src_y = (const pixel *)enc->src_y + (size_t)cu_y * cw + cu_x;
    const int cx = cu_x / 2, cy = cu_y / 2;
    const pixel *src_cb = (const pixel *)enc->src_cb + (size_t)cy * ccw + cx;
    const pixel *src_cr = (const pixel *)enc->src_cr + (size_t)cy * ccw + cx;
    r->bits = 0;
    r->tu8 = tu8;
    r->cbf_y8 = 0;
    memset(r->cbf_y, 0, sizeof(r->cbf_y));

    if (tu8) {
        /* One 8x8 transform over the whole CU. */
        int16_t res[64], rres[64];
        for (int y = 0; y < 8; y++)
            for (int x = 0; x < 8; x++)
                res[y * 8 + x] = (int16_t)(src_y[y * cw + x] - py[y * 8 + x]);
        hevc_transform_quant_8x8(res, qp + QP_BD_OFFSET, BIT_DEPTH, r->coeff_y8);
        for (int i = 0; i < 64; i++) r->cbf_y8 |= r->coeff_y8[i] != 0;
        if (r->cbf_y8) {
            hevc_dequant_itransform_8x8(r->coeff_y8, qp + QP_BD_OFFSET, BIT_DEPTH, rres);
            for (int q = 0; q < 4; q++) {
                int16_t c16[16];
                for (int k = 0; k < 16; k++)
                    c16[k] = r->coeff_y8[((q >> 1) * 4 + (k >> 2)) * 8 + (q & 1) * 4 + (k & 3)];
                r->bits += coeff_bits(c16);
            }
        }
        for (int i = 0; i < 64; i++)
            r->rec_y[i] = r->cbf_y8 ? FUNC(clip_sample)(py[i] + rres[i]) : py[i];
    }

    /* Four 4x4 luma transforms, and inter luma takes the DCT, not the DST. */
    for (int pu = 0; !tu8 && pu < 4; pu++) {
        const int ox = pu_off_x[pu], oy = pu_off_y[pu];
        int16_t res[16], rres[16];
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                res[y * 4 + x] = (int16_t)(src_y[(oy + y) * cw + ox + x] - py[(oy + y) * 8 + ox + x]);
        FUNC(hevc_transform_quant_4x4)(res, qp + QP_BD_OFFSET, 0, r->coeff_y[pu]);
        r->cbf_y[pu] = any_nonzero16(r->coeff_y[pu]);
        if (r->cbf_y[pu]) {
            FUNC(hevc_dequant_itransform_4x4)(r->coeff_y[pu], qp + QP_BD_OFFSET, 0, rres);
            r->bits += coeff_bits(r->coeff_y[pu]);
        }
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++) {
                const int p = py[(oy + y) * 8 + ox + x];
                r->rec_y[(oy + y) * 8 + ox + x] = r->cbf_y[pu]
                    ? FUNC(clip_sample)(p + rres[y * 4 + x]) : (pixel)p;
            }
    }

    const int cqp = hevc_chroma_qp_from_luma(qp) + QP_BD_OFFSET;
    const pixel *cs[2] = { src_cb, src_cr };
    const pixel *cp[2] = { pcb, pcr };
    int16_t *cc[2] = { r->coeff_cb, r->coeff_cr };
    pixel *cr[2] = { r->rec_cb, r->rec_cr };
    int *cf[2] = { &r->cbf_cb, &r->cbf_cr };
    for (int c = 0; c < 2; c++) {
        int16_t res[16], rres[16];
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                res[y * 4 + x] = (int16_t)(cs[c][y * ccw + x] - cp[c][y * 4 + x]);
        FUNC(hevc_transform_quant_4x4)(res, cqp, 0, cc[c]);
        *cf[c] = any_nonzero16(cc[c]);
        if (*cf[c]) {
            FUNC(hevc_dequant_itransform_4x4)(cc[c], cqp, 0, rres);
            r->bits += coeff_bits(cc[c]);
        }
        for (int i = 0; i < 16; i++)
            cr[c][i] = *cf[c] ? FUNC(clip_sample)(cp[c][i] + rres[i]) : cp[c][i];
    }

    r->dist = FUNC(sse)(src_y, cw, r->rec_y, 8, 8, 8)
            + FUNC(sse)(src_cb, ccw, r->rec_cb, 4, 4, 4)
            + FUNC(sse)(src_cr, ccw, r->rec_cr, 4, 4, 4);
}

static inline int FUNC(inter_any_cbf)(const FUNC(inter_res_t) *r)
{
    return r->cbf_y[0] | r->cbf_y[1] | r->cbf_y[2] | r->cbf_y[3] | r->cbf_y8
         | r->cbf_cb | r->cbf_cr;
}

/* SAD of the CU at an integer displacement that keeps it inside the
 * picture, in eight-bit units. */
static inline uint32_t FUNC(sad_int)(const hevc_encoder_t *enc, int cu_x, int cu_y, int dx, int dy)
{
    return FUNC(compute_sad_8x8_luma)(enc->src_y, enc->prev_recon_y, enc->coded_width,
                                      cu_x, cu_y, dx, dy) >> (BIT_DEPTH - 8);
}

/* Motion search for one 8x8 CU against the previous picture.
 *
 * Every starting point - both AMVP predictors, the merge candidates, zero,
 * the GPU's vector for the CTU - is tried at the nearest whole sample; the
 * best is walked downhill one sample at a time, then refined to half and to
 * quarter samples. All of it on the search planes of build_hpel(). The cost
 * is the SAD plus lambda times the bits of the vector difference from the
 * cheaper of the two predictors. Returns the vector; `*mvp_idx`, `*cost_q8`
 * and `*sad` say which predictor, at what cost, with what SAD. */
static hevc_mv_t FUNC(motion_search)(const hevc_encoder_t *enc, int cu_x, int cu_y,
                                     const hevc_mv_t mvp[2], const hevc_mv_t *starts, int n_starts,
                                     int lambda_q8, int *mvp_idx, int64_t *cost_q8, uint32_t *sad_out)
{
    const int cw = (int)enc->coded_width, ch = (int)enc->coded_height;
    /* Up to eight samples past the picture: the search planes have a margin. */
    const int min_dx = -cu_x - 8, max_dx = cw - cu_x;
    const int min_dy = -cu_y - 8, max_dy = ch - cu_y;
    const int range = 64;
    const pixel *src = (const pixel *)enc->src_y + (size_t)cu_y * enc->coded_width + cu_x;
    const int qx0 = cu_x * 4, qy0 = cu_y * 4;

#define MV_BITS(qx, qy) ({ \
        int b0_ = mvd_bits((qx) - mvp[0].x) + mvd_bits((qy) - mvp[0].y); \
        int b1_ = mvd_bits((qx) - mvp[1].x) + mvd_bits((qy) - mvp[1].y); \
        b0_ <= b1_ ? b0_ : b1_; })
#define COST(qx, qy, sad) ((int64_t)(sad) * 256 + (int64_t)lambda_q8 * MV_BITS(qx, qy))

    int bx = 0, by = 0;
    int64_t best = INT64_MAX;
    uint32_t best_sad = 0;
    for (int i = 0; i < n_starts; i++) {
        int dx = (starts[i].x + 2) >> 2, dy = (starts[i].y + 2) >> 2;
        if (dx < min_dx) dx = min_dx;
        if (dx > max_dx) dx = max_dx;
        if (dy < min_dy) dy = min_dy;
        if (dy > max_dy) dy = max_dy;
        const uint32_t s = FUNC(search_sad)(enc, src, qx0 + dx * 4, qy0 + dy * 4);
        const int64_t c = COST(dx * 4, dy * 4, s);
        if (c < best) { best = c; best_sad = s; bx = dx; by = dy; }
    }

    /* Downhill, one sample at a time, within the search range. */
    const int cx0 = bx, cy0 = by;
    for (int step = 0; step < 32; step++) {
        static const int ddx[4] = { -1, 1, 0, 0 }, ddy[4] = { 0, 0, -1, 1 };
        int nbx = bx, nby = by;
        for (int k = 0; k < 4; k++) {
            const int dx = bx + ddx[k], dy = by + ddy[k];
            if (dx < min_dx || dx > max_dx || dy < min_dy || dy > max_dy) continue;
            if (dx < cx0 - range || dx > cx0 + range || dy < cy0 - range || dy > cy0 + range) continue;
            const uint32_t s = FUNC(search_sad)(enc, src, qx0 + dx * 4, qy0 + dy * 4);
            const int64_t c = COST(dx * 4, dy * 4, s);
            if (c < best) { best = c; best_sad = s; nbx = dx; nby = dy; }
        }
        if (nbx == bx && nby == by) break;
        bx = nbx;
        by = nby;
    }

    /* Half, then quarter samples around the best so far. */
    hevc_mv_t m = { (int16_t)(bx * 4), (int16_t)(by * 4) };
    for (int s = 2; s >= 1; s >>= 1) {
        const hevc_mv_t c0 = m;
        for (int k = 0; k < 8; k++) {
            static const int ox[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
            static const int oy[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
            const int tx = c0.x + ox[k] * s, ty = c0.y + oy[k] * s;
            const uint32_t sd = FUNC(search_sad)(enc, src, qx0 + tx, qy0 + ty);
            const int64_t c = COST(tx, ty, sd);
            if (c < best) { best = c; best_sad = sd; m.x = (int16_t)tx; m.y = (int16_t)ty; }
        }
    }

    const int b0 = mvd_bits(m.x - mvp[0].x) + mvd_bits(m.y - mvp[0].y);
    const int b1 = mvd_bits(m.x - mvp[1].x) + mvd_bits(m.y - mvp[1].y);
    *mvp_idx = b1 < b0 ? 1 : 0;
    *cost_q8 = best;
    *sad_out = best_sad;
#undef COST
#undef MV_BITS
    return m;
}

static void FUNC(write_back_inter)(hevc_encoder_t *enc, int cu_x, int cu_y,
                                   const pixel rec_y[64], const pixel rec_cb[16], const pixel rec_cr[16])
{
    const uint32_t cw = enc->coded_width, ccw = cw / 2;
    pixel *ry = (pixel *)enc->recon_y + (size_t)cu_y * cw + cu_x;
    pixel *rcb = (pixel *)enc->recon_cb + (size_t)(cu_y / 2) * ccw + cu_x / 2;
    pixel *rcr = (pixel *)enc->recon_cr + (size_t)(cu_y / 2) * ccw + cu_x / 2;
    for (int y = 0; y < 8; y++) memcpy(ry + (size_t)y * cw, rec_y + y * 8, 8 * sizeof(pixel));
    for (int y = 0; y < 4; y++) {
        memcpy(rcb + (size_t)y * ccw, rec_cb + y * 4, 4 * sizeof(pixel));
        memcpy(rcr + (size_t)y * ccw, rec_cr + y * 4, 4 * sizeof(pixel));
    }
}

/* The transform tree of an inter CU: the 8x8 CU splits down to four 4x4
 * luma blocks (the maximum transform size here), chroma flags at the root,
 * chroma data after the fourth luma block. Luma blocks of an inter CU are
 * always scanned diagonally. */
static void FUNC(emit_inter_residual)(const hevc_encoder_t *enc, hevc_cabac_t *cab,
                                      const FUNC(inter_res_t) *r)
{
    /* With 8x8 transforms allowed, the CU's transform tree says whether it
     * splits. */
    if (enc->tu8) hevc_cabac_code_split_transform_flag(cab, !r->tu8, 3);
    if (r->tu8) {
        hevc_cabac_code_cbf_chroma(cab, r->cbf_cb, 0);
        hevc_cabac_code_cbf_chroma(cab, r->cbf_cr, 0);
        /* ⚠️ 7.3.8.8: at the root of an inter CU with neither chroma flag
         * set, cbf_luma is not coded - it is 1. The callers never get here
         * with nothing to code. */
        if (r->cbf_cb || r->cbf_cr) hevc_cabac_code_cbf_luma(cab, r->cbf_y8, 0);
        if (r->cbf_y8) hevc_cabac_code_residual_8x8(cab, r->coeff_y8, 1);
        if (r->cbf_cb) hevc_cabac_code_residual_4x4(cab, r->coeff_cb, 0, 0);
        if (r->cbf_cr) hevc_cabac_code_residual_4x4(cab, r->coeff_cr, 0, 0);
        return;
    }
    hevc_cabac_code_cbf_chroma(cab, r->cbf_cb, 0);
    hevc_cabac_code_cbf_chroma(cab, r->cbf_cr, 0);
    for (int pu = 0; pu < 4; pu++) {
        hevc_cabac_code_cbf_luma(cab, r->cbf_y[pu], 1);
        if (r->cbf_y[pu]) hevc_cabac_code_residual_4x4(cab, r->coeff_y[pu], 1, 0);
    }
    if (r->cbf_cb) hevc_cabac_code_residual_4x4(cab, r->coeff_cb, 0, 0);
    if (r->cbf_cr) hevc_cabac_code_residual_4x4(cab, r->coeff_cr, 0, 0);
}

/* ------------------------------------------------------------------- CU */

/* What decide_cu() chose for one 8x8 CU of a P picture, kept so that its
 * syntax can be written once the CTU has been decided as a whole. */
typedef struct {
    int kind;                 /* CU_SKIP, CU_MERGE, CU_AMVP, CU_INTRA */
    int merge_idx;
    int mvp_idx;
    hevc_mv_t mvd;
    int64_t j;                /* D * 256 + lambda * R */
    FUNC(inter_res_t) res;    /* CU_MERGE, CU_AMVP; for CU_SKIP only rec_* */
    FUNC(intra_cu_t) intra;   /* CU_INTRA */
} FUNC(cu_decision_t);

static void FUNC(set_cu_maps)(hevc_encoder_t *enc, int cu_x, int cu_y, int size_cu,
                              int skip, int inter, hevc_mv_t mv, int depth)
{
    const uint32_t stride = enc->width_ctu * 2;
    const int cux = cu_x / HEVC_CU_SIZE, cuy = cu_y / HEVC_CU_SIZE;
    for (int dy = 0; dy < size_cu; dy++)
        for (int dx = 0; dx < size_cu; dx++) {
            const uint32_t i = (uint32_t)(cuy + dy) * stride + (uint32_t)(cux + dx);
            enc->cu_skip_map[i] = (uint8_t)skip;
            enc->cu_is_inter[i] = (uint8_t)inter;
            enc->mv_x_map[i] = inter ? mv.x : 0;
            enc->mv_y_map[i] = inter ? mv.y : 0;
            enc->cu_depth[i] = (uint8_t)depth;
        }
    if (inter) {
        for (int y = 0; y < size_cu * 2; y++)
            for (int x = 0; x < size_cu * 2; x++)
                enc->luma_mode_map[(cu_y / 4 + y) * enc->mode_map_stride + (cu_x / 4 + x)] = HEVC_MODE_DC;
    }
}

/* cu_skip_flag's context: whether the CUs to the left and above were
 * skipped (9.3.4.2.2). */
static inline int FUNC(skip_ctx)(const hevc_encoder_t *enc, int cu_x, int cu_y, int y_min)
{
    const uint32_t stride = enc->width_ctu * 2;
    const uint32_t i = (uint32_t)(cu_y / HEVC_CU_SIZE) * stride + (uint32_t)(cu_x / HEVC_CU_SIZE);
    return (cu_x > 0 && enc->cu_skip_map[i - 1] ? 1 : 0)
         + (cu_y > y_min && enc->cu_skip_map[i - stride] ? 1 : 0);
}

static void FUNC(emit_cu)(hevc_encoder_t *enc, hevc_cabac_t *cab, int cu_x, int cu_y, int y_min,
                          const FUNC(cu_decision_t) *d)
{
    const int ctx = FUNC(skip_ctx)(enc, cu_x, cu_y, y_min);
    switch (d->kind) {
    case CU_SKIP:
        hevc_cabac_code_cu_skip_flag(cab, 1, ctx);
        hevc_cabac_code_merge_idx(cab, d->merge_idx);
        return;
    case CU_INTRA:
        FUNC(emit_intra)(enc, cab, cu_x, cu_y, false, ctx, &d->intra);
        return;
    default:
        break;
    }
    hevc_cabac_code_cu_skip_flag(cab, 0, ctx);
    hevc_cabac_code_pred_mode_flag(cab, 0 /* MODE_INTER */);
    hevc_cabac_code_part_mode_intra(cab, 1 /* PART_2Nx2N: part_mode's first bin, one context */);
    if (d->kind == CU_MERGE) {
        hevc_cabac_code_merge_flag(cab, 1);
        hevc_cabac_code_merge_idx(cab, d->merge_idx);
        /* rqt_root_cbf is not coded for a 2Nx2N merge: it is 1, and the
         * residual is there - an empty one would have been a skip. */
        FUNC(emit_inter_residual)(enc, cab, &d->res);
    } else {
        hevc_cabac_code_merge_flag(cab, 0);
        hevc_cabac_code_mvd(cab, d->mvd.x, d->mvd.y);
        hevc_cabac_code_mvp_idx(cab, d->mvp_idx);
        const int root = FUNC(inter_any_cbf)(&d->res);
        hevc_cabac_code_rqt_root_cbf(cab, root);
        if (root) FUNC(emit_inter_residual)(enc, cab, &d->res);
    }
}

/* What a candidate costs: its syntax run through a copy of the estimation
 * chain, the bits turned into the distortion's units with lambda. `after`
 * keeps the contexts it leaves behind, for when it is chosen. */
static int64_t FUNC(rd_cost)(hevc_encoder_t *enc, const hevc_cabac_t *chain, int cu_x, int cu_y,
                             int y_min, const FUNC(cu_decision_t) *d, int64_t dist,
                             hevc_cabac_t *after)
{
    hevc_cabac_estimator(after, chain);
    FUNC(emit_cu)(enc, after, cu_x, cu_y, y_min, d);
    return dist * 256 + ((enc->lambda_sse_q8 * (int64_t)after->est_bits) >> 15);
}

/* Squared error of a prediction alone, luma and chroma, eight-bit units. */
static int64_t FUNC(pred_dist)(const hevc_encoder_t *enc, int cu_x, int cu_y,
                               const pixel py[64], const pixel pcb[16], const pixel pcr[16])
{
    const uint32_t cw = enc->coded_width, ccw = cw / 2;
    const size_t co = (size_t)(cu_y / 2) * ccw + cu_x / 2;
    return FUNC(sse)((const pixel *)enc->src_y + (size_t)cu_y * cw + cu_x, cw, py, 8, 8, 8)
         + FUNC(sse)((const pixel *)enc->src_cb + co, ccw, pcb, 4, 4, 4)
         + FUNC(sse)((const pixel *)enc->src_cr + co, ccw, pcr, 4, 4, 4);
}

/* Decide one 8x8 CU of a P picture by rate and distortion, and reconstruct
 * it into the frame; the syntax is left for emit_cu(). Every candidate's
 * bits are counted by running its syntax through `chain`, which then moves
 * on with the contexts the chosen one leaves. */
static void FUNC(decide_cu)(hevc_encoder_t *enc, hevc_cabac_t *chain, int cu_x, int cu_y, int y_min,
                            FUNC(cu_decision_t) *d, uint32_t *sad_out)
{
    const int cux = cu_x / HEVC_CU_SIZE, cuy = cu_y / HEVC_CU_SIZE;
    const int lsad = enc->lambda_sad_q8;
    const pixel *src = (const pixel *)enc->src_y + (size_t)cu_y * enc->coded_width + cu_x;
    const uint32_t cw = enc->coded_width, ccw = cw / 2;
    const size_t co = (size_t)(cu_y / 2) * ccw + cu_x / 2;

    /* The best merge candidate, by SAD on the search planes. */
    hevc_mv_t cand[5];
    derive_merge_candidates(y_min / HEVC_CU_SIZE, enc, cux, cuy, cand);
    int best_merge = 0;
    int64_t best_merge_cost = INT64_MAX;
    for (int i = 0; i < 5; i++) {
        bool dup = false;
        for (int p = 0; p < i; p++)
            if (cand[p].x == cand[i].x && cand[p].y == cand[i].y) { dup = true; break; }
        if (dup) continue;
        const int64_t c = (int64_t)FUNC(search_sad)(enc, src, cu_x * 4 + cand[i].x,
                                                    cu_y * 4 + cand[i].y) * 256
                        + (int64_t)lsad * (i + 1);
        if (c < best_merge_cost) { best_merge_cost = c; best_merge = i; }
    }
    *sad_out += (uint32_t)(best_merge_cost >> 8);

    FUNC(cu_decision_t) cd;
    hevc_cabac_t after, best_after;
    int64_t best_j = INT64_MAX;
    hevc_mv_t best_mv = cand[best_merge];

#define TRY(kind_, dist_, mv_) do {                                              \
        cd.kind = (kind_);                                                         \
        const int64_t j_ = FUNC(rd_cost)(enc, chain, cu_x, cu_y, y_min, &cd, (dist_), &after); \
        if (j_ < best_j) { best_j = j_; *d = cd; d->j = j_; best_after = after; best_mv = (mv_); } \
    } while (0)

    /* Skip: the prediction alone, whatever the residual would have been. */
    pixel py[64], pcb[16], pcr[16];
    FUNC(predict_cu)(enc, cu_x, cu_y, cand[best_merge], py, pcb, pcr);
    cd.merge_idx = best_merge;
    memcpy(cd.res.rec_y, py, sizeof(py));
    memcpy(cd.res.rec_cb, pcb, sizeof(pcb));
    memcpy(cd.res.rec_cr, pcr, sizeof(pcr));
    TRY(CU_SKIP, FUNC(pred_dist)(enc, cu_x, cu_y, py, pcb, pcr), cand[best_merge]);

    /* Merge with its residual - unless there is none, which is the skip -
     * as four 4x4 transforms and as one 8x8. */
    {
        FUNC(inter_residual)(enc, cu_x, cu_y, py, pcb, pcr, 0, &cd.res);
        if (enc->tu8) {
            FUNC(inter_res_t) r8;
            FUNC(inter_residual)(enc, cu_x, cu_y, py, pcb, pcr, 1, &r8);
            if (r8.dist * 256 + enc->lambda_sse_q8 * r8.bits < cd.res.dist * 256 + enc->lambda_sse_q8 * cd.res.bits)
                cd.res = r8;
        }
        if (FUNC(inter_any_cbf)(&cd.res))
            TRY(CU_MERGE, cd.res.dist, cand[best_merge]);
    }

    /* A searched vector, coded against the better AMVP predictor: with its
     * residual, and without it. */
    hevc_mv_t mvp[2];
    derive_amvp_candidates(y_min / HEVC_CU_SIZE, enc, cux, cuy, mvp);
    hevc_mv_t starts[10];
    int n_starts = 0;
    starts[n_starts++] = mvp[0];
    starts[n_starts++] = mvp[1];
    for (int i = 0; i < 5; i++) starts[n_starts++] = cand[i];
    starts[n_starts].x = 0; starts[n_starts].y = 0; n_starts++;
    {
        const uint32_t ctu = ((uint32_t)cuy / 2) * enc->width_ctu + ((uint32_t)cux / 2);
        if (enc->num_gpu_mvs > 0 && ctu < enc->num_gpu_mvs) {
            starts[n_starts].x = (int16_t)enc->gpu_mvs[ctu].mvx;
            starts[n_starts].y = (int16_t)enc->gpu_mvs[ctu].mvy;
            n_starts++;
        }
    }
    int mvp_idx = 0;
    int64_t me_cost;
    uint32_t me_sad;
    const hevc_mv_t mv = FUNC(motion_search)(enc, cu_x, cu_y, mvp, starts, n_starts,
                                             lsad, &mvp_idx, &me_cost, &me_sad);
    if (mv.x != cand[best_merge].x || mv.y != cand[best_merge].y) {
        FUNC(predict_cu)(enc, cu_x, cu_y, mv, py, pcb, pcr);
        cd.mvp_idx = mvp_idx;
        cd.mvd.x = (int16_t)(mv.x - mvp[mvp_idx].x);
        cd.mvd.y = (int16_t)(mv.y - mvp[mvp_idx].y);
        bool coded = false;
        {
            FUNC(inter_residual)(enc, cu_x, cu_y, py, pcb, pcr, 0, &cd.res);
            if (enc->tu8) {
                FUNC(inter_res_t) r8;
                FUNC(inter_residual)(enc, cu_x, cu_y, py, pcb, pcr, 1, &r8);
                if (r8.dist * 256 + enc->lambda_sse_q8 * r8.bits < cd.res.dist * 256 + enc->lambda_sse_q8 * cd.res.bits)
                    cd.res = r8;
            }
            coded = FUNC(inter_any_cbf)(&cd.res) != 0;
            TRY(CU_AMVP, cd.res.dist, mv);
        }
        if (coded) {
            /* The same vector with no residual at all. */
            memset(cd.res.cbf_y, 0, sizeof(cd.res.cbf_y));
            cd.res.cbf_y8 = 0;
            cd.res.cbf_cb = cd.res.cbf_cr = 0;
            memcpy(cd.res.rec_y, py, sizeof(py));
            memcpy(cd.res.rec_cb, pcb, sizeof(pcb));
            memcpy(cd.res.rec_cr, pcr, sizeof(pcr));
            TRY(CU_AMVP, FUNC(pred_dist)(enc, cu_x, cu_y, py, pcb, pcr), mv);
        }
    }

    /* Intra, tried for real: it has to be reconstructed to be judged.
     * ⚠️ Every time. Skipping it when the motion search looked good enough
     * - better than a flat block, or than an 8x8 intra guess - saved 7 to
     * 15% of the time and cost 4 to 16% more bits on ducks_take_off, where
     * the water is exactly what 4x4 intra wins. */
    FUNC(intra_trial)(enc, cu_x, cu_y, y_min, &cd.intra);
    const int64_t dist_intra =
          FUNC(sse)(src, cw, (const pixel *)enc->recon_y + (size_t)cu_y * cw + cu_x, cw, 8, 8)
        + FUNC(sse)((const pixel *)enc->src_cb + co, ccw, (const pixel *)enc->recon_cb + co, ccw, 4, 4)
        + FUNC(sse)((const pixel *)enc->src_cr + co, ccw, (const pixel *)enc->recon_cr + co, ccw, 4, 4);
    {
        const hevc_mv_t zero = { 0, 0 };
        TRY(CU_INTRA, dist_intra, zero);
    }
#undef TRY

    *chain = best_after;
    switch (d->kind) {
    case CU_INTRA: {
        const hevc_mv_t zero = { 0, 0 };
        FUNC(set_cu_maps)(enc, cu_x, cu_y, 1, 0, 0, zero, 1);
        break;   /* its samples are already in the frame */
    }
    case CU_SKIP:
        FUNC(set_cu_maps)(enc, cu_x, cu_y, 1, 1, 1, best_mv, 1);
        FUNC(write_back_inter)(enc, cu_x, cu_y, d->res.rec_y, d->res.rec_cb, d->res.rec_cr);
        break;
    default:
        FUNC(set_cu_maps)(enc, cu_x, cu_y, 1, 0, 1, best_mv, 1);
        FUNC(write_back_inter)(enc, cu_x, cu_y, d->res.rec_y, d->res.rec_cb, d->res.rec_cr);
        break;
    }
}

/* The whole CTU as one 16x16 skip: the best merge candidate for a 16x16
 * prediction unit, no residual. Returns its distortion; the candidate and
 * the prediction come back through the pointers. */
static int64_t FUNC(skip16_dist)(hevc_encoder_t *enc, int x, int y, int y_min,
                                 int *merge_idx, hevc_mv_t *mv,
                                 pixel py[256], pixel pcb[64], pixel pcr[64])
{
    const int cux = x / HEVC_CU_SIZE, cuy = y / HEVC_CU_SIZE;
    const int cw = (int)enc->coded_width, ch = (int)enc->coded_height;
    const pixel *src = (const pixel *)enc->src_y + (size_t)y * cw + x;
    hevc_mv_t cand[5];
    derive_merge_candidates_n(y_min / HEVC_CU_SIZE, enc, cux, cuy, 2, cand);
    int best = 0;
    int64_t best_c = INT64_MAX;
    for (int i = 0; i < 5; i++) {
        bool dup = false;
        for (int p = 0; p < i; p++)
            if (cand[p].x == cand[i].x && cand[p].y == cand[i].y) { dup = true; break; }
        if (dup) continue;
        int64_t s = 0;
        for (int q = 0; q < 4; q++)
            s += FUNC(search_sad)(enc, src + (q >> 1) * 8 * cw + (q & 1) * 8,
                                  (x + (q & 1) * 8) * 4 + cand[i].x, (y + (q >> 1) * 8) * 4 + cand[i].y);
        const int64_t c = s * 256 + (int64_t)enc->lambda_sad_q8 * (i + 1);
        if (c < best_c) { best_c = c; best = i; }
    }
    *merge_idx = best;
    *mv = cand[best];
    FUNC(hevc_mc_uni)(enc->prev_recon_y, cw, cw, ch, x, y, 16, 16, mv->x, mv->y, 0, py, 16);
    FUNC(hevc_mc_uni)(enc->prev_recon_cb, cw / 2, cw / 2, ch / 2, x / 2, y / 2, 8, 8, mv->x, mv->y, 1, pcb, 8);
    FUNC(hevc_mc_uni)(enc->prev_recon_cr, cw / 2, cw / 2, ch / 2, x / 2, y / 2, 8, 8, mv->x, mv->y, 1, pcr, 8);
    const size_t co = (size_t)(y / 2) * (cw / 2) + x / 2;
    return FUNC(sse)(src, cw, py, 16, 16, 16)
         + FUNC(sse)((const pixel *)enc->src_cb + co, cw / 2, pcb, 8, 8, 8)
         + FUNC(sse)((const pixel *)enc->src_cr + co, cw / 2, pcr, 8, 8, 8);
}

static void FUNC(encode_ctu)(hevc_encoder_t *enc, hevc_cabac_t *cab, int ctu_col, int ctu_row, bool is_idr, int y_min, uint32_t *sad_out) {
    int ctu_x = ctu_col * HEVC_CTU_SIZE, ctu_y = ctu_row * HEVC_CTU_SIZE;
    static const int cu_off_x[4] = { 0, 8, 0, 8 };
    static const int cu_off_y[4] = { 0, 0, 8, 8 };

    /* split_cu_flag's context: whether the CUs to the left and above sit
     * deeper in their quadtree than this one does (9.3.4.2.2). */
    const uint32_t stride = enc->width_ctu * 2;
    const uint32_t c0 = (uint32_t)(ctu_y / HEVC_CU_SIZE) * stride + (uint32_t)(ctu_x / HEVC_CU_SIZE);
    const int split_ctx = (ctu_col > 0 && enc->cu_depth[c0 - 1] > 0 ? 1 : 0)
                        + (ctu_y > y_min && enc->cu_depth[c0 - stride] > 0 ? 1 : 0);

    if (is_idr || !enc->has_ref) {
        hevc_cabac_code_split_cu_flag(cab, 1, split_ctx);
        for (int i = 0; i < 4; i++) {
            const int x = ctu_x + cu_off_x[i], y = ctu_y + cu_off_y[i];
            FUNC(intra_cu_t) intra;
            FUNC(intra_trial)(enc, x, y, y_min, &intra);
            const hevc_mv_t zero = { 0, 0 };
            FUNC(set_cu_maps)(enc, x, y, 1, 0, 0, zero, 1);
            FUNC(emit_intra)(enc, cab, x, y, is_idr, FUNC(skip_ctx)(enc, x, y, y_min), &intra);
        }
        return;
    }

    /* The whole CTU as one skip, costed first. */
    pixel py16[256], pcb16[64], pcr16[64];
    int idx16 = 0;
    hevc_mv_t mv16 = { 0, 0 };
    int64_t j16 = INT64_MAX;
    hevc_cabac_t chain;
    hevc_cabac_estimator(&chain, cab);
    hevc_cabac_code_split_cu_flag(&chain, 1, split_ctx);
    const uint32_t split_bits = chain.est_bits;
    if (enc->cu16) {
        const int64_t dist = FUNC(skip16_dist)(enc, ctu_x, ctu_y, y_min, &idx16, &mv16, py16, pcb16, pcr16);
        hevc_cabac_t e16;
        hevc_cabac_estimator(&e16, cab);
        hevc_cabac_code_split_cu_flag(&e16, 0, split_ctx);
        hevc_cabac_code_cu_skip_flag(&e16, 1, FUNC(skip_ctx)(enc, ctu_x, ctu_y, y_min));
        hevc_cabac_code_merge_idx(&e16, idx16);
        j16 = dist * 256 + ((enc->lambda_sse_q8 * (int64_t)e16.est_bits) >> 15);
    }

    /* Four 8x8 CUs can never cost less than their split flag and about a
     * bit each. A skip already under that is taken without trying them -
     * most of a still picture, and none of the work. */
    const int64_t floor_split = (enc->lambda_sse_q8 * (int64_t)(split_bits + 4 * 32768)) >> 15;
    FUNC(cu_decision_t) d[4];
    uint32_t sad8 = 0;
    int64_t j_split = INT64_MAX;
    if (j16 >= floor_split) {
        /* Each CU decided against the contexts the ones before it leave. */
        j_split = (enc->lambda_sse_q8 * (int64_t)split_bits) >> 15;
        for (int i = 0; i < 4; i++) {
            FUNC(decide_cu)(enc, &chain, ctu_x + cu_off_x[i], ctu_y + cu_off_y[i], y_min, &d[i], &sad8);
            j_split += d[i].j;
        }
    }

    if (j16 < j_split) {
        const uint32_t cw = enc->coded_width, ccw = cw / 2;
        pixel *ry = (pixel *)enc->recon_y + (size_t)ctu_y * cw + ctu_x;
        pixel *rcb = (pixel *)enc->recon_cb + (size_t)(ctu_y / 2) * ccw + ctu_x / 2;
        pixel *rcr = (pixel *)enc->recon_cr + (size_t)(ctu_y / 2) * ccw + ctu_x / 2;
        for (int y = 0; y < 16; y++) memcpy(ry + (size_t)y * cw, py16 + y * 16, 16 * sizeof(pixel));
        for (int y = 0; y < 8; y++) {
            memcpy(rcb + (size_t)y * ccw, pcb16 + y * 8, 8 * sizeof(pixel));
            memcpy(rcr + (size_t)y * ccw, pcr16 + y * 8, 8 * sizeof(pixel));
        }
        FUNC(set_cu_maps)(enc, ctu_x, ctu_y, 2, 1, 1, mv16, 0);
        hevc_cabac_code_split_cu_flag(cab, 0, split_ctx);
        hevc_cabac_code_cu_skip_flag(cab, 1, FUNC(skip_ctx)(enc, ctu_x, ctu_y, y_min));
        hevc_cabac_code_merge_idx(cab, idx16);
        *sad_out += sad8;
        return;
    }

    hevc_cabac_code_split_cu_flag(cab, 1, split_ctx);
    for (int i = 0; i < 4; i++)
        FUNC(emit_cu)(enc, cab, ctu_x + cu_off_x[i], ctu_y + cu_off_y[i], y_min, &d[i]);
    *sad_out += sad8;
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
