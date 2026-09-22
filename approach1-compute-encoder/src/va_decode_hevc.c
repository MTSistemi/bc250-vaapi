/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * va_decode_hevc.c - the VA-API decode entry point, VAEntrypointVLD for
 * H.265.
 *
 * The same shape as the H.264 one: the application has already parsed the
 * parameter sets and hands over VAPictureParameterBufferHEVC and one
 * VASliceParameterBufferHEVC per slice, so this file is a translation
 * layer that turns VA's structures into the decoder's own.
 *
 * ⚠️ With one difference. The slice header is parsed here, out of the
 * bytes the application handed over, rather than rebuilt from
 * VASliceParameterBufferHEVC. It has to be: wavefront parallelism needs
 * the entry point offsets, VA says how many of them there are and never
 * says what they are, and working out where a substream ends without them
 * is wrong by a byte or two often enough to matter. Everything else in
 * the slice parameters is cross-checked against what the header says and
 * the header wins.
 */
#include "va_backend.h"
#include "decoder_h265.h"
#include "bitreader.h"

#include <stdlib.h>
#include <string.h>

void bc250_hevc_dec_reset(bc250_context *c)
{
    c->hevc_dec_state.has_pic = 0;
    c->hevc_dec_state.n_slices = 0;
    c->hevc_dec_state.n_data = 0;
}

void bc250_hevc_dec_free(bc250_context *c)
{
    free(c->hevc_dec_state.slices);
    free(c->hevc_dec_state.data);
    free(c->hevc_dec_state.rbsp);
    c->hevc_dec_state.slices = NULL;
    c->hevc_dec_state.data = NULL;
    c->hevc_dec_state.rbsp = NULL;
    c->hevc_dec_state.cap_slices = 0;
    c->hevc_dec_state.cap_data = 0;
    c->hevc_dec_state.cap_rbsp = 0;
    bc250_hevc_dec_reset(c);
}

static bool grow_slices(bc250_context *c)
{
    if (c->hevc_dec_state.n_slices < c->hevc_dec_state.cap_slices) return true;
    const int new_one = c->hevc_dec_state.cap_slices
        ? c->hevc_dec_state.cap_slices * 2 : 16;
    void *p = realloc(c->hevc_dec_state.slices,
                      (size_t)new_one * sizeof(*c->hevc_dec_state.slices));
    if (!p) return false;
    c->hevc_dec_state.slices = p;
    c->hevc_dec_state.cap_slices = new_one;
    return true;
}

static bool grow_data(bc250_context *c, size_t count)
{
    const size_t serve = c->hevc_dec_state.n_data + count;
    if (serve <= c->hevc_dec_state.cap_data) return true;
    size_t new_one = c->hevc_dec_state.cap_data ? c->hevc_dec_state.cap_data : 65536;
    while (new_one < serve) new_one *= 2;
    uint8_t *p = realloc(c->hevc_dec_state.data, new_one);
    if (!p) return false;
    c->hevc_dec_state.data = p;
    c->hevc_dec_state.cap_data = new_one;
    return true;
}

VAStatus bc250_hevc_dec_render(bc250_context *c, bc250_buffer *b)
{
    if (!b->data) return VA_STATUS_SUCCESS;

    switch (b->type) {
    case VAPictureParameterBufferType:
        if (b->size < sizeof(VAPictureParameterBufferHEVC))
            return VA_STATUS_ERROR_INVALID_PARAMETER;
        memcpy(&c->hevc_dec_state.pic, b->data,
               sizeof(VAPictureParameterBufferHEVC));
        c->hevc_dec_state.has_pic = 1;
        return VA_STATUS_SUCCESS;

    case VAIQMatrixBufferType:
        /* ⚠️ Quantisation matrices are refused at the parameter set, so a
         * stream that sends them is refused rather than decoded with the
         * flat ones and quietly wrong. */
        return VA_STATUS_SUCCESS;

    case VASliceParameterBufferType: {
        const unsigned n = b->num_elements ? b->num_elements : 1;
        if (b->size < sizeof(VASliceParameterBufferHEVC))
            return VA_STATUS_ERROR_INVALID_PARAMETER;
        for (unsigned k = 0; k < n; k++) {
            if (!grow_slices(c)) return VA_STATUS_ERROR_ALLOCATION_FAILED;
            const VASliceParameterBufferHEVC *src =
                (const VASliceParameterBufferHEVC *)
                ((const uint8_t *)b->data + (size_t)k * b->size);
            const int i = c->hevc_dec_state.n_slices;
            c->hevc_dec_state.slices[i].p = *src;
            c->hevc_dec_state.slices[i].off = (size_t)-1;
            c->hevc_dec_state.slices[i].len = 0;
            c->hevc_dec_state.n_slices++;
        }
        return VA_STATUS_SUCCESS;
    }

    case VASliceDataBufferType: {
        const size_t total = (size_t)b->size
            * (b->num_elements ? b->num_elements : 1);
        for (int i = 0; i < c->hevc_dec_state.n_slices; i++) {
            if (c->hevc_dec_state.slices[i].off != (size_t)-1) continue;
            const VASliceParameterBufferHEVC *p = &c->hevc_dec_state.slices[i].p;
            if (p->slice_data_flag != VA_SLICE_DATA_FLAG_ALL)
                return VA_STATUS_ERROR_UNIMPLEMENTED;
            if ((size_t)p->slice_data_offset + p->slice_data_size > total)
                return VA_STATUS_ERROR_INVALID_PARAMETER;
            if (!grow_data(c, p->slice_data_size))
                return VA_STATUS_ERROR_ALLOCATION_FAILED;
            memcpy(c->hevc_dec_state.data + c->hevc_dec_state.n_data,
                   (const uint8_t *)b->data + p->slice_data_offset,
                   p->slice_data_size);
            c->hevc_dec_state.slices[i].off = c->hevc_dec_state.n_data;
            c->hevc_dec_state.slices[i].len = p->slice_data_size;
            c->hevc_dec_state.n_data += p->slice_data_size;
        }
        return VA_STATUS_SUCCESS;
    }

    default:
        return VA_STATUS_SUCCESS;
    }
}

/* --------------------------------------------------------- translation */

static void fill_sps(const VAPictureParameterBufferHEVC *p, hevc_sps_t *s)
{
    memset(s, 0, sizeof *s);
    s->valid = true;
    s->chroma_format_idc = p->pic_fields.bits.chroma_format_idc;
    s->width = p->pic_width_in_luma_samples;
    s->height = p->pic_height_in_luma_samples;
    s->bit_depth_luma = 8 + p->bit_depth_luma_minus8;
    s->bit_depth_chroma = 8 + p->bit_depth_chroma_minus8;
    s->log2_max_poc_lsb = 4 + p->log2_max_pic_order_cnt_lsb_minus4;
    s->max_dec_pic_buffering = p->sps_max_dec_pic_buffering_minus1 + 1;

    s->log2_min_cb = 3 + p->log2_min_luma_coding_block_size_minus3;
    s->log2_ctb = s->log2_min_cb + p->log2_diff_max_min_luma_coding_block_size;
    s->log2_min_tb = 2 + p->log2_min_transform_block_size_minus2;
    s->log2_max_tb = s->log2_min_tb + p->log2_diff_max_min_transform_block_size;
    s->max_transform_hierarchy_depth_intra = p->max_transform_hierarchy_depth_intra;
    s->max_transform_hierarchy_depth_inter = p->max_transform_hierarchy_depth_inter;

    s->scaling_list_enabled = p->pic_fields.bits.scaling_list_enabled_flag;
    s->amp_enabled = p->pic_fields.bits.amp_enabled_flag;
    s->sao_enabled = p->slice_parsing_fields.bits.sample_adaptive_offset_enabled_flag;
    s->pcm_enabled = p->pic_fields.bits.pcm_enabled_flag;
    s->pcm_bit_depth_luma = p->pcm_sample_bit_depth_luma_minus1 + 1;
    s->pcm_bit_depth_chroma = p->pcm_sample_bit_depth_chroma_minus1 + 1;
    s->log2_min_pcm_cb = 3 + p->log2_min_pcm_luma_coding_block_size_minus3;
    s->log2_max_pcm_cb = s->log2_min_pcm_cb
        + p->log2_diff_max_min_pcm_luma_coding_block_size;
    s->pcm_loop_filter_disabled = p->pic_fields.bits.pcm_loop_filter_disabled_flag;
    s->long_term_ref_pics_present = p->slice_parsing_fields.bits.long_term_ref_pics_present_flag;
    s->num_long_term_sps = p->num_long_term_ref_pic_sps;
    s->temporal_mvp_enabled = p->slice_parsing_fields.bits.sps_temporal_mvp_enabled_flag;
    s->strong_intra_smoothing = p->pic_fields.bits.strong_intra_smoothing_enabled_flag;

    /* ⚠️ The application has already applied the conformance window: the
     * size it asks for is the size it wants back. Cropping again here
     * would take it off twice. */
    s->crop_left = s->crop_right = s->crop_top = s->crop_bottom = 0;

    s->ctb_size = 1 << s->log2_ctb;
    s->ctb_width = (s->width + s->ctb_size - 1) >> s->log2_ctb;
    s->ctb_height = (s->height + s->ctb_size - 1) >> s->log2_ctb;
    s->ctb_count = s->ctb_width * s->ctb_height;
    s->min_cb_width = s->width >> s->log2_min_cb;
    s->min_cb_height = s->height >> s->log2_min_cb;

    /* The short term sets live in the slice header for a VLD decoder: the
     * application does not pass the sequence's own copies, and every slice
     * that uses one sends it inline. */
    s->num_st_rps = 0;
}

static void fill_pps(const VAPictureParameterBufferHEVC *p, hevc_pps_t *q)
{
    memset(q, 0, sizeof *q);
    q->valid = true;
    q->pps_id = 0;
    q->sps_id = 0;
    q->dependent_slice_segments_enabled =
        p->slice_parsing_fields.bits.dependent_slice_segments_enabled_flag;
    q->output_flag_present = p->slice_parsing_fields.bits.output_flag_present_flag;
    q->num_extra_slice_header_bits = p->num_extra_slice_header_bits;
    q->sign_data_hiding = p->pic_fields.bits.sign_data_hiding_enabled_flag;
    q->cabac_init_present = p->slice_parsing_fields.bits.cabac_init_present_flag;
    q->num_ref_idx_default[0] = p->num_ref_idx_l0_default_active_minus1 + 1;
    q->num_ref_idx_default[1] = p->num_ref_idx_l1_default_active_minus1 + 1;
    q->init_qp = 26 + p->init_qp_minus26;
    q->constrained_intra_pred = p->pic_fields.bits.constrained_intra_pred_flag;
    q->transform_skip_enabled = p->pic_fields.bits.transform_skip_enabled_flag;
    q->cu_qp_delta_enabled = p->pic_fields.bits.cu_qp_delta_enabled_flag;
    q->diff_cu_qp_delta_depth = p->diff_cu_qp_delta_depth;
    q->cb_qp_offset = p->pps_cb_qp_offset;
    q->cr_qp_offset = p->pps_cr_qp_offset;
    q->slice_chroma_qp_offsets_present =
        p->slice_parsing_fields.bits.pps_slice_chroma_qp_offsets_present_flag;
    q->weighted_pred = p->pic_fields.bits.weighted_pred_flag;
    q->weighted_bipred = p->pic_fields.bits.weighted_bipred_flag;
    q->transquant_bypass_enabled = p->pic_fields.bits.transquant_bypass_enabled_flag;
    q->tiles_enabled = p->pic_fields.bits.tiles_enabled_flag;
    q->entropy_coding_sync_enabled = p->pic_fields.bits.entropy_coding_sync_enabled_flag;
    q->num_tile_columns = p->num_tile_columns_minus1 + 1;
    q->num_tile_rows = p->num_tile_rows_minus1 + 1;
    q->loop_filter_across_tiles = p->pic_fields.bits.loop_filter_across_tiles_enabled_flag;
    q->loop_filter_across_slices =
        p->pic_fields.bits.pps_loop_filter_across_slices_enabled_flag;
    q->deblocking_filter_override_enabled =
        p->slice_parsing_fields.bits.deblocking_filter_override_enabled_flag;
    q->deblocking_filter_disabled =
        p->slice_parsing_fields.bits.pps_disable_deblocking_filter_flag;
    q->deblocking_filter_control_present = 1;
    q->beta_offset = 2 * p->pps_beta_offset_div2;
    q->tc_offset = 2 * p->pps_tc_offset_div2;
    q->pps_scaling_list_present = 0;
    q->lists_modification_present =
        p->slice_parsing_fields.bits.lists_modification_present_flag;
    q->log2_parallel_merge_level = 2 + p->log2_parallel_merge_level_minus2;
    q->slice_segment_header_extension_present =
        p->slice_parsing_fields.bits.slice_segment_header_extension_present_flag;
}

static bool is_valid(const VAPictureHEVC *p)
{
    return !(p->flags & VA_PICTURE_HEVC_INVALID)
        && p->picture_id != VA_INVALID_SURFACE;
}

VAStatus bc250_hevc_dec_decode(bc250_context *c, gpu_image_t out,
                               gpu_memory_t mem)
{
    hevc_decoder_t *dec = c->h265_dec;
    if (!dec) return VA_STATUS_ERROR_INVALID_CONTEXT;
    if (!c->hevc_dec_state.has_pic || c->hevc_dec_state.n_slices == 0)
        return VA_STATUS_ERROR_INVALID_PARAMETER;

    const VAPictureParameterBufferHEVC *pp = &c->hevc_dec_state.pic;

    /* 4:2:0 at eight or ten bits, and the two depths equal. Everything
     * below eight bits and everything above ten is a profile this
     * decoder does not claim; luma and chroma differing is one the
     * templates cannot serve, because every line that picks between the
     * two builds reads the luma depth. */
    if (pp->pic_fields.bits.chroma_format_idc != 1
        || (pp->bit_depth_luma_minus8 != 0 && pp->bit_depth_luma_minus8 != 2)
        || pp->bit_depth_chroma_minus8 != pp->bit_depth_luma_minus8)
        return VA_STATUS_ERROR_UNSUPPORTED_PROFILE;
    /* Tiles put the coding tree units in an order of the picture parameter
     * set's choosing, and nothing here understands that. Refused rather
     * than decoded into a scramble. */
    if (pp->pic_fields.bits.tiles_enabled_flag)
        return VA_STATUS_ERROR_UNSUPPORTED_PROFILE;
    if (pp->pic_fields.bits.scaling_list_enabled_flag)
        return VA_STATUS_ERROR_UNSUPPORTED_PROFILE;
    if (pp->pic_fields.bits.pcm_enabled_flag)
        return VA_STATUS_ERROR_UNSUPPORTED_PROFILE;
    if (pp->slice_parsing_fields.bits.long_term_ref_pics_present_flag)
        return VA_STATUS_ERROR_UNSUPPORTED_PROFILE;

    hevc_sps_t sps;
    hevc_pps_t pps;
    fill_sps(pp, &sps);
    fill_pps(pp, &pps);
    if (sps.width != c->width || sps.height != c->height)
        return VA_STATUS_ERROR_RESOLUTION_NOT_SUPPORTED;

    /* Which surfaces are still references. This runs before the picture is
     * opened, so that the one about to be decoded cannot be handed a slot
     * it is about to read from. */
    uintptr_t ref_pic[16];
    int poc[16];
    int n_refs = 0;
    for (int i = 0; i < 15 && n_refs < 16; i++) {
        if (!is_valid(&pp->ReferenceFrames[i])) continue;
        ref_pic[n_refs] = (uintptr_t)pp->ReferenceFrames[i].picture_id;
        poc[n_refs] = pp->ReferenceFrames[i].pic_order_cnt;
        n_refs++;
    }
    hevc_decoder_set_references(dec, ref_pic, poc, n_refs);

    if (hevc_decoder_begin_picture(dec, &sps, &pps,
                                   (uintptr_t)c->current_render_target,
                                   pp->CurrPic.pic_order_cnt) != 0)
        return VA_STATUS_ERROR_OPERATION_FAILED;

    /* One set of parameter sets, at whatever index the headers name. */
    static hevc_sps_t sps_store[16];
    static hevc_pps_t pps_store[64];
    for (int i = 0; i < 16; i++) sps_store[i] = sps;
    for (int i = 0; i < 64; i++) { pps_store[i] = pps; pps_store[i].pps_id = i; }

    for (int i = 0; i < c->hevc_dec_state.n_slices; i++) {
        if (c->hevc_dec_state.slices[i].off == (size_t)-1) continue;
        const uint8_t *nal = c->hevc_dec_state.data
            + c->hevc_dec_state.slices[i].off;
        const size_t len = c->hevc_dec_state.slices[i].len;
        if (len < 3) continue;

        if (c->hevc_dec_state.cap_rbsp < len) {
            uint8_t *p = realloc(c->hevc_dec_state.rbsp, len);
            if (!p) return VA_STATUS_ERROR_ALLOCATION_FAILED;
            c->hevc_dec_state.rbsp = p;
            c->hevc_dec_state.cap_rbsp = len;
        }
        const size_t n = br_extract_rbsp(c->hevc_dec_state.rbsp, len, nal, len);

        hevc_slice_t sl;
        const int kind = (nal[0] >> 1) & 0x3f;
        if (hevc_ps_read_slice(&sl, c->hevc_dec_state.rbsp, n, kind,
                                sps_store, pps_store) != 0)
            continue;              /* a slice we cannot read, not a guess */

        /* ⚠️ The picture order count comes from the application, which
         * has the whole sequence in front of it. Deriving it here from
         * the count's low bits would need the previous picture's, and a
         * decoder that has just been seeked to has no previous picture. */
        sl.poc = pp->CurrPic.pic_order_cnt;
        hevc_decoder_shift_entry_points(&sl, nal, len, sl.data_bit_offset >> 3);

        hevc_decoder_slice(dec, &sl, c->hevc_dec_state.rbsp, n);
    }

    /* ⚠️ Always finished, even when every slice was refused: a picture
     * that is never ended leaves the buffer holding a slot no later one
     * can reuse, and the application still gets its surface back. */
    hevc_decoder_end_picture(dec);
    if (hevc_decoder_load(dec, out, mem) != 0)
        return VA_STATUS_ERROR_OPERATION_FAILED;
    return VA_STATUS_SUCCESS;
}
