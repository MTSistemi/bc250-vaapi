/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * va_decode.c - the VA-API decode entry point, VAEntrypointVLD for H.264.
 *
 * Under VLD the application has already parsed the sequence, picture and
 * slice headers: it hands over VAPictureParameterBufferH264 and one
 * VASliceParameterBufferH264 per slice, and the driver's work starts at the
 * macroblock layer. So this file is a translation layer and nothing more -
 * it turns VA's structures into the decoder's own and maps surfaces onto
 * frame store slots.
 *
 * ⚠️ The slices are held rather than decoded as they arrive. Decoding here
 * is CPU work measured in milliseconds, and vaRenderPicture runs with the
 * driver lock held; doing it there would serialise every other thread in
 * the application behind it. Everything is copied and the whole picture is
 * decoded from vaEndPicture with the lock dropped.
 */
#include "va_backend.h"

#include <stdlib.h>
#include <string.h>

/* PicOrderCnt of a frame, clause 8.2.1: the smaller of its two field
 * counts. This decoder is progressive-only, so the two are equal in every
 * stream it accepts, but taking the minimum is what the clause says. */
static int poc_di(const VAPictureH264 *p)
{
    if (p->flags & VA_PICTURE_H264_INVALID) return 0;
    return p->TopFieldOrderCnt < p->BottomFieldOrderCnt
         ? p->TopFieldOrderCnt : p->BottomFieldOrderCnt;
}

static bool is_valid(const VAPictureH264 *p)
{
    return !(p->flags & VA_PICTURE_H264_INVALID)
        && p->picture_id != VA_INVALID_SURFACE;
}

void bc250_dec_reset(bc250_context *c)
{
    c->dec_state.has_pic = 0;
    c->dec_state.has_iq = 0;
    c->dec_state.n_slices = 0;
    c->dec_state.n_data = 0;
}

void bc250_dec_free(bc250_context *c)
{
    free(c->dec_state.slices);
    free(c->dec_state.data);
    c->dec_state.slices = NULL;
    c->dec_state.data = NULL;
    c->dec_state.cap_slices = 0;
    c->dec_state.cap_data = 0;
    bc250_dec_reset(c);
}

static bool grow_slices(bc250_context *c)
{
    if (c->dec_state.n_slices < c->dec_state.cap_slices) return true;
    const int fresh = c->dec_state.cap_slices ? c->dec_state.cap_slices * 2 : 16;
    void *p = realloc(c->dec_state.slices, (size_t)fresh * sizeof(*c->dec_state.slices));
    if (!p) return false;
    c->dec_state.slices = p;
    c->dec_state.cap_slices = fresh;
    return true;
}

static bool grow_data(bc250_context *c, size_t count)
{
    if (c->dec_state.n_data + count <= c->dec_state.cap_data) return true;
    size_t fresh = c->dec_state.cap_data ? c->dec_state.cap_data : 65536;
    while (fresh < c->dec_state.n_data + count) fresh *= 2;
    void *p = realloc(c->dec_state.data, fresh);
    if (!p) return false;
    c->dec_state.data = p;
    c->dec_state.cap_data = fresh;
    return true;
}

/* ------------------------------------------------------ vaRenderPicture */

VAStatus bc250_dec_render(bc250_context *c, bc250_buffer *b)
{
    if (!b->data) return VA_STATUS_SUCCESS;

    switch (b->type) {
    case VAPictureParameterBufferType:
        if (b->size < sizeof(VAPictureParameterBufferH264))
            return VA_STATUS_ERROR_INVALID_PARAMETER;
        memcpy(&c->dec_state.pic, b->data, sizeof(VAPictureParameterBufferH264));
        c->dec_state.has_pic = 1;
        return VA_STATUS_SUCCESS;

    case VAIQMatrixBufferType:
        if (b->size < sizeof(VAIQMatrixBufferH264))
            return VA_STATUS_ERROR_INVALID_PARAMETER;
        memcpy(&c->dec_state.iq, b->data, sizeof(VAIQMatrixBufferH264));
        c->dec_state.has_iq = 1;
        return VA_STATUS_SUCCESS;

    case VASliceParameterBufferType: {
        /* One buffer can carry several slices, one element each. */
        const unsigned n = b->num_elements ? b->num_elements : 1;
        if (b->size < sizeof(VASliceParameterBufferH264))
            return VA_STATUS_ERROR_INVALID_PARAMETER;
        for (unsigned k = 0; k < n; k++) {
            if (!grow_slices(c)) return VA_STATUS_ERROR_ALLOCATION_FAILED;
            const VASliceParameterBufferH264 *src =
                (const VASliceParameterBufferH264 *)((const uint8_t *)b->data
                                                     + (size_t)k * b->size);
            c->dec_state.slices[c->dec_state.n_slices].p = *src;
            /* Filled in when the data buffer arrives. */
            c->dec_state.slices[c->dec_state.n_slices].off = (size_t)-1;
            c->dec_state.slices[c->dec_state.n_slices].len = 0;
            c->dec_state.n_slices++;
        }
        return VA_STATUS_SUCCESS;
    }

    case VASliceDataBufferType: {
        /* Pairs up with every slice parameter that is still waiting for its
         * bytes. ffmpeg sends parameters and data alternately, but the API
         * does not promise that, so the pairing is by "not yet filled in"
         * rather than by position. */
        const size_t total = (size_t)b->size * (b->num_elements ? b->num_elements : 1);
        for (int i = 0; i < c->dec_state.n_slices; i++) {
            if (c->dec_state.slices[i].off != (size_t)-1) continue;
            const VASliceParameterBufferH264 *p = &c->dec_state.slices[i].p;
            if (p->slice_data_flag != VA_SLICE_DATA_FLAG_ALL)
                return VA_STATUS_ERROR_UNIMPLEMENTED;
            if ((size_t)p->slice_data_offset + p->slice_data_size > total)
                return VA_STATUS_ERROR_INVALID_PARAMETER;
            if (!grow_data(c, p->slice_data_size))
                return VA_STATUS_ERROR_ALLOCATION_FAILED;
            memcpy(c->dec_state.data + c->dec_state.n_data,
                   (const uint8_t *)b->data + p->slice_data_offset,
                   p->slice_data_size);
            c->dec_state.slices[i].off = c->dec_state.n_data;
            c->dec_state.slices[i].len = p->slice_data_size;
            c->dec_state.n_data += p->slice_data_size;
        }
        return VA_STATUS_SUCCESS;
    }

    default:
        return VA_STATUS_SUCCESS;     /* not ours; the caller may know it */
    }
}

/* --------------------------------------------------------- translation */

static void fill_pic(const bc250_context *c, h264d_pic_t *out)
{
    const VAPictureParameterBufferH264 *p = &c->dec_state.pic;
    memset(out, 0, sizeof(*out));

    out->mb_width = p->picture_width_in_mbs_minus1 + 1;
    out->mb_height = p->picture_height_in_mbs_minus1 + 1;
    out->width = out->mb_width * 16;
    out->height = out->mb_height * 16;

    out->chroma_qp_index_offset = p->chroma_qp_index_offset;
    out->second_chroma_qp_index_offset = p->second_chroma_qp_index_offset;
    out->pic_init_qp = 26 + p->pic_init_qp_minus26;
    out->direct_8x8_inference = p->seq_fields.bits.direct_8x8_inference_flag;

    out->entropy_coding_mode = p->pic_fields.bits.entropy_coding_mode_flag;
    out->transform_8x8_mode = p->pic_fields.bits.transform_8x8_mode_flag;
    out->constrained_intra_pred = p->pic_fields.bits.constrained_intra_pred_flag;
    out->deblocking_filter_control_present =
        p->pic_fields.bits.deblocking_filter_control_present_flag;
    out->weighted_pred = p->pic_fields.bits.weighted_pred_flag;
    out->weighted_bipred_idc = p->pic_fields.bits.weighted_bipred_idc;
    out->pic_order_present = p->pic_fields.bits.pic_order_present_flag;

    /* ⚠️ num_ref_idx_lX_default is never used: every slice carries its own
     * active count and VA hands that over directly. It is filled in anyway
     * so nothing downstream can read a zero and think it means something. */
    out->num_ref_idx_l0 = 1;
    out->num_ref_idx_l1 = 1;

    /* VAIQMatrixBufferH264 is in raster order, which is the order the
     * dequantiser wants, so this is a copy. A picture that carries no
     * matrix uses the flat one. */
    if (c->dec_state.has_iq) {
        memcpy(out->scaling4, c->dec_state.iq.ScalingList4x4, sizeof(out->scaling4));
        /* At 4:2:0 only the two luma 8x8 lists exist. */
        memcpy(out->scaling8[0], c->dec_state.iq.ScalingList8x8[0], 64);
        memcpy(out->scaling8[1], c->dec_state.iq.ScalingList8x8[1], 64);
        for (int i = 2; i < 6; i++)
            memset(out->scaling8[i], 16, 64);
    } else {
        for (int i = 0; i < 6; i++) {
            memset(out->scaling4[i], 16, 16);
            memset(out->scaling8[i], 16, 64);
        }
    }
}

/* One reference list, VA's surfaces turned into frame store slots. Returns
 * false when the stream names a picture this decoder has never produced,
 * which happens after a seek and means the slice cannot be decoded. */
static bool fill_list(h264_decoder_t *dec, const VAPictureH264 *va,
                         int count, int8_t *slot)
{
    for (int i = 0; i < count && i < 32; i++) {
        if (!is_valid(&va[i])) { slot[i] = 0; continue; }
        h264d_frame_t *f = h264_decoder_frame_for(dec, va[i].picture_id);
        if (!f) return false;
        slot[i] = (int8_t)h264_decoder_slot_of(dec, f);
    }
    return true;
}

static void weights_of(const VASliceParameterBufferH264 *p, h264d_slice_t *s)
{
    s->luma_log2_weight_denom = p->luma_log2_weight_denom;
    s->chroma_log2_weight_denom = p->chroma_log2_weight_denom;

    const int16_t one_l = (int16_t)(1 << p->luma_log2_weight_denom);
    const int16_t one_c = (int16_t)(1 << p->chroma_log2_weight_denom);

    for (int i = 0; i < 32; i++) {
        /* ⚠️ A list whose flag is clear has no values in the buffer at all,
         * so the neutral weight has to be put there by hand: the decoder
         * multiplies unconditionally. */
        s->luma_weight[0][i] = p->luma_weight_l0_flag ? p->luma_weight_l0[i] : one_l;
        s->luma_offset[0][i] = p->luma_weight_l0_flag ? p->luma_offset_l0[i] : 0;
        s->luma_weight[1][i] = p->luma_weight_l1_flag ? p->luma_weight_l1[i] : one_l;
        s->luma_offset[1][i] = p->luma_weight_l1_flag ? p->luma_offset_l1[i] : 0;
        for (int k = 0; k < 2; k++) {
            s->chroma_weight[0][i][k] =
                p->chroma_weight_l0_flag ? p->chroma_weight_l0[i][k] : one_c;
            s->chroma_offset[0][i][k] =
                p->chroma_weight_l0_flag ? p->chroma_offset_l0[i][k] : 0;
            s->chroma_weight[1][i][k] =
                p->chroma_weight_l1_flag ? p->chroma_weight_l1[i][k] : one_c;
            s->chroma_offset[1][i][k] =
                p->chroma_weight_l1_flag ? p->chroma_offset_l1[i][k] : 0;
        }
    }
}

/* ---------------------------------------------------------- the picture */

VAStatus bc250_dec_decode(bc250_context *c, gpu_image_t out, gpu_memory_t mem)
{
    h264_decoder_t *dec = c->h264_dec;
    if (!dec) return VA_STATUS_ERROR_INVALID_CONTEXT;
    if (!c->dec_state.has_pic || c->dec_state.n_slices == 0)
        return VA_STATUS_ERROR_INVALID_PARAMETER;

    const VAPictureParameterBufferH264 *pp = &c->dec_state.pic;

    if (!h264_decoder_supports(100, pp->seq_fields.bits.chroma_format_idc,
                               8 + pp->bit_depth_luma_minus8,
                               8 + pp->bit_depth_chroma_minus8,
                               pp->seq_fields.bits.frame_mbs_only_flag,
                               pp->seq_fields.bits.mb_adaptive_frame_field_flag))
        return VA_STATUS_ERROR_UNSUPPORTED_PROFILE;

    /* Flexible macroblock ordering puts macroblocks in a scan order of the
     * picture parameter set's choosing. Nothing here understands that, and
     * a stream that uses it would decode into a scramble rather than fail,
     * so it is refused. Only Baseline allows it. */
    if (pp->num_slice_groups_minus1 > 0)
        return VA_STATUS_ERROR_UNSUPPORTED_PROFILE;

    h264d_pic_t pic;
    fill_pic(c, &pic);
    if (pic.width != c->width || pic.height != c->height)
        return VA_STATUS_ERROR_RESOLUTION_NOT_SUPPORTED;

    /* Which surfaces are still references, so the frame store can let the
     * others go. This runs before begin_picture, which is what keeps the
     * current picture's own slot from being handed to someone else. */
    uint32_t ref_pic[16];
    int poc[16];
    bool length[16];
    int n_refs = 0;
    for (int i = 0; i < 16; i++) {
        if (!is_valid(&pp->ReferenceFrames[i])) continue;
        ref_pic[n_refs] = pp->ReferenceFrames[i].picture_id;
        poc[n_refs] = poc_di(&pp->ReferenceFrames[i]);
        length[n_refs] = (pp->ReferenceFrames[i].flags
                        & VA_PICTURE_H264_LONG_TERM_REFERENCE) != 0;
        n_refs++;
    }
    h264_decoder_set_references(dec, ref_pic, poc, length, n_refs);

    if (h264_decoder_begin_picture(dec, &pic, c->current_render_target,
                                   poc_di(&pp->CurrPic), pp->frame_num) != 0)
        return VA_STATUS_ERROR_OPERATION_FAILED;

    h264d_slice_input_t *ready =
        calloc((size_t)c->dec_state.n_slices, sizeof(*ready));
    if (!ready) return VA_STATUS_ERROR_ALLOCATION_FAILED;
    int n_ready = 0;

    for (int i = 0; i < c->dec_state.n_slices; i++) {
        const VASliceParameterBufferH264 *sp = &c->dec_state.slices[i].p;
        if (c->dec_state.slices[i].off == (size_t)-1)
            continue;                    /* a parameter with no data */

        h264d_slice_t sl;
        memset(&sl, 0, sizeof(sl));
        sl.type = sp->slice_type % 5;
        sl.first_mb = sp->first_mb_in_slice;
        sl.qpy = pic.pic_init_qp + sp->slice_qp_delta;
        sl.cabac_init_idc = sp->cabac_init_idc;
        sl.direct_spatial_mv_pred = sp->direct_spatial_mv_pred_flag != 0;
        sl.disable_deblocking_filter_idc = sp->disable_deblocking_filter_idc;
        sl.alpha_c0_offset = 2 * sp->slice_alpha_c0_offset_div2;
        sl.beta_offset = 2 * sp->slice_beta_offset_div2;
        sl.num_ref_idx[0] = (sl.type == 2) ? 0 : sp->num_ref_idx_l0_active_minus1 + 1;
        sl.num_ref_idx[1] = (sl.type == 1) ? sp->num_ref_idx_l1_active_minus1 + 1 : 0;

        if (!fill_list(dec, sp->RefPicList0, sl.num_ref_idx[0], sl.ref_list[0])
            || !fill_list(dec, sp->RefPicList1, sl.num_ref_idx[1], sl.ref_list[1]))
            continue;   /* a reference we never decoded: skip, do not guess */

        weights_of(sp, &sl);

        if (n_ready < c->dec_state.n_slices) {
            ready[n_ready].slice = sl;
            ready[n_ready].data = c->dec_state.data + c->dec_state.slices[i].off;
            ready[n_ready].size = c->dec_state.slices[i].len;
            ready[n_ready].bit_offset = sp->slice_data_bit_offset;
            n_ready++;
        }
    }

    /* All of them at once: they are independent, so the decoder can read
     * several at the same time. */
    if (n_ready > 0)
        h264_decoder_slices(dec, ready, n_ready);
    free(ready);

    /* ⚠️ Always finished, even when every slice was refused. A picture that
     * is never ended leaves the frame store holding a slot that no later
     * picture can reuse, and the application still gets its surface back -
     * green rather than absent, which is what every other driver does after
     * a seek. */
    if (h264_decoder_end_picture(dec, out, mem) != 0)
        return VA_STATUS_ERROR_OPERATION_FAILED;
    return VA_STATUS_SUCCESS;
}
