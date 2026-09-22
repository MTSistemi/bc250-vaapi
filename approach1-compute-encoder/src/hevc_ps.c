/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_ps.c - reading H.265 parameter sets and slice segment headers.
 *
 * Every refusal here is deliberate. A parameter set that says something
 * this decoder does not implement is rejected by name rather than parsed
 * past, because the alternative is a picture that decodes into something
 * plausible and wrong - which costs far more to find than an error
 * message costs to read.
 */
#include "hevc_ps.h"

#include <string.h>

/* What each refusal means, so a caller can say it out loud. */
enum {
    PS_OK = 0,
    PS_TRUNCATED = -1,          /* the NAL ended in the middle of something */
    PS_UNSUPPORTED = -2,  /* legal, but not implemented here */
    PS_NONSENSE = -3,         /* outside what the standard allows */
};

/* ------------------------------------------------------- profile_tier_level */

/* Clause 7.3.3. Nothing in it changes how a picture decodes at Main
 * profile, so this reads it to get past it. The lengths are the point:
 * eighty-eight bits of profile then eight of level, and a sub-layer block
 * that is present or not per flag. */
static int read_ptl(br_t *br, int max_sub_layers_minus1)
{
    br_skip(br, 88);                 /* space, tier, idc, compatibility, flags */
    br_skip(br, 8);                  /* general_level_idc */

    bool profile[8] = { false }, level[8] = { false };
    for (int i = 0; i < max_sub_layers_minus1; i++) {
        profile[i] = br_read1(br) != 0;
        level[i] = br_read1(br) != 0;
    }
    if (max_sub_layers_minus1 > 0)
        for (int i = max_sub_layers_minus1; i < 8; i++)
            br_skip(br, 2);          /* reserved_zero_2bits */
    for (int i = 0; i < max_sub_layers_minus1; i++) {
        if (profile[i]) br_skip(br, 88);
        if (level[i]) br_skip(br, 8);
    }
    return br_overrun(br) ? PS_TRUNCATED : PS_OK;
}

/* ------------------------------------------------------------ scaling lists */

/* Clause 7.3.4. Read to be consumed, not kept: a stream that carries its
 * own quantisation matrices is refused by the caller, and reading them
 * here is what lets the caller see the rest of the parameter set before it
 * does so. */
static int skip_scaling_list(br_t *br)
{
    for (int size = 0; size < 4; size++) {
        for (int mat = 0; mat < 6; mat += (size == 3) ? 3 : 1) {
            if (!br_read1(br)) {
                br_read_ue(br);           /* pred_matrix_id_delta */
                continue;
            }
            const int count = (size == 0) ? 16 : 64;
            if (size > 1) br_read_se(br); /* dc_coef_minus8 */
            for (int i = 0; i < count; i++)
                br_read_se(br);           /* delta_coef */
        }
    }
    return br_overrun(br) ? PS_TRUNCATED : PS_OK;
}

/* ------------------------------------------------- short-term reference sets */

/* Clause 7.3.7 and the derivations of 7.4.8.
 *
 * ⚠️ A set can be written as a difference from the one before it, and then
 * the deltas in the bitstream are not the deltas that come out: they
 * describe which pictures of the earlier set survive, shifted. Reading the
 * short form and storing it as if it were the long one gives reference
 * lists that are wrong only for some pictures, which is the worst kind of
 * wrong there is. */
static int read_st_rps(br_t *br, hevc_st_rps_t *out,
                        const hevc_st_rps_t *previous_ones, int idx, int count)
{
    memset(out, 0, sizeof(*out));

    bool from_previous = false;
    if (idx != 0)
        from_previous = br_read1(br) != 0;

    if (from_previous) {
        int delta_idx = 1;
        if (idx == count)
            delta_idx = (int)br_read_ue(br) + 1;
        if (idx - delta_idx < 0)
            return PS_NONSENSE;
        const hevc_st_rps_t *ref_pic = &previous_ones[idx - delta_idx];
        const int n_refs = ref_pic->num_negative + ref_pic->num_positive;
        if (n_refs > HEVC_MAX_RPS * 2 - 1)
            return PS_NONSENSE;

        const int sign = br_read1(br) ? -1 : 1;
        const int abs_delta = (int)br_read_ue(br) + 1;
        const int delta_rps = sign * abs_delta;

        bool used_flag[HEVC_MAX_RPS * 2 + 1] = { false };
        bool usa_delta[HEVC_MAX_RPS * 2 + 1];
        for (int j = 0; j <= n_refs; j++) {
            used_flag[j] = br_read1(br) != 0;
            usa_delta[j] = used_flag[j] ? true : (br_read1(br) != 0);
        }
        if (br_overrun(br)) return PS_TRUNCATED;

        /* 7-61: the pictures before this one, built from the reference
         * set's positives walked backwards, then the reference picture
         * itself, then its negatives. */
        int i = 0;
        for (int j = ref_pic->num_positive - 1; j >= 0; j--) {
            const int d = ref_pic->delta_poc[ref_pic->num_negative + j] + delta_rps;
            if (d < 0 && usa_delta[ref_pic->num_negative + j]) {
                out->delta_poc[i] = d;
                out->used[i++] = used_flag[ref_pic->num_negative + j];
            }
        }
        if (delta_rps < 0 && usa_delta[n_refs]) {
            out->delta_poc[i] = delta_rps;
            out->used[i++] = used_flag[n_refs];
        }
        for (int j = 0; j < ref_pic->num_negative; j++) {
            const int d = ref_pic->delta_poc[j] + delta_rps;
            if (d < 0 && usa_delta[j]) {
                out->delta_poc[i] = d;
                out->used[i++] = used_flag[j];
            }
        }
        out->num_negative = i;

        /* 7-62: and the ones after, the mirror of that. */
        int k = i;
        for (int j = ref_pic->num_negative - 1; j >= 0; j--) {
            const int d = ref_pic->delta_poc[j] + delta_rps;
            if (d > 0 && usa_delta[j]) {
                out->delta_poc[k] = d;
                out->used[k++] = used_flag[j];
            }
        }
        if (delta_rps > 0 && usa_delta[n_refs]) {
            out->delta_poc[k] = delta_rps;
            out->used[k++] = used_flag[n_refs];
        }
        for (int j = 0; j < ref_pic->num_positive; j++) {
            const int d = ref_pic->delta_poc[ref_pic->num_negative + j] + delta_rps;
            if (d > 0 && usa_delta[ref_pic->num_negative + j]) {
                out->delta_poc[k] = d;
                out->used[k++] = used_flag[ref_pic->num_negative + j];
            }
        }
        out->num_positive = k - i;
        return br_overrun(br) ? PS_TRUNCATED : PS_OK;
    }

    const int neg = (int)br_read_ue(br);
    const int pos = (int)br_read_ue(br);
    if (neg > HEVC_MAX_RPS || pos > HEVC_MAX_RPS)
        return PS_NONSENSE;
    out->num_negative = neg;
    out->num_positive = pos;

    int poc = 0;
    for (int i = 0; i < neg; i++) {
        poc -= (int)br_read_ue(br) + 1;
        out->delta_poc[i] = poc;
        out->used[i] = br_read1(br) != 0;
    }
    poc = 0;
    for (int i = 0; i < pos; i++) {
        poc += (int)br_read_ue(br) + 1;
        out->delta_poc[neg + i] = poc;
        out->used[neg + i] = br_read1(br) != 0;
    }
    return br_overrun(br) ? PS_TRUNCATED : PS_OK;
}

/* --------------------------------------------------------------------- SPS */

int hevc_ps_read_sps(hevc_sps_t *out, const uint8_t *rbsp, size_t n)
{
    br_t br;
    br_init(&br, rbsp, n);
    br_skip(&br, 16);                       /* the two-byte NAL header */

    hevc_sps_t s;
    memset(&s, 0, sizeof(s));

    s.vps_id = (int)br_read(&br, 4);
    const int max_sub = (int)br_read(&br, 3);
    br_read1(&br);                          /* temporal_id_nesting_flag */
    const int r = read_ptl(&br, max_sub);
    if (r) return r;

    s.sps_id = (int)br_read_ue(&br);
    if (s.sps_id < 0 || s.sps_id > 15) return PS_NONSENSE;

    s.chroma_format_idc = (int)br_read_ue(&br);
    if (s.chroma_format_idc == 3)
        s.separate_colour_plane = br_read1(&br) != 0;
    if (s.chroma_format_idc != 1) return PS_UNSUPPORTED;   /* 4:2:0 only */

    s.width = (int)br_read_ue(&br);
    s.height = (int)br_read_ue(&br);
    if (s.width <= 0 || s.height <= 0 || s.width > 16384 || s.height > 16384)
        return PS_NONSENSE;

    if (br_read1(&br)) {                    /* conformance_window_flag */
        /* In chroma units, which at 4:2:0 is half a luma sample each way. */
        s.crop_left = (int)br_read_ue(&br) * 2;
        s.crop_right = (int)br_read_ue(&br) * 2;
        s.crop_top = (int)br_read_ue(&br) * 2;
        s.crop_bottom = (int)br_read_ue(&br) * 2;
    }

    s.bit_depth_luma = 8 + (int)br_read_ue(&br);
    s.bit_depth_chroma = 8 + (int)br_read_ue(&br);
    /* ⚠️ Main and Main 10, and the two depths must be equal. Every
     * template in the decoder is built for eight and for ten, and every
     * line that picks between them reads the LUMA depth. A stream with
     * ten-bit luma and eight-bit chroma would take the ten-bit path for
     * both planes and be wrong in the chroma without a word. */
    if ((s.bit_depth_luma != 8 && s.bit_depth_luma != 10)
        || s.bit_depth_chroma != s.bit_depth_luma)
        return PS_UNSUPPORTED;

    s.log2_max_poc_lsb = 4 + (int)br_read_ue(&br);
    if (s.log2_max_poc_lsb < 4 || s.log2_max_poc_lsb > 16) return PS_NONSENSE;

    const bool per_sub_layer = br_read1(&br) != 0;
    for (int i = per_sub_layer ? 0 : max_sub; i <= max_sub; i++) {
        s.max_dec_pic_buffering = (int)br_read_ue(&br) + 1;
        s.num_reorder_pics = (int)br_read_ue(&br);
        br_read_ue(&br);                    /* max_latency_increase_plus1 */
    }

    s.log2_min_cb = 3 + (int)br_read_ue(&br);
    s.log2_ctb = s.log2_min_cb + (int)br_read_ue(&br);
    s.log2_min_tb = 2 + (int)br_read_ue(&br);
    s.log2_max_tb = s.log2_min_tb + (int)br_read_ue(&br);
    if (s.log2_ctb < 4 || s.log2_ctb > 6 || s.log2_min_cb < 3
        || s.log2_min_tb < 2 || s.log2_max_tb > 5 || s.log2_max_tb > s.log2_ctb)
        return PS_NONSENSE;
    s.max_transform_hierarchy_depth_inter = (int)br_read_ue(&br);
    s.max_transform_hierarchy_depth_intra = (int)br_read_ue(&br);

    s.scaling_list_enabled = br_read1(&br) != 0;
    if (s.scaling_list_enabled) {
        s.sps_scaling_list_present = br_read1(&br) != 0;
        if (s.sps_scaling_list_present) {
            const int e = skip_scaling_list(&br);
            if (e) return e;
        }
    }

    s.amp_enabled = br_read1(&br) != 0;
    s.sao_enabled = br_read1(&br) != 0;

    s.pcm_enabled = br_read1(&br) != 0;
    if (s.pcm_enabled) {
        s.pcm_bit_depth_luma = 1 + (int)br_read(&br, 4);
        s.pcm_bit_depth_chroma = 1 + (int)br_read(&br, 4);
        s.log2_min_pcm_cb = 3 + (int)br_read_ue(&br);
        s.log2_max_pcm_cb = s.log2_min_pcm_cb + (int)br_read_ue(&br);
        s.pcm_loop_filter_disabled = br_read1(&br) != 0;
    }

    s.num_st_rps = (int)br_read_ue(&br);
    if (s.num_st_rps < 0 || s.num_st_rps > 64) return PS_NONSENSE;
    for (int i = 0; i < s.num_st_rps; i++) {
        const int e = read_st_rps(&br, &s.st_rps[i], s.st_rps, i, s.num_st_rps);
        if (e) return e;
    }

    s.long_term_ref_pics_present = br_read1(&br) != 0;
    if (s.long_term_ref_pics_present) {
        s.num_long_term_sps = (int)br_read_ue(&br);
        for (int i = 0; i < s.num_long_term_sps; i++) {
            br_skip(&br, s.log2_max_poc_lsb);   /* lt_ref_pic_poc_lsb_sps */
            br_read1(&br);                      /* used_by_curr_pic_lt_sps */
        }
    }

    s.temporal_mvp_enabled = br_read1(&br) != 0;
    s.strong_intra_smoothing = br_read1(&br) != 0;
    /* Everything after this - VUI and the extensions - changes nothing
     * about how a picture decodes, so it is where reading stops. */

    if (br_overrun(&br)) return PS_TRUNCATED;

    s.ctb_size = 1 << s.log2_ctb;
    s.ctb_width = (s.width + s.ctb_size - 1) >> s.log2_ctb;
    s.ctb_height = (s.height + s.ctb_size - 1) >> s.log2_ctb;
    s.ctb_count = s.ctb_width * s.ctb_height;
    s.min_cb_width = s.width >> s.log2_min_cb;
    s.min_cb_height = s.height >> s.log2_min_cb;
    s.valid = true;
    *out = s;
    return PS_OK;
}

/* --------------------------------------------------------------------- PPS */

int hevc_ps_read_pps(hevc_pps_t *out, const uint8_t *rbsp, size_t n)
{
    br_t br;
    br_init(&br, rbsp, n);
    br_skip(&br, 16);

    hevc_pps_t p;
    memset(&p, 0, sizeof(p));

    p.pps_id = (int)br_read_ue(&br);
    p.sps_id = (int)br_read_ue(&br);
    if (p.pps_id < 0 || p.pps_id > 63 || p.sps_id < 0 || p.sps_id > 15)
        return PS_NONSENSE;

    p.dependent_slice_segments_enabled = br_read1(&br) != 0;
    p.output_flag_present = br_read1(&br) != 0;
    p.num_extra_slice_header_bits = (int)br_read(&br, 3);
    p.sign_data_hiding = br_read1(&br) != 0;
    p.cabac_init_present = br_read1(&br) != 0;
    p.num_ref_idx_default[0] = (int)br_read_ue(&br) + 1;
    p.num_ref_idx_default[1] = (int)br_read_ue(&br) + 1;
    p.init_qp = 26 + br_read_se(&br);
    p.constrained_intra_pred = br_read1(&br) != 0;
    p.transform_skip_enabled = br_read1(&br) != 0;
    p.cu_qp_delta_enabled = br_read1(&br) != 0;
    if (p.cu_qp_delta_enabled)
        p.diff_cu_qp_delta_depth = (int)br_read_ue(&br);
    p.cb_qp_offset = br_read_se(&br);
    p.cr_qp_offset = br_read_se(&br);
    p.slice_chroma_qp_offsets_present = br_read1(&br) != 0;
    p.weighted_pred = br_read1(&br) != 0;
    p.weighted_bipred = br_read1(&br) != 0;
    p.transquant_bypass_enabled = br_read1(&br) != 0;
    p.tiles_enabled = br_read1(&br) != 0;
    p.entropy_coding_sync_enabled = br_read1(&br) != 0;

    /* ⚠️ Tiles and wavefront together are legal in the standard and are
     * sent by nothing we have seen; kvazaar, the only encoder here that
     * writes tiles at all, calls the combination experimental. The
     * substreams would then run per tile per row, and the state a row
     * continues from would be the tile's rather than the picture's.
     * Refused rather than guessed at - a guess here decodes into a
     * plausible-looking scramble instead of failing. */
    if (p.tiles_enabled && p.entropy_coding_sync_enabled)
        return PS_UNSUPPORTED;

    p.num_tile_columns = 1;
    p.num_tile_rows = 1;
    p.loop_filter_across_tiles = true;
    if (p.tiles_enabled) {
        p.num_tile_columns = (int)br_read_ue(&br) + 1;
        p.num_tile_rows = (int)br_read_ue(&br) + 1;
        if (p.num_tile_columns > 32 || p.num_tile_rows > 32)
            return PS_NONSENSE;
        p.uniform_spacing = br_read1(&br) != 0;
        if (!p.uniform_spacing) {
            for (int i = 0; i < p.num_tile_columns - 1; i++)
                p.column_width[i] = (int)br_read_ue(&br) + 1;
            for (int i = 0; i < p.num_tile_rows - 1; i++)
                p.row_height[i] = (int)br_read_ue(&br) + 1;
        }
        p.loop_filter_across_tiles = br_read1(&br) != 0;
    }

    p.loop_filter_across_slices = br_read1(&br) != 0;
    p.deblocking_filter_control_present = br_read1(&br) != 0;
    if (p.deblocking_filter_control_present) {
        p.deblocking_filter_override_enabled = br_read1(&br) != 0;
        p.deblocking_filter_disabled = br_read1(&br) != 0;
        if (!p.deblocking_filter_disabled) {
            p.beta_offset = 2 * br_read_se(&br);
            p.tc_offset = 2 * br_read_se(&br);
        }
    }

    p.pps_scaling_list_present = br_read1(&br) != 0;
    if (p.pps_scaling_list_present) {
        const int e = skip_scaling_list(&br);
        if (e) return e;
    }

    p.lists_modification_present = br_read1(&br) != 0;
    p.log2_parallel_merge_level = 2 + (int)br_read_ue(&br);
    p.slice_segment_header_extension_present = br_read1(&br) != 0;

    if (br_overrun(&br)) return PS_TRUNCATED;
    p.valid = true;
    *out = p;
    return PS_OK;
}

/* ----------------------------------------------------- pred_weight_table */

/* Clause 7.3.6.3 and the derivations beside it.
 *
 * ⚠️ The chroma offset is not what the bitstream carries. What is sent is
 * the difference from what the weight alone would have done to a mid-grey
 * sample, so the offset has to be put back together before it means
 * anything. */
static void read_weights(br_t *br, hevc_slice_t *s, bool chroma)
{
    s->luma_log2_weight_denom = (int)br_read_ue(br);
    s->chroma_log2_weight_denom = s->luma_log2_weight_denom;
    if (chroma)
        s->chroma_log2_weight_denom += br_read_se(br);

    const int one_l = 1 << s->luma_log2_weight_denom;
    const int one_c = 1 << s->chroma_log2_weight_denom;
    const int lists = (s->type == 0) ? 2 : 1;

    for (int l = 0; l < lists; l++) {
        bool ha_luma[16] = { false }, ha_chroma[16] = { false };
        const int count = s->num_ref_idx[l] < 16 ? s->num_ref_idx[l] : 16;

        for (int i = 0; i < count; i++)
            ha_luma[i] = br_read1(br) != 0;
        if (chroma)
            for (int i = 0; i < count; i++)
                ha_chroma[i] = br_read1(br) != 0;

        for (int i = 0; i < count; i++) {
            if (ha_luma[i]) {
                s->luma_weight[l][i] = (int16_t)(one_l + br_read_se(br));
                s->luma_offset[l][i] = (int16_t)br_read_se(br);
            } else {
                s->luma_weight[l][i] = (int16_t)one_l;
                s->luma_offset[l][i] = 0;
            }
            for (int j = 0; j < 2; j++) {
                if (ha_chroma[i]) {
                    const int w = one_c + br_read_se(br);
                    const int d = br_read_se(br);
                    int o = d - (((128 * w) >> s->chroma_log2_weight_denom) - 128);
                    o = o < -128 ? -128 : (o > 127 ? 127 : o);
                    s->chroma_weight[l][i][j] = (int16_t)w;
                    s->chroma_offset[l][i][j] = (int16_t)o;
                } else {
                    s->chroma_weight[l][i][j] = (int16_t)one_c;
                    s->chroma_offset[l][i][j] = 0;
                }
            }
        }
    }
}

/* ------------------------------------------------------- slice segment header */

int hevc_ps_read_slice(hevc_slice_t *out, const uint8_t *rbsp, size_t n,
                        int nal_type, const hevc_sps_t *sps_store,
                        const hevc_pps_t *pps_store)
{
    br_t br;
    br_init(&br, rbsp, n);
    br_skip(&br, 16);

    hevc_slice_t s;
    memset(&s, 0, sizeof(s));
    s.nal_type = nal_type;
    s.pic_output_flag = true;
    s.collocated_from_l0 = true;

    s.first_slice_in_pic = br_read1(&br) != 0;
    if (hevc_nal_e_irap(nal_type))
        s.no_output_of_prior_pics = br_read1(&br) != 0;

    s.pps_id = (int)br_read_ue(&br);
    if (s.pps_id < 0 || s.pps_id > 63 || !pps_store[s.pps_id].valid)
        return PS_NONSENSE;
    const hevc_pps_t *pps = &pps_store[s.pps_id];
    if (!sps_store[pps->sps_id].valid)
        return PS_NONSENSE;
    const hevc_sps_t *sps = &sps_store[pps->sps_id];

    if (!s.first_slice_in_pic) {
        if (pps->dependent_slice_segments_enabled)
            s.dependent_slice_segment = br_read1(&br) != 0;
        /* Ceil(Log2(PicSizeInCtbsY)) bits. */
        int bit = 0;
        while ((1 << bit) < sps->ctb_count) bit++;
        s.segment_address = (int)br_read(&br, bit);
    }

    if (s.dependent_slice_segment) {
        /* ⚠️ A dependent slice segment inherits the previous independent
         * header entire, and reconstructing that is the caller's
         * business - but it still carries its OWN entry points, header
         * extension and byte alignment. Returning here skipped all
         * three and left data_bit_offset pointing at the alignment bit
         * instead of at the slice data. So it jumps to the tail rather
         * than out. */
        goto tail;
    }

    for (int i = 0; i < pps->num_extra_slice_header_bits; i++)
        br_read1(&br);

    s.type = (int)br_read_ue(&br);
    if (s.type < 0 || s.type > 2) return PS_NONSENSE;

    if (pps->output_flag_present)
        s.pic_output_flag = br_read1(&br) != 0;

    if (!hevc_nal_e_idr(nal_type)) {
        s.poc_lsb = (int)br_read(&br, sps->log2_max_poc_lsb);
        s.short_term_ref_pic_set_sps_flag = br_read1(&br) != 0;
        if (!s.short_term_ref_pic_set_sps_flag) {
            const int e = read_st_rps(&br, &s.st_rps, sps->st_rps,
                                       sps->num_st_rps, sps->num_st_rps);
            if (e) return e;
        } else if (sps->num_st_rps > 1) {
            int bit = 0;
            while ((1 << bit) < sps->num_st_rps) bit++;
            s.short_term_ref_pic_set_idx = (int)br_read(&br, bit);
            if (s.short_term_ref_pic_set_idx >= sps->num_st_rps)
                return PS_NONSENSE;
            s.st_rps = sps->st_rps[s.short_term_ref_pic_set_idx];
        } else if (sps->num_st_rps == 1) {
            s.st_rps = sps->st_rps[0];
        }

        if (sps->long_term_ref_pics_present)
            return PS_UNSUPPORTED;   /* long-term references */

        if (sps->temporal_mvp_enabled)
            s.temporal_mvp_enabled = br_read1(&br) != 0;
    }

    if (sps->sao_enabled) {
        s.sao_luma = br_read1(&br) != 0;
        s.sao_chroma = br_read1(&br) != 0;
    }

    s.num_ref_idx[0] = pps->num_ref_idx_default[0];
    s.num_ref_idx[1] = pps->num_ref_idx_default[1];

    if (s.type != 2) {                       /* P or B */
        if (br_read1(&br)) {                 /* num_ref_idx_active_override */
            s.num_ref_idx[0] = (int)br_read_ue(&br) + 1;
            if (s.type == 0)
                s.num_ref_idx[1] = (int)br_read_ue(&br) + 1;
        }
        if (s.type != 0)
            s.num_ref_idx[1] = 0;
        if (s.num_ref_idx[0] > 15 || s.num_ref_idx[1] > 15)
            return PS_NONSENSE;

        if (pps->lists_modification_present) {
            /* NumPicTotalCurr: how many pictures the lists can be built
             * from. The modification syntax is only present when there is
             * more than one. */
            int count = 0;
            for (int i = 0; i < s.st_rps.num_negative + s.st_rps.num_positive; i++)
                if (s.st_rps.used[i]) count++;
            if (count > 1)
                return PS_UNSUPPORTED;    /* reference list modification */
        }

        if (s.type == 0)
            s.mvd_l1_zero = br_read1(&br) != 0;
        if (pps->cabac_init_present)
            s.cabac_init_flag = br_read1(&br) != 0;
        if (s.temporal_mvp_enabled) {
            /* Inferred to one when it is not sent, which for a P slice is
             * always. */
            s.collocated_from_l0 = true;
            if (s.type == 0)
                s.collocated_from_l0 = br_read1(&br) != 0;
            if ((s.collocated_from_l0 && s.num_ref_idx[0] > 1)
                || (!s.collocated_from_l0 && s.num_ref_idx[1] > 1))
                s.collocated_ref_idx = (int)br_read_ue(&br);
        }
        if ((pps->weighted_pred && s.type == 1)
            || (pps->weighted_bipred && s.type == 0))
            read_weights(&br, &s, sps->chroma_format_idc != 0);

        s.five_minus_max_num_merge_cand = (int)br_read_ue(&br);
        if (s.five_minus_max_num_merge_cand > 4) return PS_NONSENSE;
    }

    s.qp = pps->init_qp + br_read_se(&br);
    if (s.qp < -6 * (sps->bit_depth_luma - 8) || s.qp > 51) return PS_NONSENSE;

    if (pps->slice_chroma_qp_offsets_present) {
        s.cb_qp_offset = br_read_se(&br);
        s.cr_qp_offset = br_read_se(&br);
    }

    s.deblocking_filter_disabled = pps->deblocking_filter_disabled;
    s.beta_offset = pps->beta_offset;
    s.tc_offset = pps->tc_offset;
    if (pps->deblocking_filter_override_enabled && br_read1(&br)) {
        s.deblocking_filter_disabled = br_read1(&br) != 0;
        if (!s.deblocking_filter_disabled) {
            s.beta_offset = 2 * br_read_se(&br);
            s.tc_offset = 2 * br_read_se(&br);
        }
    }

    s.loop_filter_across_slices = pps->loop_filter_across_slices;
    if (pps->loop_filter_across_slices
        && (s.sao_luma || s.sao_chroma || !s.deblocking_filter_disabled))
        s.loop_filter_across_slices = br_read1(&br) != 0;

tail:
    if (pps->tiles_enabled || pps->entropy_coding_sync_enabled) {
        s.num_entry_point_offsets = (int)br_read_ue(&br);
        if (s.num_entry_point_offsets > 0) {
            const int len = (int)br_read_ue(&br) + 1;
            if (len > 32) return PS_NONSENSE;
            if (s.num_entry_point_offsets > 600) return PS_UNSUPPORTED;
            uint32_t sum = 0;
            for (int i = 0; i < s.num_entry_point_offsets; i++) {
                sum += (uint32_t)br_read(&br, len) + 1;
                s.entry_point[i] = sum;
            }
        }
    }

    if (pps->slice_segment_header_extension_present) {
        const int len = (int)br_read_ue(&br);
        br_skip(&br, len * 8);
    }

    /* byte_alignment(): a one bit, then zeros to the next byte.
     *
     * ⚠️ Read and checked, not skipped: see the note at the top of this
     * file's history. It is the cheapest possible proof that the header
     * above was read correctly. */
    if (!br_read1(&br)) return PS_NONSENSE;
    while (br.bitpos & 7)
        if (br_read1(&br)) return PS_NONSENSE;

    if (br_overrun(&br)) return PS_TRUNCATED;
    s.data_bit_offset = br.bitpos;
    *out = s;
    return PS_OK;
}
