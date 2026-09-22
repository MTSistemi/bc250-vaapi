/* bc250-encoding-decoding-fix v0.4.3 - https://github.com/simpmix/bc250-encoding-decoding-fix */
/*
 * Copyright (c) 2026 BC-250 Project Contributors
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * encoder_h265.c - H.265/HEVC encoder supporting IDR (I-slices) and
 *                  inter-frame prediction (P-slices with zero-motion CU skip).
 *
 * ============================================================================
 * DESIGN, in one place (see docs/hevc_scope_note.md and DEVLOG.md Sec. 6/8/27
 * for the history of why this replaced a non-functional stub)
 * ============================================================================
 *
 * This encoder supports GOP structures with periodic/forced IDR frames and
 * inter-predicted P-frames. For P-frames:
 *   - Reference picture set (RPS) is configured with DeltaPOC = -1 pointing
 *     to the previous reconstructed frame in the DPB.
 *   - NAL unit type is NAL_UNIT_CODED_SLICE_TRAIL_R (1) with 4-bit POC LSB.
 *   - Each 8x8 CU evaluates temporal difference against the reference picture.
 *     Static / low-motion blocks are coded as SKIP CUs (cu_skip_flag = 1,
 *     merge_idx = 0) with zero residual and zero motion vector, yielding
 *     immense bitrate reduction on typical desktop / streaming video.
 *   - Dynamic blocks are coded with cu_skip_flag = 0 and pred_mode_flag = 1
 *     (MODE_INTRA) falling back to full intra prediction and transform coding.
 *
 * Picture structure, chosen to keep every stage genuinely simple AND
 * genuinely spec-correct at the same time (see hevc_intra.h's top comment
 * for why this is NOT a case of reusing the existing GPU DCT/quantize
 * shaders - HEVC's mandatory 4x4-luma-intra DST-VII transform and its own
 * QP-to-quantizer-step mapping make that numerically wrong, not just a
 * block-size mismatch):
 *
 *   - CTU size = 16x16 (the minimum ITU-T H.265 allows - CtbLog2SizeY must
 *     be 4..6). One split_cu_flag=1 per CTU (always forced - condL/condA
 *     context still real, computed from real neighbor availability), giving
 *     exactly four 8x8 CUs per CTU, in z-order (TL,TR,BL,BR).
 *   - Every CU is intra, PartMode=PART_NxN (legal only at minimum CU size,
 *     which 8x8 always is here) - four independent 4x4 luma PUs per CU, each
 *     with its own real intra_luma_pred_mode. PartMode=NxN makes
 *     IntraSplitFlag=1, which per 7.4.9.8 FORCES (infers, no bit spent) the
 *     transform tree to split once at trafoDepth==0, landing exactly on the
 *     four 4x4 luma PUs as their own leaf TUs - no separate transform-size
 *     decision needed anywhere in this encoder.
 *   - Chroma (4:2:0) is one 4x4 Cb + one 4x4 Cr block per CU (8x8 luma / 1
 *     chroma shift = 4x4 chroma, coded once at the CU's own transform-tree
 *     root per the spec's "chroma stops splitting at the 4x4 floor" rule -
 *     see encode_cu()'s comment).
 *   - Every 4x4 TU (luma AND chroma) is therefore always exactly one
 *     coefficient group - no sig_coeff_group_flag/coded_sub_block_flag
 *     complexity anywhere (see hevc_cabac.h's scope note).
 *
 * Real per-4x4-block intra prediction (Planar/DC/Horizontal/Vertical, the
 * same four candidates the GPU's own I16x16 SAD decision already knows how
 * to choose between, conceptually) with proper z-scan reconstruction
 * chaining, real DST-VII (luma) / DCT-II (chroma) transform + real HEVC
 * quantization, and real CABAC entropy coding are implemented in
 * hevc_intra.c and hevc_cabac.c respectively - see those files.
 *
 * The one piece of the existing GPU/Vulkan infrastructure this file DOES
 * reuse unmodified is gpu_compute_download_nv12() - the real, already-
 * uploaded picture is read back from the GPU surface into host memory once
 * per frame, exactly the way va_backend.c's own CPU-side surface access
 * (bc250_MapBuffer et al) already does, and the existing
 * gpu_compute_begin_picture/dispatch_encode/end_picture/sync() sequence is
 * still called first (with its result discarded) purely to preserve the
 * exact same Vulkan image layout transitions and fence/staging-buffer
 * bookkeeping the rest of this driver (va_backend.c's EndPicture) already
 * depends on - see that call site's comment below.
 */

#include "encoder_h265.h"
#include "bitstream.h"
#include "hevc_cabac.h"
#include "hevc_intra.h"
#include "dynamic_governor.h"
#include "cpu_simd_me.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__SSE2__) || defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#endif

#define NAL_UNIT_CODED_SLICE_TRAIL_R     1
#define NAL_UNIT_CODED_SLICE_IDR_W_RADL 19
#define NAL_UNIT_VPS               32
#define NAL_UNIT_SPS               33
#define NAL_UNIT_PPS               34
#define NAL_UNIT_AUD               35

#define HEVC_MAX_SLICES 16
#define HEVC_CTU_SIZE 16
#define HEVC_CU_SIZE   8
#define HEVC_PU_SIZE   4

static inline uint8_t clip8i(int v) { return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); }

/* ============================================================================
 * Level selection (Annex A.3 MaxLumaPs table, picture-size-only heuristic -
 * a real encoder would also check bitrate/CPB constraints; this project's
 * one-QP-for-the-whole-stream design has no rate-control loop to check
 * against, so this picks the smallest level whose MaxLumaPs covers the
 * picture, which is what every simple/embedded HEVC encoder does in
 * practice for a "just make it playable" level tag).
 * ==========================================================================*/
static int hevc_pick_level_idc(uint32_t width, uint32_t height) {
    static const struct { uint64_t max_luma_ps; int level_idc; } table[] = {
        { 36864UL,        30 },
        { 122880UL,       60 },
        { 245760UL,       63 },
        { 552960UL,       90 },
        { 983040UL,       93 },
        { 2228224UL,     120 },
        { 2228224UL,     123 },
        { 8912896UL,     150 },
        { 8912896UL,     153 },
        { 8912896UL,     156 },
        { 35651584UL,    180 },
        { 35651584UL,    183 },
        { 35651584UL,    186 },
    };
    uint64_t pic_size = (uint64_t)width * (uint64_t)height;
    for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); i++)
        if (pic_size <= table[i].max_luma_ps) return table[i].level_idc;
    return 186;
}

/* ============================================================================
 * VPS / SPS / PPS
 * ==========================================================================*/

static void write_profile_tier_level(bitstream_t *bs, int level_idc) {
    bs_write_u(bs, 2, 0);   /* general_profile_space */
    bs_write1(bs, 0);       /* general_tier_flag (Main tier) */
    bs_write_u(bs, 5, 1);   /* general_profile_idc = 1 (Main) */
    for (int i = 0; i < 32; i++)
        bs_write1(bs, i == 1 ? 1 : 0); /* general_profile_compatibility_flag[1] = Main */
    bs_write1(bs, 1); /* general_progressive_source_flag */
    bs_write1(bs, 0); /* general_interlaced_source_flag */
    bs_write1(bs, 0); /* general_non_packed_constraint_flag */
    bs_write1(bs, 1); /* general_frame_only_constraint_flag */
    bs_write_u(bs, 16, 0);
    bs_write_u(bs, 16, 0);
    bs_write_u(bs, 12, 0); /* reserved_zero_44bits */
    bs_write_u(bs, 8, level_idc);
}

static size_t write_aud_hevc(uint8_t *buf, size_t buf_size, bool is_idr) {
    uint8_t rbsp[8];
    bitstream_t bs;
    bs_init(&bs, rbsp, sizeof(rbsp));
    /* pic_type: 0 for I-slices only, 1 for P and I slices */
    bs_write_u(&bs, 3, is_idr ? 0 : 1);
    bs_rbsp_trailing_bits(&bs);

    bitstream_t out_bs;
    bs_init(&out_bs, buf, buf_size);
    bs_write_nal_header_hevc(&out_bs, NAL_UNIT_AUD);
    size_t off = bs_bytes_written(&out_bs);
    if (off >= buf_size) return 0;
    return off + bs_rbsp_to_ebsp(buf + off, buf_size - off, rbsp, bs_bytes_written(&bs));
}

static size_t write_vps(uint8_t *buf, size_t buf_size) {
    uint8_t rbsp[128];
    bitstream_t bs;
    bs_init(&bs, rbsp, sizeof(rbsp));

    bs_write_u(&bs, 4, 0);   /* vps_video_parameter_set_id */
    bs_write_u(&bs, 2, 3);   /* vps_base_layer_internal/available_flag */
    bs_write_u(&bs, 6, 0);   /* vps_max_layers_minus1 */
    bs_write_u(&bs, 3, 0);   /* vps_max_sub_layers_minus1 */
    bs_write1(&bs, 1);       /* vps_temporal_id_nesting_flag */
    bs_write_u(&bs, 16, 0xffff);

    write_profile_tier_level(&bs, 30); /* level is irrelevant here; SPS carries the real one */

    bs_write1(&bs, 1); /* vps_sub_layer_ordering_info_present_flag */
    bs_write_ue(&bs, 1); /* vps_max_dec_pic_buffering_minus1 = 1 (1 ref + 1 current pic) */
    bs_write_ue(&bs, 0); /* vps_num_reorder_pics */
    bs_write_ue(&bs, 0); /* vps_max_latency_increase_plus1 */

    bs_write_u(&bs, 6, 0); /* vps_max_nuh_reserved_zero_layer_id */
    bs_write_ue(&bs, 0);   /* vps_max_op_sets_minus1 */
    bs_write1(&bs, 0);     /* vps_timing_info_present_flag */
    bs_write1(&bs, 0);     /* vps_extension_flag */

    bs_rbsp_trailing_bits(&bs);

    bitstream_t out_bs;
    bs_init(&out_bs, buf, buf_size);
    bs_write_nal_header_hevc(&out_bs, NAL_UNIT_VPS);
    size_t off = bs_bytes_written(&out_bs);
    if (off >= buf_size) return 0;
    return off + bs_rbsp_to_ebsp(buf + off, buf_size - off, rbsp, bs_bytes_written(&bs));
}

static size_t write_sps(uint8_t *buf, size_t buf_size, uint32_t coded_w, uint32_t coded_h,
                         uint32_t real_w, uint32_t real_h, int level_idc) {
    uint8_t rbsp[256];
    bitstream_t bs;
    bs_init(&bs, rbsp, sizeof(rbsp));

    bs_write_u(&bs, 4, 0);  /* sps_video_parameter_set_id */
    bs_write_u(&bs, 3, 0);  /* sps_max_sub_layers_minus1 */
    bs_write1(&bs, 1);      /* sps_temporal_id_nesting_flag */

    write_profile_tier_level(&bs, level_idc);

    bs_write_ue(&bs, 0); /* sps_seq_parameter_set_id */
    bs_write_ue(&bs, 1); /* chroma_format_idc = 1 (4:2:0) */

    bs_write_ue(&bs, coded_w);
    bs_write_ue(&bs, coded_h);

    int need_crop = (coded_w != real_w) || (coded_h != real_h);
    bs_write1(&bs, need_crop ? 1 : 0);
    if (need_crop) {
        bs_write_ue(&bs, 0);
        bs_write_ue(&bs, (coded_w - real_w) / 2);
        bs_write_ue(&bs, 0);
        bs_write_ue(&bs, (coded_h - real_h) / 2);
    }

    bs_write_ue(&bs, 0); /* bit_depth_luma_minus8 */
    bs_write_ue(&bs, 0); /* bit_depth_chroma_minus8 */
    bs_write_ue(&bs, 4); /* log2_max_pic_order_cnt_lsb_minus4 = 4 -> log2=8 (0..255) */

    bs_write1(&bs, 1); /* sps_sub_layer_ordering_info_present_flag */
    bs_write_ue(&bs, 1); /* sps_max_dec_pic_buffering_minus1 = 1 (1 ref + 1 current pic) */
    bs_write_ue(&bs, 0); /* sps_num_reorder_pics */
    bs_write_ue(&bs, 0); /* sps_max_latency_increase_plus1 */

    bs_write_ue(&bs, 0); /* log2_min_luma_coding_block_size_minus3 -> MinCb = 8 */
    bs_write_ue(&bs, 1); /* log2_diff_max_min_coding_block_size -> Ctb = 16 */
    bs_write_ue(&bs, 0); /* log2_min_luma_transform_block_size_minus2 -> MinTb = 4 */
    bs_write_ue(&bs, 0); /* log2_diff_max_min_transform_block_size -> MaxTb = MinTb = 4 */
    bs_write_ue(&bs, 0); /* max_transform_hierarchy_depth_inter */
    bs_write_ue(&bs, 0); /* max_transform_hierarchy_depth_intra (IntraSplitFlag adds +1 -> MaxTrafoDepth=1) */

    bs_write1(&bs, 0); /* scaling_list_enabled_flag */
    bs_write1(&bs, 0); /* amp_enabled_flag */
    bs_write1(&bs, 0); /* sample_adaptive_offset_enabled_flag */
    bs_write1(&bs, 0); /* pcm_enabled_flag */

    bs_write_ue(&bs, 1); /* num_short_term_ref_pic_sets = 1 */
    /* short_term_ref_pic_set(0) per Rec. ITU-T H.265 7.3.7 */
    bs_write_ue(&bs, 1); /* num_negative_pics = 1 */
    bs_write_ue(&bs, 0); /* num_positive_pics = 0 */
    bs_write_ue(&bs, 0); /* delta_poc_s0_minus1[0] = 0 -> DeltaPoc = -(0+1) = -1 */
    bs_write1(&bs, 1);   /* used_by_curr_pic_s0_flag[0] = 1 */

    bs_write1(&bs, 0);   /* long_term_ref_pics_present_flag */
    bs_write1(&bs, 0);   /* sps_temporal_mvp_enable_flag */
    bs_write1(&bs, 0);   /* sps_strong_intra_smoothing_enable_flag */
    bs_write1(&bs, 0);   /* vui_parameters_present_flag */
    bs_write1(&bs, 0);   /* sps_extension_present_flag */

    bs_rbsp_trailing_bits(&bs);

    bitstream_t out_bs;
    bs_init(&out_bs, buf, buf_size);
    bs_write_nal_header_hevc(&out_bs, NAL_UNIT_SPS);
    size_t off = bs_bytes_written(&out_bs);
    if (off >= buf_size) return 0;
    return off + bs_rbsp_to_ebsp(buf + off, buf_size - off, rbsp, bs_bytes_written(&bs));
}

static size_t write_pps(uint8_t *buf, size_t buf_size, int init_qp) {
    uint8_t rbsp[64];
    bitstream_t bs;
    bs_init(&bs, rbsp, sizeof(rbsp));

    bs_write_ue(&bs, 0); /* pps_pic_parameter_set_id */
    bs_write_ue(&bs, 0); /* pps_seq_parameter_set_id */
    bs_write1(&bs, 0);   /* dependent_slice_segments_enabled_flag */
    bs_write1(&bs, 0);   /* output_flag_present_flag */
    bs_write_u(&bs, 3, 0); /* num_extra_slice_header_bits */
    bs_write1(&bs, 0);   /* sign_data_hiding_flag */
    bs_write1(&bs, 0);   /* cabac_init_present_flag */
    bs_write_ue(&bs, 0); /* num_ref_idx_l0_default_active_minus1 */
    bs_write_ue(&bs, 0); /* num_ref_idx_l1_default_active_minus1 */
    bs_write_se(&bs, init_qp - 26); /* init_qp_minus26 */
    bs_write1(&bs, 0);   /* constrained_intra_pred_flag */
    bs_write1(&bs, 0);   /* transform_skip_enabled_flag */
    bs_write1(&bs, 0);   /* cu_qp_delta_enabled_flag */
    bs_write_se(&bs, 0); /* pps_cb_qp_offset */
    bs_write_se(&bs, 0); /* pps_cr_qp_offset */
    bs_write1(&bs, 0);   /* pps_slice_chroma_qp_offsets_present_flag */
    bs_write1(&bs, 0);   /* weighted_pred_flag */
    bs_write1(&bs, 0);   /* weighted_bipred_flag */
    bs_write1(&bs, 0);   /* transquant_bypass_enable_flag */
    bs_write1(&bs, 0);   /* tiles_enabled_flag */
    bs_write1(&bs, 0);   /* entropy_coding_sync_enabled_flag */
    bs_write1(&bs, 0);   /* pps_loop_filter_across_slices_enabled_flag = 0 */
    bs_write1(&bs, 1);   /* deblocking_filter_control_present_flag = 1 (we need to disable deblock) */
    bs_write1(&bs, 0);   /* deblocking_filter_override_enabled_flag = 0 */
    bs_write1(&bs, 1);   /* pps_deblocking_filter_disabled_flag = 1 (encoder has no deblock filter;
                           * leaving this enabled causes reference-frame mismatch drift on P-frames
                           * because the decoder deblocks its reference but our encoder doesn't) */
    bs_write1(&bs, 0);   /* pps_scaling_list_data_present_flag */
    bs_write1(&bs, 0);   /* lists_modification_present_flag */
    bs_write_ue(&bs, 0); /* log2_parallel_merge_level_minus2 */
    bs_write1(&bs, 0);   /* slice_segment_header_extension_present_flag */
    bs_write1(&bs, 0);   /* pps_extension_present_flag */

    bs_rbsp_trailing_bits(&bs);

    bitstream_t out_bs;
    bs_init(&out_bs, buf, buf_size);
    bs_write_nal_header_hevc(&out_bs, NAL_UNIT_PPS);
    size_t off = bs_bytes_written(&out_bs);
    if (off >= buf_size) return 0;
    return off + bs_rbsp_to_ebsp(buf + off, buf_size - off, rbsp, bs_bytes_written(&bs));
}

/* ============================================================================
 * Encoder state
 * ==========================================================================*/

struct hevc_encoder {
    bc250_gpu_context_t *gpu;
    uint32_t width, height;               /* real (as requested by libva) */
    uint32_t coded_width, coded_height;   /* rounded up to a 16px CTU multiple */
    uint32_t width_ctu, height_ctu;
    uint32_t fps, bitrate;
    uint32_t frame_count;
    uint32_t gop_size;
    uint32_t poc;
    bool     force_idr;
    bool     has_ref;
    int qp;
    int pps_init_qp;
    int qp_hint_applied;         /* Last QP explicitly handed to hevc_encoder_set_qp(), or -1 if never called yet */
    rate_control_t rc;
    uint32_t quality_level;      /* 1..7 (1 = Quality, 4 = Balanced, 7 = Speed) */
    uint32_t max_frame_bits;     /* Maximum frame size in bits (0 = unlimited) */

    /* Source (post-download, padded/replicated to coded dimensions) and
     * reconstructed planes. Luma at coded_w x coded_h; chroma at
     * coded_w/2 x coded_h/2 (4:2:0). */
    uint8_t *src_y, *src_cb, *src_cr;
    uint8_t *recon_y, *recon_cb, *recon_cr;
    uint8_t *prev_recon_y, *prev_recon_cb, *prev_recon_cr;

    /* Per-CU skip tracking for current frame (for condL/condA context derivation).
     * Size: (width_ctu * 2) * (height_ctu * 2). */
    uint8_t *cu_skip_map;

    /* Inter prediction & motion vector maps (for spatial merge candidate derivation).
     * Size: (width_ctu * 2) * (height_ctu * 2). MVs in 1/4-pel units. */
    uint8_t *cu_is_inter;
    int16_t *mv_x_map;
    int16_t *mv_y_map;
    uint32_t last_frame_sad;

    /* GPU compute motion vector readback for acceleration */
    gpu_mv_t *gpu_mvs;
    uint32_t num_gpu_mvs;

    /* Real per-4x4-luma-PU intra mode, for MPM derivation - one entry per
     * 4x4 position, persistent scratch (positional availability checks
     * gate every read, so stale cross-frame content is never read - see
     * hevc_derive_mpm() call sites below). */
    int8_t *luma_mode_map;
    uint32_t mode_map_stride;

    /* Raw NV12 download scratch, real width x height. */
    uint8_t *dl_y;
    uint8_t *dl_uv;

    uint8_t *slice_rbsp;
    size_t   slice_rbsp_cap;

    /* One slice per thread's worth of state. A slice is a whole number of CTU
     * rows, which is what makes a slice boundary also a CTU-row boundary - the
     * MPM derivation already refuses to look above a CTU row, so that part was
     * slice-safe before this existed. */
    int      num_slices;
    uint8_t *slice_buf[HEVC_MAX_SLICES];
    size_t   slice_len[HEVC_MAX_SLICES];
    uint32_t slice_sad[HEVC_MAX_SLICES];

    size_t   slice_buf_cap;
    uint8_t *scratch_out;
    size_t   scratch_out_cap;

    /* Dynamic asymmetric CPU/GPU load balancing governor & SIMD ME config */
    dynamic_governor_t governor;
    /* Frames the governor has told us to keep off the GPU, counted so one
     * in every step_down_hysteresis can go anyway and bring back a
     * measurement. See hevc_encoder_encode_frame(). */
    uint32_t governor_skips;
    cpu_simd_me_config_t me_cfg;
};

static uint32_t round_up16(uint32_t v) { return (v + 15u) & ~15u; }

hevc_encoder_t *hevc_encoder_create(bc250_gpu_context_t *gpu_ctx,
                                    uint32_t width, uint32_t height,
                                    uint32_t fps, uint32_t bitrate)
{
    if (width == 0 || height == 0) return NULL;
    hevc_encoder_t *enc = calloc(1, sizeof(hevc_encoder_t));
    if (!enc) return NULL;

    enc->gpu = gpu_ctx;
    enc->width = width;
    enc->height = height;
    enc->fps = fps ? fps : 30;
    enc->bitrate = bitrate;
    enc->qp = 27;
    bool qp_pinned = false;
    {
        const char *qp_env = getenv("BC250_HEVC_QP");
        if (qp_env) {
            int q = atoi(qp_env);
            if (q >= 1 && q <= 51) { enc->qp = q; qp_pinned = true; }
        }
    }
    enc->pps_init_qp = enc->qp;
    /* The rate control was built and then switched off.
     *
     * encode_core() already asks rc_get_frame_qp() for this frame's QP, the
     * tail already calls rc_update_stats() with the bits produced, and the
     * slice header already writes slice_qp_delta so a per-frame QP reaches
     * the decoder. All three are guarded by `rc.mode != RC_CQP`, and this
     * call passed RC_CQP - so none of them ever ran: every frame went out at
     * QP 27 no matter what bitrate the caller asked for, and asking for more
     * bitrate changed nothing.
     *
     * Measured on a BC-250 at 1920x1080, testsrc, 8 Mbit/s requested: the
     * stream came out at 1.5 Mbit/s and 25.3 dB PSNR, where h264_vaapi on the
     * same clip and the same request gives 51.4 dB. The H.264 path defaults
     * to RC_LOW_LATENCY (see its rc_init() call), so HEVC now does the same.
     *
     * BC250_HEVC_QP still pins the QP: somebody who names a QP is asking for
     * constant QP, and that is what RC_CQP is for.
     */
    rc_init(&enc->rc, qp_pinned ? RC_CQP : RC_LOW_LATENCY, bitrate,
            (double)enc->fps, width, height);
    enc->rc.current_qp = enc->qp;
    enc->rc.base_qp = enc->qp;
    enc->qp_hint_applied = -1; /* no explicit QP hint applied yet - see hevc_encoder_set_qp() */
    enc->quality_level = 4;
    enc->max_frame_bits = 0;

    enc->gop_size = enc->fps;
    {
        const char *gop_env = getenv("BC250_HEVC_GOP");
        if (gop_env) {
            int g = atoi(gop_env);
            if (g >= 1) enc->gop_size = (uint32_t)g;
        }
    }
    enc->poc = 0;
    enc->force_idr = false;
    enc->has_ref = false;

    enc->coded_width = round_up16(width);
    enc->coded_height = round_up16(height);
    enc->width_ctu = enc->coded_width / HEVC_CTU_SIZE;
    enc->height_ctu = enc->coded_height / HEVC_CTU_SIZE;

    size_t luma_size = (size_t)enc->coded_width * enc->coded_height;
    size_t chroma_size = (size_t)(enc->coded_width / 2) * (enc->coded_height / 2);

    enc->src_y = malloc(luma_size);
    enc->src_cb = malloc(chroma_size);
    enc->src_cr = malloc(chroma_size);
    enc->recon_y = malloc(luma_size);
    enc->recon_cb = malloc(chroma_size);
    enc->recon_cr = malloc(chroma_size);
    enc->prev_recon_y = malloc(luma_size);
    enc->prev_recon_cb = malloc(chroma_size);
    enc->prev_recon_cr = malloc(chroma_size);

    size_t num_cus = (size_t)(enc->width_ctu * 2) * (enc->height_ctu * 2);
    enc->cu_skip_map = calloc(num_cus, 1);
    enc->cu_is_inter = calloc(num_cus, 1);
    enc->mv_x_map = calloc(num_cus, sizeof(int16_t));
    enc->mv_y_map = calloc(num_cus, sizeof(int16_t));

    enc->mode_map_stride = enc->coded_width / HEVC_PU_SIZE;
    enc->luma_mode_map = malloc((size_t)enc->mode_map_stride * (enc->coded_height / HEVC_PU_SIZE));

    enc->dl_y = malloc((size_t)width * height);
    enc->dl_uv = malloc((size_t)(width / 2) * (height / 2) * 2);

    enc->slice_rbsp_cap = luma_size + 65536;
    enc->slice_rbsp = malloc(enc->slice_rbsp_cap);

    /* Four by default, measured rather than guessed.
     *
     * On a BC-250 at 1920x1080 at 60 fps, testsrc, 8 Mbit/s requested:
     *    1 slice    78.2 fps   46.41 dB   4.09 Mbit/s
     *    4 slices  111.2 fps   45.83 dB   3.85 Mbit/s
     *    8 slices  118.9 fps   45.63 dB   3.78 Mbit/s
     *   16 slices  138.7 fps   44.74 dB   3.31 Mbit/s
     * Prediction restarts at every slice boundary, so more slices cost
     * quality. Four buys 42% more speed for half a dB, which on a box whose
     * job is streaming a game while the game is running is a trade worth
     * making; sixteen gives up too much for the rest. BC250_HEVC_SLICES
     * overrides it, and 1 restores exactly the old single-slice bitstream. */
    enc->num_slices = 4;
    if (enc->num_slices > (int)enc->height_ctu) enc->num_slices = (int)enc->height_ctu;
    {
        const char *s = getenv("BC250_HEVC_SLICES");
        if (s) {
            int n = atoi(s);
            if (n >= 1 && n <= HEVC_MAX_SLICES) enc->num_slices = n;
            if (enc->num_slices > (int)enc->height_ctu) enc->num_slices = (int)enc->height_ctu;
        }
    }
    {
        size_t per = luma_size / (size_t)enc->num_slices + 262144;
        for (int i = 0; i < enc->num_slices; i++) enc->slice_buf[i] = malloc(per);
        enc->slice_buf_cap = per;
    }

    enc->scratch_out_cap = luma_size + 131072;
    enc->scratch_out = malloc(enc->scratch_out_cap);

    size_t num_mbs = (size_t)enc->width_ctu * enc->height_ctu;
    enc->gpu_mvs = calloc(num_mbs, sizeof(gpu_mv_t));

    if (!enc->src_y || !enc->src_cb || !enc->src_cr ||
        !enc->recon_y || !enc->recon_cb || !enc->recon_cr ||
        !enc->prev_recon_y || !enc->prev_recon_cb || !enc->prev_recon_cr ||
        !enc->cu_skip_map || !enc->cu_is_inter || !enc->mv_x_map || !enc->mv_y_map ||
        !enc->luma_mode_map || !enc->dl_y || !enc->dl_uv ||
        !enc->slice_rbsp || !enc->scratch_out || !enc->gpu_mvs) {
        hevc_encoder_destroy(enc);
        return NULL;
    }

    dynamic_governor_init(&enc->governor);
    cpu_simd_me_config_init(&enc->me_cfg, width, height);

    return enc;
}

void hevc_encoder_set_force_idr(hevc_encoder_t *encoder)
{
    if (encoder) encoder->force_idr = true;
}

void hevc_encoder_set_gop_size(hevc_encoder_t *encoder, uint32_t gop_size)
{
    if (encoder && gop_size >= 1) encoder->gop_size = gop_size;
}

uint32_t hevc_encoder_get_gop_size(const hevc_encoder_t *encoder)
{
    return encoder ? encoder->gop_size : 30;
}

void hevc_encoder_set_qp(hevc_encoder_t *encoder, int qp)
{
    if (encoder) {
        if (qp < 0) qp = 0;
        if (qp > 51) qp = 51;
        /* va_backend.c calls this for EVERY picture with pic_init_qp out of
         * VAEncPictureParameterBufferHEVC. To prevent per-frame unchanged hints
         * from stomping the rate controller's QP walk, only reset base_qp/current_qp
         * when the hint genuinely changes. */
        if (qp != encoder->qp_hint_applied) {
            encoder->rc.base_qp = qp;
            encoder->rc.current_qp = qp;
            encoder->qp_hint_applied = qp;
        }
        encoder->qp = qp;
    }
}

int hevc_encoder_get_qp(const hevc_encoder_t *encoder)
{
    return encoder ? encoder->qp : 27;
}

void hevc_encoder_set_bitrate(hevc_encoder_t *encoder, uint32_t bitrate)
{
    if (encoder && bitrate > 0 && bitrate != encoder->rc.target_bitrate) {
        encoder->bitrate = bitrate;
        rc_init(&encoder->rc, encoder->rc.mode, bitrate, (double)encoder->fps,
                encoder->width, encoder->height);
    }
}

uint32_t hevc_encoder_get_bitrate(const hevc_encoder_t *encoder)
{
    return encoder ? encoder->bitrate : 0;
}

void hevc_encoder_set_fps(hevc_encoder_t *encoder, uint32_t fps)
{
    if (encoder && fps > 0 && fps != encoder->fps) {
        encoder->fps = fps;
        rc_init(&encoder->rc, encoder->rc.mode, encoder->rc.target_bitrate,
                (double)fps, encoder->width, encoder->height);
    }
}

uint32_t hevc_encoder_get_fps(const hevc_encoder_t *encoder)
{
    return encoder ? encoder->fps : 30;
}

void hevc_encoder_set_rc_mode(hevc_encoder_t *encoder, rc_mode_t mode)
{
    if (encoder) {
        encoder->rc.mode = mode;
        if (mode == RC_LOW_LATENCY) {
            encoder->rc.buffer_size = encoder->rc.target_bits_per_frame * 2;
        } else if (mode == RC_CBR || mode == RC_VBR) {
            encoder->rc.buffer_size = encoder->rc.target_bitrate;
        }
        if (encoder->rc.buffer_size < 1000) encoder->rc.buffer_size = 1000;
        encoder->rc.buffer_fullness = encoder->rc.buffer_size / 2;
        encoder->rc.error_integral = 0;
    }
}

rc_mode_t hevc_encoder_get_rc_mode(const hevc_encoder_t *encoder)
{
    return encoder ? encoder->rc.mode : RC_CQP;
}

uint32_t hevc_encoder_get_last_frame_sad(const hevc_encoder_t *encoder)
{
    return encoder ? encoder->last_frame_sad : 0;
}

void hevc_encoder_set_quality_level(hevc_encoder_t *encoder, uint32_t quality_level)
{
    if (encoder) {
        if (quality_level < 1) quality_level = 1;
        if (quality_level > 7) quality_level = 7;
        encoder->quality_level = quality_level;
        rc_set_quality_level(&encoder->rc, quality_level);
    }
}

uint32_t hevc_encoder_get_quality_level(const hevc_encoder_t *encoder)
{
    return encoder ? encoder->quality_level : 4;
}

void hevc_encoder_set_max_frame_size(hevc_encoder_t *encoder, uint32_t max_frame_bits)
{
    if (encoder) {
        encoder->max_frame_bits = max_frame_bits;
        rc_set_max_frame_size(&encoder->rc, max_frame_bits);
    }
}

uint32_t hevc_encoder_get_max_frame_size(const hevc_encoder_t *encoder)
{
    return encoder ? encoder->max_frame_bits : 0;
}

int hevc_encoder_get_governor_tier(const hevc_encoder_t *encoder)
{
    return encoder ? (int)dynamic_governor_get_tier(&encoder->governor) : 0;
}

void hevc_encoder_destroy(hevc_encoder_t *encoder)
{
    if (!encoder) return;
    free(encoder->src_y); free(encoder->src_cb); free(encoder->src_cr);
    free(encoder->recon_y); free(encoder->recon_cb); free(encoder->recon_cr);
    free(encoder->prev_recon_y); free(encoder->prev_recon_cb); free(encoder->prev_recon_cr);
    free(encoder->cu_skip_map);
    free(encoder->cu_is_inter);
    free(encoder->mv_x_map);
    free(encoder->mv_y_map);
    free(encoder->luma_mode_map);
    free(encoder->dl_y); free(encoder->dl_uv);
    for (int i = 0; i < encoder->num_slices; i++) free(encoder->slice_buf[i]);
    free(encoder->slice_rbsp);
    free(encoder->scratch_out);
    free(encoder->gpu_mvs);
    free(encoder);
}

/* Replicate-pad a downloaded plane (real w x h) into a coded_w x coded_h
 * working buffer - only the bottom/right margin (if any) needs padding,
 * since coded dims are always >= real dims by construction. */
static void pad_replicate(uint8_t *dst, uint32_t dst_w, uint32_t dst_h,
                           const uint8_t *src, uint32_t src_stride, uint32_t src_w, uint32_t src_h) {
    for (uint32_t y = 0; y < dst_h; y++) {
        uint32_t sy = y < src_h ? y : src_h - 1;
        const uint8_t *srow = src + (size_t)sy * src_stride;
        uint8_t *drow = dst + (size_t)y * dst_w;
        for (uint32_t x = 0; x < dst_w; x++) {
            uint32_t sx = x < src_w ? x : src_w - 1;
            drow[x] = srow[sx];
        }
    }
}

/* ============================================================================
 * Per-CU encoding
 * ==========================================================================*/

static const int pu_off_x[4] = { 0, 4, 0, 4 };
static const int pu_off_y[4] = { 0, 0, 4, 4 };

static int any_nonzero16(const int16_t *c) {
    for (int i = 0; i < 16; i++) if (c[i]) return 1;
    return 0;
}

typedef struct {
    int16_t x;
    int16_t y;
} hevc_mv_t;

static inline uint32_t compute_sad_8x8_luma(const uint8_t *src_y,
                                            const uint8_t *ref_y,
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

static inline uint32_t compute_sad_4x4_chroma(const uint8_t *src_cb,
                                              const uint8_t *src_cr,
                                              const uint8_t *ref_cb,
                                              const uint8_t *ref_cr,
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

static inline uint32_t hevc_cu_rank(uint32_t width_ctu, int cux, int cuy) {
    uint32_t ctu_col = (uint32_t)cux / 2;
    uint32_t ctu_row = (uint32_t)cuy / 2;
    uint32_t cu_sub = ((uint32_t)cuy & 1) * 2 + ((uint32_t)cux & 1);
    return (ctu_row * width_ctu + ctu_col) * 4 + cu_sub;
}

static inline bool hevc_cu_is_available(uint32_t width_ctu, uint32_t height_ctu,
                                        int cur_cux, int cur_cuy,
                                        int nb_cux, int nb_cuy, int cuy_min)
{
    if (nb_cux < 0 || nb_cuy < cuy_min) return false;
    if (nb_cux >= (int)(width_ctu * 2) || nb_cuy >= (int)(height_ctu * 2)) return false;
    uint32_t cur_rank = hevc_cu_rank(width_ctu, cur_cux, cur_cuy);
    uint32_t nb_rank = hevc_cu_rank(width_ctu, nb_cux, nb_cuy);
    return nb_rank < cur_rank;
}

/* Derives spatial merge candidates matching ITU-T H.265 Section 8.5.3.2.2.
 * Output cand_mvs has exactly 5 candidates (padded with (0,0)), in 1/4-pel units.
 * All candidates are strictly derived from spatial neighbors or zero-vectors,
 * ensuring 100% bit-exact candidate derivation matching hardware decoders. */
static int derive_merge_candidates(int cuy_min, const hevc_encoder_t *enc,
                                   int cux, int cuy,
                                   hevc_mv_t cand_mvs[5])
{
    uint32_t w_cu = enc->width_ctu * 2;
    uint32_t h_cu = enc->height_ctu * 2;

    hevc_mv_t spatial_cand[5];
    int num_spatial = 0;

    /* 1. Candidate A1 (Left): (cux - 1, cuy) */
    bool a1_has_inter = false;
    hevc_mv_t mv_a1 = {0, 0};
    if (hevc_cu_is_available(enc->width_ctu, enc->height_ctu, cux, cuy, cux - 1, cuy, cuy_min)) {
        uint32_t a1_idx = (uint32_t)cuy * w_cu + (uint32_t)(cux - 1);
        if (enc->cu_is_inter[a1_idx]) {
            a1_has_inter = true;
            mv_a1.x = enc->mv_x_map[a1_idx];
            mv_a1.y = enc->mv_y_map[a1_idx];
            spatial_cand[num_spatial++] = mv_a1;
        }
    }

    /* 2. Candidate B1 (Above): (cux, cuy - 1) */
    bool b1_has_inter = false;
    hevc_mv_t mv_b1 = {0, 0};
    if (hevc_cu_is_available(enc->width_ctu, enc->height_ctu, cux, cuy, cux, cuy - 1, cuy_min)) {
        uint32_t b1_idx = (uint32_t)(cuy - 1) * w_cu + (uint32_t)cux;
        if (enc->cu_is_inter[b1_idx]) {
            b1_has_inter = true;
            mv_b1.x = enc->mv_x_map[b1_idx];
            mv_b1.y = enc->mv_y_map[b1_idx];
            /* Pruning: B1 against A1 */
            if (!a1_has_inter || mv_b1.x != mv_a1.x || mv_b1.y != mv_a1.y) {
                spatial_cand[num_spatial++] = mv_b1;
            }
        }
    }

    /* 3. Candidate B0 (Above-Right): (cux + 1, cuy - 1) */
    bool b0_has_inter = false;
    hevc_mv_t mv_b0 = {0, 0};
    if (hevc_cu_is_available(enc->width_ctu, enc->height_ctu, cux, cuy, cux + 1, cuy - 1, cuy_min)) {
        uint32_t b0_idx = (uint32_t)(cuy - 1) * w_cu + (uint32_t)(cux + 1);
        if (enc->cu_is_inter[b0_idx]) {
            b0_has_inter = true;
            mv_b0.x = enc->mv_x_map[b0_idx];
            mv_b0.y = enc->mv_y_map[b0_idx];
            /* Pruning: B0 against B1 */
            if (!b1_has_inter || mv_b0.x != mv_b1.x || mv_b0.y != mv_b1.y) {
                spatial_cand[num_spatial++] = mv_b0;
            }
        }
    }

    /* 4. Candidate A0 (Below-Left): (cux - 1, cuy + 1) */
    bool a0_has_inter = false;
    hevc_mv_t mv_a0 = {0, 0};
    if (hevc_cu_is_available(enc->width_ctu, enc->height_ctu, cux, cuy, cux - 1, cuy + 1, cuy_min)) {
        uint32_t a0_idx = (uint32_t)(cuy + 1) * w_cu + (uint32_t)(cux - 1);
        if (enc->cu_is_inter[a0_idx]) {
            a0_has_inter = true;
            mv_a0.x = enc->mv_x_map[a0_idx];
            mv_a0.y = enc->mv_y_map[a0_idx];
            /* Pruning: A0 against A1 */
            if (!a1_has_inter || mv_a0.x != mv_a1.x || mv_a0.y != mv_a1.y) {
                spatial_cand[num_spatial++] = mv_a0;
            }
        }
    }

    /* 5. Candidate B2 (Above-Left): (cux - 1, cuy - 1) */
    if (num_spatial < 4) {
        if (hevc_cu_is_available(enc->width_ctu, enc->height_ctu, cux, cuy, cux - 1, cuy - 1, cuy_min)) {
            uint32_t b2_idx = (uint32_t)(cuy - 1) * w_cu + (uint32_t)(cux - 1);
            if (enc->cu_is_inter[b2_idx]) {
                hevc_mv_t mv_b2;
                mv_b2.x = enc->mv_x_map[b2_idx];
                mv_b2.y = enc->mv_y_map[b2_idx];
                /* Pruning: B2 against A1 and B1 */
                if ((!a1_has_inter || mv_b2.x != mv_a1.x || mv_b2.y != mv_a1.y) &&
                    (!b1_has_inter || mv_b2.x != mv_b1.x || mv_b2.y != mv_b1.y)) {
                    spatial_cand[num_spatial++] = mv_b2;
                }
            }
        }
    }

    int num_cand = 0;
    for (int i = 0; i < num_spatial && num_cand < 5; i++) {
        cand_mvs[num_cand++] = spatial_cand[i];
    }

    while (num_cand < 5) {
        cand_mvs[num_cand].x = 0;
        cand_mvs[num_cand].y = 0;
        num_cand++;
    }

    return num_cand;
}

static void encode_cu(hevc_encoder_t *enc, hevc_cabac_t *cab, int cu_x, int cu_y, bool is_idr, int y_min, uint32_t *sad_out) {
    int qp = enc->qp;
    uint32_t cw = enc->coded_width, ch = enc->coded_height;
    uint32_t ccw = cw / 2, cch = ch / 2;
    int cux = cu_x / HEVC_CU_SIZE;
    int cuy = cu_y / HEVC_CU_SIZE;
    uint32_t cu_stride = enc->width_ctu * 2;
    uint32_t cu_idx = (uint32_t)cuy * cu_stride + (uint32_t)cux;

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
            uint32_t c_sad = compute_sad_8x8_luma(enc->src_y, enc->prev_recon_y, cw, cu_x, cu_y, c_dx, c_dy) +
                             compute_sad_4x4_chroma(enc->src_cb, enc->src_cr,
                                                    enc->prev_recon_cb, enc->prev_recon_cr,
                                                    ccw, cx, cy, cdx, cdy);
            if (c_sad < best_cand_sad) {
                best_cand_sad = c_sad;
                best_cand_idx = i;
                /* Fast path: stationary (0,0) or perfect match exits immediately */
                if (c_dx == 0 && c_dy == 0 && (best_cand_sad <= 32 || (gpu_says_static && best_cand_sad <= 96))) {
                    break;
                }
            }
        }

        uint32_t threshold = 96 * (1 + (enc->qp / 8));
        if (enc->quality_level >= 4) {
            threshold = threshold * 2;
        }
        static int s_skip_override = -2;
        if (s_skip_override == -2) {
            const char *env = getenv("BC250_HEVC_SKIP_THRESHOLD");
            s_skip_override = env ? atoi(env) : -1;
        }
        if (s_skip_override >= 0) {
            threshold = (uint32_t)s_skip_override;
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
            memcpy(&enc->recon_y[(cu_y + y) * cw + cu_x],
                   &enc->prev_recon_y[(cu_y + chosen_dy + y) * cw + (cu_x + chosen_dx)],
                   HEVC_CU_SIZE);
        }
        int cx = cu_x / 2, cy = cu_y / 2;
        int cdx = (chosen_dx >= 0) ? (chosen_dx >> 1) : ((chosen_dx - 1) >> 1);
        int cdy = (chosen_dy >= 0) ? (chosen_dy >> 1) : ((chosen_dy - 1) >> 1);
        for (int y = 0; y < HEVC_PU_SIZE; y++) {
            memcpy(&enc->recon_cb[(cy + y) * ccw + cx],
                   &enc->prev_recon_cb[(cy + cdy + y) * ccw + (cx + cdx)],
                   HEVC_PU_SIZE);
            memcpy(&enc->recon_cr[(cy + y) * ccw + cx],
                   &enc->prev_recon_cr[(cy + cdy + y) * ccw + (cx + cdx)],
                   HEVC_PU_SIZE);
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
        uint8_t pred[16];
        int mode;
        if (enc->quality_level >= 4) {
            mode = HEVC_MODE_DC;
            hevc_predict_4x4(enc->recon_y, cw, cw, ch, px, py, mode, 1, y_min, pred);
        } else {
            mode = hevc_choose_luma_mode(y_min, enc->src_y, enc->recon_y, (int)cw, (int)cw, (int)ch, px, py, pred);
        }
        pu_modes[pu] = mode;

        int16_t residual[16];
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                residual[y * 4 + x] = (int16_t)(enc->src_y[(py + y) * cw + (px + x)] - pred[y * 4 + x]);

        int16_t coeff[16];
        hevc_transform_quant_4x4(residual, qp, 1 /* DST for 4x4 luma intra */, coeff);
        memcpy(luma_coeff[pu], coeff, sizeof(coeff));
        cbf_luma[pu] = any_nonzero16(coeff);

        if (cbf_luma[pu]) {
            int16_t recon_residual[16];
            hevc_dequant_itransform_4x4(coeff, qp, 1, recon_residual);
            for (int y = 0; y < 4; y++)
                for (int x = 0; x < 4; x++)
                    enc->recon_y[(py + y) * cw + (px + x)] = clip8i(pred[y * 4 + x] + recon_residual[y * 4 + x]);
        } else {
            for (int y = 0; y < 4; y++)
                memcpy(&enc->recon_y[(py + y) * cw + px], &pred[y * 4], 4);
        }

        enc->luma_mode_map[(py / 4) * enc->mode_map_stride + (px / 4)] = (int8_t)mode;
    }

    /* Chroma: one 4x4 Cb + one 4x4 Cr per CU, DC prediction only */
    int cx = cu_x / 2, cy = cu_y / 2;
    uint8_t pred_cb[16], pred_cr[16];
    hevc_predict_4x4(enc->recon_cb, ccw, ccw, cch, cx, cy, HEVC_MODE_DC, 0, y_min / 2, pred_cb);
    hevc_predict_4x4(enc->recon_cr, ccw, ccw, cch, cx, cy, HEVC_MODE_DC, 0, y_min / 2, pred_cr);

    int16_t res_cb[16], res_cr[16];
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) {
            res_cb[y * 4 + x] = (int16_t)(enc->src_cb[(cy + y) * ccw + (cx + x)] - pred_cb[y * 4 + x]);
            res_cr[y * 4 + x] = (int16_t)(enc->src_cr[(cy + y) * ccw + (cx + x)] - pred_cr[y * 4 + x]);
        }

    /* Chroma quantizes at QpC, not QpY - Rec. ITU-T H.265 Table 8-10.
     * Passing luma QP directly causes divergence from the standard when QP >= 30. */
    int cqp = hevc_chroma_qp_from_luma(qp);

    int16_t coeff_cb[16], coeff_cr[16];
    hevc_transform_quant_4x4(res_cb, cqp, 0, coeff_cb);
    hevc_transform_quant_4x4(res_cr, cqp, 0, coeff_cr);
    int cbf_cb = any_nonzero16(coeff_cb);
    int cbf_cr = any_nonzero16(coeff_cr);

    if (cbf_cb) {
        int16_t rres_cb[16];
        hevc_dequant_itransform_4x4(coeff_cb, cqp, 0, rres_cb);
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                enc->recon_cb[(cy + y) * ccw + (cx + x)] = clip8i(pred_cb[y * 4 + x] + rres_cb[y * 4 + x]);
    } else {
        for (int y = 0; y < 4; y++)
            memcpy(&enc->recon_cb[(cy + y) * ccw + cx], &pred_cb[y * 4], 4);
    }

    if (cbf_cr) {
        int16_t rres_cr[16];
        hevc_dequant_itransform_4x4(coeff_cr, cqp, 0, rres_cr);
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                enc->recon_cr[(cy + y) * ccw + (cx + x)] = clip8i(pred_cr[y * 4 + x] + rres_cr[y * 4 + x]);
    } else {
        for (int y = 0; y < 4; y++)
            memcpy(&enc->recon_cr[(cy + y) * ccw + cx], &pred_cr[y * 4], 4);
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

static void encode_ctu(hevc_encoder_t *enc, hevc_cabac_t *cab, int ctu_col, int ctu_row, bool is_idr, int y_min, uint32_t *sad_out) {
    int ctu_x = ctu_col * HEVC_CTU_SIZE, ctu_y = ctu_row * HEVC_CTU_SIZE;
    int cond_l = ctu_col > 0 ? 1 : 0;
    int cond_a = (ctu_row * HEVC_CTU_SIZE > y_min) ? 1 : 0;
    hevc_cabac_code_split_cu_flag(cab, 1, cond_l + cond_a);

    static const int cu_off_x[4] = { 0, 8, 0, 8 };
    static const int cu_off_y[4] = { 0, 0, 8, 8 };
    for (int i = 0; i < 4; i++)
        encode_cu(enc, cab, ctu_x + cu_off_x[i], ctu_y + cu_off_y[i], is_idr, y_min, sad_out);
}

/* ============================================================================
 * Frame entry point
 * ==========================================================================*/

/* Core encode: assumes encoder->dl_y / encoder->dl_uv (real width x height,
 * NV12: dl_y row-pitch == width, dl_uv row-pitch == width with Cb/Cr
 * interleaved) are already populated. Both hevc_encoder_encode_frame()
 * (GPU-surface readback) and hevc_encoder_encode_raw() (direct host
 * pointers, no GPU involved - see encoder_h265.h) fill those in their own
 * way and then share everything from here on. */
/* Split NV12's interleaved chroma into separate Cb and Cr planes.
 *
 * This ran one byte at a time: 518400 iterations per 1080p frame, inside
 * encode_core(), which the profiler put at the top of the HEVC encode once
 * the surface readback had been dealt with. PSHUFB gathers the even bytes
 * into one half of a register and the odd bytes into the other, so sixteen
 * samples of each plane come out per pair of loads.
 *
 * Runtime-dispatched like cpu_simd_me.c, and the scalar tail keeps widths
 * that are not a multiple of sixteen working.
 */
#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("ssse3")))
static void deinterleava_uv_ssse3(uint8_t *cb, uint8_t *cr,
                                  const uint8_t *uv, size_t n) {
    const __m128i sh = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14,
                                     1, 3, 5, 7, 9, 11, 13, 15);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i a = _mm_loadu_si128((const __m128i *)(uv + 2 * i));
        __m128i b = _mm_loadu_si128((const __m128i *)(uv + 2 * i + 16));
        a = _mm_shuffle_epi8(a, sh);
        b = _mm_shuffle_epi8(b, sh);
        _mm_storeu_si128((__m128i *)(cb + i), _mm_unpacklo_epi64(a, b));
        _mm_storeu_si128((__m128i *)(cr + i), _mm_unpackhi_epi64(a, b));
    }
    for (; i < n; i++) { cb[i] = uv[2 * i]; cr[i] = uv[2 * i + 1]; }
}
#endif

static void deinterleava_uv(uint8_t *cb, uint8_t *cr, const uint8_t *uv, size_t n) {
#if defined(__x86_64__) || defined(_M_X64)
    static int ha_ssse3 = -1;
    if (ha_ssse3 < 0) ha_ssse3 = __builtin_cpu_supports("ssse3") ? 1 : 0;
    if (ha_ssse3) { deinterleava_uv_ssse3(cb, cr, uv, n); return; }
#endif
    for (size_t i = 0; i < n; i++) { cb[i] = uv[2 * i]; cr[i] = uv[2 * i + 1]; }
}

static int encode_core(hevc_encoder_t *encoder, uint8_t *output_buf, size_t output_size)
{
    bool is_idr = (encoder->frame_count % encoder->gop_size == 0) || encoder->force_idr || !encoder->has_ref;
    encoder->force_idr = false;
    if (is_idr) {
        encoder->poc = 0;
    }

    /* In VBR/CBR/LOW_LATENCY mode, update QP via rate control model */
    if (encoder->rc.mode != RC_CQP) {
        int target_qp = rc_get_frame_qp(&encoder->rc, is_idr ? 0 : encoder->last_frame_sad);
        if (target_qp >= 1 && target_qp <= 51) {
            encoder->qp = target_qp;
        }
    }
    encoder->last_frame_sad = 0;

    pad_replicate(encoder->src_y, encoder->coded_width, encoder->coded_height,
                  encoder->dl_y, encoder->width, encoder->width, encoder->height);

    uint32_t cw2 = encoder->width / 2, ch2 = encoder->height / 2;
    uint32_t ccw = encoder->coded_width / 2, cch = encoder->coded_height / 2;
    for (uint32_t y = 0; y < ch2; y++) {
        const uint8_t *uvrow = encoder->dl_uv + (size_t)y * encoder->width;
        deinterleava_uv(&encoder->src_cb[y * ccw], &encoder->src_cr[y * ccw],
                        uvrow, cw2);
    }
    pad_replicate(encoder->src_cb, ccw, cch, encoder->src_cb, ccw, cw2, ch2);
    pad_replicate(encoder->src_cr, ccw, cch, encoder->src_cr, ccw, cw2, ch2);

    size_t num_cus = (size_t)(encoder->width_ctu * 2) * (encoder->height_ctu * 2);
    memset(encoder->luma_mode_map, 0, (size_t)encoder->mode_map_stride * (encoder->coded_height / HEVC_PU_SIZE));
    memset(encoder->cu_skip_map, 0, num_cus);
    memset(encoder->cu_is_inter, 0, num_cus);
    memset(encoder->mv_x_map, 0, num_cus * sizeof(int16_t));
    memset(encoder->mv_y_map, 0, num_cus * sizeof(int16_t));

    bool write_param_sets = is_idr;
    if (getenv("BC250_HEVC_REPEAT_HEADERS")) {
        write_param_sets = true;
    }
    if (write_param_sets) {
        encoder->pps_init_qp = encoder->qp;
    }
    int slice_qp_delta = encoder->qp - encoder->pps_init_qp;

    /* One slice per region of CTU rows.
     *
     * A slice is a whole number of CTU rows, so a slice boundary is also a CTU
     * row boundary - which is why the MPM derivation, which already refuses to
     * look above a CTU row, needed nothing. Everything else that reads a
     * neighbour takes y_min and stops there: the intra reference samples, the
     * merge candidates and the two CABAC contexts that ask whether the block
     * above was skipped.
     *
     * Each slice gets its own CABAC engine, its own buffer and its own SAD
     * accumulator, so nothing is shared while a slice is being encoded. That
     * is what lets the loop be handed to OpenMP.
     */
    const int ns = encoder->num_slices;
    const uint32_t ctu_rows = encoder->height_ctu;
    uint32_t bit_address = 0;
    {
        uint32_t n = encoder->width_ctu * encoder->height_ctu;
        while ((1u << bit_address) < n) bit_address++;
    }

    /* Nothing is shared while a slice is being encoded.
     *
     * Each slice has its own CABAC engine, its own output buffer, its own SAD
     * accumulator and its own region of the reconstruction and of the mode and
     * skip maps. Every neighbour lookup stops at y_min, so no slice ever reads
     * a pixel or a mode another slice is writing. The shared reads - the source
     * planes and the previous reconstruction - are read-only for the whole
     * frame.
     */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (ns > 1)
#endif
    for (int s = 0; s < ns; s++) {
        uint32_t r0 = (uint32_t)(((uint64_t)ctu_rows * (uint32_t)s) / (uint32_t)ns);
        uint32_t r1 = (uint32_t)(((uint64_t)ctu_rows * (uint32_t)(s + 1)) / (uint32_t)ns);
        int y_min = (int)(r0 * HEVC_CTU_SIZE);
        uint32_t sad = 0;

        bitstream_t slice_bs;
        bs_init(&slice_bs, encoder->slice_buf[s], encoder->slice_buf_cap);

        bs_write1(&slice_bs, s == 0 ? 1 : 0);   /* first_slice_segment_in_pic_flag */
        if (is_idr) {
            bs_write1(&slice_bs, 1);            /* no_output_of_prior_pics_flag */
        }
        bs_write_ue(&slice_bs, 0);              /* slice_pic_parameter_set_id */
        /* dependent_slice_segments_enabled_flag is 0 in the PPS, so no
         * dependent_slice_segment_flag here - just the address, in
         * Ceil(Log2(PicSizeInCtbsY)) bits, per Rec. ITU-T H.265 7.3.6.1. */
        if (s != 0) {
            bs_write_u(&slice_bs, (int)bit_address, r0 * encoder->width_ctu);
        }
        bs_write_ue(&slice_bs, is_idr ? 2 : 1); /* slice_type: 2 = I, 1 = P */

        if (!is_idr) {
            bs_write_u(&slice_bs, 8, encoder->poc & 0xFF);
            bs_write1(&slice_bs, 1);
            bs_write1(&slice_bs, 0);
            bs_write_ue(&slice_bs, 0);
        }

        bs_write_se(&slice_bs, slice_qp_delta);
        bs_rbsp_trailing_bits(&slice_bs);

        hevc_cabac_t cab;
        hevc_cabac_init(&cab, &slice_bs);
        hevc_cabac_reset_contexts(&cab, encoder->qp, is_idr ? 2 : 1);
        hevc_cabac_start(&cab);

        uint32_t ctus_slice = (r1 - r0) * encoder->width_ctu;
        uint32_t k = 0;
        for (uint32_t row = r0; row < r1; row++) {
            for (uint32_t col = 0; col < encoder->width_ctu; col++) {
                encode_ctu(encoder, &cab, (int)col, (int)row, is_idr, y_min, &sad);
                k++;
                /* end_of_slice_segment_flag: the last CTU of THIS slice */
                hevc_cabac_encode_terminate(&cab, k == ctus_slice ? 1 : 0);
            }
        }

        hevc_cabac_finish(&cab);
        bs_rbsp_trailing_bits(&slice_bs);
        encoder->slice_len[s] = bs_bytes_written(&slice_bs);
        encoder->slice_sad[s] = sad;
    }

    for (int s = 0; s < ns; s++) encoder->last_frame_sad += encoder->slice_sad[s];

    size_t total = 0;
    total += write_aud_hevc(encoder->scratch_out + total, encoder->scratch_out_cap - total, is_idr);
    if (write_param_sets) {
        total += write_vps(encoder->scratch_out + total, encoder->scratch_out_cap - total);
        total += write_sps(encoder->scratch_out + total, encoder->scratch_out_cap - total,
                            encoder->coded_width, encoder->coded_height,
                            encoder->width, encoder->height,
                            hevc_pick_level_idc(encoder->coded_width, encoder->coded_height));
        total += write_pps(encoder->scratch_out + total, encoder->scratch_out_cap - total, encoder->qp);
    }

    for (int s = 0; s < ns; s++) {
        size_t needed = total + 32 + encoder->slice_len[s] * 2;
        if (needed > encoder->scratch_out_cap) {
            size_t new_cap = encoder->scratch_out_cap * 2;
            if (new_cap < needed + 131072) new_cap = needed + 131072;
            uint8_t *new_buf = realloc(encoder->scratch_out, new_cap);
            if (new_buf) {
                encoder->scratch_out = new_buf;
                encoder->scratch_out_cap = new_cap;
            }
        }
        bitstream_t out_bs;
        bs_init(&out_bs, encoder->scratch_out + total, encoder->scratch_out_cap - total);
        bs_write_nal_header_hevc(&out_bs, is_idr ? NAL_UNIT_CODED_SLICE_IDR_W_RADL : NAL_UNIT_CODED_SLICE_TRAIL_R);
        size_t off = bs_bytes_written(&out_bs);
        size_t ebsp = bs_rbsp_to_ebsp(encoder->scratch_out + total + off,
                                      encoder->scratch_out_cap - total - off,
                                      encoder->slice_buf[s], encoder->slice_len[s]);
        total += off + ebsp;
    }

    if (total > output_size) return -1;
    memcpy(output_buf, encoder->scratch_out, total);

    if (total > 0 && encoder->rc.mode != RC_CQP) {
        rc_update_stats(&encoder->rc, (int)(total * 8));
    }

    /* Update reference buffers for subsequent P-frames */
    size_t luma_size = (size_t)encoder->coded_width * encoder->coded_height;
    size_t chroma_size = (size_t)(encoder->coded_width / 2) * (encoder->coded_height / 2);
    /* The reference picture changes hands, it does not get copied.
     *
     * This was three megabytes of memcpy per 1080p frame. prev_recon_* is only
     * ever READ (motion search and the skip path's copy) and recon_* is only
     * ever WRITTEN - every pixel of the coded area, by one CU or another - so
     * swapping the pointers leaves both sides holding exactly what the copy
     * used to give them. Both buffers stay allocated and are freed together,
     * so nothing changes for hevc_encoder_destroy().
     */
    {
        uint8_t *t;
        t = encoder->prev_recon_y;  encoder->prev_recon_y  = encoder->recon_y;  encoder->recon_y  = t;
        t = encoder->prev_recon_cb; encoder->prev_recon_cb = encoder->recon_cb; encoder->recon_cb = t;
        t = encoder->prev_recon_cr; encoder->prev_recon_cr = encoder->recon_cr; encoder->recon_cr = t;
    }
    (void)luma_size; (void)chroma_size;
    encoder->has_ref = true;
    encoder->poc++;

    /* Debug-only: dump this encoder's own idea of the reconstructed picture
     * (i.e. what a bug-free decoder given this exact bitstream SHOULD
     * reproduce) - lets a diff against a real decoder's actual output
     * localize whether a mismatch is in the prediction/transform/quant
     * math (this dump would ALSO look wrong) or in CABAC/bitstream framing
     * (this dump looks right, but a real decoder's output doesn't). */
    if (getenv("BC250_HEVC_DEBUG_RECON")) {
        /* After the swap it is prev_recon_y that holds this frame. */
        FILE *fy = fopen("bc250_hevc_debug_recon_y.raw", "wb");
        if (fy) { fwrite(encoder->prev_recon_y, 1, (size_t)encoder->coded_width * encoder->coded_height, fy); fclose(fy); }
    }

    encoder->frame_count++;
    return (int)total;
}

int hevc_encoder_encode_frame(hevc_encoder_t *encoder,
                              bc250_gpu_context_t *gpu_ctx,
                              gpu_image_t input_surface,
                              gpu_memory_t input_memory,
                              uint8_t *output_buf, size_t output_size)
{
    if (!encoder || !output_buf) return -1;

    encoder->num_gpu_mvs = 0;
    bool is_idr = (encoder->frame_count % encoder->gop_size == 0) || encoder->force_idr || !encoder->has_ref;

    if (gpu_ctx && input_surface.y_plane != VK_NULL_HANDLE) {
        /* ⚠️ The governor's tiers mean something narrower here than in the
         * H.264 encoder. The GPU's only job in this one is the motion
         * search, so there is no reduced-search dispatch to fall back on:
         * tiers 0 and 1 both run it, and tiers 2 and 3 do not run it at
         * all. Not running it is already a complete fallback - the CPU
         * search is what happens when num_gpu_mvs stays at zero, which is
         * exactly the state an I frame is in. */
        const governor_tier_t tier = dynamic_governor_get_tier(&encoder->governor);
        bool use_gpu_me = (tier < GOV_TIER_2_CPU_OFFLOAD);

        /* ⚠️ And that search is also the only thing that measures the GPU.
         * Skipping it leaves the governor with no new latency, so the
         * moving average never decays and the encoder would stay on the
         * CPU for the rest of the stream. One frame in every
         * step_down_hysteresis goes to the GPU anyway, purely to bring
         * back a reading. */
        if (!use_gpu_me) {
            const uint32_t every = encoder->governor.step_down_hysteresis;
            encoder->governor_skips++;
            use_gpu_me = every && (encoder->governor_skips % every == 0);
        }

        /* Run lightweight subgroup-accelerated GPU motion estimation on P-frames (~0.4ms) */
        if (!is_idr && encoder->has_ref && use_gpu_me) {
            gpu_compute_begin_picture(gpu_ctx, input_surface);
            gpu_compute_dispatch_me_only(gpu_ctx, input_surface, (int)encoder->width, (int)encoder->height);
            gpu_compute_end_picture(gpu_ctx);
            gpu_compute_sync(gpu_ctx);
            dynamic_governor_update(&encoder->governor,
                                    gpu_compute_get_last_latency_ms(gpu_ctx));

            void *mv_data = NULL;
            size_t mv_size = 0;
            if (gpu_compute_get_mv_staging_data(gpu_ctx, &mv_data, &mv_size) == 0 && mv_data) {
                size_t max_bytes = (size_t)encoder->width_ctu * encoder->height_ctu * sizeof(gpu_mv_t);
                size_t copy_bytes = (mv_size < max_bytes) ? mv_size : max_bytes;
                memcpy(encoder->gpu_mvs, mv_data, copy_bytes);
                encoder->num_gpu_mvs = (uint32_t)(copy_bytes / sizeof(gpu_mv_t));
            }
        } else if (tier == GOV_TIER_3_FAILOVER && !is_idr && encoder->has_ref) {
            /* One skipped frame is the whole emergency. Step back down so
             * the next frame tries the GPU again instead of waiting for a
             * measurement that can only come from trying. */
            dynamic_governor_notify_failover_handled(&encoder->governor);
        }

        gpu_compute_download_nv12(gpu_ctx, &input_surface, input_memory,
                                   encoder->dl_y, (int)encoder->width,
                                   encoder->dl_uv, (int)encoder->width,
                                   (int)encoder->width, (int)encoder->height);
    } else {
        memset(encoder->dl_y, 128, (size_t)encoder->width * encoder->height);
        memset(encoder->dl_uv, 128, (size_t)(encoder->width / 2) * (encoder->height / 2) * 2);
    }

    return encode_core(encoder, output_buf, output_size);
}

int hevc_encoder_encode_raw(hevc_encoder_t *encoder,
                            const uint8_t *y_plane, int y_pitch,
                            const uint8_t *uv_plane, int uv_pitch,
                            uint8_t *output_buf, size_t output_size)
{
    if (!encoder || !output_buf || !y_plane || !uv_plane) return -1;

    encoder->num_gpu_mvs = 0;

    for (uint32_t y = 0; y < encoder->height; y++)
        memcpy(encoder->dl_y + (size_t)y * encoder->width, y_plane + (size_t)y * y_pitch, encoder->width);
    for (uint32_t y = 0; y < encoder->height / 2; y++)
        memcpy(encoder->dl_uv + (size_t)y * encoder->width, uv_plane + (size_t)y * uv_pitch, encoder->width);

    return encode_core(encoder, output_buf, output_size);
}
