/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * decoder_h264.h - H.264 decoding for the BC-250.
 *
 * What this decoder does not have to do is parse a bitstream from the top.
 * Under VAEntrypointVLD the application has already read the SPS, the PPS and
 * every slice header, and hands them over as VAPictureParameterBufferH264 and
 * VASliceParameterBufferH264. So the work starts at the macroblock layer:
 * entropy decoding, prediction, the inverse transform and the deblocking
 * filter.
 *
 * The output frame is built in ordinary system memory and uploaded into the
 * VA surface once, at the end of the picture.
 *
 * ⚠️ Reference pictures are kept here as plain planes rather than being read
 * back out of their VA surfaces. Surface memory is write-combining: writing
 * to it is fast, reading from it is roughly an order of magnitude slower than
 * system memory, and motion compensation reads references constantly. That is
 * the same trap the encoder fell into with its surface readback.
 *
 * Scope: 4:2:0 8-bit, progressive, I/P/B slices, CABAC and CAVLC, both the
 * 4x4 and the 8x8 transform. Interlaced coding (PAFF and MBAFF), 4:2:2 and
 * 4:4:4, and the lossless and predictive-lossless modes are not implemented;
 * h264_decoder_supports() says so before a context is ever created, so those
 * streams fall back to the application's own decoder instead of decoding
 * into rubbish.
 */
#ifndef BC250_DECODER_H264_H
#define BC250_DECODER_H264_H

#include <stdbool.h>
#include <stdint.h>

#include "gpu_compute.h"

#define H264D_MAX_REFS 16
#define H264D_DPB_SIZE (H264D_MAX_REFS + 1)

/* Macroblock types, after the mb_type of the standard has been resolved
 * against the slice type. The decoder works with these rather than with the
 * raw numbers, because the same mb_type value means different things in an
 * I, P or B slice. */
typedef enum {
    H264D_MB_I_NxN = 0,     /* Intra_4x4 or Intra_8x8, per transform_size flag */
    H264D_MB_I_16x16,
    H264D_MB_I_PCM,
    H264D_MB_P_16x16,
    H264D_MB_P_16x8,
    H264D_MB_P_8x16,
    H264D_MB_P_8x8,
    H264D_MB_P_SKIP,
    H264D_MB_B_DIRECT,
    H264D_MB_B_16x16,
    H264D_MB_B_16x8,
    H264D_MB_B_8x16,
    H264D_MB_B_8x8,
    H264D_MB_B_SKIP,
} h264d_mb_type_t;

/* Per-macroblock state that outlives the macroblock, because a neighbour or
 * the deblocking filter will ask for it. One of these per macroblock of the
 * picture; at 1080p that is 8160 of them. */
typedef struct {
    uint8_t  type;              /* h264d_mb_type_t */
    uint8_t  intra;             /* 1 for every I_ type */
    uint8_t  transform8x8;
    /* Set while the partitions are read when any of them is smaller than
     * 8x8. The 8x8 transform is not offered for such a macroblock, so the
     * flag has to be known before coded_block_pattern is read. */
    uint8_t  sub_8x8;
    /* Which of the four 8x8 partitions are direct. The ref_idx
     * context of 9.3.3.1.1.6 skips a direct neighbour. */
    uint8_t  direct;
    /* Debug only: the four sub_mb_types of a P_8x8 or B_8x8. */
    int8_t   sub_type[4];
    uint8_t  cbp;               /* bits 0..3 luma 8x8s, bits 4..5 chroma */
    int8_t   qpy;
    int8_t   chroma_pred_mode;
    int8_t   ipred[16];         /* Intra_4x4 / Intra_8x8 modes, raster in mb */
    /* Per 8x8 partition, per list: the DPB slot of the reference picture, or
     * -1 when that list is unused here.
     *
     * ⚠️ The DPB slot, not the reference index the slice header gave. The
     * deblocking filter compares reference PICTURES, and the same picture
     * can sit at different indices in two slices' lists, so a decoder that
     * stores indices puts an edge where the encoder put none. Resolving it
     * once here also means the filter never has to know which slice a
     * macroblock came from. */
    int8_t   ref[2][4];
    /* The reference INDEX the slice signalled, per 8x8 partition.
     * ⚠️ Kept alongside the DPB slot above because the two clauses ask
     * different questions. Motion vector prediction (8.4.1.3) compares
     * indices; the deblocking filter (8.7.2.1) compares pictures. They agree
     * within a slice unless reference list modification has put the same
     * picture at two indices, which is exactly when one of them is wrong. */
    int8_t   ref_idx[2][4];
    int16_t  mv[2][16][2];      /* per 4x4 block, per list */
    /* The motion vector differences as they were coded, per 4x4 block.
     * ⚠️ Kept because the CABAC context for mvd is the sum of the
     * neighbours' DIFFERENCES, not of their vectors: two blocks can end up
     * far apart having each coded almost nothing, which is the case the
     * context exists to catch. */
    int16_t  mvd[2][16][2];
    uint8_t  nnz[3][16];        /* non-zero coefficients: CAVLC context and
                                 * the deblocking filter's bS both need it */
    uint8_t  cbf_dc[3];         /* coded_block_flag of the DC blocks */
} h264d_mb_t;

/* A decoded picture. The planes are ours; `surface` is the VA surface the
 * application knows it by, which is how a reference in
 * VAPictureParameterBufferH264 is matched back to its pixels. */
typedef struct {
    uint8_t *y, *cb, *cr;
    int      stride_y, stride_c;
    int      poc;                /* the smaller of the two field orders */
    int      frame_num;
    uint32_t surface;            /* VASurfaceID, or ~0u when free */
    bool     used;
    bool     is_long_term;

    /* The motion field this picture was decoded with, kept because a later
     * B picture's direct macroblocks read the co-located one (8.4.1.2).
     * Laid out per macroblock: [mb][list][8x8] and [mb][list][4x4][xy]. */
    int8_t  *col_ref;
    int16_t *col_mv;
    /* ⚠️ The POC of the picture each of those indices pointed at. Temporal
     * direct needs the picture, not the index: the index means something
     * only inside the list the co-located slice built. */
    int32_t *col_poc;
    bool     col_intra_only;     /* nothing here to derive from */
} h264d_frame_t;

/* Everything the picture parameter buffer tells us, flattened. */
typedef struct {
    int width, height;
    int mb_width, mb_height;
    int chroma_qp_index_offset;
    int second_chroma_qp_index_offset;
    int pic_init_qp;
    int num_ref_idx_l0, num_ref_idx_l1;
    bool entropy_coding_mode;
    bool transform_8x8_mode;
    bool constrained_intra_pred;
    bool direct_8x8_inference;
    bool deblocking_filter_control_present;
    bool weighted_pred;
    int  weighted_bipred_idc;
    bool pic_order_present;
    uint8_t scaling4[6][16];
    uint8_t scaling8[6][64];
} h264d_pic_t;

/* Per-slice state, rebuilt at every slice header. */
typedef struct {
    int  type;                  /* 0 P, 1 B, 2 I, after the %5 */
    int  qpy;
    int  cabac_init_idc;
    int  num_ref_idx[2];
    int  disable_deblocking_filter_idc;
    int  alpha_c0_offset, beta_offset;
    int  first_mb;
    bool direct_spatial_mv_pred;
    /* Reference lists, as indices into the decoder's DPB. */
    int8_t ref_list[2][32];
    int    ref_poc[2][32];
    /* Explicit weighted prediction, clause 7.4.3.2. */
    int  luma_log2_weight_denom, chroma_log2_weight_denom;
    int16_t luma_weight[2][32], luma_offset[2][32];
    int16_t chroma_weight[2][32][2], chroma_offset[2][32][2];
} h264d_slice_t;

typedef struct h264_decoder h264_decoder_t;

/* Whether this decoder can handle a stream with these properties at all.
 * Called before a context is created, so that an unsupported stream is
 * refused rather than decoded into rubbish. */
bool h264_decoder_supports(int profile_idc, int chroma_format_idc,
                           int bit_depth_luma, int bit_depth_chroma,
                           bool frame_mbs_only, bool mb_adaptive);

h264_decoder_t *h264_decoder_create(bc250_gpu_context_t *gpu_ctx,
                                    uint32_t width, uint32_t height);
void h264_decoder_destroy(h264_decoder_t *decoder);

/* Start a picture. Fills in the decoder's idea of the picture parameters and
 * picks the frame buffer that `surface` names. */
int h264_decoder_begin_picture(h264_decoder_t *dec, const h264d_pic_t *pic,
                               uint32_t surface, int poc, int frame_num);

/* One slice, ready to decode. `data` is the slice NAL payload including
 * its header and its emulation prevention bytes; `bit_offset` is where the
 * slice data itself begins, counted in that buffer, which is what
 * VASliceParameterBufferH264::slice_data_bit_offset gives. */
typedef struct {
    h264d_slice_t slice;
    const uint8_t *data;
    size_t size;
    int bit_offset;
    /* Not used by the decoder: somewhere for a caller to keep the
     * macroblock the slice starts at, for its own error messages. */
    int first_mb_diag;
} h264d_slice_input_t;

/* Decode one slice. */
int h264_decoder_slice(h264_decoder_t *dec, const h264d_slice_t *slice,
                       const uint8_t *data, size_t size, int bit_offset);

/* Decode a picture's slices. They are independent - no prediction crosses a
 * slice boundary - so with more than one and a thread pool to run them on,
 * they are decoded at the same time, entropy decoding included. Returns the
 * first failure, or zero. */
int h264_decoder_slices(h264_decoder_t *dec, const h264d_slice_input_t *in,
                        int n);

/* Finish the picture: deblock, then upload into the VA surface. */
int h264_decoder_end_picture(h264_decoder_t *dec, gpu_image_t out,
                             gpu_memory_t out_memory);

/* Tell the decoder which surfaces are still references, so the frames behind
 * the others can be reused. `refs` holds `n` VASurfaceIDs. */
void h264_decoder_set_references(h264_decoder_t *dec, const uint32_t *refs,
                                 const int *pocs, const bool *long_term, int n);

/* The frame a reference surface maps to, or NULL when the application is
 * referring to a picture this decoder never produced (a stream that starts
 * mid-GOP, typically). */
h264d_frame_t *h264_decoder_frame_for(h264_decoder_t *dec, uint32_t surface);

/* The frame store slot a frame sits in. The reference lists a slice carries
 * name slots, not surfaces, so that the macroblock layer never has to search
 * for a picture while it is decoding. */
int h264_decoder_slot_of(const h264_decoder_t *dec, const h264d_frame_t *f);

#endif /* BC250_DECODER_H264_H */
