/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevcps.c - read an H.265 stream's parameter sets and slice headers, and
 * say what is in them.
 *
 *     hevcps [-q] <file.265>
 *
 * Nothing is decoded. This is the first half of a decoder's harness: it
 * proves the syntax above the slice data can be read, which is the part
 * VA-API hands over ready-made and which a file does not. It exits
 * non-zero if any of it is refused, so it can be run over a spread of
 * streams as a test.
 *
 * ⚠️ The picture order count is derived here rather than read: only its
 * low bits are sent, and the high bits come from the picture before. Get
 * it wrong and every reference list points at the wrong picture, which is
 * why it is worth checking on its own before anything depends on it.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bitreader.h"
#include "hevc_ps.h"
#include "decoder_h265.h"
#include "hevc_dec_internal.h"

static const char *nome_nal(int t)
{
    switch (t) {
    case HEVC_NAL_TRAIL_N: return "TRAIL_N";
    case HEVC_NAL_TRAIL_R: return "TRAIL_R";
    case HEVC_NAL_TSA_N: return "TSA_N";
    case HEVC_NAL_TSA_R: return "TSA_R";
    case HEVC_NAL_STSA_N: return "STSA_N";
    case HEVC_NAL_STSA_R: return "STSA_R";
    case HEVC_NAL_RADL_N: return "RADL_N";
    case HEVC_NAL_RADL_R: return "RADL_R";
    case HEVC_NAL_RASL_N: return "RASL_N";
    case HEVC_NAL_RASL_R: return "RASL_R";
    case HEVC_NAL_BLA_W_LP: return "BLA_W_LP";
    case HEVC_NAL_BLA_W_RADL: return "BLA_W_RADL";
    case HEVC_NAL_BLA_N_LP: return "BLA_N_LP";
    case HEVC_NAL_IDR_W_RADL: return "IDR_W_RADL";
    case HEVC_NAL_IDR_N_LP: return "IDR_N_LP";
    case HEVC_NAL_CRA: return "CRA";
    case HEVC_NAL_VPS: return "VPS";
    case HEVC_NAL_SPS: return "SPS";
    case HEVC_NAL_PPS: return "PPS";
    case HEVC_NAL_AUD: return "AUD";
    case HEVC_NAL_EOS: return "EOS";
    case HEVC_NAL_EOB: return "EOB";
    case HEVC_NAL_FD: return "FD";
    case HEVC_NAL_SEI_PREFIX: return "SEI";
    case HEVC_NAL_SEI_SUFFIX: return "SEI_SUFFIX";
    default: return "?";
    }
}

static const char *perche(int r)
{
    switch (r) {
    case -1: return "NAL truncated";
    case -2: return "not supported";
    case -3: return "outside the standard";
    default: return "?";
    }
}

static const char slice_type_of[3] = { 'B', 'P', 'I' };

/* Between one IDR and the next, the picture order counts must be exactly
 * 0..k-1: every one of them present, and none of them twice. They arrive
 * in decode order, which with B pictures is not display order, so the
 * question is about the SET and not about the sequence. */
static bool check_group(const int *poc, int count)
{
    if (count <= 0) return true;
    for (int expected = 0; expected < count; expected++) {
        int n_found = 0;
        for (int i = 0; i < count; i++)
            if (poc[i] == expected) n_found++;
        if (n_found != 1) {
            fprintf(stderr, "poc %d seen %d times across %d pictures\n",
                    expected, n_found, count);
            return false;
        }
    }
    return true;
}



/* The decoded picture buffer.
 *
 * ⚠️ Every picture here is one another may still point at. A picture is
 * let go when the reference picture set of a later one stops naming it,
 * and not a moment before: the set is the only thing that says so, and a
 * decoder that frees on its own idea of "old" loses a reference the
 * stream was still counting on. */







/* Pictures wait here until the stream ends, because the order they come
 * out in is not the order they were decoded in. */
typedef struct {
    long order;
    uint8_t *data;
    size_t n;
} frame_t;

static frame_t *output_order;
static int n_output, cap_output;

static int compare_order(const void *a, const void *b)
{
    const long x = ((const frame_t *)a)->order;
    const long y = ((const frame_t *)b)->order;
    return x < y ? -1 : (x > y ? 1 : 0);
}

/* Cropped on the way out: the coded picture is a whole number of smallest
 * coding blocks and the visible one is not. */
static void write_picture(FILE *f, hevc_decoder_t *dc,
                            const hevc_sps_t *sps, long order)
{
    int stride[3];
    const uint8_t *plane[3];
    /* ⚠️ The loop filters run here and not at the end of each slice: 8.7.2
     * is defined over the whole picture, and an edge between two coding
     * tree units cannot be filtered until both of them exist. */
    hevc_decoder_end_picture(dc);
    for (int i = 0; i < 3; i++) plane[i] = hevc_decoder_plane(dc, i, &stride[i]);
    if (!f || !plane[0]) return;

    const int x0 = sps->crop_left, y0 = sps->crop_top;
    const int w = sps->width - sps->crop_left - sps->crop_right;
    const int h = sps->height - sps->crop_top - sps->crop_bottom;
    /* ⚠️ The strides come back in SAMPLES, so every offset into the plane
     * is multiplied here and nowhere else. Above eight bits a sample is a
     * native uint16_t, which on this machine is exactly what ffmpeg calls
     * yuv420p10le - no byte swapping, by luck rather than design, and
     * worth knowing if this ever runs somewhere big endian. */
    const size_t bytes = sps->bit_depth_luma > 8 ? 2 : 1;
    const size_t n = ((size_t)w * h + 2 * (size_t)(w / 2) * (h / 2)) * bytes;

    if (n_output == cap_output) {
        const int new_one = cap_output ? cap_output * 2 : 32;
        frame_t *p = realloc(output_order, (size_t)new_one * sizeof *p);
        if (!p) return;
        output_order = p;
        cap_output = new_one;
    }
    uint8_t *data = malloc(n);
    if (!data) return;

    size_t o = 0;
    for (int y = 0; y < h; y++) {
        memcpy(data + o,
               plane[0] + ((size_t)(y0 + y) * stride[0] + x0) * bytes,
               (size_t)w * bytes);
        o += (size_t)w * bytes;
    }
    for (int p = 1; p < 3; p++)
        for (int y = 0; y < h / 2; y++) {
            memcpy(data + o,
                   plane[p] + ((size_t)(y0 / 2 + y) * stride[p]
                               + x0 / 2) * bytes,
                   (size_t)(w / 2) * bytes);
            o += (size_t)(w / 2) * bytes;
        }

    output_order[n_output].order = order;
    output_order[n_output].data = data;
    output_order[n_output].n = n;
    n_output++;
}

/* Display order at last: sorted by picture order count, and by which
 * instantaneous refresh they belong to, since the count restarts at every
 * one of those. */
static void drain_output(FILE *f)
{
    if (output_order) qsort(output_order, (size_t)n_output,
                             sizeof *output_order, compare_order);
    for (int i = 0; i < n_output; i++) {
        if (f) fwrite(output_order[i].data, 1, output_order[i].n, f);
        free(output_order[i].data);
    }
    free(output_order);
    output_order = NULL;
    n_output = cap_output = 0;
}



/* The harness runs the decoder with no GPU context, so the finished
 * picture stays in memory instead of going to a surface. The upload is
 * still referenced from the object file, so it needs a body to link
 * against; reaching it would mean the null-context path had been lost,
 * which is why it says so rather than returning quietly. */
int gpu_compute_upload_nv12(gpu_context_t *ctx, gpu_image_t *image,
                            gpu_memory_t memory,
                            const uint8_t *y, int ys,
                            const uint8_t *uv, int uvs, int w, int h);
int gpu_compute_upload_nv12(gpu_context_t *ctx, gpu_image_t *image,
                            gpu_memory_t memory,
                            const uint8_t *y, int ys,
                            const uint8_t *uv, int uvs, int w, int h)
{
    (void)ctx; (void)image; (void)memory;
    (void)y; (void)ys; (void)uv; (void)uvs; (void)w; (void)h;
    fprintf(stderr, "the harness has no GPU to hand the picture to\n");
    abort();
}

int main(int argc, char **argv)
{
    bool quiet = false, headers_only = false;
    const char *nome = NULL, *output = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-q") == 0) quiet = true;
        else if (strcmp(argv[i], "-h") == 0) headers_only = true;
        else if (!nome) nome = argv[i];
        else output = argv[i];
    }
    if (!nome) {
        fprintf(stderr, "usage: hevcps [-q] [-h] <file.265> [output.yuv]\n");
        return 2;
    }

    FILE *f = fopen(nome, "rb");
    if (!f) { perror(nome); return 2; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc((size_t)len);
    uint8_t *rbsp = malloc((size_t)len);
    if (!buf || !rbsp || fread(buf, 1, (size_t)len, f) != (size_t)len) {
        fprintf(stderr, "%s: cannot read it\n", nome);
        return 2;
    }
    fclose(f);

    hevc_sps_t *sps = calloc(16, sizeof(hevc_sps_t));
    hevc_pps_t *pps = calloc(64, sizeof(hevc_pps_t));
    if (!sps || !pps) return 2;

    /* Clause 8.3.1, the picture order count: only its low bits are sent. */
    int prev_poc_lsb = 0, prev_poc_msb = 0;
    int pictures = 0, total_slices = 0, refused = 0;
    int slices_of_this = 0;

    /* The picture order counts seen since the last IDR. They must come out
     * as exactly 0..k-1: every one present, none twice. An IDR restarts
     * the count, so the check is per group and not over the whole file. */
    int *poc_seen = calloc(4096, sizeof(int));
    int n_seen = 0, broken_groups = 0;
    if (!poc_seen) return 2;

    hevc_decoder_t *dec = hevc_decoder_create(NULL, 0, 0);
    if (!dec) return 2;
    int slices_read = 0, slices_lost = 0, slices_skipped = 0;
    FILE *fo = output ? fopen(output, "wb") : NULL;
    if (output && !fo) { perror(output); return 2; }
    bool picture_open = false;
    long base_order = 0, next_order = 0, current_order = 0;

    for (long i = 0; i + 3 < len; ) {
        /* Find the start code, then the next one. */
        if (!(buf[i] == 0 && buf[i + 1] == 0
              && (buf[i + 2] == 1 || (buf[i + 2] == 0 && buf[i + 3] == 1)))) {
            i++;
            continue;
        }
        long start = i + ((buf[i + 2] == 1) ? 3 : 4);
        long fine = start;
        while (fine + 3 < len
               && !(buf[fine] == 0 && buf[fine + 1] == 0
                    && (buf[fine + 2] == 1
                        || (buf[fine + 2] == 0 && buf[fine + 3] == 1))))
            fine++;
        if (fine + 3 >= len) fine = len;
        i = fine;
        if (start >= fine) continue;

        const int kind = (buf[start] >> 1) & 0x3f;
        const size_t n = br_extract_rbsp(rbsp, (size_t)len, buf + start,
                                         (size_t)(fine - start));

        if (kind == HEVC_NAL_SPS) {
            hevc_sps_t s;
            const int r = hevc_ps_read_sps(&s, rbsp, n);
            if (r) {
                fprintf(stderr, "SPS refused: %s\n", perche(r));
                refused++;
                continue;
            }
            sps[s.sps_id] = s;
            if (!quiet)
                printf("SPS %d: %dx%d, crop %d/%d/%d/%d, CTB %d, "
                       "CB %d..%d, TB %d..%d, poc_lsb %d bit, "
                       "sao %d amp %d smi %d tmvp %d, %d sets of "
                       "references\n",
                       s.sps_id, s.width, s.height,
                       s.crop_left, s.crop_right, s.crop_top, s.crop_bottom,
                       s.ctb_size, 1 << s.log2_min_cb, 1 << s.log2_ctb,
                       1 << s.log2_min_tb, 1 << s.log2_max_tb,
                       s.log2_max_poc_lsb, s.sao_enabled, s.amp_enabled,
                       s.strong_intra_smoothing, s.temporal_mvp_enabled,
                       s.num_st_rps);
        } else if (kind == HEVC_NAL_PPS) {
            hevc_pps_t p;
            const int r = hevc_ps_read_pps(&p, rbsp, n);
            if (r) {
                fprintf(stderr, "PPS refused: %s\n", perche(r));
                refused++;
                continue;
            }
            pps[p.pps_id] = p;
            if (!quiet)
                printf("PPS %d (SPS %d): qp %d, cu_qp_delta %d/%d, "
                       "chroma %+d%+d, weights %d/%d, tile %dx%d, wpp %d, "
                       "deblk %s, lists %d/%d, merge %d\n",
                       p.pps_id, p.sps_id, p.init_qp,
                       p.cu_qp_delta_enabled, p.diff_cu_qp_delta_depth,
                       p.cb_qp_offset, p.cr_qp_offset,
                       p.weighted_pred, p.weighted_bipred,
                       p.num_tile_columns, p.num_tile_rows,
                       p.entropy_coding_sync_enabled,
                       p.deblocking_filter_disabled ? "no" : "si",
                       p.num_ref_idx_default[0], p.num_ref_idx_default[1],
                       p.log2_parallel_merge_level);
        } else if (hevc_nal_e_slice(kind)) {
            hevc_slice_t s;
            const int r = hevc_ps_read_slice(&s, rbsp, n, kind, sps, pps);
            if (r) {
                fprintf(stderr, "slice refused: %s\n", perche(r));
                refused++;
                continue;
            }
            if (s.dependent_slice_segment) {
                fprintf(stderr, "dependent slice segments: not supported\n");
                refused++;
                continue;
            }

            if (s.first_slice_in_pic) {
                if (pictures && !quiet)
                    printf("     (%d slice)\n", slices_of_this);
                slices_of_this = 0;
                pictures++;

                /* 8.3.1. An IRAP that starts the sequence resets it. */
                if (hevc_nal_e_idr(kind)) {
                    /* An IDR ends one group and starts the next. */
                    if (!check_group(poc_seen, n_seen)) broken_groups++;
                    n_seen = 0;
                    s.poc = 0;
                    prev_poc_lsb = 0;
                    prev_poc_msb = 0;
                } else {
                    const hevc_sps_t *sp = &sps[pps[s.pps_id].sps_id];
                    const int max = 1 << sp->log2_max_poc_lsb;
                    int msb;
                    if (s.poc_lsb < prev_poc_lsb
                        && prev_poc_lsb - s.poc_lsb >= max / 2)
                        msb = prev_poc_msb + max;
                    else if (s.poc_lsb > prev_poc_lsb
                             && s.poc_lsb - prev_poc_lsb > max / 2)
                        msb = prev_poc_msb - max;
                    else
                        msb = prev_poc_msb;
                    s.poc = msb + s.poc_lsb;
                    prev_poc_lsb = s.poc_lsb;
                    prev_poc_msb = msb;
                }

                if (n_seen < 4096) poc_seen[n_seen++] = s.poc;

                if (!quiet)
                    printf("  %-10s %c poc %3d qp %2d rif %d/%d sao %d%d "
                           "merge %d",
                           nome_nal(kind), slice_type_of[s.type], s.poc, s.qp,
                           s.num_ref_idx[0], s.num_ref_idx[1],
                           s.sao_luma, s.sao_chroma,
                           5 - s.five_minus_max_num_merge_cand);
                if (!quiet && s.type != 2) {
                    printf(" rps");
                    for (int k = 0; k < s.st_rps.num_negative
                                        + s.st_rps.num_positive; k++)
                        printf(" %+d%s", s.st_rps.delta_poc[k],
                               s.st_rps.used[k] ? "" : "-");
                }
                if (!quiet) printf("\n");
            }
            slices_of_this++;
            total_slices++;

            if (!headers_only) {
                const hevc_sps_t *sp = &sps[pps[s.pps_id].sps_id];
                hevc_decoder_shift_entry_points(
                    &s, buf + start, (size_t)(fine - start),
                    s.data_bit_offset >> 3);
                if (s.first_slice_in_pic) {
                    if (picture_open) {
                        write_picture(fo, dec, sp, current_order);
                        picture_open = false;
                    }
                    if (s.nal_type == HEVC_NAL_IDR_W_RADL
                        || s.nal_type == HEVC_NAL_IDR_N_LP)
                        base_order = next_order;
                    hevc_decoder_unescape(dec, &s);
                    if (hevc_decoder_begin_picture(dec, sp, &pps[s.pps_id],
                                                   (uintptr_t)pictures, s.poc)) {
                        slices_lost++;
                        continue;
                    }
                    current_order = base_order + s.poc;
                    if (current_order >= next_order)
                        next_order = current_order + 1;
                }
                const int e = hevc_decoder_slice(dec, &s, rbsp, n);
                if (e == 4) {
                    slices_skipped++;
                    if (picture_open)
                        write_picture(fo, dec, sp, current_order);
                    picture_open = false;
                    if (fo) { fclose(fo); fo = NULL; }
                } else if (e) {
                    slices_lost++;
                    if (!quiet)
                        printf("     ^ %s\n", hevc_decoder_reason(e));
                } else {
                    slices_read++;
                    picture_open = true;
                }
            }
        }
    }
    if (pictures && !quiet)
        printf("     (%d slice)\n", slices_of_this);

    if (picture_open) {
        const hevc_sps_t *sp = NULL;
        for (int k = 0; k < 16; k++) if (sps[k].valid) { sp = &sps[k]; break; }
        if (sp) write_picture(fo, dec, sp, current_order);
    }
    drain_output(fo);
    if (fo) fclose(fo);
    if (!check_group(poc_seen, n_seen)) broken_groups++;

    printf("%d pictures, %d slices, %d refused, %d groups with broken poc, "
           "%d walked, %d skipped, %d lost\n",
           pictures, total_slices, refused, broken_groups,
           slices_read, slices_skipped, slices_lost);
    free(buf); free(rbsp); free(sps); free(pps); free(poc_seen);
    hevc_decoder_destroy(dec);
    return (refused || broken_groups || slices_lost) ? 1 : 0;
}
