/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * decoder_h265.c - the decoded picture buffer, the reference lists and the
 * slice loop.
 *
 * Everything below the slice header knows nothing about where pictures
 * come from or where they go. This is the piece that does.
 */
#include "decoder_h265.h"
#include "hevc_dec_internal.h"
#include "gpu_compute.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IMG_SLOTS 20

struct hevc_decoder {
    hevcd_t d;
    hevcd_img_t buffer[IMG_SLOTS];
    uintptr_t surface_id[IMG_SLOTS];  /* what the caller calls each picture */
    void *gpu;
    int width, height;
    hevc_sps_t sps;
    hevc_pps_t pps;
    hevc_slice_t last_one;
    bool is_open;
};

static void free_img(hevcd_img_t *g)
{
    for (int i = 0; i < 3; i++) { free(g->plane[i]); g->plane[i] = NULL; }
    free(g->mvf);
    g->mvf = NULL;
    g->n_planes = 0;
    g->n_mvf = 0;
    g->is_valid = false;
}

static hevcd_img_t *find_img(const hevcd_t *d, int poc)
{
    for (int i = 0; i < d->n_buf; i++)
        if (d->buf[i].is_valid && d->buf[i].poc == poc) return &d->buf[i];
    return NULL;
}

/* Everything the reference picture set no longer names can go. */
static void unescape(hevcd_t *d, const hevc_slice_t *sl)
{
    if (sl->nal_type == HEVC_NAL_IDR_W_RADL || sl->nal_type == HEVC_NAL_IDR_N_LP) {
        for (int i = 0; i < d->n_buf; i++) d->buf[i].is_valid = false;
        return;
    }
    const hevc_st_rps_t *r = &sl->st_rps;
    const int count = r->num_negative + r->num_positive;
    for (int i = 0; i < d->n_buf; i++) {
        if (!d->buf[i].is_valid) continue;
        bool serve = false;
        for (int k = 0; k < count && !serve; k++)
            if (d->buf[i].poc == sl->poc + r->delta_poc[k]) serve = true;
        if (!serve) d->buf[i].is_valid = false;
    }
}

/* The three planes of one picture at the coded size, and the motion field
 * a later picture will read for its temporal candidate. The visible size
 * is smaller and is applied when writing out. */
static int open_picture(hevcd_t *d, const hevc_sps_t *sps, int poc)
{
    const int w = sps->width, h = sps->height;
    const size_t n_mvf = (size_t)(w >> 2) * (h >> 2);
    /* ⚠️ Above eight bits a sample is two bytes. n_planes counts BYTES,
     * because that is what the SAO snapshot mallocs and copies, and it
     * doubles as the test below: a sequence that changes depth finds the
     * size different and reallocates, instead of writing ten-bit samples
     * into a buffer sized for eight. Luma and chroma share the figure,
     * which holds while we refuse streams whose two depths differ. */
    const size_t bytes = sps->bit_depth_luma > 8 ? 2 : 1;

    hevcd_img_t *g = NULL;
    for (int i = 0; i < d->n_buf && !g; i++)
        if (!d->buf[i].is_valid) g = &d->buf[i];
    if (!g) return -1;

    if (g->n_planes != (size_t)w * h * bytes) {
        free_img(g);
        g->plane[0] = malloc((size_t)w * h * bytes);
        g->plane[1] = malloc((size_t)(w / 2) * (h / 2) * bytes);
        g->plane[2] = malloc((size_t)(w / 2) * (h / 2) * bytes);
        g->n_planes = (size_t)w * h * bytes;
    }
    if (g->n_mvf != n_mvf) {
        free(g->mvf);
        g->mvf = malloc(n_mvf * sizeof *g->mvf);
        g->n_mvf = n_mvf;
    }
    if (!g->plane[0] || !g->plane[1] || !g->plane[2] || !g->mvf) return -1;

    memset(g->mvf, 0, n_mvf * sizeof *g->mvf);
    g->stride[0] = w;
    g->stride[1] = g->stride[2] = w / 2;
    g->poc = poc;
    g->is_valid = true;
    g->n_list[0] = g->n_list[1] = 0;

    d->current = g;
    d->mvf = g->mvf;
    for (int i = 0; i < 3; i++) { d->plane[i] = g->plane[i]; d->stride[i] = g->stride[i]; }
    d->n_planes = g->n_planes;
    return 0;
}

/* 8.3.4: the lists are the pictures before this one, then the ones after,
 * repeated until the list is as long as the slice header asked for.
 *
 * ⚠️ List one starts from the other end. That is the whole point of having
 * two: a B picture with one reference each way sends the shorter index
 * for whichever direction it meant. */
static void build_lists(hevcd_t *d, const hevc_slice_t *sl)
{
    const hevc_st_rps_t *r = &sl->st_rps;
    const hevcd_img_t *before[16], *after[16];
    int np = 0, nd = 0;

    for (int i = 0; i < r->num_negative && np < 16; i++) {
        if (!r->used[i]) continue;
        const hevcd_img_t *g = find_img(d, sl->poc + r->delta_poc[i]);
        if (g) before[np++] = g;
    }
    for (int i = r->num_negative; i < r->num_negative + r->num_positive
             && nd < 16; i++) {
        if (!r->used[i]) continue;
        const hevcd_img_t *g = find_img(d, sl->poc + r->delta_poc[i]);
        if (g) after[nd++] = g;
    }

    for (int l = 0; l < 2; l++) {
        d->n_refs[l] = 0;
        const int how_many = sl->num_ref_idx[l];
        const hevcd_img_t **a = l ? after : before;
        const hevcd_img_t **b = l ? before : after;
        const int na = l ? nd : np, nb = l ? np : nd;
        if (!na && !nb) continue;
        while (d->n_refs[l] < how_many) {
            for (int i = 0; i < na && d->n_refs[l] < how_many; i++)
                d->ref_pic[l][d->n_refs[l]++] = a[i];
            for (int i = 0; i < nb && d->n_refs[l] < how_many; i++)
                d->ref_pic[l][d->n_refs[l]++] = b[i];
        }
    }

    /* What the indices mean, kept with the picture: a later one that takes
     * this as its collocated picture asks what it pointed at, and an index
     * means nothing outside the slice that wrote it. */
    for (int l = 0; l < 2; l++) {
        d->current->n_list[l] = d->n_refs[l];
        for (int i = 0; i < d->n_refs[l]; i++)
            d->current->poc_list[l][i] = d->ref_pic[l][i]->poc;
    }

    d->col = NULL;
    if (sl->temporal_mvp_enabled) {
        const int l = sl->collocated_from_l0 ? 0 : 1;
        if (sl->collocated_ref_idx < d->n_refs[l])
            d->col = d->ref_pic[l][sl->collocated_ref_idx];
    }
}

/* 7.4.7.1: the entry point offsets count the NAL unit's bytes, the
 * emulation prevention ones included. The decoder reads the payload with
 * those already removed, so each offset has to lose however many of them
 * it has passed.
 *
 * ⚠️ The rule for which 0x03 is an emulation prevention byte is the one
 * in br_extract_rbsp and not "every 0x03 after two zeros": a 0x03
 * followed by a byte above 0x03 is ordinary payload. Counting them any
 * other way moves the offsets by the wrong amount on exactly the streams
 * where it matters.
 */
static void shift_entry_points(hevc_slice_t *s, const uint8_t *grezzo,
                               size_t n_grezzo, size_t first)
{
    if (s->num_entry_point_offsets <= 0) return;

    size_t i = 0, r = 0, zeros = 0, i0 = 0, r0 = 0;
    bool started = false;
    int k = 0;

    while (i < n_grezzo && k < s->num_entry_point_offsets) {
        if (!started && r == first) {
            i0 = i; r0 = r; started = true;
        }
        if (started && (size_t)(i - i0) == (size_t)s->entry_point[k]) {
            s->entry_point[k] = (uint32_t)(r - r0);
            k++;
            continue;
        }
        const uint8_t c = grezzo[i];
        if (zeros >= 2 && c == 0x03
            && !(i + 1 < n_grezzo && grezzo[i + 1] > 0x03)) {
            zeros = 0;
            i++;
            continue;                 /* removed, so the payload stands still */
        }
        zeros = (c == 0x00) ? zeros + 1 : 0;
        i++;
        r++;
    }
    /* An offset past the end of the NAL is a broken stream; leave the
     * rest where the payload ended and let the row check refuse it. */
    while (k < s->num_entry_point_offsets) {
        s->entry_point[k] = (uint32_t)(r - r0);
        k++;
    }
}

/* Why a slice could not be walked through. */
static const char *slice_reason(int e)
{
    switch (e) {
    case 1: return "ended before the last CTU";
    case 2: return "did not end where it should";
    case 3: return "read past the end of the NAL";
    case 4: return "slice type not walkable yet";
    case 5: return "out of memory";
    case 6: return "end of subset was not one";
    case 7: return "a row is not as long as the header says";
    default: return "?";
    }
}

/* Read every bin of a slice's coding tree, and check it lands.
 *
 * ⚠️ Nothing is reconstructed. What this proves is that the syntax was
 * read correctly, which CABAC makes checkable without any samples: a slice
 * read correctly ends with end_of_slice_segment_flag set exactly after the
 * last coding tree unit, and with the arithmetic decoder at the end of the
 * NAL. One bin read against the wrong context almost never lands there. */
static int walk_slice(hevcd_t *d, const hevc_sps_t *sps,
                          const hevc_pps_t *pps, const hevc_slice_t *sl,
                          const uint8_t *rbsp, size_t n)
{
    /* ⚠️ P and B slices are read through, not reconstructed. Their
     * samples are meaningless until motion compensation exists; what the
     * walk proves is that every bin of their syntax was read against the
     * right context. */

    const size_t serve_cb = (size_t)sps->min_cb_width * sps->min_cb_height;
    const size_t serve_pu = (size_t)(sps->width >> 2) * (sps->height >> 2);
    if (!d->ct_depth || d->n_ct_depth < serve_cb) {
        free(d->ct_depth);
        d->ct_depth = calloc(serve_cb, 1);
        d->n_ct_depth = serve_cb;
    }
    if (!d->intra_mode || d->n_intra_mode < serve_pu) {
        free(d->intra_mode);
        d->intra_mode = calloc(serve_pu, 1);
        d->n_intra_mode = serve_pu;
    }
    const size_t edges_needed = (size_t)((sps->width + 7) >> 3)
                            * (size_t)((sps->height + 7) >> 3);
    if (!d->edges || d->n_edges < edges_needed) {
        free(d->edges);
        d->edges = calloc(edges_needed, 1);
        d->n_edges = edges_needed;
    }
    if (!d->no_filter || d->n_no_filter < serve_cb) {
        free(d->no_filter);
        d->no_filter = calloc(serve_cb, 1);
        d->n_no_filter = serve_cb;
    }
    if (!d->sao || d->n_sao < (size_t)sps->ctb_count) {
        free(d->sao);
        d->sao = calloc((size_t)sps->ctb_count, sizeof *d->sao);
        d->n_sao = (size_t)sps->ctb_count;
    }
    if (!d->cbf_map || d->n_cbf < serve_pu) {
        free(d->cbf_map);
        d->cbf_map = calloc(serve_pu, 1);
        d->n_cbf = serve_pu;
    }
    if (!d->cbf_map) return 5;
    memset(d->cbf_map, 0, serve_pu);
    if (!d->skip || d->n_skip < serve_cb) {
        free(d->skip);
        d->skip = calloc(serve_cb, 1);
        d->n_skip = serve_cb;
    }
    if (!d->edges || !d->no_filter || !d->sao || !d->skip) return 5;
    d->edges_stride = (sps->width + 7) >> 3;
    if (!d->qp_y_map || d->n_qp < serve_cb) {
        free(d->qp_y_map);
        d->qp_y_map = calloc(serve_cb, 1);
        d->n_qp = serve_cb;
    }
    if (!d->ct_depth || !d->intra_mode || !d->qp_y_map) return 5;
    memset(d->ct_depth, 0, serve_cb);
    memset(d->edges, 0, edges_needed);
    memset(d->no_filter, 0, serve_cb);
    memset(d->skip, 0, serve_cb);
    memset(d->intra_mode, HEVCD_INTRA_DC, serve_pu);

    d->sps = sps;
    d->pps = pps;
    if (!d->current) return 5;
    /* ⚠️ Before the z-scan, which is built out of these. */
    if (hevcd_prepare_tiles(d)) return 5;
    if (hevcd_prepare_zscan(d)) return 5;
    d->slice = sl;
    d->min_pu_width = sps->width >> 2;
    d->min_pu_height = sps->height >> 2;
    d->qp_y = sl->qp;
    d->qp_y_pred = sl->qp;
    d->qp_y_prev = sl->qp;
    d->qg_restarts = true;
    d->slice_end = false;

    const size_t first = sl->data_bit_offset >> 3;
    if (first >= n) return 3;
    const uint8_t *base = rbsp + first;
    size_t rest = n - first;
    hevcd_cabac_init(&d->cabac, base, rest,
                     sl->type, sl->cabac_init_flag, sl->qp);

    const bool wpp = pps->entropy_coding_sync_enabled;
    const int init_type = hevcd_init_type(sl->type, sl->cabac_init_flag);
    uint8_t snapshot[HEVCD_CTX];
    bool have_snapshot = false;

    /* ⚠️ Only a slice that starts at the first unit and carries one entry
     * point per row: anything else - a slice segment starting mid picture,
     * tiles, a header that does not say where the rows are - is walked one
     * unit at a time rather than guessed at. */
    if (wpp && sl->segment_address == 0 && sps->ctb_height >= 2
        && sl->num_entry_point_offsets == sps->ctb_height - 1) {
        const int e = hevcd_wavefront(d, sps, pps, sl, base, rest, init_type);
        if (e >= 0) return e;
    }

    const int count = sps->ctb_count;
    const bool tiles = pps->tiles_enabled;
    int progress = 0;
    int substream = 0;      /* how many substream boundaries have passed */

    /* ⚠️ Tile scan. Without tiles the map is the identity and this is the
     * raster walk it always was. */
    for (int ts = d->rs_to_ts[sl->segment_address]; ts < count; ts++) {
        const int addr = d->ts_to_rs[ts];
        const int cx = addr % sps->ctb_width;
        const int x = cx << sps->log2_ctb;
        const int y = (addr / sps->ctb_width) << sps->log2_ctb;
        if (hevcd_read_ctu(d, x, y)) return 4;   /* refused inside */
        progress++;

        /* 9.3.2.3: after the second unit of a row, so the row below can
         * start from here. */
        if (wpp && cx == 1) {
            memcpy(snapshot, d->cabac.state, HEVCD_CTX);
            have_snapshot = true;
        }
        /* HEVC_TRACE: how far into the NAL each coding tree unit got.
         * When a slice does not land, this says where it stopped being
         * right - a unit that consumed implausibly little is where to
         * look, not the one that ran out of data. */
        if (getenv("HEVC_TRACE")) {
            const long n_read = (long)((d->cabac.ptr - d->cabac.start) * 8
                                      - d->cabac.cache_bits);
            fprintf(stderr, "ctu ts %d rs %d (%d,%d): %ld bit su %ld" "\n",
                    ts, addr, x, y, n_read, (long)(n - first) * 8);
        }
        if (hevcd_overrun(&d->cabac)) return 3;

        const int fine = hevcd_terminate(&d->cabac);
        if (fine) {
            /* ⚠️ It has to end after the LAST one in TILE SCAN, which
             * with tiles is not the bottom right unit of the picture. */
            return (ts == count - 1) ? 0 : 1;
        }

        /* 7.3.8.1: the substream ends when the next unit starts a new
         * tile, or under wavefront a new row. */
        const bool tile_break = tiles && ts + 1 < count
                                && d->tile_of_ts[ts + 1] != d->tile_of_ts[ts];
        const bool row_break = wpp && addr + 1 < count
                               && (addr + 1) % sps->ctb_width == 0;
        if (tile_break || row_break) {
            if (!hevcd_terminate(&d->cabac))
                return 6;                  /* the bit is defined to be one */

            const size_t n_used = h264d_cabac_byte_pos(&d->cabac);

            /* Where the header says this substream ends. One entry point
             * per substream, counted in the order they are walked, which
             * is why this is a running index and not a row number: under
             * tiles the substreams are tiles. */
            size_t step = n_used;
            if (substream < sl->num_entry_point_offsets) {
                const uint32_t fin = sl->entry_point[substream];
                const uint32_t ini = substream > 0
                                     ? sl->entry_point[substream - 1] : 0;
                step = (size_t)(fin - ini);
                /* A substream may stop short of what the header allows -
                 * the bytes left over are the engine's own look-ahead. It
                 * may not run past it: that is one read wrongly. */
                if (n_used > step) return 7;
            }
            substream++;
            if (step >= rest) return 3;
            base += step;
            rest -= step;
            h264d_cabac_init_engine(&d->cabac, base, rest);
            /* 8.6.1: a new substream predicts its first quantisation
             * group from the slice's own parameter, not from where the
             * previous one happened to end. */
            d->qg_restarts = true;

            /* 9.3.1. A tile starts from nothing: no other tile's state
             * carries into it, which is the whole point of tiles.
             * ⚠️ A wavefront row is the opposite - it continues from the
             * snapshot taken two units into the row above - so the two
             * cases cannot share this line. */
            if (tile_break)
                hevcd_cabac_ctx_init(d->cabac.state, init_type, sl->qp);
            else if (have_snapshot && sps->ctb_width >= 2)
                memcpy(d->cabac.state, snapshot, HEVCD_CTX);
            else
                hevcd_cabac_ctx_init(d->cabac.state, init_type, sl->qp);
        }
    }
    (void)progress;
    return 2;                                 /* ran out of CTUs first */
}

/* ------------------------------------------------------------- the API */

hevc_decoder_t *hevc_decoder_create(void *gpu, int width, int height)
{
    hevc_decoder_t *h = calloc(1, sizeof *h);
    if (!h) return NULL;
    h->gpu = gpu;
    h->width = width;
    h->height = height;
    h->d.buf = h->buffer;
    h->d.n_buf = IMG_SLOTS;
    return h;
}

void hevc_decoder_destroy(hevc_decoder_t *h)
{
    if (!h) return;
    for (int i = 0; i < IMG_SLOTS; i++) free_img(&h->buffer[i]);
    hevcd_t *d = &h->d;
    hevcd_free_tiles(d);
    free(d->ct_depth); free(d->intra_mode); free(d->min_tb_addr_zs);
    free(d->qp_y_map); free(d->edges); free(d->no_filter);
    free(d->skip); free(d->cbf_map);
    hevcd_free_filters(d);
    free(h);
}

void hevc_decoder_set_references(hevc_decoder_t *h, const uintptr_t *id,
                                 const int *poc, int n)
{
    for (int i = 0; i < IMG_SLOTS; i++) {
        if (!h->buffer[i].is_valid) continue;
        bool serve = false;
        for (int k = 0; k < n && !serve; k++)
            if (h->surface_id[i] == id[k] && h->buffer[i].poc == poc[k]) serve = true;
        if (!serve) h->buffer[i].is_valid = false;
    }
}

int hevc_decoder_begin_picture(hevc_decoder_t *h, const hevc_sps_t *sps,
                               const hevc_pps_t *pps, uintptr_t id, int poc)
{
    h->sps = *sps;
    h->pps = *pps;
    if (open_picture(&h->d, &h->sps, poc)) return -1;
    for (int i = 0; i < IMG_SLOTS; i++)
        if (&h->buffer[i] == h->d.current) h->surface_id[i] = id;
    h->is_open = true;
    return 0;
}

int hevc_decoder_slice(hevc_decoder_t *h, const hevc_slice_t *sl,
                       const uint8_t *rbsp, size_t n)
{
    if (!h->is_open || !h->d.current) return 5;
    h->last_one = *sl;
    h->d.slice = &h->last_one;
    build_lists(&h->d, &h->last_one);
    return walk_slice(&h->d, &h->sps, &h->pps, &h->last_one, rbsp, n);
}

void hevc_decoder_end_picture(hevc_decoder_t *h)
{
    if (!h->is_open || !h->d.current) return;
    if (h->d.slice) { hevcd_deblock(&h->d); hevcd_sao(&h->d); }
    h->is_open = false;
}

const uint8_t *hevc_decoder_plane(const hevc_decoder_t *h, int plane,
                                  int *stride)
{
    if (!h->d.current || plane < 0 || plane > 2) return NULL;
    if (stride) *stride = h->d.current->stride[plane];
    return h->d.current->plane[plane];
}

void hevc_decoder_unescape(hevc_decoder_t *h, const hevc_slice_t *sl)
{
    unescape(&h->d, sl);
}

void hevc_decoder_shift_entry_points(hevc_slice_t *s, const uint8_t *grezzo,
                                     size_t n_grezzo, size_t first)
{
    shift_entry_points(s, grezzo, n_grezzo, first);
}

/* ⚠️ The surface wants the two chroma planes interleaved, and it is
 * written once, here, rather than plane by plane as the picture is
 * decoded: surface memory is write-combining, which is fast to write
 * straight through and very slow to read back or revisit. */
int hevc_decoder_load(hevc_decoder_t *h, gpu_image_t out, gpu_memory_t mem)
{
    if (!h->gpu) return 0;                 /* the harness keeps the planes */
    const hevcd_img_t *g = h->d.current;
    if (!g || !g->plane[0]) return -1;

    const int cw = h->width / 2, ch = h->height / 2;
    const int ten_bit = h->sps.bit_depth_luma > 8;
    const size_t sample = ten_bit ? 2 : 1;

    uint8_t *uv = malloc((size_t)cw * 2 * ch * sample);
    if (!uv) return -1;

    if (ten_bit) {
        /* ⚠️ Every one of these is a sample count, so every offset is
         * multiplied. The shift into P010's high bits is not here - it
         * belongs to the upload, which is the part that knows what a
         * surface format is. */
        for (int r = 0; r < ch; r++) {
            const uint16_t *a = (const uint16_t *)g->plane[1]
                                + (size_t)r * g->stride[1];
            const uint16_t *b = (const uint16_t *)g->plane[2]
                                + (size_t)r * g->stride[2];
            uint16_t *o = (uint16_t *)uv + (size_t)r * cw * 2;
            for (int x = 0; x < cw; x++) { o[2 * x] = a[x]; o[2 * x + 1] = b[x]; }
        }
        const int r = gpu_compute_upload_p010(h->gpu, &out, mem,
                                              (const uint16_t *)g->plane[0],
                                              g->stride[0] * 2,
                                              (const uint16_t *)uv, cw * 4,
                                              h->width, h->height);
        free(uv);
        return r;
    }

    for (int r = 0; r < ch; r++) {
        const uint8_t *a = g->plane[1] + (size_t)r * g->stride[1];
        const uint8_t *b = g->plane[2] + (size_t)r * g->stride[2];
        uint8_t *o = uv + (size_t)r * cw * 2;
        for (int x = 0; x < cw; x++) { o[2 * x] = a[x]; o[2 * x + 1] = b[x]; }
    }
    const int r = gpu_compute_upload_nv12(h->gpu, &out, mem,
                                          g->plane[0], g->stride[0],
                                          uv, cw * 2, h->width, h->height);
    free(uv);
    return r;
}

const char *hevc_decoder_reason(int e)
{
    return slice_reason(e);
}
