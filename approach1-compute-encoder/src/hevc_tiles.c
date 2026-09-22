/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_tiles.c - the tile layout and the two scan orders, Rec. ITU-T
 * H.265 clause 6.5.1.
 *
 * A picture without tiles is a picture with one tile covering all of it.
 * The maps below are built that way too, so there is no second code path
 * to keep in step: without tiles the two scans are the same map and every
 * unit carries tile zero, and every caller can stop asking.
 */
#include "hevc_dec_internal.h"

#include <stdlib.h>
#include <string.h>

/* 6.5.1: the column widths, in coding tree units.
 *
 * ⚠️ Uniform spacing is not "the same width each". It is the running
 * difference of a rounded division, which spreads the remainder over the
 * columns instead of piling it on the last one. Writing it as a division
 * plus a fixed-up last column gives the same total and the wrong
 * boundaries, and a wrong boundary here puts every unit after it in the
 * wrong tile. */
static void column_widths(const hevc_pps_t *pps, int in_ctbs, int n, int *out)
{
    if (pps->uniform_spacing) {
        for (int i = 0; i < n; i++)
            out[i] = ((i + 1) * in_ctbs) / n - (i * in_ctbs) / n;
        return;
    }
    int used = 0;
    for (int i = 0; i < n - 1; i++) {
        out[i] = pps->column_width[i];
        used += out[i];
    }
    out[n - 1] = in_ctbs - used;
}

static void row_heights(const hevc_pps_t *pps, int in_ctbs, int n, int *out)
{
    if (pps->uniform_spacing) {
        for (int i = 0; i < n; i++)
            out[i] = ((i + 1) * in_ctbs) / n - (i * in_ctbs) / n;
        return;
    }
    int used = 0;
    for (int i = 0; i < n - 1; i++) {
        out[i] = pps->row_height[i];
        used += out[i];
    }
    out[n - 1] = in_ctbs - used;
}

/* Builds, for this picture, the raster-to-tile-scan map, its inverse, and
 * which tile each unit belongs to. Returns 0, or -1 if the layout does
 * not fit the picture or there is no memory.
 *
 * Called once per picture rather than per slice: the layout comes from
 * the picture parameter set and every slice of the picture shares it. */
int hevcd_prepare_tiles(hevcd_t *d)
{
    const hevc_sps_t *sps = d->sps;
    const hevc_pps_t *pps = d->pps;
    const int w = sps->ctb_width, h = sps->ctb_height;
    const int count = sps->ctb_count;
    const int nc = pps->tiles_enabled ? pps->num_tile_columns : 1;
    const int nr = pps->tiles_enabled ? pps->num_tile_rows : 1;

    if (nc < 1 || nr < 1 || nc > w || nr > h) return -1;

    if (d->n_tile_map < (size_t)count) {
        free(d->rs_to_ts); free(d->ts_to_rs); free(d->tile_of_ts);
        d->rs_to_ts = malloc((size_t)count * sizeof *d->rs_to_ts);
        d->ts_to_rs = malloc((size_t)count * sizeof *d->ts_to_rs);
        d->tile_of_ts = malloc((size_t)count * sizeof *d->tile_of_ts);
        d->n_tile_map = (size_t)count;
        if (!d->rs_to_ts || !d->ts_to_rs || !d->tile_of_ts) {
            d->n_tile_map = 0;
            return -1;
        }
    }

    int col_w[32], row_h[32];
    int col_bd[33], row_bd[33];
    column_widths(pps, w, nc, col_w);
    row_heights(pps, h, nr, row_h);

    col_bd[0] = 0;
    for (int i = 0; i < nc; i++) {
        if (col_w[i] <= 0) return -1;
        col_bd[i + 1] = col_bd[i] + col_w[i];
    }
    row_bd[0] = 0;
    for (int j = 0; j < nr; j++) {
        if (row_h[j] <= 0) return -1;
        row_bd[j + 1] = row_bd[j] + row_h[j];
    }
    if (col_bd[nc] != w || row_bd[nr] != h) return -1;

    for (int rs = 0; rs < count; rs++) {
        const int tx = rs % w, ty = rs / w;
        int tile_x = 0, tile_y = 0;
        for (int i = 0; i < nc; i++) if (tx >= col_bd[i]) tile_x = i;
        for (int j = 0; j < nr; j++) if (ty >= row_bd[j]) tile_y = j;

        /* Every whole tile to the left in this band, then every whole
         * band above, then the position inside this tile. */
        int ts = 0;
        for (int i = 0; i < tile_x; i++) ts += row_h[tile_y] * col_w[i];
        for (int j = 0; j < tile_y; j++) ts += w * row_h[j];
        ts += (ty - row_bd[tile_y]) * col_w[tile_x] + (tx - col_bd[tile_x]);

        d->rs_to_ts[rs] = ts;
        d->ts_to_rs[ts] = rs;
    }

    /* ⚠️ Indexed by TILE-SCAN address, not raster. The loop that walks the
     * picture asks "did the tile change since the last unit", and it walks
     * in tile scan. */
    int tile = 0;
    for (int j = 0; j < nr; j++) {
        for (int i = 0; i < nc; i++, tile++) {
            for (int y = row_bd[j]; y < row_bd[j + 1]; y++)
                for (int x = col_bd[i]; x < col_bd[i + 1]; x++)
                    d->tile_of_ts[d->rs_to_ts[y * w + x]] = tile;
        }
    }

    d->n_tiles = nc * nr;
    return 0;
}

void hevcd_free_tiles(hevcd_t *d)
{
    free(d->rs_to_ts); d->rs_to_ts = NULL;
    free(d->ts_to_rs); d->ts_to_rs = NULL;
    free(d->tile_of_ts); d->tile_of_ts = NULL;
    d->n_tile_map = 0;
    d->n_tiles = 0;
}

/* Which tile covers the unit at these LUMA coordinates. Used by the
 * availability rule and by the loop filters, both of which think in
 * samples rather than in unit addresses. */
int hevcd_tile_at(const hevcd_t *d, int x, int y)
{
    /* The common case by far, and worth one branch: with a single tile
     * every answer is zero and the two map lookups are waste. */
    if (d->n_tiles <= 1 || !d->tile_of_ts) return 0;
    const hevc_sps_t *sps = d->sps;
    const int rs = (y >> sps->log2_ctb) * sps->ctb_width + (x >> sps->log2_ctb);
    if (rs < 0 || rs >= sps->ctb_count) return 0;
    return d->tile_of_ts[d->rs_to_ts[rs]];
}
