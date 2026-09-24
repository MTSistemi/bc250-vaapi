/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_residual.c - residual_coding(), Rec. ITU-T H.265 clause 7.3.8.11.
 *
 * A transform block's coefficients, read backwards. Where the last one is
 * comes first; then the block is walked from there towards the DC in
 * groups of sixteen, and each group is read in passes - which coefficients
 * are there, which are bigger than one, which one is bigger than two, and
 * what is left over.
 *
 * ⚠️ Almost every context here depends on where in the block the
 * coefficient sits, and several on what the groups to the right and below
 * turned out to be. Nothing can be approximated: reading one bin against
 * the wrong context does not fail, it desynchronises the arithmetic
 * decoder several bins later, somewhere else entirely.
 */
#include "hevc_dec_internal.h"

#include <string.h>

/* 9.3.3.11: what is left of a level after the flags. A Golomb-Rice code
 * whose prefix escapes into exp-Golomb once it reaches three. */
static int read_rest(hevcd_cabac_t *c, int rice)
{
    int prefix = 0;
    while (prefix < 32 && hevcd_bypass(c))
        prefix++;

    if (prefix < 3)
        return (prefix << rice) + (int)(rice ? hevcd_bypass_n(c, rice) : 0);

    const int count = prefix - 3 + rice;
    if (count > 30) return 0;            /* a stream this broken is over */
    const uint32_t suffix = count ? hevcd_bypass_n(c, count) : 0;
    return (((1 << (prefix - 3)) + 2) << rice) + (int)suffix;
}

void hevcd_read_residual(hevcd_t *d, int x0, int y0, int log2_size, int c_idx)
{
    const hevc_pps_t *pps = d->pps;
    hevcd_cabac_t *c = &d->cabac;

    const int side = 1 << log2_size;
    memset(d->coeff, 0, (size_t)side * side * sizeof(int16_t));
    d->n_nz = 0;
    d->nz_max_x = -1;
    d->nz_max_y = -1;

    d->transform_skip = false;
    if (pps->transform_skip_enabled && !d->cu.transquant_bypass
        && log2_size == 2)
        d->transform_skip =
            hevcd_bin(c, HEVCD_CTX_TRANSFORM_SKIP_FLAG + (c_idx ? 1 : 0)) != 0;

    /* 8.4.4.2. An intra block small enough has its scan chosen by the
     * prediction mode: a near-horizontal prediction leaves a residual with
     * vertical structure, and the other way round. Anything larger, and
     * anything inter, is scanned diagonally. */
    int scan_idx = HEVCD_SCAN_DIAG;
    if (d->cu.pred_mode == HEVCD_MODE_INTRA
        && (log2_size == 2 || (log2_size == 3 && c_idx == 0))) {
        const int mode = c_idx
            ? d->cu.intra_mode_c
            : d->intra_mode[(y0 >> 2) * d->min_pu_width + (x0 >> 2)];
        if (mode >= 6 && mode <= 14) scan_idx = HEVCD_SCAN_VERT;
        else if (mode >= 22 && mode <= 30) scan_idx = HEVCD_SCAN_HORIZ;
    }

    /* --- where the last coefficient is, 9.3.3.10 -------------------- */
    const int max = (log2_size << 1) - 1;
    int off, shift;
    if (c_idx == 0) {
        off = 3 * (log2_size - 2) + ((log2_size - 1) >> 2);
        shift = (log2_size + 1) >> 2;
    } else {
        off = 15;
        shift = log2_size - 2;
    }

    int ux = 0, uy = 0;
    while (ux < max && hevcd_bin(c, HEVCD_CTX_LAST_SIGNIFICANT_COEFF_X_PREFIX
                                    + (ux >> shift) + off))
        ux++;
    while (uy < max && hevcd_bin(c, HEVCD_CTX_LAST_SIGNIFICANT_COEFF_Y_PREFIX
                                    + (uy >> shift) + off))
        uy++;
    if (ux > 3) {
        const int q = (ux >> 1) - 1;
        ux = (1 << q) * (2 + (ux & 1)) + (int)hevcd_bypass_n(c, q);
    }
    if (uy > 3) {
        const int q = (uy >> 1) - 1;
        uy = (1 << q) * (2 + (uy & 1)) + (int)hevcd_bypass_n(c, q);
    }
    if (scan_idx == HEVCD_SCAN_VERT) { const int t = ux; ux = uy; uy = t; }
    if (ux >= side || uy >= side) return;        /* malformed */

    /* --- the scans, and how far into them the last coefficient is --- */
    const uint8_t *sx, *sy, *gx, *gy;
    int count;
    if (scan_idx == HEVCD_SCAN_DIAG) {
        sx = hevcd_diag4_x; sy = hevcd_diag4_y;
        count = hevcd_diag4_inv[uy & 3][ux & 3];
        switch (side) {
        case 4: {
            static const uint8_t solo[1] = { 0 };
            gx = gy = solo;                 /* one group, at the corner */
            break;
        }
        case 8:
            count += hevcd_diag2_inv[uy >> 2][ux >> 2] << 4;
            gx = hevcd_diag2_x; gy = hevcd_diag2_y;
            break;
        case 16:
            count += hevcd_diag4_inv[uy >> 2][ux >> 2] << 4;
            gx = hevcd_diag4_x; gy = hevcd_diag4_y;
            break;
        default:
            count += hevcd_diag8_inv[uy >> 2][ux >> 2] << 4;
            gx = hevcd_diag8_x; gy = hevcd_diag8_y;
            break;
        }
    } else if (scan_idx == HEVCD_SCAN_HORIZ) {
        /* ⚠️ Only 4x4 and 8x8 blocks are ever scanned this way, so the
         * groups are always 2x2 and the inverse scan is always the 8x8
         * one. There is no sixteen-wide horizontal scan to look for. */
        gx = hevcd_horiz2_x; gy = hevcd_horiz2_y;
        sx = hevcd_horiz4_x; sy = hevcd_horiz4_y;
        count = hevcd_horiz8_inv[uy][ux];
    } else {
        gx = hevcd_horiz2_y; gy = hevcd_horiz2_x;
        sx = hevcd_horiz4_y; sy = hevcd_horiz4_x;
        count = hevcd_horiz8_inv[ux][uy];
    }
    count++;
    const int last_group = (count - 1) >> 4;
    const int side_g = side >> 2;

    uint8_t group[8][8];
    memset(group, 0, sizeof(group));

    /* ⚠️ Carried from one group to the next, not reset with them: the set
     * of contexts the next group starts from depends on how the previous
     * one ended. */
    int greater1_ctx = 1;

    for (int i = last_group; i >= 0; i--) {
        const int x_cg = gx[i], y_cg = gy[i];
        bool implicit_dc = false;

        if (i < last_group && i > 0) {
            int ctx = 0;
            if (x_cg < side_g - 1) ctx += group[y_cg][x_cg + 1];
            if (y_cg < side_g - 1) ctx += group[y_cg + 1][x_cg];
            const int inc = (ctx > 1 ? 1 : ctx) + (c_idx ? 2 : 0);
            group[y_cg][x_cg] = (uint8_t)
                hevcd_bin(c, HEVCD_CTX_SIGNIFICANT_COEFF_GROUP_FLAG + inc);
            implicit_dc = true;
        } else {
            /* The group holding the last coefficient, and the one holding
             * the DC, are coded by definition. */
            group[y_cg][x_cg] = 1;
        }
        if (!group[y_cg][x_cg]) continue;

        int prev = 0;
        if (x_cg < side_g - 1) prev = group[y_cg][x_cg + 1] ? 1 : 0;
        if (y_cg < side_g - 1) prev += group[y_cg + 1][x_cg] ? 2 : 0;

        /* 9.3.4.2.5 */
        int scf_off = c_idx ? 27 : 0;
        const uint8_t *map_of;
        if (log2_size == 2) {
            map_of = &hevcd_ctx_idx_map[0];
        } else {
            map_of = &hevcd_ctx_idx_map[(prev + 1) << 4];
            if (c_idx == 0) {
                if (x_cg > 0 || y_cg > 0) scf_off += 3;
                scf_off += (log2_size == 3)
                         ? ((scan_idx == HEVCD_SCAN_DIAG) ? 9 : 15) : 21;
            } else {
                scf_off += (log2_size == 3) ? 9 : 12;
            }
        }

        uint8_t sig[16];
        int n_sig = 0;
        int n_alto;
        if (i == last_group) {
            const int last_pos = (count - 1) & 15;
            sig[n_sig++] = (uint8_t)last_pos;
            n_alto = last_pos - 1;
        } else {
            n_alto = 15;
        }

        /* ⚠️ Written every time and counted only when significant. Which
         * way a significance bin goes is the least predictable thing in
         * the stream, and a branch on it was most of the mispredictions in
         * the whole decoder. */
        for (int n = n_alto; n > 0; n--) {
            const int inc = map_of[(sy[n] << 2) + sx[n]] + scf_off;
            const int b = hevcd_bin(c, HEVCD_CTX_SIGNIFICANT_COEFF_FLAG + inc);
            sig[n_sig] = (uint8_t)n;
            n_sig += b;
            implicit_dc = implicit_dc && !b;
        }
        if (n_alto >= 0) {
            if (implicit_dc) {
                sig[n_sig++] = 0;
            } else {
                const int inc = (i == 0) ? (c_idx ? 27 : 0) : (2 + scf_off);
                if (hevcd_bin(c, HEVCD_CTX_SIGNIFICANT_COEFF_FLAG + inc))
                    sig[n_sig++] = 0;
            }
        }
        if (n_sig == 0) continue;

        /* --- the levels, 9.3.4.2.6 and 9.3.4.2.7 ------------------- */
        int ctx_set = (i > 0 && c_idx == 0) ? 2 : 0;
        if (i != last_group && greater1_ctx == 0)
            ctx_set++;
        greater1_ctx = 1;

        uint8_t g1[8];
        memset(g1, 0, sizeof(g1));
        int first_g1 = -1;
        const int n_g1 = n_sig < 8 ? n_sig : 8;
        /* 9.3.4.2.6: a one sends the context to zero for good, a zero
         * moves it on from 1 to 3 and leaves 0 and 3 where they are. As a
         * table, so that the bin picks the next context without a branch. */
        static const uint8_t next_g1[2][4] = { { 0, 2, 3, 3 }, { 0, 0, 0, 0 } };
        for (int k = 0; k < n_g1; k++) {
            const int inc = (ctx_set << 2) + greater1_ctx + (c_idx ? 16 : 0);
            const int b =
                hevcd_bin(c, HEVCD_CTX_COEFF_ABS_LEVEL_GREATER1_FLAG + inc);
            g1[k] = (uint8_t)b;
            first_g1 = (b && first_g1 < 0) ? k : first_g1;
            greater1_ctx = next_g1[b][greater1_ctx];
        }
        if (first_g1 >= 0)
            g1[first_g1] += (uint8_t)
                hevcd_bin(c, HEVCD_CTX_COEFF_ABS_LEVEL_GREATER2_FLAG
                             + ctx_set + (c_idx ? 4 : 0));

        /* 7.4.9.11: when the run of coefficients is long enough, the sign
         * of the lowest one is not sent - it is carried by the parity of
         * the sum, which costs nothing because the sum has to be right
         * anyway. */
        const int alto = sig[0], basso = sig[n_sig - 1];
        const bool hidden_sign = pps->sign_data_hiding
                                 && !d->cu.transquant_bypass
                                 && (alto - basso) >= 4;

        const int n_signs = hidden_sign ? n_sig - 1 : n_sig;
        uint32_t signs = n_signs ? hevcd_bypass_n(c, n_signs) : 0;

        int rice = 0, sum = 0;
        int levels[16];
        for (int k = 0; k < n_sig; k++) {
            int base = (k < 8) ? 1 + g1[k] : 1;
            const int threshold = (k < 8) ? ((k == first_g1) ? 3 : 2) : 1;
            if (base == threshold) {
                base += read_rest(c, rice);
                if (base > (3 << rice) && rice < 4) rice++;
            }
            levels[k] = base;
            sum += base;
        }

        for (int k = 0; k < n_sig; k++) {
            bool negative;
            if (hidden_sign && k == n_sig - 1)
                negative = (sum & 1) != 0;
            else
                negative = ((signs >> (n_signs - 1 - k)) & 1) != 0;
            const int n = sig[k];
            const int xc = (x_cg << 2) + sx[n], yc = (y_cg << 2) + sy[n];
            d->coeff[yc * side + xc] =
                (int16_t)(negative ? -levels[k] : levels[k]);
            d->nz_pos[d->n_nz++] = (uint16_t)(yc * side + xc);
            d->nz_max_x = xc > d->nz_max_x ? xc : d->nz_max_x;
            d->nz_max_y = yc > d->nz_max_y ? yc : d->nz_max_y;
        }
    }
}
