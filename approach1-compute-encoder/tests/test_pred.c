/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * test_pred.c - intra prediction against a literal reading of the standard.
 *
 * h264_pred.c collapses the standard's cases: it lays the neighbours out as
 * one array around the corner so that, for instance, the three cases of
 * Diagonal Down Right become a single expression. That is a rewrite, and a
 * rewrite is where an off-by-one gets in.
 *
 * So this file transcribes clause 8.3.1.2 and 8.3.2.2 the other way -
 * literally, case by case, p[x,y] by p[x,y], with no cleverness and no
 * regard for speed - and checks the two agree on every sample of every mode
 * over random references. If the compact form is wrong anywhere, the sample
 * it is wrong on is printed.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "h264_pred.h"

static uint64_t seed = 0x243F6A8885A308D3ull;
static uint32_t random_u32(void)
{
    seed ^= seed >> 12;
    seed ^= seed << 25;
    seed ^= seed >> 27;
    return (uint32_t)((seed * 0x2545F4914F6CDD1Dull) >> 32);
}

/* ---- the standard, read literally ------------------------------------ */

typedef struct {
    const uint8_t *t;      /* p[0,-1] .. */
    const uint8_t *l;      /* p[-1,0] .. */
    uint8_t c;             /* p[-1,-1] */
} ref_t;

static int p(const ref_t *r, int x, int y)
{
    if (x == -1 && y == -1) return r->c;
    if (y == -1) return r->t[x];
    if (x == -1) return r->l[y];
    abort();
}

/* clause 8.3.1.2.1 to 8.3.1.2.9 */
static void literal4(uint8_t out[16], int mode, const ref_t *r,
                       int avail_top, int avail_left)
{
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            int v = 0;
            switch (mode) {
            case 0: v = p(r, x, -1); break;
            case 1: v = p(r, -1, y); break;
            case 2:
                if (avail_top && avail_left)
                    v = (p(r,0,-1)+p(r,1,-1)+p(r,2,-1)+p(r,3,-1)
                       + p(r,-1,0)+p(r,-1,1)+p(r,-1,2)+p(r,-1,3) + 4) >> 3;
                else if (avail_left)
                    v = (p(r,-1,0)+p(r,-1,1)+p(r,-1,2)+p(r,-1,3) + 2) >> 2;
                else if (avail_top)
                    v = (p(r,0,-1)+p(r,1,-1)+p(r,2,-1)+p(r,3,-1) + 2) >> 2;
                else
                    v = 128;
                break;
            case 3:
                if (x == 3 && y == 3)
                    v = (p(r,6,-1) + 3*p(r,7,-1) + 2) >> 2;
                else
                    v = (p(r,x+y,-1) + 2*p(r,x+y+1,-1) + p(r,x+y+2,-1) + 2) >> 2;
                break;
            case 4:
                if (x > y)
                    v = (p(r,x-y-2,-1) + 2*p(r,x-y-1,-1) + p(r,x-y,-1) + 2) >> 2;
                else if (x < y)
                    v = (p(r,-1,y-x-2) + 2*p(r,-1,y-x-1) + p(r,-1,y-x) + 2) >> 2;
                else
                    v = (p(r,0,-1) + 2*p(r,-1,-1) + p(r,-1,0) + 2) >> 2;
                break;
            case 5: {
                int z = 2*x - y;
                if (z == 0 || z == 2 || z == 4 || z == 6)
                    v = (p(r,x-(y>>1)-1,-1) + p(r,x-(y>>1),-1) + 1) >> 1;
                else if (z == 1 || z == 3 || z == 5)
                    v = (p(r,x-(y>>1)-2,-1) + 2*p(r,x-(y>>1)-1,-1) + p(r,x-(y>>1),-1) + 2) >> 2;
                else if (z == -1)
                    v = (p(r,-1,0) + 2*p(r,-1,-1) + p(r,0,-1) + 2) >> 2;
                else
                    v = (p(r,-1,y-1) + 2*p(r,-1,y-2) + p(r,-1,y-3) + 2) >> 2;
                break;
            }
            case 6: {
                int z = 2*y - x;
                if (z == 0 || z == 2 || z == 4 || z == 6)
                    v = (p(r,-1,y-(x>>1)-1) + p(r,-1,y-(x>>1)) + 1) >> 1;
                else if (z == 1 || z == 3 || z == 5)
                    v = (p(r,-1,y-(x>>1)-2) + 2*p(r,-1,y-(x>>1)-1) + p(r,-1,y-(x>>1)) + 2) >> 2;
                else if (z == -1)
                    v = (p(r,-1,0) + 2*p(r,-1,-1) + p(r,0,-1) + 2) >> 2;
                else
                    v = (p(r,x-1,-1) + 2*p(r,x-2,-1) + p(r,x-3,-1) + 2) >> 2;
                break;
            }
            case 7:
                if ((y & 1) == 0)
                    v = (p(r,x+(y>>1),-1) + p(r,x+(y>>1)+1,-1) + 1) >> 1;
                else
                    v = (p(r,x+(y>>1),-1) + 2*p(r,x+(y>>1)+1,-1) + p(r,x+(y>>1)+2,-1) + 2) >> 2;
                break;
            case 8: {
                int z = x + 2*y;
                if (z == 0 || z == 2 || z == 4)
                    v = (p(r,-1,y+(x>>1)) + p(r,-1,y+(x>>1)+1) + 1) >> 1;
                else if (z == 1 || z == 3)
                    v = (p(r,-1,y+(x>>1)) + 2*p(r,-1,y+(x>>1)+1) + p(r,-1,y+(x>>1)+2) + 2) >> 2;
                else if (z == 5)
                    v = (p(r,-1,2) + 3*p(r,-1,3) + 2) >> 2;
                else
                    v = p(r,-1,3);
                break;
            }
            default: abort();
            }
            out[y * 4 + x] = (uint8_t)v;
        }
    }
}

/* clause 8.3.2.2.1, reference sample filtering */
static void filter_literal(const uint8_t t[16], const uint8_t l[8], uint8_t c,
                             int avail_top, int avail_left, int avail_corner,
                             uint8_t ft[16], uint8_t fl[8], uint8_t *fc)
{
    memcpy(ft, t, 16);
    memcpy(fl, l, 8);
    *fc = c;
    if (avail_top) {
        ft[0] = (uint8_t)((avail_corner ? c : t[0]) + 2*t[0] + t[1] + 2) >> 0;
        ft[0] = (uint8_t)(((avail_corner ? c : t[0]) + 2*t[0] + t[1] + 2) >> 2);
        for (int i = 1; i <= 14; i++)
            ft[i] = (uint8_t)((t[i-1] + 2*t[i] + t[i+1] + 2) >> 2);
        ft[15] = (uint8_t)((t[14] + 2*t[15] + t[15] + 2) >> 2);
    }
    if (avail_left) {
        fl[0] = (uint8_t)(((avail_corner ? c : l[0]) + 2*l[0] + l[1] + 2) >> 2);
        for (int i = 1; i <= 6; i++)
            fl[i] = (uint8_t)((l[i-1] + 2*l[i] + l[i+1] + 2) >> 2);
        fl[7] = (uint8_t)((l[6] + 2*l[7] + l[7] + 2) >> 2);
    }
    if (avail_corner) {
        int a = avail_top  ? t[0] : c;
        int b = avail_left ? l[0] : c;
        *fc = (uint8_t)((a + 2*c + b + 2) >> 2);
    }
}

/* clause 8.3.2.2.2 to 8.3.2.2.10 */
static void literal8(uint8_t out[64], int mode, const ref_t *r,
                       int avail_top, int avail_left)
{
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            int v = 0;
            switch (mode) {
            case 0: v = p(r, x, -1); break;
            case 1: v = p(r, -1, y); break;
            case 2: {
                int s = 0;
                if (avail_top && avail_left) {
                    for (int i = 0; i < 8; i++) s += p(r,i,-1) + p(r,-1,i);
                    v = (s + 8) >> 4;
                } else if (avail_top) {
                    for (int i = 0; i < 8; i++) s += p(r,i,-1);
                    v = (s + 4) >> 3;
                } else if (avail_left) {
                    for (int i = 0; i < 8; i++) s += p(r,-1,i);
                    v = (s + 4) >> 3;
                } else {
                    v = 128;
                }
                break;
            }
            case 3:
                if (x == 7 && y == 7)
                    v = (p(r,14,-1) + 3*p(r,15,-1) + 2) >> 2;
                else
                    v = (p(r,x+y,-1) + 2*p(r,x+y+1,-1) + p(r,x+y+2,-1) + 2) >> 2;
                break;
            case 4:
                if (x > y)
                    v = (p(r,x-y-2,-1) + 2*p(r,x-y-1,-1) + p(r,x-y,-1) + 2) >> 2;
                else if (x < y)
                    v = (p(r,-1,y-x-2) + 2*p(r,-1,y-x-1) + p(r,-1,y-x) + 2) >> 2;
                else
                    v = (p(r,0,-1) + 2*p(r,-1,-1) + p(r,-1,0) + 2) >> 2;
                break;
            case 5: {
                int z = 2*x - y;
                if (z >= 0 && (z & 1) == 0)
                    v = (p(r,x-(y>>1)-1,-1) + p(r,x-(y>>1),-1) + 1) >> 1;
                else if (z >= 0)
                    v = (p(r,x-(y>>1)-2,-1) + 2*p(r,x-(y>>1)-1,-1) + p(r,x-(y>>1),-1) + 2) >> 2;
                else if (z == -1)
                    v = (p(r,-1,0) + 2*p(r,-1,-1) + p(r,0,-1) + 2) >> 2;
                else
                    v = (p(r,-1,y-2*x-1) + 2*p(r,-1,y-2*x-2) + p(r,-1,y-2*x-3) + 2) >> 2;
                break;
            }
            case 6: {
                int z = 2*y - x;
                if (z >= 0 && (z & 1) == 0)
                    v = (p(r,-1,y-(x>>1)-1) + p(r,-1,y-(x>>1)) + 1) >> 1;
                else if (z >= 0)
                    v = (p(r,-1,y-(x>>1)-2) + 2*p(r,-1,y-(x>>1)-1) + p(r,-1,y-(x>>1)) + 2) >> 2;
                else if (z == -1)
                    v = (p(r,-1,0) + 2*p(r,-1,-1) + p(r,0,-1) + 2) >> 2;
                else
                    v = (p(r,x-2*y-1,-1) + 2*p(r,x-2*y-2,-1) + p(r,x-2*y-3,-1) + 2) >> 2;
                break;
            }
            case 7:
                if ((y & 1) == 0)
                    v = (p(r,x+(y>>1),-1) + p(r,x+(y>>1)+1,-1) + 1) >> 1;
                else
                    v = (p(r,x+(y>>1),-1) + 2*p(r,x+(y>>1)+1,-1) + p(r,x+(y>>1)+2,-1) + 2) >> 2;
                break;
            case 8: {
                int z = x + 2*y;
                if (z < 13 && (z & 1) == 0)
                    v = (p(r,-1,y+(x>>1)) + p(r,-1,y+(x>>1)+1) + 1) >> 1;
                else if (z < 13)
                    v = (p(r,-1,y+(x>>1)) + 2*p(r,-1,y+(x>>1)+1) + p(r,-1,y+(x>>1)+2) + 2) >> 2;
                else if (z == 13)
                    v = (p(r,-1,6) + 3*p(r,-1,7) + 2) >> 2;
                else
                    v = p(r,-1,7);
                break;
            }
            default: abort();
            }
            out[y * 8 + x] = (uint8_t)v;
        }
    }
}


/* clause 8.3.3, Intra_16x16 */
static void literal16(uint8_t out[256], int mode, const ref_t *r,
                        int avail_top, int avail_left)
{
    int H = 0, V = 0, a = 0, b = 0, cc = 0;
    if (mode == 3) {
        for (int i = 0; i < 8; i++) {
            H += (i + 1) * (p(r, 8 + i, -1) - p(r, 6 - i, -1));
            V += (i + 1) * (p(r, -1, 8 + i) - p(r, -1, 6 - i));
        }
        a = 16 * (p(r, -1, 15) + p(r, 15, -1));
        b = (5 * H + 32) >> 6;
        cc = (5 * V + 32) >> 6;
    }
    for (int y = 0; y < 16; y++) {
        for (int x = 0; x < 16; x++) {
            int v = 0;
            switch (mode) {
            case 0: v = p(r, x, -1); break;
            case 1: v = p(r, -1, y); break;
            case 2: {
                int s = 0;
                if (avail_top && avail_left) {
                    for (int i = 0; i < 16; i++) s += p(r,i,-1) + p(r,-1,i);
                    v = (s + 16) >> 5;
                } else if (avail_top) {
                    for (int i = 0; i < 16; i++) s += p(r,i,-1);
                    v = (s + 8) >> 4;
                } else if (avail_left) {
                    for (int i = 0; i < 16; i++) s += p(r,-1,i);
                    v = (s + 8) >> 4;
                } else v = 128;
                break;
            }
            case 3: {
                v = (a + b * (x - 7) + cc * (y - 7) + 16) >> 5;
                v = v < 0 ? 0 : (v > 255 ? 255 : v);
                break;
            }
            default: abort();
            }
            out[y * 16 + x] = (uint8_t)v;
        }
    }
}

/* clause 8.3.4, chroma. Mode numbering is DC, horizontal, vertical, plane. */
static void letteraleC(uint8_t out[64], int mode, const ref_t *r,
                       int avail_top, int avail_left)
{
    int H = 0, V = 0, a = 0, b = 0, cc = 0;
    if (mode == 3) {
        for (int i = 0; i < 4; i++) {
            H += (i + 1) * (p(r, 4 + i, -1) - p(r, 2 - i, -1));
            V += (i + 1) * (p(r, -1, 4 + i) - p(r, -1, 2 - i));
        }
        a = 16 * (p(r, -1, 7) + p(r, 7, -1));
        b = (34 * H + 32) >> 6;
        cc = (34 * V + 32) >> 6;
    }
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            int v = 0;
            switch (mode) {
            case 0: {
                /* 8.3.4.1, one DC per 4x4 quadrant */
                int x0 = (x >> 2) * 4, y0 = (y >> 2) * 4;
                int st = 0, sl = 0;
                for (int i = 0; i < 4; i++) { st += p(r, x0 + i, -1); sl += p(r, -1, y0 + i); }
                if ((x0 == 0 && y0 == 0) || (x0 > 0 && y0 > 0)) {
                    if (avail_top && avail_left) v = (st + sl + 4) >> 3;
                    else if (avail_left)         v = (sl + 2) >> 2;
                    else if (avail_top)          v = (st + 2) >> 2;
                    else                         v = 128;
                } else if (x0 > 0 && y0 == 0) {
                    if (avail_top)               v = (st + 2) >> 2;
                    else if (avail_left)         v = (sl + 2) >> 2;
                    else                         v = 128;
                } else {
                    if (avail_left)              v = (sl + 2) >> 2;
                    else if (avail_top)          v = (st + 2) >> 2;
                    else                         v = 128;
                }
                break;
            }
            case 1: v = p(r, -1, y); break;
            case 2: v = p(r, x, -1); break;
            case 3:
                v = (a + b * (x - 3) + cc * (y - 3) + 16) >> 5;
                v = v < 0 ? 0 : (v > 255 ? 255 : v);
                break;
            default: abort();
            }
            out[y * 8 + x] = (uint8_t)v;
        }
    }
}

static int check16(void)
{
    uint8_t t[16], l[16], c, ours[256], its[256];
    int faults = 0;
    for (int pass_index = 0; pass_index < 20000; pass_index++) {
        for (int i = 0; i < 16; i++) { t[i] = (uint8_t)random_u32(); l[i] = (uint8_t)random_u32(); }
        c = (uint8_t)random_u32();
        for (int mode = 0; mode < 4; mode++) {
            for (int disp = (mode == 2 ? 0 : 3); disp <= 3; disp++) {
                int at = (disp >> 1) & 1, al = disp & 1;
                ref_t r = { t, l, c };
                h264d_pred16x16(ours, 16, mode, t, l, c, at, al);
                literal16(its, mode, &r, at, al);
                if (memcmp(ours, its, 256)) {
                    if (faults < 3)
                        printf("  16x16 mode %d, top=%d left=%d: differ\n", mode, at, al);
                    faults++;
                }
            }
        }
    }
    return faults;
}

static int provaC(void)
{
    uint8_t t[8], l[8], c, ours[64], its[64];
    int faults = 0;
    for (int pass_index = 0; pass_index < 20000; pass_index++) {
        for (int i = 0; i < 8; i++) { t[i] = (uint8_t)random_u32(); l[i] = (uint8_t)random_u32(); }
        c = (uint8_t)random_u32();
        for (int mode = 0; mode < 4; mode++) {
            for (int disp = (mode == 0 ? 0 : 3); disp <= 3; disp++) {
                int at = (disp >> 1) & 1, al = disp & 1;
                ref_t r = { t, l, c };
                h264d_pred_chroma(ours, 8, mode, t, l, c, at, al);
                letteraleC(its, mode, &r, at, al);
                if (memcmp(ours, its, 64)) {
                    if (faults < 3) {
                        printf("  chroma mode %d, top=%d left=%d:\n", mode, at, al);
                        for (int y = 0; y < 8; y++) {
                            printf("    ");
                            for (int x = 0; x < 8; x++)
                                printf("%4d%c", ours[y*8+x],
                                       ours[y*8+x] == its[y*8+x] ? ' ' : '*');
                            printf("\n");
                        }
                    }
                    faults++;
                }
            }
        }
    }
    return faults;
}

/* ---- the comparison --------------------------------------------------- */

static const char *mode_name[9] = {
    "Vertical", "Horizontal", "DC", "Diagonal Down Left", "Diagonal Down Right",
    "Vertical Right", "Horizontal Down", "Vertical Left", "Horizontal Up"
};

static int check4(void)
{
    uint8_t t[8], l[4], c, ours[16], its[16];
    int faults = 0;

    for (int pass_index = 0; pass_index < 20000; pass_index++) {
        for (int i = 0; i < 8; i++) t[i] = (uint8_t)random_u32();
        for (int i = 0; i < 4; i++) l[i] = (uint8_t)random_u32();
        c = (uint8_t)random_u32();
        /* The directional modes are only ever chosen when their references
         * exist, so only DC is exercised with a side missing. */
        for (int mode = 0; mode < 9; mode++) {
            for (int disp = (mode == 2 ? 0 : 3); disp <= 3; disp++) {
                int at = (disp >> 1) & 1, al = disp & 1;
                ref_t r = { t, l, c };
                h264d_pred4x4(ours, 4, mode, t, l, c, at, al);
                literal4(its, mode, &r, at, al);
                if (memcmp(ours, its, 16)) {
                    if (faults < 6) {
                        printf("  4x4 mode %d (%s), top=%d left=%d:\n",
                               mode, mode_name[mode], at, al);
                        for (int y = 0; y < 4; y++) {
                            printf("    ");
                            for (int x = 0; x < 4; x++)
                                printf("%4d%c", ours[y*4+x],
                                       ours[y*4+x] == its[y*4+x] ? ' ' : '*');
                            printf("   norma: ");
                            for (int x = 0; x < 4; x++) printf("%4d", its[y*4+x]);
                            printf("\n");
                        }
                    }
                    faults++;
                }
            }
        }
    }
    return faults;
}

static int check8(void)
{
    uint8_t t[16], l[8], c, ours[64], its[64];
    int faults = 0;

    for (int pass_index = 0; pass_index < 20000; pass_index++) {
        for (int i = 0; i < 16; i++) t[i] = (uint8_t)random_u32();
        for (int i = 0; i < 8; i++) l[i] = (uint8_t)random_u32();
        c = (uint8_t)random_u32();
        int top_right = (int)(random_u32() & 1);

        uint8_t tt[16];
        memcpy(tt, t, 16);
        if (!top_right) memset(tt + 8, tt[7], 8);

        for (int mode = 0; mode < 9; mode++) {
            for (int disp = (mode == 2 ? 0 : 3); disp <= 3; disp++) {
                int at = (disp >> 1) & 1, al = disp & 1;
                int ac = at && al;   /* the corner exists when both do */

                uint8_t ft[16], fl[8], fc;
                filter_literal(tt, l, c, at, al, ac, ft, fl, &fc);
                ref_t r = { ft, fl, fc };

                h264d_pred8x8_luma(ours, 8, mode, t, l, c, at, al, ac, top_right);
                literal8(its, mode, &r, at, al);
                if (memcmp(ours, its, 64)) {
                    if (faults < 4) {
                        printf("  8x8 mode %d (%s), top=%d left=%d top-right=%d:\n",
                               mode, mode_name[mode], at, al, top_right);
                        for (int y = 0; y < 8; y++) {
                            printf("    ");
                            for (int x = 0; x < 8; x++)
                                printf("%4d%c", ours[y*8+x],
                                       ours[y*8+x] == its[y*8+x] ? ' ' : '*');
                            printf("\n");
                        }
                    }
                    faults++;
                }
            }
        }
    }
    return faults;
}

int main(void)
{
    printf("1. Intra_4x4, nine modes against the literal standard\n");
    int g4 = check4();
    if (g4) { printf("   %d blocks differ\n", g4); return 1; }
    printf("   20000 rounds x 9 modes: identical\n");

    printf("2. Intra_8x8, nine modes plus the reference filter\n");
    int g8 = check8();
    if (g8) { printf("   %d blocks differ\n", g8); return 1; }
    printf("   20000 rounds x 9 modes: identical\n");

    printf("3. Intra_16x16, four modes\n");
    int g16 = check16();
    if (g16) { printf("   %d blocks differ\n", g16); return 1; }
    printf("   20000 rounds x 4 modes: identical\n");

    printf("4. chroma 8x8, four modes\n");
    int gc = provaC();
    if (gc) { printf("   %d blocks differ\n", gc); return 1; }
    printf("   20000 rounds x 4 modes: identical\n");

    printf("\nOK\n");
    return 0;
}
