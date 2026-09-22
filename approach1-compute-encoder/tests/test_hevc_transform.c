/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * test_hevc_transform.c - the inverse transforms against the matrices
 * they are supposed to be.
 *
 * hevcd_transform does not multiply by a matrix. The DCT reads one row of
 * a shared table with a stride, and the 4x4 intra luma DST is factored
 * into four multiplications instead of sixteen. Both are worth doing and
 * both are easy to get wrong in a way that leaves most of the output
 * right: a DST written with x1 where x2 belongs gives three correct
 * coefficients out of four, which does not look like a transform bug at
 * all when you meet it in a picture.
 *
 * ⚠️ The DST matrix here is typed from clause 8.6.4.2 and not derived
 * from the factored form, which is the whole point. A reference computed
 * from the same expression would agree with any mistake in it.
 */
#include "hevc_dec_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int faults;

/* Table in 8.6.4.2, transMatrix for the 4x4 luma intra case. */
static const int dst[4][4] = {
    { 29, 55, 74, 84 },
    { 74, 74,  0, -74 },
    { 84, -29, -74, 55 },
    { 55, -84, 74, -29 },
};

static int clip16(int v)
{
    return v < -32768 ? -32768 : (v > 32767 ? 32767 : v);
}

/* 8.6.4.2 written the long way: columns, shift by seven, rows, shift by
 * twenty minus the bit depth. Both stages clipped to sixteen bits. */
static void reference(const int16_t *in, int16_t *out, int n, bool usa_dst)
{
    int16_t tmp[32 * 32];

    for (int x = 0; x < n; x++)
        for (int y = 0; y < n; y++) {
            int s = 0;
            for (int k = 0; k < n; k++) {
                const int m = usa_dst ? dst[k][y]
                                      : hevcd_dct[k * (32 / n)][y];
                s += m * in[k * n + x];
            }
            tmp[y * n + x] = (int16_t)clip16((s + 64) >> 7);
        }

    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++) {
            int s = 0;
            for (int k = 0; k < n; k++) {
                const int m = usa_dst ? dst[k][x]
                                      : hevcd_dct[k * (32 / n)][x];
                s += m * tmp[y * n + k];
            }
            out[y * n + x] = (int16_t)clip16((s + 2048) >> 12);
        }
}

static unsigned semi = 12345;

static int next_up(int amplitude)
{
    semi = semi * 1103515245u + 12345u;
    return (int)((semi >> 16) % (unsigned)(2 * amplitude + 1)) - amplitude;
}

static void compare(const char *what, const int16_t *a, const int16_t *b,
                      int count)
{
    int differing = 0, worst = 0;
    for (int i = 0; i < count; i++)
        if (a[i] != b[i]) {
            differing++;
            const int e = abs(a[i] - b[i]);
            if (e > worst) worst = e;
        }
    if (differing) {
        printf("  %-34s %d of %d differ, worst by %d\n",
               what, differing, count, worst);
        faults++;
    } else {
        printf("  %-34s equal\n", what);
    }
}

/* One impulse at a time: each coefficient on its own says which basis
 * function came out, so a swapped row shows up on its own line instead of
 * being averaged away with thirty-one others. */
static void impulsi(int log2_size, bool usa_dst)
{
    const int n = 1 << log2_size;
    int16_t ours[32 * 32], expected[32 * 32], input[32 * 32];
    char what[64];
    int differing = 0;

    for (int k = 0; k < n * n; k++) {
        memset(input, 0, sizeof(int16_t) * (size_t)n * n);
        input[k] = 256;
        memcpy(ours, input, sizeof(int16_t) * (size_t)n * n);
        hevcd_transform(ours, log2_size, usa_dst, 8);
        reference(input, expected, n, usa_dst);
        if (memcmp(ours, expected, sizeof(int16_t) * (size_t)n * n))
            differing++;
    }
    snprintf(what, sizeof what, "%s %dx%d, one impulse at a time",
             usa_dst ? "DST" : "DCT", n, n);
    if (differing) {
        printf("  %-34s %d basis vectors of %d wrong\n", what, differing, n * n);
        faults++;
    } else {
        printf("  %-34s equal\n", what);
    }
}

static void at_random(int log2_size, bool usa_dst)
{
    const int n = 1 << log2_size;
    int16_t ours[32 * 32], expected[32 * 32], input[32 * 32];
    char what[64];

    snprintf(what, sizeof what, "%s %dx%d, random blocks",
             usa_dst ? "DST" : "DCT", n, n);

    for (int pass_index = 0; pass_index < 64; pass_index++) {
        for (int i = 0; i < n * n; i++) input[i] = (int16_t)next_up(4000);
        memcpy(ours, input, sizeof(int16_t) * (size_t)n * n);
        hevcd_transform(ours, log2_size, usa_dst, 8);
        reference(input, expected, n, usa_dst);
        if (memcmp(ours, expected, sizeof(int16_t) * (size_t)n * n)) {
            compare(what, ours, expected, n * n);
            return;
        }
    }
    printf("  %-34s equal\n", what);
}

int main(void)
{
    printf("the inverse transforms against their own matrices\n");
    impulsi(2, true);
    at_random(2, true);
    for (int l = 2; l <= 5; l++) {
        impulsi(l, false);
        at_random(l, false);
    }

    printf("\n");
    if (faults) {
        printf("%d prove fallite\n", faults);
        return 1;
    }
    printf("all good\n");
    return 0;
}
