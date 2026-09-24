/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_filter.c - the deblocking filter, Rec. ITU-T H.265 clause 8.7.2.
 *
 * Block transforms leave steps at the block edges, and the eye finds a
 * straight edge that is not in the picture far more readily than it finds
 * the error that produced it. So the standard smooths them - and does it
 * normatively, in the decoding loop, because the filtered picture is what
 * the next one predicts from. A decoder that skips it does not merely look
 * worse: it drifts.
 *
 * ⚠️ Every vertical edge in the picture is filtered before any horizontal
 * one, and the horizontal pass reads what the vertical pass wrote. Doing
 * it edge by edge, both directions at once, gives a different picture.
 * Two passes over the whole picture is the simplest way to be sure, and
 * the passes are independent inside themselves: the edges of one direction
 * are eight samples apart and reach four, so no two of them touch.
 */
#include "hevc_dec_internal.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* The edge offset at eight bits, sixteen samples at a time. SSSE3 for
 * _mm_shuffle_epi8, which looks the five cases up in one instruction; the
 * decoder asks for it at run time, as hevc_mc.c does, and BC250_HEVC_NOSIMD
 * turns it off to compare with the scalar loop beside it.
 *
 * ⚠️ The samples are compared as signed bytes after flipping the top bit,
 * which orders them the same way as unsigned ones: SSE has no unsigned
 * byte comparison. */
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

/* ⚠️ Atomic, because the first call can come from several filter threads
 * at once. They would all store the same answer, but two plain stores to
 * one int from two threads are a data race all the same, and
 * ThreadSanitizer says so. */
static int sao_vector(void)
{
    static _Atomic int answer = -1;
    int a = atomic_load_explicit(&answer, memory_order_relaxed);
    if (a < 0) {
        a = !getenv("BC250_HEVC_NOSIMD")
            && __builtin_cpu_supports("ssse3") ? 1 : 0;
        atomic_store_explicit(&answer, a, memory_order_relaxed);
    }
    return a;
}

__attribute__((target("ssse3")))
static void sao_edge_row_ssse3(uint8_t *out, const uint8_t *cur,
                               const uint8_t *a, const uint8_t *b, int n,
                               const int8_t table[16])
{
    const __m128i flip = _mm_set1_epi8((char)0x80);
    const __m128i two = _mm_set1_epi8(2);
    const __m128i zero = _mm_setzero_si128();
    const __m128i lut = _mm_loadu_si128((const __m128i *)table);
    int x = 0;
    for (; x + 16 <= n; x += 16) {
        const __m128i v = _mm_loadu_si128((const __m128i *)(cur + x));
        const __m128i vs = _mm_xor_si128(v, flip);
        const __m128i as = _mm_xor_si128(
            _mm_loadu_si128((const __m128i *)(a + x)), flip);
        const __m128i bs = _mm_xor_si128(
            _mm_loadu_si128((const __m128i *)(b + x)), flip);
        /* sign(v - a): the two comparisons are all ones where true. */
        const __m128i sa = _mm_sub_epi8(_mm_cmpgt_epi8(as, vs),
                                        _mm_cmpgt_epi8(vs, as));
        const __m128i sb = _mm_sub_epi8(_mm_cmpgt_epi8(bs, vs),
                                        _mm_cmpgt_epi8(vs, bs));
        const __m128i idx = _mm_add_epi8(_mm_add_epi8(sa, sb), two);
        const __m128i off = _mm_shuffle_epi8(lut, idx);
        const __m128i neg = _mm_cmpgt_epi8(zero, off);
        const __m128i lo = _mm_add_epi16(_mm_unpacklo_epi8(v, zero),
                                         _mm_unpacklo_epi8(off, neg));
        const __m128i hi = _mm_add_epi16(_mm_unpackhi_epi8(v, zero),
                                         _mm_unpackhi_epi8(off, neg));
        _mm_storeu_si128((__m128i *)(out + x), _mm_packus_epi16(lo, hi));
    }
    for (; x < n; x++) {
        const int v = cur[x];
        const int idx = 2 + (v > a[x]) - (v < a[x]) + (v > b[x]) - (v < b[x]);
        const int r = v + table[idx];
        out[x] = (uint8_t)(r < 0 ? 0 : (r > 255 ? 255 : r));
    }
}
#endif

#define BIT_DEPTH 8
#include "hevc_pixel.h"
#include "hevc_filter_template.c"
#undef BIT_DEPTH

#define BIT_DEPTH 10
#include "hevc_pixel.h"
#include "hevc_filter_template.c"
#undef BIT_DEPTH

/* ------------------------------------------- the filters on many threads
 *
 * Deblocking and SAO used to run on one thread after the wavefront had
 * finished with sixteen, and at 4K they were most of what was left: the
 * picture was decoded in parallel and then filtered in series.
 *
 * Four stages, each split by coding tree block rows, with every thread
 * waiting for the others between two stages:
 *
 *   0. the vertical edges      - 8.7.2 filters all of them first,
 *   1. the horizontal edges    - and these read what stage 0 wrote,
 *   2. the copy SAO reads      - the whole deblocked picture,
 *   3. SAO                     - which reads the copy around each block.
 *
 * Inside a stage the rows are independent (see one_direction() for why
 * the horizontal edges are), so the rows are simply handed out in order.
 *
 * ⚠️ One thread runs exactly the same stages: the single-threaded suites
 * exercise this code and not some other serial path beside it. */

#define FILTER_MAX_THREAD 16

typedef struct {
    hevcd_t *d;
    bool ten, deblock, sao;
    int rows;
    _Atomic int next[4];            /* the next row of each stage */

    /* Who takes part, and the wait between two stages. ⚠️ `taking_part`
     * is only known once the threads exist - one may fail to start - so
     * every worker waits for it before anything else. */
    pthread_mutex_t m;
    pthread_cond_t c;
    int taking_part, arrived, generation;
} filter_job_t;

static void stage_done(filter_job_t *j)
{
    pthread_mutex_lock(&j->m);
    const int gen = j->generation;
    if (++j->arrived == j->taking_part) {
        j->arrived = 0;
        j->generation++;
        pthread_cond_broadcast(&j->c);
    } else {
        while (gen == j->generation) pthread_cond_wait(&j->c, &j->m);
    }
    pthread_mutex_unlock(&j->m);
}

static void stage_row(filter_job_t *j, int stage, int ry)
{
    hevcd_t *d = j->d;
    switch (stage) {
    case 0: if (j->ten) deblock_row_10(d, true, ry);  else deblock_row_8(d, true, ry);  break;
    case 1: if (j->ten) deblock_row_10(d, false, ry); else deblock_row_8(d, false, ry); break;
    case 2: if (j->ten) sao_copy_row_10(d, ry);       else sao_copy_row_8(d, ry);       break;
    case 3: if (j->ten) sao_row_10(d, ry);            else sao_row_8(d, ry);            break;
    }
}

static void run_stage(filter_job_t *j, int stage)
{
    for (;;) {
        const int ry = atomic_fetch_add_explicit(&j->next[stage], 1,
                                                 memory_order_relaxed);
        if (ry >= j->rows) break;
        stage_row(j, stage, ry);
    }
}

static void *filter_worker(void *arg)
{
    filter_job_t *j = arg;
    pthread_mutex_lock(&j->m);
    while (!j->taking_part) pthread_cond_wait(&j->c, &j->m);
    pthread_mutex_unlock(&j->m);

    if (j->deblock) {
        run_stage(j, 0);
        stage_done(j);
        run_stage(j, 1);
        stage_done(j);
    }
    if (j->sao) {
        run_stage(j, 2);
        stage_done(j);
        run_stage(j, 3);
    }
    return NULL;
}

static int filter_threads(int rows)
{
    const char *s = getenv("BC250_HEVC_THREAD");
    int n = s ? atoi(s) : (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) n = 1;
    if (n > FILTER_MAX_THREAD) n = FILTER_MAX_THREAD;
    if (n > rows) n = rows;
    return n;
}

void hevcd_loop_filters(hevcd_t *d)
{
    if (!d->sps || !d->plane[0]) return;

    filter_job_t j;
    memset(&j, 0, sizeof j);
    j.d = d;
    j.ten = d->sps->bit_depth_luma > 8;
    j.rows = d->sps->ctb_height;
    j.deblock = d->edges != NULL;
    /* ⚠️ After deciding about deblocking, not before, and on this thread:
     * it allocates the copy, which the workers then only fill. */
    j.sao = j.ten ? sao_prepare_10(d) : sao_prepare_8(d);
    if (!j.deblock && !j.sao) return;

    pthread_mutex_init(&j.m, NULL);
    pthread_cond_init(&j.c, NULL);

    const int want = filter_threads(j.rows);
    pthread_t t[FILTER_MAX_THREAD];
    int alive = 0;
    for (int i = 1; i < want; i++)
        if (pthread_create(&t[alive], NULL, filter_worker, &j) == 0) alive++;

    pthread_mutex_lock(&j.m);
    j.taking_part = alive + 1;
    pthread_cond_broadcast(&j.c);
    pthread_mutex_unlock(&j.m);

    filter_worker(&j);                  /* this thread works too */
    for (int i = 0; i < alive; i++) pthread_join(t[i], NULL);

    pthread_mutex_destroy(&j.m);
    pthread_cond_destroy(&j.c);
}

void hevcd_free_filters(hevcd_t *d)
{
    free(d->sao);
    d->sao = NULL;
    d->n_sao = 0;
    for (int c = 0; c < 3; c++) { free(d->copy_of[c]); d->copy_of[c] = NULL; }
    d->n_copy = 0;
}
