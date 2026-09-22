/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * h264_threads.c - reconstruction across several cores, as a wavefront.
 *
 * H.264 has no entropy_coding_sync: a slice's arithmetic decoder can only
 * be started at the slice's beginning, so reading the syntax is strictly
 * serial and stays on one thread. Reconstruction is a different matter.
 * Once the syntax and the residuals are in, what a macroblock needs from
 * its own picture is only the reconstructed samples its intra prediction
 * reads, and those come from three neighbours: left, above, and above
 * right.
 *
 * That is a wavefront. Row r may reconstruct column x as soon as row r - 1
 * has finished column x + 1, so the rows run two macroblocks apart and as
 * many of them are in flight as there are threads.
 *
 * ⚠️ The dependency really is x + 1 and not x. Intra prediction reads the
 * macroblock above right, so column x of row r needs column x + 1 of row
 * r - 1 finished, not merely column x. Getting that wrong gives a picture
 * that is right almost everywhere and wrong along some diagonals, which is
 * exactly the kind of bug that survives a casual look.
 *
 * Each worker takes a copy of the decoder and uses it as a cursor of its
 * own. Reconstruction reads the decoder and never writes to it, so the
 * copies share the macroblock array, the frame store and the residuals,
 * and differ only in where they are.
 */
#include "h264_dec_internal.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

/* A worker and which of the per-worker buffers are its own. */
struct worker_arg { struct h264d_pool *p; int io; };
typedef struct worker_arg worker_arg_t;

struct h264d_pool {
    pthread_t *thread;
    struct worker_arg *arg;
    int n;

    pthread_mutex_t m;
    pthread_cond_t via, finished;

    /* The job being run, if any. */
    int kind;                   /* 0 reconstruct, 1 deblock, 2 slices */
    h264_decoder_t *d;
    h264d_deblock_pic_t dbl;
    int first, count, number;
    int first_row, last_row;

    /* Slice jobs. */
    const h264d_slice_input_t *in;
    int n_in, first_number;
    atomic_int next_slice;
    atomic_int error;

    /* Per worker, for slice jobs only: somewhere to un-escape a NAL into
     * and a residual ring. Allocated the first time they are needed. */
    uint8_t **rbsp;
    size_t *rbsp_cap;
    h264d_residual_t **residuals;
    int n_residuals;

    unsigned generation;       /* bumped once per job */
    int active;                 /* workers still inside this job */
    bool close_picture;

    atomic_int next_row;   /* next row to claim */
    atomic_int *progress_of;      /* per row: the first column not yet done */
    int n_rows;
};

/* How far a row has to have got before the row below may touch column x. */
static inline void await_row(const atomic_int *p, int up_to)
{
    int rounds = 0;
    while (atomic_load_explicit(p, memory_order_acquire) < up_to) {
        /* A short spin, because the row above is usually only a macroblock
         * or two ahead and about to move. Then out of the way, because a
         * thread that spins through its slice stops the one it is waiting
         * for from being scheduled at all. */
        if (++rounds < 256) {
#if defined(__x86_64__)
            __builtin_ia32_pause();
#endif
        } else {
            sched_yield();
            rounds = 0;
        }
    }
}

/* A slice job: take slices until there are none left. */
static void work_slice(struct h264d_pool *p, int io)
{
    h264_decoder_t c = *p->d;
    /* âš ï¸ No pool on the cursor: the workers are the slices, so this one
     * reconstructs what it reads, itself, as it goes. */
    c.pool = NULL;
    c.rbsp = p->rbsp[io];
    c.rbsp_cap = p->rbsp_cap[io];
    c.residuals = p->residuals[io];
    c.n_residuals = p->n_residuals;
    c.band_rows = 1;

    for (;;) {
        const int i = atomic_fetch_add(&p->next_slice, 1);
        if (i >= p->n_in) break;
        const int r = h264d_decode_slice(&c, &p->in[i], p->first_number + i);
        if (r) {
            int expected = 0;
            atomic_compare_exchange_strong(&p->error, &expected, r);
        }
    }
}

static void work(struct h264d_pool *p)
{
    const int mb_w = (p->kind == 0) ? p->d->mb_w : p->dbl.mb_w;
    const int last = p->first + p->count - 1;
    h264_decoder_t c;
    if (p->kind == 0) {
        c = *p->d;
        c.slice = p->d->slices[p->number];
    }

    for (;;) {
        const int r = atomic_fetch_add(&p->next_row, 1);
        if (r > p->last_row) break;

        const int x0 = (r == p->first_row) ? p->first % mb_w : 0;
        const int x1 = (r == p->last_row) ? last % mb_w : mb_w - 1;
        const bool wait_for = (r > p->first_row);

        for (int x = x0; x <= x1; x++) {
            if (wait_for)
                await_row(&p->progress_of[r - 1], x + 2);

            if (p->kind == 0) {
                c.mb_idx = r * mb_w + x;
                c.mb_x = x;
                c.mb_y = r;
                c.res = &p->d->residuals[c.mb_idx % p->d->n_residuals];
                h264d_reconstruct_mb(&c);
            } else {
                h264d_deblock_mb(&p->dbl, x, r);
            }

            atomic_store_explicit(&p->progress_of[r], x + 1,
                                  memory_order_release);
        }
        /* The row is finished: let anything still waiting on its tail
         * through, including the columns it never had. */
        atomic_store_explicit(&p->progress_of[r], mb_w + 2,
                              memory_order_release);
    }
}

static void *worker(void *arg)
{
    const worker_arg_t *a = arg;
    struct h264d_pool *p = a->p;
    const int io = a->io;
    unsigned mia = 0;
    for (;;) {
        pthread_mutex_lock(&p->m);
        while (p->generation == mia && !p->close_picture)
            pthread_cond_wait(&p->via, &p->m);
        if (p->close_picture) {
            pthread_mutex_unlock(&p->m);
            return NULL;
        }
        mia = p->generation;
        const int kind = p->kind;
        pthread_mutex_unlock(&p->m);

        if (kind == 2) work_slice(p, io);
        else           work(p);

        pthread_mutex_lock(&p->m);
        if (--p->active == 0)
            pthread_cond_signal(&p->finished);
        pthread_mutex_unlock(&p->m);
    }
}

/* How many threads to use. BC250_H264_THREADS overrides; 0 or 1 turns
 * threading off entirely and every picture is reconstructed in place. */
static int thread_count(void)
{
    const char *e = getenv("BC250_H264_THREADS");
    if (e) {
        const int n = atoi(e);
        return n < 0 ? 0 : n;
    }
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) n = 1;
    /* ⚠️ Beyond about eight the wavefront stops paying: the rows are two
     * macroblocks apart, so the ninth thread is nine rows behind the first
     * and spends most of its time waiting for the eighth. */
    if (n > 8) n = 8;
    return (int)n;
}

int h264d_pool_start(h264_decoder_t *d)
{
    if (d->pool) return 0;
    const int n = thread_count();
    if (n < 2) return 0;                 /* one thread: no pool at all */

    struct h264d_pool *p = calloc(1, sizeof(*p));
    if (!p) return -1;
    p->n_rows = d->mb_h;
    p->progress_of = calloc((size_t)p->n_rows, sizeof(atomic_int));
    p->thread = calloc((size_t)n, sizeof(pthread_t));
    if (!p->progress_of || !p->thread) {
        free(p->progress_of); free(p->thread); free(p);
        return -1;
    }
    pthread_mutex_init(&p->m, NULL);
    pthread_cond_init(&p->via, NULL);
    pthread_cond_init(&p->finished, NULL);
    p->generation = 0;

    p->arg = calloc((size_t)n, sizeof(worker_arg_t));
    if (!p->arg) {
        free(p->progress_of); free(p->thread); free(p);
        return -1;
    }
    for (int i = 0; i < n; i++) {
        p->arg[i].p = p;
        p->arg[i].io = i;
        if (pthread_create(&p->thread[i], NULL, worker, &p->arg[i]) != 0) {
            p->n = i;
            break;
        }
        p->n = i + 1;
    }
    if (p->n < 2) {
        /* Not enough threads came up to be worth the synchronisation. */
        pthread_mutex_lock(&p->m);
        p->close_picture = true;
        pthread_cond_broadcast(&p->via);
        pthread_mutex_unlock(&p->m);
        for (int i = 0; i < p->n; i++) pthread_join(p->thread[i], NULL);
        pthread_mutex_destroy(&p->m);
        pthread_cond_destroy(&p->via);
        pthread_cond_destroy(&p->finished);
        free(p->progress_of); free(p->thread); free(p);
        return 0;
    }
    d->pool = p;
    return 0;
}

void h264d_pool_stop(h264_decoder_t *d)
{
    struct h264d_pool *p = d->pool;
    if (!p) return;
    pthread_mutex_lock(&p->m);
    p->close_picture = true;
    pthread_cond_broadcast(&p->via);
    pthread_mutex_unlock(&p->m);
    for (int i = 0; i < p->n; i++)
        pthread_join(p->thread[i], NULL);
    pthread_mutex_destroy(&p->m);
    pthread_cond_destroy(&p->via);
    pthread_cond_destroy(&p->finished);
    for (int i = 0; i < p->n; i++) {
        if (p->rbsp) free(p->rbsp[i]);
        if (p->residuals) free(p->residuals[i]);
    }
    free(p->rbsp);
    free(p->rbsp_cap);
    free(p->residuals);
    free(p->arg);
    free(p->progress_of);
    free(p->thread);
    free(p);
    d->pool = NULL;
}

/* One macroblock at a time, in order, on this thread. */
static void enqueue(h264_decoder_t *d, int first, int count, int number)
{
    h264_decoder_t c = *d;
    c.slice = d->slices[number];
    for (int i = 0; i < count; i++) {
        c.mb_idx = first + i;
        if (c.mb_idx >= d->mb_count) break;
        c.mb_x = c.mb_idx % d->mb_w;
        c.mb_y = c.mb_idx / d->mb_w;
        c.res = &d->residuals[c.mb_idx % d->n_residuals];
        h264d_reconstruct_mb(&c);
    }
}

void h264d_reconstruct_range(h264_decoder_t *d, int first, int count,
                             int number)
{
    if (count <= 0) return;
    if (first + count > d->mb_count)
        count = d->mb_count - first;

    struct h264d_pool *p = d->pool;
    const int first_row = first / d->mb_w;
    const int last_row = (first + count - 1) / d->mb_w;

    /* ⚠️ Below a few rows the threads cost more than they save: waking
     * them, the wavefront's two-macroblock lag and the join at the end all
     * have to be paid before the first sample is any faster. */
    if (!p || last_row - first_row < 3) {
        enqueue(d, first, count, number);
        return;
    }

    pthread_mutex_lock(&p->m);
    p->kind = 0;
    p->d = d;
    p->first = first;
    p->count = count;
    p->number = number;
    p->first_row = first_row;
    p->last_row = last_row;
    atomic_store(&p->next_row, first_row);
    for (int r = first_row; r <= last_row; r++) {
        const int x0 = (r == first_row) ? first % d->mb_w : 0;
        atomic_store(&p->progress_of[r], x0);
    }
    p->active = p->n;
    p->generation++;
    pthread_cond_broadcast(&p->via);
    while (p->active > 0)
        pthread_cond_wait(&p->finished, &p->m);
    pthread_mutex_unlock(&p->m);
}

/* The deblocking filter over a whole picture, as the same wavefront. */
void h264d_deblock_wavefront(h264_decoder_t *d, const h264d_deblock_pic_t *dp)
{
    struct h264d_pool *p = d->pool;
    if (!p || dp->mb_h < 4) {
        for (int my = 0; my < dp->mb_h; my++)
            for (int mx = 0; mx < dp->mb_w; mx++)
                h264d_deblock_mb(dp, mx, my);
        return;
    }

    pthread_mutex_lock(&p->m);
    p->kind = 1;
    p->d = d;
    p->dbl = *dp;
    p->first = 0;
    p->count = dp->mb_w * dp->mb_h;
    p->number = 0;
    p->first_row = 0;
    p->last_row = dp->mb_h - 1;
    atomic_store(&p->next_row, 0);
    for (int r = 0; r <= p->last_row; r++)
        atomic_store(&p->progress_of[r], 0);
    p->active = p->n;
    p->generation++;
    pthread_cond_broadcast(&p->via);
    while (p->active > 0)
        pthread_cond_wait(&p->finished, &p->m);
    pthread_mutex_unlock(&p->m);
}

/* Make sure every worker has a NAL buffer big enough and a residual ring.
 * Done the first time slice parallelism is used, and then only grown. */
static bool prepare_buffer(struct h264d_pool *p, h264_decoder_t *d,
                           size_t serve)
{
    if (!p->rbsp) {
        p->rbsp = calloc((size_t)p->n, sizeof(uint8_t *));
        p->rbsp_cap = calloc((size_t)p->n, sizeof(size_t));
        p->residuals = calloc((size_t)p->n, sizeof(h264d_residual_t *));
        if (!p->rbsp || !p->rbsp_cap || !p->residuals) return false;
    }
    if (!p->n_residuals) {
        p->n_residuals = 2 * d->mb_w;
        for (int i = 0; i < p->n; i++) {
            p->residuals[i] = calloc((size_t)p->n_residuals,
                                   sizeof(h264d_residual_t));
            if (!p->residuals[i]) return false;
        }
    }
    for (int i = 0; i < p->n; i++) {
        if (p->rbsp_cap[i] >= serve) continue;
        uint8_t *new_one = realloc(p->rbsp[i], serve);
        if (!new_one) return false;
        p->rbsp[i] = new_one;
        p->rbsp_cap[i] = serve;
    }
    return true;
}

int h264d_slices_pool(h264_decoder_t *d, const h264d_slice_input_t *in,
                      int n, int first_number)
{
    struct h264d_pool *p = d->pool;
    size_t bigger = 0;
    for (int i = 0; i < n; i++)
        if (in[i].size > bigger) bigger = in[i].size;

    pthread_mutex_lock(&p->m);
    if (!prepare_buffer(p, d, bigger + 64)) {
        pthread_mutex_unlock(&p->m);
        /* Out of memory for the parallel path: the serial one needs none. */
        int first_error = 0;
        for (int i = 0; i < n; i++) {
            const int r = h264d_decode_slice(d, &in[i], first_number + i);
            if (r && !first_error) first_error = r;
        }
        return first_error;
    }

    p->kind = 2;
    p->d = d;
    p->in = in;
    p->n_in = n;
    p->first_number = first_number;
    atomic_store(&p->next_slice, 0);
    atomic_store(&p->error, 0);
    p->active = p->n;
    p->generation++;
    pthread_cond_broadcast(&p->via);
    while (p->active > 0)
        pthread_cond_wait(&p->finished, &p->m);
    pthread_mutex_unlock(&p->m);
    return atomic_load(&p->error);
}
