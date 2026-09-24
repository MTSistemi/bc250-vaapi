/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * decoder_h265.h - the H.265 decoder as something other code can hold.
 *
 * Everything below the slice header lives in hevc_cu.c and its neighbours
 * and knows nothing about where pictures come from or go. This is the
 * piece that does: it keeps the decoded pictures a stream still points at,
 * builds each slice's reference lists, and hands a finished picture
 * either to a caller that wants the planes or to the GPU.
 */
#ifndef BC250_DECODER_H265_H
#define BC250_DECODER_H265_H

#include "hevc_ps.h"

#include <stddef.h>
#include <stdint.h>

#include "gpu_compute.h"

typedef struct hevc_decoder hevc_decoder_t;

/* How many pictures one decoder can have in flight at once. */
#define HEVC_DECODER_FRAMES 2

/* `gpu` may be NULL: without it the decoder keeps the planes and a caller
 * reads them with hevc_decoder_plane(). */
hevc_decoder_t *hevc_decoder_create(void *gpu, int width, int height);
void hevc_decoder_destroy(hevc_decoder_t *d);

/* Which pictures a later one may still point at, named by whatever the
 * caller uses to name them: a surface, or a counter.
 *
 * ⚠️ Called before begin_picture, so that the picture about to be decoded
 * cannot be handed the slot of one it is about to read. */
void hevc_decoder_set_references(hevc_decoder_t *d, const uintptr_t *id,
                                 const int *poc, int n);

/* Find a reference image in the decoder DPB matching surface id and/or poc. */
const void *hevc_decoder_find_ref(const hevc_decoder_t *d, uintptr_t id, int poc);

/* Find the closest valid image in the decoder DPB by POC (for frame drop concealment). */
const void *hevc_decoder_find_closest(const hevc_decoder_t *d, int poc);

/* A new picture. Returns 0, or non-zero when the buffer is full. */
int hevc_decoder_begin_picture(hevc_decoder_t *d, const hevc_sps_t *sps,
                               const hevc_pps_t *pps, uintptr_t id, int poc);

/* One slice, from the payload with its emulation prevention bytes already
 * removed. Returns 0, or a reason hevc_decoder_reason() can name. */
int hevc_decoder_slice(hevc_decoder_t *d, const hevc_slice_t *sl,
                       const uint8_t *rbsp, size_t n);

/* Let go of everything the reference picture set no longer names. */
void hevc_decoder_unescape(hevc_decoder_t *d, const hevc_slice_t *sl);

/* Whether the picture a caller named `id` is still held as a reference.
 * The output process needs it: a picture that has been output but is
 * still referenced still takes a place in the buffer. */
bool hevc_decoder_holds(const hevc_decoder_t *d, uintptr_t id);

/* The loop filters, which are defined over the whole picture and so can
 * only run once every slice of it is in. */
void hevc_decoder_end_picture(hevc_decoder_t *d);

/* Where the picture just finished is, when there is no GPU to hand it to. */
const uint8_t *hevc_decoder_plane(const hevc_decoder_t *d, int plane,
                                  int *stride);

/* The picture just finished, onto the surface, as NV12. Does nothing
 * when the decoder was made without a GPU: the caller keeps the planes. */
int hevc_decoder_load(hevc_decoder_t *d, gpu_image_t out, gpu_memory_t mem);

const char *hevc_decoder_reason(int e);

/* The same, for a caller decoding more than one picture at a time - the
 * functions above are frame 0 of these.
 *
 * A picture is begun, sliced, ended and loaded on frame k, 0 <= k <
 * HEVC_DECODER_FRAMES; two frames may be in any of those at the same time,
 * on different threads. ⚠️ Pictures are BEGUN in decoding order, one
 * after the other, each after hevc_decoder_set_references() for it -
 * nothing else of two pictures needs ordering. `refs` are the reference
 * pictures the caller resolved (hevc_decoder_find_ref()) before beginning,
 * and are held, like the picture itself, until frame k begins its next:
 * a later picture's reference set cannot free them while they are read.
 * A picture reading one still being decoded waits for the rows it needs. */
int hevc_decoder_begin_frame(hevc_decoder_t *d, int k, const hevc_sps_t *sps,
                             const hevc_pps_t *pps, uintptr_t id, int poc,
                             const void *const *refs, int n_refs);
int hevc_decoder_frame_slice(hevc_decoder_t *d, int k, const hevc_slice_t *sl,
                             const uint8_t *rbsp, size_t n);
void hevc_decoder_end_frame(hevc_decoder_t *d, int k);
int hevc_decoder_load_frame(hevc_decoder_t *d, int k, gpu_image_t out,
                            gpu_memory_t mem);

/* 7.4.7.1: the entry point offsets count the NAL unit's bytes, emulation
 * prevention included, and everything downstream reads the payload with
 * those removed. */
void hevc_decoder_shift_entry_points(hevc_slice_t *s, const uint8_t *grezzo,
                                     size_t n_grezzo, size_t first);

#endif /* BC250_DECODER_H265_H */
