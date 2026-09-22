/*
 * Copyright (c) 2026 BC-250 Project
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * hevc_pixel.h - what a sample is, for one compilation of one file.
 *
 * A ten-bit sample does not fit in a byte, so the files that touch
 * samples are compiled twice: once with a sample as uint8_t and once as
 * uint16_t, and the two sets of functions carry the depth in their names.
 *
 * ⚠️ The alternative was one path with sixteen-bit samples throughout,
 * which is shorter to write and halves the throughput of eight-bit video
 * - twice the memory traffic for a picture where the top eight bits of
 * every sample are zero. Eight bit is what the board actually decodes, so
 * it keeps its own path and its vector code.
 *
 * ⚠️ A stride is counted in SAMPLES, never in bytes. They are the same
 * number at eight bits, which is exactly why getting it wrong there would
 * stay invisible until the day it is not.
 */
#ifndef BIT_DEPTH
#error "hevc_pixel.h: define BIT_DEPTH before including it"
#endif

#undef pixel
#undef PIXEL_MAX
#undef FUNC
#undef FUNC_JOIN
#undef FUNC_JOIN_

#if BIT_DEPTH > 8
#define pixel uint16_t
#else
#define pixel uint8_t
#endif

#define PIXEL_MAX ((1 << BIT_DEPTH) - 1)

/* Two levels, so BIT_DEPTH is expanded before it is pasted. */
#define FUNC_JOIN_(a, b) a ## _ ## b
#define FUNC_JOIN(a, b) FUNC_JOIN_(a, b)
#define FUNC(a) FUNC_JOIN(a, BIT_DEPTH)
