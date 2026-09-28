/* Copyright 2026. Pulserver contributors.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef _DIMTYPES_H
#define _DIMTYPES_H

#include <stdint.h>
#include <inttypes.h>

/*
 * The integer types of the multidimensional array machinery.
 *
 * They are fixed-width so that an array is limited by memory rather
 * than by the width of the platform's `long`, which is 32 bits on
 * LLP64 systems (64-bit Windows).  Format with PRId64 / PRIu64 and
 * PRIx64; write constants with INT64_C / UINT64_C.
 */

/* Extents of dimensions, positions along them, and element counts. */
typedef int64_t bart_dim_t;

/* Strides and offsets, in bytes or elements, and sizes in bytes.
 * Signed: a stride may be negative. */
typedef int64_t bart_stride_t;

/* Sets of dimensions: bit i selects dimension i. */
typedef uint64_t bart_flags_t;

#define BART_FLAGS_BITS 64

#ifndef __cplusplus
_Static_assert(8 == sizeof(bart_dim_t), "bart_dim_t must be 64 bits");
_Static_assert(8 == sizeof(bart_stride_t), "bart_stride_t must be 64 bits");
_Static_assert(8 == sizeof(bart_flags_t), "bart_flags_t must be 64 bits");
#else
static_assert(8 == sizeof(bart_dim_t), "bart_dim_t must be 64 bits");
static_assert(8 == sizeof(bart_stride_t), "bart_stride_t must be 64 bits");
static_assert(8 == sizeof(bart_flags_t), "bart_flags_t must be 64 bits");
#endif

#endif	// _DIMTYPES_H
