/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef _BLOCKPROC_H
#define _BLOCKPROC_H

#include "misc/cppwrap.h"

extern float lineproc2(int D, const bart_dim_t dims[__VLA(D)], const bart_dim_t blkdims[__VLA(D)], const bart_dim_t line_dims[__VLA(D)], const void * data,
			 float (*op)(const void* data, const bart_dim_t blkdims[__VLA(D)], complex float* dst, const complex float* src),
			 const bart_stride_t ostrs[__VLA(D)], complex float* dst, const bart_stride_t istrs[__VLA(D)], const complex float* src);

extern float lineproc(int D, const bart_dim_t dims[__VLA(D)], const bart_dim_t blkdims[__VLA(D)], const bart_dim_t line_dims[__VLA(D)], const void * data,
			float (*op)(const void* data, const bart_dim_t blkdims[__VLA(D)], complex float* dst, const complex float* src),
			complex float* dst, const complex float* src);

extern float blockproc2(int D, const bart_dim_t dims[__VLA(D)], const bart_dim_t blkdims[__VLA(D)], const void * data,
			 float (*op)(const void* data, const bart_dim_t blkdims[__VLA(D)], complex float* dst, const complex float* src),
			 const bart_stride_t ostrs[__VLA(D)], complex float* dst, const bart_stride_t istrs[__VLA(D)], const complex float* src);

extern float blockproc(int D, const bart_dim_t dims[__VLA(D)], const bart_dim_t blkdims[__VLA(D)], const void * data,
			float (*op)(const void* data, const bart_dim_t blkdims[__VLA(D)], complex float* dst, const complex float* src),
			complex float* dst, const complex float* src);

extern float blockproc_shift2(int D, const bart_dim_t dims[__VLA(D)], const bart_dim_t blkdims[__VLA(D)], const bart_dim_t shifts[__VLA(D)], const void* data,
			       float (*op)(const void* data, const bart_dim_t blkdims[__VLA(D)], complex float* dst, const complex float* src), 
			       const bart_stride_t ostrs[__VLA(D)], complex float* dst, const bart_stride_t istrs[__VLA(D)], const complex float* src);

extern float blockproc_shift(int D, const bart_dim_t dims[__VLA(D)], const bart_dim_t blkdims[__VLA(D)], const bart_dim_t shifts[__VLA(D)], const void* data,
			      float (*op)(const void* data, const bart_dim_t blkdims[__VLA(D)], complex float* dst, const complex float* src), 
			      complex float* dst, const complex float* src);

extern float blockproc_circshift(int D, const bart_dim_t dims[__VLA(D)], const bart_dim_t blkdims[__VLA(D)], const bart_dim_t shifts[__VLA(D)], const void* data,
			      float (*op)(const void* data, const bart_dim_t blkdims[__VLA(D)], complex float* dst, const complex float* src), 
			      complex float* dst, const complex float* src);

extern float blockproc_shift_mult2(int D, const bart_dim_t dims[__VLA(D)], const bart_dim_t blkdims[__VLA(D)], const bart_dim_t shifts[__VLA(D)], const bart_dim_t mult[__VLA(D)], const void* data,
			       float (*op)(const void* data, const bart_dim_t blkdims[__VLA(D)], complex float* dst, const complex float* src),
			       const bart_stride_t ostrs[__VLA(D)], complex float* dst, const bart_stride_t istrs[__VLA(D)], const complex float* src);

extern float blockproc_shift_mult(int D, const bart_dim_t dims[__VLA(D)], const bart_dim_t blkdims[__VLA(D)], const bart_dim_t shifts[__VLA(D)], const bart_dim_t mult[__VLA(D)], const void* data,
			      float (*op)(const void* data, const bart_dim_t blkdims[__VLA(D)], complex float* dst, const complex float* src),
			      complex float* dst, const complex float* src);

extern float stackproc2(int D, const bart_dim_t dims[__VLA(D)], const bart_dim_t blkdims[__VLA(D)], int stkdim, const void* data,
			float (*op)(const void* data, const bart_dim_t stkdims[__VLA(D)], complex float* dst, const complex float* src),
			const bart_stride_t ostrs[__VLA(D)], complex float* dst, const bart_stride_t istrs[__VLA(D)], const complex float* src);

extern float stackproc(int D, const bart_dim_t dims[__VLA(D)], const bart_dim_t blkdims[__VLA(D)], int stkdim, const void* data,
			float (*op)(const void* data, const bart_dim_t stkdims[__VLA(D)], complex float* dst, const complex float* src),
			complex float* dst, const complex float* src);

#include "misc/cppwrap.h"

#endif


