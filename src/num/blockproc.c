/* Copyright 2015. The Regents of the University of California.
 * Copyright 2024. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2014 Frank Ong
 */

#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>

#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "blockproc.h"


float lineproc2( int D,  const bart_dim_t dims[D], const bart_dim_t blkdims[D], const bart_dim_t line_dims[D], const void* data,
		 float (*op)(const void* data, const bart_dim_t blkdims[D], complex float* dst, const complex float* src), 
		 const bart_stride_t ostrs[D], complex float* dst, const bart_stride_t istrs[D], const complex float* src)
{
	// Get number of blocks per dimension
	bart_dim_t nblocks[D];
	bart_dim_t shifts[D];

	for (int i = 0; i < D; i++) {

		nblocks[i] = dims[i] - blkdims[i] + 1;
		shifts[i] = (dims[i] - nblocks[i] * line_dims[i]) / 2;
	}

	bart_stride_t line_strs[D];
	md_calc_strides(D, line_strs, line_dims, CFL_SIZE);

	bart_dim_t numblocks = md_calc_size(D, nblocks);
	float info = 0;

	// Loop over blocks
	complex float* blk = md_alloc_sameplace(D, blkdims, CFL_SIZE, src);
	complex float* line = md_alloc_sameplace(D, line_dims, CFL_SIZE, src);

	for (bart_dim_t b = 0; b < numblocks; b++) {

		// Get block position and actual block size
		bart_dim_t blkpos[D];
		bart_dim_t linepos[D];

		bart_dim_t ind = b;
		for (int i = 0; i < D; i++) {

			bart_dim_t blkind = ind % nblocks[i];
			blkpos[i] = blkind;
			linepos[i] = blkind + shifts[i];
			ind = (ind - blkind) / nblocks[i];
		}

		bart_stride_t blkstrs[D];
		md_calc_strides(D, blkstrs, blkdims, CFL_SIZE);

		md_copy_block2(D, blkpos, blkdims, blkstrs, blk, dims, istrs, src, CFL_SIZE);

		// Process block
		info += (*op)(data, blkdims, line, blk);

		md_copy_block2(D, linepos, dims, ostrs, dst, line_dims, line_strs, line, CFL_SIZE);
	}

	md_free(blk);
	md_free(line);

	return info;
}


float lineproc(  int D, const bart_dim_t dims[D], const bart_dim_t blkdims[D], const bart_dim_t line_dims[D], const void* data,
		 float (*op)(const void* data, const bart_dim_t blkdims[D], complex float* dst, const complex float* src), 
		 complex float* dst, const complex float* src)
{
	bart_stride_t strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	return lineproc2(D, dims, blkdims, line_dims, data, op, strs, dst, strs, src);
}


float blockproc_shift_mult2(int D, const bart_dim_t dims[D], const bart_dim_t blkdims[D], const bart_dim_t shifts[D], const bart_dim_t mult[D], const void* data,
			float (*op)(const void* data, const bart_dim_t blkdims[D], complex float* dst, const complex float* src), 
			const bart_stride_t ostrs[D], complex float* dst, const bart_stride_t istrs[D], const complex float* src)
{
	float info = 0;
	bart_dim_t pos[D];

	for (int i = 0; i < D; i++) {	

		pos[i] = shifts[i];

		while (pos[i] < 0)
			pos[i] += dims[i];
	}

	int i = 0;

	while ((i < D) && (0 == pos[i]))
		i++;

	if (D == i) {

		info += blockproc2( D, dims, blkdims, data, op, ostrs, dst, istrs, src );
		return info;
	}

	bart_dim_t shift = pos[i];

	assert(shift != 0);

	bart_dim_t dim0[D];
	bart_dim_t dim1[D];
	bart_dim_t dim2[D];
	bart_dim_t dim3[D];

	md_copy_dims(D, dim0, dims);
	md_copy_dims(D, dim1, dims);
	md_copy_dims(D, dim2, dims);
	md_copy_dims(D, dim3, dims);

	dim0[i] = shift - (shift/mult[i]) * mult[i];
	dim1[i] = (shift/mult[i]) * mult[i];
	dim2[i] = ((dims[i] - shift) / mult[i]) * mult[i];
	dim3[i] = dims[i] - (((dims[i] - shift) / mult[i]) * mult[i]) - shift;

	bart_stride_t off0 = 0;
	bart_stride_t off1 = off0 + dim0[i] * ostrs[i] / (bart_stride_t)CFL_SIZE;
	bart_stride_t off2 = off1 + dim1[i] * ostrs[i] / (bart_stride_t)CFL_SIZE;
	bart_stride_t off3 = off2 + dim2[i] * ostrs[i] / (bart_stride_t)CFL_SIZE;

	pos[i] = 0;

	info += blockproc_shift_mult2(D, dim0, blkdims, pos, mult, data, op, ostrs, dst + off0, istrs, src + off0);
	info += blockproc_shift_mult2(D, dim1, blkdims, pos, mult, data, op, ostrs, dst + off1, istrs, src + off1);
	info += blockproc_shift_mult2(D, dim2, blkdims, pos, mult, data, op, ostrs, dst + off2, istrs, src + off2);
	info += blockproc_shift_mult2(D, dim3, blkdims, pos, mult, data, op, ostrs, dst + off3, istrs, src + off3);

	return info;
}


float blockproc_shift_mult(int D, const bart_dim_t dims[D], const bart_dim_t blkdims[D], const bart_dim_t shifts[D], const bart_dim_t mult[D], const void* data,
		 float (*op)(const void* data, const bart_dim_t blkdims[D], complex float* dst, const complex float* src), 
		 complex float* dst, const complex float* src)
{
	bart_stride_t strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	return blockproc_shift_mult2( D, dims, blkdims, shifts, mult, data, op, strs, dst, strs, src );
}



float blockproc_shift2(int D, const bart_dim_t dims[D], const bart_dim_t blkdims[D], const bart_dim_t shifts[D], const void* data,
			float (*op)(const void* data, const bart_dim_t blkdims[D], complex float* dst, const complex float* src), 
			const bart_stride_t ostrs[D], complex float* dst, const bart_stride_t istrs[D], const complex float* src)
{
	float info = 0;
	bart_dim_t pos[D];

	for (int i = 0; i < D; i++) {

		pos[i] = shifts[i];

		while (pos[i] < 0)
			pos[i] += dims[i];
	}

	int i = 0;

	while ((i < D) && (0 == pos[i]))
		i++;

	if (D == i) {

		info += blockproc2(D, dims, blkdims, data, op, ostrs, dst, istrs, src);

		return info;
	}

	bart_dim_t shift = pos[i];

	assert(shift != 0);

	bart_dim_t dim1[D];
	bart_dim_t dim2[D];

	md_copy_dims(D, dim1, dims);
	md_copy_dims(D, dim2, dims);

	dim1[i] = shift;
	dim2[i] = dims[i] - shift;

	pos[i] = 0;

	info += blockproc_shift2(D, dim1, blkdims, pos, data, op, ostrs, dst, istrs, src);
	info += blockproc_shift2(D, dim2, blkdims, pos, data, op, ostrs,
			dst + dim1[i] * ostrs[i] / (bart_stride_t)CFL_SIZE, istrs, src + dim1[i] * istrs[i] / (bart_stride_t)CFL_SIZE);

	return info;
}


float blockproc_shift(int D,  const bart_dim_t dims[D], const bart_dim_t blkdims[D], const bart_dim_t shifts[D], const void* data,
		 float (*op)(const void* data, const bart_dim_t blkdims[D], complex float* dst, const complex float* src), 
		 complex float* dst, const complex float* src)
{
	bart_stride_t strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	return blockproc_shift2(D, dims, blkdims, shifts, data, op, strs, dst, strs, src);
}


float blockproc_circshift(int D,  const bart_dim_t dims[D], const bart_dim_t blkdims[D], const bart_dim_t shifts[D], const void* data,
		 float (*op)(const void* data, const bart_dim_t blkdims[D], complex float* dst, const complex float* src), 
		 complex float* dst, const complex float* src)
{
	complex float* tmp = md_alloc( D, dims, CFL_SIZE );
	
	bart_dim_t unshifts[D];
	for (int i = 0; i < D; i++)
		unshifts[i] = -shifts[i];

	md_circ_shift(D, dims, shifts, tmp, src, CFL_SIZE);

	float info = blockproc(D, dims, blkdims, data, op, tmp, tmp);

	md_circ_shift(D, dims, unshifts, dst, tmp, CFL_SIZE);

	md_free(tmp);

	return info;
}


float blockproc2(int D,  const bart_dim_t dims[D], const bart_dim_t blkdims[D], const void* data,
		 float (*op)(const void* data, const bart_dim_t blkdims[D], complex float* dst, const complex float* src), 
		 const bart_stride_t ostrs[D], complex float* dst, const bart_stride_t istrs[D], const complex float* src)
{
	// Get number of blocks per dimension
	bart_dim_t nblocks[D];
	for (int i = 0; i < D; i++)
		nblocks[i] = (float)(dims[i] + blkdims[i] - 1) / (float)blkdims[i];

	bart_dim_t numblocks = md_calc_size(D, nblocks);
	float info = 0;

	// Loop over blocks
	complex float* blk = md_alloc_sameplace(D, blkdims, CFL_SIZE, src);

	for (bart_dim_t b = 0; b < numblocks; b++) {

		// Get block position and actual block size
		bart_dim_t blkpos[D];
		bart_dim_t blkdims_b[D]; // actual block size

		bart_dim_t ind = b;
		for (int i = 0; i < D; i++) {

			bart_dim_t blkind = ind % nblocks[i];
			blkpos[i] = blkind * blkdims[i];
			ind = (ind - blkind) / nblocks[i];

			blkdims_b[i] = MIN(dims[i] - blkpos[i], blkdims[i]);
		}

		bart_stride_t blkstrs[D];
		md_calc_strides(D, blkstrs, blkdims_b, CFL_SIZE);

		md_copy_block2(D, blkpos, blkdims_b, blkstrs, blk, dims, istrs, src, CFL_SIZE);

		// Process block
		info += (*op)(data, blkdims_b, blk, blk);

		md_copy_block2( D, blkpos, dims, ostrs, dst, blkdims_b, blkstrs, blk, CFL_SIZE);
	}

	md_free(blk);
	return info;
}


float blockproc( int D, const bart_dim_t dims[D], const bart_dim_t blkdims[D], const void* data,
		 float (*op)(const void* data, const bart_dim_t blkdims[D], complex float* dst, const complex float* src), 
		 complex float* dst, const complex float* src )
{
	bart_stride_t strs[D];
	md_calc_strides( D, strs, dims, CFL_SIZE );

	return blockproc2( D, dims, blkdims, data, op, strs, dst, strs, src );
}


float stackproc2(int D, const bart_dim_t dims[D], const bart_dim_t blkdims[D], int stkdim, const void* data,
		float (*op)(const void* data, const bart_dim_t stkdims[D], complex float* dst, const complex float* src), 
		 const bart_stride_t ostrs[D], complex float* dst, const bart_stride_t istrs[D], const complex float* src)
{
	// Get number of blocks per dimension
	bart_dim_t nblocks[D];
	for (int i = 0; i < D; i++)
		nblocks[i] = (float)(dims[i] + blkdims[i] - 1) / (float)blkdims[i];

	bart_dim_t numblocks = md_calc_size(D, nblocks);
	float info = 0;

	// Initialize stack
	bart_dim_t stkdims[D];
	md_copy_dims(D, stkdims, blkdims);
	stkdims[stkdim] = numblocks;

	bart_stride_t stkstrs[D];
	md_calc_strides(D, stkstrs, stkdims, CFL_SIZE);

	bart_stride_t stkstr1[D];
	md_calc_strides(D, stkstr1, stkdims, 1);

	complex float* stk = md_alloc(D, stkdims, CFL_SIZE);
	md_clear(D, stkdims, stk, CFL_SIZE);

	// Loop over blocks and stack them up
	for (bart_dim_t b = 0; b < numblocks; b++)
	{
		// Get block position and actual block size
		bart_dim_t blkpos[D];
		bart_dim_t blkdims_b[D]; // actual block size
		bart_dim_t ind = b;

		for (int i = 0; i < D; i++) {

			bart_dim_t blkind = ind % nblocks[i];
			blkpos[i] = blkind * blkdims[i];
			ind = (ind - blkind) / nblocks[i];

			blkdims_b[i] = MIN(dims[i] - blkpos[i], blkdims[i]);
		}

		bart_stride_t blkstrs[D];
		md_calc_strides(D, blkstrs, blkdims_b, CFL_SIZE);

		md_copy_block2(D, blkpos, blkdims_b, blkstrs, stk + stkstr1[stkdim] * b, dims, istrs, src, CFL_SIZE);
	}

	bart_stride_t blkstrs[D];
	md_calc_strides(D, blkstrs, blkdims, CFL_SIZE);

	// Process block
	info = (*op)(data, stkdims, stk, stk);

	// Put back block
	for (bart_dim_t b = 0; b < numblocks; b++) {

		// Get block position and actual block size
		bart_dim_t blkpos[D];
		bart_dim_t blkdims_b[D]; // actual block size
		bart_dim_t ind = b;

		for (int i = 0; i < D; i++) {

			bart_dim_t blkind = ind % nblocks[i];
			blkpos[i] = blkind * blkdims[i];
			ind = (ind - blkind) / nblocks[i];

			blkdims_b[i] = MIN(dims[i] - blkpos[i], blkdims[i]);
		}

		bart_stride_t blkstrs[D];
		md_calc_strides(D, blkstrs, blkdims_b, CFL_SIZE);

		md_copy_block2(D, blkpos, dims, ostrs, dst, blkdims_b, blkstrs, stk + stkstr1[stkdim] * b, CFL_SIZE);
	}

	md_free(stk);

	return info;
}


float stackproc(int D, const bart_dim_t dims[D], const bart_dim_t blkdims[D], int stkdim, const void* data,
		float (*op)(const void* data, const bart_dim_t stkdims[D], complex float* dst, const complex float* src), 
		complex float* dst, const complex float* src)
{
	bart_stride_t strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	return stackproc2(D, dims, blkdims, stkdim, data, op, strs, dst, strs, src);
}

