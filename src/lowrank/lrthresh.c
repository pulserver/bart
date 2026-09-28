/* Copyright 2015-2018. The Regents of the University of California.
 * Copyright 2015. Tao Zhang and Joseph Cheng.
 * Copyright 2016-2019. Martin Uecker.
 * Copyright 2024-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014-2015 Frank Ong
 * 2014 Tao Zhang
 * 2014 Joseph Cheng 
 * 2014 Jon Tamir 
 * 2014-2018 Martin Uecker
 */

#include <stdlib.h>
#include <complex.h>
#include <math.h>

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/linalg.h"
#include "num/ops.h"
#include "num/ops_p.h"
#include "num/blockproc.h"
#include "num/casorati.h"

#include "iter/thresh.h"

#include "lowrank/batchsvd.h"
#include "lowrank/svthresh.h"

#include "lrthresh.h"


struct lrthresh_data_s {

	operator_data_t super;

	float lambda;
	bool randshift;
	bool noise;
	int remove_mean; 

	bart_stride_t strs_lev[DIMS];
	bart_stride_t strs[DIMS];

	bart_dim_t dims_decom[DIMS];
	bart_dim_t dims[DIMS];

	bart_flags_t mflags;
	bart_flags_t flags;
	bart_dim_t levels;
	bart_dim_t blkdims[MAX_LEV][DIMS];

	bool overlapping_blocks;
};

static DEF_TYPEID(lrthresh_data_s);



static struct lrthresh_data_s* lrthresh_create_data(const bart_dim_t dims_decom[DIMS], bool randshift, bart_flags_t mflags, const bart_dim_t blkdims[MAX_LEV][DIMS], float lambda, bool noise, int remove_mean, bool overlapping_blocks);
static void lrthresh_free_data(const operator_data_t* data);
static void lrthresh_apply(const operator_data_t* _data, float lambda, complex float* dst, const complex float* src);



/**
 * Initialize lrthresh operator
 *
 * @param dims_decom - decomposition dimensions
 * @param randshift - randshift boolean
 * @param mflags - selects which dimensions gets reshaped as the first dimension in matrix
 * @param blkdims - contains block dimensions for all levels
 *
 */
const struct operator_p_s* lrthresh_create(const bart_dim_t dims_lev[DIMS], bool randshift, bart_flags_t mflags, const bart_dim_t blkdims[MAX_LEV][DIMS], float lambda, bool noise, int remove_mean, bool overlapping_blocks)
{
	struct lrthresh_data_s* data = lrthresh_create_data(dims_lev, randshift, mflags, blkdims, lambda, noise, remove_mean, overlapping_blocks);

	return operator_p_create(DIMS, dims_lev, DIMS, dims_lev, CAST_UP(data), lrthresh_apply, lrthresh_free_data);
}



/**
 * Initialize lrthresh data
 *
 * @param dims_decom - dimensions with levels at LEVEL_DIMS
 * @param randshift - randshift boolean
 * @param mflags - selects which dimensions gets reshaped as the first dimension in matrix
 * @param blkdims - contains block dimensions for all levels
 *
 */
static struct lrthresh_data_s* lrthresh_create_data(const bart_dim_t dims_decom[DIMS], bool randshift, bart_flags_t mflags, const bart_dim_t blkdims[MAX_LEV][DIMS], float lambda, bool noise, int remove_mean, bool overlapping_blocks)
{
	PTR_ALLOC(struct lrthresh_data_s, data);
	SET_TYPEID(lrthresh_data_s, data);

	data->randshift = randshift;
	data->mflags = mflags;
	data->lambda = lambda;
	data->noise = noise;
	data->remove_mean = remove_mean;

	data->overlapping_blocks = overlapping_blocks;

	// level dimensions
	md_copy_dims(DIMS, data->dims_decom, dims_decom);
	md_calc_strides(DIMS, data->strs_lev, dims_decom, CFL_SIZE);

	// image dimensions
	data->levels = dims_decom[LEVEL_DIM];
	md_select_dims(DIMS, ~LEVEL_FLAG, data->dims, dims_decom);
	md_calc_strides(DIMS, data->strs, data->dims, CFL_SIZE);

	// blkdims
	for(bart_dim_t l = 0; l < data->levels; l++)
		md_copy_dims(DIMS, data->blkdims[l], blkdims[l]);

	return PTR_PASS(data);
}



/**
 * Free lrthresh operator
 */
static void lrthresh_free_data(const operator_data_t* _data)
{
	xfree(CAST_DOWN(lrthresh_data_s, _data));
}



/*
 * Return a random number between 0 and limit inclusive.
 */
static int rand_lim(int limit)
{
	int divisor = RAND_MAX / (limit + 1);
	int retval;

	do { 
		retval = rand() / divisor;

	} while (retval > limit);

	return retval;
}



/*
 * Low rank threhsolding for arbitrary block sizes
 */
static void lrthresh_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(lrthresh_data_s, _data);

	float lambda = mu * data->lambda;

	bart_stride_t strs1[DIMS];
	md_calc_strides(DIMS, strs1, data->dims_decom, 1);

//#pragma omp parallel for
	for (int l = 0; l < data->levels; l++) {

		complex float* dstl = dst + l * strs1[LEVEL_DIM];
		const complex float* srcl = src + l * strs1[LEVEL_DIM];

		bart_dim_t blkdims[DIMS];
		bart_dim_t shifts[DIMS];
		bart_dim_t unshifts[DIMS];
		bart_dim_t zpad_dims[DIMS];
		bart_dim_t M = 1;

		for (int i = 0; i < DIMS; i++) {

			blkdims[i] = data->blkdims[l][i];
			zpad_dims[i] = (data->dims[i] + blkdims[i] - 1) / blkdims[i];
			zpad_dims[i] *= blkdims[i];

			if (MD_IS_SET(data->mflags, i))
				M *= blkdims[i];

			shifts[i] = 0.;

			if (data->randshift)
				shifts[i] = rand_lim((int)MIN(blkdims[i] - 1, zpad_dims[i] - blkdims[i]));

			unshifts[i] = -shifts[i];
		}

		bart_stride_t zpad_strs[DIMS];
		md_calc_strides(DIMS, zpad_strs, zpad_dims, CFL_SIZE);

		bart_dim_t blk_size = md_calc_size(DIMS, blkdims);
		bart_dim_t img_size = md_calc_size(DIMS, zpad_dims);
		bart_dim_t N = blk_size / M;
		bart_dim_t B = img_size / blk_size;

		if (data->noise && (l == data->levels - 1)) {

			M = img_size;
			N = 1;
			B = 1;
		}


		complex float* tmp = md_alloc_sameplace(DIMS, zpad_dims, CFL_SIZE, dst);

		md_circ_ext(DIMS, zpad_dims, tmp, data->dims, srcl, CFL_SIZE);

		md_circ_shift(DIMS, zpad_dims, shifts, tmp, tmp, CFL_SIZE);


		bart_dim_t mat_dims[2];

		(data->overlapping_blocks ? casorati_dims : basorati_dims)(DIMS, mat_dims, blkdims, zpad_dims);

		complex float* tmp_mat = md_alloc_sameplace(2, mat_dims, CFL_SIZE, dst);
		complex float* tmp_mat2 = tmp_mat;

		// Reshape image into a blk_size x number of blocks matrix
		(data->overlapping_blocks ? casorati_matrix : basorati_matrix)(DIMS, blkdims, mat_dims, tmp_mat, zpad_dims, zpad_strs, tmp);

		bart_dim_t num_blocks = mat_dims[1];
		bart_dim_t mat2_dims[2] = { mat_dims[0], mat_dims[1] };

		// FIXME: casorati and basorati are transposes of each other
		if (data->overlapping_blocks) {

			mat2_dims[0] = mat_dims[1];
			mat2_dims[1] = mat_dims[0];

			tmp_mat2 = md_alloc_sameplace(2, mat2_dims, CFL_SIZE, dst);

			md_transpose(2, 0, 1, mat2_dims, tmp_mat2, mat_dims, tmp_mat, CFL_SIZE);
			num_blocks = mat2_dims[1];

			if (B > 1)
				B = mat2_dims[1];
		}


		debug_printf(DP_DEBUG4, "M=%" PRId64 ", N=%" PRId64 ", B=%" PRId64 ", num_blocks=%" PRId64 ", img_size=%" PRId64 ", blk_size=%" PRId64 "\n", M, N, B, num_blocks, img_size, blk_size);

		batch_svthresh(M, N, num_blocks, lambda * GWIDTH(M, N, B), *(complex float (*)[mat2_dims[1]][M][N])tmp_mat2);
		//	for ( int b = 0; b < mat_dims[1]; b++ )
		//	svthresh(M, N, lambda * GWIDTH(M, N, B), tmp_mat, tmp_mat);

		if (data->overlapping_blocks)
			md_transpose(2, 0, 1, mat_dims, tmp_mat, mat2_dims, tmp_mat2, CFL_SIZE);

		(data->overlapping_blocks ? casorati_matrixH : basorati_matrixH)(DIMS, blkdims, zpad_dims, zpad_strs, tmp, mat_dims, tmp_mat);

		if (data->overlapping_blocks) {

			md_zsmul(DIMS, zpad_dims, tmp, tmp, 1. / M);
			md_free(tmp_mat2);
		}

		md_circ_shift(DIMS, zpad_dims, unshifts, tmp, tmp, CFL_SIZE);

		md_resize(DIMS, data->dims, dstl, zpad_dims, tmp, CFL_SIZE);

		md_free(tmp);
		md_free(tmp_mat);
	}
}



/*
 * Nuclear norm calculation for arbitrary block sizes
 */
float lrnucnorm(const struct operator_p_s* op, const complex float* src)
{
	auto data = CAST_DOWN(lrthresh_data_s, operator_p_get_data(op));

	bart_stride_t strs1[DIMS];
	md_calc_strides(DIMS, strs1, data->dims_decom, 1);
	float nnorm = 0.;


	for (int l = 0; l < data->levels; l++) {

		const complex float* srcl = src + l * strs1[LEVEL_DIM];

		bart_dim_t blkdims[DIMS];
		bart_dim_t blksize = 1;

		for (int i = 0; i < DIMS; i++) {

			blkdims[i] = data->blkdims[l][i];
			blksize *= blkdims[i];
		}

		if (1 == blksize) {

			for (bart_dim_t j = 0; j < md_calc_size(DIMS, data->dims); j++)
				nnorm += 2 * cabsf(srcl[j]);
				
			continue;
		}

		struct svthresh_blockproc_data* svdata = svthresh_blockproc_create(data->mflags, 0., 0);

		complex float* tmp = md_alloc_sameplace(DIMS, data->dims, CFL_SIZE, src);

		//debug_print_dims(DP_DEBUG1, DIMS, data->dims);
		md_copy(DIMS, data->dims, tmp, srcl, CFL_SIZE);

		// Block SVD Threshold
		nnorm = blockproc(DIMS, data->dims, blkdims, (void*)svdata, nucnorm_blockproc, tmp, tmp);

		xfree(svdata);
		md_free(tmp);
	}

	return nnorm;
}


static void llr_blkdims0(bart_dim_t blkdims[DIMS], bart_flags_t flags, const bart_dim_t idims[DIMS], int llrblk)
{
	md_copy_dims(DIMS, blkdims, idims);

	for (int i = 0; i < DIMS; i++) {

		if (!MD_IS_SET(flags, i))
			continue;

		blkdims[i] = MIN(llrblk, idims[i]);
	}
}




/*************
 * Block dimensions functions
 *************/


/**
 * Generates multiscale low rank block sizes
 *
 * @param blkdims - block sizes to be written
 * @param flags  - specifies which dimensions to do the blocks. The other dimensions will be the same as input
 * @param idims - input dimensions
 * @param blkskip - scale each level by blkskip to generate the next level
 *
 * returns number of levels
 */
int multilr_blkdims(bart_dim_t blkdims[MAX_LEV][DIMS], bart_flags_t flags, const bart_dim_t idims[DIMS], int blkskip, int initblk)
{
	// Multiscale low rank block sizes
	bart_dim_t tmp_block[DIMS];
	llr_blkdims0(tmp_block, flags, idims, initblk);

	bool done;
	// Loop block_sizes
	int levels = 0;

	do {
		md_copy_dims(DIMS, blkdims[levels], tmp_block);

		debug_print_dims(DP_INFO, DIMS, blkdims[levels]);

		done = true;

		for (int i = 0; i < DIMS; i++) {

			if (!MD_IS_SET(flags, i) || (1 == idims[i]))
				continue;

			tmp_block[i] = MIN(tmp_block[i] * blkskip, idims[i]);

			if (blkdims[levels][i] != idims[i])
				done = false;
		}

		levels++;

	} while (!done);

	return levels;
}



void add_lrnoiseblk(int* levels, bart_dim_t blkdims[MAX_LEV][DIMS], const bart_dim_t idims[DIMS])
{
	levels[0]++;

	debug_print_dims(DP_DEBUG1, DIMS, idims);
	md_copy_dims(DIMS, blkdims[levels[0] - 1], idims);
}



/**
 * Generates locally low rank block sizes
 *
 * @param blkdims - block sizes to be written
 * @param flags  - specifies which dimensions to do the blocks. The other dimensions will be the same as input
 * @param idims - input dimensions
 * @param llkblk - the block size
 *
 * returns number of levels = 1
 */
int llr_blkdims(bart_dim_t blkdims[MAX_LEV][DIMS], bart_flags_t flags, const bart_dim_t idims[DIMS], int llrblk)
{
	llr_blkdims0(blkdims[0], flags, idims, llrblk);
	return 1;
}



/**
 * Generates low rank + sparse block sizes
 *
 * @param blkdims - block sizes to be written
 * @param idims - input dimensions
 *
 * returns number of levels = 2
 */
int ls_blkdims(bart_dim_t blkdims[MAX_LEV][DIMS], const bart_dim_t idims[DIMS])
{
	for (int i = 0; i < DIMS; i++) {

		blkdims[0][i] = 1;
		blkdims[1][i] = idims[i];
	}

	return 2;
}


float get_lrthresh_lambda(const struct operator_p_s* o)
{
	auto data = CAST_DOWN(lrthresh_data_s, operator_p_get_data(o));

	return data->lambda;
}

