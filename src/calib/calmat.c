/* Copyright 2013-2016 The Regents of the University of California.
 * Copyright 2015-2021. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2024-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/casorati.h"
#include "num/lapack.h"
#include "num/linalg.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/mri2.h"
#include "misc/debug.h"

#include "linops/casorati.h"

#include "calmat.h"


/**
 *	Compute basic calibration matrix
 */
complex float* calibration_matrix(bart_dim_t calmat_dims[2], const bart_dim_t kdims[3], const bart_dim_t calreg_dims[4], const complex float* data)
{
	bart_dim_t kernel_dims[4];
	md_copy_dims(3, kernel_dims, kdims);
	kernel_dims[3] = calreg_dims[3];

	casorati_dims(4, calmat_dims, kernel_dims, calreg_dims);

	complex float* cm = md_alloc_sameplace(2, calmat_dims, CFL_SIZE, data);

	bart_stride_t calreg_strs[4];
	md_calc_strides(4, calreg_strs, calreg_dims, CFL_SIZE);
	casorati_matrix(4, kernel_dims, calmat_dims, cm, calreg_dims, calreg_strs, data);

	return cm;
}



/**
 *	Compute calibration matrix - but mask out a specified patch shape
 */
complex float* calibration_matrix_mask(bart_dim_t calmat_dims[2], const bart_dim_t kdims[3], const complex float* msk, const bart_dim_t calreg_dims[4], const complex float* data)
{
	complex float* tmp = calibration_matrix(calmat_dims, kdims, calreg_dims, data);

	if (NULL == msk)
		return tmp;

	bart_dim_t msk_dims[2];
	md_select_dims(2, ~MD_BIT(0), msk_dims, calmat_dims);

	assert(md_calc_size(3, kdims) * calreg_dims[3] == md_calc_size(2, msk_dims));

	bart_stride_t msk_strs[2];
	md_calc_strides(2, msk_strs, msk_dims, CFL_SIZE);

	// mask out un-sampled samples...
	bart_stride_t calmat_strs[2];
	md_calc_strides(2, calmat_strs, calmat_dims, CFL_SIZE);
	md_zmul2(2, calmat_dims, calmat_strs, tmp, calmat_strs, tmp, msk_strs, msk);

	return tmp;
}


/**
 *	Compute pattern matrix - mask out a specified patch shape
 */
static complex float* pattern_matrix(bart_dim_t pcm_dims[2], const bart_dim_t kdims[3], const complex float* mask, const bart_dim_t calreg_dims[4], const complex float* data)
{
	// estimate pattern
	bart_dim_t pat_dims[4];
	md_select_dims(4, ~MD_BIT(COIL_DIM), pat_dims, calreg_dims);
	complex float* pattern = md_alloc_sameplace(4, pat_dims, CFL_SIZE, data);
	estimate_pattern(4, calreg_dims, COIL_FLAG, pattern, data);

	// compute calibration matrix of pattern
	complex float* pm = calibration_matrix_mask(pcm_dims, kdims, mask, pat_dims, pattern);
	md_free(pattern);

	return pm;
}




complex float* calibration_matrix_mask2(bart_dim_t calmat_dims[2], const bart_dim_t kdims[3], const complex float* mask, const bart_dim_t calreg_dims[4], const complex float* data)
{
	bart_dim_t pcm_dims[2];
	complex float* pm = pattern_matrix(pcm_dims, kdims, mask, calreg_dims, data);

	bart_stride_t pcm_strs[2];
	md_calc_strides(2, pcm_strs, pcm_dims, CFL_SIZE);

	// number of samples for each patch
	bart_dim_t msk_dims[2];
	md_select_dims(2, ~MD_BIT(1), msk_dims, pcm_dims);

	bart_stride_t msk_strs[2];
	md_calc_strides(2, msk_strs, msk_dims, CFL_SIZE);

	complex float* msk = md_alloc(2, msk_dims, CFL_SIZE);
	md_clear(2, msk_dims, msk, CFL_SIZE);
	md_zfmacc2(2, pcm_dims, msk_strs, msk, pcm_strs, pm, pcm_strs, pm);
	md_free(pm);

	// fully sampled?
	md_zcmp2(2, msk_dims, msk_strs, msk, msk_strs, msk,
			(bart_dim_t[2]){ 0, 0 }, &(complex float){ /* pcm_dims[1] */ 15 }); // FIXME

	debug_printf(DP_DEBUG1, "%" PRId64 "/%" PRId64 " fully-sampled patches.\n",
				(bart_dim_t)pow(md_znorm(2, msk_dims, msk), 2.), pcm_dims[0]);

	complex float* tmp = calibration_matrix_mask(calmat_dims, kdims, mask, calreg_dims, data);

	// mask out incompletely sampled patches...
	bart_stride_t calmat_strs[2];
	md_calc_strides(2, calmat_strs, calmat_dims, CFL_SIZE);
	md_zmul2(2, calmat_dims, calmat_strs, tmp, calmat_strs, tmp, msk_strs, msk);

	return tmp;
}




#if 0
static void circular_patch_mask(const bart_dim_t kdims[3], int channels, complex float mask[channels * md_calc_size(3, kdims)])
{
	bart_dim_t kpos[3] = { };
	bart_dim_t kcen[3];

	for (int i = 0; i < 3; i++)
		kcen[i] = (1 == kdims[i]) ? 0 : (kdims[i] - 1) / 2;

	do {
		float dist = 0.;

		for (int i = 0; i < 3; i++)
			dist += (float)llabs(kpos[i] - kcen[i]) / (float)kdims[i];

		for (int c = 0; c < channels; c++)
			mask[((c * kdims[2] + kpos[2]) * kdims[1] + kpos[1]) * kdims[0] + kpos[0]] = (dist <= 0.5) ? 1 : 0;

	} while (md_next(3, kdims, 1 | 2 | 4, kpos));
}
#endif

void covariance_function(const bart_dim_t kdims[3], int N, complex float cov[N][N], const bart_dim_t calreg_dims[4], const complex float* data)
{
	bart_dim_t calmat_dims[2];
#if 1
	complex float* cm = calibration_matrix(calmat_dims, kdims, calreg_dims, data);
#else
	bart_dim_t channels = calreg_dims[3];
	complex float msk[channels * md_calc_size(3, kdims)];
	circular_patch_mask(kdims, channels, msk);
	complex float* cm = calibration_matrix_mask2(calmat_dims, kdims, msk, calreg_dims, data);
#endif
	int L = calmat_dims[0];
	assert(N == calmat_dims[1]);

	gram_matrix(N, cov, L, MD_CAST_ARRAY2(const complex float, 2, calmat_dims, cm, 0, 1));

	md_free(cm);
}

// use FFT to compute covariance function
// R. A. Lobos, C. -C. Chan and J. P. Haldar. New Theory and Faster Computations
// for Subspace-Based Sensitivity Map Estimation in Multichannel MRI. IEEE TMI 2024; 43: 286-296
void covariance_function_fft(const bart_dim_t kdims[3], int N, complex float cov[N][N], const bart_dim_t calreg_dims[4], const complex float* data)
{
	bart_dim_t kdims2[4] = { kdims[0], kdims[1], kdims[2], calreg_dims[3] };

	casorati_gram(N, cov, 4, kdims2, calreg_dims, data);
}



void calmat_svd(const bart_dim_t kdims[3], int N, complex float cov[N][N], float* S, const bart_dim_t calreg_dims[4], const complex float* data)
{
	bart_dim_t calmat_dims[2];
	complex float* cm = calibration_matrix(calmat_dims, kdims, calreg_dims, data);

	int L = calmat_dims[0];
	assert(N == calmat_dims[1]);

	PTR_ALLOC(complex float[L][L], U);

	// initialize to zero in case L < N not all written to
	for (int i = 0; i < N; i++)
		S[i] = 0.;

	lapack_svd_econ(L, N, *U, cov, S, MD_CAST_ARRAY2(complex float, 2, calmat_dims, cm, 0, 1));

	PTR_FREE(U);
	md_free(cm);
}

