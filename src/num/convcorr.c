/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <stddef.h>
#include <complex.h>

#include "num/flpmath.h"
#ifdef USE_GPU
#include "num/gpuops.h"
#include "num/gpukrnls.h"
#include "num/gpu_conv.h"
#ifdef USE_CUDNN
#include "num/cudnn_wrapper.h"
#endif
#endif
#include "num/multind.h"
#include "num/optimize.h"
#include "num/vecops.h"
#include "num/blas.h"
#include "num/rand.h"
#include "num/init.h"
#include "num/vecops_strided.h"
#include "num/vptr.h"

#include "misc/nested.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "convcorr.h"

static bool use_simple_convcorr = true;


//#define CONVCORR_OPTIMIZE_CPU_ONLY
//#define CONVCORR_OPTIMIZE_GPU_ONLY

/**
 * Copy from num/flpmath.c
 * Optimized threeop wrapper. Use when inputs are constants
 *
 * @param D number of dimensions
 * @param dim dimensions
 * @param ostr output strides
 * @param optr output
 * @param istr1 input 1 strides
 * @param iptr1 input 1 (constant)
 * @param istr2 input 2 strides
 * @param iptr2 input 2 (constant)
 * @param size size of data structures, e.g. complex float
 * @param too three-op multiply function
 */
static void optimized_threeop_oii(int D, const bart_dim_t dim[D], const bart_stride_t ostr[D], void* optr, const bart_stride_t istr1[D], const void* iptr1, const bart_stride_t istr2[D], const void* iptr2, size_t sizes[3], md_nary_opt_fun_t too)
{
	const bart_stride_t (*nstr[3])[D?D:1] = { (const bart_stride_t (*)[D?D:1])ostr, (const bart_stride_t (*)[D?D:1])istr1, (const bart_stride_t (*)[D?D:1])istr2 };
	void *nptr[3] = { optr, (void*)iptr1, (void*)iptr2 };

	bart_flags_t io = UINT64_C(1) + ((iptr1 == optr) ? 2 : 0) + ((iptr2 == optr) ? 4 : 0);

	(optimized_nop)(3, io, D, dim, nstr, nptr, sizes, too);
}



zconvcorr_fwd_algo_f* algos_fwd_cpu[] = { zconvcorr_fwd_im2col_cf_cpu, };
zconvcorr_bwd_krn_algo_f* algos_bwd_krn_cpu[] = { zconvcorr_bwd_krn_im2col_cf_cpu, };
zconvcorr_bwd_in_algo_f* algos_bwd_in_cpu[] = {	zconvcorr_bwd_in_im2col_cf_cpu, };

#ifdef USE_GPU
zconvcorr_bwd_krn_algo_f* algos_bwd_krn_gpu[] = {
#ifdef USE_CUDNN
	zconvcorr_bwd_krn_cudnn,
#endif
	zconvcorr_bwd_krn_im2col_cf_gpu,
};

zconvcorr_fwd_algo_f* algos_fwd_gpu[] = {
#ifdef USE_CUDNN
	zconvcorr_fwd_cudnn,
#endif
	zconvcorr_fwd_im2col_cf_gpu,
};

zconvcorr_bwd_in_algo_f* algos_bwd_in_gpu[] = {
#ifdef USE_CUDNN
	zconvcorr_bwd_in_cudnn,
#endif
	zconvcorr_bwd_in_im2col_cf_gpu,
};
#endif




//detect if strides describe convolution
static bool detect_convcorr(	int N,
				bart_dim_t nodims[N], bart_dim_t nidims[N], bart_dim_t nkdims[N],
				bart_stride_t nostrs[N], bart_stride_t nistrs[N], bart_stride_t nkstrs[N],
				bart_dim_t dilation[N], bart_stride_t strides[N],
				bart_flags_t* ptr_flag, bool* ptr_conv,
				const bart_dim_t dims[2 * N], const bart_stride_t ostrs[2 * N], const bart_stride_t istrs[2 * N], const bart_stride_t kstrs[2 * N],
				size_t size);


//functions detecting strides for a specific call and running the algorithms
static bool simple_zconvcorr_fwd(	int N, const bart_dim_t dims[N],
					const bart_stride_t ostrs[N], complex float* optr,
					const bart_stride_t istrs1[N], const complex float* iptr1,
					const bart_stride_t istrs2[N], const complex float* iptr2);
static bool simple_zconvcorr_bwd_in(	int N, const bart_dim_t dims[N],
					const bart_stride_t ostrs[N], complex float* optr,
					const bart_stride_t istrs1[N], const complex float* iptr1,
					const bart_stride_t istrs2[N], const complex float* iptr2);
static bool simple_zconvcorr_bwd_krn(	int N, const bart_dim_t dims[N],
					const bart_stride_t ostrs[N], complex float* optr,
					const bart_stride_t istrs1[N], const complex float* iptr1,
					const bart_stride_t istrs2[N], const complex float* iptr2);


static bool detect_convcorr(	int N,
				bart_dim_t nodims[N], bart_dim_t nidims[N], bart_dim_t nkdims[N],
				bart_stride_t nostrs[N], bart_stride_t nistrs[N], bart_stride_t nkstrs[N],
				bart_dim_t dilation[N], bart_stride_t strides[N],
				bart_flags_t* ptr_flag, bool* ptr_conv,
				const bart_dim_t dims[2 * N], const bart_stride_t ostrs[2 * N], const bart_stride_t istrs[2 * N], const bart_stride_t kstrs[2 * N],
				size_t size)
{
	bart_dim_t istrs_triv = (bart_stride_t)size;

	*ptr_flag = 0;
	*ptr_conv = true;

	md_singleton_dims(N, dilation);
	md_singleton_dims(N, strides);

	for (int i = 0; i < N; i++) {

		if ((1 != dims[i]) && (1 != dims[N + i])) {

			*ptr_flag = MD_SET(*ptr_flag, i);

			nodims[i] = dims[0 + i];
			nkdims[i] = dims[N + i];

			if (0 != kstrs[i])
				return false;

			nkstrs[i] = kstrs[N + i];

			if (0 != ostrs[N + i])
				return false;

			nostrs[i] = ostrs[i];

			bart_stride_t test_strides[] = { istrs[i] / istrs_triv, 1, 2, 3, 4, 5, 6, 7, 8 };
			bool found = false;

			for (int j = 0; !found && j < (int)ARRAY_SIZE(test_strides); j++) {

				strides[i] = test_strides[j];

				if (1 > strides[i])
					continue;

				if (0 != istrs[i] % strides[i])
					continue;

				nistrs[i] = istrs[i] / strides[i];

				if ((0 == nistrs[i]) || (0 != istrs[N + i] % nistrs[i]))
					continue;

				dilation[i] = istrs[N + i] / nistrs[i];

				nidims[i] = strides[i] * (nodims[i] - 1) + 1 + dilation[i] * (nkdims[i] - 1);

				found = true;
			}

			istrs_triv *= nidims[i];

			*ptr_conv = *ptr_conv && (0 >= nkstrs[i]);

			if (!found)
				return false;

		} else {

			if (1 != dims[N +  i])
				return false;

			nostrs[i] = ostrs[i];
			nistrs[i] = istrs[i];
			nkstrs[i] = kstrs[i];

			nodims[i] = (0 == nostrs[i]) ? 1 : dims[i];
			nkdims[i] = (0 == nkstrs[i]) ? 1 : dims[i];
			nidims[i] = (0 == nistrs[i]) ? 1 : dims[i];

			dilation[i] = 1;
			strides[i] = 1;

			if (0 != nistrs[i])
				istrs_triv *= nidims[i];
		}
	}

	for (int i = 0; i < N; i++)
		if (MD_IS_SET(*ptr_flag, i))
			if (*ptr_conv)
				nkstrs[i] = -nkstrs[i];

	if (0 == *ptr_flag)
		return false;

#if 1 // this is a cross check, that the detected dims/strides reproduce the input strides/dims
	bart_dim_t tdims[2 * N];
	bart_stride_t tostrs[2 * N];
	bart_stride_t tistrs[2 * N];
	bart_stride_t tkstrs[2 * N];

	calc_convcorr_geom_strs_dil(	N, *ptr_flag,
					tdims, tostrs, tkstrs, tistrs,
					nodims, nostrs, nkdims, nkstrs, nidims, nistrs,
					dilation, strides, *ptr_conv, false);

	assert(md_check_equal_dims(2 * N, tdims, dims, ~UINT64_C(0)));
	assert(md_check_equal_dims(2 * N, tostrs, ostrs, md_nontriv_dims(2 * N, dims)));
	assert(md_check_equal_dims(2 * N, tistrs, istrs, md_nontriv_dims(2 * N, dims)));
	assert(md_check_equal_dims(2 * N, tkstrs, kstrs, md_nontriv_dims(2 * N, dims)));
#endif
	return true;
}


bool simple_zconvcorr(	int N, const bart_dim_t dims[N],
			const bart_stride_t ostrs[N], complex float* optr,
			const bart_stride_t istrs1[N], const complex float* iptr1,
			const bart_stride_t istrs2[N], const complex float* iptr2)
{
	if (!use_simple_convcorr)
		return false;

	if (is_vptr(optr) || is_vptr(iptr1) || is_vptr(iptr2))
		return false;

	if (simple_zconvcorr_fwd(N, dims, ostrs, optr, istrs1, iptr1, istrs2, iptr2))
		return true;

	if (simple_zconvcorr_bwd_in(N, dims, ostrs, optr, istrs1, iptr1, istrs2, iptr2))
		return true;

	if (simple_zconvcorr_bwd_krn(N, dims, ostrs, optr, istrs1, iptr1, istrs2, iptr2))
		return true;

	return false;
}


//The following three function detect a (transposed) convolution and run the specific algorithms
static bool simple_zconvcorr_fwd(	int N, const bart_dim_t dims[N],
					const bart_stride_t ostrs[N], complex float* optr,
					const bart_stride_t istrs1[N], const complex float* iptr1,
					const bart_stride_t istrs2[N], const complex float* iptr2)
{
	if (0 != N % 2)
		return false;

	N /= 2;

	size_t size = CFL_SIZE;

	bart_flags_t flags;
	bool conv;
	bart_dim_t nodims[N];
	bart_dim_t nidims[N];
	bart_dim_t nkdims[N];

	bart_stride_t nostrs[N];
	bart_stride_t nistrs[N];
	bart_stride_t nkstrs[N];

	bart_dim_t dilation[N];
	bart_stride_t strides[N];

	complex float* out = NULL;
	const complex float* in = NULL;
	const complex float* krn = NULL;

	bool result = false;

	if (detect_convcorr(	N,
				nodims, nidims, nkdims,
				nostrs, nistrs, nkstrs,
				dilation, strides,
				&flags, &conv,
				dims, ostrs, istrs1, istrs2,
				size)) {

		out = optr;
		in = iptr1;
		krn = iptr2;
		result = true;
	}

	if ((!result) && (detect_convcorr(	N,
						nodims, nidims, nkdims,
						nostrs, nistrs, nkstrs,
						dilation, strides,
						&flags, &conv,
						dims, ostrs, istrs2, istrs1,
						size))) {

		out = optr;
		in = iptr2;
		krn = iptr1;
		result = true;
	}

	if (!result)
		return false;

	bart_dim_t tdims[2 * N];
	bart_stride_t tostrs[2 * N];
	bart_stride_t tistrs[2 * N];
	bart_stride_t tkstrs[2 * N];

	krn -= calc_convcorr_geom_strs_dil(	N, flags,
						tdims, tostrs, tkstrs, tistrs,
						nodims, nostrs,
						nkdims, nkstrs,
						nidims, nistrs,
						dilation, strides, conv, false) / (bart_stride_t)size;

#ifdef USE_GPU
	if (cuda_ondevice(out))
		for (int i = 0; (bart_flags_t)i < sizeof(algos_fwd_gpu) / sizeof(algos_fwd_gpu[0]); i++)
			if (algos_fwd_gpu[i](	N,
						nodims, nostrs, out,
						nidims, nistrs, in,
						nkdims, nkstrs, krn,
						flags, dilation, strides, conv))
				return true;

	if (!cuda_ondevice(out))
#endif
		for (int i = 0; (bart_flags_t)i < sizeof(algos_fwd_cpu) / sizeof(algos_fwd_cpu[0]); i++)
			if (algos_fwd_cpu[i](	N,
						nodims, nostrs, out,
						nidims, nistrs, in,
						nkdims, nkstrs, krn,
						flags, dilation, strides, conv))
				return true;

	return false;
}


static bool simple_zconvcorr_bwd_in(	int N, const bart_dim_t dims[N],
					const bart_stride_t ostrs[N], complex float* optr,
					const bart_stride_t istrs1[N], const complex float* iptr1,
					const bart_stride_t istrs2[N], const complex float* iptr2)
{
	if (0 != N % 2)
		return false;

	N /= 2;

	size_t size = CFL_SIZE;

	bart_flags_t flags;
	bool conv;
	bart_dim_t nodims[N];
	bart_dim_t nidims[N] = { };	// GCC ANALYZER
	bart_dim_t nkdims[N];

	bart_stride_t nostrs[N];
	bart_stride_t nistrs[N];
	bart_stride_t nkstrs[N];

	bart_dim_t dilation[N];
	bart_stride_t strides[N];

	const complex float* out = NULL;
	complex float* in = NULL;
	const complex float* krn = NULL;

	bool result = false;

	if (detect_convcorr(	N,
				nodims, nidims, nkdims,
				nostrs, nistrs, nkstrs,
				dilation, strides,
				&flags, &conv,
				dims, istrs1, ostrs, istrs2,
				size)) {

		out = iptr1;
		in = optr;
		krn = iptr2;
		result = true;
	}

	if ((!result) && (detect_convcorr(	N,
						nodims, nidims, nkdims,
						nostrs, nistrs, nkstrs,
						dilation, strides,
						&flags, &conv,
						dims, istrs2, ostrs, istrs1,
						size))) {

		out = iptr2;
		in = optr;
		krn = iptr1;
		result = true;
	}

	if (!result)
		return false;

	bart_dim_t tdims[2 * N];
	bart_stride_t tostrs[2 * N];
	bart_stride_t tistrs[2 * N];
	bart_stride_t tkstrs[2 * N];

	krn -= calc_convcorr_geom_strs_dil(	N, flags,
						tdims, tostrs, tkstrs, tistrs,
						nodims, nostrs,
						nkdims, nkstrs,
						nidims, nistrs,
						dilation, strides, conv, false) / (bart_stride_t)size;

#ifdef USE_GPU
	if (cuda_ondevice(out))
		for(int i = 0; (bart_flags_t)i < sizeof(algos_bwd_in_gpu) / sizeof(algos_bwd_in_gpu[0]); i++)
			if (algos_bwd_in_gpu[i](	N,
							nodims, nostrs, out,
							nidims, nistrs, in,
							nkdims, nkstrs, krn,
							flags, dilation, strides, conv))
				return true;
#endif


#ifdef USE_GPU
	if (!cuda_ondevice(out))
#else
	if (true)
#endif
	for(int i = 0; (bart_flags_t)i < sizeof(algos_bwd_in_cpu) / sizeof(algos_bwd_in_cpu[0]); i++)
		if (algos_bwd_in_cpu[i](	N,
						nodims, nostrs, out,
						nidims, nistrs, in,
						nkdims, nkstrs, krn,
						flags, dilation, strides, conv))
			return true;

	return false;
}


static bool simple_zconvcorr_bwd_krn(	int N, const bart_dim_t dims[N],
					const bart_stride_t ostrs[N], complex float* optr,
					const bart_stride_t istrs1[N], const complex float* iptr1,
					const bart_stride_t istrs2[N], const complex float* iptr2)
{
	if (0 != N % 2)
		return false;

	N /= 2;

	size_t size = CFL_SIZE;

	bart_flags_t flags;
	bool conv;
	bart_dim_t nodims[N];
	bart_dim_t nidims[N] = { };	// GCC ANAYLZER
	bart_dim_t nkdims[N];

	bart_stride_t nostrs[N];
	bart_stride_t nistrs[N];
	bart_stride_t nkstrs[N];

	bart_dim_t dilation[N];
	bart_stride_t strides[N];

	const complex float* out = NULL;
	const complex float* in = NULL;
	complex float* krn = NULL;

	bool result = false;

		if (detect_convcorr(	N,
				nodims, nidims, nkdims,
				nostrs, nistrs, nkstrs,
				dilation, strides,
				&flags, &conv,
				dims, istrs1, istrs2, ostrs,
				size)) {

		out = iptr1;
		in = iptr2;
		krn = optr;
		result = true;
	}

	if ((!result) && (detect_convcorr(	N,
						nodims, nidims, nkdims,
						nostrs, nistrs, nkstrs,
						dilation, strides,
						&flags, &conv,
						dims, istrs2, istrs1, ostrs,
						size))) {

		out = iptr2;
		in = iptr1;
		krn = optr;
		result = true;
	}

	if (!result)
		return false;

	bart_dim_t tdims[2 * N];
	bart_stride_t tostrs[2 * N];
	bart_stride_t tistrs[2 * N];
	bart_stride_t tkstrs[2 * N];

	krn -= calc_convcorr_geom_strs_dil(	N, flags,
						tdims, tostrs, tkstrs, tistrs,
						nodims, nostrs,
						nkdims, nkstrs,
						nidims, nistrs,
						dilation, strides, conv, false) / (bart_stride_t)size;

#ifdef USE_GPU
	if (cuda_ondevice(out))
		for(int i = 0; (bart_flags_t)i < sizeof(algos_bwd_krn_gpu) / sizeof(algos_bwd_krn_gpu[0]); i++)
			if (algos_bwd_krn_gpu[i](	N,
							nodims, nostrs, out,
							nidims, nistrs, in,
							nkdims, nkstrs, krn,
							flags, dilation, strides, conv))
				return true;
#endif

#ifdef USE_GPU
	if (!cuda_ondevice(out))
#else
	if (true)
#endif
		for(int i = 0; (bart_flags_t)i < sizeof(algos_bwd_krn_cpu) / sizeof(algos_bwd_krn_cpu[0]); i++)
			if (algos_bwd_krn_cpu[i](	N,
							nodims, nostrs, out,
							nidims, nistrs, in,
							nkdims, nkstrs, krn,
							flags, dilation, strides, conv))
				return true;

	return false;
}


/**
 * Checks if params correspond to convcorr which is channel first and contiguous in memory
 */
static bool check_trivial_cf(	int N,
				bart_dim_t odims[N], bart_stride_t ostrs[N],
				bart_dim_t idims[N], bart_stride_t istrs[N],
				bart_dim_t kdims[N], bart_stride_t kstrs[N],
				bart_flags_t flags,
				size_t size)
{
	// Check conv dims
	for (int i = 2; i < N; i++)
		if ((!MD_IS_SET(flags, i)) && ((1 != idims[i]) && (1 != kdims[i])))
			return false;

	// Check matmul dims
	if (MD_IS_SET(flags, 0) || MD_IS_SET(flags, 1) || (1 != idims[0]) || (1 != odims[1]))
		return false;

	// check contiguous memory
	if (N > md_calc_blockdim(N, odims, ostrs, size))
		return false;

	if (N > md_calc_blockdim(N, idims, istrs, size))
		return false;

	if (N > md_calc_blockdim(N, kdims, kstrs, size))
		return false;

	return true;
}

static bool check_trivial_strs_dil(int N, const bart_dim_t dilation[N], const bart_stride_t strides[N])
{
	if ((NULL != dilation) && (!md_check_equal_dims(N, dilation, MD_SINGLETON_DIMS(N), ~UINT64_C(0))))
		return false;

	if ((NULL != strides) && (!md_check_equal_dims(N, strides, MD_SINGLETON_DIMS(N), ~UINT64_C(0))))
		return false;

	return true;
}


bool zconvcorr_fwd_im2col_cf_cpu(int N,
				bart_dim_t odims[N], bart_stride_t ostrs[N], complex float* out,
				bart_dim_t idims[N], bart_stride_t istrs[N], const complex float* in,
				bart_dim_t kdims[N], bart_stride_t kstrs[N], const complex float* krn,
				bart_flags_t flags, const bart_dim_t dilation[N], const bart_stride_t strides[N], bool conv)
{
#ifdef NO_BLAS
	return false;
#else
#ifdef USE_GPU
	if (cuda_ondevice(out))
		return false;
#endif
	if (5 > N)
		return false;

	if (!check_trivial_cf(5, odims, ostrs, idims, istrs, kdims, kstrs, flags, CFL_SIZE))
		return false;

	if (!check_trivial_strs_dil(5, dilation, strides))
		return false;

	if (conv)
		return false;

	bart_dim_t dims_mat[8]; // (1 | nr_in_channel, kx, ky, kz | outx, outy, outz)

	md_copy_dims(5, dims_mat, kdims);
	md_copy_dims(3, dims_mat + 5, odims + 2);


	bart_dim_t kdims_mat[8]; // (nr_filter | nr_in_channel, kx, ky, kz | 1, 1, 1 )

	md_select_dims(8, MD_BIT(5) - 1, kdims_mat, dims_mat);


	bart_dim_t idims_mat[N + 3]; // (1 | nr_in_channel, kx, ky, kz | outx, outy, outz | ... )

	md_select_dims(8, ~UINT64_C(1) , idims_mat, dims_mat);
	md_copy_dims(N - 5, idims_mat + 8, idims + 5);


	bart_dim_t odims_mat[8]; // (nr_filter | 1, 1, 1, 1 | outx, outy, outz)

	md_select_dims(8, MD_BIT(0) | MD_BIT(5) | MD_BIT(6) | MD_BIT(7), odims_mat, dims_mat);


	bart_dim_t istrs_mat[8];

	md_copy_strides(5, istrs_mat, MD_STRIDES(5, idims, CFL_SIZE));
	md_copy_strides(3, istrs_mat + 5, MD_STRIDES(5, idims, CFL_SIZE) + 2);


	bart_dim_t osize = odims[0] * odims[1] * odims[2] * odims[3] * odims[4];
	bart_dim_t ksize = kdims[0] * kdims[1] * kdims[2] * kdims[3] * kdims[4];
	bart_dim_t isize = idims[0] * idims[1] * idims[2] * idims[3] * idims[4];

	bart_dim_t M1 = dims_mat[0];
	bart_dim_t K1 = dims_mat[1] * dims_mat[2] * dims_mat[3] * dims_mat[4];
	bart_dim_t N1 = dims_mat[5] * dims_mat[6] * dims_mat[7];

	bart_dim_t* idims_matP = idims_mat; // clang
	bart_dim_t* istrs_matP = istrs_mat;

	bart_dim_t mdims[N - 5];

	md_tenmul_dims(N - 5, mdims, odims + 5, idims + 5, kdims + 5);


	NESTED(void, nary_zconvcorr3D_I2C_CF, (struct nary_opt_data_s* data, void* ptr[]))
	{
		for (bart_dim_t i = 0; i < data->size; i++){

			complex float* imat_tmp = md_alloc_sameplace(8, idims_matP, CFL_SIZE, in);

			md_copy2(8, idims_matP, MD_STRIDES(8, idims_matP, CFL_SIZE), imat_tmp,
					istrs_matP, (const complex float*)ptr[1] + i * isize, CFL_SIZE);

			blas_matrix_zfmac(	M1, N1, K1,
						(complex float*)ptr[0] + i * osize,
						(complex float*)ptr[2] + i * ksize, 'N',
						imat_tmp, 'N'
						);

			md_free(imat_tmp);
		}
	};

	optimized_threeop_oii(N - 5, mdims, ostrs + 5, (void*)out, istrs + 5, (void*)in, kstrs + 5, (void*)krn,
				(size_t[3]){ (size_t)(osize * (bart_stride_t)CFL_SIZE), (size_t)(isize * (bart_stride_t)CFL_SIZE), (size_t)(ksize * (bart_stride_t)CFL_SIZE) },
				CLOSURE(md_nary_opt_fun_t, nary_zconvcorr3D_I2C_CF));

	debug_printf(DP_DEBUG3, "conv by %s \n", __func__);

	return true;
#endif
}


bool zconvcorr_bwd_krn_im2col_cf_cpu(int N,
				bart_dim_t odims[N], bart_stride_t ostrs[N], const complex float* out,
				bart_dim_t idims[N], bart_stride_t istrs[N], const complex float* in,
				bart_dim_t kdims[N], bart_stride_t kstrs[N], complex float* krn,
				bart_flags_t flags, const bart_dim_t dilation[N], const bart_stride_t strides[N], bool conv)
{
#ifdef NO_BLAS
	return false;
#else
#ifdef USE_GPU
	if (cuda_ondevice(out))
		return false;
#endif
	size_t size = CFL_SIZE;

	if (5 > N)
		return false;

	if (!check_trivial_cf(5, odims, ostrs, idims, istrs, kdims, kstrs, flags, size))
		return false;

	if (!check_trivial_strs_dil(5, dilation, strides))
		return false;

	if (conv)
		return false;


	bart_dim_t dims_mat[8]; // (1 | nr_in_channel, kx, ky, kz | outx, outy, outz)

	md_copy_dims(5, dims_mat, kdims);
	md_copy_dims(3, dims_mat + 5, odims + 2);


	bart_dim_t kdims_mat[8]; // (nr_filter | nr_in_channel, kx, ky, kz | 1, 1, 1 )

	md_select_dims(8, MD_BIT(5) - 1, kdims_mat, dims_mat);


	bart_dim_t idims_mat[N + 3]; // (1 | nr_in_channel, kx, ky, kz | outx, outy, outz | ... )

	md_select_dims(8, ~UINT64_C(1) , idims_mat, dims_mat);
	md_copy_dims(N - 5, idims_mat + 8, idims + 5);


	bart_dim_t odims_mat[8]; // (nr_filter | 1, 1, 1, 1 | outx, outy, outz)

	md_select_dims(8, MD_BIT(0) | MD_BIT(5) | MD_BIT(6) | MD_BIT(7), odims_mat, dims_mat);


	bart_dim_t istrs_mat[8];

	md_copy_strides(5, istrs_mat, MD_STRIDES(5, idims, size));
	md_copy_strides(3, istrs_mat + 5, MD_STRIDES(5, idims, size) + 2);

	bart_dim_t osize = odims[0] * odims[1] * odims[2] * odims[3] * odims[4];
	bart_dim_t ksize = kdims[0] * kdims[1] * kdims[2] * kdims[3] * kdims[4];
	bart_dim_t isize = idims[0] * idims[1] * idims[2] * idims[3] * idims[4];

	bart_dim_t M1 = dims_mat[0];
	bart_dim_t K1 = dims_mat[1] * dims_mat[2] * dims_mat[3] * dims_mat[4];
	bart_dim_t N1 = dims_mat[5] * dims_mat[6] * dims_mat[7];

	bart_dim_t* idims_matP = idims_mat; // clang
	bart_dim_t* istrs_matP = istrs_mat;

	bart_dim_t mdims[N - 5];

	md_tenmul_dims(N - 5, mdims, odims + 5, idims + 5, kdims + 5);


	NESTED(void, nary_zconvcorr_im2col, (struct nary_opt_data_s* data, void* ptr[]))
	{
		for (bart_dim_t i = 0; i < data->size; i++){

			complex float* imat_tmp = md_alloc_sameplace(8, idims_matP, size, in);

			md_copy2(8, idims_matP, MD_STRIDES(8, idims_matP, size), imat_tmp, istrs_matP, (const complex float*)ptr[1] + i * isize, size);

			blas_matrix_zfmac(	M1, K1, N1,
						(complex float*)ptr[0] + i * ksize,
						(complex float*)ptr[2] + i * osize, 'N',
						imat_tmp, 'T'
						);

			md_free(imat_tmp);
		}
	};

	optimized_threeop_oii(N - 5, mdims, kstrs + 5, (void*)krn, istrs + 5, (void*)in, ostrs + 5, (void*)out,
				(size_t[3]){ (size_t)(ksize * (bart_stride_t)CFL_SIZE), (size_t)(isize * (bart_stride_t)CFL_SIZE), (size_t)(osize * (bart_stride_t)CFL_SIZE) },
				CLOSURE(md_nary_opt_fun_t, nary_zconvcorr_im2col));

	debug_printf(DP_DEBUG3, "conv by %s \n", __func__);

	return true;
#endif
}


bool zconvcorr_bwd_in_im2col_cf_cpu(int N,
				bart_dim_t odims[N], bart_stride_t ostrs[N], const complex float* out,
				bart_dim_t idims[N], bart_stride_t istrs[N], complex float* in,
				bart_dim_t kdims[N], bart_stride_t kstrs[N], const complex float* krn,
				bart_flags_t flags, const bart_dim_t dilation[N], const bart_stride_t strides[N], bool conv)
{
#ifdef NO_BLAS
	return false;
#else
#ifdef USE_GPU
	if (cuda_ondevice(out))
		return false;
#endif

	if (5 > N)
		return false;

	if (!check_trivial_cf(5, odims, ostrs, idims, istrs, kdims, kstrs, flags, CFL_SIZE))
		return false;

	if (!check_trivial_strs_dil(5, dilation, strides))
		return false;

	if (conv)
		return false;


	bart_dim_t dims_mat[8]; // (1 | nr_in_channel, kx, ky, kz | outx, outy, outz)

	md_copy_dims(5, dims_mat, kdims);
	md_copy_dims(3, dims_mat + 5, odims + 2);


	bart_dim_t kdims_mat[8]; // (nr_filter | nr_in_channel, kx, ky, kz | 1, 1, 1 )

	md_select_dims(8, MD_BIT(5) - 1, kdims_mat, dims_mat);


	bart_dim_t idims_mat[N + 3]; // (1 | nr_in_channel, kx, ky, kz | outx, outy, outz | ... )

	md_select_dims(8, ~UINT64_C(1) , idims_mat, dims_mat);
	md_copy_dims(N - 5, idims_mat + 8, idims + 5);


	bart_dim_t odims_mat[8]; // (nr_filter | 1, 1, 1, 1 | outx, outy, outz)

	md_select_dims(8, MD_BIT(0) | MD_BIT(5) | MD_BIT(6) | MD_BIT(7), odims_mat, dims_mat);


	bart_dim_t istrs_mat[8];

	md_copy_strides(5, istrs_mat, MD_STRIDES(5, idims, CFL_SIZE));
	md_copy_strides(3, istrs_mat + 5, MD_STRIDES(5, idims, CFL_SIZE) + 2);


	bart_dim_t osize = odims[0] * odims[1] * odims[2] * odims[3] * odims[4];
	bart_dim_t ksize = kdims[0] * kdims[1] * kdims[2] * kdims[3] * kdims[4];
	bart_dim_t isize = idims[0] * idims[1] * idims[2] * idims[3] * idims[4];

	bart_dim_t M1 = dims_mat[0];
	bart_dim_t K1 = dims_mat[1] * dims_mat[2] * dims_mat[3] * dims_mat[4];
	bart_dim_t N1 = dims_mat[5] * dims_mat[6] * dims_mat[7];

	bart_dim_t* idims_matP = idims_mat; // clang
	bart_dim_t* istrs_matP = istrs_mat;

	bart_dim_t mdims[N - 5];

	md_tenmul_dims(N - 5, mdims, odims + 5, idims + 5, kdims + 5);


	NESTED(void, nary_zconvcorr3D_I2C_CF, (struct nary_opt_data_s* data, void* ptr[]))
	{
		for (bart_dim_t i = 0; i < data->size; i++){

			complex float* imat_tmp = md_alloc_sameplace(8, idims_matP, CFL_SIZE, in);
			md_clear(8, idims_matP, imat_tmp, CFL_SIZE);

			blas_matrix_zfmac(	K1, N1, M1,
						imat_tmp,
						(complex float*)ptr[2] + i * ksize, 'T',
						(complex float*)ptr[1] + i * osize, 'N'
						);

			md_zadd2(8, idims_matP, istrs_matP, (complex float*)ptr[0] + i * isize,
					istrs_matP, (const complex float*)ptr[0] + i * isize,
					MD_STRIDES(8, idims_matP, CFL_SIZE), imat_tmp);

			md_free(imat_tmp);
		}
	};

	optimized_threeop_oii(N - 5, mdims, istrs + 5, (void*)in, ostrs + 5, (void*)out, kstrs + 5, (void*)krn,
				(size_t[3]){ (size_t)(osize * (bart_stride_t)CFL_SIZE), (size_t)(isize * (bart_stride_t)CFL_SIZE), (size_t)(ksize * (bart_stride_t)CFL_SIZE) },
				CLOSURE(md_nary_opt_fun_t, nary_zconvcorr3D_I2C_CF));

	debug_printf(DP_DEBUG3, "conv by %s \n", __func__);

	return true;
#endif
}


#ifdef USE_GPU
bool zconvcorr_fwd_im2col_cf_gpu(int N,
				bart_dim_t odims[N], bart_stride_t ostrs[N], complex float* out,
				bart_dim_t idims[N], bart_stride_t istrs[N], const complex float* in,
				bart_dim_t kdims[N], bart_stride_t kstrs[N], const complex float* krn,
				bart_flags_t flags, const bart_dim_t dilation[N], const bart_stride_t strides[N], bool conv)
{
	if (!cuda_ondevice(out))
		return false;

	if (5 > N)
		return false;

	if (!check_trivial_cf(5, odims, ostrs, idims, istrs, kdims, kstrs, flags, CFL_SIZE))
		return false;

	if (conv)
		return false;

	// mim2col dims (nr_out_channel | nr_in_channel, kx, ky, kz | outx, outy, outz)
	// kernel	(nr_filter | nr_in_channel, kx, ky, kz | 1, 1, 1 )
	// image	(1 | nr_in_channel, kx, ky, kz | outx, outy, outz | ... )
	// output 	(nr_filter | 1, 1, 1, 1 | outx, outy, outz)

	bart_dim_t osize = odims[0] * odims[1] * odims[2] * odims[3] * odims[4];
	bart_dim_t ksize = kdims[0] * kdims[1] * kdims[2] * kdims[3] * kdims[4];
	bart_dim_t isize = idims[0] * idims[1] * idims[2] * idims[3] * idims[4];

	bart_dim_t M1 = kdims[0];
	bart_dim_t K1 = kdims[1] * kdims[2] * kdims[3] * kdims[4];
	bart_dim_t N1 = odims[2] * odims[3] * odims[4];

	bart_dim_t imat_size = K1 * N1;

	bart_dim_t mdims[N - 5];

	md_tenmul_dims(N - 5, mdims, odims + 5, idims + 5, kdims + 5);

	//clang
	const bart_dim_t* odimsp = odims;
	const bart_dim_t* idimsp = idims;
	const bart_dim_t* kdimsp = kdims;
	const bart_dim_t* dilationp = dilation;
	const bart_dim_t* stridesp = strides;

	NESTED(void, nary_zconvcorr_im2col, (struct nary_opt_data_s* data, void* ptr[]))
	{
		for (bart_dim_t i = 0; i < data->size; i++){

			complex float* imat_tmp = md_alloc_gpu(1, &imat_size, CFL_SIZE);
			cuda_im2col(imat_tmp, (const complex float*)ptr[1] + i * isize, odimsp, idimsp, kdimsp, dilationp, stridesp);

			blas_matrix_zfmac(	M1, N1, K1,
						(complex float*)ptr[0] + i * osize,
						(complex float*)ptr[2] + i * ksize, 'N',
						imat_tmp, 'N'
						);
			md_free(imat_tmp);
		}
	};

	optimized_threeop_oii(N - 5, mdims, ostrs + 5, (void*)out, istrs + 5, (void*)in, kstrs + 5, (void*)krn,
				(size_t[3]){ (size_t)((bart_stride_t)CFL_SIZE * osize), (size_t)((bart_stride_t)CFL_SIZE * isize), (size_t)((bart_stride_t)CFL_SIZE * ksize) },
				nary_zconvcorr_im2col);

	debug_printf(DP_DEBUG3, "conv by %s \n", __func__);

	return true;
}
#endif


#ifdef USE_GPU
bool zconvcorr_bwd_krn_im2col_cf_gpu(int N,
				bart_dim_t odims[N], bart_stride_t ostrs[N], const complex float* out,
				bart_dim_t idims[N], bart_stride_t istrs[N], const complex float* in,
				bart_dim_t kdims[N], bart_stride_t kstrs[N], complex float* krn,
				bart_flags_t flags, const bart_dim_t dilation[N], const bart_stride_t strides[N], bool conv)
{
	if (!cuda_ondevice(out))
		return false;

	size_t size = CFL_SIZE;

	if (5 > N)
		return false;

	if (!check_trivial_cf(5, odims, ostrs, idims, istrs, kdims, kstrs, flags, size))
		return false;

	if (conv)
		return false;

	// mim2col dims (nr_out_channel | nr_in_channel, kx, ky, kz | outx, outy, outz)
	// kernel	(nr_filter | nr_in_channel, kx, ky, kz | 1, 1, 1 )
	// image	(1 | nr_in_channel, kx, ky, kz | outx, outy, outz | ... )
	// output 	(nr_filter | 1, 1, 1, 1 | outx, outy, outz)

	bart_dim_t osize = odims[0] * odims[1] * odims[2] * odims[3] * odims[4];
	bart_dim_t ksize = kdims[0] * kdims[1] * kdims[2] * kdims[3] * kdims[4];
	bart_dim_t isize = idims[0] * idims[1] * idims[2] * idims[3] * idims[4];

	bart_dim_t M1 = kdims[0];
	bart_dim_t K1 = kdims[1] * kdims[2] * kdims[3] * kdims[4];
	bart_dim_t N1 = odims[2] * odims[3] * odims[4];

	bart_dim_t imat_size = K1 * N1;

	bart_dim_t mdims[N - 5];

	md_tenmul_dims(N - 5, mdims, odims + 5, idims + 5, kdims + 5);

		//clang
	const bart_dim_t* odimsp = odims;
	const bart_dim_t* idimsp = idims;
	const bart_dim_t* kdimsp = kdims;
	const bart_dim_t* dilationp = dilation;
	const bart_dim_t* stridesp = strides;

	NESTED(void, nary_zconvcorr_im2col, (struct nary_opt_data_s* data, void* ptr[]))
	{
		for (bart_dim_t i = 0; i < data->size; i++){

			complex float* imat_tmp = md_alloc_gpu(1, &imat_size, size);
			cuda_im2col(imat_tmp, (const complex float*)ptr[1] + i * isize, odimsp, idimsp, kdimsp, dilationp, stridesp);

			blas_matrix_zfmac(	M1, K1, N1,
						(complex float*)ptr[0] + i * ksize,
						(complex float*)ptr[2] + i * osize, 'N',
						imat_tmp, 'T'
						);
			md_free(imat_tmp);
		}
	};

	optimized_threeop_oii(N - 5, mdims, kstrs + 5, (void*)krn, istrs + 5, (void*)in, ostrs + 5, (void*)out,
				(size_t[3]){ (size_t)((bart_stride_t)CFL_SIZE * ksize), (size_t)((bart_stride_t)CFL_SIZE * isize), (size_t)((bart_stride_t)CFL_SIZE * osize) },
				nary_zconvcorr_im2col);

	debug_printf(DP_DEBUG3, "conv by %s \n", __func__);

	return true;
}
#endif


#ifdef USE_GPU
bool zconvcorr_bwd_in_im2col_cf_gpu(int N,
				bart_dim_t odims[N], bart_stride_t ostrs[N], const complex float* out,
				bart_dim_t idims[N], bart_stride_t istrs[N], complex float* in,
				bart_dim_t kdims[N], bart_stride_t kstrs[N], const complex float* krn,
				bart_flags_t flags, const bart_dim_t dilation[N], const bart_stride_t strides[N], bool conv)
{
	if (!cuda_ondevice(out))
		return false;

	if (5 > N)
		return false;

	if (!check_trivial_cf(5, odims, ostrs, idims, istrs, kdims, kstrs, flags, CFL_SIZE))
		return false;

	if (conv)
		return false;

#ifndef NON_DETERMINISTIC
	if ((NULL != dilation) && 1 != md_calc_size(N, dilation))
		return false;

	if ((NULL != strides) && 1 != md_calc_size(N, strides))
		return false;
#endif

	// mim2col dims (nr_out_channel | nr_in_channel, kx, ky, kz | outx, outy, outz)
	// kernel	(nr_filter | nr_in_channel, kx, ky, kz | 1, 1, 1 )
	// image	(1 | nr_in_channel, kx, ky, kz | outx, outy, outz | ... )
	// output 	(nr_filter | 1, 1, 1, 1 | outx, outy, outz)

	bart_dim_t osize = odims[0] * odims[1] * odims[2] * odims[3] * odims[4];
	bart_dim_t ksize = kdims[0] * kdims[1] * kdims[2] * kdims[3] * kdims[4];
	bart_dim_t isize = idims[0] * idims[1] * idims[2] * idims[3] * idims[4];

	bart_dim_t M1 = kdims[0];
	bart_dim_t K1 = kdims[1] * kdims[2] * kdims[3] * kdims[4];
	bart_dim_t N1 = odims[2] * odims[3] * odims[4];

	bart_dim_t imat_size = K1 * N1;

	bart_dim_t mdims[N - 5];

	md_tenmul_dims(N - 5, mdims, odims + 5, idims + 5, kdims + 5);

	//clang
	const bart_dim_t* odimsp = odims;
	const bart_dim_t* idimsp = idims;
	const bart_dim_t* kdimsp = kdims;
	const bart_dim_t* dilationp = dilation;
	const bart_dim_t* stridesp = strides;

	NESTED(void, nary_zconvcorr_im2col, (struct nary_opt_data_s* data, void* ptr[]))
	{
		for (bart_dim_t i = 0; i < data->size; i++){

			complex float* imat_tmp = md_alloc_gpu(1, &imat_size, CFL_SIZE);
			md_clear(1, &imat_size, imat_tmp, CFL_SIZE);

			blas_matrix_zfmac(	K1, N1, M1,
						imat_tmp,
						(complex float*)ptr[2] + i * ksize, 'T',
						(complex float*)ptr[1] + i * osize, 'N'
						);


			cuda_im2col_transp((complex float*)ptr[0] + i * isize, imat_tmp , odimsp, idimsp, kdimsp, dilationp, stridesp);

			md_free(imat_tmp);
		}
	};

	optimized_threeop_oii(N - 5, mdims, istrs + 5, (void*)in, ostrs + 5, (void*)out, kstrs + 5, (void*)krn,
				(size_t[3]){ (size_t)((bart_stride_t)CFL_SIZE * isize), (size_t)((bart_stride_t)CFL_SIZE * osize), (size_t)((bart_stride_t)CFL_SIZE * ksize) },
				nary_zconvcorr_im2col);

	debug_printf(DP_DEBUG3, "conv by %s \n", __func__);

	return true;
}
#endif

static void test_zconvcorr_fwd_ref(	int N,
					bart_dim_t odims[N], bart_stride_t ostrs[N], complex float* optr,
					bart_dim_t idims[N], bart_stride_t istrs[N], const complex float* iptr,
					bart_dim_t kdims[N], bart_stride_t kstrs[N], const complex float* kptr,
					bart_flags_t flags, const bart_dim_t dilation[N], const bart_stride_t strides[N], bool conv
				)
{
	bart_dim_t tdims[2 * N];
	bart_stride_t tostrs[2 * N];
	bart_stride_t tistrs[2 * N];
	bart_stride_t tkstrs[2 * N];

	int shift = calc_convcorr_geom_strs_dil(N, flags, tdims, tostrs, tkstrs, tistrs,
						odims, ostrs,
						kdims, kstrs,
						idims, istrs,
						dilation, strides, conv, false);

	deactivate_strided_vecops();
	md_zfmac2(2 * N, tdims, tostrs, optr, tistrs, iptr, tkstrs, kptr + shift);
	activate_strided_vecops();
}

static void test_zconvcorr_bwd_krn_ref(	int N,
					bart_dim_t odims[N], bart_stride_t ostrs[N], const complex float* optr,
					bart_dim_t idims[N], bart_stride_t istrs[N], const complex float* iptr,
					bart_dim_t kdims[N], bart_stride_t kstrs[N], complex float* kptr,
					bart_flags_t flags, const bart_dim_t dilation[N], const bart_stride_t strides[N], bool conv
				)
{
	bart_dim_t tdims[2 * N];
	bart_stride_t tostrs[2 * N];
	bart_stride_t tistrs[2 * N];
	bart_stride_t tkstrs[2 * N];

	int shift = calc_convcorr_geom_strs_dil(N, flags, tdims, tostrs, tkstrs, tistrs,
						odims, ostrs,
						kdims, kstrs,
						idims, istrs,
						dilation, strides, conv, false);

	deactivate_strided_vecops();
	md_zfmac2(2 * N, tdims, tkstrs, kptr + shift, tostrs, optr, tistrs, iptr);
	activate_strided_vecops();
}

static void test_zconvcorr_bwd_in_ref(	int N,
					bart_dim_t odims[N], bart_stride_t ostrs[N], const complex float* optr,
					bart_dim_t idims[N], bart_stride_t istrs[N], complex float* iptr,
					bart_dim_t kdims[N], bart_stride_t kstrs[N], const complex float* kptr,
					bart_flags_t flags, const bart_dim_t dilation[N], const bart_stride_t strides[N], bool conv
				)
{
	bart_dim_t tdims[2 * N];
	bart_stride_t tostrs[2 * N];
	bart_stride_t tistrs[2 * N];
	bart_stride_t tkstrs[2 * N];

	int shift = calc_convcorr_geom_strs_dil(N, flags, tdims, tostrs, tkstrs, tistrs,
						odims, ostrs,
						kdims, kstrs,
						idims, istrs,
						dilation, strides, conv, false);

	deactivate_strided_vecops();
	md_zfmac2(2 * N, tdims, tistrs, iptr, tostrs, optr, tkstrs, kptr + shift);
	activate_strided_vecops();
}


bool test_zconvcorr_fwd(	int N,
				bart_dim_t odims[N], bart_stride_t ostrs[N],
				bart_dim_t idims[N], bart_stride_t istrs[N],
				bart_dim_t kdims[N], bart_stride_t kstrs[N],
				bart_flags_t flags, const bart_dim_t dilation[N], const bart_stride_t strides[N], bool conv,
				float max_nrmse, bool gpu, bart_dim_t min_no_algos)
{
	bool result = true;

#ifdef USE_GPU
	void* ref_ptr = gpu ? md_alloc_gpu(1, MD_DIMS(1), CFL_SIZE) : md_alloc(1, MD_DIMS(1), CFL_SIZE);
#else
	assert(!gpu);
	void* ref_ptr = md_alloc(1, MD_DIMS(1), CFL_SIZE);
#endif

	complex float* optr_ref = md_alloc_sameplace(N, odims, CFL_SIZE, ref_ptr);
	complex float* optr_tst = md_alloc_sameplace(N, odims, CFL_SIZE, ref_ptr);
	complex float* optr_ini = md_alloc_sameplace(N, odims, CFL_SIZE, ref_ptr);

	complex float* iptr = md_alloc_sameplace(N, idims, CFL_SIZE, ref_ptr);
	complex float* kptr = md_alloc_sameplace(N, kdims, CFL_SIZE, ref_ptr);

	md_gaussian_rand(N, odims, optr_ini);
	md_gaussian_rand(N, idims, iptr);
	md_gaussian_rand(N, kdims, kptr);

	md_copy(N, odims, optr_ref, optr_ini, CFL_SIZE);

	test_zconvcorr_fwd_ref(N, odims, ostrs, optr_ref, idims, istrs, iptr, kdims, kstrs, kptr, flags, dilation, strides, conv);

	bart_dim_t counter = 0;

#ifdef USE_GPU
	int nr_algos = gpu ? ARRAY_SIZE(algos_fwd_gpu) : ARRAY_SIZE(algos_fwd_cpu);
#else
	int nr_algos = ARRAY_SIZE(algos_fwd_cpu);
#endif

	for(int i = 0; i < nr_algos; i++) {

#ifdef USE_GPU
		zconvcorr_fwd_algo_f* algo = gpu ? algos_fwd_gpu[i] : algos_fwd_cpu[i];
#else
		zconvcorr_fwd_algo_f* algo = algos_fwd_cpu[i];
#endif

		md_copy(N, odims, optr_tst, optr_ini, CFL_SIZE);

		if (algo(N, odims, ostrs, optr_tst, idims, istrs, iptr, kdims, kstrs, kptr, flags, dilation, strides, conv)) {

			float err = md_znrmse(N, odims, optr_ref, optr_tst);
			debug_printf((err >= max_nrmse) ? DP_WARN : DP_DEBUG1, "error zconvcorr_fwd algo %d: %.8f\n", i, err);

			counter += 1;
			result = result && (max_nrmse > err);
		}
	}

	md_free(optr_tst);
	md_free(optr_ini);
	md_free(optr_ref);


	md_free(iptr);
	md_free(kptr);

	md_free(ref_ptr);

	return result && (counter >= min_no_algos);
}

bool test_zconvcorr_bwd_in(	int N,
				bart_dim_t odims[N], bart_stride_t ostrs[N],
				bart_dim_t idims[N], bart_stride_t istrs[N],
				bart_dim_t kdims[N], bart_stride_t kstrs[N],
				bart_flags_t flags, const bart_dim_t dilation[N], const bart_stride_t strides[N], bool conv,
				float max_nrmse, bool gpu, bart_dim_t min_no_algos)
{
	bool result = true;

#ifdef USE_GPU
	void* ref_ptr = gpu ? md_alloc_gpu(1, MD_DIMS(1), CFL_SIZE) : md_alloc(1, MD_DIMS(1), CFL_SIZE);
#else
	assert(!gpu);
	void* ref_ptr = md_alloc(1, MD_DIMS(1), CFL_SIZE);
#endif

	complex float* iptr_ref = md_alloc_sameplace(N, idims, CFL_SIZE, ref_ptr);
	complex float* iptr_tst = md_alloc_sameplace(N, idims, CFL_SIZE, ref_ptr);
	complex float* iptr_ini = md_alloc_sameplace(N, idims, CFL_SIZE, ref_ptr);

	complex float* optr = md_alloc_sameplace(N, odims, CFL_SIZE, ref_ptr);
	complex float* kptr = md_alloc_sameplace(N, kdims, CFL_SIZE, ref_ptr);

	md_gaussian_rand(N, idims, iptr_ini);
	md_gaussian_rand(N, odims, optr);
	md_gaussian_rand(N, kdims, kptr);

	md_copy(N, idims, iptr_ref, iptr_ini, CFL_SIZE);

	test_zconvcorr_bwd_in_ref(N, odims, ostrs, optr, idims, istrs, iptr_ref, kdims, kstrs, kptr, flags, dilation, strides, conv);

	bart_dim_t counter = 0;

#ifdef USE_GPU
	int nr_algos = gpu ? ARRAY_SIZE(algos_bwd_in_gpu) : ARRAY_SIZE(algos_bwd_in_cpu);
#else
	int nr_algos = ARRAY_SIZE(algos_bwd_in_cpu);
#endif

	for(int i = 0; i < nr_algos; i++) {

#ifdef USE_GPU
		zconvcorr_bwd_in_algo_f* algo = gpu ? algos_bwd_in_gpu[i] : algos_bwd_in_cpu[i];
#else
		zconvcorr_bwd_in_algo_f* algo = algos_bwd_in_cpu[i];
#endif

		md_copy(N, idims, iptr_tst, iptr_ini, CFL_SIZE);

		if (algo(N, odims, ostrs, optr, idims, istrs, iptr_tst, kdims, kstrs, kptr, flags, dilation, strides, conv)) {

			float err = md_znrmse(N, idims, iptr_ref, iptr_tst);
			debug_printf((err >= max_nrmse) ? DP_WARN : DP_DEBUG1, "error zconvcorr_bwd_in algo %d: %.8f\n", i, err);

			counter += 1;
			result = result && (max_nrmse > err);
		}
	}

	md_free(iptr_tst);
	md_free(iptr_ini);
	md_free(iptr_ref);

	md_free(optr);
	md_free(kptr);

	md_free(ref_ptr);

	return result && (counter >= min_no_algos);
}

bool test_zconvcorr_bwd_krn(	int N,
				bart_dim_t odims[N], bart_stride_t ostrs[N],
				bart_dim_t idims[N], bart_stride_t istrs[N],
				bart_dim_t kdims[N], bart_stride_t kstrs[N],
				bart_flags_t flags, const bart_dim_t dilation[N], const bart_stride_t strides[N], bool conv,
				float max_nrmse, bool gpu, bart_dim_t min_no_algos)
{
	bool result = true;

#ifdef USE_GPU
	void* ref_ptr = gpu ? md_alloc_gpu(1, MD_DIMS(1), CFL_SIZE) : md_alloc(1, MD_DIMS(1), CFL_SIZE);
#else
	assert(!gpu);
	void* ref_ptr = md_alloc(1, MD_DIMS(1), CFL_SIZE);
#endif

	complex float* kptr_ref = md_alloc_sameplace(N, kdims, CFL_SIZE, ref_ptr);
	complex float* kptr_tst = md_alloc_sameplace(N, kdims, CFL_SIZE, ref_ptr);
	complex float* kptr_ini = md_alloc_sameplace(N, kdims, CFL_SIZE, ref_ptr);

	complex float* optr = md_alloc_sameplace(N, odims, CFL_SIZE, ref_ptr);
	complex float* iptr = md_alloc_sameplace(N, idims, CFL_SIZE, ref_ptr);

	md_gaussian_rand(N, kdims, kptr_ini);
	md_gaussian_rand(N, odims, optr);
	md_gaussian_rand(N, idims, iptr);

	md_copy(N, kdims, kptr_ref, kptr_ini, CFL_SIZE);

	test_zconvcorr_bwd_krn_ref(N, odims, ostrs, optr, idims, istrs, iptr, kdims, kstrs, kptr_ref, flags, dilation, strides, conv);

	bart_dim_t counter = 0;

#ifdef USE_GPU
	int nr_algos = gpu ? ARRAY_SIZE(algos_bwd_krn_gpu) : ARRAY_SIZE(algos_bwd_krn_cpu);
#else
	int nr_algos = ARRAY_SIZE(algos_bwd_krn_cpu);
#endif

	for(int i = 0; i < nr_algos; i++) {

#ifdef USE_GPU
		zconvcorr_bwd_krn_algo_f* algo = gpu ? algos_bwd_krn_gpu[i] : algos_bwd_krn_cpu[i];
#else
		zconvcorr_bwd_krn_algo_f* algo = algos_bwd_krn_cpu[i];
#endif

		md_copy(N, kdims, kptr_tst, kptr_ini, CFL_SIZE);

		if (algo(N, odims, ostrs, optr, idims, istrs, iptr, kdims, kstrs, kptr_tst, flags, dilation, strides, conv)) {

			float err = md_znrmse(N, kdims, kptr_ref, kptr_tst);
			debug_printf((err >= max_nrmse) ? DP_WARN : DP_DEBUG1, "error zconvcorr_bwd_krn algo %d: %.8f\n", i, err);

			counter += 1;
			result = result && (max_nrmse > err);
		}
	}

	md_free(kptr_tst);
	md_free(kptr_ini);
	md_free(kptr_ref);

	md_free(optr);
	md_free(iptr);

	md_free(ref_ptr);

	return result && (counter >= min_no_algos);
}
