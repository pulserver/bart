/* Copyright 2014. The Regents of the University of California.
 * Copyright 2017. Martin Uecker.
 * Copyright 2022-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013 Frank Ong
 * 2013-2022 Martin Uecker
 */

/*
 * md_*-based multi-dimensional wavelet implementation
 *
 * - 3 levels (1d, md, md-hierarchical)
 * - all higher-level code should work for GPU as well
 *
 * Bugs:
 *
 * - GPU version is not optimized
 * - memory use could possible be reduced
 *
 * Missing:
 *
 * - different boundary conditions
 *   (symmetric, periodic, zero)
 */

#include <stdio.h>
#include <complex.h>
#include <assert.h>
#include <limits.h>
#include <stdbool.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/ops.h"
#include "num/vptr_fun.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#include "wavelet/wl3-cuda.h"
#endif

#include "wavelet.h"


// layer 1 - 1-dimensional wavelet transform

static int bandsize(int imsize, int flen)
{
	return (imsize + flen - 1) / 2;
}

static complex float* access(const bart_stride_t str[3], complex float* x, bart_dim_t i, bart_dim_t j, bart_dim_t k)
{
	return (void*)x + str[2] * i + str[1] * j + str[0] * k;
}

static const complex float* caccess(const bart_stride_t str[3], const complex float* x, bart_dim_t i, bart_dim_t j, bart_dim_t k)
{
	return (const void*)x + str[2] * i + str[1] * j + str[0] * k;
}

/*
 * l: [3210]		(flen=4)
 * n:    [0123456789]	(dims[1])
 * j:   0 1 2 | | |	(bandsize)
 * */

static int coord(int j, int x, int flen, int l)
{
	int n = 2 * j + 1 - (flen - 1) + l;

	if (n < 0)
		n = -n - 1;

	if (n >= x)
		n = x - 1 - (n - x);

	return n;
}


static void wavelet_down3(const bart_dim_t dims[3], const bart_stride_t out_str[3], complex float* out, const bart_stride_t in_str[3], const complex float* in, int flen, const float filter[flen])
{
#pragma omp parallel for collapse(3)
	for (int i = 0; i < dims[2]; i++) {

		for (int j = 0; j < bandsize(dims[1], flen); j++) {

			for (int k = 0; k < dims[0]; k++) {

				*access(out_str, out, i, j, k) = 0.;

				for (int l = 0; l < flen; l++) {

					int n = coord(j, dims[1], flen, l);

					*access(out_str, out, i, j, k) +=
						*(caccess(in_str, in, i, n, k)) * filter[flen - l - 1];
				}
			}
		}
	}
}

static void wavelet_up3(const bart_dim_t dims[3], const bart_stride_t out_str[3], complex float* out, const bart_stride_t in_str[3],  const complex float* in, int flen, const float filter[flen])
{
//	md_clear2(3, dims, out_str, out, CFL_SIZE);

#pragma omp parallel for collapse(3)
	for (int i = 0; i < dims[2]; i++) {

		for (int n = 0; n < dims[1]; n++) {

			for (int k = 0; k < dims[0]; k++) {

		//		*access(out_str, out, i, j, k) = 0.;

				int odd = (n + 1) % 2;

				for (int l = odd; l < flen; l += 2) {

					int j = (n + l - 1) / 2;
#if 0
					assert(1 == (n + l) % 2);
					assert(n == coord(j, dims[1], flen, flen - l - 1));
#endif
					if ((j < 0) || (bandsize(dims[1], flen) <= j))
						continue;

					*access(out_str, out, i, n, k) +=
						*caccess(in_str, in, i, j, k) * filter[flen - l - 1];
				}
			}
		}
	}
}


void fwt1(int N, int d, const bart_dim_t dims[N], const bart_stride_t ostr[N], complex float* low, complex float* hgh, const bart_stride_t istr[N], const complex float* in, const bart_dim_t flen, const float filter[2][2][flen])
{
	debug_printf(DP_DEBUG4, "fwt1: %d/%d\n", d, N);
	debug_print_dims(DP_DEBUG4, N, dims);

	assert(dims[d] >= 2);

	bart_dim_t odims[N];
	md_copy_dims(N, odims, dims);
	odims[d] = bandsize(dims[d], flen);

	debug_print_dims(DP_DEBUG4, N, odims);

	bart_dim_t o = d + 1;
	bart_dim_t u = N - o;

	// 0 1 2 3 4 5 6|7
	// --d-- * --u--|N
	// ---o---

	assert(d == md_calc_blockdim(d, dims + 0, istr + 0, CFL_SIZE));
	assert(u == md_calc_blockdim(u, dims + o, istr + o, CFL_SIZE * (size_t)md_calc_size(o, dims)));

	assert(d == md_calc_blockdim(d, odims + 0, ostr + 0, CFL_SIZE));
	assert(u == md_calc_blockdim(u, odims + o, ostr + o, CFL_SIZE * (size_t)md_calc_size(o, odims)));

	// merge dims

	bart_dim_t wdims[3] = { md_calc_size(d, dims), dims[d], md_calc_size(u, dims + o) };
	bart_stride_t wistr[3] = { CFL_SIZE, istr[d], (bart_stride_t)CFL_SIZE * md_calc_size(o, dims) };
	bart_stride_t wostr[3] = { CFL_SIZE, ostr[d], (bart_stride_t)CFL_SIZE * md_calc_size(o, odims) };

#ifdef  USE_CUDA
	if (cuda_ondevice(in)) {

		assert(cuda_ondevice(low));
		assert(cuda_ondevice(hgh));

		float* flow = md_gpu_move(1, MD_DIMS(flen), filter[0][0], FL_SIZE);
		float* fhgh = md_gpu_move(1, MD_DIMS(flen), filter[0][1], FL_SIZE);

		wl3_cuda_down3(wdims, wostr, low, wistr, in, flen, flow);
		wl3_cuda_down3(wdims, wostr, hgh, wistr, in, flen, fhgh);

		md_free(flow);
		md_free(fhgh);
		return;
	}
#endif

	// no clear needed
	wavelet_down3(wdims, wostr, low, wistr, in, flen, filter[0][0]);
	wavelet_down3(wdims, wostr, hgh, wistr, in, flen, filter[0][1]);
}


void iwt1(int N, int d, const bart_dim_t dims[N], const bart_stride_t ostr[N], complex float* out, const bart_stride_t istr[N], const complex float* low, const complex float* hgh, const bart_dim_t flen, const float filter[2][2][flen])
{
	debug_printf(DP_DEBUG4, "ifwt1: %d/%d\n", d, N);
	debug_print_dims(DP_DEBUG4, N, dims);

	assert(dims[d] >= 2);

	bart_dim_t idims[N];
	md_copy_dims(N, idims, dims);
	idims[d] = bandsize(dims[d], flen);

	debug_print_dims(DP_DEBUG4, N, idims);

	int o = d + 1;
	int u = N - o;

	// 0 1 2 3 4 5 6|7
	// --d-- * --u--|N
	// ---o---

	assert(d == md_calc_blockdim(d, dims + 0, ostr + 0, CFL_SIZE));
	assert(u == md_calc_blockdim(u, dims + o, ostr + o, (size_t)((bart_stride_t)CFL_SIZE * md_calc_size(o, dims))));
	assert(d == md_calc_blockdim(d, idims + 0, istr + 0, CFL_SIZE));
	assert(u == md_calc_blockdim(u, idims + o, istr + o, (size_t)((bart_stride_t)CFL_SIZE * md_calc_size(o, idims))));

	bart_dim_t wdims[3] = { md_calc_size(d, dims), dims[d], md_calc_size(u, dims + o) };
	bart_stride_t wistr[3] = { CFL_SIZE, istr[d], (bart_stride_t)CFL_SIZE * md_calc_size(o, idims) };
	bart_stride_t wostr[3] = { CFL_SIZE, ostr[d], (bart_stride_t)CFL_SIZE * md_calc_size(o, dims) };

	md_clear(3, wdims, out, CFL_SIZE);	// we cannot clear because we merge outputs

#ifdef  USE_CUDA
	if (cuda_ondevice(out)) {

		assert(cuda_ondevice(low));
		assert(cuda_ondevice(hgh));

		float* flow = md_gpu_move(1, MD_DIMS(flen), filter[1][0], FL_SIZE);
		float* fhgh = md_gpu_move(1, MD_DIMS(flen), filter[1][1], FL_SIZE);

		wl3_cuda_up3(wdims, wostr, out, wistr, low, flen, flow);
		wl3_cuda_up3(wdims, wostr, out, wistr, hgh, flen, fhgh);

		md_free(flow);
		md_free(fhgh);
		return;
	}
#endif

	wavelet_up3(wdims, wostr, out, wistr, low, flen, filter[1][0]);
	wavelet_up3(wdims, wostr, out, wistr, hgh, flen, filter[1][1]);
}


// layer 2 - multi-dimensional wavelet transform

static void wavelet_dims_r(int N, int n, bart_flags_t flags, bart_dim_t odims[2 * N], const bart_dim_t dims[N], const bart_dim_t flen)
{
	if (MD_IS_SET(flags, n)) {

		odims[0 + n] = bandsize(dims[n], flen);
		odims[N + n] = 2;
	}

	if (n > 0)
		wavelet_dims_r(N, n - 1, flags, odims, dims, flen);
}

void wavelet_dims(int N, bart_flags_t flags, bart_dim_t odims[2 * N], const bart_dim_t dims[N], const bart_dim_t flen)
{
	md_copy_dims(N, odims, dims);
	md_singleton_dims(N, odims + N);

	wavelet_dims_r(N, N - 1, flags, odims, dims, flen);
}


void fwtN(int N, bart_flags_t flags, const bart_dim_t shifts[N], const bart_dim_t dims[N], const bart_stride_t ostr[2 * N], complex float* out, const bart_stride_t istr[N], const complex float* in, const bart_dim_t flen, const float filter[2][2][flen])
{
	bart_dim_t odims[2 * N];
	wavelet_dims(N, flags, odims, dims, flen);

	assert(md_calc_size(2 * N, odims) >= md_calc_size(N, dims));

	// FIXME one of these is unnecessary if we use the output

	complex float* tmpA = md_alloc_sameplace(2 * N, odims, CFL_SIZE, out);
	complex float* tmpB = md_alloc_sameplace(2 * N, odims, CFL_SIZE, out);

	bart_dim_t tidims[2 * N];
	md_copy_dims(N, tidims, dims);
	md_singleton_dims(N, tidims + N);

	bart_stride_t tistrs[2 * N];
	md_calc_strides(2 * N, tistrs, tidims, CFL_SIZE);

	bart_dim_t todims[2 * N];
	md_copy_dims(2 * N, todims, tidims);

	bart_stride_t tostrs[2 * N];

	// maybe we should push the randshift into lower levels

	//md_copy2(N, dims, tistrs, tmpA, istr, in, CFL_SIZE);
	md_circ_shift2(N, dims, shifts, tistrs, tmpA, istr, in, CFL_SIZE);

	for (int i = 0; i < N; i++) {

		if (MD_IS_SET(flags, i)) {

			todims[0 + i] = odims[0 + i];
			todims[N + i] = odims[N + i];

			md_calc_strides(2 * N, tostrs, todims, CFL_SIZE);

			fwt1(2 * N, i, tidims, tostrs, tmpB, (void*)tmpB + tostrs[N + i], tistrs, tmpA, flen, filter);

			md_copy_dims(2 * N, tidims, todims);
			md_copy_dims(2 * N, tistrs, tostrs);

			complex float* swap = tmpA;
			tmpA = tmpB;
			tmpB = swap;
		}
	}

	md_copy2(2 * N, todims, ostr, out, tostrs, tmpA, CFL_SIZE);

	md_free(tmpA);
	md_free(tmpB);
}


void iwtN(int N, bart_flags_t flags, const bart_dim_t shifts[N], const bart_dim_t dims[N], const bart_stride_t ostr[N], complex float* out, const bart_stride_t istr[2 * N], const complex float* in, const bart_dim_t flen, const float filter[2][2][flen])
{
	bart_dim_t idims[2 * N];
	wavelet_dims(N, flags, idims, dims, flen);

	assert(md_calc_size(2 * N, idims) >= md_calc_size(N, dims));

	complex float* tmpA = md_alloc_sameplace(2 * N, idims, CFL_SIZE, out);
	complex float* tmpB = md_alloc_sameplace(2 * N, idims, CFL_SIZE, out);

	bart_dim_t tidims[2 * N];
	md_copy_dims(2 * N, tidims, idims);

	bart_stride_t tistrs[2 * N];
	md_calc_strides(2 * N, tistrs, tidims, CFL_SIZE);

	bart_dim_t todims[2 * N];
	md_copy_dims(2 * N, todims, tidims);

	bart_stride_t tostrs[2 * N];

	bart_dim_t ishifts[N];
	for (int i = 0; i < N; i++)
		ishifts[i] = -shifts[i];

	md_copy2(2 * N, tidims, tistrs, tmpA, istr, in, CFL_SIZE);

	for (int i = N - 1; i >= 0; i--) {	// run backwards to maintain contiguous blocks

		if (MD_IS_SET(flags, i)) {

			todims[0 + i] = dims[0 + i];
			todims[N + i] = 1;

			md_calc_strides(2 * N, tostrs, todims, CFL_SIZE);

			iwt1(2 * N, i, todims, tostrs, tmpB, tistrs, tmpA, (void*)tmpA + tistrs[N + i], flen, filter);

			md_copy_dims(2 * N, tidims, todims);
			md_copy_dims(2 * N, tistrs, tostrs);

			complex float* swap = tmpA;
			tmpA = tmpB;
			tmpB = swap;
		}
	}

	//md_copy2(N, dims, ostr, out, tostrs, tmpA, CFL_SIZE);
	md_circ_shift2(N, dims, ishifts, ostr, out, tostrs, tmpA, CFL_SIZE);

	md_free(tmpA);
	md_free(tmpB);
}

// layer 3 - hierarchical multi-dimensional wavelet transform

static bart_flags_t wavelet_filter_flags(int N, bart_flags_t flags, const bart_dim_t dims[N], const bart_dim_t min[N])
{
	for (int i = 0; i < N; i++)
		if (dims[i] < min[i])	// CHECK
			flags = MD_CLEAR(flags, i);

	return flags;
}

int wavelet_num_levels(int N, bart_flags_t flags, const bart_dim_t dims[N], const bart_dim_t min[N], const bart_dim_t flen)
{
	if (0 == flags)
		return 1;

	bart_dim_t wdims[2 * N];
	wavelet_dims(N, flags, wdims, dims, flen);

	return 1 + wavelet_num_levels(N, wavelet_filter_flags(N, flags, wdims, min), wdims, min, flen);
}

static int wavelet_coeffs_r(int levels, int N, bart_flags_t flags, const bart_dim_t dims[N], const bart_dim_t min[N], const bart_dim_t flen)
{
	bart_dim_t wdims[2 * N];
	wavelet_dims(N, flags, wdims, dims, flen);

	bart_dim_t coeffs = md_calc_size(N, wdims);
	bart_dim_t bands = md_calc_size(N, wdims + N);

	assert((0 == flags) == (0 == levels));

	if (0 == flags)
		return bands * coeffs;

	return coeffs * (bands - 1) + wavelet_coeffs_r(levels - 1, N, wavelet_filter_flags(N, flags, wdims, min), wdims, min, flen);
}

bart_dim_t wavelet_coeffs(int N, bart_flags_t flags, const bart_dim_t dims[N], const bart_dim_t min[N], const bart_dim_t flen)
{
	int levels = wavelet_num_levels(N, flags, dims, min, flen);

	assert(levels > 0);

	return wavelet_coeffs_r(levels - 1, N, flags, dims, min, flen);
}





void wavelet_thresh(int N, float lambda, bart_flags_t flags, bart_flags_t jflags, const bart_dim_t shifts[N], const bart_dim_t dims[N], complex float* out, const complex float* in, const bart_dim_t minsize[N], bart_dim_t flen, const float filter[2][2][flen])
{
	assert(0 == (flags & jflags));

	bart_dim_t wdims[N];
	wavelet_coeffs2(N, flags, wdims, dims, minsize, flen);

	bart_stride_t wstr[N];
	md_calc_strides(N, wstr, wdims, CFL_SIZE);

	complex float* tmp = md_alloc_sameplace(N, wdims, CFL_SIZE, out);

	bart_stride_t str[N];
	md_calc_strides(N, str, dims, CFL_SIZE);

	fwt2(N, flags, shifts, wdims, wstr, tmp, dims, str, in, minsize, flen, filter);

	md_zsoftthresh(N, wdims, lambda, jflags, tmp, tmp);

	iwt2(N, flags, shifts, dims, str, out, wdims, wstr, tmp, minsize, flen, filter);

	md_free(tmp);
}


void wavelet_coeffs2(int N, bart_flags_t flags, bart_dim_t odims[N], const bart_dim_t dims[N], const bart_dim_t min[N], const bart_dim_t flen)
{
	md_select_dims(N, ~flags, odims, dims);

	if (0 == flags)
		return;

	int levels = wavelet_num_levels(N, flags, dims, min, flen);

	assert(levels > 0);

	bart_dim_t wdims[N];
	md_select_dims(N, flags, wdims, dims);	// remove unmodified dims

	int b = md_min_idx(flags);

	odims[b] = wavelet_coeffs_r(levels - 1, N, flags, wdims, min, flen);
}


static bool wavelet_check_dims(int N, bart_flags_t flags, const bart_dim_t dims[N], const bart_dim_t minsize[N])
{
	for (int i = 0; i < N; i++)
		if (MD_IS_SET(flags, i))
			if ((minsize[i] <= 2) || (dims[i] < minsize[i]))
				return false;

	return true;
}


static void embed(int N, bart_flags_t flags, bart_stride_t ostr[N], const bart_dim_t dims[N], const bart_stride_t str[N])
{
	int b = md_min_idx(flags);

	bart_dim_t dims1[N];
	md_select_dims(N, flags, dims1, dims);

	md_calc_strides(N, ostr, dims1, (size_t)str[b]);

	for (int i = 0; i < N; i++)
		if (!MD_IS_SET(flags, i))
			ostr[i] = str[i];
}


static void fwt2_int(int N, bart_flags_t flags, const bart_dim_t shifts[N], const bart_dim_t odims[N], const bart_stride_t ostr[N], complex float* out, const bart_dim_t idims[N], const bart_stride_t istr[N], const complex float* in, const bart_dim_t minsize[N], bart_dim_t flen, const float filter[2][2][flen])
{
	assert(wavelet_check_dims(N, flags, idims, minsize));

	if (0 == flags) {	// note: recursion does *not* end here

		assert(md_check_compat(N, 0u, odims, idims));

		md_copy2(N, idims, ostr, out, istr, in, CFL_SIZE);

		return;
	}

	// check output dimensions

	bart_dim_t odims2[N];
	wavelet_coeffs2(N, flags, odims2, idims, minsize, flen);

	assert(md_check_compat(N, 0u, odims2, odims));

	bart_dim_t wdims2[2 * N];
	wavelet_dims(N, flags, wdims2, idims, flen);

	// only consider transform dims...

	bart_dim_t dims1[N];
	md_select_dims(N, flags, dims1, idims);

	bart_dim_t wdims[2 * N];
	wavelet_dims(N, flags, wdims, dims1, flen);
	bart_dim_t level_coeffs = md_calc_size(2 * N, wdims);

	// ... which get embedded in dimension b

	int b = md_min_idx(flags);

	bart_stride_t ostr2[2 * N];
	md_calc_strides(2 * N, ostr2, wdims, (size_t)ostr[b]);

	// merge with original strides

	for (int i = 0; i < N; i++)
		if (!MD_IS_SET(flags, i))
			ostr2[i] = ostr[i];

	assert(odims[b] >= level_coeffs);

	bart_stride_t offset = (odims[b] - level_coeffs) * (ostr[b] / (bart_stride_t)CFL_SIZE);

	bart_dim_t bands = md_calc_size(N, wdims + N);
	bart_dim_t coeffs = md_calc_size(N, wdims + 0);

	debug_printf(DP_DEBUG4, "fwt2: flags:%" PRIu64 " lcoeffs:%" PRId64 " coeffs:%" PRId64 " (space:%" PRId64 ") bands:%" PRId64 " str:%" PRId64 " off:%" PRId64 "\n", flags, level_coeffs, coeffs, odims2[b], bands, ostr[b], offset / istr[b]);

	// subtract coefficients in high band

	odims2[b] -= (bands - 1) * coeffs;

	assert(odims2[b] > 0);

	bart_dim_t shifts0[N];
	for (int i = 0; i < N; i++)
		shifts0[i] = 0;

	bart_flags_t flags2 = wavelet_filter_flags(N, flags, wdims, minsize);

	assert((0 == offset) || (0u != flags2));

	fwtN(N, flags, shifts, idims, ostr2, out + offset, istr, in, flen, filter);

	if (0 != flags2) {

		bart_dim_t odims3[N];
		wavelet_coeffs2(N, flags2, odims3, wdims2, minsize, flen);

		bart_stride_t ostr3[N];
		embed(N, flags, ostr3, odims3, ostr);

		fwt2_int(N, flags2, shifts0, odims3, ostr3, out, wdims2, ostr2, out + offset, minsize, flen, filter);
	}
}


static void iwt2_int(int N, bart_flags_t flags, const bart_dim_t shifts[N], const bart_dim_t odims[N], const bart_stride_t ostr[N], complex float* out, const bart_dim_t idims[N], const bart_stride_t istr[N], const complex float* in, const bart_dim_t minsize[N], const bart_dim_t flen, const float filter[2][2][flen])
{
	assert(wavelet_check_dims(N, flags, odims, minsize));

	if (0 == flags) {	// note: recursion does *not* end here

		assert(md_check_compat(N, 0u, odims, idims));

		md_copy2(N, idims, ostr, out, istr, in, CFL_SIZE);

		return;
	}

	// check input dimensions

	bart_dim_t idims2[N];
	wavelet_coeffs2(N, flags, idims2, odims, minsize, flen);

	assert(md_check_compat(N, 0u, idims2, idims));

	bart_dim_t wdims2[2 * N];
	wavelet_dims(N, flags, wdims2, odims, flen);

	// only consider transform dims...

	bart_dim_t dims1[N];
	md_select_dims(N, flags, dims1, odims);

	bart_dim_t wdims[2 * N];
	wavelet_dims(N, flags, wdims, dims1, flen);
	bart_dim_t level_coeffs = md_calc_size(2 * N, wdims);

	// ... which get embedded in dimension b

	int b = md_min_idx(flags);

	bart_stride_t istr2[2 * N];
	md_calc_strides(2 * N, istr2, wdims, (size_t)istr[b]);

	// merge with original strides

	for (int i = 0; i < N; i++)
		if (!MD_IS_SET(flags, i))
			istr2[i] = istr[i];

	assert(idims[b] >= level_coeffs);

	bart_stride_t offset = (idims[b] - level_coeffs) * (istr[b] / (bart_stride_t)CFL_SIZE);

	bart_dim_t bands = md_calc_size(N, wdims + N);
	bart_dim_t coeffs = md_calc_size(N, wdims + 0);

	// subtract coefficients in high band

	idims2[b] -= (bands - 1) * coeffs;

	assert(idims2[b] > 0);

	debug_printf(DP_DEBUG4, "ifwt2: flags:%" PRIu64 " lcoeffs:%" PRId64 " coeffs:%" PRId64 " (space:%" PRId64 ") bands:%" PRId64 " str:%" PRId64 " off:%" PRId64 "\n", flags, level_coeffs, coeffs, idims2[b], bands, istr[b], offset / ostr[b]);

	// fix me we need temp storage
	complex float* tmp = md_alloc_sameplace(2 * N, wdims2, CFL_SIZE, out);

	bart_stride_t tstr[2 * N];
	md_calc_strides(2 * N, tstr, wdims2, CFL_SIZE);

	md_copy2(2 * N, wdims2, tstr, tmp, istr2, in + offset, CFL_SIZE);

	bart_dim_t shifts0[N];
	for (int i = 0; i < N; i++)
		shifts0[i] = 0;

	bart_flags_t flags2 = wavelet_filter_flags(N, flags, wdims, minsize);

	assert((0 == offset) || (0u != flags2));

	if (0u != flags2) {

		bart_dim_t idims3[N];
		wavelet_coeffs2(N, flags2, idims3, wdims2, minsize, flen);

		bart_stride_t istr3[N];
		embed(N, flags, istr3, idims3, istr);

		iwt2_int(N, flags2, shifts0, wdims2, tstr, tmp, idims3, istr3, in, minsize, flen, filter);
	}

	iwtN(N, flags, shifts, odims, ostr, out, tstr, tmp, flen, filter);

	md_free(tmp);
}

struct vptr_wt_s {

	vptr_fun_data_t super;
	bart_flags_t flags;
	bool backwards;

	int N;
	const bart_dim_t* shifts;
	const bart_dim_t* minsize;
	bart_dim_t flen;
	const void* filter;
};

DEF_TYPEID(vptr_wt_s);

static void vptr_wt_del(vptr_fun_data_t* _d)
{
	auto d = CAST_DOWN(vptr_wt_s, _d);

	xfree(d->shifts);
	xfree(d->minsize);
	xfree(d->filter);
}

static void wt2_wrap(vptr_fun_data_t* _data, int N, int D, const bart_dim_t* dims[N], const bart_stride_t* strs[N], void* args[N])
{
	auto d = CAST_DOWN(vptr_wt_s, _data);

	assert(D == d->N);

	bart_stride_t ostr[D];
	bart_stride_t istr[D];

	md_select_strides(D, md_nontriv_dims(D, dims[0]), ostr, strs[0]);
	md_select_strides(D, md_nontriv_dims(D, dims[1]), istr, strs[1]);

	const float (*pfilter)[2][2][d->flen] = d->filter;

	(d->backwards ? iwt2_int : fwt2_int)(D, d->flags, d->shifts, dims[0], ostr, args[0], dims[1], istr, args[1], d->minsize, d->flen, (*pfilter));
}


void fwt2(int N, bart_flags_t flags, const bart_dim_t shifts[N], const bart_dim_t odims[N], const bart_stride_t ostr[N], complex float* out, const bart_dim_t idims[N], const bart_stride_t istr[N], const complex float* in, const bart_dim_t minsize[N], bart_dim_t flen, const float filter[2][2][flen])
{
	PTR_ALLOC(struct vptr_wt_s, _d);
	SET_TYPEID(vptr_wt_s, _d);
	_d->super.del = vptr_wt_del;
	_d->N = N;
	_d->flags = flags;
	_d->backwards = false;
	_d->shifts = ARR_CLONE(bart_dim_t[N], shifts);
	_d->minsize = ARR_CLONE(bart_dim_t[N], minsize);
	_d->flen = flen;

	float (*pfilter)[2][2][flen] = TYPE_ALLOC(float[2][2][flen]);
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
			for (int k = 0; k < flen; k++)
				(*pfilter)[i][j][k] = filter[i][j][k];

	_d->filter = pfilter;

	assert(md_check_equal_dims(N, MD_SINGLETON_STRS(N), shifts, ~flags));
	//FIXME: minsize does not make sense for batch dim
	//assert(md_check_equal_dims(N, MD_SINGLETON_DIMS(N), minsize, ~flags));

	exec_vptr_zfun(wt2_wrap, CAST_UP(PTR_PASS(_d)), 2, N, ~flags & ~md_nontriv_dims(N, minsize), MD_BIT(0), MD_BIT(1), (const bart_dim_t*[2]) { odims, idims }, (const bart_dim_t*[2]) { ostr, istr }, (complex float*[2]) { out, (void*)in});
}

void iwt2(int N, bart_flags_t flags, const bart_dim_t shifts[N], const bart_dim_t odims[N], const bart_stride_t ostr[N], complex float* out, const bart_dim_t idims[N], const bart_stride_t istr[N], const complex float* in, const bart_dim_t minsize[N], bart_dim_t flen, const float filter[2][2][flen])
{
	PTR_ALLOC(struct vptr_wt_s, _d);
	SET_TYPEID(vptr_wt_s, _d);
	_d->super.del = vptr_wt_del;
	_d->N = N;
	_d->flags = flags;
	_d->backwards = true;
	_d->shifts = ARR_CLONE(bart_dim_t[N], shifts);
	_d->minsize = ARR_CLONE(bart_dim_t[N], minsize);
	_d->flen = flen;

	float (*pfilter)[2][2][flen] = TYPE_ALLOC(float[2][2][flen]);
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
			for (int k = 0; k < flen; k++)
				(*pfilter)[i][j][k] = filter[i][j][k];

	_d->filter = pfilter;

	assert(md_check_equal_dims(N, MD_SINGLETON_STRS(N), shifts, ~flags));
	//FIXME: minsize does not make sense for batch dim
	//assert(md_check_equal_dims(N, MD_SINGLETON_DIMS(N), minsize, ~flags));

	exec_vptr_zfun(wt2_wrap, CAST_UP(PTR_PASS(_d)), 2, N, ~flags & ~md_nontriv_dims(N, minsize), MD_BIT(0), MD_BIT(1), (const bart_dim_t*[2]) { odims, idims }, (const bart_dim_t*[2]) { ostr, istr }, (complex float*[2]) { out, (void*)in});
}





void fwt(int N, bart_flags_t flags, const bart_dim_t shifts[N], const bart_dim_t odims[N], complex float* out, const bart_dim_t idims[N], const complex float* in, const bart_dim_t minsize[N], bart_dim_t flen, const float filter[2][2][flen])
{
	fwt2(N, flags, shifts, odims, MD_STRIDES(N, odims, CFL_SIZE), out, idims, MD_STRIDES(N, idims, CFL_SIZE), in, minsize, flen, filter);
}


void iwt(int N, bart_flags_t flags, const bart_dim_t shifts[N], const bart_dim_t odims[N], complex float* out, const bart_dim_t idims[N], const complex float* in, const bart_dim_t minsize[N], const bart_dim_t flen, const float filter[2][2][flen])
{
	iwt2(N, flags, shifts, odims, MD_STRIDES(N, odims, CFL_SIZE), out, idims, MD_STRIDES(N, idims, CFL_SIZE), in, minsize, flen, filter);
}


// 1D Wavelet coefficients.
// The first dimension indexes along forward wavelet decomposition and reconstruction filters for fwt and iwt
// The second dimension indexes along low-pass and high-pass filters
// The third dimension is the number of filter taps.
const float wavelet_haar[2][2][2] = {
	{ { +0.7071067811865475, +0.7071067811865475 },
	  { -0.7071067811865475, +0.7071067811865475 }, },
	{ { +0.7071067811865475, +0.7071067811865475 },
	  { +0.7071067811865475, -0.7071067811865475 }, },
};

const float wavelet_dau2[2][2][4] = {
	{ { -0.1294095225512603, +0.2241438680420134, +0.8365163037378077, +0.4829629131445341 },
	  { -0.4829629131445341, +0.8365163037378077, -0.2241438680420134, -0.1294095225512603 }, },
	{ { +0.4829629131445341, +0.8365163037378077, +0.2241438680420134, -0.1294095225512603 },
	  { -0.1294095225512603, -0.2241438680420134, +0.8365163037378077, -0.4829629131445341 }, },
};

// Cohen-Daubechies-Feaveau wavelet
const float wavelet_cdf44[2][2][10] = {
	{ { +0.00000000000000000, +0.03782845550726404 , -0.023849465019556843, -0.11062440441843718 , +0.37740285561283066,
	    +0.85269867900889385, +0.37740285561283066 , -0.11062440441843718 , -0.023849465019556843, +0.03782845550726404 },
	  { +0.00000000000000000, -0.064538882628697058, +0.040689417609164058, +0.41809227322161724 , -0.7884856164055829,
	    +0.41809227322161724, +0.040689417609164058, -0.064538882628697058, +0.00000000000000000 , +0.00000000000000000 }, },
	{ { +0.00000000000000000, -0.064538882628697058, -0.040689417609164058, +0.41809227322161724 , +0.7884856164055829,
	    +0.41809227322161724, -0.040689417609164058, -0.064538882628697058, +0.000000000000000000, +0.00000000000000000 },
	  { +0.00000000000000000, -0.03782845550726404 , -0.023849465019556843, +0.11062440441843718 , +0.37740285561283066,
            -0.85269867900889385, +0.37740285561283066 , +0.11062440441843718 , -0.023849465019556843, -0.03782845550726404 }, },
};




