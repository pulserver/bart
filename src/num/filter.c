/* Copyright 2015-2017. The Regents of the University of California.
 * Copyright 2016-2017. Martin Uecker.
 * Copyright 2023-2026. Institute of Biomedical Imaging. TU Graz.
 * Copyright 2026. Department of Radiology. Boston Children's Hospital.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2017 Martin Uecker
 * 2017 Jon Tamir
 * 2026 Moritz Blumenthal
 */

#include <assert.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/loop.h"

#include "misc/misc.h"
#include "misc/nested.h"

#include "filter.h"


static int cmp_float(const void* a, const void* b)
{
	float av = *(const float*)a;
	float bv = *(const float*)b;

	return (av > bv) - (av < bv);
}

static int cmp_complex_float(const void* a, const void* b) // gives sign for 0. (not 0)
{
	float av = cabsf(*(const complex float*)a);
	float bv = cabsf(*(const complex float*)b);

	return (av > bv) - (av < bv);
}

static void sort_floats(int N, float ar[N])
{
	qsort((void*)ar, (size_t)N, sizeof(float), cmp_float);
}

static void sort_complex_floats(int N, complex float ar[N])
{
	qsort((void*)ar, (size_t)N, sizeof(complex float), cmp_complex_float);
}

float median_float(int N, const float ar[N])
{
	float tmp[N]; // = { };
	memcpy(tmp, ar, sizeof tmp);

	sort_floats(N, tmp);

	return (1 == N % 2) ? tmp[(N - 1) / 2] : ((tmp[(N - 1) / 2 + 0] + tmp[(N - 1) / 2 + 1]) / 2.);
}

complex float median_complex_float(int N, const complex float ar[N])
{
	complex float (*tmp)[N] = xmalloc(sizeof *tmp);
	memcpy(*tmp, ar, sizeof *tmp);

	sort_complex_floats(N, *tmp);

	complex float result;

	if (1 == N % 2)
		result = (*tmp)[(N - 1) / 2];
	else
		result = ((*tmp)[(N - 1) / 2 + 0] + (*tmp)[(N - 1) / 2 + 1]) / 2.;

	free(tmp);

	return result;
}


static float vec_dist(int D, const float x[D], const float y[D])
{
	float sum = 0.;

	for (int i = 0; i < D; i++)
		sum += powf(x[i] - y[i], 2.);

	return sqrtf(sum);
}

void weiszfeld(int iter, int N, int D, float x[D], const float in[N][D])
{
	for (int i = 0; i < D; i++)
		x[i] = 0.;

	for (int l = 0; l < iter; l++) {

		float sum = 0;
		float d[N];

		for (int i = 0; i < N; i++) {

			d[i] = vec_dist(D, x, in[i]);

			if (0. == d[i])
				return;

			sum += 1. / d[i];
		}

		for (int i = 0; i < D; i++)
			x[i] = 0.;

		for (int i = 0; i < N; i++)
			for (int j = 0; j < D; j++)
				x[j] += in[i][j] / d[i];

		for (int i = 0; i < D; i++)
			x[i] /= sum;
	}
}

static complex float median_geometric_complex_float(int N, const complex float ar[N])
{
	complex float x;

	// Weiszfeld's algorithm
	weiszfeld(10, N, 2, *(float(*)[2])&x, *(float(*)[N][2])ar);

	return x;
}

void md_medianz2(int D, int M, const long dim[D], const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr)
{
	assert(M < D);
	const long* nstr[2] = { ostr, istr };
	void* nptr[2] = { optr, (void*)iptr };

        long length = dim[M];
        long stride = istr[M];

	long dim2[D];
	md_select_dims(D, ~(1u << M), dim2, dim);

	NESTED(void, nary_medianz, (void* ptr[]))
	{
		complex float tmp[length];

		for (long i = 0; i < length; i++)
			tmp[i] = *((complex float*)(ptr[1] + i * stride));

		*(complex float*)ptr[0] = median_complex_float(length, tmp);
	};

	md_nary(2, D, dim2, nstr, nptr, nary_medianz);
}

void md_medianz(int D, int M, const long dim[D], complex float* optr, const complex float* iptr)
{
	assert(M < D);

	long dim2[D];
	md_select_dims(D, ~(1u << M), dim2, dim);

	long istr[D];
	long ostr[D];

	md_calc_strides(D, istr, dim, CFL_SIZE);
	md_calc_strides(D, ostr, dim2, CFL_SIZE);

	md_medianz2(D, M, dim, ostr, optr, istr, iptr);
}

void md_geometric_medianz2(int D, int M, const long dim[D], const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr)
{
	assert(M < D);
	const long* nstr[2] = { ostr, istr };
	void* nptr[2] = { optr, (void*)iptr };

	long dim2[D];
        long length = dim[M];
	long stride = istr[M];

	md_select_dims(D, ~(1u << M), dim2, dim);

	NESTED(void, nary_medianz, (void* ptr[]))
	{
		complex float tmp[length];

		for (long i = 0; i < length; i++)
			tmp[i] = *((complex float*)(ptr[1] + i * stride));

		*(complex float*)ptr[0] = median_geometric_complex_float(length, tmp);
	};

	md_nary(2, D, dim2, nstr, nptr, nary_medianz);
}

void md_geometric_medianz(int D, int M, const long dim[D], complex float* optr, const complex float* iptr)
{
	assert(M < D);

	long dim2[D];

	md_select_dims(D, ~(1u << M), dim2, dim);

	long istr[D];
	long ostr[D];

	md_calc_strides(D, istr, dim, CFL_SIZE);
	md_calc_strides(D, ostr, dim2, CFL_SIZE);

	md_medianz2(D, M, dim, ostr, optr, istr, iptr);
}


void md_moving_avgz2(int D, int M, const long dim[D], const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr)
{
	assert(M < D);
	assert(0 == ostr[M]);

	md_zavg2(D, dim, (1u << M), ostr, optr, istr, iptr);
}

void md_moving_avgz(int D, int M, const long dim[D], complex float* optr, const complex float* iptr)
{
	assert(M < D);

	long dim2[D];
	md_copy_dims(D, dim2, dim);

	dim2[M] = 1;

	long istr[D];
	long ostr[D];

	md_calc_strides(D, istr, dim, CFL_SIZE);
	md_calc_strides(D, ostr, dim2, CFL_SIZE);

	md_moving_avgz2(D, M, dim, ostr, optr, istr, iptr);
}




void centered_gradient(int N, const long dims[N], const complex float grad[N], complex float* out)
{
	md_zgradient(N, dims, out, grad);

	complex float cn = 0.;

	for (int n = 0; n < N; n++)
		 cn -= grad[n] * (float)(dims[n] / 2);

	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	md_zsadd2(N, dims, strs, out, strs, out, cn);
}

void linear_phase(int N, const long dims[N], const float pos[N], complex float* out)
{
	complex float grad[N];

	for (int n = 0; n < N; n++)
		grad[n] = 2. * M_PI * pos[n] / ((float)dims[n]);

	centered_gradient(N, dims, grad, out);
	md_zexpj(N, dims, out, out);
}


void klaplace_scaled(int N, const long dims[N], unsigned long flags, const float sc[N], complex float* out)
{
	unsigned long flags2 = flags;

	complex float* tmp = md_alloc_sameplace(N, dims, CFL_SIZE, out);

	md_clear(N, dims, out, CFL_SIZE);

	for (int i = 0; i < bitcount(flags); i++) {

		int lsb = ffs(flags2) - 1;
		flags2 = MD_CLEAR(flags2, lsb);

		complex float grad[N];
		for (int j = 0; j < N; j++)
			grad[j] = 0.;

		grad[lsb] = sc[lsb];
		centered_gradient(N, dims, grad, tmp);
		md_zspow(N, dims, tmp, tmp, 2.);
		md_zadd(N, dims, out, out, tmp);
	}

	md_free(tmp);
}



/* Definition of klaplace corresponds to conventions of continuous Fourier
 * transform (up to a sign), i.e. when used as filter by
 * ifftuc(klaplace * fftuc(x)), the resulting laplacian is scaled by a
 * factor of -(2pi)^2 assuming voxel spacing of dx=1 (c.f. test_klaplace_filter)
 * Usually, it needs to be rescaled for physical units. */

void klaplace(int N, const long dims[N], unsigned long flags, complex float* out)
{
	float sc[N];
	for (int j = 0; j < N; j++)
		sc[j] = 1.;

	klaplace_scaled(N, dims, flags, sc, out);
	md_zsmul(N, dims, out, out, -4. * M_PI * M_PI);
}



static void nary_zwindow(const long N, const float alpha, const float beta, complex float* ptr)
{
	if (1 == N) {

		ptr[0] = 1.;
		return;
	}

#pragma omp parallel for
	for (long i = 0; i < N; i++)
		ptr[i] = alpha - beta * cosf(2. * M_PI * (double)i / (double)(N - 1));
}

static void nary_zhamming(const long N, complex float* ptr)
{
#if 0
	const float alpha = 0.53836;
	const float beta = 0.46164;
#else
	const float alpha = 0.54;
	const float beta = 0.46;
#endif

	return nary_zwindow(N, alpha, beta, ptr);

}

static void nary_zhann(const long N, complex float* ptr)
{
	const float alpha = 0.5;
	const float beta = 0.5;

	return nary_zwindow(N, alpha, beta, ptr);
}

enum window_type { WINDOW_HAMMING, WINDOW_HANN };

static void md_zwindow2(int D, const long dims[D], unsigned long flags, const long ostrs[D], complex float* optr, const long istrs[D], const complex float* iptr, enum window_type wt)
{
	if (0 == flags) {

		md_copy2(D, dims, ostrs, optr, istrs, iptr, CFL_SIZE);
		return;
	}

	// process first flagged dimension

	int lsb = ffs(flags) - 1;

	long win_dims[D];
	long win_strs[D];

	md_select_dims(D, MD_BIT(lsb), win_dims, dims);
	md_calc_strides(D, win_strs, win_dims, CFL_SIZE);

	complex float* win = md_alloc_sameplace(D, win_dims, CFL_SIZE, iptr);

	switch (wt) {
	case WINDOW_HAMMING: nary_zhamming(dims[lsb], win); break;
	case WINDOW_HANN: nary_zhann(dims[lsb], win); break;
	};

	md_zmul2(D, dims, ostrs, optr, istrs, iptr, win_strs, win);

	md_free(win);

	flags = MD_CLEAR(flags, lsb);

	// process other dimensions

	if (0 != flags)
		md_zwindow2(D, dims, flags, ostrs, optr, ostrs, optr, wt);

	return;
}


#if 0
static void md_zwindow(const int D, const long dims[D], const long flags, complex float* optr, const complex float* iptr, bool hamming)
{
	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	md_zwindow2(D, dims, flags, strs, optr, strs, iptr, hamming);
}
#endif


/*
 * Apply Hamming window to iptr along flags
 */
void md_zhamming(int D, const long dims[D], const unsigned long flags, complex float* optr, const complex float* iptr)
{
	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	return md_zhamming2(D, dims, flags, strs, optr, strs, iptr);
}


/*
 * Apply Hamming window to iptr along flags (with strides)
 */
void md_zhamming2(int D, const long dims[D], const unsigned long flags, const long ostrs[D], complex float* optr, const long istrs[D], const complex float* iptr)
{
	return md_zwindow2(D, dims, flags, ostrs, optr, istrs, iptr, WINDOW_HAMMING);
}


/*
 * Apply Hann window to iptr along flags
 */
void md_zhann(int D, const long dims[D], const unsigned long flags, complex float* optr, const complex float* iptr)
{
	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	return md_zhann2(D, dims, flags, strs, optr, strs, iptr);
}


/*
 * Apply Hann window to iptr along flags (with strides)
 */
void md_zhann2(int D, const long dims[D], const unsigned long flags, const long ostrs[D], complex float* optr, const long istrs[D], const complex float* iptr)
{
	return md_zwindow2(D, dims, flags, ostrs, optr, istrs, iptr, WINDOW_HANN);
}



void md_zsample_filter(int N, const long dims[N], unsigned long flags, const float resolution[N], complex float* z, sample_filter_fun fun, bool centered)
{
	if (NULL == resolution) {

		float resolution[N];
		for (int i = 0; i < N; i++)
			resolution[i] = 1.;

		md_zsample_filter(N, dims, flags, resolution, z, fun, centered);
		return;
	}

	float scale[N];
	for (int i = 0; i < N; i++)
		scale[i] = MD_IS_SET(flags, i) ? 2 * M_PI / (dims[i] * resolution[i]) : 0.;

	const long* dimsp = dims;	// because of clang
	const float* scalep = scale;	// because of clang

	NESTED(complex float, filter_kernel, (const long pos[]))
	{
		float kpos[N];
		for (int i = 0; i < N; i++) {

			if (centered)
				kpos[i] = (pos[i] - (dimsp[i] / 2)) * scalep[i];
			else
				kpos[i] = (pos[i] - (pos[i] > dimsp[i] / 2 ? dimsp[i] : 0)) * scalep[i];
		}

		complex float val = NESTED_CALL(fun, (pos, kpos));

		return val;
	};

	md_parallel_zsample(N, dims, z, filter_kernel);
}

void klaplace_fd_scaled_uncentered(int N, const long dims[N], const float scale[N], complex float* z)
{
	const float* scalep = scale;	// because of clang

	NESTED(complex float, filter_kernel_laplace, (const long /*pos*/[], const float kpos[]))
	{
		complex float val = 0;

		for (int i = 0; i < N; i++)
			val += scalep[i] * (cexpf(-1.i * kpos[i]) + cexpf(1.i * kpos[i]) - 2.);

		return val;
	};

	md_zsample_filter(N, dims, ~0UL, NULL, z, CLOSURE(sample_filter_fun, filter_kernel_laplace), false);
}

void klaplace_fd_uncentered(int N, const long dims[N], complex float* z)
{
	float scale[N];
	for (int i = 0; i < N; i++)
		scale[i] = 1.;

	klaplace_fd_scaled_uncentered(N, dims, scale, z);
}




