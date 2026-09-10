/* Copyright 2014-2016. The Regents of the University of California.
 * Copyright 2015-2021. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2023-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors: 
 * 2014-2018 Martin Uecker
 * 2014 Jonathan Tamir
 */

#include <stdio.h>
#include <assert.h>
#include <complex.h>
#include <math.h>
#include <string.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/init.h"
#include "num/ops_p.h"
#include "num/mdfft.h"
#include "num/fft.h"
#include "num/ode.h"
#include "num/filter.h"
#ifdef USE_GPU
#include "num/gpuops.h"
#endif
#include "num/mpi_ops.h"

#include "wavelet/wavthresh.h"

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/opts.h"

#define DIMS 8

static bool use_distributed_computing = false;

static void* bench_alloc(int D, unsigned long mpi_flags, const long dimensions[D], size_t size)
{
	if (use_distributed_computing) {
#ifdef USE_GPU
		if (bart_use_gpu)
			return md_alloc_gpu_mpi(D, mpi_flags, dimensions, size);
		else
#endif
			return md_alloc_mpi(D, mpi_flags, dimensions, size);
	
	} else {
#ifdef USE_GPU
		if (bart_use_gpu)
			return md_alloc_gpu(D, dimensions, size);
		else
#endif
			return md_alloc(D, dimensions, size);
	}
}

static double bench_timestamp(void)
{
	mpi_sync();
#ifdef USE_GPU
	if (bart_use_gpu)
		cuda_sync_device();
#endif
	return timestamp();
}

static double bench_generic_copy(long dims[DIMS])
{
	long strs[DIMS];

	md_calc_strides(DIMS, strs, dims, CFL_SIZE);

	complex float* x = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* y = md_alloc(DIMS, dims, CFL_SIZE);

	md_gaussian_rand(DIMS, dims, x);

	double tic = bench_timestamp();

	md_copy2(DIMS, dims, strs, y, strs, x, CFL_SIZE);

	double toc = bench_timestamp();

	md_free(x);
	md_free(y);

	return toc - tic;
}

static double bench_generic_circ_shift(long dims[DIMS], unsigned long mpi_flags, unsigned long shift_dim, long shift)
{
	long center[DIMS] = {};
	long strs[DIMS];

	md_calc_strides(DIMS, strs, dims, CFL_SIZE);
	center[shift_dim] = shift;

	complex float* x = bench_alloc(DIMS, mpi_flags, dims, CFL_SIZE);
	complex float* y = bench_alloc(DIMS, mpi_flags, dims, CFL_SIZE);

	md_gaussian_rand(DIMS, dims, x);

	double tic = bench_timestamp();

	md_circ_shift2(DIMS, dims, center, strs, y, strs, x, CFL_SIZE);

	double toc = bench_timestamp();

	md_free(x);
	md_free(y);

	return toc - tic;
}
	
static double bench_generic_matrix_multiply(long dims[DIMS])
{
	long dimsX[DIMS];
	long dimsY[DIMS];
	long dimsZ[DIMS];
#if 1
	md_select_dims(DIMS, 2 * 3 + 17, dimsX, dims);	// 1 110 1
	md_select_dims(DIMS, 2 * 6 + 17, dimsY, dims);	// 1 011 1
	md_select_dims(DIMS, 2 * 5 + 17, dimsZ, dims);	// 1 101 1
#else
	md_select_dims(DIMS, 2 * 5 + 17, dimsZ, dims);	// 1 101 1
	md_select_dims(DIMS, 2 * 3 + 17, dimsY, dims);	// 1 110 1
	md_select_dims(DIMS, 2 * 6 + 17, dimsX, dims);	// 1 011 1
#endif
	complex float* x = md_alloc(DIMS, dimsX, CFL_SIZE);
	complex float* y = md_alloc(DIMS, dimsY, CFL_SIZE);
	complex float* z = md_alloc(DIMS, dimsZ, CFL_SIZE);

	md_gaussian_rand(DIMS, dimsX, x);
	md_gaussian_rand(DIMS, dimsY, y);

	double tic = bench_timestamp();

	md_ztenmul(DIMS, dimsZ, z, dimsX, x, dimsY, y);

	double toc = bench_timestamp();


	md_free(x);
	md_free(y);
	md_free(z);

	return toc - tic;
}


static double bench_generic_add(long dims[DIMS], unsigned long flags, bool forloop)
{
	long dimsX[DIMS];
	long dimsY[DIMS];

	long dimsC[DIMS];

	md_select_dims(DIMS, flags, dimsX, dims);
	md_select_dims(DIMS, ~flags, dimsC, dims);
	md_select_dims(DIMS, ~0UL, dimsY, dims);

	long strsX[DIMS];
	long strsY[DIMS];

	md_calc_strides(DIMS, strsX, dimsX, CFL_SIZE);
	md_calc_strides(DIMS, strsY, dimsY, CFL_SIZE);

	complex float* x = md_alloc(DIMS, dimsX, CFL_SIZE);
	complex float* y = md_alloc(DIMS, dimsY, CFL_SIZE);

	md_gaussian_rand(DIMS, dimsX, x);
	md_gaussian_rand(DIMS, dimsY, y);

	long L = md_calc_size(DIMS, dimsC);
	long T = md_calc_size(DIMS, dimsX);

	double tic = bench_timestamp();

	if (forloop) {

		for (long i = 0; i < L; i++) {

			for (long j = 0; j < T; j++)
				y[i + j * L] += x[j];
		}

	} else {

		md_zaxpy2(DIMS, dims, strsY, y, 1., strsX, x);
	}

	double toc = bench_timestamp();


	md_free(x);
	md_free(y);

	return toc - tic;
}


static double bench_generic_sum(long dims[DIMS], unsigned long flags, bool forloop)
{
	long dimsX[DIMS];
	long dimsY[DIMS];
	long dimsC[DIMS];

	md_select_dims(DIMS, ~0UL, dimsX, dims);
	md_select_dims(DIMS, flags, dimsY, dims);
	md_select_dims(DIMS, ~flags, dimsC, dims);

	long strsX[DIMS];
	long strsY[DIMS];

	md_calc_strides(DIMS, strsX, dimsX, CFL_SIZE);
	md_calc_strides(DIMS, strsY, dimsY, CFL_SIZE);

	complex float* x = md_alloc(DIMS, dimsX, CFL_SIZE);
	complex float* y = md_alloc(DIMS, dimsY, CFL_SIZE);

	md_gaussian_rand(DIMS, dimsX, x);
	md_clear(DIMS, dimsY, y, CFL_SIZE);

	long L = md_calc_size(DIMS, dimsC);
	long T = md_calc_size(DIMS, dimsY);

	double tic = bench_timestamp();

	if (forloop) {

		for (long i = 0; i < L; i++) {

			for (long j = 0; j < T; j++)
				y[j] = y[j] + x[i + j * L];
		}

	} else {

		md_zaxpy2(DIMS, dims, strsY, y, 1., strsX, x);
	}

	double toc = bench_timestamp();


	md_free(x);
	md_free(y);

	return toc - tic;
}

static double bench_copy1(long scale)
{
	long dims[DIMS] = { 1, 128 * scale, 128 * scale, 1, 1, 16, 1, 16 };
	return bench_generic_copy(dims);
}

static double bench_copy2(long scale)
{
	long dims[DIMS] = { 262144 * scale, 16, 1, 1, 1, 1, 1, 1 };
	return bench_generic_copy(dims);
}


static double bench_circ_shift(long scale)
{
	long dims[DIMS] = { 1, 256 * scale, 256 * scale, 1, 1, 16, 1, 16 };
	unsigned long mpi_flags = MD_BIT(5);
	return bench_generic_circ_shift(dims, mpi_flags, 5, 1);
}


static double bench_matrix_mult(long scale)
{
	long dims[DIMS] = { 1, 256 * scale, 256 * scale, 256 * scale, 1, 1, 1, 1 };
	return bench_generic_matrix_multiply(dims);
}



static double bench_batch_matmul1(long scale)
{
	long dims[DIMS] = { 30000 * scale, 8, 8, 8, 1, 1, 1, 1 };
	return bench_generic_matrix_multiply(dims);
}



static double bench_batch_matmul2(long scale)
{
	long dims[DIMS] = { 1, 8, 8, 8, 30000 * scale, 1, 1, 1 };
	return bench_generic_matrix_multiply(dims);
}


static double bench_tall_matmul1(long scale)
{
	long dims[DIMS] = { 1, 8, 8, 100000 * scale, 1, 1, 1, 1 };
	return bench_generic_matrix_multiply(dims);
}


static double bench_tall_matmul2(long scale)
{
	long dims[DIMS] = { 1, 100000 * scale, 8, 8, 1, 1, 1, 1 };
	unsigned long mpi_flags = MD_BIT(3);
	long dimsX[DIMS];
	long dimsY[DIMS];
	long dimsZ[DIMS];

	md_select_dims(DIMS, 2 * 3 + 17, dimsX, dims);	// 1 110 1
	md_select_dims(DIMS, 2 * 6 + 17, dimsY, dims);	// 1 011 1
	md_select_dims(DIMS, 2 * 5 + 17, dimsZ, dims);	// 1 101 1

	complex float* x = bench_alloc(DIMS, mpi_flags, dimsX, CFL_SIZE);
	complex float* y = bench_alloc(DIMS, mpi_flags, dimsY, CFL_SIZE);
	complex float* z = bench_alloc(DIMS, mpi_flags, dimsZ, CFL_SIZE);

	md_gaussian_rand(DIMS, dimsX, x);
	md_gaussian_rand(DIMS, dimsY, y);

	double tic = bench_timestamp();

	md_ztenmul(DIMS, dimsZ, z, dimsX, x, dimsY, y);

	double toc = bench_timestamp();

	md_free(x);
	md_free(y);
	md_free(z);

	return toc - tic;
}


static double bench_add(long scale)
{
	long dims[DIMS] = { 65536 * scale, 1, 50 * scale, 1, 1, 1, 1, 1 };
	return bench_generic_add(dims, MD_BIT(2), false);
}

static double bench_addf(long scale)
{
	long dims[DIMS] = { 65536 * scale, 1, 50 * scale, 1, 1, 1, 1, 1 };
	return bench_generic_add(dims, MD_BIT(2), true);
}

static double bench_add2(long scale)
{
	long dims[DIMS] = { 50 * scale, 1, 65536 * scale, 1, 1, 1, 1, 1 };
	return bench_generic_add(dims, MD_BIT(0), false);
}

static double bench_sum2(long scale)
{
	long dims[DIMS] = { 50 * scale, 1, 65536 * scale, 1, 1, 1, 1, 1 };
	return bench_generic_sum(dims, MD_BIT(0), false);
}

static double bench_sum(long scale)
{
	long dims[DIMS] = { 65536 * scale, 1, 50 * scale, 1, 1, 1, 1, 1 };
	return bench_generic_sum(dims, MD_BIT(2), false);
}

static double bench_sumf(long scale)
{
	long dims[DIMS] = { 65536 * scale, 1, 50 * scale, 1, 1, 1, 1, 1 };
	return bench_generic_sum(dims, MD_BIT(2), true);
}


static double bench_zmul(long scale)
{
	long dimsx[DIMS] = { 256, 256, 1, 1, 90 * scale, 1, 1, 1 };
	long dimsy[DIMS] = { 256, 256, 1, 1,  1, 1, 1, 1 };
	long dimsz[DIMS] = {   1,   1, 1, 1, 90 * scale, 1, 1, 1 };

	complex float* x = md_alloc(DIMS, dimsx, CFL_SIZE);
	complex float* y = md_alloc(DIMS, dimsy, CFL_SIZE);
	complex float* z = md_alloc(DIMS, dimsz, CFL_SIZE);

	md_gaussian_rand(DIMS, dimsy, y);
	md_gaussian_rand(DIMS, dimsz, z);

	long strsx[DIMS];
	long strsy[DIMS];
	long strsz[DIMS];

	md_calc_strides(DIMS, strsx, dimsx, CFL_SIZE);
	md_calc_strides(DIMS, strsy, dimsy, CFL_SIZE);
	md_calc_strides(DIMS, strsz, dimsz, CFL_SIZE);

	double tic = bench_timestamp();

	md_zmul2(DIMS, dimsx, strsx, x, strsy, y, strsz, z);

	double toc = bench_timestamp();

	md_free(x);
	md_free(y);
	md_free(z);

	return toc - tic;
}


static double bench_transpose(long scale)
{
	long dims[DIMS] = { 2000 * scale, 2000 * scale, 1, 1, 1, 1, 1, 1 };

	complex float* x = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* y = md_alloc(DIMS, dims, CFL_SIZE);
	
	md_gaussian_rand(DIMS, dims, x);
	md_clear(DIMS, dims, y, CFL_SIZE);

	double tic = bench_timestamp();

	md_transpose(DIMS, 0, 1, dims, y, dims, x, CFL_SIZE);

	double toc = bench_timestamp();

	md_free(x);
	md_free(y);
	
	return toc - tic;
}


static double bench_transpose2(long scale)
{
	long dims[DIMS] = { 200 * scale, 200 * scale, 1, 1, 1, 16, 1, 1 };
	unsigned long mpi_flags = MD_BIT(1);

	complex float* x = bench_alloc(DIMS, mpi_flags, dims, CFL_SIZE);
	complex float* y = bench_alloc(DIMS, mpi_flags, dims, CFL_SIZE);
	
	md_gaussian_rand(DIMS, dims, x);
	md_clear(DIMS, dims, y, CFL_SIZE);

	double tic = bench_timestamp();

	md_transpose(DIMS, 0, 1, dims, y, dims, x, CFL_SIZE);

	double toc = bench_timestamp();

	md_free(x);
	md_free(y);
	
	return toc - tic;
}


static double bench_resize(long scale)
{
	long dimsX[DIMS] = { 2000 * scale, 1000 * scale, 1, 1, 1, 1, 1, 1 };
	long dimsY[DIMS] = { 1000 * scale, 2000 * scale, 1, 1, 1, 1, 1, 1 };

	complex float* x = md_alloc(DIMS, dimsX, CFL_SIZE);
	complex float* y = md_alloc(DIMS, dimsY, CFL_SIZE);

	md_gaussian_rand(DIMS, dimsX, x);
	md_clear(DIMS, dimsY, y, CFL_SIZE);

	double tic = bench_timestamp();

	md_resize(DIMS, dimsY, y, dimsX, x, CFL_SIZE);

	double toc = bench_timestamp();

	md_free(x);
	md_free(y);
	
	return toc - tic;
}


static double bench_norm(int s, long scale)
{
	long dims[DIMS] = { 256 * scale, 256 * scale, 1, 16, 1, 1, 1, 1 };
	complex float* x = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* y = md_alloc(DIMS, dims, CFL_SIZE);
	
	md_gaussian_rand(DIMS, dims, x);
	md_gaussian_rand(DIMS, dims, y);

	double tic = bench_timestamp();

	switch (s) {
	case 0:
		md_zscalar(DIMS, dims, x, y);
		break;
	case 1:
		md_zscalar_real(DIMS, dims, x, y);
		break;
	case 2:
		md_z1norm(DIMS, dims, x);
		break;
	}

	double toc = bench_timestamp();

	md_free(x);
	md_free(y);
	
	return toc - tic;
}

static double bench_zscalar(long scale)
{
	return bench_norm(0, scale);
}

static double bench_zscalar_real(long scale)
{
	return bench_norm(1, scale);
}

static double bench_zl1norm(long scale)
{
	return bench_norm(2, scale);
}

static double bench_znorm(long scale)
{
	complex float* x;
	complex float* y;
	long dims[DIMS] = { 256 * scale, 256 * scale, 1, 16, 1, 1, 1, 1 };
	unsigned long mpi_flags = MD_BIT(3);
	x = bench_alloc(DIMS, mpi_flags, dims, CFL_SIZE);
	y = bench_alloc(DIMS, mpi_flags, dims, CFL_SIZE);
	
	md_gaussian_rand(DIMS, dims, x);
	md_gaussian_rand(DIMS, dims, y);

	double tic = bench_timestamp();

	md_znorm(DIMS, dims, x);

	double toc = bench_timestamp();

	md_free(x);
	md_free(y);
	
	return toc - tic;
}

static double bench_wavelet(long scale)
{
	long dims[DIMS] = { 1, 256 * scale, 256 * scale, 1, 16, 1, 1, 1 };
	unsigned long mpi_flags = MD_BIT(4);
	long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
	minsize[0] = MIN(dims[0], 16);
	minsize[1] = MIN(dims[1], 16);
	minsize[2] = MIN(dims[2], 16);

	const struct operator_p_s* p = prox_wavelet_thresh_create(DIMS, dims, 6, 0u, WAVELET_DAU2, minsize, 1.1, true);

	complex float* x = bench_alloc(DIMS, mpi_flags, dims, CFL_SIZE);
	md_gaussian_rand(DIMS, dims, x);

	double tic = bench_timestamp();

	operator_p_apply(p, 0.98, DIMS, dims, x, DIMS, dims, x);

	double toc = bench_timestamp();

	md_free(x);
	operator_p_free(p);

	return toc - tic;
}


static double bench_generic_mdfft(long dims[DIMS], unsigned long flags, unsigned long mpi_flags)
{
	complex float* x = bench_alloc(DIMS, mpi_flags, dims, CFL_SIZE);
	complex float* y = bench_alloc(DIMS, mpi_flags, dims, CFL_SIZE);

	md_gaussian_rand(DIMS, dims, x);

	double tic = bench_timestamp();

	md_fft(DIMS, dims, flags, 0u, y, x);

	double toc = bench_timestamp();

	md_free(x);
	md_free(y);

	return toc - tic;
}

static double bench_mdfft(long scale)
{
	long dims[DIMS] = { 1, 128 * scale, 128 * scale, 1, 1, 4, 1, 4 };
	unsigned long mpi_flags = MD_BIT(5);
	return bench_generic_mdfft(dims, 6ul, mpi_flags);
}



static double bench_generic_fft(long dims[DIMS], unsigned long flags, unsigned long mpi_flags)
{
	complex float* x = bench_alloc(DIMS, mpi_flags, dims, CFL_SIZE);
	complex float* y = bench_alloc(DIMS, mpi_flags, dims, CFL_SIZE);

	md_gaussian_rand(DIMS, dims, x);

	double tic = bench_timestamp();

	fft(DIMS, dims, flags, y, x);

	double toc = bench_timestamp();

	md_free(x);
	md_free(y);

	return toc - tic;
}



static double bench_fft(long scale)
{
	long dims[DIMS] = { 1, 256 * scale, 256 * scale, 1, 1, 16, 1, 8 };
	unsigned long mpi_flags = MD_BIT(5);
	return bench_generic_fft(dims, 6ul, mpi_flags);
}




static double bench_generic_fftmod(long dims[DIMS], unsigned long flags)
{
	complex float* x = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* y = md_alloc(DIMS, dims, CFL_SIZE);

	md_gaussian_rand(DIMS, dims, x);

	double tic = bench_timestamp();

	fftmod(DIMS, dims, flags, y, x);

	double toc = bench_timestamp();

	md_free(x);
	md_free(y);

	return toc - tic;
}



static double bench_fftmod(long scale)
{
	long dims[DIMS] = { 1, 256 * scale, 256 * scale, 1, 1, 16, 1, 16 };
	return bench_generic_fftmod(dims, 6ul);
}


enum bench_typ { BENCH_ZFILL, BENCH_ZSMUL, BENCH_LINPHASE };

static double bench_generic_expand(enum bench_typ typ, long scale)
{
	long dims[DIMS] = { 1, 256 * scale, 256 * scale, 1, 1, 16, 1, 16 };
	unsigned long mpi_flags = MD_BIT(5);

	float linphase_pos[DIMS] = { 0.5, 0.1 };

	complex float* x = bench_alloc(DIMS, mpi_flags, dims, CFL_SIZE);

	double tic = bench_timestamp();

	switch (typ) {

	case BENCH_ZFILL:
		md_zfill(DIMS, dims, x, 1.);
		break;

	case BENCH_ZSMUL:
		md_zsmul(DIMS, dims, x, x, 1.);
		break;

	case BENCH_LINPHASE:

		linear_phase(DIMS, dims, linphase_pos, x);
		break;

	default:
		assert(0);
	}

	double toc = bench_timestamp();

	md_free(x);

	return toc - tic;
}


static double bench_zfill(long scale)
{
	return bench_generic_expand(BENCH_ZFILL, scale);
}

static double bench_zsmul(long scale)
{
	return bench_generic_expand(BENCH_ZSMUL, scale);
}

static double bench_linphase(long scale)
{
	return bench_generic_expand(BENCH_LINPHASE, scale);
}



static double bench_ode(long scale)
{
	float mat[2][2] = { { 0., +1. }, { -1., 0. } };

	float x[2] = { 1., 0. };
	float h = 10.;
	float tol = 1.E-6;

	double tic = bench_timestamp();

	ode_matrix_interval(h, tol, 2, x, 0., scale * 10001. * M_PI, mat);

	double err = pow(fabs(x[0] + 1.), 2.) + pow(fabs(x[1] - 0.), 2.);
	assert(err < 1.E-2);

	double toc = bench_timestamp();

	return toc - tic;
}



enum bench_indices { REPETITION_IND, SCALE_IND, THREADS_IND, TESTS_IND, BENCH_DIMS };

typedef double (*bench_fun)(long scale);

static void do_test(const long dims[BENCH_DIMS], complex float* out, long scale, bench_fun fun, const char* str)
{
	if (mpi_is_main_proc())
		printf("%30.30s |", str);
	
	int N = (int)dims[REPETITION_IND];
	double sum = 0.;
	double min = 1.E10;
	double max = 0.;

	for (int i = 0; i < N; i++) {

		double dt = fun(scale);
		sum += dt;
		min = MIN(dt, min);
		max = MAX(dt, max);

		if (mpi_is_main_proc()) {
			printf(" %3.5f", (float)dt);
			fflush(stdout);
		}

		assert(0 == REPETITION_IND);
		out[i] = dt;
	}

	if (mpi_is_main_proc())
		printf(" | Avg: %3.5f Max: %3.5f Min: %3.5f\n", (float)(sum / N), max, min); 
}


const struct benchmark_s {

	bench_fun fun;
	bool mpi_bench;
	bool gpu_bench;
	const char* str;

} benchmarks[] = {
	{ bench_add,		false, 	false,	"add (md_zaxpy)" },
	{ bench_add2,		false,	false,	"add (md_zaxpy), contiguous" },
	{ bench_addf,		false, 	false,	"add (for loop)" },
	{ bench_sum,   		false, 	false,	"sum (md_zaxpy)" },
	{ bench_sum2,   	false, 	false,	"sum (md_zaxpy), contiguous" },
	{ bench_sumf,   	false, 	false,	"sum (for loop)" },
	{ bench_zmul,   	false, 	false,	"complex mult. (md_zmul2)" },
	{ bench_transpose,	false, 	false,	"complex transpose 1" },
	{ bench_transpose2,	true, 	true,	"complex transpose 2" },
	{ bench_resize,   	false, 	false,	"complex resize" },
	{ bench_matrix_mult,	false, 	false,	"complex matrix multiply" },
	{ bench_batch_matmul1,	false, 	false,	"batch matrix multiply 1" },
	{ bench_batch_matmul2,	false, 	false,	"batch matrix multiply 2" },
	{ bench_tall_matmul1,	false, 	false,	"tall matrix multiply 1" },
	{ bench_tall_matmul2,	true, 	true,	"tall matrix multiply 2" },
	{ bench_zscalar,	false, 	false,	"complex dot product" },
	{ bench_zscalar,	false, 	false,	"complex dot product" },
	{ bench_zscalar_real,	false, 	false,	"real complex dot product" },
	{ bench_znorm,		true, 	false,	"l2 norm" },
	{ bench_zl1norm,	false, 	false,	"l1 norm" },
	{ bench_copy1,		false, 	false,	"copy 1" },
	{ bench_copy2,		false, 	false,	"copy 2" },
	{ bench_circ_shift,	true, 	true,	"circ shift" },
	{ bench_zfill,		true, 	true,	"complex fill" },
	{ bench_zsmul,		true, 	true,	"complex scalar multiplication" },
	{ bench_linphase,	false, 	false,	"linear phase" },
	{ bench_wavelet,	true, 	true,	"wavelet soft thresh" },
	{ bench_mdfft,		true, 	false,	"(MD-)FFT" },
	{ bench_fft,		false, 	true,	"FFT" },
	{ bench_fftmod,		false, 	false,	"fftmod" },
	{ bench_ode,		false, 	false,	"ODE" },
};


static const char help_str[] = "Performs a series of micro-benchmarks.";



int main_bench(int argc, char* argv[argc])
{
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_OUTFILE(false, &out_file, "output"),
	};

	bool threads = false;
	bool scaling = false;
	unsigned long flags = ~0UL;

	const struct opt_s opts[] = {
		OPT_SET('T', &threads, "varying number of threads"),
		OPT_SET('S', &scaling, "varying problem size"),
		OPT_ULONG('s', &flags, "flags", "select benchmarks"),
		OPT_SET('g', &bart_use_gpu,  "perform benchmark on GPU"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	long dims[BENCH_DIMS] = { [0 ... BENCH_DIMS - 1] = 1 };
	long strs[BENCH_DIMS];
	long pos[BENCH_DIMS] = { };

	dims[REPETITION_IND] = 5;
	dims[THREADS_IND] = threads ? 8 : 1;
	dims[SCALE_IND] = scaling ? 5 : 1;
	dims[TESTS_IND] = sizeof(benchmarks) / sizeof(benchmarks[0]);

	md_calc_strides(BENCH_DIMS, strs, dims, CFL_SIZE);

	bool outp = (NULL != out_file);
	complex float* out = (outp ? create_cfl : anon_cfl)(out_file, BENCH_DIMS, dims);

	if ((mpi_get_num_procs() > 1))
		 use_distributed_computing = true;

#ifdef USE_GPU
	num_init_gpu_support();
#else
	num_init();
	if (bart_use_gpu)
		error("Copmiled without GPU support!");
#endif

	debug_printf(DP_INFO, "Running benchmarks on %d process(es)\n", mpi_get_num_procs());

	md_clear(BENCH_DIMS, dims, out, CFL_SIZE);

	do {
		if (!(flags & MD_BIT(pos[TESTS_IND])))
			continue;

		if ((bart_use_gpu && !benchmarks[pos[TESTS_IND]].gpu_bench) ||
		    (use_distributed_computing && !benchmarks[pos[TESTS_IND]].mpi_bench))
			continue;

		if (threads) {

			num_set_num_threads((int)pos[THREADS_IND] + 1);
			debug_printf(DP_INFO, "%02ld threads. ", pos[THREADS_IND] + 1);
		}

		do_test(dims, &MD_ACCESS(BENCH_DIMS, strs, pos, out), pos[SCALE_IND] + 1,
			benchmarks[pos[TESTS_IND]].fun, benchmarks[pos[TESTS_IND]].str);

	} while (md_next(BENCH_DIMS, dims, ~MD_BIT(REPETITION_IND), pos));

	unmap_cfl(BENCH_DIMS, dims, out);

	return 0;
}
