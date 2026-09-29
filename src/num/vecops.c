/* Copyright 2013-2018. The Regents of the University of California.
 * Copyright 2016-2019. Martin Uecker.
 * Copyright 2017. University of Oxford.
 * Copyright 2026. Graz University of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2011-2017 Martin Uecker
 * 2014 Frank Ong
 * 2014-2018 Jon Tamir
 * 2017 Sofia Dimoudi
 *
 *
 * This file defines basic operations on vectors of floats/complex floats
 * for operations on the CPU which are are used by higher level code
 * (mainly num/flpmath.c and num/italgos.c) to implement more complex
 * operations. The functions are exported by pointers stored in the
 * global variable cpu_ops of type struct vec_ops. Identical functions
 * are implemented for the GPU in gpukrnls.cu.
 */

#include <assert.h>
#include <math.h>
#include <complex.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "num/rand.h"
#include "num/vec_iter.h"

#include "vecops.h"



/**
 * Allocate memory for array of floats.
 * Note: be sure to pass 2*N if allocating for complex float
 *
 * @param N number of elements
 */
static float* allocate(bart_dim_t N)
{
	assert(N >= 0);
	return xmalloc(sizeof(float[N]));
}

static void del(float* vec)
{
	xfree(vec);
}

static void copy(bart_dim_t N, float* dst, const float* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = src[i];
}

static void float2double(bart_dim_t N, double* dst, const float* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = src[i];
}

static void double2float(bart_dim_t N, float* dst, const double* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = src[i];
}

/*
 * Set vector to all-zeros
 *
 * @param N vector length
 * @param vec vector
 */
static void clear(bart_dim_t N, float* vec)
{
	for (bart_dim_t i = 0; i < N; i++)
		vec[i] = 0.;
}

static void sadd(bart_dim_t N, float* vec, float src)
{
	for (bart_dim_t i = 0; i < N; i++)
		vec[i] += src;
}

static double dot(bart_dim_t N, const float* vec1, const float* vec2)
{
	double res = 0.;

	for (bart_dim_t i = 0; i < N; i++)
		res += vec1[i] * vec2[i];
	//res = fma((double)vec1[i], (double)vec2[i], res);

	return res;
}

static complex double zdot(bart_dim_t N, const complex float* vec1, const complex float* vec2)
{
	complex double res = 0.;

	for (bart_dim_t i = 0; i < N; i++)
		res += vec1[i] * conjf(vec2[i]);

	return res;
}

/**
 * Compute l2 norm of vec
 *
 * @param N vector length
 * @param vec vector
 */
static double norm(bart_dim_t N, const float* vec)
{
	double res = 0.;

	for (bart_dim_t i = 0; i < N; i++)
		res += vec[i] * vec[i];
	//res = fma((double)vec[i], (double)vec[i], res);

	return sqrt(res);
}


/**
 * Compute l1 norm of vec
 *
 * @param N vector length
 * @param vec vector
 */
static double asum(bart_dim_t N, const float* vec)
{
	double res = 0.;

	for (bart_dim_t i = 0; i < N; i++)
		res += fabsf(vec[i]);

	return res;
}


/**
 * Compute l1 norm of complex vec
 *
 * @param N vector length
 * @param vec vector
 */
static double zl1norm(bart_dim_t N, const complex float* vec)
{
	double res = 0.;

	for (bart_dim_t i = 0; i < N; i++)
		res += cabsf(vec[i]);

	return res;
}



// we should probably replace asum and zl1norm
static void zsum(bart_dim_t N, complex float* vec)
{
	complex float res = 0.;

	for (bart_dim_t i = 0; i < N; i++)
		res += vec[i];

	vec[0] = res;
}



static void axpbz(bart_dim_t N, float* dst, const float a1, const float* src1, const float a2, const float* src2)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = a1 * src1[i] + a2 * src2[i];
}

static void axpy(bart_dim_t N, float* dst, float alpha, const float* src)
{
	axpbz(N, dst, 1., dst, alpha, src);
	//dst[i] = fmaf(alpha, src[i], dst[i]);
}

static void xpay(bart_dim_t N, float beta, float* dst, const float* src)
{
	axpbz(N, dst, beta, dst, 1., src);
	//dst[i] = fmaf(beta, dst[i], src[i]);
}


static void smul(bart_dim_t N, float alpha, float* dst, const float* src)
{
	axpbz(N, dst, 0., src, alpha, src);
	//dst[i] = fmaf(alpha, src[i], 0.f);
}

static void add(bart_dim_t N, float* dst, const float* src1, const float* src2)
{
#if 1
	if (dst == src1) {

		for (bart_dim_t i = 0; i < N; i++)
			dst[i] += src2[i];
	} else
#endif
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = src1[i] + src2[i];
}

static void sadd_update(bart_dim_t N, float val, float* dst, const float* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = src[i] + val;
}

static void zsadd(bart_dim_t N, complex float val, complex float* dst, const complex float* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = src[i] + val;
}

static void sub(bart_dim_t N, float* dst, const float* src1, const float* src2)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = src1[i] - src2[i];
}

static void mul(bart_dim_t N, float* dst, const float* src1, const float* src2)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = src1[i] * src2[i];
}

static void vec_div(bart_dim_t N, float* dst, const float* src1, const float* src2)
{
	for (bart_dim_t i = 0; i < N; i++)
		//dst[i] = src1[i] / src2[i];
		dst[i] = (src2[i] == 0) ? 0.f : src1[i] / src2[i];
}

static void sdiv(bart_dim_t N, float* dst, float src1, const float* src2)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = (src2[i] == 0) ? 0.f : src1 / src2[i];
}

static void fmac(bart_dim_t N, float* dst, const float* src1, const float* src2)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] += src1[i] * src2[i];
	//dst[i] = fmaf(src1[i], src2[i], dst[i]);
}

static void fmacD(bart_dim_t N, double* dst, const float* src1, const float* src2)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] += src1[i] * src2[i];
}

static void zsmul(bart_dim_t N, complex float val, complex float* dst, const complex float* src1)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = src1[i] * val;
}

static void zmul(bart_dim_t N, complex float* dst, const complex float* src1, const complex float* src2)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = src1[i] * src2[i];
}

static void zdiv(bart_dim_t N, complex float* dst, const complex float* src1, const complex float* src2)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = (src2[i] == 0.) ? 0. : (src1[i] / src2[i]);
}

static void zpow(bart_dim_t N, complex float* dst, const complex float* src1, const complex float* src2)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = cpowf(src1[i], src2[i]);
}

static void zfmac(bart_dim_t N, complex float* dst, const complex float* src1, const complex float* src2)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] += src1[i] * src2[i];
}

static void zfmacD(bart_dim_t N, complex double* dst, const complex float* src1, const complex float* src2)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] += src1[i] * src2[i];
}

static void zmulc(bart_dim_t N, complex float* dst, const complex float* src1, const complex float* src2)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = src1[i] * conjf(src2[i]);
}

static void zfmacc(bart_dim_t N, complex float* dst, const complex float* src1, const complex float* src2)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] += src1[i] * conjf(src2[i]);
}

static void zfmaccD(bart_dim_t N, complex double* dst, const complex float* src1, const complex float* src2)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] += src1[i] * conjf(src2[i]);
}

static void zfsq2(bart_dim_t N, complex float* dst, const complex float* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] += crealf(src[i]) * crealf(src[i]) + cimagf(src[i]) * cimagf(src[i]);
}


static void zconj(bart_dim_t N, complex float* dst, const complex float* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = conjf(src[i]);
}

static void zcmp(bart_dim_t N, complex float* dst, const complex float* src1, const complex float* src2)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = (src1[i] == src2[i]) ? 1. : 0.;
}

static void zdiv_reg(bart_dim_t N, complex float* dst, const complex float* src1, const complex float* src2, complex float lambda)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = (src2[i] == 0) ? 0.f : src1[i] / (lambda + src2[i]);
}

static void zphsr(bart_dim_t N, complex float* dst, const complex float* src)
{
	for (bart_dim_t i = 0; i < N; i++) {

		float s = cabsf(src[i]);

		/* Note: the comparison (0 == src[i]) is not enough with `--fast-math`
		 * with gcc 4.4.3 (but seems to work for 4.7.3, different computer)
		 * Test:
		 * complex float a = FLT_MIN;
		 * complex float c = a / cabsf(a);
		 * assert(!(isnan(creal(c)) || isnan(cimag(c))));
		 */

		dst[i] = (0. == s) ? 1. : (src[i] / s);
	}
}

static void zexp(bart_dim_t N, complex float* dst, const complex float* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = cexpf(src[i]);
}

static void zexpj(bart_dim_t N, complex float* dst, const complex float* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = cexpf(1.I * src[i]);
}

static void zlog(bart_dim_t N, complex float* dst, const complex float* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = (src[i] == (complex float)0.) ? 0. : clogf(src[i]);
}

static void vec_exp(bart_dim_t N, float* dst, const float* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = expf(src[i]);
}

static void vec_log(bart_dim_t N, float* dst, const float* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = (src[i] == 0.) ? 0. : logf(src[i]);
}

static void zarg(bart_dim_t N, complex float* dst, const complex float* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = cargf(src[i]);
}

static void zabs(bart_dim_t N, complex float* dst, const complex float* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = cabsf(src[i]);
}

static void zatanr(bart_dim_t N, complex float* dst, const complex float* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = atan(crealf(src[i])) + 0.I;
}

static void zatan2r(bart_dim_t N, complex float* dst, const complex float* src1, const complex float* src2)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = atan2f(crealf(src1[i]), crealf(src2[i])) + 0.I;
}

static void zsin(bart_dim_t N, complex float* dst, const complex float* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = csinf(src[i]);
}

static void zcos(bart_dim_t N, complex float* dst, const complex float* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = ccosf(src[i]);
}

static void zasin(bart_dim_t N, complex float* dst, const complex float* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = casinf(src[i]);
}

static void zacos(bart_dim_t N, complex float* dst, const complex float* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = cacosf(src[i]);
}

static void zsinh(bart_dim_t N, complex float* dst, const complex float* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = csinhf(src[i]);
}

static void zcosh(bart_dim_t N, complex float* dst, const complex float* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = ccoshf(src[i]);
}

static void zacosr(bart_dim_t N, complex float* dst, const complex float* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = acosf(crealf(src[i])) + 0.I;
}

static void zmax(bart_dim_t N, complex float* dst, const complex float* src1, const complex float* src2)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = MAX(crealf(src1[i]), crealf(src2[i]));
}


static void max(bart_dim_t N, float* dst, const float* src1, const float* src2)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = MAX(src1[i], src2[i]);
}

static void smax(bart_dim_t N, float val, float* dst, const float* src1)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = MAX(src1[i], val);
}


static void min(bart_dim_t N, float* dst, const float* src1, const float* src2)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = MIN(src1[i], src2[i]);
}


static void smin(bart_dim_t N, float val, float* dst, const float* src1)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = MIN(src1[i], val);
}


static void zsmax(bart_dim_t N, float val, complex float* dst, const complex float* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = MAX(crealf(src[i]), val);
}

static void zsmin(bart_dim_t N, float val, complex float* dst, const complex float* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = MIN(crealf(src[i]), val);
}


static void vec_pow(bart_dim_t N, float* dst, const float* src1, const float* src2)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = powf(src1[i], src2[i]);
}


static void vec_sqrt(bart_dim_t N, float* dst, const float* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = sqrtf(src[i]);
}


static void vec_round(bart_dim_t N, float* dst, const float* src)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = roundf(src[i]);
}


static void vec_zle(bart_dim_t N, complex float* dst, const complex float* src1, const complex float* src2)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = (crealf(src1[i]) <= crealf(src2[i])) ? 1. : 0.;
}


static void vec_le(bart_dim_t N, float* dst, const float* src1, const float* src2)
{
	for (bart_dim_t i = 0; i < N; i++)
		dst[i] = (src1[i] <= src2[i]) ? 1. : 0.;
}

/**
 * Step (1) of soft thesholding, y = ST(x, lambda).
 * Only computes the residual, resid = MAX( (abs(x) - lambda)/abs(x)), 0 )
 *
 * @param N number of elements
 * @param lambda threshold parameter
 * @param d pointer to destination, resid
 * @param x pointer to input
 */
static void zsoftthresh_half(bart_dim_t N, float lambda, complex float* d, const complex float* x)
{
	for (bart_dim_t i = 0; i < N; i++) {

		float norm = cabsf(x[i]);
		float red = norm - lambda;
		d[i] = (red > 0.) ? (red / norm) : 0.;
	}
}


static void zsoftthresh(bart_dim_t N, float lambda, complex float* d, const complex float* x)
{
	for (bart_dim_t i = 0; i < N; i++) {

		float norm = cabsf(x[i]);
		float red = norm - lambda;
		d[i] = (red > 0.) ? (red / norm) * x[i]: 0.;
	}
}



static void softthresh_half(bart_dim_t N, float lambda, float* d, const float* x)
{
	for (bart_dim_t i = 0; i < N; i++) {

		float norm = fabsf(x[i]);
		float red = norm - lambda;
		d[i] = (red > 0.) ? (red / norm) : 0.;
	}
}



static void softthresh(bart_dim_t N, float lambda, float* d, const float* x)
{
	for (bart_dim_t i = 0; i < N; i++) {

		float norm = fabsf(x[i]);
		float red = norm - lambda;
		d[i] = (red > 0.) ? (red / norm) * x[i] : 0.;
	}
}

/**
 * Return the absolute value of the kth largest array element
 * To be used for hard thresholding
 *
 * @param N number of elements
 * @param k the sorted element index to pick
 * @param ar the input complex array
 *
 * @returns the absolute value of the kth largest array element.
 *
 */

static float klargest_complex_partsort(int N, int k, const complex float* ar)
{
	assert(k <= N);

	complex float* tmp = xmalloc(sizeof(complex float[N]));
	copy(2 * N, (float*)tmp, (const float*)ar);

	float thr = quickselect_complex(tmp, N, k);

	xfree(tmp);

	return thr;
}

/**
 * Hard thesholding, y = HT(x, thr).
 * computes the thresholded vector, y = x * (abs(x) >= t(kmax))
 *
 * @param N number of elements
 * @param k threshold parameter, index of kth largest element of sorted x
 * @param d pointer to destination, y
 * @param x pointer to input
 */

static void zhardthresh(bart_dim_t N, int k, complex float* d, const complex float* x)
{
	float thr = klargest_complex_partsort(N, k, x);

	for (bart_dim_t i = 0; i < N; i++) {

		float norm = cabsf(x[i]);
		d[i] = (norm > thr) ? x[i] : 0.;
	}
}

/**
 * Hard thesholding mask, m = HS(x, thr).
 * computes the non-zero complex support vector, m = 1.0 * (abs(x) >= t(kmax))
 * This mask should be applied by complex multiplication.
 *
 * @param N number of elements
 * @param k threshold parameter, index of kth largest element of sorted x
 * @param d pointer to destination
 * @param x pointer to input
 */

static void zhardthresh_mask(bart_dim_t N, int k, complex float* d, const complex float* x)
{
	float thr = klargest_complex_partsort(N, k, x);

	for (bart_dim_t i = 0; i < N; i++) {

		float norm = cabsf(x[i]);
		d[i] = (norm > thr) ? 1. : 0.;
	}
}

static void swap(bart_dim_t N, float* a, float* b)
{
	for (bart_dim_t i = 0; i < N; i++) {

		float tmp = a[i];
		a[i] = b[i];
		b[i] = tmp;
	}
}


// identical copy in num/fft.c
static double fftmod_phase(bart_dim_t length, int j)
{
	bart_dim_t center1 = length / 2;
	double shift = (double)center1 / (double)length;
	return ((double)j - (double)center1 / 2.) * shift;
}

static complex double fftmod_phase2(bart_dim_t n, int j, bool inv, double phase)
{
	phase += fftmod_phase(n, j);
	double rem = phase - floor(phase);
	double sgn = inv ? -1. : 1.;
#if 1
	if (rem == 0.)
		return 1.;

	if (rem == 0.5)
		return -1.;

	if (rem == 0.25)
		return 1.i * sgn;

	if (rem == 0.75)
		return -1.i * sgn;
#endif
	return cexp(M_PI * 2.i * sgn * rem);
}

static void zfftmod(bart_dim_t N, complex float* dst, const complex float* src, int n, bool inv, double phase)
{
#if 1
	if (0 == n % 2) {

		complex float ph = fftmod_phase2(n, 0, inv, phase);

		for (bart_dim_t i = 0; i < N; i++)
			for (int j = 0; j < n; j++)
				dst[i * n + j] = src[i * n + j] * ((0 == j % 2) ? ph : -ph);

		return;
	}
#endif

	for (bart_dim_t i = 0; i < N; i++)
		for (int j = 0; j < n; j++)
			dst[i * n + j] = src[i * n + j] * fftmod_phase2(n, j, inv, phase);
}


static void pdf_gauss(bart_dim_t N, float mu, float sig, float* dst, const float* src)
{
	for (int i = 0; i < N; i++)
		dst[i] = expf(-(src[i] - mu) * (src[i] - mu) / (2 * sig * sig)) / (sqrtf(2 * M_PI) * sig);
}


static void vec_real(bart_dim_t N, float* dst, const complex float* src)
{
	for (int i = 0; i < N; i++)
		dst[i] = crealf(src[i]);
}

static  void vec_imag(bart_dim_t N, float* dst, const complex float* src)
{
	for (int i = 0; i < N; i++)
		dst[i] = cimagf(src[i]);
}

static void vec_zcmpl_real(bart_dim_t N, complex float* dst, const float* src)
{
	for (int i = 0; i < N; i++)
		dst[i] = src[i];
}

static void vec_zcmpl_imag(bart_dim_t N, complex float* dst, const float* src)
{
	for (int i = 0; i < N; i++)
		dst[i] = src[i] * 1.i;
}

static void vec_zcmpl(bart_dim_t N, complex float* dst, const float* real_src, const float* imag_src)
{
	for (int i = 0; i < N; i++)
		dst[i] = real_src[i] + imag_src[i] * 1.i;
}

static void vec_zfill(bart_dim_t N, complex float val, complex float* dst)
{
	for (int i = 0; i < N; i++)
		dst[i] = val;
}


static void xpay_bat(bart_dim_t Bi, bart_dim_t N, bart_dim_t Bo, const float* beta, float* a, const float* x)
{
	for (bart_dim_t bi = 0; bi < Bi; bi++) {

		for (bart_dim_t bo = 0; bo < Bo; bo++) {

			for (bart_dim_t i = 0; i < N; i++) {

				bart_dim_t idx = 2 * bi + 2 * Bi * i + 2 * Bi * N * bo;
				bart_dim_t idx_beta = bi + Bi * bo;

				a[idx] = x[idx] + beta[idx_beta] * a[idx];
				a[idx + 1] = x[idx + 1] + beta[idx_beta] * a[idx + 1];
			}
		}
	}

}

static void dot_bat(bart_dim_t Bi, bart_dim_t N, bart_dim_t Bo, float* dst, const float* src1, const float* src2)
{
	for (bart_dim_t bi = 0; bi < Bi; bi++) {

		for (bart_dim_t bo = 0; bo < Bo; bo++) {

			double ret = 0.;

			for (bart_dim_t i = 0; i < N; i++) {

				bart_dim_t idx = 2 * bi + 2 * Bi * i + 2 * Bi * N * bo;
				ret += src1[idx] * src2[idx] + src1[idx + 1] * src2[idx + 1];
			}

			dst[bi + Bi * bo] =(float)ret;
		}
	}
}

static void axpy_bat(bart_dim_t Bi, bart_dim_t N, bart_dim_t Bo, float* a, const float* alpha, const float* x)
{
	for (bart_dim_t bi = 0; bi < Bi; bi++) {

		for (bart_dim_t bo = 0; bo < Bo; bo++) {

			for (bart_dim_t i = 0; i < N; i++) {

				bart_dim_t idx = 2 * bi + 2 * Bi * i + + 2 * Bi * N * bo;
				bart_dim_t idx_alpha = bi + Bi * bo;

				a[idx] += alpha[idx_alpha] * x[idx];
				a[idx + 1] += alpha[idx_alpha] * x[idx + 1];
			}
		}
	}

}

static void zsetnanzero(bart_dim_t N, complex float* dst, const complex float* src)
{
	for (bart_dim_t i = 0; i < N; i++) {

		if (safe_isnanf(crealf(src[i])) || safe_isnanf(cimagf(src[i])))
			dst[i] = 0;
		else
			dst[i] = src[i];
	}
}

/*
 * If you add functions here, please also add to gpuops.c/gpukrnls.cu
 */
const struct vec_ops cpu_ops = {

	.float2double = float2double,
	.double2float = double2float,
	.dot = dot,
	.asum = asum,
	.zsum = zsum,
	.zl1norm = zl1norm,

	.zdot = zdot,

	.add = add,
	.sub = sub,
	.mul = mul,
	.div = vec_div,
	.fmac = fmac,
	.fmacD = fmacD,
	.sadd = sadd_update,

	.smul = smul,

	.axpy = axpy,

	.pow = vec_pow,
	.sqrt = vec_sqrt,
	.round = vec_round,

	.zle = vec_zle,
	.le = vec_le,

	.zmul = zmul,
	.zdiv = zdiv,
	.zfmac = zfmac,
	.zfmacD = zfmacD,
	.zmulc = zmulc,
	.zfmacc = zfmacc,
	.zfmaccD = zfmaccD,
	.zfsq2 = zfsq2,

	.zsmax = zsmax,
	.zsmin = zsmin,
	.zsmul = zsmul,
	.zsadd = zsadd,

	.zpow = zpow,
	.zphsr = zphsr,
	.zconj = zconj,
	.zexpj = zexpj,
	.zexp = zexp,
	.zlog = zlog,
	.zarg = zarg,
	.zabs = zabs,
	.zatanr = zatanr,
	.zatan2r = zatan2r,

	.zsin = zsin,
	.zcos = zcos,
	.zasin = zasin,
	.zacos = zacos,
	.zacosr = zacosr,

	.zsinh = zsinh,
	.zcosh = zcosh,

	.zcmp = zcmp,
	.zdiv_reg = zdiv_reg,
	.zfftmod = zfftmod,

	.zmax = zmax,

	.smax = smax,
	.max = max,
	.min = min,

	.zsoftthresh = zsoftthresh,
	.zsoftthresh_half = zsoftthresh_half,
	.softthresh = softthresh,
	.softthresh_half = softthresh_half,
	.zhardthresh = zhardthresh,
	.zhardthresh_mask = zhardthresh_mask,

	.exp = vec_exp,
	.log = vec_log,

	.pdf_gauss = pdf_gauss,

	.real = vec_real,
	.imag = vec_imag,
	.zcmpl_real = vec_zcmpl_real,
	.zcmpl_imag = vec_zcmpl_imag,
	.zcmpl = vec_zcmpl,

	.zfill = vec_zfill,

	.zsetnanzero = zsetnanzero,
};


extern const struct vec_iter_s cpu_iter_ops;
const struct vec_iter_s cpu_iter_ops = {

	.allocate = allocate,
	.del = del,
	.clear = clear,
	.copy = copy,
	.dot = dot,
	.norm = norm,
	.axpy = axpy,
	.xpay = xpay,
	.axpbz = axpbz,
	.smul = smul,
	.add = add,
	.sub = sub,
	.swap = swap,
	.zmul = zmul,
	.fmac = fmac,
	.sqrt = vec_sqrt,
	.sdiv = sdiv,
	.sadd = sadd,
	.mul = mul,
	.div = vec_div,
	.smax = smax,
	.smin = smin,
	.rand = gaussian_rand_vec,
	.uniform = uniform_rand_vec,
	.le = vec_le,

	.xpay_bat = xpay_bat,
	.dot_bat = dot_bat,
	.axpy_bat = axpy_bat,
};

