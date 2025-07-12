/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

 #include <complex.h>
#include <stdbool.h>
#include <cuComplex.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>

#include "num/gpuops.h"

#include "gpu_bloch.h"

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/* *
 * Per thread working memory buffers need to be either allocated in global or in
 * shared memory. To achieve coalesced access to buffers in global memory, we use
 * float mem[T][S] layout where T is total floats per thread and S is a stride.
 * For shared memory, we use mem[T] which is achieved by S=1
 *
 * General naming conventions:
 * long S - strided for local memory access
 * int N  - dimension of state
 *
 * */


__device__ static void vec_saxpy(long S, int N, float dst[/*N][S*/], const float a[/*N][S*/], float alpha, const float b[/*N][S*/])
{
	for (int i = 0; i < N; i++)
		dst[i * S] = a[i * S] + alpha * b[i * S];
}

__device__ static void vec_copy(long S, int N, float dst[/*N][S*/], const float src[/*N][S*/])
{
	vec_saxpy(S, N, dst, src, 0., src);
}

__device__ static float vec_sdot(long S, int N, const float a[/*N][S*/], const float b[/*N][S*/])
{
	float ret = 0.;

	for (int i = 0; i < N; i++)
		ret += a[i * S] * b[i * S];

	return ret;
}

__device__ static float vec_norm(long S, int N, const float x[/*N][S*/])
{
	return sqrtf(vec_sdot(S, N, x, x));
}

typedef void (*ode_fun_f)(long S, float out[/*N][S*/], float t, const float yn[/*N][S*/], const void* data);

#define tridiag(s) (s * (s + 1) / 2)

__device__ static void runge_kutta_step(long S, float h, int s, const float a[/*tridiag(s)*/], const float b[/*s*/], const float c[/*s - 1*/], int N, int K, float k[/*K][N][S*/], float ynp[/*N][S*/], float tmp[/*N][S*/], float tn, const float yn[/*N][S*/], ode_fun_f f, const void* data)
{
	vec_saxpy(S, N, ynp, yn, h * b[0], k + 0);

	for (int l = 0, t = 1; t < s; t++) {

		vec_copy(S, N, tmp, yn);

		for (int r = 0; r < t; r++, l++)
			vec_saxpy(S, N, tmp, tmp, h * a[l], k + (N * (r % K)) * S);

		f(S, k + (N * (t % K)) * S, tn + h * c[t - 1], tmp, data);

		vec_saxpy(S, N, ynp, ynp, h * b[t], k + (N * (t % K)) * S);
	}
}


/*
 * Dormand JR, Prince PJ. A family of embedded Runge-Kutta formulae,
 * Journal of Computational and Applied Mathematics 6:19-26 (1980).
 */


__device__ float dormand_prince_scale(float tol, float err)
{
#if 0
	float sc = 0.75 * powf(tol / err, 1. / 5.);

	return (sc < 2.) ? sc : 2.;
#else
	float sc = 1.25 * powf(err / tol, 1. / 5.);

	return 1. / ((sc > 1. / 2.) ? sc : (1. / 2.));
#endif
}


__constant__ static float a_dps[tridiag(7)] = {
	1. / 5.,
	3. / 40.,	9. / 40.,
	44. / 45.,	-56. / 15.,	32. / 9.,
	19372. / 6561.,	-25360. / 2187., 64448. / 6561., -212. / 729.,
	9017. / 3168.,  -355. / 33.,	46732. / 5247.,	49. / 176.,	-5103. / 18656.,
	35. / 384.,	0.,		500. / 1113.,	125. / 192.,	-2187. / 6784.,	11. / 84.,
};

__constant__ static float b_dps[7] = { 5179. / 57600., 0.,  7571. / 16695., 393. / 640., -92097. / 339200., 187. / 2100., 1. / 40. };

__constant__ static float c_dps[6] = { 1. / 5., 3. / 10., 4. / 5., 8. / 9., 1., 1. };


__device__ static float kern_dormand_prince_step2(long S, float h, int N, float ynp[/*N][S*/], float tn, const float yn[/*N][S*/], float k[/*6][N][S*/], float tmp[/*N][S*/], ode_fun_f f, const void* data)
{
	runge_kutta_step(S, h, 7, a_dps, b_dps, c_dps, N, 6, k, ynp, tmp, tn, yn, f, data);

	vec_saxpy(S, N, tmp, tmp, -1., ynp);
	return vec_norm(S, N, tmp);
}

__device__ static void kern_ode_interval(long S, float h, float tol, int N, float mem[/*9][N][S*/], float st, float end, ode_fun_f f, const void* data)
{
	float* yn/*[N][S]*/	= mem + 0 * N * S;
	float* ynp/*[N][S]*/	= mem + 1 * N * S;
	float* tmp/*[N][S]*/	= mem + 2 * N * S;
	float* k/*[6][N][S]*/	= mem + 3 * N * S;

	f(S, k, st, yn, data);

	for (float t = st; t < end; ) {

		if (h > end - t)
			h = end - t;

		float err = kern_dormand_prince_step2(S, h, N, ynp, t, yn, k, tmp, f, data);

		float h_new = h * dormand_prince_scale(tol, err);

		if (err > tol) {

			f(S, k, t, yn, data);	// recreate correct k[0] which has been overwritten
		} else {

			t += h;
			vec_copy(S, N, yn, ynp);
		}

		h = h_new;
	}
}

struct bloch_fields_s {

	float TP;
	long NP;

	const cuFloatComplex* pulse;
	float pr;
	float pi;
};

__device__ static cuFloatComplex kern_eval_pulse(float t, struct bloch_fields_s* p)
{
	float b = t / p->TP * (p->NP - 1);

	float r = p->pulse[(int)floorf(b)].x + (b - floorf(b)) * (p->pulse[(int)floorf(b) + 1].x - p->pulse[(int)floorf(b)].x);
	float i = p->pulse[(int)floorf(b)].y + (b - floorf(b)) * (p->pulse[(int)floorf(b) + 1].y - p->pulse[(int)floorf(b)].y);

	return make_cuFloatComplex(r, i);
}

struct bloch_data_s
{
	struct bloch_fields_s* p;

	float R1;
	float R2;
	float B1;
	float B0;
};

__device__ static void compute_fields(float t, float field[3], const struct bloch_data_s* data)
{
	cuFloatComplex b1 = kern_eval_pulse(t, data->p);

	field[0] = b1.x;
	field[1] = -b1.y;
	field[2] = data->B0;
}



__device__ inline void kern_vec3_rot(long So, float out[/*3][So*/], long S1, const float src1[/*3][S1*/], long S2, const float src2[/*3][S2*/])
{
	out[0 * So] = src1[1 * S1] * src2[2 * S2] - src1[2 * S1] * src2[1 * S2];
	out[1 * So] = src1[2 * S1] * src2[0 * S2] - src1[0 * S1] * src2[2 * S2];
	out[2 * So] = src1[0 * S1] * src2[1 * S2] - src1[1 * S1] * src2[0 * S2];
}


__device__ void f_bloch(long S, float out[/*3][S*/], float t, const float yn[/*3][S*/], const void* data)
{
	const struct bloch_data_s* d = (const struct bloch_data_s*)data;

	float gb[3];
	compute_fields(t, gb, d);

	gb[0] *= d->B1;
	gb[1] *= d->B1;

	kern_vec3_rot(S, out, S, yn, 1, gb);

	out[0 * S] -=  yn[0 * S] * d->R2;
	out[1 * S] -=  yn[1 * S] * d->R2;
	out[2 * S] -= (yn[2 * S] - 1.) * d->R1;
}

__device__ void f_bloch_pdy(long S, float out[/*3][3][S*/], float t, const float in[/*3][S*/], const void* data)
{
	const struct bloch_data_s* d = (const struct bloch_data_s*)data;

	float gb[3];
	compute_fields(t, gb, d);

	gb[0] *= d->B1;
	gb[1] *= d->B1;

	kern_vec3_rot(S, out + (0 * 3) * S, 1, (float[3]){ 1., 0., 0. }, 1, gb);
	kern_vec3_rot(S, out + (1 * 3) * S, 1, (float[3]){ 0., 1., 0. }, 1, gb);
	kern_vec3_rot(S, out + (2 * 3) * S, 1, (float[3]){ 0., 0., 1. }, 1, gb);

	out[(0 * 3 + 0) * S] -= d->R2;
	out[(1 * 3 + 1) * S] -= d->R2;
	out[(2 * 3 + 2) * S] -= d->R1;
}

__device__ void f_bloch_b1b0_pdp(long S, float out[/*3][3][S*/], float t, const float in[/*3][S*/], const void* data)
{
	const struct bloch_data_s* d = (const struct bloch_data_s*)data;

	cuFloatComplex b1 = kern_eval_pulse(t, d->p);

	float m0 = 1.;
	out[(0 * 3 + 0) * S] = 0.;
	out[(0 * 3 + 1) * S] = 0.;
	out[(0 * 3 + 2) * S] = -(in[2 * S] - m0);
	out[(1 * 3 + 0) * S] = -in[0 * S];
	out[(1 * 3 + 1) * S] = -in[1 * S];
	out[(1 * 3 + 2) * S] = 0.;
	out[(2 * 3 + 0) * S] = in[2 * S] * b1.y;
	out[(2 * 3 + 1) * S] = in[2 * S] * b1.x;
	out[(2 * 3 + 2) * S] = -b1.y * in[0 * S] - b1.x * in[1 * S];
	kern_vec3_rot(S, out + (3 * 3) * S, S, in, 1, (float[3]){ 0., 0., 1. });

}


__device__ void f_bloch_init_pdp(long S, float out[/*3][3][S*/], float t, const float in[/*3][S*/], const void* data)
{
	out[(0 * 3 + 0) * S] = 0.;
	out[(0 * 3 + 1) * S] = 0.;
	out[(0 * 3 + 2) * S] = 0.;
	out[(1 * 3 + 0) * S] = 0.;
	out[(1 * 3 + 1) * S] = 0.;
	out[(1 * 3 + 2) * S] = 0.;
	out[(2 * 3 + 0) * S] = 0.;
	out[(2 * 3 + 1) * S] = 0.;
	out[(2 * 3 + 2) * S] = 0.;
}

__device__ void f_bloch_init_b1b0_pdp(long S, float out[/*6][3][S*/], float t, const float in[/*3][S*/], const void* data)
{
	f_bloch_init_pdp(S, out + 0 * 3 * S, t, in, data);
	f_bloch_b1b0_pdp(  S, out + 3 * 3 * S, t, in, data);
}


struct sa_data_s
{
	const void* data;

	ode_fun_f f;
	ode_fun_f pdp;
	ode_fun_f pdy;

	int N;
	int P;

	float* tmp; //[N][N][S]
};

__device__ static void f_sa(long S, float out[/*P + 1][N][S*/], float t, const float yn[/*P + 1][N][S*/], const void* data)
{
	const struct sa_data_s* d = (const struct sa_data_s*)data;

	d->f(S, out, t, yn, d->data);

	d->pdp(S, out + d->N * S, t, yn, d->data);

	//CAVEAT: tmp[N][N] is transposed of df/dy as in ode.c
	d->pdy(S, d->tmp, t, yn, d->data);

	for (int i = 0; i < d->P; i++)
		for (int j = 0; j < d->N; j++)
			for (int k = 0; k < d->N; k++)
				out[((1 + i) * d->N + j) * S] += d->tmp[(d->N * k + j) * S] * yn[((1 + i) * d->N + k) * S];
}

__device__ const struct bloch_data_s load_bloch_data(long i, struct bloch_fields_s* p, long SPP, long SPV, const cuFloatComplex* par)
{
	struct bloch_data_s db;
	db.p = p;
	db.R1 = par[SPV * i + SPP * 0].x;
	db.R2 = par[SPV * i + SPP * 1].x;
	db.B1 = par[SPV * i + SPP * 2].x;
	db.B0 = par[SPV * i + SPP * 3].x;

	return db;
}

__device__ __host__ static int kern_bloch_mem(int N, int P)
{
	int mem = N * (1 + P) * 9;	// yn, ynp, tmp, k

	if (0 < P)
		mem += N * N;		// df / dx

	return mem;
}

__global__ static void kern_ode_interval_bloch(long M, float* buf_glb, long SMM, long SMV, cuFloatComplex* omag, const cuFloatComplex* imag, long SPP, long SPV, const cuFloatComplex* par, struct bloch_fields_s p, float h, float tol, float st, float end)
{
	extern __shared__ float buf_shm[];

	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	int N = 3;
	long S = 1;
	long T = kern_bloch_mem(N, 0);

	float* buf;

	if (NULL == buf_glb) {

		buf = buf_shm + T * threadIdx.x;
	} else {

		S = stride;
		buf = buf_glb + start;
	}

	for (long i = start; i < M; i += stride) {

		struct bloch_data_s d = load_bloch_data(i, &p, SPP, SPV, par);

		for (int j = 0; j < N; j++)
			buf[j * S] = imag[SMM * j + SMV * i].x;

		kern_ode_interval(S, h, tol, N, buf, st, end, f_bloch, (const void*)&d);

		for (int j = 0; j < N; j++)
			omag[SMM * j + SMV * i] = make_cuFloatComplex(buf[j * S], 0);
	}
}

extern "C" void cuda_ode_interval_bloch(long M, long SMM, long SMV, _Complex float* mag, long SPP, long SPV, const _Complex float* par, long Np, float Tp, const _Complex float* pulse, float h, float tol, float st, float end)
{
	int memsize = kern_bloch_mem(3, 0) * sizeof(float);
	int max_blocksize = (48 * 1024) / memsize;
	int blocksize = 1;

	while (blocksize < 1024 && 2 * blocksize <= max_blocksize && blocksize < M)
		blocksize *= 2;

	int gridsize = MIN((M + blocksize - 1) / blocksize, 65536 - 1);

	struct bloch_fields_s field = {

		.TP = Tp,
		.NP = Np,
		.pulse = (const cuFloatComplex*)pulse,
	};

	float* buf = blocksize * memsize > (48 * 1024) ? (float*)cuda_malloc(memsize * gridsize * blocksize) : NULL;

	kern_ode_interval_bloch<<<gridsize, blocksize, (NULL != buf) ? 0 : blocksize * memsize, cuda_get_stream()>>>(M, buf, SMM, SMV, (cuFloatComplex*)mag, (cuFloatComplex*)mag, SPP, SPV, (cuFloatComplex*)par, field, h, tol, st, end);

	if (NULL != buf)
		cuda_free(buf);
}


__global__ static void kern_ode_interval_bloch_sa(long M, float* buf_glb, long SMM, long SMV, cuFloatComplex* omag, const cuFloatComplex* imag, long SDMM, long SDMV, cuFloatComplex* odmag, const cuFloatComplex* idmag, long SDPP, long SDPV, cuFloatComplex* odpar, const cuFloatComplex* idpar, long SPP, long SPV, const cuFloatComplex* par, struct bloch_fields_s p, float h, float tol, float st, float end)
{
	extern __shared__ float buf_shm[];

	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	int N = 3;
	int P = 0;
	int Pp = 0; // number of parameter derivative directions
	long S = 1;

	if (NULL != idmag)
		P += 3; // (M0_x M0_y M0_z)

	if (NULL != idpar) {

		Pp = 4; // (R1 R2 B1 B0)
		P += Pp;
	}

	long T = kern_bloch_mem(N, P);

	float* buf;

	if (NULL == buf_glb) {

		buf = buf_shm + T * threadIdx.x;
	} else {

		S = stride;
		buf = buf_glb + start;
	}

	for (long i = start; i < M; i += stride) {

		int j = 0;
		for (int k = 0; k < N; k++)
			buf[(j++) * S] = imag[SMM * k + SMV * i].x;

		for (int k = 0; k < N * N && NULL != idmag; k++)
			buf[(j++) * S] = idmag[SDMM * k + SDMV * i].x;

		for (int k = 0; k < N * Pp && NULL != idpar; k++)
			buf[(j++) * S] = idpar[SDPP * k + SDPV * i].x;

		struct bloch_data_s db = load_bloch_data(i, &p, SPP, SPV, par);

		struct sa_data_s d;
		d.f = f_bloch;
		d.data = (const void*)&db;
		d.N = N;
		d.P = P;
		d.pdy = f_bloch_pdy;
		d.tmp = buf + S * (N * (P + 1) * 9);

		d.pdp = NULL;

		if (NULL != idmag && NULL != idpar)
			d.pdp = f_bloch_init_b1b0_pdp;

		if (NULL != idmag && NULL == idpar)
			d.pdp = f_bloch_init_pdp;

		if (NULL == idmag && NULL != idpar)
			d.pdp = f_bloch_b1b0_pdp;

		kern_ode_interval(S, h, tol, N * (P + 1), buf, st, end, f_sa, (const void*)&d);

		j = 0;

		for (int k = 0; k < N; k++)
			omag[SMM * k + SMV * i] = make_cuFloatComplex(buf[(j++) * S], 0);

		for (int k = 0; k < N * N && NULL != odmag; k++)
			odmag[SDMM * k + SDMV * i] = make_cuFloatComplex(buf[(j++) * S], 0);

		for (int k = 0; k < N * Pp && NULL != odpar; k++)
			odpar[SDPP * k + SDPV * i] = make_cuFloatComplex(buf[(j++) * S], 0);
	}
}




extern "C" void cuda_ode_interval_bloch_sa(long M, long SMM, long SMV, _Complex float* mag, long SDMM, long SDMV, _Complex float* dmag, long SDPP, long SDPV, _Complex float* dpar, long SPP, long SPV, const _Complex float* par, long Np, float Tp, const _Complex float* pulse, float h, float tol, float st, float end)
{
	int N = 3;
	int P = 0;

	if (NULL != dmag)
		P += 3;

	if (NULL != dpar)
		P += 4;

	if (0 == P) {

		cuda_ode_interval_bloch(M, SMM, SMV, mag, SPP, SPV, par, Np, Tp, pulse, h, tol, st, end);
		return;
	}

	int memsize = kern_bloch_mem(N, P) * sizeof(float);
	int max_blocksize = (48 * 1024) / memsize;
	int blocksize = 1;

	while (blocksize < 1024 && 2 * blocksize <= max_blocksize && blocksize < M)
		blocksize *= 2;

	int gridsize = MIN((M + blocksize - 1) / blocksize, 65536 - 1);

	struct bloch_fields_s field = {

		.TP = Tp,
		.NP = Np,
		.pulse = (const cuFloatComplex*)pulse,
	};

	float* buf = blocksize * memsize > (48 * 1024) ? (float*)cuda_malloc(memsize * gridsize * blocksize) : NULL;
	assert(NULL == buf);

	kern_ode_interval_bloch_sa<<<gridsize, blocksize, (NULL != buf) ? 0 : blocksize * memsize, cuda_get_stream()>>>(M, buf, SMM, SMV, (cuFloatComplex*)mag, (cuFloatComplex*)mag, SDMM, SDMV, (cuFloatComplex*)dmag, (cuFloatComplex*)dmag, SDPP, SDPV, (cuFloatComplex*)dpar, (cuFloatComplex*)dpar, SPP, SPV, (cuFloatComplex*)par, field, h, tol, st, end);

	if (NULL != buf)
		cuda_free(buf);
}


