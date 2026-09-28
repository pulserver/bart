/* Copyright 2024-2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <math.h>
#include <complex.h>
#include <assert.h>

#include "misc/nested.h"
#include "misc/misc.h"

#include "num/multind.h"
#include "num/loop.h"
#include "num/flpmath.h"
#include "num/multiplace.h"
#include "num/vptr_fun.h"
#ifdef USE_GPU
#include "num/gpuops.h"
#endif

#include "linops/linop.h"

#include "nlops/nlop.h"

#ifdef USE_GPU
#include "motion/gpu_interpolate.h"
#endif

#include "interpolate.h"




typedef CLOSURE_TYPE(void, (bart_dim_t ind, float d)) interp_update_t;

#ifndef __clang__
#define VLA(x) x
#else
// blocks extension does not play well even with arguments which
// just look like variably-modified types
#define VLA(x)
#endif


void md_positions(int N, int d, bart_flags_t flags, const bart_dim_t sdims[N], const bart_dim_t pdims[N], complex float* pos)
{
	assert(d < N);
	assert(pdims[d] == bitcount(flags));
	assert(sdims[d] == bitcount(flags) || 1 == sdims[d]);

#ifdef USE_GPU

	if (cuda_ondevice(pos)) {

		cuda_positions(N, d, flags, sdims, pdims, pos);
		return;
	}
#endif

	bart_dim_t map[bitcount(flags)];
	for (int i = 0, ip = 0; i < N; i++)
		if (MD_IS_SET(flags, i))
			map[ip++] = i;

	const bart_dim_t* mapp = map;

	const bart_dim_t* sdimsp = sdims;
	const bart_dim_t* pdimsp = pdims;

	NESTED(complex float, pos_kernel, (const bart_dim_t pos[]))
	{
		float world = (pos[mapp[pos[d]]] - (pdimsp[mapp[pos[d]]] / 2)) / (float)pdimsp[mapp[pos[d]]];
		complex float ret = sdimsp[mapp[pos[d]]] * world + (sdimsp[mapp[pos[d]]] / 2);

		return ret;
	};

	complex float* cpu_pos = md_alloc(N, pdims, CFL_SIZE);
	md_parallel_zsample(N, pdims, cpu_pos, pos_kernel);

	md_copy(N, pdims, pos, cpu_pos, CFL_SIZE);
	md_free(cpu_pos);
}



static float spline_nn(float x)
{
	if (-0.5 <= x && x < 0.5)
		return 1;

	return 0.;
}

static float dspline_nn(float /*x*/)
{
	return 0.;
}

static float spline_lerp(float x)
{
	if (-1. <= x && x < 1.)
		return 1. - fabsf(x);

	return 0.;
}

static float dspline_lerp(float x)
{
	if (-1. <= x && x < 1.)
		return (x < 0) ? 1. : -1.;

	return 0.;
}


/* *
 * R. Keys, "Cubic convolution interpolation for digital image processing," in
 * IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 29,
 * pp. 1153-1160, December 1981, doi: 10.1109/TASSP.1981.1163711
 * */

static float spline_keys(float x)
{
	float a = -0.5;
	x = fabsf(x);

	if (x < 1)
		return (a + 2.) * x * x * x - (a + 3.) * x * x + 1.;

	if (x < 2)
		return a * x * x * x - 5. * a * x * x + 8. * a * x - 4 * a;

	return 0.;
}

static float dspline_keys(float x)
{
	float a = -0.5;
	float s = x < 0. ? -1. : 1.;
	x = fabsf(x);


	if (x < 1)
		return s * ((a + 2.) * 3. * x * x - (a + 3.) * 2. * x );

	if (x < 2)
		return s * (a * 3. * x * x - 5. * a * 2. * x + 8. * a * 1.);

	return 0.;
}



static float spline(int ord, float x)
{
	if (0 == ord)
		return spline_nn(x);

	if (1 == ord)
		return spline_lerp(x);

	if (3 == ord)
		return spline_keys(x);

	assert(0);
	return 0;
}

static float dspline(int ord, float x)
{
	if (0 == ord)
		return dspline_nn(x);

	if (1 == ord)
		return dspline_lerp(x);

	if (3 == ord)
		return dspline_keys(x);

	assert(0);
	return 0;
}


// Naming
// grid
// intp
// coor



static void interp_point_gen(int N, const bart_dim_t gdims[VLA(N)], const bart_stride_t gstrs[VLA(N)], const float coor[VLA(N)], int ord, float width, interp_update_t update)
{
#ifndef __clang__
	int sti[N];
	int eni[N];
#else
	// blocks extension does not play well with variably-modified types
	int* sti = alloca(sizeof(int[N]));
	int* eni = alloca(sizeof(int[N]));
#endif
	for (int j = 0; j < N; j++) {

		sti[j] = (int)ceil(coor[j] - 0.5 * width);
		eni[j] = (int)floor(coor[j] + 0.5 * width);
		sti[j] = MAX(sti[j], 0);
		eni[j] = MIN(eni[j], gdims[j] - 1);

		if (sti[j] > eni[j])
			return;
	}

	__block NESTED(void, interp_point_r, (int N, bart_dim_t ind, float d))	// __block for recursion
	{
		if (0 == N) {

			NESTED_CALL(update, (ind, d));

		} else {

			N--;

			for (int w = sti[N]; w <= eni[N]; w++) {

				float d2 = d * spline(ord,  coor[N] - (float)w);
				bart_dim_t ind2 = ind + w * gstrs[N] / (bart_stride_t)CFL_SIZE;

				interp_point_r(N, ind2, d2);
			}
		}
	};

	interp_point_r(N, 0, 1.);
}

static void interp_point(int N, const bart_dim_t gdims[VLA(N)], const bart_stride_t gstrs[VLA(N)], const complex float* grid, const float coor[VLA(N)], complex float* intp, int ord, float width)
{
	NESTED(void, update, (bart_dim_t ind, float d))
	{
		__real(intp[0]) += __real(grid[ind]) * d;
		__imag(intp[0]) += __imag(grid[ind]) * d;
	};

	interp_point_gen(N, gdims, gstrs, coor, ord, width, CLOSURE(interp_update_t, update));
}

static void interp_pointH(int N, const bart_dim_t gdims[VLA(N)], const bart_stride_t gstrs[VLA(N)], complex float* grid, const float coor[VLA(N)], const complex float* intp, int ord, float width)
{
	NESTED(void, update, (bart_dim_t ind, float d))
	{
		// we are allowed to update real and imaginary part independently which works atomically
#pragma 	omp atomic
		__real(grid[ind]) += __real(intp[0]) * d;
#pragma 	omp atomic
		__imag(grid[ind]) += __imag(intp[0]) * d;
	};

	interp_point_gen(N, gdims, gstrs, coor, ord, width, CLOSURE(interp_update_t, update));
}

static void interpolate2(int ord, int M, const bart_dim_t dims[M], const bart_stride_t istrs[M], complex float* intp, const bart_stride_t cstrs[M], bart_dim_t cstrs_dir, const complex float* coor, const bart_dim_t gdims[M], const bart_stride_t gstrs[M], const complex float* grid)
{
	const bart_dim_t* istrsp = istrs;
	const bart_dim_t* gstrsp = gstrs;
	const bart_dim_t* cstrsp = cstrs;
	const bart_dim_t* gdimsp = gdims;

	NESTED(void, interp_kernel, (const bart_dim_t pos[]))
	{
		bart_stride_t ioffset = md_calc_offset(M, istrsp, pos) / (bart_stride_t)CFL_SIZE;
		bart_stride_t coffset = md_calc_offset(M, cstrsp, pos) / (bart_stride_t)CFL_SIZE;

		float coord[M];
		for (int i = 0; i < M; i++)
			coord[i] = crealf(coor[coffset + i * cstrs_dir / (bart_stride_t)CFL_SIZE]);

		interp_point(M, gdimsp, gstrsp, grid, coord, intp + ioffset, ord, ord + 1);
	};

	md_parallel_loop(M, dims, ~UINT64_C(1), interp_kernel);
}

static void interpolateH2(int ord, int M, const bart_dim_t gdims[M], const bart_stride_t gstrs[M], complex float* grid, const bart_dim_t dims[M], const bart_stride_t istrs[M], const complex float* intp, const bart_stride_t cstrs[M], bart_dim_t cstrs_dir, const complex float* coor)
{
	const bart_dim_t* istrsp = istrs;
	const bart_dim_t* gstrsp = gstrs;
	const bart_dim_t* cstrsp = cstrs;
	const bart_dim_t* gdimsp = gdims;

	NESTED(void, interp_kernel, (const bart_dim_t pos[]))
	{
		bart_stride_t ioffset = md_calc_offset(M, istrsp, pos) / (bart_stride_t)CFL_SIZE;
		bart_stride_t coffset = md_calc_offset(M, cstrsp, pos) / (bart_stride_t)CFL_SIZE;

		float coord[M];
		for (int i = 0; i < M; i++)
			coord[i] = crealf(coor[coffset + i * cstrs_dir / (bart_stride_t)CFL_SIZE]);

		interp_pointH(M, gdimsp, gstrsp, grid, coord, intp + ioffset, ord, ord + 1);
	};

	md_parallel_loop(M, dims, ~UINT64_C(1), interp_kernel);
}

static void interpolate_compute_red(int d, bart_flags_t flags,
					int M, bart_dim_t idims_red[M], bart_dim_t istrs_red[M], bart_dim_t cstrs_red[M], bart_dim_t gdims_red[M], bart_dim_t gstrs_red[M],
					int N, const bart_dim_t dims[N], const bart_stride_t istrs[N], const bart_stride_t cstrs[N], const bart_dim_t gdims[N], const bart_stride_t gstrs[N])
{
	assert(!MD_IS_SET(flags, d));
	assert((0 <= d) && (d < N));
	assert(dims[d] == bitcount(flags));
	assert(M == bitcount(flags));

	bart_flags_t bflags = ~MD_BIT(d) & ~flags;
	assert(md_check_equal_dims(N, gdims, dims, bflags & md_nontriv_dims(N, gdims)));

	for (int i = 0, ip = 0; i < N; i++) {

		if (!MD_IS_SET(flags, i))
			continue;

		gdims_red[ip] = gdims[i];
		gstrs_red[ip] = gstrs[i];
		idims_red[ip] = dims[i];
		istrs_red[ip] = istrs[i];
		cstrs_red[ip] = cstrs[i];
		ip++;
	}
}

static void md_interpolate2_int(int d, bart_flags_t flags, int ord, int N, const bart_dim_t dims[N], const bart_stride_t istrs[N], complex float* intp, const bart_stride_t cstrs[N], const complex float* coor, const bart_dim_t gdims[N], const bart_stride_t gstrs[N], const complex float* grid)
{
	int M = bitcount(flags);
	bart_dim_t gdims_red[M];
	bart_dim_t gstrs_red[M];
	bart_dim_t idims_red[M];
	bart_dim_t istrs_red[M];
	bart_dim_t cstrs_red[M];

	interpolate_compute_red(d, flags, M, idims_red, istrs_red, cstrs_red, gdims_red, gstrs_red,
				N, dims, istrs, cstrs, gdims, gstrs);

	bart_dim_t pos[N];
	md_set_dims(N, pos, 0);

	do {
		bart_stride_t ioffset = md_calc_offset(N, istrs, pos) / (bart_stride_t)CFL_SIZE;
		bart_stride_t coffset = md_calc_offset(N, cstrs, pos) / (bart_stride_t)CFL_SIZE;
		bart_stride_t goffset = md_calc_offset(N, gstrs, pos) / (bart_stride_t)CFL_SIZE;

#ifdef USE_GPU
		if (cuda_ondevice(coor))
			cuda_interpolate2(ord, M, idims_red, istrs_red, intp + ioffset, cstrs_red, cstrs[d], coor + coffset, gdims_red, gstrs_red, grid + goffset);
		else
#endif
			interpolate2(ord, M, idims_red, istrs_red, intp + ioffset, cstrs_red, cstrs[d], coor + coffset, gdims_red, gstrs_red, grid + goffset);

	} while (md_next(N, dims, ~flags & ~MD_BIT(d), pos));
}

static void md_interpolateH2_int(int d, bart_flags_t flags, int ord, int N, const bart_dim_t gdims[N], const bart_stride_t gstrs[N], complex float* grid, const bart_dim_t dims[N], const bart_stride_t istrs[N], const complex float* intp, const bart_stride_t cstrs[N], const complex float* coor)
{
	int M = bitcount(flags);
	bart_dim_t gdims_red[M];
	bart_dim_t gstrs_red[M];
	bart_dim_t idims_red[M];
	bart_dim_t istrs_red[M];
	bart_dim_t cstrs_red[M];

	interpolate_compute_red(d, flags, M, idims_red, istrs_red, cstrs_red, gdims_red, gstrs_red,
				N, dims, istrs, cstrs, gdims, gstrs);

	bart_dim_t pos[N];
	md_set_dims(N, pos, 0);

	do {
		bart_stride_t ioffset = md_calc_offset(N, istrs, pos) / (bart_stride_t)CFL_SIZE;
		bart_stride_t coffset = md_calc_offset(N, cstrs, pos) / (bart_stride_t)CFL_SIZE;
		bart_stride_t goffset = md_calc_offset(N, gstrs, pos) / (bart_stride_t)CFL_SIZE;

#ifdef USE_GPU
		if (cuda_ondevice(coor))
			cuda_interpolateH2(ord, M, gdims_red, gstrs_red, grid + goffset, idims_red, istrs_red, intp + ioffset, cstrs_red, cstrs[d], coor + coffset);
		else
#endif
			interpolateH2(ord, M, gdims_red, gstrs_red, grid + goffset, idims_red, istrs_red, intp + ioffset, cstrs_red, cstrs[d], coor + coffset);

	} while (md_next(N, dims, ~flags & ~MD_BIT(d), pos));
}



struct vptr_interpolate_s {

	vptr_fun_data_t super;

	int d;
	int ord;
	bart_flags_t flags;
};

DEF_TYPEID(vptr_interpolate_s);

static void md_interpolate2_vptr(vptr_fun_data_t* _data, int N, int D, const bart_dim_t* dims[N], const bart_stride_t* strs[N], void* args[N])
{
	auto d = CAST_DOWN(vptr_interpolate_s, _data);

	md_interpolate2_int(d->d, d->flags, d->ord, D, dims[0], strs[0], args[0], strs[1], args[1], dims[2], strs[2], args[2]);
}

static void md_interpolate2H_vptr(vptr_fun_data_t* _data, int N, int D, const bart_dim_t* dims[N], const bart_stride_t* strs[N], void* args[N])
{
	auto d = CAST_DOWN(vptr_interpolate_s, _data);

	md_interpolateH2_int(d->d, d->flags, d->ord, D, dims[0], strs[0], args[0], dims[1], strs[1], args[1], strs[2], args[2]);
}


void md_interpolate2(int d, bart_flags_t flags, int ord, int N, const bart_dim_t dims[N], const bart_stride_t istrs[N], complex float* intp, const bart_stride_t cstrs[N], const complex float* coor, const bart_dim_t gdims[N], const bart_stride_t gstrs[N], const complex float* grid)
{
	PTR_ALLOC(struct vptr_interpolate_s, _d);
	SET_TYPEID(vptr_interpolate_s, _d);
	_d->super.del = NULL;
	_d->d = d;
	_d->ord = ord;
	_d->flags = flags;

	exec_vptr_zfun(md_interpolate2_vptr, CAST_UP(PTR_PASS(_d)), 3, N, ~flags & ~MD_BIT(d), MD_BIT(0), MD_BIT(0) | MD_BIT(1) | MD_BIT(2),
			(const bart_dim_t*[3]) { dims, dims, gdims },
			(const bart_dim_t*[3]) { istrs, cstrs, gstrs },
			(complex float*[3]) { intp, (void*)coor, (void*)grid });
}

void md_interpolateH2(int d, bart_flags_t flags, int ord, int N, const bart_dim_t gdims[N], const bart_stride_t gstrs[N], complex float* grid, const bart_dim_t dims[N], const bart_stride_t istrs[N], const complex float* intp, const bart_stride_t cstrs[N], const complex float* coor)
{
	PTR_ALLOC(struct vptr_interpolate_s, _d);
	SET_TYPEID(vptr_interpolate_s, _d);
	_d->super.del = NULL;
	_d->d = d;
	_d->ord = ord;
	_d->flags = flags;

	exec_vptr_zfun(md_interpolate2H_vptr, CAST_UP(PTR_PASS(_d)), 3, N, ~flags & ~MD_BIT(d), MD_BIT(0), MD_BIT(0) | MD_BIT(1) | MD_BIT(2),
			(const bart_dim_t*[3]) { gdims, dims, dims },
			(const bart_dim_t*[3]) { gstrs, istrs, cstrs },
			(complex float*[3]) { grid, (void*)intp, (void*)coor });
}






static void interp_point_adj_coor_gen(int N, const bart_dim_t gdims[VLA(N)], const bart_stride_t gstrs[VLA(N)], const float coor[VLA(N)], int ord, float width, int dir, interp_update_t update)
{
#ifndef __clang__
	int sti[N];
	int eni[N];
#else
	// blocks extension does not play well with variably-modified types
	int* sti = alloca(sizeof(int[N]));
	int* eni = alloca(sizeof(int[N]));
#endif
	for (int j = 0; j < N; j++) {

		sti[j] = (int)ceil(coor[j] - 0.5 * width);
		eni[j] = (int)floor(coor[j] + 0.5 * width);
		sti[j] = MAX(sti[j], 0);
		eni[j] = MIN(eni[j], gdims[j] - 1);

		if (sti[j] > eni[j])
			return;
	}

	__block NESTED(void, interp_point_r, (int N, bart_dim_t ind, float d))	// __block for recursion
	{
		if (0 == N) {

			NESTED_CALL(update, (ind, d));

		} else {

			N--;

			for (int w = sti[N]; w <= eni[N]; w++) {

				float d2 = d * ((N == dir) ? dspline: spline)(ord,  coor[N] - (float)w);
				bart_dim_t ind2 = ind + w * gstrs[N] / (bart_stride_t)CFL_SIZE;

				interp_point_r(N, ind2, d2);
			}
		}
	};

	interp_point_r(N, 0, 1.);
}



static void interp_point_adj_coor(int N, const bart_dim_t gdims[VLA(N)], const bart_stride_t gstrs[VLA(N)], const complex float* grid, const float coor[VLA(N)], complex float* dcoor, int ord, float width, int dir, complex float dintp)
{
	NESTED(void, update, (bart_dim_t ind, float d))
	{
		float tmp = crealf(conjf(grid[ind]) * dintp * d);
#pragma 	omp atomic
		__real(dcoor[0]) += tmp;
	};

	interp_point_adj_coor_gen(N, gdims, gstrs, coor, ord, width, dir, CLOSURE(interp_update_t, update));
}

static void interpolate_adj_coor2(int ord, int M, const bart_dim_t dims[M], const bart_stride_t istrs[M], const complex float* dintp, const bart_stride_t cstrs[M], bart_dim_t cstrs_dir, const complex float* coor, complex float* dcoor, const bart_dim_t gdims[M], const bart_stride_t gstrs[M], const complex float* grid)
{
	const bart_dim_t* istrsp = istrs;
	const bart_dim_t* gstrsp = gstrs;
	const bart_dim_t* cstrsp = cstrs;
	const bart_dim_t* gdimsp = gdims;

	NESTED(void, interp_kernel, (const bart_dim_t pos[]))
	{
		bart_stride_t ioffset = md_calc_offset(M, istrsp, pos) / (bart_stride_t)CFL_SIZE;
		bart_stride_t coffset = md_calc_offset(M, cstrsp, pos) / (bart_stride_t)CFL_SIZE;

		float coord[M];
		for (int i = 0; i < M; i++)
			coord[i] = crealf(coor[coffset + i * cstrs_dir / (bart_stride_t)CFL_SIZE]);

		for (int i = 0; i < M; i++)
			interp_point_adj_coor(M, gdimsp, gstrsp, grid, coord, dcoor + coffset + i * cstrs_dir / (bart_stride_t)CFL_SIZE, ord, ord + 1, i, dintp[ioffset]);
	};

	md_parallel_loop(M, dims, ~UINT64_C(1), interp_kernel);
}

static void md_interpolate_adj_coor2_int(int d, bart_flags_t flags, int ord, int N, const bart_dim_t dims[N], const bart_stride_t cstrs[N], const complex float* coor, complex float* dcoor, const bart_stride_t istrs[N], const complex float* dintp, const bart_dim_t gdims[N], const bart_stride_t gstrs[N], const complex float* grid)
{
	int M = bitcount(flags);
	bart_dim_t gdims_red[M];
	bart_dim_t gstrs_red[M];
	bart_dim_t idims_red[M];
	bart_dim_t istrs_red[M];
	bart_dim_t cstrs_red[M];

	interpolate_compute_red(d, flags, M, idims_red, istrs_red, cstrs_red, gdims_red, gstrs_red,
				N, dims, istrs, cstrs, gdims, gstrs);

	bart_dim_t pos[N];
	md_set_dims(N, pos, 0);

	do {
		bart_stride_t ioffset = md_calc_offset(N, istrs, pos) / (bart_stride_t)CFL_SIZE;
		bart_stride_t coffset = md_calc_offset(N, cstrs, pos) / (bart_stride_t)CFL_SIZE;
		bart_stride_t goffset = md_calc_offset(N, gstrs, pos) / (bart_stride_t)CFL_SIZE;

#ifdef USE_GPU
		if (cuda_ondevice(dcoor))
			cuda_interpolate_adj_coor2(ord, M, idims_red, istrs_red, dintp + ioffset, cstrs_red, cstrs[d], coor + coffset, dcoor + coffset, gdims_red, gstrs_red, grid + goffset);
		else
#endif
			interpolate_adj_coor2(ord, M, idims_red, istrs_red, dintp + ioffset, cstrs_red, cstrs[d], coor + coffset, dcoor + coffset, gdims_red, gstrs_red, grid + goffset);

	} while (md_next(N, dims, ~flags & ~MD_BIT(d), pos));
}






static void interp_point_der_gen(int N, const bart_dim_t gdims[VLA(N)], const bart_stride_t gstrs[VLA(N)], const float coor[VLA(N)], const float dcoor[VLA(N)], int ord, float width, interp_update_t update)
{
#ifndef __clang__
	int sti[N];
	int eni[N];
#else
	// blocks extension does not play well with variably-modified types
	int* sti = alloca(sizeof(int[N]));
	int* eni = alloca(sizeof(int[N]));
#endif
	for (int j = 0; j < N; j++) {

		sti[j] = (int)ceil(coor[j] - 0.5 * width);
		eni[j] = (int)floor(coor[j] + 0.5 * width);
		sti[j] = MAX(sti[j], 0);
		eni[j] = MIN(eni[j], gdims[j] - 1);

		if (sti[j] > eni[j])
			return;
	}

	__block NESTED(void, dinterp_point_r, (int N, bart_dim_t ind, float d, float dd))	// __block for recursion
	{
		if (0 == N) {

			NESTED_CALL(update, (ind, d));

		} else {

			N--;

			for (int w = sti[N]; w <= eni[N]; w++) {

				float d2 = d * spline(ord, coor[N] - (float)w);
				d2 += dd * dspline(ord, coor[N] - (float)w) * dcoor[N];

				float dd2 = dd * spline(ord, coor[N] - (float)w);

				bart_dim_t ind2 = ind + w * gstrs[N] / (bart_stride_t)CFL_SIZE;

				dinterp_point_r(N, ind2, d2, dd2);
			}
		}
	};

	dinterp_point_r(N, 0, 0., 1.);
}

static void der_interp_point(int N, const bart_dim_t gdims[VLA(N)], const bart_stride_t gstrs[VLA(N)], const complex float* grid, const float coor[VLA(N)], const float dcoor[VLA(N)], complex float* dintp, int ord, float width)
{
	NESTED(void, update, (bart_dim_t ind, float d))
	{
		__real(dintp[0]) += __real(grid[ind]) * d;
		__imag(dintp[0]) += __imag(grid[ind]) * d;
	};

	interp_point_der_gen(N, gdims, gstrs, coor, dcoor, ord, width, CLOSURE(interp_update_t, update));
}

static void interpolate_der_coor2(int ord, int M, const bart_dim_t dims[M], const bart_stride_t istrs[M], complex float* dintp, const bart_stride_t cstrs[M], bart_dim_t cstrs_dir, const complex float* coor, const complex float* dcoor, const bart_dim_t gdims[M], const bart_stride_t gstrs[M], const complex float* grid)
{
	const bart_dim_t* istrsp = istrs;
	const bart_dim_t* gstrsp = gstrs;
	const bart_dim_t* cstrsp = cstrs;
	const bart_dim_t* gdimsp = gdims;

	NESTED(void, interp_kernel, (const bart_dim_t pos[]))
	{
		bart_stride_t ioffset = md_calc_offset(M, istrsp, pos) / (bart_stride_t)CFL_SIZE;
		bart_stride_t coffset = md_calc_offset(M, cstrsp, pos) / (bart_stride_t)CFL_SIZE;

		float coord[M];
		float dcoord[M];

		for (int i = 0; i < M; i++) {

			coord[i] = crealf(coor[coffset + i * cstrs_dir / (bart_stride_t)CFL_SIZE]);
			dcoord[i] = crealf(dcoor[coffset + i * cstrs_dir / (bart_stride_t)CFL_SIZE]);
		}

		der_interp_point(M, gdimsp, gstrsp, grid, coord, dcoord, dintp + ioffset, ord, ord + 1);
	};

	md_parallel_loop(M, dims, ~UINT64_C(1), interp_kernel);
}


static void md_interpolate_der_coor2_int(int d, bart_flags_t flags, int ord, int N, const bart_dim_t dims[N], const bart_stride_t istrs[N], complex float* dintp, const bart_stride_t cstrs[N], const complex float* coor, const complex float* dcoor, const bart_dim_t gdims[N], const bart_stride_t gstrs[N], const complex float* grid)
{
	int M = bitcount(flags);
	bart_dim_t gdims_red[M];
	bart_dim_t gstrs_red[M];
	bart_dim_t idims_red[M];
	bart_dim_t istrs_red[M];
	bart_dim_t cstrs_red[M];

	interpolate_compute_red(d, flags, M, idims_red, istrs_red, cstrs_red, gdims_red, gstrs_red,
				N, dims, istrs, cstrs, gdims, gstrs);

	bart_dim_t pos[N];
	md_set_dims(N, pos, 0);

	do {
		bart_stride_t ioffset = md_calc_offset(N, istrs, pos) / (bart_stride_t)CFL_SIZE;
		bart_stride_t coffset = md_calc_offset(N, cstrs, pos) / (bart_stride_t)CFL_SIZE;
		bart_stride_t goffset = md_calc_offset(N, gstrs, pos) / (bart_stride_t)CFL_SIZE;

#ifdef USE_GPU
		if (cuda_ondevice(dcoor))
			cuda_interpolate_der_coor2(ord, M, idims_red, istrs_red, dintp + ioffset, cstrs_red, cstrs[d], coor + coffset, dcoor + coffset, gdims_red, gstrs_red, grid + goffset);
		else
#endif
			interpolate_der_coor2(ord, M, idims_red, istrs_red, dintp + ioffset, cstrs_red, cstrs[d], coor + coffset, dcoor + coffset, gdims_red, gstrs_red, grid + goffset);

	} while (md_next(N, dims, ~flags & ~MD_BIT(d), pos));
}


struct vptr_interpolate_der_coor_s {

	vptr_fun_data_t super;

	int d;
	int ord;
	bart_flags_t flags;
};

DEF_TYPEID(vptr_interpolate_der_coor_s);

static void md_interpolate_der_coor2_vptr(vptr_fun_data_t* _data, int N, int D, const bart_dim_t* dims[N], const bart_stride_t* strs[N], void* args[N])
{
	auto d = CAST_DOWN(vptr_interpolate_der_coor_s, _data);

	md_interpolate_der_coor2_int(d->d, d->flags, d->ord, D, dims[0], strs[0], args[0], strs[1], args[1], args[2], dims[3], strs[3], args[3]);
}

static void md_interpolate_adj_coor2_vptr(vptr_fun_data_t* _data, int N, int D, const bart_dim_t* dims[N], const bart_stride_t* strs[N], void* args[N])
{
	auto d = CAST_DOWN(vptr_interpolate_der_coor_s, _data);

	md_interpolate_adj_coor2_int(d->d, d->flags, d->ord, D, dims[0], strs[0], args[0], args[1], strs[2], args[2], dims[3], strs[3], args[3]);
}

void md_interpolate_der_coor2(int d, bart_flags_t flags, int ord, int N, const bart_dim_t dims[N], const bart_stride_t istrs[N], complex float* dintp, const bart_stride_t cstrs[N], const complex float* coor, const complex float* dcoor, const bart_dim_t gdims[N], const bart_stride_t gstrs[N], const complex float* grid)
{
	PTR_ALLOC(struct vptr_interpolate_der_coor_s, _d);
	SET_TYPEID(vptr_interpolate_der_coor_s, _d);
	_d->super.del = NULL;
	_d->d = d;
	_d->ord = ord;
	_d->flags = flags;

	exec_vptr_zfun(md_interpolate_der_coor2_vptr, CAST_UP(PTR_PASS(_d)), 4, N, ~flags & ~ MD_BIT(d), MD_BIT(0), MD_BIT(0) | MD_BIT(1) | MD_BIT(2) | MD_BIT(3),
			(const bart_dim_t*[4]) { dims, dims, dims, gdims },
			(const bart_dim_t*[4]) { istrs, cstrs, cstrs, gstrs },
			(complex float*[4]) { dintp, (void*)coor, (void*)dcoor, (void*)grid });
}


void md_interpolate_adj_coor2(int d, bart_flags_t flags, int ord, int N, const bart_dim_t dims[N], const bart_stride_t cstrs[N], const complex float* coor, complex float* dcoor, const bart_stride_t istrs[N], const complex float* dintp, const bart_dim_t gdims[N], const bart_stride_t gstrs[N], const complex float* grid)
{
	PTR_ALLOC(struct vptr_interpolate_der_coor_s, _d);
	SET_TYPEID(vptr_interpolate_der_coor_s, _d);
	_d->super.del = NULL;
	_d->d = d;
	_d->ord = ord;
	_d->flags = flags;

	exec_vptr_zfun(md_interpolate_adj_coor2_vptr, CAST_UP(PTR_PASS(_d)), 4, N, ~flags & ~ MD_BIT(d), MD_BIT(1), MD_BIT(0) | MD_BIT(1) | MD_BIT(2) | MD_BIT(3),
			(const bart_dim_t*[4]) { dims, dims, dims, gdims },
			(const bart_dim_t*[4]) { cstrs, cstrs, istrs, gstrs },
			(complex float*[4]) { (void*)coor, dcoor, (void*)dintp, (void*)grid });
}


void md_interpolate(int d, bart_flags_t flags, int ord, int N, const bart_dim_t idims[N], complex float* intp, const bart_dim_t cdims[N], const complex float* coor, const bart_dim_t gdims[N], const complex float* grid)
{
	assert(md_check_compat(N, ~UINT64_C(0), idims, cdims));

	bart_dim_t dims[N];
	md_max_dims(N, ~UINT64_C(0), dims, idims, cdims);

	md_clear(N, idims, intp, CFL_SIZE);
	md_interpolate2(d, flags, ord, N, dims, MD_STRIDES(N, idims, CFL_SIZE), intp, MD_STRIDES(N, cdims, CFL_SIZE), coor, gdims, MD_STRIDES(N, gdims, CFL_SIZE), grid);
}

void md_interpolateH(int d, bart_flags_t flags, int ord, int N, const bart_dim_t gdims[N], complex float* grid, const bart_dim_t idims[N], const complex float* intp, const bart_dim_t cdims[N], const complex float* coor)
{
	assert(md_check_compat(N, ~UINT64_C(0), idims, cdims));

	bart_dim_t dims[N];
	md_max_dims(N, ~UINT64_C(0), dims, idims, cdims);

	md_clear(N, gdims, grid, CFL_SIZE);
	md_interpolateH2(d, flags, ord, N, gdims, MD_STRIDES(N, gdims, CFL_SIZE), grid, dims, MD_STRIDES(N, idims, CFL_SIZE), intp, MD_STRIDES(N, cdims, CFL_SIZE), coor);
}

void md_interpolate_adj_coor(int d, bart_flags_t flags, int ord, int N, const bart_dim_t cdims[N], const complex float* coor, complex float* dcoor, const bart_dim_t idims[N], const complex float* dintp, const bart_dim_t gdims[N], const complex float* grid)
{
	assert(md_check_compat(N, ~UINT64_C(0), idims, cdims));

	bart_dim_t dims[N];
	md_max_dims(N, ~UINT64_C(0), dims, idims, cdims);

	md_clear(N, cdims, dcoor, CFL_SIZE);
	md_interpolate_adj_coor2(d, flags, ord, N, dims, MD_STRIDES(N, cdims, CFL_SIZE), coor, dcoor, MD_STRIDES(N, idims, CFL_SIZE), dintp, gdims, MD_STRIDES(N, gdims, CFL_SIZE), grid);
}

static void md_interpolate_adj_coor_shifted(int d, bart_flags_t flags, int ord, int N, const bart_dim_t cdims[N], const complex float* coor, complex float* dcoor, const bart_dim_t idims[N], const complex float* dintp, const bart_dim_t gdims[N], const complex float* grid)
{
	assert(md_check_compat(N, ~UINT64_C(0), idims, cdims));

	bart_dim_t dims[N];
	md_max_dims(N, ~UINT64_C(0), dims, idims, cdims);

	md_clear(N, cdims, dcoor, CFL_SIZE);

	complex float* tmp = md_alloc_sameplace(N, cdims, CFL_SIZE, dcoor);
	md_clear(N, cdims, tmp, CFL_SIZE);

	for (int i = 0; i < cdims[d]; i++) {

		md_copy(N, cdims, tmp, coor, CFL_SIZE);

		complex float* _coor = tmp + i * MD_STRIDES(N, cdims, CFL_SIZE)[d] / (bart_stride_t)CFL_SIZE;
		md_zsadd2(N, idims, MD_STRIDES(N, cdims, CFL_SIZE), _coor, MD_STRIDES(N, cdims, CFL_SIZE), _coor, -0.5);

		complex float* _dcoor = dcoor + i * MD_STRIDES(N, cdims, CFL_SIZE)[d] / (bart_stride_t)CFL_SIZE;
		md_interpolate2(d, flags, ord, N, dims, MD_STRIDES(N, cdims, CFL_SIZE), _dcoor, MD_STRIDES(N, cdims, CFL_SIZE), tmp, gdims, MD_STRIDES(N, gdims, CFL_SIZE), grid);

		md_zsmul2(N, idims, MD_STRIDES(N, cdims, CFL_SIZE), _dcoor, MD_STRIDES(N, cdims, CFL_SIZE), _dcoor, -1);

		md_zsadd2(N, idims, MD_STRIDES(N, cdims, CFL_SIZE), _coor, MD_STRIDES(N, cdims, CFL_SIZE), _coor, 1.);
		md_interpolate2(d, flags, ord, N, dims, MD_STRIDES(N, cdims, CFL_SIZE), _dcoor, MD_STRIDES(N, cdims, CFL_SIZE), tmp, gdims, MD_STRIDES(N, gdims, CFL_SIZE), grid);
	}

	md_free(tmp);

	md_zmulc2(N, dims, MD_STRIDES(N, cdims, CFL_SIZE), dcoor, MD_STRIDES(N, idims, CFL_SIZE), dintp, MD_STRIDES(N, cdims, CFL_SIZE), dcoor);
}

void md_interpolate_der_coor(int d, bart_flags_t flags, int ord, int N, const bart_dim_t idims[N], complex float* dintp, const bart_dim_t cdims[N], const complex float* coor, const complex float* dcoor, const bart_dim_t gdims[N], const complex float* grid)
{
	assert(md_check_compat(N, ~UINT64_C(0), idims, cdims));

	bart_dim_t dims[N];
	md_max_dims(N, ~UINT64_C(0), dims, idims, cdims);

	md_clear(N, idims, dintp, CFL_SIZE);
	md_interpolate_der_coor2(d, flags, ord, N, dims, MD_STRIDES(N, idims, CFL_SIZE), dintp, MD_STRIDES(N, cdims, CFL_SIZE), coor, dcoor, gdims, MD_STRIDES(N, gdims, CFL_SIZE), grid);
}

void md_resample(bart_flags_t flags, int ord, int N, const bart_dim_t _odims[N], complex float* dst, const bart_dim_t _idims[N], const complex float* src)
{
	assert(md_check_equal_dims(N, _odims, _idims, ~flags));

	bart_dim_t odims[N + 1];
	bart_dim_t idims[N + 1];
	bart_dim_t cdims[N + 1];

	md_copy_dims(N, odims, _odims);
	md_copy_dims(N, idims, _idims);
	md_select_dims(N, flags, cdims, odims);

	odims[N] = 1;
	idims[N] = 1;
	cdims[N] = bitcount(flags);

	complex float* pos = md_alloc_sameplace(N + 1, cdims, CFL_SIZE, dst);
	md_positions(N + 1, N, flags, idims, cdims, pos);

	md_interpolate(N, flags, ord, N + 1, odims, dst, cdims, pos, idims, src);

	md_free(pos);
}









struct lop_interp_s {

	linop_data_t super;

	int N;
	bart_dim_t* idims;
	bart_dim_t* cdims;
	bart_dim_t* gdims;

	int d;
	int ord;
	bart_flags_t flags;

	struct multiplace_array_s* coor;
};

static DEF_TYPEID(lop_interp_s);

static void lop_interp_free(const linop_data_t* _data)
{
	auto data = CAST_DOWN(lop_interp_s, _data);

	multiplace_free(data->coor);

	xfree(data->idims);
	xfree(data->cdims);
	xfree(data->gdims);

	xfree(data);
}

static void lop_interp(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto d = CAST_DOWN(lop_interp_s, _data);

	md_interpolate(d->d, d->flags, d->ord, d->N, d->idims, dst, d->cdims, multiplace_read(d->coor, dst), d->gdims, src);
}

static void lop_interpH(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto d = CAST_DOWN(lop_interp_s, _data);

	md_interpolateH(d->d, d->flags, d->ord, d->N, d->gdims, dst, d->idims, src, d->cdims, multiplace_read(d->coor, dst));
}

const struct linop_s* linop_interpolate_create(int d, bart_flags_t flags, int ord, int N, const bart_dim_t idims[N], const bart_dim_t cdims[N], const complex float* coor, const bart_dim_t gdims[N])
{
	PTR_ALLOC(struct lop_interp_s, data);
	SET_TYPEID(lop_interp_s, data);

	data->N = N;
	data->idims = ARR_CLONE(bart_dim_t[N], idims);
	data->cdims = ARR_CLONE(bart_dim_t[N], cdims);
	data->gdims = ARR_CLONE(bart_dim_t[N], gdims);

	data->d = d;
	data->ord = ord;
	data->flags = flags;

	data->coor = multiplace_move(N, cdims, CFL_SIZE, coor);

	return linop_create(N, idims, N, gdims,	CAST_UP(PTR_PASS(data)), lop_interp, lop_interpH, NULL, NULL, lop_interp_free);
}



struct nlop_interp_s {

	nlop_data_t super;

	int N;
	bart_dim_t* idims;
	bart_dim_t* cdims;
	bart_dim_t* gdims;

	int d;
	int ord;
	bart_flags_t flags;

	complex float* coor;
	complex float* grid;

	bool shifted_grad;
};

static DEF_TYPEID(nlop_interp_s);

static void nlop_interp_free(const nlop_data_t* _data)
{
	auto data = CAST_DOWN(nlop_interp_s, _data);

	md_free(data->coor);
	md_free(data->grid);

	xfree(data->idims);
	xfree(data->cdims);
	xfree(data->gdims);

	xfree(data);
}

static void nlop_interp_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto d = CAST_DOWN(nlop_interp_s, _data);
	assert(3 == N);

	complex float* intp = args[0];
	const complex float* grid = args[1];

	if (NULL == d->coor)
		d->coor = md_alloc_sameplace(d->N, d->cdims, CFL_SIZE, intp);

	if (NULL == d->grid)
		d->grid = md_alloc_sameplace(d->N, d->gdims, CFL_SIZE, intp);

	md_copy(d->N, d->cdims, d->coor, args[2], CFL_SIZE);
	md_copy(d->N, d->gdims, d->grid, args[1], CFL_SIZE);

	md_interpolate(d->d, d->flags, d->ord, d->N, d->idims, intp, d->cdims, d->coor, d->gdims, grid);
}

static void nlop_interp(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	auto d = CAST_DOWN(nlop_interp_s, _data);

	md_interpolate(d->d, d->flags, d->ord, d->N, d->idims, dst, d->cdims, d->coor, d->gdims, src);
}

static void nlop_interpH(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	auto d = CAST_DOWN(nlop_interp_s, _data);

	md_interpolateH(d->d, d->flags, d->ord, d->N, d->gdims, dst, d->idims, src, d->cdims, d->coor);
}

static void nlop_interp_der(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	auto d = CAST_DOWN(nlop_interp_s, _data);

	md_interpolate_der_coor(d->d, d->flags, d->ord, d->N, d->idims, dst, d->cdims, d->coor, src, d->gdims, d->grid);
}

static void nlop_interp_adj(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	auto d = CAST_DOWN(nlop_interp_s, _data);

	(d->shifted_grad ? md_interpolate_adj_coor_shifted : md_interpolate_adj_coor)(d->d, d->flags, d->ord, d->N, d->cdims, d->coor, dst, d->idims, src, d->gdims, d->grid);
}

const struct nlop_s* nlop_interpolate_create(int d, bart_flags_t flags, int ord, bool shifted_grad, int N, const bart_dim_t idims[N], const bart_dim_t cdims[N], const bart_dim_t gdims[N])
{
	PTR_ALLOC(struct nlop_interp_s, data);
	SET_TYPEID(nlop_interp_s, data);

	data->N = N;
	data->idims = ARR_CLONE(bart_dim_t[N], idims);
	data->cdims = ARR_CLONE(bart_dim_t[N], cdims);
	data->gdims = ARR_CLONE(bart_dim_t[N], gdims);

	data->shifted_grad = shifted_grad;

	data->d = d;
	data->ord = ord;
	data->flags = flags;

	data->coor = NULL;
	data->grid = NULL;

	bart_dim_t nl_odims[1][N];
	bart_dim_t nl_idims[2][N];

	md_copy_dims(N, nl_odims[0], idims);
	md_copy_dims(N, nl_idims[0], gdims);
	md_copy_dims(N, nl_idims[1], cdims);


	return nlop_generic_create(1, N, nl_odims, 2, N, nl_idims, CAST_UP(PTR_PASS(data)), nlop_interp_fun,
				   (nlop_der_fun_t[2][1]){ { nlop_interp }, { nlop_interp_der } },
				   (nlop_der_fun_t[2][1]){ { nlop_interpH }, { nlop_interp_adj } },
				   NULL, NULL, nlop_interp_free);
}




