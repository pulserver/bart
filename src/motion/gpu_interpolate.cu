/* Copyright 2024. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include "num/gpu_compat_runtime.h"
#include "num/gpu_compat_complex.h"

#include <math.h>
#include <complex.h>
#include <assert.h>

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/gpukrnls_misc.h"
#include "num/gpuops.h"

#include "misc/misc.h"

#include "gpu_interpolate.h"


__device__ void device_unravel_index(int D, bart_dim_t pos[__VLA(D)], bart_flags_t flags, const bart_dim_t dims[__VLA(D)], bart_dim_t index)
{
	bart_dim_t ind = index;

	for (int d = 0; d < D; ++d) {

		if (!MD_IS_SET(flags, d))
			continue;

		pos[d] = ind % dims[d];
		ind /= dims[d];
	}
}

#define MAXPOS 17

struct pos_data {

	int N;
	int d;
	bart_dim_t sdims[MAXPOS];
	bart_dim_t pdims[MAXPOS];
	bart_dim_t map[MAXPOS];
};


__global__ void kern_positions(const struct pos_data pd, bart_dim_t N, cuFloatComplex* dst)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	bart_dim_t pos[MAXPOS] = { 0 };

	for (bart_dim_t i = start; i < N; i += stride) {

		device_unravel_index(pd.N, pos, ~UINT64_C(0), pd.pdims, i);

		float world = (pos[pd.map[pos[pd.d]]] - (pd.pdims[pd.map[pos[pd.d]]] / 2)) / (float)pd.pdims[pd.map[pos[pd.d]]];
		float ret = pd.sdims[pd.map[pos[pd.d]]] * world + (pd.sdims[pd.map[pos[pd.d]]] / 2);

		dst[i] = make_cuFloatComplex(ret, 0.);
	}
}

void cuda_positions(int N, int d, bart_flags_t flags, const bart_dim_t sdims[__VLA(N)], const bart_dim_t pdims[__VLA(N)], _Complex float* pos)
{
	assert(N <= MAXPOS);

	struct pos_data pd;
	pd.N = N;
	pd.d = d;

	md_copy_dims(N, pd.sdims, sdims);
	md_copy_dims(N, pd.pdims, pdims);

	for (int i = 0, ip = 0; i < N; i++)
		if (MD_IS_SET(flags, i))
			pd.map[ip++] = i;

	bart_dim_t tot = md_calc_size(N, pdims);

	dim3 cu_block = getBlockSize(tot, (const void*)kern_positions);
	dim3 cu_grid = getGridSize(tot, (const void*)kern_positions);

	kern_positions<<<cu_grid, cu_block, 0, cuda_get_stream() >>>(pd, tot, (cuFloatComplex*)pos);

	CUDA_KERNEL_ERROR;
}




__device__ static float spline_nn(float x)
{
	if (-0.5 <= x && x < 0.5)
		return 1;

	return 0.;
}

__device__ static float dspline_nn(float x)
{
	return 0.;
}

__device__ static float spline_lerp(float x)
{
	if (-1. <= x && x < 1.)
		return 1. - fabs(x);

	return 0.;
}

__device__ static float dspline_lerp(float x)
{
	if (-1. <= x && x < 1.)
		return (x < 0) ? 1 : -1;

	return 0.;
}

__device__ static float spline_keys(float x)
{
	float a = -0.5;
	x = fabsf(x);

	if (x < 1)
		return (a + 2.) * x * x * x - (a + 3.) * x * x + 1.;

	if (x < 2)
		return a * x * x * x - 5. * a * x * x + 8. * a * x - 4 * a;

	return 0.;
}

__device__ static float dspline_keys(float x)
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

__device__ static float spline(int ord, float x)
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

__device__ static float dspline(int ord, float x)
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

#define INTP_DIMS 3

struct intp_data {

	int ord;
	float width;
	bart_stride_t coor_dir_dim_str;

	int N;
	bart_dim_t grid_dims[INTP_DIMS];
	bart_dim_t intp_dims[INTP_DIMS];

	bart_stride_t intp_strs[INTP_DIMS];
	bart_stride_t coor_strs[INTP_DIMS];
	bart_stride_t grid_strs[INTP_DIMS];

};

struct intp_data_device {

	float coor[INTP_DIMS];
	float dcoor[INTP_DIMS];
	int pos_grid[INTP_DIMS];
	int sti[INTP_DIMS];
	int eni[INTP_DIMS];
	int dir;
};

__device__ static __inline__ void dev_atomic_zadd_scl(cuFloatComplex* arg, cuFloatComplex val, float scl)
{
	atomicAdd(&(arg->x), val.x * scl);
	atomicAdd(&(arg->y), val.y * scl);
}

__device__ static inline struct intp_data_device get_intp_data_device(const struct intp_data* conf, const cuFloatComplex* coor, const cuFloatComplex* dcoor)
{
	struct intp_data_device idd;

	for (int j = 0; j < INTP_DIMS; j++) {

		idd.coor[j] = 0.;
		idd.dcoor[j] = 0.;
		idd.sti[j] = 0;
		idd.eni[j] = 0;
	}

	for (int j = 0; j < conf->N; j++) {
	
		idd.coor[j] = coor[conf->coor_dir_dim_str * j].x;

		if (NULL != dcoor)
			idd.dcoor[j] = dcoor[conf->coor_dir_dim_str * j].x;

		idd.sti[j] = (int)ceil(idd.coor[j] - 0.499999 * conf->width);
		idd.eni[j] = (int)floor(idd.coor[j] + 0.499999 * conf->width);
		
		idd.sti[j] = MAX(idd.sti[j], 0);
		idd.eni[j] = MIN(idd.eni[j], conf->grid_dims[j] - 1);
	}

	return idd;
}


template<bool adjoint>
__device__ static void intp_point_r(const struct intp_data* id, const struct intp_data_device* idd, cuFloatComplex* dst, const cuFloatComplex* src)
{
	assert(3 == INTP_DIMS);

	float d[INTP_DIMS];
	bart_dim_t ind[INTP_DIMS];

	for (bart_dim_t z = idd->sti[2]; z <= idd->eni[2]; z++) {

		d[2] = spline(id->ord, (idd->coor[2] - (float)z));
		ind[2] = z * id->grid_strs[2];

		for (bart_dim_t y = idd->sti[1]; y <= idd->eni[1]; y++) {

			d[1] = spline(id->ord, (idd->coor[1] - (float)y)) * d[2];
			ind[1] = y * id->grid_strs[1] + ind[2];

			for (bart_dim_t x = idd->sti[0]; x <= idd->eni[0]; x++) {

				d[0] = spline(id->ord, (idd->coor[0] - (float)x)) * d[1];
				ind[0] = x * id->grid_strs[0] + ind[1];

				if (adjoint)
					dev_atomic_zadd_scl(dst + ind[0] , src[0], d[0]);
				else
					dev_atomic_zadd_scl(dst, src[ind[0]], d[0]);
			}
		}
	}
}


template<bool adjoint>
__global__ static void kern_intp(struct intp_data conf, const cuFloatComplex* coor, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start[3];
	int stride[3];

	start[0] = threadIdx.x + blockDim.x * blockIdx.x;
	start[1] = threadIdx.y + blockDim.y * blockIdx.y;
	start[2] = threadIdx.z + blockDim.z * blockIdx.z;

	stride[0] = blockDim.x * gridDim.x;
	stride[1] = blockDim.y * gridDim.y;
	stride[2] = blockDim.z * gridDim.z;

	bart_dim_t pos[3];

	struct intp_data_device idd;

	for (pos[2] = start[2]; pos[2] < conf.intp_dims[2]; pos[2] += stride[2])
	for (pos[1] = start[1]; pos[1] < conf.intp_dims[1]; pos[1] += stride[1])
	for (pos[0] = start[0]; pos[0] < conf.intp_dims[0]; pos[0] += stride[0]) {


		bart_stride_t offset_coor = 0;
		bart_stride_t offset_intp = 0;

		for (int i = 0; i < conf.N; i++) {

			offset_coor += conf.coor_strs[i] * pos[i];
			offset_intp += conf.intp_strs[i] * pos[i];
		}
		
		idd = get_intp_data_device(&conf, coor + offset_coor, NULL);

		if (adjoint)
			intp_point_r<true>(&conf, &idd, dst, src + offset_intp);
		else
			intp_point_r<false>(&conf, &idd, dst + offset_intp, src);
	}
}

static struct intp_data cuda_intp_get_data(int M, 
			const bart_dim_t grid_dims[__VLA(M)], const bart_stride_t grid_strs[__VLA(M)],
			const bart_dim_t intp_dims[__VLA(M)], const bart_stride_t intp_strs[__VLA(M)],
							const bart_stride_t coor_strs[__VLA(M)], bart_stride_t coor_dir_dim_str, int ord, float width)
{
	struct intp_data id = {

		.ord = ord,
		.width = width,
		.coor_dir_dim_str = coor_dir_dim_str / (bart_stride_t)CFL_SIZE,
		
		.N = M,
	};

	md_copy_dims(M, id.grid_dims, grid_dims);
	md_singleton_dims(INTP_DIMS, id.intp_dims);
	md_copy_dims(M, id.intp_dims, intp_dims);

	md_copy_dims(M, id.grid_strs, grid_strs);
	md_copy_dims(M, id.intp_strs, intp_strs);
	md_copy_dims(M, id.coor_strs, coor_strs);

	for (int i = 0; i < M; i++) {

		id.coor_strs[i] /= CFL_SIZE;
		id.intp_strs[i] /= CFL_SIZE;
		id.grid_strs[i] /= CFL_SIZE;
	}

	return id;
}

template<bool adjoint>
static void cuda_intp_temp(int M, 
			const bart_dim_t grid_dims[__VLA(M)], const bart_stride_t grid_strs[__VLA(M)], _Complex float* grid,
			const bart_dim_t intp_dims[__VLA(M)], const bart_stride_t intp_strs[__VLA(M)], _Complex float* intp,
							const bart_stride_t coor_strs[__VLA(M)], bart_stride_t coor_dir_dim_str, const _Complex float* coor,
			int ord, float width)
{
	struct intp_data id = cuda_intp_get_data(M, grid_dims, grid_strs, intp_dims, intp_strs, coor_strs, coor_dir_dim_str, ord, width);

	dim3 cu_block = getBlockSize3((const bart_dim_t*)id.intp_dims, (const void*)kern_intp<adjoint>);
	dim3 cu_grid = getGridSize3((const bart_dim_t*)id.intp_dims, (const void*)kern_intp<adjoint>);

	if (adjoint)
		kern_intp<true><<<cu_grid, cu_block, 0, cuda_get_stream()>>>(id, (const cuFloatComplex*)coor, (cuFloatComplex*)grid, (const cuFloatComplex*)intp);
	else
		kern_intp<false><<<cu_grid, cu_block, 0, cuda_get_stream()>>>(id, (const cuFloatComplex*)coor, (cuFloatComplex*)intp, (const cuFloatComplex*)grid);

	CUDA_KERNEL_ERROR;
}

void cuda_interpolate2(int ord, int M, 
			const bart_dim_t intp_dims[__VLA(M)], const bart_stride_t intp_strs[__VLA(M)], _Complex float* intp,
							const bart_stride_t coor_strs[__VLA(M)], bart_stride_t coor_dir_dim_str, const _Complex float* coor,
			const bart_dim_t grid_dims[__VLA(M)], const bart_stride_t grid_strs[__VLA(M)], const _Complex float* grid)
{
	cuda_intp_temp<false>(M, grid_dims, grid_strs, (_Complex float*)grid, intp_dims, intp_strs, intp, coor_strs, coor_dir_dim_str, coor, ord, ord + 1);
}

void cuda_interpolateH2(int ord, int M, 
		const bart_dim_t grid_dims[__VLA(M)], const bart_stride_t grid_strs[__VLA(M)], _Complex float* grid,
		const bart_dim_t intp_dims[__VLA(M)], const bart_stride_t intp_strs[__VLA(M)], const _Complex float* intp,
						const bart_stride_t coor_strs[__VLA(M)], bart_stride_t coor_dir_dim_str, const _Complex float* coor)
{
	cuda_intp_temp<true>(M, grid_dims, grid_strs, grid, intp_dims, intp_strs, (_Complex float*)intp, coor_strs, coor_dir_dim_str, coor, ord, ord + 1);
}













__device__ static void intp_point_adj_coor(const struct intp_data* id, const struct intp_data_device* idd, cuFloatComplex* dcoor, const cuFloatComplex dintp, const cuFloatComplex* grid)
{
	assert(3 == INTP_DIMS);

	float d[INTP_DIMS];
	bart_dim_t ind[INTP_DIMS];

	for (bart_dim_t z = idd->sti[2]; z <= idd->eni[2]; z++) {

		d[2] = ((2 == idd->dir) ? dspline: spline)(id->ord, (idd->coor[2] - (float)z));
		ind[2] = z * id->grid_strs[2];

		for (bart_dim_t y = idd->sti[1]; y <= idd->eni[1]; y++) {

			d[1] = ((1 == idd->dir) ? dspline: spline)(id->ord, (idd->coor[1] - (float)y)) * d[2];
			ind[1] = y * id->grid_strs[1] + ind[2];

			for (bart_dim_t x = idd->sti[0]; x <= idd->eni[0]; x++) {

				d[0] = ((0 == idd->dir) ? dspline: spline)(id->ord, (idd->coor[0] - (float)x)) * d[1];
				ind[0] = x * id->grid_strs[0] + ind[1];

				cuFloatComplex val = cuCmulf(dintp, cuConjf(grid[ind[0]]));
				val.y = 0.;

				dev_atomic_zadd_scl(dcoor, val, d[0]);
			}
		}
	}
}


__global__ static void kern_intp_point_adj_coor(struct intp_data conf, const cuFloatComplex* coor, cuFloatComplex* dcoor, const cuFloatComplex* grid, const cuFloatComplex* dintp)
{
	int start[3];
	int stride[3];

	start[0] = threadIdx.x + blockDim.x * blockIdx.x;
	start[1] = threadIdx.y + blockDim.y * blockIdx.y;
	start[2] = threadIdx.z + blockDim.z * blockIdx.z;

	stride[0] = blockDim.x * gridDim.x;
	stride[1] = blockDim.y * gridDim.y;
	stride[2] = blockDim.z * gridDim.z;

	bart_dim_t pos[3];

	struct intp_data_device idd;

	for (pos[2] = start[2]; pos[2] < conf.intp_dims[2]; pos[2] += stride[2])
	for (pos[1] = start[1]; pos[1] < conf.intp_dims[1]; pos[1] += stride[1])
	for (pos[0] = start[0]; pos[0] < conf.intp_dims[0]; pos[0] += stride[0]) {

		bart_stride_t offset_coor = 0;
		bart_stride_t offset_intp = 0;

		for (int i = 0; i < conf.N; i++) {

			offset_coor += conf.coor_strs[i] * pos[i];
			offset_intp += conf.intp_strs[i] * pos[i];
		}

		idd = get_intp_data_device(&conf, coor + offset_coor, NULL);

		for (int i = 0; i < conf.N; i++) {

			idd.dir = i;
			intp_point_adj_coor(&conf, &idd, dcoor + offset_coor + i * conf.coor_dir_dim_str, dintp[offset_intp], grid);
		}
	}
}


void cuda_interpolate_adj_coor2(int ord, int M, 
			const bart_dim_t intp_dims[__VLA(M)], const bart_stride_t intp_strs[__VLA(M)], const _Complex float* dintp,
							const bart_stride_t coor_strs[__VLA(M)], bart_stride_t coor_dir_dim_str, const _Complex float* coor, _Complex float* dcoor,
			const bart_dim_t grid_dims[__VLA(M)], const bart_stride_t grid_strs[__VLA(M)], const _Complex float* grid)
{
	struct intp_data id = cuda_intp_get_data(M, grid_dims, grid_strs, intp_dims, intp_strs, coor_strs, coor_dir_dim_str, ord, ord + 1);

	dim3 cu_block = getBlockSize3((const bart_dim_t*)id.intp_dims, (const void*)kern_intp_point_adj_coor);
	dim3 cu_grid = getGridSize3((const bart_dim_t*)id.intp_dims, (const void*)kern_intp_point_adj_coor);

	kern_intp_point_adj_coor<<<cu_grid, cu_block, 0, cuda_get_stream() >>>(id, (const cuFloatComplex*)coor, (cuFloatComplex*)dcoor, (const cuFloatComplex*)grid, (const cuFloatComplex*)dintp);

	CUDA_KERNEL_ERROR;
}





__device__ static void intp_point_der_coor(const struct intp_data* id, const struct intp_data_device* idd, cuFloatComplex* dintp, const cuFloatComplex* grid)
{
	assert(3 == INTP_DIMS);

	float d[INTP_DIMS];
	float dd[INTP_DIMS];
	bart_dim_t ind[INTP_DIMS];

	for (bart_dim_t z = idd->sti[2]; z <= idd->eni[2]; z++) {

		dd[2] = spline(id->ord, (idd->coor[2] - (float)z));
		
		d[2] = dspline(id->ord, (idd->coor[2] - (float)z)) * idd->dcoor[2];

		ind[2] = z * id->grid_strs[2];

		for (bart_dim_t y = idd->sti[1]; y <= idd->eni[1]; y++) {

			dd[1] = spline(id->ord, (idd->coor[1] - (float)y)) * dd[2];
			
			d[1] = dspline(id->ord, (idd->coor[1] - (float)y)) * idd->dcoor[1] * dd[2];
			d[1] += spline(id->ord, (idd->coor[1] - (float)y)) * d[2];
			
			ind[1] = y * id->grid_strs[1] + ind[2];

			for (bart_dim_t x = idd->sti[0]; x <= idd->eni[0]; x++) {

				dd[0] = spline(id->ord, (idd->coor[0] - (float)x)) * dd[1];
				
				d[0] = dspline(id->ord, (idd->coor[0] - (float)x)) * idd->dcoor[0] * dd[1];
				d[0] += spline(id->ord, (idd->coor[0] - (float)x)) * d[1];
				
				ind[0] = x * id->grid_strs[0] + ind[1];

				dev_atomic_zadd_scl(dintp, grid[ind[0]], d[0]);
			}
		}
	}
}


__global__ static void kern_intp_point_der_coor(struct intp_data conf, const cuFloatComplex* coor, const cuFloatComplex* dcoor, const cuFloatComplex* grid, cuFloatComplex* dintp)
{
	int start[3];
	int stride[3];

	start[0] = threadIdx.x + blockDim.x * blockIdx.x;
	start[1] = threadIdx.y + blockDim.y * blockIdx.y;
	start[2] = threadIdx.z + blockDim.z * blockIdx.z;

	stride[0] = blockDim.x * gridDim.x;
	stride[1] = blockDim.y * gridDim.y;
	stride[2] = blockDim.z * gridDim.z;

	bart_dim_t pos[3];

	struct intp_data_device idd;

	for (pos[2] = start[2]; pos[2] < conf.intp_dims[2]; pos[2] += stride[2])
	for (pos[1] = start[1]; pos[1] < conf.intp_dims[1]; pos[1] += stride[1])
	for (pos[0] = start[0]; pos[0] < conf.intp_dims[0]; pos[0] += stride[0]) {

		bart_stride_t offset_coor = 0;
		bart_stride_t offset_intp = 0;

		for (int i = 0; i < conf.N; i++) {

			offset_coor += conf.coor_strs[i] * pos[i];
			offset_intp += conf.intp_strs[i] * pos[i];
		}

		idd = get_intp_data_device(&conf, coor + offset_coor, dcoor + offset_coor);

		intp_point_der_coor(&conf, &idd, dintp + offset_intp, grid);
	}
}


void cuda_interpolate_der_coor2(int ord, int M, 
			const bart_dim_t intp_dims[__VLA(M)], const bart_stride_t intp_strs[__VLA(M)], _Complex float* dintp,
							const bart_stride_t coor_strs[__VLA(M)], bart_stride_t coor_dir_dim_str, const _Complex float* coor, const _Complex float* dcoor,
			const bart_dim_t grid_dims[__VLA(M)], const bart_stride_t grid_strs[__VLA(M)], const _Complex float* grid)
{
	struct intp_data id = cuda_intp_get_data(M, grid_dims, grid_strs, intp_dims, intp_strs, coor_strs, coor_dir_dim_str, ord, ord + 1);

	dim3 cu_block = getBlockSize3((const bart_dim_t*)id.intp_dims, (const void*)kern_intp_point_der_coor);
	dim3 cu_grid = getGridSize3((const bart_dim_t*)id.intp_dims, (const void*)kern_intp_point_der_coor);

	kern_intp_point_der_coor<<<cu_grid, cu_block, 0, cuda_get_stream() >>>(id, (const cuFloatComplex*)coor, (const cuFloatComplex*)dcoor, (const cuFloatComplex*)grid, (cuFloatComplex*)dintp);

	CUDA_KERNEL_ERROR;
}





