/* Copryight 2024-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <assert.h>

#include "num/gpu_compat_runtime.h"

#include "misc/debug.h"
#include "misc/misc.h"

#include "num/gpuops.h"

#include "gpukrnls_triagmat.h"



struct cuda_strides_upper_triagmat {

	bart_dim_t N;	 //Continous vector
	bart_dim_t NC; //Coils dimension (stride N)
	bart_dim_t NM; //Matrix dimension

	//strides of matrix dimension (do not need to be contigous)
	bart_stride_t ostr;
	bart_stride_t istr;
	bart_stride_t mstr;
};

__device__ static bart_dim_t upper_triag_idx(bart_dim_t i, bart_dim_t j)
{
	if (i > j)
		return -(i + ((j + 1) * j) / 2);
	else
		return i + ((j + 1) * j) / 2;
}


__global__ static void kern_zrfmac_upper_triagmat(struct cuda_strides_upper_triagmat strs, float2* dst, const float2* src, const float* mat)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (bart_dim_t i = start; i < strs.N; i += stride) {

		for (bart_dim_t m = 0; m < strs.NM; m++) {

			for (bart_dim_t n = 0; n <= m; n++) {

				float val = mat[i + upper_triag_idx(n, m) * strs.mstr];

				if (0. == val)
					continue;

				for (bart_dim_t c = 0; c < strs.NC; c++) {

					dst[i + strs.N * c + strs.ostr * m].x += val * src[i + strs.N * c + strs.istr * n].x;
					dst[i + strs.N * c + strs.ostr * m].y += val * src[i + strs.N * c + strs.istr * n].y;

					if (n != m) {

						dst[i + strs.N * c + strs.ostr * n].x += val * src[i + strs.N * c + strs.istr * m].x;
						dst[i + strs.N * c + strs.ostr * n].y += val * src[i + strs.N * c + strs.istr * m].y;
					}
				}
			}
		}
	}
}

#define BLOCKSIZE 1024

static int blocksize(bart_dim_t N)
{
	return BLOCKSIZE;
}

static bart_dim_t gridsize(bart_dim_t N)
{
	// to ensure that "start" does not overflow we need to restrict gridsize!
	return MIN((N + BLOCKSIZE - 1) / BLOCKSIZE, 65536 - 1);
}


extern "C" void cuda_zrfmac_upper_triagmat(bart_dim_t N, bart_dim_t NC, bart_dim_t NM, bart_stride_t ostr, bart_stride_t istr, bart_stride_t mstr, float* dst, const float* src, const float* mat)
{
	cuda_strides_upper_triagmat conf;
	conf.N = N;
	conf.NC = NC;
	conf.NM = NM;
	conf.ostr = ostr / 2; //use float2 for faster access
	conf.istr = istr / 2; //use float2 for faster access
	conf.mstr = mstr;

	kern_zrfmac_upper_triagmat<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(conf, (float2*)dst, (float2*)src, mat);
	CUDA_KERNEL_ERROR;
}
