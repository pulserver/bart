/* Copyright 2013-2015 The Regents of the University of California.
 * Copyright 2026. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 **/

#include "num/multind.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "misc/debug.h"
#include "misc/misc.h"

#include "shuffle.h"


#if 0
void md_shuffle2(int N, const bart_dim_t dims[N], const bart_dim_t factors[N],
		const bart_stride_t ostrs[N], void* out, const bart_stride_t istrs[N], const void* in, size_t size)
{
	bart_dim_t dims2[2 * N];
	bart_stride_t ostrs2[2 * N];
	bart_stride_t istrs2[2 * N];

	for (int i = 0; i < N; i++) {

		assert(0 == dims[i] % factors[i]);

		bart_dim_t f2 = dims[i] / factors[i];

		dims2[0 * N + i] = f2;
		dims2[1 * N + i] = factors[i];

		ostrs2[1 * N + i] = ostrs[i];
		ostrs2[0 * N + i] = ostrs[i] * f2;

		istrs2[0 * N + i] = istrs[i] * factors[i];
		istrs2[1 * N + i] = istrs[i];
	}

	md_copy2(2 * N, dims2, ostrs2, out, istrs2, in, size);
}

void md_shuffle(int N, const bart_dim_t dims[N], const bart_dim_t factors[N],
		void* out, const void* in, size_t size)
{
	bart_stride_t strs[N];
	md_calc_strides(N, strs, dims, size);

	md_shuffle2(N, dims, factors, strs, out, strs, in, size);
}
#endif


static void decompose_dims(int N, bart_dim_t dims2[2 * N], bart_stride_t ostrs2[2 * N], bart_stride_t istrs2[2 * N],
		const bart_dim_t factors[N], const bart_dim_t odims[N + 1], const bart_stride_t ostrs[N + 1], const bart_dim_t idims[N], const bart_stride_t istrs[N])
{
	bart_dim_t prod = 1;

	for (int i = 0; i < N; i++) {

		bart_dim_t f2 = idims[i] / factors[i];

		assert(0 == idims[i] % factors[i]);
		assert(odims[i] == idims[i] / factors[i]);

		dims2[1 * N + i] = factors[i];
		dims2[0 * N + i] = f2;

		istrs2[0 * N + i] = istrs[i] * factors[i];
		istrs2[1 * N + i] = istrs[i];

		ostrs2[0 * N + i] = ostrs[i];
		ostrs2[1 * N + i] = ostrs[N] * prod;

		prod *= factors[i];
	}

	assert(odims[N] == prod);
}

void md_decompose2(int N, const bart_dim_t factors[N],
		const bart_dim_t odims[N + 1], const bart_stride_t ostrs[N + 1], void* out,
		const bart_dim_t idims[N], const bart_stride_t istrs[N], const void* in, size_t size)
{
	bart_dim_t dims2[2 * N];
	bart_stride_t ostrs2[2 * N];
	bart_stride_t istrs2[2 * N];

	decompose_dims(N, dims2, ostrs2, istrs2, factors, odims, ostrs, idims, istrs);

	md_copy2(2 * N, dims2, ostrs2, out, istrs2, in, size);
}

void md_decompose(int N, const bart_dim_t factors[N], const bart_dim_t odims[N + 1],
		void* out, const bart_dim_t idims[N], const void* in, size_t size)
{
	bart_stride_t ostrs[N + 1];
	md_calc_strides(N + 1, ostrs, odims, size);

	bart_stride_t istrs[N];
	md_calc_strides(N, istrs, idims, size);

	md_decompose2(N, factors, odims, ostrs, out, idims, istrs, in, size);
}

void md_recompose2(int N, const bart_dim_t factors[N],
		const bart_dim_t odims[N], const bart_stride_t ostrs[N], void* out,
		const bart_dim_t idims[N + 1], const bart_stride_t istrs[N + 1], const void* in, size_t size)
{
	bart_dim_t dims2[2 * N];
	bart_stride_t ostrs2[2 * N];
	bart_stride_t istrs2[2 * N];

	decompose_dims(N, dims2, istrs2, ostrs2, factors, idims, istrs, odims, ostrs);

	md_copy2(2 * N, dims2, ostrs2, out, istrs2, in, size);
}

void md_recompose(int N, const bart_dim_t factors[N], const bart_dim_t odims[N],
		void* out, const bart_dim_t idims[N + 1], const void* in, size_t size)
{
	bart_stride_t ostrs[N];
	md_calc_strides(N, ostrs, odims, size);

	bart_stride_t istrs[N + 1];
	md_calc_strides(N + 1, istrs, idims, size);

	md_recompose2(N, factors, odims, ostrs, out, idims, istrs, in, size);
}

