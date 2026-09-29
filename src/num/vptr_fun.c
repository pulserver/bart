/* Copyright 2024-2026. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 *
*/

#include <assert.h>
#include <string.h>
#include <complex.h>

#include "misc/misc.h"

#include "num/multind.h"
#include "num/vptr.h"
#include "num/delayed.h"

#include "vptr_fun.h"


void exec_vptr_fun_internal(vptr_fun_t fun, vptr_fun_data_t* data, int N, int D, bart_flags_t lflags, const bart_dim_t* dims[N], const bart_stride_t* strs[N], void* ptr[N], size_t sizes[N], bool resolve)
{
	bart_dim_t ldims[D];
	md_select_dims(D, lflags, ldims, dims[0]);

	for (int i = 1; i < N; i++) {

		bart_dim_t tdims[D];
		md_select_dims(D, lflags, tdims, dims[i]);

		assert(md_check_compat(D, ~UINT64_C(0), tdims, ldims));
		md_max_dims(D, ~UINT64_C(0), ldims, ldims, tdims);
	}

	bart_flags_t vptr_loop_flags = 0;

	for (int i = 0; i < N; i++)
		vptr_loop_flags |= vptr_block_loop_flags(D, ldims, strs[i], ptr[i], sizes[i], true);

	bart_dim_t tdims[N][D];
	bart_stride_t tstrs[N][D];

	const bart_dim_t* ndims[N];
	const bart_stride_t* nstrs[N];
	void* sptr[N];

	for (int i = 0; i < N; i++) {

		md_select_dims(D, ~vptr_loop_flags, tdims[i], dims[i]);
		md_select_strides(D, ~vptr_loop_flags, tstrs[i], strs[i]);

		ndims[i] = tdims[i];
		nstrs[i] = tstrs[i];
	}

	bart_dim_t pos[D];
	md_set_dims(D, pos, 0);

	do {
		for(int i = 0; i < N; i++)
			sptr[i] = ptr[i] + md_calc_offset(D, strs[i], pos);

		if (resolve && !mpi_accessible_mult(N, (const void**)sptr))
			continue;

		for (int i = 0; i < N && resolve; i++) {

			vptr_contiguous_strs(D, sptr[i], vptr_loop_flags, tstrs[i], strs[i]);
			sptr[i] = vptr_resolve(sptr[i]);
		}

		fun(data, N, D, ndims, nstrs, sptr);

	} while (md_next(D, ldims, vptr_loop_flags, pos));
}


void exec_vptr_fun_gen(vptr_fun_t fun, vptr_fun_data_t* data, int N, int D, bart_flags_t lflags, bart_flags_t wflags, bart_flags_t rflags, const bart_dim_t* dims[N], const bart_stride_t* strs[N], void* _ptr[N], size_t sizes[N], bool resolve)
{
	for (int i = 1; i < N; i++)
		assert(is_vptr(_ptr[0]) == is_vptr(_ptr[i]));

	void* ptr[N];
	for(int i = 0; i < N; i++)
		ptr[i] = vptr_resolve_range(_ptr[i]);

	if (!is_vptr(ptr[0])) {

		fun(data, N, D, dims, strs, ptr);

		if (NULL != data->del)
			data->del(data);

		xfree(data);

		return;
	}

	if (is_delayed(ptr[0])) {

		exec_vptr_fun_delayed(fun, data, N, D, lflags, wflags, rflags, dims, strs, ptr, sizes, resolve);
		return;
	}

	exec_vptr_fun_internal(fun, data, N, D, lflags, dims, strs, ptr, sizes, resolve);

	if (NULL != data->del)
		data->del(data);

	xfree(data);
}


void exec_vptr_zfun(vptr_fun_t fun, vptr_fun_data_t* data, int N, int D, bart_flags_t lflags, bart_flags_t wflags, bart_flags_t rflags, const bart_dim_t* dims[N], const bart_stride_t* strs[N], complex float* cptr[N])
{
	size_t sizes[N];
	void* ptr[N];

	for(int i = 0; i < N; i++) {

		ptr[i] = cptr[i];
		sizes[i] = sizeof(complex float);
	}

	exec_vptr_fun_gen(fun, data, N, D, lflags, wflags, rflags, dims, strs, ptr, sizes, true);
}


void exec_vptr_fun(vptr_fun_t fun, vptr_fun_data_t* data, int N, int D, bart_flags_t lflags, bart_flags_t wflags, bart_flags_t rflags, const bart_dim_t* dims[N], const bart_stride_t* strs[N], float* cptr[N])
{
	size_t sizes[N];
	void* ptr[N];

	for(int i = 0; i < N; i++) {

		ptr[i] = cptr[i];
		sizes[i] = sizeof(float);
	}

	exec_vptr_fun_gen(fun, data, N, D, lflags, wflags, rflags, dims, strs, ptr, sizes, true);
}


