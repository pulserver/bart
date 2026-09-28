/* Copyright 2023. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Bernhard Rapp
 *
*/

#include <complex.h>
#include <assert.h>
#include <stdio.h>

#include "misc/debug.h"
#include "num/mpi_ops.h"
#include "num/vptr.h"
#include "num/multind.h"

#include "utest.h"

#define N 10
#define D 16

//shift flags according to complex to real conversion
static bool test_mpi_get_flags_C2R(void)
{
	const size_t CFL_SIZE = sizeof(complex float);
	const size_t FL_SIZE = sizeof(float);

	/* so far dims is ignored in mpi_get_flags */

	const bart_flags_t flags = 8;
	const bart_dim_t cdims[N] = { 32, 32, 1, 8, 1, 1, 1, 1, 1, 1};

	complex float* ptr = md_alloc_mpi(N, flags, cdims, CFL_SIZE);

	bart_stride_t cstrs[N];
	md_calc_strides(N, cstrs, cdims, CFL_SIZE);

	const bart_dim_t rdims[N + 1] = { 2, 32, 32, 1, 8, 1, 1, 1, 1, 1, 1};
	bart_stride_t rstrs[N + 1];
	md_calc_strides(N, rstrs, rdims, FL_SIZE);

	//return
	const bart_flags_t complex_flags = vptr_block_loop_flags(N, cdims, cstrs, ptr, CFL_SIZE, false);
	const bart_flags_t real_flags = vptr_block_loop_flags(N, rdims, rstrs, ptr, FL_SIZE, false);

	md_free(ptr);

#ifdef USE_MPI
	UT_RETURN_ASSERT((flags == complex_flags) && (complex_flags == (real_flags >> 1)));
#else
	UT_RETURN_ASSERT(0 == (complex_flags | real_flags));
#endif
}
UT_REGISTER_TEST(test_mpi_get_flags_C2R);

//Clear flags on sliced dims
static bool test_mpi_get_flags_slice(void)
{
	const size_t CFL_SIZE = sizeof(complex float);

	/* so far dims is ignored in mpi_get_flags */

	const bart_flags_t flags = 8;
	const bart_dim_t dims[N] = { 32, 32, 1, 8, 1, 1, 1, 1, 1, 1};

	bart_stride_t strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	complex float* ptr = md_alloc_mpi(N, flags, dims, CFL_SIZE);

	bart_stride_t sstrs[N];
	md_select_strides(N, ~flags, sstrs, strs);

	bart_flags_t f = vptr_block_loop_flags(N, dims, sstrs, ptr, CFL_SIZE, false);

	md_free(ptr);

	UT_RETURN_ASSERT(0 == f);
}

UT_REGISTER_TEST(test_mpi_get_flags_slice);

//keep dims if not reshaped on distributed dimensions
static bool test_mpi_get_flags_reshape(void)
{
	const size_t CFL_SIZE = sizeof(complex float);

	const bart_flags_t flags = 8;
	const bart_dim_t dims[N] = { 32, 32, 1, 8, 1, 1, 1, 1, 1, 1};

	complex float* ptr = md_alloc_mpi(N, flags, dims, CFL_SIZE);

	const bart_dim_t reshape1_dims[N] = { 32, 16, 2, 8, 1, 1, 1, 1, 1};
	bart_stride_t reshape1_strs[N];
	md_calc_strides(N, reshape1_strs, reshape1_dims, CFL_SIZE);

	bart_flags_t f = vptr_block_loop_flags(N, dims, reshape1_strs, ptr, CFL_SIZE, false);

	md_free(ptr);

#ifdef USE_MPI
	UT_RETURN_ASSERT(flags == f);
#else
	UT_RETURN_ASSERT(0 == f);
#endif
}
UT_REGISTER_TEST(test_mpi_get_flags_reshape);


static bool test_mpi_get_flags_roi(void)
{
	const size_t CFL_SIZE = sizeof(complex float);

	const bart_flags_t flags = 8;

	bart_dim_t dims[N] = { 128, 128, 1, 8, 1, 1, 1, 1, 1, 1};
	bart_stride_t strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	complex float* ptr = md_alloc_mpi(N, flags, dims, CFL_SIZE);

	bart_dim_t roi[N] = { 128, 32, 1, 8, 1, 1, 1, 1, 1, 1};


	bart_flags_t f1 = vptr_block_loop_flags(N, roi, strs, ptr, CFL_SIZE, false);

	md_free(ptr);
#ifdef USE_MPI
	UT_RETURN_ASSERT(f1 == flags);
#else
	UT_RETURN_ASSERT(0 == f1);
#endif
}

UT_REGISTER_TEST(test_mpi_get_flags_roi);

