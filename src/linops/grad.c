/* Copyright 2014-2015. The Regents of the University of California.
 * Copyright 2016-2019. Martin Uecker.
 * Copyright 2024-2026. Institute of Biomedical Imaging. TU Graz.
 * Copyright 2026. Department of Radiology. Boston Children's Hospital.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <assert.h>
#include <strings.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/laplace.h"

#include "linops/linop.h"

#include "misc/misc.h"

#include "grad.h"


typedef void (*md_zfdiff_core_t)(int D, const bart_dim_t dims[D], int d, bool adj, const bart_stride_t ostr[D], complex float* out, const bart_stride_t istr[D], const complex float* in);

static void md_zfdiff_core2(int D, const bart_dim_t dims[D], int d, bool dir, bool adj, const bart_stride_t ostr[D], complex float* out, const bart_stride_t istr[D], const complex float* in)
{
	bart_dim_t pos[D];
	md_set_dims(D, pos, 0);

	if (adj)
		pos[d] = dir ? 1 : -1;
	else
		pos[d] = dir ? -1 : 1;

	md_circ_shift2(D, dims, pos, ostr, out, istr, in, CFL_SIZE);

	if (dir)
		md_zsub2(D, dims, ostr, out, ostr, out, istr, in);
	else
		md_zsub2(D, dims, ostr, out, istr, in, ostr, out);
}

static void md_zfdiff_f_core2(int D, const bart_dim_t dims[D], int d, bool adj, const bart_stride_t ostr[D], complex float* out, const bart_stride_t istr[D], const complex float* in)
{
	md_zfdiff_core2(D, dims, d, false, adj, ostr, out, istr, in);
}

static void md_zfdiff_b_core2(int D, const bart_dim_t dims[D], int d, bool adj, const bart_stride_t ostr[D], complex float* out, const bart_stride_t istr[D], const complex float* in)
{
	md_zfdiff_core2(D, dims, d, true, adj, ostr, out, istr, in);
}


static void md_zfdiff_z_core2(int D, const bart_dim_t dims[D], int d, bool adj, const bart_stride_t ostr[D], complex float* out, const bart_stride_t istr[D], const complex float* in)
{
	bart_dim_t pos[D];
	md_set_dims(D, pos, 0);

	pos[d] = -1;
	md_circ_shift2(D, dims, pos, ostr, out, istr, in, CFL_SIZE);

	complex float* tmp = md_alloc_sameplace(D, dims, CFL_SIZE, out);

	pos[d] = 1;
	md_circ_shift2(D, dims, pos, MD_STRIDES(D, dims, CFL_SIZE), tmp, istr, in, CFL_SIZE);

	md_zsub2(D, dims, ostr, out, ostr, out, MD_STRIDES(D, dims, CFL_SIZE), tmp);

	md_free(tmp);

	md_zsmul2(D, dims, ostr, out, ostr, out, adj ? -0.5 : 0.5);
}



static void grad_op(md_zfdiff_core_t grad, int D, const bart_dim_t dims[D], int d, bart_flags_t flags, complex float* out, const complex float* in)
{
	int N = bitcount(flags);

	assert(N == dims[d]);
	assert(!MD_IS_SET(flags, d));

	bart_stride_t strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	bart_dim_t dims1[D];
	md_select_dims(D, ~MD_BIT(d), dims1, dims);

	bart_stride_t strs1[D];
	md_calc_strides(D, strs1, dims1, CFL_SIZE);

	bart_flags_t flags2 = flags;

	for (int i = 0; i < N; i++) {

		int lsb = md_min_idx(flags2);
		flags2 = MD_CLEAR(flags2, lsb);

		grad(D, dims1, lsb, false, strs, (void*)out + i * strs[d], strs1, in);
	}

	assert(0 == flags2);
}


static void grad_adjoint(md_zfdiff_core_t grad, int D, const bart_dim_t dims[D], int d, bart_flags_t flags, complex float* out, const complex float* in)
{
	int N = bitcount(flags);

	assert(N == dims[d]);
	assert(!MD_IS_SET(flags, d));

	bart_stride_t strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	bart_dim_t dims1[D];
	md_select_dims(D, ~MD_BIT(d), dims1, dims);

	bart_stride_t strs1[D];
	md_calc_strides(D, strs1, dims1, CFL_SIZE);

	bart_flags_t flags2 = flags;

	complex float* tmp = md_alloc_sameplace(D, dims1, CFL_SIZE, out);

	md_clear(D, dims1, out, CFL_SIZE);
	md_clear(D, dims1, tmp, CFL_SIZE);

	for (int i = 0; i < N; i++) {

		int lsb = md_min_idx(flags2);
		flags2 = MD_CLEAR(flags2, lsb);

		grad(D, dims1, lsb, true, strs1, tmp, strs, (const void*)in + i * strs[d]);
		md_zadd(D, dims1, out, out, tmp);
	}

	md_free(tmp);

	assert(0 == flags2);
}




struct grad_s {

	linop_data_t super;

	md_zfdiff_core_t grad;

	int N;
	int d;
	bart_dim_t* dims;
	bart_flags_t flags;
};

static DEF_TYPEID(grad_s);

static void grad_op_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(grad_s, _data);

	grad_op(data->grad, data->N, data->dims, data->d, data->flags, dst, src);
}

static void grad_op_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(grad_s, _data);

	grad_adjoint(data->grad, data->N, data->dims, data->d, data->flags, dst, src);
}


static void grad_op_free(const linop_data_t* _data)
{
	const auto data = CAST_DOWN(grad_s, _data);

	xfree(data->dims);
	xfree(data);
}

static struct linop_s* linop_grad_internal_create(md_zfdiff_core_t grad, bart_dim_t N, const bart_dim_t dims[N], int d, bart_flags_t flags)
{
	PTR_ALLOC(struct grad_s, data);
	SET_TYPEID(grad_s, data);

	data->grad = grad;

	int NO = N;

	if (N == d) {

		// as a special case, id d is one after the last dimensions,
		// we extend the output dimensions by one.

		NO++;

	} else {

		assert(1 == dims[d]);
	}

	bart_dim_t dims2[NO];
	md_copy_dims(N, dims2, dims);

	assert(!MD_IS_SET(flags, d));

	dims2[d] = bitcount(flags);

	data->N = NO;
	data->d = d;
	data->flags = flags;

	data->dims = *TYPE_ALLOC(bart_dim_t[N + 1]);

	md_copy_dims(NO, data->dims, dims2);

	return linop_create(NO, dims2, N, dims, CAST_UP(PTR_PASS(data)), grad_op_apply, grad_op_adjoint, NULL, NULL, grad_op_free);
}

struct linop_s* linop_grad_forward_create(bart_dim_t N, const bart_dim_t dims[N], int d, bart_flags_t flags)
{
	return linop_grad_internal_create(md_zfdiff_f_core2, N, dims, d, flags);
}

struct linop_s* linop_grad_backward_create(bart_dim_t N, const bart_dim_t dims[N], int d, bart_flags_t flags)
{
	return linop_grad_internal_create(md_zfdiff_b_core2, N, dims, d, flags);
}

struct linop_s* linop_grad_zentral_create(bart_dim_t N, const bart_dim_t dims[N], int d, bart_flags_t flags)
{
	return linop_grad_internal_create(md_zfdiff_z_core2, N, dims, d, flags);
}

struct linop_s* linop_grad_create(bart_dim_t N, const bart_dim_t dims[N], int d, bart_flags_t flags)
{
	return linop_grad_backward_create(N, dims, d, flags);
}

struct symmetrize_s {

	linop_data_t super;

	int N;
	const bart_dim_t* dims;
	const bart_dim_t* sdims;
	const bart_stride_t* strs;

	int dim1;
	int dim2;
};

static DEF_TYPEID(symmetrize_s);

static void symmetrize_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(symmetrize_s, _data);

	assert(dst != src);
	md_copy2(data->N, data->dims, data->strs, dst, data->strs, src, CFL_SIZE);

	bart_dim_t pos[data->N] = { };

	for (int i = 0; i < data->dims[data->dim1]; i++) {

		for (int j = 0; j < data->dims[data->dim2]; j++) {

			pos[data->dim1] = i;
			pos[data->dim2] = j;
			const complex float* tmp_in = MD_ACCESS_PTR(data->N, data->strs, pos, src);

			SWAP(pos[data->dim1], pos[data->dim2]);

			complex float* tmp_out = MD_ACCESS_PTR(data->N, data->strs, pos, dst);

			md_zadd2(data->N, data->sdims, data->strs, tmp_out, data->strs, tmp_out, data->strs, tmp_in);
			md_zsmul2(data->N, data->sdims, data->strs, tmp_out, data->strs, tmp_out, 0.5);
		}
	}
}

static void symmetrize_free(const linop_data_t* _data)
{
	const auto data = CAST_DOWN(symmetrize_s, _data);

	xfree(data->dims);
	xfree(data->sdims);
	xfree(data->strs);
	xfree(data);
}


struct linop_s* linop_symmetrize_create(bart_dim_t N, const bart_dim_t dims[N], bart_flags_t flags)
{
	PTR_ALLOC(struct symmetrize_s, data);
	SET_TYPEID(symmetrize_s, data);

	flags &= (MD_BIT(N) - 1);

	data->N = N;
	assert(2 == bitcount(flags));

	bart_stride_t strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	bart_dim_t sdims[N];
	md_select_dims(N, ~flags, sdims, dims);

	data->dims = ARR_CLONE(bart_dim_t[N], dims);
	data->sdims = ARR_CLONE(bart_dim_t[N], sdims);
	data->strs = ARR_CLONE(bart_dim_t[N], strs);

	data->dim1 = md_min_idx(flags);
	data->dim2 = md_max_idx(flags);

	assert(dims[data->dim1] == dims[data->dim2]);

	return linop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), symmetrize_apply, symmetrize_apply, NULL, NULL, symmetrize_free);
}




struct laplace_s {

	linop_data_t super;

	int N;
	bart_dim_t* dims;
	bart_flags_t flags;
	const float* scaling;
};

static DEF_TYPEID(laplace_s);

static void laplace_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(laplace_s, _data);

	md_laplace_fd_scaled(data->N, data->dims, data->flags, data->scaling, dst, src);
}

static void laplace_free(const linop_data_t* _data)
{
	const auto data = CAST_DOWN(laplace_s, _data);

	xfree(data->dims);
	xfree(data->scaling);
	xfree(data);
}


struct linop_s* linop_scaled_laplace_create(bart_dim_t N, const bart_dim_t dims[N], bart_flags_t flags, const float scaling[N])
{
	PTR_ALLOC(struct laplace_s, data);
	SET_TYPEID(laplace_s, data);

	data->N = N;
	data->flags = flags;
	data->dims = ARR_CLONE(bart_dim_t[N], dims);
	data->scaling = ARR_CLONE(float[N], scaling);

	return linop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), laplace_apply, laplace_apply, NULL, NULL, laplace_free);
}


struct linop_s* linop_laplace_create(bart_dim_t N, const bart_dim_t dims[N], bart_flags_t flags)
{
	float scaling[N];
	for (int i = 0; i < N; i++)
		scaling[i] = 1.;

	return linop_scaled_laplace_create(N, dims, flags, scaling);

}
