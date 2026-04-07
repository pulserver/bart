/* Copyright 2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

 #include <complex.h>

#include "misc/misc.h"

#include "misc/mri.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/multiplace.h"
#include "num/iovec.h"

#include "linops/linop.h"

#include "nlops/nlop.h"

#include "utils.h"

struct moba_rvc_s {

	linop_data_t super;

	int N;
	const long* dims;

	unsigned long rvc;
};

DEF_TYPEID(moba_rvc_s);

static void moba_rvc_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(moba_rvc_s, _data);

	md_copy(data->N, data->dims, dst, src, CFL_SIZE);

	long pos[data->N];
	md_set_dims(data->N, pos, 0);

	long map_dims[data->N];
	md_select_dims(data->N, ~COEFF_FLAG, map_dims, data->dims);

	long strs[data->N];
	md_calc_strides(data->N, strs, data->dims, CFL_SIZE);

	for (; pos[COEFF_DIM] < data->dims[COEFF_DIM]; pos[COEFF_DIM]++)
		if (MD_IS_SET(data->rvc, pos[COEFF_DIM]))
			md_zreal2(data->N, map_dims, strs, MD_ACCESS_PTR(data->N, strs, pos, dst), strs, MD_ACCESS_PTR(data->N, strs, pos, dst));
}

static void moba_rvc_del(const linop_data_t* _data)
{
	const auto data = CAST_DOWN(moba_rvc_s, _data);

	xfree(data->dims);

	xfree(data);
}

const struct linop_s* moba_rvc_create(int N, const long in_dims[N], unsigned long flags)
{
	PTR_ALLOC(struct moba_rvc_s, data);
	SET_TYPEID(moba_rvc_s, data);

	data->N = N;
	data->dims = ARR_CLONE(long[N], in_dims);
	data->rvc = flags;

	return linop_create(N, in_dims, N, in_dims, CAST_UP(PTR_PASS(data)), moba_rvc_apply, moba_rvc_apply, moba_rvc_apply, NULL, moba_rvc_del);
}



struct moba_precond_s {

	nlop_data_t super;

	int N;
	const long* dims;
	const long* map_dims;

	const long* strs;
	const long* scl_strs;

	struct multiplace_array_s* diag;
	const struct linop_s** map_linops;
	const float* init_val;
};

DEF_TYPEID(moba_precond_s);

static void moba_precond_derivative(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(moba_precond_s, _data);

	const complex float* diag = multiplace_read(data->diag, src);

	md_zmul2(data->N, data->dims, data->strs, dst, data->strs, src, data->scl_strs, diag);

	complex float* tmp1 = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
	complex float* tmp2 = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	long pos[data->N];
	md_set_dims(data->N, pos, 0);

	for (; pos[COEFF_DIM] < data->dims[COEFF_DIM]; pos[COEFF_DIM]++) {

		const struct linop_s* lop = data->map_linops[pos[COEFF_DIM]];

		if (NULL == lop)
			continue;

		md_copy_block(data->N, pos, data->map_dims, tmp1, data->dims, dst, CFL_SIZE);

		linop_forward(lop, data->N, data->map_dims, tmp2, data->N, data->map_dims, tmp1);

		md_copy_block(data->N, pos, data->dims, dst, data->map_dims, tmp2, CFL_SIZE);
	};

	md_free(tmp1);
	md_free(tmp2);

}

static void moba_precond_adjoint(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(moba_precond_s, _data);

	const complex float* diag = multiplace_read(data->diag, src);

	md_zmulc2(data->N, data->dims, data->strs, dst, data->strs, src, data->scl_strs, diag);

	complex float* tmp1 = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
	complex float* tmp2 = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	long pos[data->N];
	md_set_dims(data->N, pos, 0);

	for (; pos[COEFF_DIM] < data->dims[COEFF_DIM]; pos[COEFF_DIM]++) {

		const struct linop_s* lop = data->map_linops[pos[COEFF_DIM]];

		if (NULL == lop)
			continue;

		md_copy_block(data->N, pos, data->map_dims, tmp1, data->dims, dst, CFL_SIZE);

		linop_adjoint(lop, data->N, data->map_dims, tmp2, data->N, data->map_dims, tmp1);

		md_copy_block(data->N, pos, data->dims, dst, data->map_dims, tmp2, CFL_SIZE);
	};

	md_free(tmp1);
	md_free(tmp2);
}

static void moba_precond_apply(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	moba_precond_derivative(_data, 0, 0, dst, src);

	const auto data = CAST_DOWN(moba_precond_s, _data);

	long pos[data->N];
	md_set_dims(data->N, pos, 0);

	const complex float* diag = multiplace_read(data->diag, NULL);

	for (; pos[COEFF_DIM] < data->dims[COEFF_DIM]; pos[COEFF_DIM]++)
		if (0. == cabsf(diag[pos[COEFF_DIM]]) && (0. != data->init_val[pos[COEFF_DIM]]))
			md_zfill2(data->N, data->map_dims, data->strs, MD_ACCESS_PTR(data->N, data->strs, pos, dst), data->init_val[pos[COEFF_DIM]]);

}

static void moba_precond_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(moba_precond_s, _data);

	for (long i = 0; i < data->dims[COEFF_DIM]; i++)
		if (NULL != data->map_linops[i])
			linop_free(data->map_linops[i]);

	xfree(data->dims);
	xfree(data->map_dims);
	xfree(data->strs);
	xfree(data->scl_strs);

	multiplace_free(data->diag);

	xfree(data->map_linops);
	xfree(data->init_val);

	xfree(data);
}

const struct nlop_s* moba_precond_create(int N, const long in_dims[N], const struct linop_s* linops[in_dims[COEFF_DIM]], const float scaling[in_dims[COEFF_DIM]], const float init[in_dims[COEFF_DIM]])
{
	assert(COEFF_DIM < N);

	PTR_ALLOC(struct moba_precond_s, data);
	SET_TYPEID(moba_precond_s, data);

	data->N = N;
	data->dims = ARR_CLONE(long[N], in_dims);

	long map_dims[N];
	md_select_dims(N, ~COEFF_FLAG, map_dims, in_dims);
	data->map_dims = ARR_CLONE(long[N], map_dims);

	long strs[N];
	md_calc_strides(N, strs, in_dims, CFL_SIZE);
	data->strs = ARR_CLONE(long[N], strs);

	long scl_strs[N];
	md_calc_strides_selected(N, COEFF_FLAG, scl_strs, in_dims, CFL_SIZE);
	data->scl_strs = ARR_CLONE(long[N], scl_strs);

	complex float scale_diag[in_dims[COEFF_DIM]];
	for (int i = 0; i < in_dims[COEFF_DIM]; i++)
		scale_diag[i] = scaling[i];

	long scl_dims[N];
	md_select_dims(N, COEFF_FLAG, scl_dims, in_dims);

	data->diag = multiplace_move(N, scl_dims, CFL_SIZE, scale_diag);

	const struct linop_s* map_linops[in_dims[COEFF_DIM]];
	for (int i = 0; i < in_dims[COEFF_DIM]; i++)
		map_linops[i] = (NULL != linops && NULL != linops[i]) ? linop_clone(linops[i]) : NULL;

	float init_val2[in_dims[COEFF_DIM]];
	for (int i = 0; i < in_dims[COEFF_DIM]; i++)
		init_val2[i] = (NULL != init) ? init[i] : 0.f;

	data->map_linops = ARR_CLONE(const struct linop_s*[in_dims[COEFF_DIM]], map_linops);
	data->init_val = ARR_CLONE(float[in_dims[COEFF_DIM]], init_val2);

	return nlop_create(N, in_dims, N, in_dims, CAST_UP(PTR_PASS(data)), moba_precond_apply, moba_precond_derivative, moba_precond_adjoint, NULL, NULL, moba_precond_del);
}





struct moba_attach_trafo_s {

	nlop_data_t super;

	const struct nlop_s* nlop;

	const struct linop_s* linop;
};

DEF_TYPEID(moba_attach_trafo_s);

static void moba_attach_trafo_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(moba_attach_trafo_s, _data);

	const struct iovec_s* cod = nlop_codomain(data->nlop);
	const struct iovec_s* dom = nlop_domain(data->nlop);

	nlop_apply(data->nlop, cod->N, cod->dims, dst, dom->N, dom->dims, src);
}

static void moba_attach_trafo_der(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(moba_attach_trafo_s, _data);

	const struct iovec_s* cod = nlop_codomain(data->nlop);
	const struct iovec_s* dom = nlop_domain(data->nlop);

	nlop_derivative(data->nlop, cod->N, cod->dims, dst, dom->N, dom->dims, src);
}

static void moba_attach_trafo_adj(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(moba_attach_trafo_s, _data);

	const struct iovec_s* cod = nlop_codomain(data->nlop);
	const struct iovec_s* dom = nlop_domain(data->nlop);

	nlop_adjoint(data->nlop, dom->N, dom->dims, dst, cod->N, cod->dims, src);
}

static void moba_attach_trafo_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(moba_attach_trafo_s, _data);

	nlop_free(data->nlop);
	linop_free(data->linop);

	xfree(data);
}

const struct nlop_s* moba_attach_trafo_F(const struct nlop_s* nlop, const struct linop_s* linop)
{
	PTR_ALLOC(struct moba_attach_trafo_s, data);
	SET_TYPEID(moba_attach_trafo_s, data);

	data->nlop = nlop;
	data->linop = linop_clone(linop);

	const struct iovec_s* cod = nlop_codomain(nlop);
	const struct iovec_s* dom = nlop_domain(nlop);

	return nlop_create(cod->N, cod->dims, dom->N, dom->dims, CAST_UP(PTR_PASS(data)), moba_attach_trafo_fun, moba_attach_trafo_der, moba_attach_trafo_adj, NULL, NULL, moba_attach_trafo_del);
}

const struct linop_s* moba_attach_trafo_get_linop(struct nlop_s* nlop)
{
	const nlop_data_t* _data = nlop_get_data(nlop);
	const auto data = CAST_DOWN(moba_attach_trafo_s, _data);

	return data->linop;
}
