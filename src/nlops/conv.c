/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2022-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <assert.h>
#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "linops/someops.h"
#include "linops/linop.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/iovec.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/tenmul.h"
#include "nlops/nlop.h"

#include "conv.h"

#ifdef USE_GPU
#include "num/gpuops.h"
#endif


struct convcorr_geom_s {

	nlop_data_t super;

	bart_dim_t N;

	const bart_dim_t* odims;
	const bart_dim_t* idims1;
	const bart_dim_t* idims2;

	const bart_dim_t* mdims;
	const bart_stride_t* ostrs;
	const bart_stride_t* istrs1;
	const bart_stride_t* istrs2;

	bart_flags_t flags;

	complex float* der1;
	complex float* der2;

	bart_dim_t shift;
};

DEF_TYPEID(convcorr_geom_s);

static void convcorr_init(struct convcorr_geom_s* data, const complex float* ref) {

	bool der1 = nlop_der_requested(CAST_UP(data), 0, 0);
	bool der2 = nlop_der_requested(CAST_UP(data), 1, 0);

	if (der1) {

		if (NULL == data->der2)
			data->der2 = md_alloc_sameplace(data->N, data->idims2, CFL_SIZE, ref);

	} else {

		md_free(data->der2);
		data->der2 = NULL;
	}

	if (der2) {

		if (NULL == data->der1)
			data->der1 = md_alloc_sameplace(data->N, data->idims1, CFL_SIZE, ref);

	} else {

		md_free(data->der1);
		data->der1 = NULL;
	}
}

static void convcorr_clear_der(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(convcorr_geom_s, _data);

	md_free(data->der1);
	md_free(data->der2);

	data->der1 = NULL;
	data->der2 = NULL;
}

static void convcorr_geom_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(convcorr_geom_s, _data);
	assert(3 == N);

	complex float* dst = args[0];
	const complex float* src1 = args[1];
	const complex float* src2 = args[2];

	convcorr_init(data, dst);

#ifdef USE_GPU
	assert((cuda_ondevice(dst) == cuda_ondevice(src1)) && (cuda_ondevice(src1) == cuda_ondevice(src2)));
#endif

	// conj to have benefits of fmac optimization in adjoints

	if (nlop_der_requested(_data, 1, 0))
		md_zconj(data->N, data->idims1, data->der1, src1);

	if (nlop_der_requested(_data, 0, 0))
		md_zconj(data->N, data->idims2, data->der2, src2);

	md_clear(data->N, data->odims, dst, CFL_SIZE);
	md_zfmac2(2 * data->N, data->mdims, data->ostrs, dst, data->istrs1, src1, data->istrs2, src2 + data->shift);
}


static void convcorr_geom_der2(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(convcorr_geom_s, _data);
	complex float* x1 = data->der1;

	if (NULL == x1)
		error("Convcorr %p derivative not available\n", data);

	md_clear(data->N, data->odims, dst, CFL_SIZE);
	md_zfmacc2(2 * data->N, data->mdims, data->ostrs, dst, data->istrs2, src + data->shift, data->istrs1, x1);
}

static void convcorr_geom_adj2(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(convcorr_geom_s, _data);
	complex float* x1 = data->der1;

	if (NULL == x1)
		error("Convcorr %p derivative not available\n", data);

	md_clear(data->N, data->idims2, dst, CFL_SIZE);
	md_zfmac2(2 * data->N, data->mdims, data->istrs2, dst + data->shift, data->ostrs, src, data->istrs1, x1);
}

static void convcorr_geom_der1(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(convcorr_geom_s, _data);
	complex float* x2 = data->der2;

	if (NULL == x2)
		error("Convcorr %p derivative not available\n", data);

	md_clear(data->N, data->odims, dst, CFL_SIZE);
	md_zfmacc2(2 * data->N, data->mdims, data->ostrs, dst, data->istrs1, src, data->istrs2, x2 + data->shift);
}

static void convcorr_geom_adj1(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(convcorr_geom_s, _data);
	complex float* x2 = data->der2;

	if (NULL == x2)
		error("Convcorr %p derivative not available\n", data);

	md_clear(data->N, data->idims1, dst, CFL_SIZE);
	md_zfmac2(2 * data->N, data->mdims, data->istrs1, dst, data->ostrs, src, data->istrs2, x2 + data->shift);
}


static void convcorr_geom_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(convcorr_geom_s, _data);

	xfree(data->odims);
	xfree(data->idims1);
	xfree(data->idims2);

	xfree(data->mdims);
	xfree(data->ostrs);
	xfree(data->istrs1);
	xfree(data->istrs2);

	md_free(data->der1);
	md_free(data->der2);

	xfree(data);
}

static struct nlop_s* nlop_convcorr_geom_valid_create(bart_dim_t N, bart_flags_t flags, const bart_dim_t odims[N], const bart_dim_t idims[N], const bart_dim_t kdims[N],
							bool conv, const bart_stride_t strides[N], const bart_dim_t dilations[N], bool transp)
{
	for (int i = 0; i < N; i++) {

		if (MD_IS_SET(flags, i)) {

			assert(idims[i] == strides[i] * (odims[i] - 1) + 1 + (kdims[i] - 1) * dilations[i]);
		}
	}

	PTR_ALLOC(struct convcorr_geom_s, data);
	SET_TYPEID(convcorr_geom_s, data);

	data->flags = flags;

	bart_dim_t nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], transp ? idims : odims);

	bart_dim_t nl_idims[2][N];
	md_copy_dims(N, nl_idims[0], transp ? odims : idims);
	md_copy_dims(N, nl_idims[1], kdims);

	data->N = N;

	PTR_ALLOC(bart_dim_t[N], nodims);
	PTR_ALLOC(bart_dim_t[N], nidims1);
	PTR_ALLOC(bart_dim_t[N], nidims2);

	PTR_ALLOC(bart_dim_t[2 * N], nmdims);
	PTR_ALLOC(bart_dim_t[2 * N], nostrs);
	PTR_ALLOC(bart_dim_t[2 * N], nistrs1);
	PTR_ALLOC(bart_dim_t[2 * N], nistrs2);

	md_copy_dims(N, *nodims, nl_odims[0]);
	md_copy_dims(N, *nidims1, nl_idims[0]);
	md_copy_dims(N, *nidims2, nl_idims[1]);

	if (transp)
		data->shift = calc_convcorr_geom_strs_dil(N, flags,
							*nmdims, *nistrs1, *nistrs2, *nostrs,
							odims, MD_STRIDES(N, odims, CFL_SIZE),
							kdims, MD_STRIDES(N, kdims, CFL_SIZE),
							idims, MD_STRIDES(N, idims, CFL_SIZE),
							dilations, strides, conv, false) / (bart_stride_t)CFL_SIZE;
	else
		data->shift = calc_convcorr_geom_strs_dil(N, flags,
							*nmdims, *nostrs, *nistrs2, *nistrs1,
							odims, MD_STRIDES(N, odims, CFL_SIZE),
							kdims, MD_STRIDES(N, kdims, CFL_SIZE),
							idims, MD_STRIDES(N, idims, CFL_SIZE),
							dilations, strides, conv, false) / (bart_stride_t)CFL_SIZE;

	data->odims = *PTR_PASS(nodims);
	data->idims1 = *PTR_PASS(nidims1);
	data->idims2 = *PTR_PASS(nidims2);

	data->mdims = *PTR_PASS(nmdims);
	data->ostrs = *PTR_PASS(nostrs);
	data->istrs1 = *PTR_PASS(nistrs1);
	data->istrs2 = *PTR_PASS(nistrs2);

	data->der1 = NULL;
	data->der2 = NULL;

	return nlop_generic_managed_create(1, N, nl_odims, 2, N, nl_idims, CAST_UP(PTR_PASS(data)), convcorr_geom_fun,
					   (nlop_der_fun_t[2][1]){ { convcorr_geom_der1 }, { convcorr_geom_der2 } },
					   (nlop_der_fun_t[2][1]){ { convcorr_geom_adj1 }, { convcorr_geom_adj2 } }, NULL, NULL, convcorr_geom_del, convcorr_clear_der, NULL);
}


struct nlop_s* nlop_convcorr_geom_create(int N, bart_flags_t flags, const bart_dim_t odims[N], const bart_dim_t idims[N], const bart_dim_t kdims[N],
					enum PADDING conv_pad, bool conv, const bart_stride_t strides[N], const bart_dim_t dilations[N], char transpc)
{
	bart_dim_t ones[N];
	md_singleton_dims(N, ones);

	if (NULL == strides)
		strides = ones;

	if (NULL == dilations)
		dilations = ones;

	struct nlop_s* result = NULL;

	assert(('N' == transpc) || ('T' == transpc) || ('C' == transpc));

	bool transp = ('N' != transpc);

	bart_dim_t pad_for[N];
	bart_dim_t pad_after[N];
	md_singleton_strides(N, pad_for);
	md_singleton_strides(N, pad_after);

	if (PAD_VALID == conv_pad) {

		result = nlop_convcorr_geom_valid_create(N, flags, odims, idims, kdims, conv, strides, dilations, transp);

	} else {

		bart_dim_t nidims[N];

		for (int i = 0; i < N; i++) {

			if (MD_IS_SET(flags, i)) {

				assert(idims[i] == strides[i] * odims[i]);
				nidims[i] = strides[i] * (odims[i] - 1) + 1 + (kdims[i] - 1) * dilations[i];

			} else {

				nidims[i] = idims[i];
			}

			bart_dim_t pos = llabs((nidims[i] / 2) - (idims[i] / 2)); // center corresponds to resize_center/ceter of fft

			// from md_resize_center:
			// if idim[d] > nidim[d], then optr[i] = iptr[pos + i] for 0 <= i < nidim[d]
			// if idim[d] < nidim[d], then optr[pos + i] = iptr[i] for 0 <= i < idim[d]

			pad_for[i] = (nidims[i] > idims[i]) ? pos : -pos;
			pad_after[i] = nidims[i] - idims[i] - pad_for[i];
		}

		result = nlop_convcorr_geom_valid_create(N, flags, odims, nidims, kdims, conv, strides, dilations, transp);

		struct linop_s* pad_op = linop_padding_create(N, idims, conv_pad, pad_for, pad_after);

		if (transp) {

			result = nlop_chain2_FF(result, 0, nlop_from_linop_F(linop_get_adjoint(pad_op)), 0);
			linop_free(pad_op);

		} else {

			result = nlop_chain2_FF(nlop_from_linop_F(pad_op), 0, result, 0);
			result = nlop_permute_inputs_F(result, 2, MAKE_ARRAY(1, 0));
		}
	}

	if ('C' == transpc)
		result = nlop_chain2_FF(nlop_from_linop_F(linop_zconj_create(N, kdims)), 0, result, 1);

	return result;
}

