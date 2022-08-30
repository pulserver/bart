/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"
#include "nlops/nlop_jacobian.h"

#include "ztrigon.h"


struct zsin_s {

	INTERFACE(nlop_data_t);
};

DEF_TYPEID(zsin_s);

static void zsin_free(const nlop_data_t* _data)
{
	xfree(_data);
}

static void zsin_apply(const nlop_data_t* /*data*/, int N, const long dims[N], complex float* dst, const complex float* src, complex float* der)
{

	if (NULL != der)
		md_zcos(N, dims, der, src);

	md_zsin(N, dims, dst, src);
}

const struct nlop_s* nlop_zsin_create(int N, const long dims[N])
{
	PTR_ALLOC(struct zsin_s, data);
	SET_TYPEID(zsin_s, data);

	return nlop_zdiag_create(N, dims, CAST_UP(PTR_PASS(data)), zsin_apply, zsin_free);
}






struct zcos_s {

	INTERFACE(nlop_data_t);
};

DEF_TYPEID(zcos_s);

static void zcos_free(const nlop_data_t* _data)
{
	xfree(_data);
}

static void zcos_apply(const nlop_data_t* /*_data*/, int N, const long dims[N], complex float* dst, const complex float* src, complex float* der)
{

	if (NULL != der) {

		md_zsin(N, dims, der, src);
		md_zsmul(N, dims, der, der, -1.);
	}

	md_zcos(N, dims, dst, src);
}

const struct nlop_s* nlop_zcos_create(int N, const long dims[N])
{
	PTR_ALLOC(struct zcos_s, data);
	SET_TYPEID(zcos_s, data);

	return nlop_zdiag_create(N, dims, CAST_UP(PTR_PASS(data)), zcos_apply, zcos_free);
}


struct zasin_s {

	INTERFACE(nlop_data_t);
};

DEF_TYPEID(zasin_s);

static void zasin_free(const nlop_data_t* _data)
{
	xfree(_data);
}

static void zasin_apply(const nlop_data_t* /*data*/, int N, const long dims[N], complex float* dst, const complex float* src, complex float* der)
{

	if (NULL != der) {

		//1 / sqrt(1 - x^2)
		md_zfill(N, dims, der, 1.);
		md_zmul(N, dims, dst, src, src);
		md_zsub(N, dims, der, der, dst);
		md_zspow(N, dims, der, der, -0.5);
	}

	md_zasin(N, dims, dst, src);
}

const struct nlop_s* nlop_zasin_create(int N, const long dims[N])
{
	PTR_ALLOC(struct zsin_s, data);
	SET_TYPEID(zsin_s, data);

	return nlop_zdiag_create(N, dims, CAST_UP(PTR_PASS(data)), zasin_apply, zasin_free);
}




struct zacos_s {

	INTERFACE(nlop_data_t);
};

DEF_TYPEID(zacos_s);

static void zacos_free(const nlop_data_t* _data)
{
	xfree(_data);
}

static void zacos_apply(const nlop_data_t* /*_data*/, int N, const long dims[N], complex float* dst, const complex float* src, complex float* der)
{

	if (NULL != der) {

		//-1 / sqrt(1 - x^2)
		md_zfill(N, dims, der, 1.);
		md_zmul(N, dims, dst, src, src);
		md_zsub(N, dims, der, der, dst);
		md_zspow(N, dims, der, der, -0.5);
		md_zsmul(N, dims, der, der, -1.);
	}

	md_zacos(N, dims, dst, src);
}

const struct nlop_s* nlop_zacos_create(int N, const long dims[N])
{
	PTR_ALLOC(struct zcos_s, data);
	SET_TYPEID(zcos_s, data);

	return nlop_zdiag_create(N, dims, CAST_UP(PTR_PASS(data)), zacos_apply, zacos_free);
}

