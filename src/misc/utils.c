/* Copyright 2014. The Regents of the University of California.
 * Copyright 2018,2021 Martin Uecker.
 * Copyright 2024. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/loop.h"

#include "misc/misc.h"

#include "utils.h"



complex float* compute_mask(int N, const bart_dim_t msk_dims[N], const float restrict_fov[N])
{
	complex float* mask = md_alloc(N, msk_dims, CFL_SIZE);

	bart_dim_t small_dims[N];

	for (int i = 0; i < N; i++)
		small_dims[i] = (1 == msk_dims[i]) ? 1 : (msk_dims[i] * restrict_fov[i]);

	complex float* small_mask = md_alloc(N, small_dims, CFL_SIZE);

	md_fill(N, small_dims, small_mask, &(complex float){ 1. }, CFL_SIZE);
	md_resize_center(N, msk_dims, mask, small_dims, small_mask, CFL_SIZE);

	md_free(small_mask);

	return mask;
}


void apply_mask(int N, const bart_dim_t dims[N], complex float* x, const float restrict_fov[N])
{
	bart_flags_t flags = 0;

	for (int i = 0; i < N; i++)
		if (1. != restrict_fov[i])
			flags = MD_SET(flags, i);

	bart_dim_t msk_dims[N];
	md_select_dims(N, flags, msk_dims, dims);

	bart_stride_t msk_strs[N];
	md_calc_strides(N, msk_strs, msk_dims, CFL_SIZE);

	complex float* mask = compute_mask(N, msk_dims, restrict_fov);

	bart_stride_t strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);
	md_zmul2(N, dims, strs, x, strs, x, msk_strs, mask);
	md_free(mask);
}


void normalize(int N, bart_flags_t flags, const bart_dim_t dims[N], complex float* maps)
{
	bart_dim_t dims_img[N];
	md_select_dims(N, ~flags, dims_img, dims);

	complex float* maps_norm = md_alloc(N, dims_img, CFL_SIZE);

	md_zrss(N, dims, flags, maps_norm, maps);

	bart_stride_t str[N];
	bart_stride_t str_img[N];

	md_calc_strides(N, str, dims, CFL_SIZE);
	md_calc_strides(N, str_img, dims_img, CFL_SIZE);

	md_zdiv2(N, dims, str, maps, str, maps, str_img, maps_norm);
	md_free(maps_norm);
}


void normalizel1(int N, bart_flags_t flags, const bart_dim_t dims[N], complex float* maps)
{
	bart_dim_t dims_img[N];
	md_select_dims(N, ~flags, dims_img, dims);

	complex float* maps_norm = md_alloc(N, dims_img, CFL_SIZE);
	complex float* maps_abs = md_alloc(N, dims, CFL_SIZE);

	md_zabs(N, dims, maps_abs, maps);

	bart_stride_t strs[N];
	bart_stride_t strs_img[N];
	md_calc_strides(N, strs_img, dims_img, CFL_SIZE);
	md_calc_strides(N, strs, dims, CFL_SIZE);

	md_clear(N, dims_img, maps_norm, CFL_SIZE);
	md_zadd2(N, dims, strs_img, maps_norm, strs_img, maps_norm, strs, maps_abs);

	md_free(maps_abs);

	bart_stride_t str[N];
	bart_stride_t str_img[N];

	md_calc_strides(N, str, dims, CFL_SIZE);
	md_calc_strides(N, str_img, dims_img, CFL_SIZE);

	md_zdiv2(N, dims, str, maps, str, maps, str_img, maps_norm);
	md_free(maps_norm);
}




/*
 * rotate phase jointly along dim so that the 0-th slice along dim has phase = 0
 *
 */
void fixphase(int N, const bart_dim_t dims[N], int dim, complex float* out, const complex float* in)
{
	assert(dim < N);

	bart_dim_t dims2[N];
	md_select_dims(N, ~MD_BIT(dim), dims2, dims);

	complex float* tmp = md_alloc_sameplace(N, dims2, CFL_SIZE, in);

	bart_dim_t pos[N];
	for (int i = 0; i < N; i++)
		pos[i] = 0;

	md_slice(N, MD_BIT(dim), pos, dims, tmp, in, CFL_SIZE);

	md_zphsr(N, dims2, tmp, tmp);

	bart_stride_t strs[N];
	bart_stride_t strs2[N];

	md_calc_strides(N, strs, dims, CFL_SIZE);
	md_calc_strides(N, strs2, dims2, CFL_SIZE);

	md_zmulc2(N, dims, strs, out, strs, in, strs2, tmp);

	md_free(tmp);
}

void fixphase2(int N, const bart_dim_t dims[N], int dim, const complex float rot[dims[dim]], complex float* out, const complex float* in)
{
	assert(dim < N);

	bart_stride_t strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	bart_dim_t dims2[N];
	bart_stride_t strs2[N];
	md_select_dims(N, ~MD_BIT(dim), dims2, dims);
	md_calc_strides(N, strs2, dims2, CFL_SIZE);

	complex float* tmp = md_alloc_sameplace(N, dims2, CFL_SIZE, in);

	bart_dim_t tdims[N];
	bart_stride_t tstrs[N];
	md_select_dims(N, MD_BIT(dim), tdims, dims);
	md_calc_strides(N, tstrs, tdims, CFL_SIZE);

	md_clear(N, dims2, tmp, CFL_SIZE);
	md_zfmacc2(N, dims, strs2, tmp, strs, in, tstrs, rot);
	md_zphsr(N, dims2, tmp, tmp);

	md_zmulc2(N, dims, strs, out, strs, in, strs2, tmp);

	md_free(tmp);
}

