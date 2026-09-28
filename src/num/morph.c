/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2025 Moritz Blumenthal
 */

#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/loop.h"
#include "num/conv.h"

#include "morph.h"


complex float* md_structuring_element_cube(int N, bart_dim_t dims[N], int radius, bart_flags_t flags, const void* ref)
{
	for (int i = 0; i < N; i++)
		dims[i] = (MD_IS_SET(flags, i)) ? 1 + 2 * radius : 1;

	complex float* structure = md_alloc_sameplace(N, dims, CFL_SIZE, ref);
	md_zfill(N, dims, structure, 1.);

	return structure;
}

complex float* md_structuring_element_ball(int N, bart_dim_t dims[N], int radius, bart_flags_t flags, const void* ref)
{
	for (int i = 0; i < N; i++)
		dims[i] = (MD_IS_SET(flags, i)) ? 1 + 2 * radius : 1;

	complex float* structure = md_alloc_sameplace(N, dims, CFL_SIZE, ref);
	const bart_dim_t* dimsp = dims;

	NESTED(complex float, ball_kernel, (const bart_dim_t pos[]))
	{
		complex float val = 0.;

		float rad = 0.;
		for (int i = 0; i < N; i++)
			rad += powf(llabs(pos[i] - dimsp[i] / 2), 2);

		if (rad <= powf(radius, 2))
			val = 1.;


		return val;
	};

	md_zsample(N, dims, structure, ball_kernel);

	return structure;
}

complex float* md_structuring_element_cross(int N, bart_dim_t dims[N], int radius, bart_flags_t flags, const void* ref)
{
	for (int i = 0; i < N; i++)
		dims[i] = (MD_IS_SET(flags, i)) ? 1 + 2 * radius : 1;

	complex float* structure = md_alloc_sameplace(N, dims, CFL_SIZE, ref);
	const bart_dim_t* dimsp = dims;

	NESTED(complex float, ball_kernel, (const bart_dim_t pos[]))
	{
		complex float val = 0.;

		for (int i = 0; i < N; i++)
			if (pos[i] == dimsp[i] / 2)
				val = 1.;

		return val;
	};

	md_zsample(N, dims, structure, ball_kernel);

	return structure;
}


static void mask_conv(int D, const bart_dim_t mask_dims[D], complex float* mask, const bart_dim_t dims[D], complex float* out, const complex float* in, enum conv_type ctype)
{
	conv(D, md_nontriv_dims(D, mask_dims), ctype, CONV_SYMMETRIC, dims, out, dims, in, mask_dims, mask);
}

void md_erosion(int D, const bart_dim_t mask_dims[D], complex float* mask, const bart_dim_t dims[D], complex float* out, const complex float* in, enum conv_type ctype)
{
	complex float* tmp = md_alloc_sameplace(D, dims, CFL_SIZE, in);

	mask_conv(D, mask_dims, mask, dims, tmp, in, ctype);

	// take relative error into account due to floating points
	md_zsgreatequal(D, dims, out, tmp, (1 - 0.0001) * md_zasum(D, mask_dims, mask));

	md_free(tmp);
}


void md_dilation(int D, const bart_dim_t mask_dims[D], complex float* mask, const bart_dim_t dims[D], complex float* out, const complex float* in, enum conv_type ctype)
{
	complex float* tmp = md_alloc_sameplace(D, dims, CFL_SIZE, in);

	mask_conv(D, mask_dims, mask, dims, tmp, in, ctype);

	// take relative error into account due to floating points
	md_zsgreatequal(D, dims, out, tmp, (1 - md_zasum(D, mask_dims, mask) * 0.0001));

	md_free(tmp);
}

void md_opening(int D, const bart_dim_t mask_dims[D], complex float* mask, const bart_dim_t dims[D], complex float* out, const complex float* in, enum conv_type ctype)
{
	complex float* tmp = md_alloc_sameplace(D, dims, CFL_SIZE, in);

	md_erosion(D, mask_dims, mask, dims, tmp, in, ctype);

	md_dilation(D, mask_dims, mask, dims, out, tmp, ctype);

	md_free(tmp);
}

void md_closing(int D, const bart_dim_t mask_dims[D], complex float* mask, const bart_dim_t dims[D], complex float* out, const complex float* in, enum conv_type ctype)
{
	complex float* tmp = md_alloc_sameplace(D, dims, CFL_SIZE, in);

	md_dilation(D, mask_dims, mask, dims, tmp, in, ctype);

	md_erosion(D, mask_dims, mask, dims, out, tmp, ctype);

	md_free(tmp);
}



// this assumes a zero padded input such that if pos exceeds the input it evaluates to false
static bool extend_label(int N, const bart_stride_t lstrs[N], complex float* labels, const bart_stride_t istrs[N], const complex float* in, const bart_dim_t sdims[N], const bart_stride_t sstrs[N], const complex float* structure, bart_dim_t label)
{
	if (0. == *in)
		return false;

	if (0. != *labels)
		return false;

	*labels = label;

	bart_dim_t pos[N];
	md_set_dims(N, pos, 0);

	bart_stride_t offset[N];
	for (int i = 0; i < N; i++)
		offset[i] = -sdims[i] / 2;

	labels = &MD_ACCESS(N, lstrs, offset, labels);
	in = &MD_ACCESS(N, istrs, offset, in);

	while (md_next(N, sdims, ~UINT64_C(0), pos)) {

		if (0. == MD_ACCESS(N, sstrs, pos, structure))
			continue;

		extend_label(N, lstrs, &MD_ACCESS(N, lstrs, pos, labels), istrs, &MD_ACCESS(N, istrs, pos, in), sdims, sstrs, structure, label);
	}

	return true;
}

static bart_dim_t md_label_int2(int N, const bart_dim_t dims[N], const bart_stride_t lstrs[N], complex float* labels, const bart_stride_t istrs[N], const _Complex float* in, const bart_dim_t sdims[N], const complex float* structure)
{
	md_clear2(N, dims, lstrs, labels, CFL_SIZE);

	bart_dim_t pos[N];
	md_set_dims(N, pos, 0);

	bart_stride_t sstrs[N];
	md_calc_strides(N, sstrs, sdims, CFL_SIZE);

	bart_dim_t label = 0;

	do {

		label++;
		if (!extend_label(N, lstrs, &MD_ACCESS(N, lstrs, pos, labels), istrs, &MD_ACCESS(N, istrs, pos, in), sdims, sstrs, structure, label))
			label--;

	} while (md_next(N, dims, ~UINT64_C(0), pos));

	return label;
}


bart_dim_t md_label(int N, const bart_dim_t dims[N], _Complex float* labels, const _Complex float* src, const bart_dim_t sdims[N], const complex float* structure)
{
	bart_dim_t ndims[N];
	for (int i = 0; i < N; i++)
		ndims[i] = dims[i] + sdims[i] - 1;

	complex float* tin = md_alloc(N, ndims, CFL_SIZE);
	md_resize_center(N, ndims, tin, dims, src, CFL_SIZE);

	bart_stride_t tstrs[N];
	md_calc_strides(N, tstrs, ndims, CFL_SIZE);

	bart_stride_t offset[N];
	for (int i = 0; i < N; i++)
		offset[i] = (sdims[i] / 2);

	complex float* cpu_structure = md_alloc(N, sdims, CFL_SIZE);
	md_copy(N, sdims, cpu_structure, structure, CFL_SIZE);

	complex float* cpu_labels = md_alloc(N, dims, CFL_SIZE);

	bart_dim_t label = md_label_int2(N, dims, MD_STRIDES(N, dims, CFL_SIZE), cpu_labels, tstrs, &MD_ACCESS(N, tstrs, offset, tin), sdims, cpu_structure);
	md_free(tin);

	md_copy(N, dims, labels, cpu_labels, CFL_SIZE);
	md_free(cpu_labels);
	md_free(cpu_structure);

	return label;
}


complex float* md_label_simple_connection(int N, bart_dim_t dims[N], float radius, bart_flags_t flags)
{
	int r = (int)floor(radius);
	assert(0 < r);

	for (int i = 0; i < N; i++)
		dims[i] = (MD_IS_SET(flags, i)) ? 1 + 2 * r : 1;

	complex float* structure = md_alloc(N, dims, CFL_SIZE);

	bart_dim_t pos[N];
	md_set_dims(N, pos, 0);

	bart_stride_t strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	bart_dim_t center[N];
	for (int i = 0; i < N; i++)
		center[i] = dims[i] / 2;

	do {
		bart_dim_t sum = 0;
		for (int i = 0; i < N; i++)
			sum += pow(llabs(pos[i] - center[i]), 2);

		MD_ACCESS(N, strs, pos, structure) = (pow(radius, 2) >= sum) ? 1. : 0.;
	} while (md_next(N, dims, ~UINT64_C(0), pos));

	return structure;
}

void md_center_of_mass(int N_labels, int N, float com[N_labels][N], const bart_dim_t dims[N], const complex float* labels, const complex float* wgh)
{
	complex float* labels_cpu = md_alloc(N, dims, CFL_SIZE);
	md_copy(N, dims, labels_cpu, labels, CFL_SIZE);

	complex float* wgh_cpu = NULL;
	if (NULL != wgh) {

		wgh_cpu = md_alloc(N, dims, CFL_SIZE);
		md_copy(N, dims, wgh_cpu, wgh, CFL_SIZE);
	}

	float count[N_labels];

	bart_dim_t pos[N];
	md_set_dims(N, pos, 0);

	bart_stride_t strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	for (int i = 0; i < N_labels; i++) {

		count[i] = 0.;

		for (int j = 0; j < N; j++)
			com[i][j] = 0.;
	}

	do {
		bart_dim_t label = MD_ACCESS(N, strs, pos, labels_cpu) - 1;

		if (-1 == label)
			continue;

		assert(label < N_labels);

		float val = 1.;
		if (NULL != wgh_cpu)
			val = crealf(MD_ACCESS(N, strs, pos, wgh_cpu));

		count[label] += val;

		for (int j = 0; j < N; j++)
			com[label][j] += pos[j] * val;

	} while (md_next(N, dims, ~UINT64_C(0), pos));

	for (int i = 0; i < N_labels; i++) {

		assert(0. != count[i]);

		for (int j = 0; j < N; j++)
			com[i][j] /= count[i];
	}

	md_free(labels_cpu);
	if (NULL != wgh_cpu)
		md_free(wgh_cpu);
}
