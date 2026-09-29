/* Copyright 2025-2026. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2025 Moritz Blumenthal
 *
 * References:
 *
 * Lee T.-C., Kashyap R.L. and Chu C.-N., Building skeleton models
 * via 3-D medial surface/axis thinning algorithms.
 * Computer Vision, Graphics, and Image Processing, 56(6):462-478, 1994.
 *
 *
 */

#include <assert.h>
#include <complex.h>
#include <math.h>

#include "misc/misc.h"

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

static bart_dim_t md_label_int2(int N, const bart_dim_t dims[N], const bart_stride_t lstrs[N], complex float* labels, const bart_stride_t istrs[N], const complex float* in, const bart_dim_t sdims[N], const complex float* structure)
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


bart_dim_t md_label(int N, const bart_dim_t dims[N], complex float* labels, const complex float* src, const bart_dim_t sdims[N], const complex float* structure)
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

static void extract_neighborhood(bool neighbor[3][3][3], bart_stride_t strs[3], complex float* src)
{
	for (int p = -1; p < 2; p++)
		for (int r = -1; r < 2; r++)
			for (int c = -1; c < 2; c++)
				neighbor[p + 1][c + 1][r + 1] = (0. != src[(c * strs[0] + r * strs[1] + p * strs[2]) / (bart_dim_t)sizeof(*src)]) ? true : false;
}

static int euler[256] = {
	0,  1, 0, -1, 0, -1, 0, 1, 0, -3, 0, -1, 0, -1, 0, 1, 0, -1, 0, 1, 0, 1, 0, -1, 0, 3, 0, 1, 0, 1, 0, -1,
	0, -3, 0, -1, 0,  3, 0, 1, 0,  1, 0, -1, 0,  3, 0, 1, 0, -1, 0, 1, 0, 1, 0, -1, 0, 3, 0, 1, 0, 1, 0, -1,
	0, -3, 0,  3, 0, -1, 0, 1, 0,  1, 0,  3, 0, -1, 0, 1, 0, -1, 0, 1, 0, 1, 0, -1, 0, 3, 0, 1, 0, 1, 0, -1,
	0,  1, 0,  3, 0,  3, 0, 1, 0,  5, 0,  3, 0,  3, 0, 1, 0, -1, 0, 1, 0, 1, 0, -1, 0, 3, 0, 1, 0, 1, 0, -1,
	0, -7, 0, -1, 0, -1, 0, 1, 0, -3, 0, -1, 0, -1, 0, 1, 0, -1, 0, 1, 0, 1, 0, -1, 0, 3, 0, 1, 0, 1, 0, -1,
	0, -3, 0, -1, 0,  3, 0, 1, 0,  1, 0, -1, 0,  3, 0, 1, 0, -1, 0, 1, 0, 1, 0, -1, 0, 3, 0, 1, 0, 1, 0, -1,
	0, -3, 0,  3, 0, -1, 0, 1, 0,  1, 0,  3, 0, -1, 0, 1, 0, -1, 0, 1, 0, 1, 0, -1, 0, 3, 0, 1, 0, 1, 0, -1,
	0,  1, 0,  3, 0,  3, 0, 1, 0,  5, 0,  3, 0,  3, 0, 1, 0, -1, 0, 1, 0, 1, 0, -1, 0, 3, 0, 1, 0, 1, 0, -1
};

static int oct_idx[8][7] = {
	{  2,  1, 11, 10,  5,  4, 14 }, //NEB
	{  0,  9,  3, 12,  1, 10,  4 }, //NWB
	{  8,  7, 17, 16,  5,  4, 14 }, //SEB
	{  6, 15,  7, 16,  3, 12,  4 }, //SWB
	{ 20, 23, 19, 22, 11, 14, 10 }, //NEU
	{ 18, 21,  9, 12, 19, 22, 10 }, //NWU
	{ 26, 23, 17, 14, 25, 22, 16 }, //SEU
	{ 24, 25, 15, 16, 21, 22, 12 }, //SWU
};

static bool is_euler_invariant(bool neighbors[3][3][3])
{
	int ret = 0;

	for (int o = 0; o < 8; o++) {

		unsigned int idx = 1;

		for (int j = 0; j < 7; j++) {

			if ((&(neighbors[0][0][0]))[oct_idx[o][j]])
				idx |= 1u << (7 - j);
		}

		ret += euler[idx];
	}

	return 0 == ret;
}

static int count_neighbors(bool neighbors[3][3][3])
{
	int count = 0;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 3; k++)
				count += (neighbors[i][j][k] ? 1 : 0);

	return count - 1;
}

static bool extend_local_label(int labels[3][3][3], const bool neighbors[3][3][3], int i, int j, int k, int label)
{
	if (!neighbors[i][j][k])
		return false;

	if (0 != labels[i][j][k])
		return false;

	labels[i][j][k] = label;

	for (int ip = MAX(0, i - 1); ip < MIN(3, i + 2); ip++)
		for (int jp = MAX(0, j - 1); jp < MIN(3, j + 2); jp++)
			for (int kp = MAX(0, k - 1); kp < MIN(3, k + 2); kp++)
				extend_local_label(labels, neighbors, ip, jp, kp, label);

	return true;
}

static int local_label(const bool neighbors[3][3][3])
{
	int labels[3][3][3];

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 3; k++)
				labels[i][j][k] = 0;

	int label = 1;

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 3; k++)
				if (extend_local_label(labels, neighbors, i, j, k, label))
					label++;

	return label - 1;
}


static bool is_simple(bool neighbors[3][3][3])
{
	neighbors[1][1][1] = false;
	int labels = local_label(neighbors);
	neighbors[1][1][1] = true;
	return (1 == labels);

}


static bool is_border(int type, bool neighbors[3][3][3])
{
	switch (type) {

		case 0: return !neighbors[1][1][0];
		case 1: return !neighbors[1][1][2];
		case 2: return !neighbors[1][0][1];
		case 3: return !neighbors[1][2][1];
		case 4: return !neighbors[0][1][1];
		case 5: return !neighbors[2][1][1];
		default: assert(false); return false;
	};
}

static void thinning_3D(const bart_dim_t dims[3], complex float* dst, const complex float* keep)
{
	bart_dim_t count = 0;
	bart_dim_t size = md_calc_size(3, dims);
	for (bart_dim_t i = 0; i < size; i++)
		if (0. != dst[i])
			count++;

	bart_dim_t pos[6][count?:1][3];

	bart_stride_t strs[3];
	md_calc_strides(3, strs, dims, CFL_SIZE);

	bool redo = true;

	while (redo) {

		redo = false;

		bart_dim_t count_boarders[6] = { 0, 0, 0, 0, 0, 0 };

		for (bart_dim_t i = 0; i < size; i++) {

			if (0. == dst[i])
				continue;

			if (NULL != keep && 0. != crealf(keep[i]))
				continue;

			bool neighbors[3][3][3];

			extract_neighborhood(neighbors, strs, dst + i);

			if (1 == count_neighbors(neighbors))
				continue;

			if (!is_simple(neighbors))
				continue;

			if (!is_euler_invariant(neighbors))
				continue;

			for (int j = 0; j < 6; j++) {

				if (!is_border(j, neighbors))
					continue;

				int idx = count_boarders[j]++;
				md_unravel_index(3, pos[j][idx], 7, dims, i);
			}
		}

		for (int j = 0; j < 6; j++) {

			bool neighbors[3][3][3];

			for (bart_dim_t i = 0; i < count_boarders[j]; i++) {

				extract_neighborhood(neighbors, strs, &MD_ACCESS(3, strs, pos[j][i], dst));

				if (is_simple(neighbors) && is_euler_invariant(neighbors)) {

					redo = true;
					MD_ACCESS(3, strs, pos[j][i], dst) = 0.;
				}
			}
		}
	}
}

void md_thinning_3D(int N, const bart_dim_t dims[N], complex float* dst, const complex float* src, const complex float* keep)
{
	assert(3 >= bitcount(md_nontriv_dims(N, dims)));

	bart_dim_t rdims[3] = { 1, 1, 1 };
	bart_dim_t ndims[3] = { 1, 1, 1 };

	for (int i =0, ip = 0; i < N; i++)
		if (1 < dims[i])
			rdims[ip++] = dims[i];

	for (int i = 0; i < 3; i++)
		ndims[i] = rdims[i] + 2;

	complex float* tmp = md_alloc(3, ndims, CFL_SIZE);
	md_resize_center(3, ndims, tmp, rdims, src, CFL_SIZE);

	complex float* tmp_keep = NULL;

	if (NULL != keep) {
		tmp_keep = md_alloc(3, ndims, CFL_SIZE);
		md_resize_center(3, ndims, tmp_keep, rdims, keep, CFL_SIZE);
	}

	thinning_3D(ndims, tmp, tmp_keep);

	if (NULL != tmp_keep)
		md_free(tmp_keep);

	md_resize_center(3, rdims, dst, ndims, tmp, CFL_SIZE);
	md_free(tmp);
}


