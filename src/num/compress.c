/* Copyright 2024. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/types.h"

#include "num/multind.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#include "num/gpukrnls_copy.h"
#endif
#include "num/vptr_fun.h"

#include "compress.h"



bart_dim_t md_compress_mask_to_index(bart_dim_t N, const bart_dim_t dims[N], bart_dim_t* index, const complex float* mask)
{
	bart_dim_t tot = md_calc_size(N, dims);
	bart_dim_t max = 0;

	for (bart_dim_t i = 0; i < tot; i++)
		if (0. != crealf(mask[i]))
			index[i] = (max++);
		else
			index[i] = -1;

	return max;
}

void md_compress_dims(bart_dim_t N, bart_dim_t cdims[N], const bart_dim_t dcdims[N], const bart_dim_t mdims[N], bart_dim_t max)
{
	md_select_dims(N, ~md_nontriv_dims(N, mdims), cdims, dcdims);

	int cdim = ffs(md_nontriv_dims(N, mdims)) - 1;
	cdims[cdim] = max;
}

void md_decompress_dims(bart_dim_t N, bart_dim_t dcdims[N], const bart_dim_t cdims[N], const bart_dim_t mdims[N])
{
	int cdim = ffs(md_nontriv_dims(N, mdims)) - 1;

	md_select_dims(N, ~MD_BIT(cdim), dcdims, cdims);
	md_max_dims(N, ~UINT64_C(0), dcdims, dcdims, mdims);
}

static void decompress_kern(bart_stride_t stride, bart_dim_t N, bart_stride_t dcstrs, void* dst, bart_stride_t istrs, const bart_dim_t* index, const void* src, size_t size)
{
	for (int i = 0; i < N; i++)
		if (index[i * istrs] >= 0)
			memcpy(dst + dcstrs * i, src + index[istrs * i] * stride, size);
}

static void compress_kern(bart_stride_t stride, bart_dim_t N, void* dst, bart_stride_t istrs, const bart_dim_t* index, bart_stride_t dcstrs, const void* src, size_t size)
{
	for (int i = 0; i < N; i++)
		if (index[i * istrs] >= 0)
			memcpy(dst + index[istrs * i] * stride, src + dcstrs * i, size);
}


struct vptr_decompress_s {

	vptr_fun_data_t super;
	size_t size;
};

DEF_TYPEID(vptr_decompress_s);


static void md_decompress2_int(vptr_fun_data_t* _data, int N, int D, const bart_dim_t* dims[N], const bart_stride_t* strs[N], void* args[N])
{
	auto d = CAST_DOWN(vptr_decompress_s, _data);

	const bart_dim_t* odims = dims[0];
	const bart_dim_t* idims = dims[1];
	const bart_dim_t* mdims = dims[2];

	const bart_stride_t* ostrs = strs[0];
	const bart_stride_t* istrs = strs[1];
	const bart_stride_t* mstrs = strs[2];

	void* dst = args[0];
	const void* src = args[1];
	const bart_dim_t* index = args[2];

	assert(1 >= bitcount(md_nontriv_dims(D, mdims) & md_nontriv_dims(D, idims)));

	int flat_idx = 0;
	for (int i = 0; i < D; i++)
		if (1 != mdims[i] && 1 != idims[i])
			flat_idx = i;

	bart_dim_t midx = ffs(md_nontriv_dims(D, mdims)) - 1;
	if (0 == (bitcount(md_nontriv_dims(D, mdims) & md_nontriv_dims(D, idims))))
		flat_idx = midx;

	bart_stride_t istrs2[D];
	md_select_strides(D, ~MD_BIT(flat_idx), istrs2, istrs);

	bart_dim_t pos[D];
	md_set_dims(D, pos, 0);

	bart_flags_t merge_flags = MD_BIT(midx);
	bart_dim_t merge_size = mdims[midx];

	for (int i = midx + 1; i < D; i++) {

		if (mdims[i] != odims[i])
			break;

		if (1 == mdims[i])
			continue;

		if ((mstrs[i] != mstrs[midx] * merge_size) || (ostrs[i] != ostrs[midx] * merge_size))
			break;

		merge_flags |= MD_BIT(i);
		merge_size *= mdims[i];
	}

#ifdef USE_CUDA
	bool gpu = cuda_ondevice(dst);
	assert(gpu == cuda_ondevice(src));
	assert(gpu == cuda_ondevice(index));
#endif

	do {
		void* tdst = dst + md_calc_offset(D, ostrs, pos);
		const void* tsrc = src + md_calc_offset(D, istrs2, pos);
		const bart_dim_t* tindex = &MD_ACCESS(D, mstrs, pos, index);

#ifdef USE_CUDA
		if (gpu)
			cuda_decompress(istrs[flat_idx], merge_size, ostrs[midx], tdst, mstrs[midx] / (bart_stride_t)sizeof(bart_dim_t), tindex, tsrc, d->size);
		else
#endif
		decompress_kern(istrs[flat_idx], merge_size, ostrs[midx], tdst, mstrs[midx] / (bart_stride_t)sizeof(bart_dim_t), tindex, tsrc, d->size);

	} while (md_next(D, odims, ~merge_flags, pos));
}

struct vptr_compress_s {

	vptr_fun_data_t super;
	size_t size;
};

DEF_TYPEID(vptr_compress_s);


static void md_compress2_int(vptr_fun_data_t* _data, int N, int D, const bart_dim_t* dims[N], const bart_stride_t* strs[N], void* args[N])
{
	auto d = CAST_DOWN(vptr_compress_s, _data);

	const bart_dim_t* odims = dims[0];
	const bart_dim_t* idims = dims[1];
	const bart_dim_t* mdims = dims[2];

	const bart_stride_t* ostrs = strs[0];
	const bart_stride_t* istrs = strs[1];
	const bart_stride_t* mstrs = strs[2];

	void* dst = args[0];
	const void* src = args[1];
	const bart_dim_t* index = args[2];


	assert(1 >= bitcount(md_nontriv_dims(D, mdims) & md_nontriv_dims(D, odims)));

	int flat_idx = 0;
	for (int i = 0; i < D; i++)
		if (1 != mdims[i] && 1 != odims[i])
			flat_idx = i;

	bart_dim_t midx = ffs(md_nontriv_dims(D, mdims)) - 1;
	if (0 == (bitcount(md_nontriv_dims(D, mdims) & md_nontriv_dims(D, odims))))
		flat_idx = midx;

	bart_stride_t ostrs2[D];
	md_select_strides(D, ~MD_BIT(flat_idx), ostrs2, ostrs);

	bart_dim_t pos[D];
	md_set_dims(D, pos, 0);

	bart_flags_t merge_flags = MD_BIT(midx);
	bart_dim_t merge_size = mdims[midx];

	for (int i = midx + 1; i < D; i++) {

		if (mdims[i] != idims[i])
			break;

		if (1 == mdims[i])
			continue;

		if ((mstrs[i] != mstrs[midx] * merge_size) || (istrs[i] != istrs[midx] * merge_size))
			break;

		merge_flags |= MD_BIT(i);
		merge_size *= mdims[i];
	}

#ifdef USE_CUDA
	bool gpu = cuda_ondevice(dst);
	assert(gpu == cuda_ondevice(src));
	assert(gpu == cuda_ondevice(index));
#endif


	do {
		void* tdst = dst + md_calc_offset(D, ostrs2, pos);
		const void* tsrc = src + md_calc_offset(D, istrs, pos);
		const bart_dim_t* tindex = &MD_ACCESS(D, mstrs, pos, index);

#ifdef USE_CUDA
		if (gpu)
			cuda_compress(ostrs[flat_idx], merge_size, tdst, mstrs[midx] / (bart_stride_t)sizeof(bart_dim_t), tindex, istrs[midx], tsrc, d->size);
		else
#endif
		compress_kern(ostrs[flat_idx], merge_size, tdst, mstrs[midx] / (bart_stride_t)sizeof(bart_dim_t), tindex, istrs[midx], tsrc, d->size);

	} while (md_next(D, idims, ~merge_flags, pos));
}



void md_decompress2(int N, const bart_dim_t odims[N], const bart_stride_t ostrs[N], void* dst, const bart_dim_t idims[N], const bart_stride_t istrs[N], const void* src, const bart_dim_t mdims[N], const bart_stride_t mstrs[N], const bart_dim_t* index, const void* fill, size_t size)
{
	if (NULL != fill)
		md_fill2(N, odims, ostrs, dst, fill, size);

	PTR_ALLOC(struct vptr_decompress_s, _d);
	SET_TYPEID(vptr_decompress_s, _d);
	_d->super.del = NULL;
	_d->size = size;

	bart_flags_t lflags = ~md_nontriv_strides(N, mstrs);

	// 0 is read as not completely over written
	exec_vptr_fun_gen(md_decompress2_int, CAST_UP(PTR_PASS(_d)), 3, N, lflags, MD_BIT(0), MD_BIT(0) | MD_BIT(1) | MD_BIT(2),
				(const bart_dim_t*[3]) { odims, idims, mdims }, (const bart_dim_t*[3]) { ostrs, istrs, mstrs },
				(void* [3]) { dst, (void*)src, (void*)index}, (size_t[3]) { size, size, sizeof(bart_dim_t)}, true);
}


void md_compress2(int N, const bart_dim_t odims[N], const bart_stride_t ostrs[N], void* dst, const bart_dim_t idims[N], const bart_stride_t istrs[N], const void* src, const bart_dim_t mdims[N], const bart_stride_t mstrs[N], const bart_dim_t* index, size_t size)
{
	PTR_ALLOC(struct vptr_compress_s, _d);
	SET_TYPEID(vptr_compress_s, _d);
	_d->super.del = NULL;
	_d->size = size;

	bart_flags_t lflags = ~md_nontriv_strides(N, mstrs);

	exec_vptr_fun_gen(md_compress2_int, CAST_UP(PTR_PASS(_d)), 3, N, lflags, MD_BIT(0), MD_BIT(1) | MD_BIT(2),
				(const bart_dim_t*[3]) { odims, idims, mdims }, (const bart_dim_t*[3]) { ostrs, istrs, mstrs },
				(void* [3]) { dst, (void*)src, (void*)index}, (size_t[3]) { size, size, sizeof(bart_dim_t)}, true);
}


void md_decompress(int N, const bart_dim_t odims[N], void* dst, const bart_dim_t idims[N], const void* src, const bart_dim_t mdims[N], const bart_dim_t* index, const void* fill, size_t size)
{
	md_decompress2(N, odims, MD_STRIDES(N, odims, size), dst, idims, MD_STRIDES(N, idims, size), src, mdims, MD_STRIDES(N, mdims, sizeof(bart_dim_t)), index, fill, size);
}

void md_compress(int N, const bart_dim_t odims[N], void* dst, const bart_dim_t idims[N], const void* src, const bart_dim_t mdims[N], const bart_dim_t* index, size_t size)
{
	md_compress2(N, odims, MD_STRIDES(N, odims, size), dst, idims, MD_STRIDES(N, idims, size), src, mdims, MD_STRIDES(N, mdims, sizeof(bart_dim_t)), index, size);
}



