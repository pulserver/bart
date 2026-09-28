
#ifndef _MULTIND_H
#define _MULTIND_H	1

#include <string.h>
#ifndef assert
#include <assert.h>
#endif
#include <stdint.h>
#ifdef _WIN32
#include <malloc.h>
#else
#include <alloca.h>
#endif

#include "misc/cppwrap.h"
#include "misc/nested.h"
#include "misc/types.h"
#include "misc/misc.h"

#define MD_BIT(x) (UINT64_C(1) << (x))
#define MD_IS_SET(x, y)	((x) & MD_BIT(y))
#define MD_CLEAR(x, y) ((x) & ~MD_BIT(y))
#define MD_SET(x, y)	((x) | MD_BIT(y))

typedef CLOSURE_TYPE(void, (void* ptr[])) md_nary_fun_t;
typedef CLOSURE_TYPE(void, (int C, void* ptr[__VLA(C)], int N, const bart_dim_t dim[__VLA(N)], const bart_stride_t* str[__VLA(C)])) md_nary_resolve_fun_t;
typedef CLOSURE_TYPE(void, (bart_dim_t N, bart_stride_t str, void* ptr)) md_trafo_fun_t;
typedef CLOSURE_TYPE(void, (const bart_dim_t* pos)) md_loop_fun_t;
typedef CLOSURE_TYPE(void, (bart_flags_t flags, bart_dim_t* pos)) md_loop_fun2_t;

extern void md_unravel_index(int D, bart_dim_t pos[__VLA(D)], bart_flags_t flags, const bart_dim_t dims[__VLA(D)], bart_dim_t index);
extern void md_unravel_index_permuted(int D, bart_dim_t pos[__VLA(D)], bart_flags_t flags, const bart_dim_t dims[__VLA(D)], bart_dim_t index, const int order[__VLA(D)]);
extern bart_dim_t md_ravel_index(int D, const bart_dim_t pos[__VLA(D)], bart_flags_t flags, const bart_dim_t dims[__VLA(D)]);
extern bart_dim_t md_ravel_index_permuted(int D, const bart_dim_t pos[__VLA(D)], bart_flags_t flags, const bart_dim_t dims[__VLA(D)], const int order[__VLA(D)]);
extern bart_dim_t md_reravel_index(int D, bart_flags_t rflags, bart_flags_t uflags, const bart_dim_t dims[__VLA(D)], bart_dim_t index);

extern void md_nary(int C, int D, const bart_dim_t dim[__VLA(D)], const bart_stride_t* str[__VLA(C)], void* ptr[__VLA(C)], md_nary_fun_t fun);
#define md_nary(C, D, dim, str, ptr, fun) \
	md_nary(C, D, dim, str, ptr, CLOSURE(md_nary_fun_t, fun))

extern void md_nary_resolve(int C, int D, const bart_dim_t dim[__VLA(D)], const bart_stride_t* str[__VLA(C)], void* ptr[__VLA(C)], md_nary_resolve_fun_t fun);
#define md_nary_resolve(C, D, dim, str, ptr, fun) \
	md_nary_resolve(C, D, dim, str, ptr, CLOSURE(md_nary_resolve_fun_t, fun))

extern void md_parallel_nary(int C, int D, const bart_dim_t dim[__VLA(D)], bart_flags_t flags, const bart_stride_t* str[__VLA(C)], void* ptr[__VLA(C)], md_nary_fun_t fun);
#define md_parallel_nary(C, D, dim, flags, str, ptr, fun) \
	md_parallel_nary(C, D, dim, flags, str, ptr, CLOSURE(md_nary_fun_t, fun))

extern void md_parallel_loop(int D, const bart_dim_t dim[__VLA(D)], bart_flags_t flags, md_loop_fun_t fun);
#define md_parallel_loop(D, dim, flags, fun) \
	md_parallel_loop(D, dim, flags, CLOSURE(md_loop_fun_t, fun))

extern void md_parallel_loop_split(int D, const bart_dim_t dim[__VLA(D)], bart_flags_t flags, md_loop_fun2_t fun);
#define md_parallel_loop_split(D, dim, flags, fun) \
	md_parallel_loop_split(D, dim, flags, CLOSURE(md_loop_fun2_t, fun))

extern void md_loop(int D, const bart_dim_t dim[__VLA(D)], md_loop_fun_t fun);
#define md_loop(D, dim, fun) \
	md_loop(D, dim, CLOSURE(md_loop_fun_t, fun))

extern void md_septrafo2(int D, const bart_dim_t dimensions[__VLA(D)], bart_flags_t flags, const bart_stride_t strides[__VLA(D)], void* ptr, md_trafo_fun_t fun);
extern void md_septrafo(int D, const bart_dim_t dimensions[__VLA(D)], bart_flags_t flags, void* ptr, size_t size, md_trafo_fun_t fun);


extern void md_clear2(int D, const bart_dim_t dim[__VLA(D)], const bart_stride_t str[__VLA(D)], void* ptr, size_t size);
extern void md_clear(int D, const bart_dim_t dim[__VLA(D)], void* ptr, size_t size);
extern void md_swap2(int D, const bart_dim_t dim[__VLA(D)], const bart_stride_t ostr[__VLA(D)], void* optr, const bart_stride_t istr[__VLA(D)], void* iptr, size_t size);
extern void md_swap(int D, const bart_dim_t dim[__VLA(D)],  void* optr, void* iptr, size_t size);
extern void md_circular_swap2(int M, int D, const bart_dim_t dims[__VLA(D)], const bart_stride_t* strs[__VLA(M)], void* ptr[__VLA(M)], size_t size);
extern void md_circular_swap(int M, int D, const bart_dim_t dims[__VLA(D)], void* ptr[__VLA(M)], size_t size);

extern void md_copy2(int D, const bart_dim_t dim[__VLA(D)], const bart_stride_t ostr[__VLA(D)], void* optr, const bart_stride_t istr[__VLA(D)], const void* iptr, size_t size);
extern void md_copy(int D, const bart_dim_t dim[__VLA(D)],  void* optr, const void* iptr, size_t size);
extern void md_copy_block2(int D, const bart_dim_t pos[__VLA(D)], const bart_dim_t odim[__VLA(D)], const bart_stride_t ostr[__VLA(D)], void* optr, const bart_dim_t idim[__VLA(D)], const bart_stride_t istr[__VLA(D)], const void* iptr, size_t size);
extern void md_copy_block(int D, const bart_dim_t pos[__VLA(D)], const bart_dim_t odim[__VLA(D)], void* optr, const bart_dim_t idim[__VLA(D)], const void* iptr, size_t size);
extern void md_move_block2(int D, const bart_dim_t dim[__VLA(D)], const bart_dim_t opos[__VLA(D)], const bart_dim_t odim[__VLA(D)], const bart_stride_t ostr[__VLA(D)], void* optr, const bart_dim_t ipos[__VLA(D)], const bart_dim_t idim[__VLA(D)], const bart_stride_t istr[__VLA(D)], const void* iptr, size_t size);
extern void md_move_block(int D, const bart_dim_t dim[__VLA(D)], const bart_dim_t opos[__VLA(D)], const bart_dim_t odim[__VLA(D)], void* optr, const bart_dim_t ipos[__VLA(D)], const bart_dim_t idim[__VLA(D)], const void* iptr, size_t size);

extern void md_pad(int D, const void* val, const bart_dim_t odim[__VLA(D)], void* optr, const bart_dim_t idim[__VLA(D)], const void* iptr, size_t size);
extern void md_pad_center(int D, const void* val, const bart_dim_t odim[__VLA(D)], void* optr, const bart_dim_t idim[__VLA(D)], const void* iptr, size_t size);
extern void md_reflectpad_center2(int D, const bart_dim_t odim[__VLA(D)], const bart_stride_t ostr[__VLA(D)], void* optr, const bart_dim_t idim[__VLA(D)], const bart_stride_t istr[__VLA(D)], const void* iptr, size_t size);
extern void md_reflectpad_center(int D, const bart_dim_t odim[__VLA(D)], void* optr, const bart_dim_t idim[__VLA(D)], const void* iptr, size_t size);
extern void md_resize(int D, const bart_dim_t odim[__VLA(D)], void* optr, const bart_dim_t idim[__VLA(D)], const void* iptr, size_t size);
extern void md_resize_center(int D, const bart_dim_t odim[__VLA(D)], void* optr, const bart_dim_t idim[__VLA(D)], const void* iptr, size_t size);
extern void md_resize_front(int D, const bart_dim_t odim[__VLA(D)], void* optr, const bart_dim_t idim[__VLA(D)], const void* iptr, size_t size);
extern void md_fill2(int D, const bart_dim_t dim[__VLA(D)], const bart_stride_t str[__VLA(D)], void* ptr, const void* iptr, size_t size);
extern void md_fill(int D, const bart_dim_t dim[__VLA(D)], void* ptr, const void* iptr, size_t size);
extern void md_slice2(int D, bart_flags_t flags, const bart_dim_t pos[__VLA(D)], const bart_dim_t dim[__VLA(D)], const bart_stride_t ostr[__VLA(D)], void* optr, const bart_stride_t istr[__VLA(D)], const void* iptr, size_t size);
extern void md_slice(int D, bart_flags_t flags, const bart_dim_t pos[__VLA(D)], const bart_dim_t dim[__VLA(D)], void* optr, const void* iptr, size_t size);
extern void md_transpose2(int D, int dim1, int dim2, const bart_dim_t odims[__VLA(D)], const bart_stride_t ostr[__VLA(D)], void* optr, const bart_dim_t idims[__VLA(D)], const bart_stride_t istr[__VLA(D)], const void* iptr, size_t size);
extern void md_transpose(int D, int dim1, int dim2, const bart_dim_t odims[__VLA(D)], void* optr, const bart_dim_t idims[__VLA(D)], const void* iptr, size_t size);
extern void md_permute2(int D, const int order[__VLA(D)], const bart_dim_t odims[__VLA(D)], const bart_stride_t ostr[__VLA(D)], void* optr, const bart_dim_t idims[__VLA(D)], const bart_stride_t istr[__VLA(D)], const void* iptr, size_t size);
extern void md_permute(int D, const int order[__VLA(D)], const bart_dim_t odims[__VLA(D)], void* optr, const bart_dim_t idims[__VLA(D)], const void* iptr, size_t size);
extern void md_flip2(int D, const bart_dim_t dims[__VLA(D)], bart_flags_t flags, const bart_stride_t ostr[__VLA(D)], void* optr, const bart_stride_t istr[__VLA(D)], const void* iptr, size_t size);
extern void md_flip(int D, const bart_dim_t dims[__VLA(D)], bart_flags_t flags, void* optr, const void* iptr, size_t size);

extern void md_swap_flip2(int D, const bart_dim_t dims[__VLA(D)], bart_flags_t flags, const bart_stride_t ostr[__VLA(D)], void* optr, const bart_stride_t istr[__VLA(D)], void* iptr, size_t size);
extern void md_swap_flip(int D, const bart_dim_t dims[__VLA(D)], bart_flags_t flags, void* optr, void* iptr, size_t size);

extern void md_reshape(int D, bart_flags_t flags, const bart_dim_t odims[__VLA(D)], void* optr, const bart_dim_t idims[__VLA(D)], const void* iptr, size_t size);
extern void md_reshape2(int D, bart_flags_t flags, const bart_dim_t odims[__VLA(D)], const bart_stride_t ostrs[__VLA(D)], void* optr, const bart_dim_t idims[__VLA(D)], const bart_stride_t istrs[__VLA(D)], const void* iptr, size_t size);

extern void md_copy_diag2(int D, const bart_dim_t dims[__VLA(D)], bart_flags_t flags, const bart_stride_t str1[__VLA(D)], void* dst, const bart_stride_t str2[__VLA(D)], const void* src, size_t size);
extern void md_copy_diag(int D, const bart_dim_t dims[__VLA(D)], bart_flags_t flags, void* dst, const void* src, size_t size);
extern void md_fill_diag(int D, const bart_dim_t dims[__VLA(D)], bart_flags_t flags, void* dst, const void* src, size_t size);

extern void md_circ_shift2(int D, const bart_dim_t dim[__VLA(D)], const bart_dim_t center[__VLA(D)], const bart_stride_t str1[__VLA(D)], void* dst, const bart_stride_t str2[__VLA(D)], const void* src, size_t size);
extern void md_circ_shift(int D, const bart_dim_t dim[__VLA(D)], const bart_dim_t center[__VLA(D)], void* dst, const void* src, size_t size);

extern void md_circ_ext2(int D, const bart_dim_t dims1[__VLA(D)], const bart_stride_t strs1[__VLA(D)], void* dst, const bart_dim_t dims2[__VLA(D)], const bart_stride_t strs2[__VLA(D)], const void* src, size_t size);
extern void md_circ_ext(int D, const bart_dim_t dims1[__VLA(D)], void* dst, const bart_dim_t dims2[__VLA(D)], const void* src, size_t size);


extern void md_periodic2(int D, const bart_dim_t dims1[__VLA(D)], const bart_stride_t strs1[__VLA(D)], void* dst, const bart_dim_t dims2[__VLA(D)], const bart_stride_t strs2[__VLA(D)], const void* src, size_t size);
extern void md_periodic(int D, const bart_dim_t dims1[__VLA(D)], void* dst, const bart_dim_t dims2[__VLA(D)], const void* src, size_t size);

extern bool md_compare2(int D, const bart_dim_t dims[__VLA(D)], const bart_stride_t str1[__VLA(D)], const void* src1,
			const bart_stride_t str2[__VLA(D)], const void* src2, size_t size);
extern bool md_compare(int D, const bart_dim_t dims[__VLA(D)], const void* src1, const void* src2, size_t size);

inline bart_dim_t md_calc_size_r(int D, const bart_dim_t dim[__VLA(D)], size_t size)
{
	if (0 == D)
		return (bart_stride_t)size;

	return md_calc_size_r(D - 1, dim, (size_t)((bart_stride_t)size * dim[D - 1]));
}

inline bart_dim_t md_calc_size(int D, const bart_dim_t dim[__VLA(D)])
{
	return md_calc_size_r(D, dim, 1);
}

inline bart_stride_t* md_calc_strides_selected(int D, bart_flags_t flags, bart_stride_t str[__VLA2(D)], const bart_dim_t dim[__VLA(D)], size_t size)
{
	bart_dim_t old = (bart_stride_t)size;

	for (int i = 0; i < D; i++) {

		if (!MD_IS_SET(flags, i)) {

			str[i] = 0;
			continue;
		}

		str[i] = (1 == dim[i]) ? 0 : old;
		old *= dim[i];
	}

	return str;
}

inline bart_stride_t* md_calc_strides(int D, bart_stride_t str[__VLA2(D)], const bart_dim_t dim[__VLA(D)], size_t size)
{
	return md_calc_strides_selected(D, ~UINT64_C(0), str, dim, size);
}

inline void md_copy_strides(int D, bart_stride_t ostrs[__VLA(D)], const bart_stride_t istrs[__VLA(D)])
{
	memcpy(ostrs, istrs, sizeof(bart_dim_t[D]));
}

inline void md_copy_dims(int D, bart_dim_t odims[__VLA(D)], const bart_dim_t idims[__VLA(D)])
{
	memcpy(odims, idims, sizeof(bart_dim_t[D]));
}


typedef void* (*md_alloc_fun_t)(int D, const bart_dim_t dimensions[__VLA(D)], size_t size);

extern void* md_alloc_safe(int D, const bart_dim_t dimensions[__VLA(D)], size_t size, size_t total_size) alloc_size(4);

inline void* md_alloc(int D, const bart_dim_t dimensions[__VLA(D)], size_t size)
{
	return md_alloc_safe(D, dimensions, size, (size_t)(md_calc_size(D, dimensions) * (bart_stride_t)size));
}

extern void* md_calloc(int D, const bart_dim_t dimensions[__VLA(D)], size_t size);
#ifdef USE_GPU
extern void* md_alloc_gpu(int D, const bart_dim_t dimensions[__VLA(D)], size_t size);
extern void* md_gpu_move(int D, const bart_dim_t dims[__VLA(D)], const void* ptr, size_t size);
extern void* md_gpu_mpi_move(int D, unsigned long dist_flags, const bart_dim_t dims[__VLA(D)], const void* ptr, size_t size);
extern void* md_alloc_gpu_mpi(int D, unsigned long dist_flags, const bart_dim_t dims[__VLA(D)], size_t size);
#endif
extern void* md_alloc_sameplace(int D, const bart_dim_t dimensions[__VLA(D)], size_t size, const void* ptr);
extern void md_free(const void* p);
extern bool md_is_sameplace(const void* ptr1, const void* ptr2);

extern void* md_alloc_mpi(int D, bart_flags_t dist_flags, const bart_dim_t dims[__VLA(D)], size_t size);
extern void* md_mpi_move(int D, bart_flags_t dist_flags, const bart_dim_t dims[__VLA(D)], const void* ptr, size_t size);
extern void* md_mpi_moveF(int D, bart_flags_t dist_flags, const bart_dim_t dims[__VLA(D)], const void* ptr, size_t size);
extern void* md_mpi_wrap(int D, bart_flags_t dist_flags, const bart_dim_t dims[__VLA(D)], const void* ptr, size_t size, bool writeback);

extern bart_stride_t md_calc_offset(int D, const bart_stride_t strides[__VLA(D)], const bart_dim_t position[__VLA(D)]);
extern int md_calc_blockdim(int D, const bart_dim_t dim[__VLA(D)], const bart_stride_t str[__VLA(D)], size_t size);
extern void md_select_dims(int D, bart_flags_t flags, bart_dim_t odims[__VLA(D)], const bart_dim_t idims[__VLA(D)]);
extern void md_select_strides(int D, bart_flags_t flags, bart_stride_t ostrs[__VLA(D)], const bart_stride_t istrs[__VLA(D)]);

extern void md_copy_order(int D, int odims[__VLA(D)], const int idims[__VLA(D)]);
extern void md_merge_dims(int D, bart_dim_t odims[__VLA(D)], const bart_dim_t dims1[__VLA(D)], const bart_dim_t dims2[__VLA(D)]);
extern bool md_check_compat(int D, bart_flags_t flags, const bart_dim_t dim1[__VLA(D)], const bart_dim_t dim2[__VLA(D)]);
extern bool md_check_bounds(int D, bart_flags_t flags, const bart_dim_t dim1[__VLA(D)], const bart_dim_t dim2[__VLA(D)]);
extern bool md_check_order_bounds(int D, bart_flags_t flags, const int order1[__VLA(D)], const int order2[__VLA(D)]);
extern void md_singleton_dims(int D, bart_dim_t dims[__VLA(D)]);
extern void md_singleton_strides(int D, bart_stride_t strs[__VLA(D)]);
extern void md_set_dims(int D, bart_dim_t dims[__VLA(D)], bart_dim_t val);
extern void md_min_dims(int D, bart_flags_t flags, bart_dim_t odims[__VLA(D)], const bart_dim_t idims1[__VLA(D)], const bart_dim_t idims2[__VLA(D)]);
extern void md_max_dims(int D, bart_flags_t flags, bart_dim_t odims[__VLA(D)], const bart_dim_t idims1[__VLA(D)], const bart_dim_t idims2[__VLA(D)]);
extern bool md_is_index(int D, const bart_dim_t pos[__VLA(D)], const bart_dim_t dims[__VLA(D)]);
extern bool md_check_dimensions(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t flags);
extern bool md_check_equal_dims(int N, const bart_dim_t dims1[__VLA(N)], const bart_dim_t dims2[__VLA(N)], bart_flags_t flags);
extern bool md_check_equal_order(int N, const int order1[__VLA(N)], const int order2[__VLA(N)], bart_flags_t flags);
extern void md_permute_dims(int D, const int order[__VLA(D)], bart_dim_t odims[__VLA(D)], const bart_dim_t idims[__VLA(D)]);
extern void md_transpose_dims(int D, int dim1, int dim2, bart_dim_t odims[__VLA(D)], const bart_dim_t idims[__VLA(D)]);

extern bool md_next_permuted(int D, const int order[__VLA(D)], const bart_dim_t dims[__VLA(D)], bart_flags_t flags, bart_dim_t pos[__VLA(D)]);

extern void md_mask_compress(int D, const bart_dim_t dims[__VLA(D)], bart_dim_t M, uint32_t dst[__VLA(M)], const float* src);
extern void md_mask_decompress(int D, const bart_dim_t dims[__VLA(D)], float* dst, bart_dim_t M, const uint32_t src[__VLA(M)]);

extern bart_flags_t md_permute_flags(int D, const int order[__VLA(D)], bart_flags_t flags);
extern void md_permute_invert(int D, int inv_order[__VLA(D)], const int order[__VLA(D)]);

extern bart_flags_t md_nontriv_dims(int D, const bart_dim_t dims[__VLA(D)]);
extern bart_flags_t md_nontriv_strides(int D, const bart_dim_t dims[__VLA(D)]);

extern bool md_overlap(int D1, const bart_dim_t dims1[__VLA(D1)], const bart_stride_t strs1[__VLA(D1)], const void* ptr1, size_t size1,
			int D2, const bart_dim_t dims2[__VLA(D2)], const bart_stride_t strs2[__VLA(D2)], const void* ptr2, size_t size2);


#define MD_MAKE_ARRAY(T, ...) ((T[]){ __VA_ARGS__ })
#define MD_DIMS(...) MD_MAKE_ARRAY(bart_dim_t, __VA_ARGS__)


extern int md_max_idx(bart_flags_t flags);
extern int md_min_idx(bart_flags_t flags);

#define MD_CAST_ARRAY2_PTR(T, N, dims, x, a, b) \
({						\
	int _a = (a), _b = (b);			\
	const bart_dim_t* _dims = dims;		\
	assert(_a < _b);			\
	assert(!md_check_dimensions((N), _dims, (1 << _a) | (1 << _b))); \
	(T (*)[_dims[_b]][_dims[_a]])(x);	\
})
#define MD_CAST_ARRAY3_PTR(T, N, dims, x, a, b, c) \
({						\
	int _a = (a), _b = (b), _c = (c);	\
	const bart_dim_t* _dims = dims;		\
	assert((_a < _b) && (_b < _c));		\
	assert(!md_check_dimensions((N), _dims, (1 << _a) | (1 << _b | (1 << _c)))); \
	(T (*)[_dims[_c]][_dims[_b]][_dims[_a]])(x); \
})

#define MD_CAST_ARRAY2(T, N, dims, x, a, b) (*MD_CAST_ARRAY2_PTR(T, N, dims, x, a, b))
#define MD_CAST_ARRAY3(T, N, dims, x, a, b, c) (*MD_CAST_ARRAY3_PTR(T, N, dims, x, a, b, c))


#define MD_ACCESS(N, strs, pos, x)	(*({ auto _x = (x); &((_x)[md_calc_offset((N), (strs), (pos)) / (bart_stride_t)sizeof((_x)[0])]); }))
#define MD_ACCESS_PTR(N, strs, pos, x)	((NULL == (x) ? NULL : &MD_ACCESS(N, strs, pos, x)))

#define MD_STRIDES(N, dims, elsize)	(md_calc_strides((N), alloca((bart_flags_t)(N) * sizeof(bart_dim_t)), (dims), (elsize)))

#define MD_SINGLETON_DIMS(N)				\
({							\
	int _N = (N);					\
	bart_dim_t* _dims = alloca((bart_flags_t)_N * sizeof(bart_dim_t));	\
	md_singleton_dims(_N, _dims);			\
	_dims;						\
})

#define MD_SINGLETON_STRS(N)				\
({							\
	int _N = (N);					\
	bart_dim_t* _dims = alloca((bart_flags_t)_N * sizeof(bart_dim_t)); 	\
	md_singleton_strides(_N, _dims); 		\
	_dims; 						\
})


inline bool md_next(int D, const bart_dim_t dims[__VLA(D)], bart_flags_t flags, bart_dim_t pos[__VLA(D)])
{
	if (0 == D--)
		return false;

	if (md_next(D, dims, flags, pos))
		return true;

	if (MD_IS_SET(flags, D)) {

		assert((0 <= pos[D]) && (pos[D] < dims[D]));

		if (++pos[D] < dims[D])
			return true;

		pos[D] = 0;
	}

	return false;
}


#include "misc/cppwrap.h"

#endif // _MULTIND_H

