
#include <stddef.h>

#include "misc/cppwrap.h"

extern bart_dim_t md_compress_mask_to_index(bart_dim_t N, const bart_dim_t dims[__VLA(N)], bart_dim_t* index, const _Complex float* mask);

extern void md_compress_dims(bart_dim_t N, bart_dim_t cdims[__VLA(N)], const bart_dim_t dcdims[__VLA(N)], const bart_dim_t mdims[__VLA(N)], bart_dim_t max);
extern void md_decompress_dims(bart_dim_t N, bart_dim_t dcdims[__VLA(N)], const bart_dim_t cdims[__VLA(N)], const bart_dim_t mdims[__VLA(N)]);

extern void md_decompress2(int N, const bart_dim_t odims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], void* dst, const bart_dim_t idims[__VLA(N)], const bart_stride_t istrs[__VLA(N)], const void* src, const bart_dim_t mdims[__VLA(N)], const bart_stride_t mstrs[__VLA(N)], const bart_dim_t* index, const void* fill, size_t size);
extern void md_decompress(int N, const bart_dim_t odims[__VLA(N)], void* dst, const bart_dim_t idims[__VLA(N)], const void* src, const bart_dim_t mdims[__VLA(N)], const bart_dim_t* index, const void* fill, size_t size);
extern void md_compress2(int N, const bart_dim_t odims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], void* dst, const bart_dim_t idims[__VLA(N)], const bart_stride_t istrs[__VLA(N)], const void* src, const bart_dim_t mdims[__VLA(N)], const bart_stride_t mstrs[__VLA(N)], const bart_dim_t* index, size_t size);
extern void md_compress(int N, const bart_dim_t odims[__VLA(N)], void* dst, const bart_dim_t idims[__VLA(N)], const void* src, const bart_dim_t mdims[__VLA(N)], const bart_dim_t* index, size_t size);

#include "misc/cppwrap.h"
