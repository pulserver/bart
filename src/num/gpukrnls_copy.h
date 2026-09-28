#include "misc/dimtypes.h"

#ifdef __cplusplus
extern "C" {
#endif

extern void cuda_copy_ND(int D, const bart_dim_t dims[], const bart_stride_t ostrs[], void* dst, const bart_stride_t istrs[], const void* src, size_t size);
extern _Bool cuda_memequal(bart_dim_t size, const void* src1, const void* src2);

extern void cuda_decompress(bart_stride_t stride, bart_dim_t N, bart_stride_t dcstrs, void* dst, bart_stride_t istrs, const bart_dim_t* index, const void* src, size_t size);
extern void cuda_compress(bart_stride_t stride, bart_dim_t N, void* dst, bart_stride_t istrs, const bart_dim_t* index, bart_stride_t dcstrs, const void* src, size_t size);


#ifdef __cplusplus
}
#endif
