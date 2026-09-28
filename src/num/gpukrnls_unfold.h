#include "misc/dimtypes.h"


#ifdef __cplusplus
extern "C" {
#endif

extern void cuda_add_unfold(int D, const bart_dim_t dims[], const bart_stride_t ostrs[], float* dst, const bart_stride_t istrs1[], const float* src1, const bart_stride_t istrs2[], const float* src2);
extern void cuda_zadd_unfold(int D, const bart_dim_t dims[], const bart_stride_t ostrs[], _Complex float* dst, const bart_stride_t istrs1[], const _Complex float* src1, const bart_stride_t istrs2[], const _Complex float* src2);

extern void cuda_mul_unfold(int D, const bart_dim_t dims[], const bart_stride_t ostrs[], float* dst, const bart_stride_t istrs1[], const float* src1, const bart_stride_t istrs2[], const float* src2);
extern void cuda_zmul_unfold(int D, const bart_dim_t dims[], const bart_stride_t ostrs[], _Complex float* dst, const bart_stride_t istrs1[], const _Complex float* src1, const bart_stride_t istrs2[], const _Complex float* src2);
extern void cuda_zmulc_unfold(int D, const bart_dim_t dims[], const bart_stride_t ostrs[], _Complex float* dst, const bart_stride_t istrs1[], const _Complex float* src1, const bart_stride_t istrs2[], const _Complex float* src2);

extern void cuda_fmac_unfold(int D, const bart_dim_t dims[], const bart_stride_t ostrs[], float* dst, const bart_stride_t istrs1[], const float* src1, const bart_stride_t istrs2[], const float* src2);
extern void cuda_zfmac_unfold(int D, const bart_dim_t dims[], const bart_stride_t ostrs[], _Complex float* dst, const bart_stride_t istrs1[], const _Complex float* src1, const bart_stride_t istrs2[], const _Complex float* src2);
extern void cuda_zfmacc_unfold(int D, const bart_dim_t dims[], const bart_stride_t ostrs[], _Complex float* dst, const bart_stride_t istrs1[], const _Complex float* src1, const bart_stride_t istrs2[], const _Complex float* src2);


#ifdef __cplusplus
}
#endif
