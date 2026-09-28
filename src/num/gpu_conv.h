
#ifdef __cplusplus
extern "C" {
#endif

extern void cuda_im2col(_Complex float* dst, const _Complex float* src, const bart_dim_t odims[5], const bart_dim_t idims[5], const bart_dim_t kdims[5], const bart_dim_t dilation[5], const bart_stride_t strides[5]);
extern void cuda_im2col_transp(_Complex float* dst, const _Complex float* src, const bart_dim_t odims[5], const bart_dim_t idims[5], const bart_dim_t kdims[5], const bart_dim_t dilation[5], const bart_stride_t strides[5]);

#ifdef __cplusplus
}
#endif
