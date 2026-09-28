
#ifdef __cplusplus
extern "C" {
#endif

extern void cuda_zrfmac_upper_triagmat(bart_dim_t N, bart_dim_t NC, bart_dim_t NM, bart_stride_t ostr, bart_stride_t istr, bart_stride_t mstr, float* dst, const float* src, const float* mat);

#ifdef __cplusplus
}
#endif
