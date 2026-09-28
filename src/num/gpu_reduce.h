#ifdef __cplusplus
extern "C" {
#endif

extern void cuda_reduce_zadd_inner(bart_dim_t dim_reduce, bart_dim_t dim_batch, _Complex float* dst, const _Complex float* src);
extern void cuda_reduce_zadd_outer(bart_dim_t dim_reduce, bart_dim_t dim_batch, _Complex float* dst, const _Complex float* src);

extern void cuda_reduce_zmax_inner(bart_dim_t dim_reduce, bart_dim_t dim_batch, _Complex float* dst, const _Complex float* src);
extern void cuda_reduce_zmax_outer(bart_dim_t dim_reduce, bart_dim_t dim_batch, _Complex float* dst, const _Complex float* src);

extern void cuda_reduce_add_inner(bart_dim_t dim_reduce, bart_dim_t dim_batch, float* dst, const float* src);
extern void cuda_reduce_add_outer(bart_dim_t dim_reduce, bart_dim_t dim_batch, float* dst, const float* src);

#ifdef __cplusplus
}
#endif
