
#ifdef __cplusplus
extern "C" {
#endif

#include "misc/dimtypes.h"
#include "misc/mri.h"

extern void rss_combine(const bart_dim_t dims[DIMS], _Complex float* image, const _Complex float* data);
extern void optimal_combine(const bart_dim_t dims[DIMS], float alpha, _Complex float* image, const _Complex float* sens, const _Complex float* data);
extern float estimate_scaling_norm(float rescale, int imsize, _Complex float* tmpnorm, bool compat, float p);
extern float estimate_scaling(const bart_dim_t dims[DIMS], const _Complex float* sens, const _Complex float* data, float p);
extern float estimate_scaling2(const bart_dim_t dims[DIMS], const _Complex float* sens, const bart_stride_t strs[DIMS], const _Complex float* data, float p);
extern float estimate_scaling_old2(const bart_dim_t dims[DIMS], const _Complex float* sens, const bart_stride_t strs[DIMS], const _Complex float* data);
extern void fake_kspace(const bart_dim_t dims[DIMS], _Complex float* kspace, const _Complex float* sens, const _Complex float* image);
extern void replace_kspace(const bart_dim_t dims[DIMS], _Complex float* out, const _Complex float* kspace, const _Complex float* sens, const _Complex float* image);
extern void replace_kspace2(const bart_dim_t dims[DIMS], _Complex float* out, const _Complex float* kspace, const _Complex float* sens, const _Complex float* image);

extern float estimate_scaling_cal(const bart_dim_t dims[DIMS], const _Complex float* sens, const bart_dim_t cal_dims[DIMS], const _Complex float* cal_data, bool compat, float p);

#ifdef __cplusplus
}
#endif

