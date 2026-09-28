#include "misc/dimtypes.h"


#ifdef __cplusplus
extern "C" {
#endif

extern void cuda_xpay_bat(bart_dim_t Bi, bart_dim_t N, bart_dim_t Bo, const float* beta, float* a, const float* x);
extern void cuda_axpy_bat(bart_dim_t Bi, bart_dim_t N, bart_dim_t Bo, float* a, const float* alpha, const float* x);
extern void cuda_dot_bat(bart_dim_t Bi, bart_dim_t N, bart_dim_t Bo, float* dst, const float* x, const float* y);

#ifdef __cplusplus
}
#endif
