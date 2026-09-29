#include "misc/dimtypes.h"

#ifdef __cplusplus
extern "C" {
#endif

extern void cuda_ode_interval_bloch(bart_dim_t N, bart_dim_t SMM, bart_dim_t SMV, _Complex float* mag, bart_dim_t SPP, bart_dim_t SPV, const _Complex float* par, bart_dim_t Np, float Tp, const _Complex float* pulse, float h, float tol, float st, float end);
extern void cuda_ode_interval_bloch_sa(bart_dim_t N, bart_dim_t SMM, bart_dim_t SMV, _Complex float* mag, bart_dim_t SDMM, bart_dim_t SDMV, _Complex float* dmag, bart_dim_t SDPM, bart_dim_t SDPV, _Complex float* dpar, bart_dim_t SPM, bart_dim_t SPV, const _Complex float* par, bart_dim_t Np, float Tp, const _Complex float* pulse, float h, float tol, float st, float end);

#ifdef __cplusplus
}
#endif