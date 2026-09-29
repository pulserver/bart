#ifdef __cplusplus
extern "C" {
#endif

extern void cuda_ode_interval_bloch(long N, long SMM, long SMV, _Complex float* mag, long SPP, long SPV, const _Complex float* par, long Np, float Tp, const _Complex float* pulse, float h, float tol, float st, float end);
extern void cuda_ode_interval_bloch_sa(long N, long SMM, long SMV, _Complex float* mag, long SDMM, long SDMV, _Complex float* dmag, long SDPM, long SDPV, _Complex float* dpar, long SPM, long SPV, const _Complex float* par, long Np, float Tp, const _Complex float* pulse, float h, float tol, float st, float end);

#ifdef __cplusplus
}
#endif