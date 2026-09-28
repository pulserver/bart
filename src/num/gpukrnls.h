
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

extern double cuda_dot(bart_dim_t N, const float* src1, const float* src2);
extern double cuda_norm(bart_dim_t N, const float* src);
extern _Complex double cuda_cdot(bart_dim_t N, const _Complex float* src1, const _Complex float* src2);

extern void cuda_float2double(bart_dim_t size, double* dst, const float* src);
extern void cuda_double2float(bart_dim_t size, float* dst, const double* src);
extern void cuda_sxpay(bart_dim_t size, float* y, float alpha, const float* src);
extern void cuda_xpay(bart_dim_t N, float beta, float* dst, const float* src);
extern void cuda_axpbz(bart_dim_t N, float* dst, const float a, const float* x, const float b, const float* z);
extern void cuda_smul(bart_dim_t N, float alpha, float* dst, const float* src);
extern void cuda_mul(bart_dim_t N, float* dst, const float* src1, const float* src2);
extern void cuda_div(bart_dim_t N, float* dst, const float* src1, const float* src2);
extern void cuda_add(bart_dim_t N, float* dst, const float* src1, const float* src2);
extern void cuda_sadd(bart_dim_t N, float val, float* dst, const float* src1);
extern void cuda_zsadd(bart_dim_t N, _Complex float val, _Complex float* dst, const _Complex float* src1);
extern void cuda_sub(bart_dim_t N, float* dst, const float* src1, const float* src2);
extern void cuda_fmac(bart_dim_t N, float* dst, const float* src1, const float* src2);
extern void cuda_fmacD(bart_dim_t N, double* dst, const float* src1, const float* src2);
extern void cuda_zsmul(bart_dim_t N, _Complex float alpha, _Complex float* dst, const _Complex float* src1);
extern void cuda_zmul(bart_dim_t N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
extern void cuda_zdiv(bart_dim_t N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
extern void cuda_zfmac(bart_dim_t N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
extern void cuda_zfmacD(bart_dim_t N, _Complex double* dst, const _Complex float* src1, const _Complex float* src2);
extern void cuda_zmulc(bart_dim_t N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
extern void cuda_zfmacc(bart_dim_t N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
extern void cuda_zfmaccD(bart_dim_t N, _Complex double* dst, const _Complex float* src1, const _Complex float* src2);
extern void cuda_zfsq2(bart_dim_t N, _Complex float* dst, const _Complex float* src);
extern void cuda_pow(bart_dim_t N, float* dst, const float* src1, const float* src2);
extern void cuda_zpow(bart_dim_t N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
extern void cuda_sqrt(bart_dim_t N, float* dst, const float* src);
extern void cuda_round(bart_dim_t N, float* dst, const float* src);
extern void cuda_zconj(bart_dim_t N, _Complex float* dst, const _Complex float* src);
extern void cuda_zphsr(bart_dim_t N, _Complex float* dst, const _Complex float* src);
extern void cuda_zexpj(bart_dim_t N, _Complex float* dst, const _Complex float* src);
extern void cuda_zexp(bart_dim_t N, _Complex float* dst, const _Complex float* src);
extern void cuda_zsin(bart_dim_t N, _Complex float* dst, const _Complex float* src);
extern void cuda_zcos(bart_dim_t N, _Complex float* dst, const _Complex float* src);
extern void cuda_zasin(bart_dim_t N, _Complex float* dst, const _Complex float* src);
extern void cuda_zacos(bart_dim_t N, _Complex float* dst, const _Complex float* src);
extern void cuda_zsinh(bart_dim_t N, _Complex float* dst, const _Complex float* src);
extern void cuda_zcosh(bart_dim_t N, _Complex float* dst, const _Complex float* src);
extern void cuda_zlog(bart_dim_t N, _Complex float* dst, const _Complex float* src);
extern void cuda_zarg(bart_dim_t N, _Complex float* dst, const _Complex float* src);
extern void cuda_zabs(bart_dim_t N, _Complex float* dst, const _Complex float* src);
extern void cuda_zatanr(bart_dim_t N, _Complex float* dst, const _Complex float* src);
extern void cuda_zatan2r(bart_dim_t N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
extern void cuda_zacosr(bart_dim_t N, _Complex float* dst, const _Complex float* src);
extern void cuda_exp(bart_dim_t N, float* dst, const float* src);
extern void cuda_log(bart_dim_t N, float* dst, const float* src);
extern void cuda_zsoftthresh_half(bart_dim_t N, float lambda, _Complex float* d, const _Complex float* x);
extern void cuda_zsoftthresh(bart_dim_t N, float lambda, _Complex float* d, const _Complex float* x);
extern void cuda_softthresh_half(bart_dim_t N, float lambda, float* d, const float* x);
extern void cuda_softthresh(bart_dim_t N, float lambda, float* d, const float* x);
extern void cuda_zreal(bart_dim_t N, _Complex float* dst, const _Complex float* src);
extern void cuda_zcmp(bart_dim_t N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
extern void cuda_zdiv_reg(bart_dim_t N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2, _Complex float lambda);
extern void cuda_le(bart_dim_t N, float* dst, const float* src1, const float* src2);
extern void cuda_zfftmod(bart_dim_t N, _Complex float* dst, const _Complex float* src, int n, bool inv, double phase);
extern void cuda_zfftmod_3d(const bart_dim_t dims[3], _Complex float* dst, const _Complex float* src, bool inv, double phase);
extern void cuda_zfftmod_1d(bart_dim_t N, _Complex float* dst, const _Complex float* src, bool inv, double phase);
extern void cuda_zmax(bart_dim_t N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
extern void cuda_zle(bart_dim_t N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
extern void cuda_smax(bart_dim_t N, float val, float* dst, const float* src1);
extern void cuda_smin(bart_dim_t N, float val, float* dst, const float* src1);
extern void cuda_max(bart_dim_t N, float* dst, const float* src1, const float* src2);
extern void cuda_min(bart_dim_t N, float* dst, const float* src1, const float* src2);
extern void cuda_zsum(bart_dim_t N, _Complex float* dst);
extern void cuda_zsmax(bart_dim_t N, float alpha, _Complex float* dst, const _Complex float* src);
extern void cuda_zsmin(bart_dim_t N, float alpha, _Complex float* dst, const _Complex float* src);
extern void cuda_pdf_gauss(bart_dim_t N, float mu, float sig, float* dst, const float* src);
extern void cuda_real(bart_dim_t N, float* dst, const _Complex float* src);
extern void cuda_imag(bart_dim_t N, float* dst, const _Complex float* src);
extern void cuda_zcmpl_real(bart_dim_t N, _Complex float* dst, const float* src);
extern void cuda_zcmpl_imag(bart_dim_t N, _Complex float* dst, const float* src);
extern void cuda_zcmpl(bart_dim_t N, _Complex float* dst, const float* real_src, const float* imag_src);
extern void cuda_zfill(bart_dim_t N, _Complex float val, _Complex float* dst);

extern void cuda_mask_compress(bart_dim_t N, uint32_t* dst, const float* src);
extern void cuda_mask_decompress(bart_dim_t N, float* dst, const uint32_t* src);

extern void cuda_zfmac_strided(bart_dim_t N, bart_dim_t dims[3], bart_flags_t oflags, bart_flags_t iflags1, bart_flags_t iflags2, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
extern void cuda_zfmacc_strided(bart_dim_t N, bart_dim_t dims[3], bart_flags_t oflags, bart_flags_t iflags1, bart_flags_t iflags2, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);

extern void cuda_fmacD_dot(bart_dim_t N, double* dst, const float* src1, const float* src2);
extern void cuda_zfmaccD_dot(bart_dim_t N, _Complex double* dst, const _Complex float* src1, const _Complex float* src2);

extern void cuda_addD(bart_dim_t N, double* dst, const double* src1, const double* src2);

#ifdef __cplusplus
}
#endif
