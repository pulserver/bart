#include "misc/dimtypes.h"

void blas_zfmac_cgemm(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], _Complex float* optr, const bart_stride_t istr1[__VLA(N)], const _Complex float* iptr1, const bart_stride_t istr2[__VLA(N)], const _Complex float* iptr2);
void blas_zfmac_cgemv(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], _Complex float* optr, const bart_stride_t istr1[__VLA(N)], const _Complex float* iptr1, const bart_stride_t istr2[__VLA(N)], const _Complex float* iptr2);
void blas_zfmac_caxpy(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], _Complex float* optr, const bart_stride_t istr1[__VLA(N)], const _Complex float* iptr1, const bart_stride_t istr2[__VLA(N)], const _Complex float* iptr2);
void blas_zfmac_cgeru(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], _Complex float* optr, const bart_stride_t istr1[__VLA(N)], const _Complex float* iptr1, const bart_stride_t istr2[__VLA(N)], const _Complex float* iptr2);
void blas_zfmac_cdotu(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], _Complex float* optr, const bart_stride_t istr1[__VLA(N)], const _Complex float* iptr1, const bart_stride_t istr2[__VLA(N)], const _Complex float* iptr2);

void blas_fmac_sgemm(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], float* optr, const bart_stride_t istr1[__VLA(N)], const float* iptr1, const bart_stride_t istr2[__VLA(N)], const float* iptr2);
void blas_fmac_sgemv(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], float* optr, const bart_stride_t istr1[__VLA(N)], const float* iptr1, const bart_stride_t istr2[__VLA(N)], const float* iptr2);
void blas_fmac_saxpy(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], float* optr, const bart_stride_t istr1[__VLA(N)], const float* iptr1, const bart_stride_t istr2[__VLA(N)], const float* iptr2);
void blas_fmac_sger(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], float* optr, const bart_stride_t istr1[__VLA(N)], const float* iptr1, const bart_stride_t istr2[__VLA(N)], const float* iptr2);
void blas_fmac_sdot(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], float* optr, const bart_stride_t istr1[__VLA(N)], const float* iptr1, const bart_stride_t istr2[__VLA(N)], const float* iptr2);

void blas_zmul_cmatcopy(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], _Complex float* optr, const bart_stride_t istr1[__VLA(N)], const _Complex float* iptr1, const bart_stride_t istr2[__VLA(N)], const _Complex float* iptr2);
void blas_zsmul_cmatcopy(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], _Complex float* optr, const bart_stride_t istr[__VLA(N)], const _Complex float* iptr, _Complex float val);
void blas_zmul_cdgmm(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], _Complex float* optr, const bart_stride_t istr1[__VLA(N)], const _Complex float* iptr1, const bart_stride_t istr2[__VLA(N)], const _Complex float* iptr2);
void blas_zmul_cgeru(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], _Complex float* optr, const bart_stride_t istr1[__VLA(N)], const _Complex float* iptr1, const bart_stride_t istr2[__VLA(N)], const _Complex float* iptr2);
void blas_zmul_cscal(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], _Complex float* optr, const bart_stride_t istr1[__VLA(N)], const _Complex float* iptr1, const bart_stride_t istr2[__VLA(N)], const _Complex float* iptr2);

void blas_mul_smatcopy(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], float* optr, const bart_stride_t istr1[__VLA(N)], const float* iptr1, const bart_stride_t istr2[__VLA(N)], const float* iptr2);
void blas_smul_smatcopy(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], float* optr, const bart_stride_t istr[__VLA(N)], const float* iptr, float val);
void blas_mul_sdgmm(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], float* optr, const bart_stride_t istr1[__VLA(N)], const float* iptr1, const bart_stride_t istr2[__VLA(N)], const float* iptr2);
void blas_mul_sger(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], float* optr, const bart_stride_t istr1[__VLA(N)], const float* iptr1, const bart_stride_t istr2[__VLA(N)], const float* iptr2);
void blas_mul_sscal(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], float* optr, const bart_stride_t istr1[__VLA(N)], const float* iptr1, const bart_stride_t istr2[__VLA(N)], const float* iptr2);
