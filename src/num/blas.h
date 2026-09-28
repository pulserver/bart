
#include <complex.h>

#include "misc/misc.h"

#ifdef USE_CUDA
void cublas_init(void);
void cublas_deinit(void);

extern double cuda_asum(bart_dim_t size, const float* src);
extern void cuda_saxpy(bart_dim_t size, float* y, float alpha, const float* src);
extern void cuda_swap(bart_dim_t size, float* a, float* b);
#endif



extern void blas_cgemm(char transa, char transb, bart_dim_t M, bart_dim_t N, bart_dim_t K, const _Complex float alpha, bart_dim_t lda, const _Complex float* A, bart_dim_t ldb, const _Complex float* B, const _Complex float beta, bart_dim_t ldc, _Complex float* C);
extern void blas2_cgemm(char transa, char transb, bart_dim_t M, bart_dim_t N, bart_dim_t K, const _Complex float* alpha, bart_dim_t lda, const _Complex float* A, bart_dim_t ldb, const _Complex float* B, const _Complex float* beta, bart_dim_t ldc, _Complex float* C);
extern void blas_sgemm(char transa, char transb, bart_dim_t M, bart_dim_t N, bart_dim_t K, const float alpha, bart_dim_t lda, const  float* A, bart_dim_t ldb, const  float* B, const  float beta, bart_dim_t ldc,  float* C);
extern void blas2_sgemm(char transa, char transb, bart_dim_t M, bart_dim_t N, bart_dim_t K, const float* alpha, bart_dim_t lda, const  float* A, bart_dim_t ldb, const  float* B, const  float* beta, bart_dim_t ldc,  float* C);

extern void blas_cgemv(char trans, bart_dim_t M, bart_dim_t N, _Complex float alpha, bart_dim_t lda, const _Complex float* A, bart_dim_t incx, const _Complex float* x, _Complex float beta, bart_dim_t incy, _Complex float* y);
extern void blas2_cgemv(char trans, bart_dim_t M, bart_dim_t N, const _Complex float* alpha, bart_dim_t lda, const _Complex float* A, bart_dim_t incx, const _Complex float* x, _Complex float* beta, bart_dim_t incy, _Complex float* y);
extern void blas_sgemv(char trans, bart_dim_t M, bart_dim_t N, float alpha, bart_dim_t lda, const float* A, bart_dim_t incx, const float* x, float beta, bart_dim_t incy, float* y);
extern void blas2_sgemv(char trans, bart_dim_t M, bart_dim_t N, const float* alpha, bart_dim_t lda, const float* A, bart_dim_t incx, const float* x, float* beta, bart_dim_t incy, float* y);

extern void blas_sger(bart_dim_t M, bart_dim_t N, float alpha, bart_dim_t incx, const float* x, bart_dim_t incy, const float* y, bart_dim_t lda, float* A);
extern void blas2_sger(bart_dim_t M, bart_dim_t N, const float* alpha, bart_dim_t incx, const float* x, bart_dim_t incy, const float* y, bart_dim_t lda, float* A);
extern void blas_cgeru(bart_dim_t M, bart_dim_t N, _Complex float alpha, bart_dim_t incx, const _Complex float* x, bart_dim_t incy, const _Complex float* y, bart_dim_t lda, _Complex float* A);
extern void blas2_cgeru(bart_dim_t M, bart_dim_t N, const _Complex float* alpha, bart_dim_t incx, const _Complex float* x, bart_dim_t incy, const _Complex float* y, bart_dim_t lda, _Complex float* A);

extern void blas2_caxpy(bart_dim_t N, const _Complex float* alpha, bart_dim_t incx, const _Complex float* x, bart_dim_t incy, _Complex float* y);
extern void blas_caxpy(bart_dim_t N, _Complex float alpha, bart_dim_t incx, const _Complex float* x, bart_dim_t incy, _Complex float* y);
extern void blas2_saxpy(bart_dim_t N, const float* alpha, bart_dim_t incx, const float* x, bart_dim_t incy, float* y);
extern void blas_saxpy(bart_dim_t N, float alpha, bart_dim_t incx, const float* x, bart_dim_t incy, float* y);

extern void blas2_cscal(bart_dim_t N, const _Complex float* alpha, bart_dim_t incx, _Complex float* x);
extern void blas_cscal(bart_dim_t N, _Complex float alpha, bart_dim_t incx, _Complex float* x);
extern void blas2_sscal(bart_dim_t N, const float* alpha, bart_dim_t incx, float* x);
extern void blas_sscal(bart_dim_t N, float alpha, bart_dim_t incx, float* x);

extern void blas_cdgmm(bart_dim_t M, bart_dim_t N, _Bool left_mul, const _Complex float* A, bart_dim_t lda, const _Complex float* x, bart_dim_t incx, _Complex float* C, bart_dim_t ldc);
extern void blas_sdgmm(bart_dim_t M, bart_dim_t N, _Bool left_mul, const float* A, bart_dim_t lda, const float* x, bart_dim_t incx, float* C, bart_dim_t ldc);

extern void blas2_cdotu(_Complex float* result, bart_dim_t N, bart_dim_t incx, const _Complex float* x, bart_dim_t incy, const _Complex float* y);
extern void blas2_sdot(float* result, bart_dim_t N, bart_dim_t incx, const float* x, bart_dim_t incy, const float* y);

extern void blas_cmatcopy(char trans, bart_dim_t M, bart_dim_t N, _Complex float alpha, const _Complex float* A, bart_dim_t lda, _Complex float* B, bart_dim_t ldb);
extern void blas2_cmatcopy(char trans, bart_dim_t M, bart_dim_t N, const _Complex float* alpha, const _Complex float* A, bart_dim_t lda, _Complex float* B, bart_dim_t ldb);
extern void blas_smatcopy(char trans, bart_dim_t M, bart_dim_t N, float alpha, const float* A, bart_dim_t lda, float* B, bart_dim_t ldb);
extern void blas2_smatcopy(char trans, bart_dim_t M, bart_dim_t N, const float* alpha, const float* A, bart_dim_t lda, float* B, bart_dim_t ldb);

extern void blas_csyrk(char uplow, char trans, bart_dim_t N, bart_dim_t K, _Complex float alpha, bart_dim_t lda, const _Complex float A[][lda], _Complex float beta, bart_dim_t ldc, _Complex float C[][ldc]);

extern void blas_matrix_multiply(bart_dim_t M, bart_dim_t N, bart_dim_t K, _Complex float C[N][M], const _Complex float A[K][M], const _Complex float B[N][K]);

extern void blas_matrix_zfmac(bart_dim_t M, bart_dim_t N, bart_dim_t K, _Complex float* C, const _Complex float* A, char transa, const _Complex float* B, char transb);
extern void blas_gemv_zfmac(bart_dim_t M, bart_dim_t N, _Complex float* y, const _Complex float* A, char trans, const _Complex float* x);
extern void blas_gemv_fmac(bart_dim_t M, bart_dim_t N, float* y, const float* A, char trans, const float* x);
extern void blas_sger_fmac(bart_dim_t M, bart_dim_t N, float* A, const float* x, const float* y);
