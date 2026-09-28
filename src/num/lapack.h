
#include "misc/dimtypes.h"
#include <complex.h>

extern void lapack_eig(bart_dim_t N, float eigenval[N], complex float matrix[N][N]);
extern void lapack_geig(bart_dim_t N, float eigenval[N], complex float A[N][N], complex float B[N][N]);
extern void lapack_svd(bart_dim_t M, bart_dim_t N, complex float U[M][M], complex float VH[N][N], float S[(N > M) ? M : N], complex float A[N][M]);
extern void lapack_svd_econ(bart_dim_t M, bart_dim_t N,
		     complex float U[(N > M) ? M : N][M],
		     complex float VH[N][(N > M) ? M : N],
		     float S[(N > M) ? M : N],
		     complex float A[N][M]);

extern void lapack_qr_econ(bart_dim_t M, bart_dim_t N,
		    complex float R[N][(N > M) ? M : N],
		    complex float A[N][M]);

extern void lapack_eig_double(bart_dim_t N, double eigenval[N], complex double matrix[N][N]);
extern void lapack_svd_double(bart_dim_t M, bart_dim_t N, complex double U[M][M], complex double VH[N][N], double S[(N > M) ? M : N], complex double A[N][M]);
extern void lapack_matrix_multiply(bart_dim_t M, bart_dim_t N, bart_dim_t K, complex float C[M][N], const complex float A[M][K], const complex float B[K][N]);

extern void lapack_cholesky(bart_dim_t N, complex float A[N][N]);
extern void lapack_cholesky_lower(bart_dim_t N, complex float A[N][N]);

extern void lapack_trimat_inverse(bart_dim_t N, complex float A[N][N]);
extern void lapack_trimat_inverse_lower(bart_dim_t N, complex float A[N][N]);
extern void lapack_trimat_solve(bart_dim_t N, bart_dim_t M, complex float A[N][N], complex float B[M][N], bool upper);

extern void lapack_schur(bart_dim_t N, complex float W[N], complex float VS[N][N], complex float A[N][N]);
extern void lapack_schur_double(bart_dim_t N, complex double W[N], complex double VS[N][N], complex double A[N][N]);

extern void lapack_sylvester(bart_dim_t N, bart_dim_t M, float* scale, complex float A[N][N], complex float B[M][M], complex float C[M][N]);

extern void lapack_cinverse_UL(bart_dim_t N, complex float A[N][N]);
extern void lapack_sinverse_UL(bart_dim_t N, float A[N][N]);

extern void lapack_solve_real(bart_dim_t N, float A[N][N], float B[N]);


