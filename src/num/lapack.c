/* Copyright 2017. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Martin Uecker
 * 2017 Jon Tamir
 */

#include <complex.h>

#include "misc/misc.h"

#ifndef NO_LAPACK
#ifdef NOLAPACKE
#include "lapacke/lapacke.h"
#elif USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif
#endif

#include "lapack.h"

#ifdef NO_LAPACK
#define LAPACKE(x, ...) \
	if ((__VA_ARGS__, 0))	\
		error("LAPACK: " # x " failed.\n");
#else
#define LAPACKE(x, ...) \
	if (0 != LAPACKE_##x(LAPACK_COL_MAJOR, __VA_ARGS__))	\
		error("LAPACK: " # x " failed.\n");
#endif

/* ATTENTION: blas and lapack use column-major matrices
 * while native C uses row-major. All matrices are
 * transposed to what one would expect.
 *
 * LAPACK svd destroys its input matrix
 **/

void lapack_eig(bart_dim_t N, float eigenval[N], complex float matrix[N][N])
{
	LAPACKE(cheev, 'V', 'U', N, &matrix[0][0], N, eigenval);
}

// A*x = (lambda)*B*x
void lapack_geig(bart_dim_t N, float eigenval[N], complex float A[N][N], complex float B[N][N])
{
	LAPACKE(chegv, 1, 'V', 'U', N, &A[0][0], N, &B[0][0], N, eigenval);
}

void lapack_svd(bart_dim_t M, bart_dim_t N, complex float U[M][M], complex float VH[N][N], float S[(N > M) ? M : N], complex float A[N][M])
{
	LAPACKE(cgesdd, 'A', M, N, &A[0][0], M, S, &U[0][0], M, &VH[0][0], N);
}

// AT = VHT ST UT
void lapack_svd_econ(bart_dim_t M, bart_dim_t N,
		     complex float U[(N > M) ? M : N][M],
		     complex float VH[N][(N > M) ? M : N],
		     float S[(N > M) ? M : N],
		     complex float A[N][M])
{
	PTR_ALLOC(float[MIN(M, N) - 1], superb);
	LAPACKE(cgesvd, NULL != U ? 'S' : 'N', NULL != VH ? 'S' : 'N', M, N, &A[0][0], M, S, NULL != U ? &U[0][0] : NULL, M, NULL != VH ? &VH[0][0] : NULL, MIN(M, N), *superb);
	PTR_FREE(superb);
}

// A = QR in Fortran notation
// A is overwritten with Q only A[MIN(M,N)][M] are valid on exit
void lapack_qr_econ(bart_dim_t M, bart_dim_t N,
		    complex float R[N][(N > M) ? M : N],
		    complex float A[N][M])
{
	PTR_ALLOC(complex float[MIN(M, N)], tau);
	LAPACKE(cgeqrf, M, N, &A[0][0], M, *tau);

	if (NULL != R) {

		for (int i = 0; i < N; i++)
			for (int j = 0; j < MIN(M, N); j++)
				R[i][j] = (i <= j) ? A[i][j] : 0.;
	}

	LAPACKE(cungqr, M, MIN(M, N), MIN(M, N), &A[0][0], M, *tau);
	PTR_FREE(tau);
}

void lapack_eig_double(bart_dim_t N, double eigenval[N], complex double matrix[N][N])
{
	LAPACKE(zheev, 'V', 'U', N, &matrix[0][0], N, eigenval);
}

void lapack_svd_double(bart_dim_t M, bart_dim_t N, complex double U[M][M], complex double VH[N][N], double S[(N > M) ? M : N], complex double A[N][M])
{
	LAPACKE(zgesdd, 'A', M, N, &A[0][0], M, S, &U[0][0], M, &VH[0][0], N);
}

static void lapack_cholesky_UL(bart_dim_t N, char UL, complex float A[N][N])
{
	LAPACKE(cpotrf, UL, N, &A[0][0], N);
}

void lapack_cholesky(bart_dim_t N, complex float A[N][N])
{
	lapack_cholesky_UL(N, 'U', A);
}

void lapack_cholesky_lower(bart_dim_t N, complex float A[N][N])
{
	lapack_cholesky_UL(N, 'L', A);
}


static void lapack_trimat_inverse_UL(bart_dim_t N, char UL, complex float A[N][N])
{
	LAPACKE(ctrtri, UL, 'N', N, &A[0][0], N);
}

void lapack_trimat_inverse(bart_dim_t N, complex float A[N][N])
{
	lapack_trimat_inverse_UL(N, 'U', A);
}

void lapack_trimat_inverse_lower(bart_dim_t N, complex float A[N][N])
{
	lapack_trimat_inverse_UL(N, 'L', A);
}

// Solve A x = B for x
void lapack_trimat_solve(bart_dim_t N, bart_dim_t M, complex float A[N][N], complex float B[M][N], bool upper)
{
	// for non-unit ('N') triangular matrix A
	// on output: B overwritten by solution matrix X
	LAPACKE(ctrtrs, (upper ? 'U' : 'L'), 'N', 'N', N, M, &A[0][0], N, &B[0][0], N);
}


void lapack_cinverse_UL(bart_dim_t N, complex float A[N][N])
{
	int ipiv[N];

	LAPACKE(cgetrf, N, N, &A[0][0], N, ipiv);
	LAPACKE(cgetri, N, &A[0][0], N, ipiv);
}

void lapack_sinverse_UL(bart_dim_t N, float A[N][N])
{
	int ipiv[N];

	LAPACKE(sgetrf, N, N, &A[0][0], N, ipiv);
	LAPACKE(sgetri, N, &A[0][0], N, ipiv);
}


void lapack_schur(bart_dim_t N, complex float W[N], complex float VS[N][N], complex float A[N][N])
{
	int sdim = 0;

	// On output, A overwritten by Schur form T
	LAPACKE(cgees, 'V', 'N', NULL, N, &A[0][0], N, &sdim, &W[0], &VS[0][0], N);
}

void lapack_schur_double(bart_dim_t N, complex double W[N], complex double VS[N][N], complex double A[N][N])
{
	int sdim = 0;

	// On output, A overwritten by Schur form T
	LAPACKE(zgees, 'V', 'N', NULL, N, &A[0][0], N, &sdim, &W[0], &VS[0][0], N);
}

// Solves the complex Sylvester matrix equation
// op(A)*X + X*op(B) = scale*C
void lapack_sylvester(bart_dim_t N, bart_dim_t M, float* scale, complex float A[N][N], complex float B[M][M], complex float C[M][N])
{
	// A -> triangluar
	// On output: C overwritten by X
	LAPACKE(ctrsyl, 'N', 'N', +1, N, M, &A[0][0], N, &B[0][0], M, &C[0][0], N, scale);
}

void lapack_solve_real(bart_dim_t N, float A[N][N], float B[N])
{
	int ipiv[N];
	LAPACKE(sgesv, N, 1, &A[0][0], N, ipiv, B, N);
}


