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
	LAPACKE(cheev, 'V', 'U', checked_int(N), &matrix[0][0], checked_int(N), eigenval);
}

// A*x = (lambda)*B*x
void lapack_geig(bart_dim_t N, float eigenval[N], complex float A[N][N], complex float B[N][N])
{
	LAPACKE(chegv, 1, 'V', 'U', checked_int(N), &A[0][0], checked_int(N), &B[0][0], checked_int(N), eigenval);
}

void lapack_svd(bart_dim_t M, bart_dim_t N, complex float U[M][M], complex float VH[N][N], float S[(N > M) ? M : N], complex float A[N][M])
{
	LAPACKE(cgesdd, 'A', checked_int(M), checked_int(N), &A[0][0], checked_int(M), S, &U[0][0], checked_int(M), &VH[0][0], checked_int(N));
}

// AT = VHT ST UT
void lapack_svd_econ(bart_dim_t M, bart_dim_t N,
		     complex float U[(N > M) ? M : N][M],
		     complex float VH[N][(N > M) ? M : N],
		     float S[(N > M) ? M : N],
		     complex float A[N][M])
{
	PTR_ALLOC(float[MIN(M, N) - 1], superb);
	LAPACKE(cgesvd, NULL != U ? 'S' : 'N', NULL != VH ? 'S' : 'N', checked_int(M), checked_int(N), &A[0][0], checked_int(M), S, NULL != U ? &U[0][0] : NULL, checked_int(M), NULL != VH ? &VH[0][0] : NULL, checked_int(MIN(M, N)), *superb);
	PTR_FREE(superb);
}

// A = QR in Fortran notation
// A is overwritten with Q only A[MIN(M,N)][M] are valid on exit
void lapack_qr_econ(bart_dim_t M, bart_dim_t N,
		    complex float R[N][(N > M) ? M : N],
		    complex float A[N][M])
{
	PTR_ALLOC(complex float[MIN(M, N)], tau);
	LAPACKE(cgeqrf, checked_int(M), checked_int(N), &A[0][0], checked_int(M), *tau);

	if (NULL != R) {

		for (int i = 0; i < N; i++)
			for (int j = 0; j < MIN(M, N); j++)
				R[i][j] = (i <= j) ? A[i][j] : 0.;
	}

	LAPACKE(cungqr, checked_int(M), checked_int(MIN(M, N)), checked_int(MIN(M, N)), &A[0][0], checked_int(M), *tau);
	PTR_FREE(tau);
}

void lapack_eig_double(bart_dim_t N, double eigenval[N], complex double matrix[N][N])
{
	LAPACKE(zheev, 'V', 'U', checked_int(N), &matrix[0][0], checked_int(N), eigenval);
}

void lapack_svd_double(bart_dim_t M, bart_dim_t N, complex double U[M][M], complex double VH[N][N], double S[(N > M) ? M : N], complex double A[N][M])
{
	LAPACKE(zgesdd, 'A', checked_int(M), checked_int(N), &A[0][0], checked_int(M), S, &U[0][0], checked_int(M), &VH[0][0], checked_int(N));
}

static void lapack_cholesky_UL(bart_dim_t N, char UL, complex float A[N][N])
{
	LAPACKE(cpotrf, UL, checked_int(N), &A[0][0], checked_int(N));
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
	LAPACKE(ctrtri, UL, 'N', checked_int(N), &A[0][0], checked_int(N));
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
	LAPACKE(ctrtrs, (upper ? 'U' : 'L'), 'N', 'N', checked_int(N), checked_int(M), &A[0][0], checked_int(N), &B[0][0], checked_int(N));
}


void lapack_cinverse_UL(bart_dim_t N, complex float A[N][N])
{
	int ipiv[N];

	LAPACKE(cgetrf, checked_int(N), checked_int(N), &A[0][0], checked_int(N), ipiv);
	LAPACKE(cgetri, checked_int(N), &A[0][0], checked_int(N), ipiv);
}

void lapack_sinverse_UL(bart_dim_t N, float A[N][N])
{
	int ipiv[N];

	LAPACKE(sgetrf, checked_int(N), checked_int(N), &A[0][0], checked_int(N), ipiv);
	LAPACKE(sgetri, checked_int(N), &A[0][0], checked_int(N), ipiv);
}


void lapack_schur(bart_dim_t N, complex float W[N], complex float VS[N][N], complex float A[N][N])
{
	int sdim = 0;

	// On output, A overwritten by Schur form T
	LAPACKE(cgees, 'V', 'N', NULL, checked_int(N), &A[0][0], checked_int(N), &sdim, &W[0], &VS[0][0], checked_int(N));
}

void lapack_schur_double(bart_dim_t N, complex double W[N], complex double VS[N][N], complex double A[N][N])
{
	int sdim = 0;

	// On output, A overwritten by Schur form T
	LAPACKE(zgees, 'V', 'N', NULL, checked_int(N), &A[0][0], checked_int(N), &sdim, &W[0], &VS[0][0], checked_int(N));
}

// Solves the complex Sylvester matrix equation
// op(A)*X + X*op(B) = scale*C
void lapack_sylvester(bart_dim_t N, bart_dim_t M, float* scale, complex float A[N][N], complex float B[M][M], complex float C[M][N])
{
	// A -> triangluar
	// On output: C overwritten by X
	LAPACKE(ctrsyl, 'N', 'N', +1, checked_int(N), checked_int(M), &A[0][0], checked_int(N), &B[0][0], checked_int(M), &C[0][0], checked_int(N), scale);
}

void lapack_solve_real(bart_dim_t N, float A[N][N], float B[N])
{
	int ipiv[N];
	LAPACKE(sgesv, checked_int(N), 1, &A[0][0], checked_int(N), ipiv, B, checked_int(N));
}


