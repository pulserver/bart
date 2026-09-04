/* Copyright 2018-2023. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "misc/nested.h"

#include "num/linalg.h"
#include "num/quadrature.h"

#include "ode.h"

#define ode_type float
#define ODE_SUFFIX(x) x
#define ODE_FUN ode_fun_t
#define ODE_POW powf
#define ODE_VEC_COPY vecf_copy
#define ODE_VEC_SAXPY vecf_saxpy
#define ODE_VEC_NORM vecf_norm
#include "ode.inc"

#undef ode_type
#define ode_type double
#undef ODE_SUFFIX
#define ODE_SUFFIX(x) x##_d
#undef ODE_FUN
#define ODE_FUN ode_fun_t_d
#undef ODE_POW
#define ODE_POW pow
#undef ODE_VEC_COPY
#define ODE_VEC_COPY vecd_copy
#undef ODE_VEC_SAXPY
#define ODE_VEC_SAXPY vecd_saxpy
#undef ODE_VEC_NORM
#define ODE_VEC_NORM vecd_norm
#include "ode.inc"

#if 0
static void euler(float h, int N, float x[N], float st, float end,
	void CLOSURE_TYPE(f)(int N, float (*matrix)[N][N], float t))
{
	for (float t = st; t < end; ) {

		float A[N][N];
		NESTED_CALL(f, (N, &A, t));

		float tmp[N];

		matf_vecmul(N, N, tmp, A, x);

		t += h;

		vecf_saxpy(N, x, h, tmp);

		if (t + h > end)
			h = end - t;
	}
}
#endif

void (crank_nicolson)(float h, int N, float x[N], float st, float end, ode_cn_f f)
{
	for (float t = st; t < end; ) {

		float A[N][N];
		NESTED_CALL(f, (N, &A, t + h / 2.));

		/* (x_n - x_{n-1}) / dt = A (x_n + x_{n-1}) / 2
		 * x_n - dt / 2 A x_n = x_{n-1} + dt / 2 A x_{n-1}
		 * (I - dt / 2 A) x_n = (I + dt / 2 A) x_{n-1}
		 * x_{n} = (I - dt / 2 A)^{-1} (I + dt / 2 A) x_{n-1}
		 */

		float B[N][N];
		// (I + dt/2 * A)
		matf_identity(N, N, B);
		matf_saxpy(N, N, B, +h / 2., A);

		float tmp[N];
		matf_vecmul(N, N, tmp, B, x); // tmp = (I + dt/2 * A) * x_{n-1}

		t += h;

		// (I - dt/2 * A)
		matf_identity(N, N, B);
		matf_saxpy(N, N, B, -h / 2., A);
		matf_solve(N, x, B, tmp);

		if (t + h > end)
			h = end - t;
	}
}


void crank_nicolson_matrix(float h, int N, float x[N], float st, float end, const float matrix[N][N])
{
#ifdef __clang__
	const void* matrix2 = matrix;	// clang workaround
#endif
	NESTED(void, ode_matrix_fun, (int N, float (*A)[N][N], float t))
	{
		(void)t;
#ifdef __clang__
		const float (*matrix)[N] = matrix2;
#endif
		matf_copy(N, N, *A, matrix);
	};

	crank_nicolson(h, N, x, st, end, ode_matrix_fun);
}

void (crank_nicolson_adjoint)(float h, int N, float x[N], float st, float end, ode_cn2_f f)
{
	for (float t = end; t > st; ) {

		float Ak[N][N];
		float Akp1[N][N];
		NESTED_CALL(f, (N, &Ak, &Akp1, t - h / 2.));

		/* (I - dt / 2 Ak) x_n = (I + dt / 2 Akp1) x_{n+1}
		 * x_{n} = (I - dt / 2 Ak)^{-1} (I + dt / 2 Akp1) x_{n+1}
		 */

		 // (I + dt/2 * Akp1)
		float B[N][N];
		matf_identity(N, N, B);
		matf_saxpy(N, N, B, +h / 2., Akp1);

		float tmp[N];
		matf_vecmul(N, N, tmp, B, x); // tmp = (I + dt/2 * Akp1) * x_{n+1}

		t -= h;

		// (I - dt/2 * Ak)
		matf_identity(N, N, B);
		matf_saxpy(N, N, B, -h / 2., Ak);
		matf_solve(N, x, B, tmp);

		if (t - h < st)
			h = t - st;
	}
}

void crank_nicolson_matrix_adjoint(float h, int N, float x[N], float st, float end, 
	const float matrix_ak[N][N], const float matrix_akp1[N][N])
{
#ifdef __clang__
	const void* matrix1 = matrix_ak;	// clang workaround
	const void* matrix2 = matrix_akp1;	// clang workaround
#endif
	NESTED(void, ode_matrix_fun, (int N, float (*Ak)[N][N], float (*Akp1)[N][N], float t))
	{
		(void)t;
#ifdef __clang__
		const float (*matrix_ak)[N] = matrix1;
		const float (*matrix_akp1)[N] = matrix2;
#endif
		matf_copy(N, N, *Ak, matrix_ak);
		matf_copy(N, N, *Akp1, matrix_akp1);
	};

	crank_nicolson_adjoint(h, N, x, st, end, ode_matrix_fun);
}

// Runge-Kutta 4

void rk4_step(float h, int N, float ynp[N], float tn, const float yn[N], ode_fun_t f)
{
	const float c[3] = { 0.5, 0.5, 1. };

	const float a[6] = {
		0.5,
		0.0, 0.5,
		0.0, 0.0, 1.0,
	};
	const float b[4] = { 1. / 6., 1. / 3., 1. / 3., 1. / 6. };

	float k[1][N];	// K = 1 because only diagonal elements are used
	NESTED_CALL(f, (k[0], tn, yn));

	float tmp[N];
	runge_kutta_step(h, 4, a, b, c, N, 1, k, ynp, tmp, tn, yn, f);
}



/*
 * Dormand JR, Prince PJ. A family of embedded Runge-Kutta formulae,
 * Journal of Computational and Applied Mathematics 6:19-26 (1980).
 */
void dormand_prince_step(float h, int N, float ynp[N], float tn, const float yn[N], ode_fun_t f)
{
	const float c[6] = { 1. / 5., 3. / 10., 4. / 5., 8. / 9., 1., 1. };

	const float a[tridiag(7)] = {
		1. / 5.,
		3. / 40.,	9. / 40.,
		44. / 45.,	-56. / 15.,	32. / 9.,
		19372. / 6561.,	-25360. / 2187., 64448. / 6561., -212. / 729.,
		9017. / 3168.,  -355. / 33.,	46732. / 5247.,	49. / 176.,	-5103. / 18656.,
		35. / 384.,	0.,		500. / 1113.,	125. / 192.,	-2187. / 6784.,	11. / 84.,
	};

	const float b[7] = { 5179. / 57600., 0.,  7571. / 16695., 393. / 640., -92097. / 339200., 187. / 2100., 1. / 40. };

	float k[6][N];
	NESTED_CALL(f, (k[0], tn, yn));

	float tmp[N];
	runge_kutta_step(h, 7, a, b, c, N, 6, k, ynp, tmp, tn, yn, f);
}


void ode_interval2(float h, float tol,
	int N, const float t[N + 1],
	int M, float x[N + 1][M],
	ode_fun_t sys)
{
	for (int i = 0; i < N; i++) {

		for (int m = 0; m < M; m++)
			x[i + 1][m] = x[i][m];

		(ode_interval)(h, -1, tol, M, x[i + 1], t[i], t[i + 1], sys);
	}
}



void ode_matrix_interval(float h, float tol, int N, float x[N], float st, float end, const float matrix[N][N])
{
#ifdef __clang__
	const void* matrix2 = matrix;	// clang workaround
#endif
	NESTED(void, ode_matrix_fun, (float* x, float t, const float* in))
	{
		(void)t;
#ifdef __clang__
		const float (*matrix)[N] = matrix2;
#endif
		for (int i = 0; i < N; i++) {

			x[i] = 0.;

			for (int j = 0; j < N; j++)
				x[i] += matrix[i][j] * in[j];
		}
	};

	ode_interval(h, -1, tol, N, x, st, end, ode_matrix_fun);
}


// the adjoint method for sensitivity analysis
// void (*s)(void* data, float* out, float t)
// void M

void ode_adjoint_sa_noinit(float h, float tol,
	int N, const float t[N + 1],
	int M, float z[N + 1][M],
	ode_sys_t sysT,
	ode_cost_t cost)
{
	// adjoint state

	for (int i = N; 0 < i; i--) {

		for (int m = 0; m < M; m++)
			z[i - 1][m] = z[i][m];

		// invert time -> neg. sign on RHS

		NESTED(void, asa_eval, (float out[M], float t, const float yn[M]))
		{
			NESTED_CALL(sysT, (out, -t, yn));

			float off[M];
			NESTED_CALL(cost, (off, -t));

			for (int m = 0; m < M; m++)
				out[m] += off[m];
		};

		ode_interval(h, -1, tol, M, z[i - 1], -t[i], -t[i - 1], asa_eval);
	}
}


void ode_adjoint_sa(float h, float tol,
	int N, const float t[N + 1],
	int M, float x[N + 1][M], float z[N + 1][M],
	const float x0[M],
	ode_sys_t sys,
	ode_sys_t sysT,
	ode_cost_t cost)
{
	// forward solution

	for (int m = 0; m < M; m++)
		x[0][m] = x0[m];

	ode_interval2(h, tol, N, t, M, x, sys);

	// adjoint solution

	for (int m = 0; m < M; m++)
		z[N][m] = 0.;

	ode_adjoint_sa_noinit(h, tol, N, t, M, z, sysT, cost);
}

void ode_matrix_adjoint_sa(float h, float tol,
	int N, const float t[N + 1],
	int M, float x[N + 1][M], float z[N + 1][M],
	const float x0[M], const float sys[N][M][M],
	const float cost[N][M])
{
	// forward solution

	for (int m = 0; m < M; m++)
		x[0][m] = x0[m];

	for (int i = 0; i < N; i++) {

		for (int m = 0; m < M; m++)
			x[i + 1][m] = x[i][m];

		ode_matrix_interval(h, tol, M, x[i + 1], t[i], t[i + 1], sys[i]);
	}

	// adjoint state

	for (int m = 0; m < M; m++)
		z[N][m] = 0.;

	for (int i = N; 0 < i; i--) {

		for (int m = 0; m < M; m++)
			z[i - 1][m] = z[i][m];
#ifdef __clang__
		const void* cost2 = cost;
		const void* sys2 = sys;
#endif
		// invert time -> neg. sign on RHS

		NESTED(void, matrix_fun, (float x[M], float t, const float in[M]))
		{
			(void)t;
#ifdef __clang__
			const float (*cost)[M] = cost2;
			const float (*sys)[M][M] = sys2;
#endif
			for (int l = 0; l < M; l++) {

				x[l] = cost[i - 1][l];

				for (int k = 0; k < M; k++)
					x[l] += sys[i - 1][k][l] * in[k];
			}
		};

		ode_interval(h, -1, tol, M, z[i - 1], -t[i], -t[i - 1], matrix_fun);
	}
}

static float adj_eval(int M, const float x[M], const float z[M], const float Adp[M][M])
{
	float ret = 0.;

	for (int l = 0; l < M; l++)
		for (int k = 0; k < M; k++)
			ret += z[l] * Adp[l][k] * x[k];

	return ret;
}

void ode_adjoint_sa_eval(int N, const float t[N + 1], int M,
		int P, float dj[P],
		const float x[N + 1][M], const float z[N + 1][M],
		const float Adp[P][M][M])
{
#ifdef __clang__
	const void* x2 = x;
	const void* z2 = z;
	const void* Adp2 = Adp;
#endif

	NESTED(void, eval, (float out[P], int i))
	{
#ifdef __clang__
		const float (*x)[M] = x2;
		const float (*z)[M] = z2;
		const float (*Adp)[M][M] = Adp2;
#endif
		for (int p = 0; p < P; p++)
			out[p] = adj_eval(M, x[i], z[i], Adp[p]);
	};

	quadrature_trapezoidal(N, t, P, dj, CLOSURE(quadrature_fun_t, eval));
}


void ode_adjoint_sa_eq_eval(int N, int M, int P, float dj[P],
		const float x[N + 1][M], const float z[N + 1][M],
		const float Adp[P][M][M])
{
#ifdef __clang__
	const void* x2 = x;
	const void* z2 = z;
	const void* Adp2 = Adp;
#endif

	NESTED(void, eval, (float out[P], int i))
	{
#ifdef __clang__
		const float (*x)[M] = x2;
		const float (*z)[M] = z2;
		const float (*Adp)[M][M] = Adp2;
#endif
		for (int p = 0; p < P; p++)
			out[p] = adj_eval(M, x[i], z[i], Adp[p]);
	};

	quadrature_simpson_ext(N, 1., P, dj, CLOSURE(quadrature_fun_t, eval));
}

