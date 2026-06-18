
#include "misc/nested.h"

#ifndef ODE_FUN_T
#define ODE_FUN_T
typedef CLOSURE_TYPE(void, (float* out, float t, const float* yn)) ode_fun_t;
#endif

typedef ode_fun_t ode_sys_t;
typedef CLOSURE_TYPE(void, (float dst[], float t)) ode_cost_t;

extern void rk4_step(float h, int N, float ynp[N], float tn, const float yn[N], ode_fun_t f);

extern void dormand_prince_step(float h, int N, float ynp[N], float tn, const float yn[N], ode_fun_t f);

extern float dormand_prince_step2(float h, int N, float ynp[N], float tn, const float yn[N], float tmp[6][N], ode_fun_t f);

extern float dormand_prince_scale(float tol, float err);

extern void ode_interval(float h, float tol, int N, float x[N], float st, float end, ode_fun_t f);
#define ode_interval(h, tol, N, x, st, end, f) \
	ode_interval(h, tol, N, x, st, end, CLOSURE(ode_fun_t, f))


extern void ode_interval2(float h, float tol,
	int N, const float t[N + 1], int M, float x[N + 1][M], ode_fun_t sys);

extern void ode_matrix_interval(float h, float tol, int N, float x[N], float st, float end, const float matrix[N][N]);

extern void ode_direct_sa(float h, float tol, int N, int P, float x[P + 1][N],
	float st, float end,
	ode_fun_t f, ode_fun_t pdy, ode_fun_t pdp);
#define ode_direct_sa(h, tol, N, P, x, st, end, f, pdy, pdp) \
	ode_direct_sa(h, tol, N, P, x, st, end, CLOSURE(ode_fun_t, f), CLOSURE(ode_fun_t, pdy), CLOSURE(ode_fun_t, pdp))



extern void ode_adjoint_sa(float h, float tol,
	int N, const float t[N + 1],
	int M, float x[N + 1][M], float z[N + 1][M],
	const float x0[M],
	ode_sys_t sys, ode_sys_t sysT, ode_cost_t cost);

void ode_adjoint_sa_noinit(float h, float tol,
	int N, const float t[N + 1],
	int M, float z[N + 1][M],
	ode_sys_t sysT, ode_cost_t cost);

extern void ode_matrix_adjoint_sa(float h, float tol,
	int N, const float t[N + 1],
	int M, float x[N + 1][M], float z[N + 1][M],
	const float x0[M], const float sys[N][M][M],
	const float cost[N][M]);
#if 0
extern void ode_matrix_adjoint_sa(int N, const float t[N + 1],
	int M, float z[N + 1][M], const float x0[M],
	void (*sys)(float sys[M][M], float t),
	void (*cost)(float c[N], float t));
#endif

extern void ode_adjoint_sa_eval(int N, const float t[N + 1], int M,
		int P, float dj[P],
		const float x[N + 1][M], const float z[N + 1][M],
		const float Adp[P][M][M]);

void ode_adjoint_sa_eq_eval(int N, int M, int P, float dj[P],
		const float x[N + 1][M], const float z[N + 1][M],
		const float Adp[P][M][M]);

typedef CLOSURE_TYPE(void, (int N, float (*matrix)[N][N], float t)) ode_cn_f;
typedef CLOSURE_TYPE(void, (int N, float (*matrix_ak)[N][N], float (*matrix_akp1)[N][N], float t)) ode_cn2_f;

void crank_nicolson(float h, int N, float x[N], float st, float end, ode_cn_f f);
#define crank_nicolson(h, N, x, st, end, f) \
	crank_nicolson(h, N, x, st, end, CLOSURE(ode_cn_f, f))

void crank_nicolson_matrix(float h, int N, float x[N], float st, float end, const float matrix[N][N]);

void crank_nicolson_adjoint(float h, int N, float x[N], float st, float end, ode_cn2_f f);
#define crank_nicolson_adjoint(h, N, x, st, end, f) \
	crank_nicolson_adjoint(h, N, x, st, end, CLOSURE(ode_cn2_f, f))

void crank_nicolson_matrix_adjoint(float h, int N, float x[N], float st, float end, 
		const float matrix_ak[N][N], const float matrix_akp1[N][N]);
