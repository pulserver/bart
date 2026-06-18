
#include <complex.h>

#include "misc/nested.h"

#ifndef ODE_FUN_T
#define ODE_FUN_T
typedef CLOSURE_TYPE(void, (float* out, float t, const float* yn)) ode_fun_t;
#endif

extern void mat_exp(int N, float t, float out[N][N], const float in[N][N]);
extern void zmat_exp(int N, float t, complex float out[N][N], const complex float in[N][N]);
extern void mat_to_exp(int N, float st, float en, float out[N][N], float tol, ode_fun_t f);

