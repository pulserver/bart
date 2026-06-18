
#ifndef _QUADRATURE_H
#define _QUADRATURE_H 1

#include "misc/nested.h"

typedef CLOSURE_TYPE(void, (float out[/*P*/], int i)) quadrature_fun_t;

extern void quadrature_trapezoidal(int N, const float t[static N + 1], int P, float out[P],
		quadrature_fun_t sample);

extern void quadrature_simpson_ext(int N, float T, int P, float out[P],
		quadrature_fun_t sample);

#endif // _QUADRATURE_H

