
#ifndef _QUADRATURE_H
#define _QUADRATURE_H 1

#include "misc/nested.h"

extern void quadrature_trapezoidal(int N, const float t[static N + 1], int P, float out[P],
		CLOSURE_TYPE(void, (float out[P], int i)) sample);

extern void quadrature_simpson_ext(int N, float T, int P, float out[P],
		CLOSURE_TYPE(void, (float out[P], int i)) sample);

#endif // _QUADRATURE_H

