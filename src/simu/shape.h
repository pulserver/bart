
#ifndef _SHAPE
#define _SHAPE

#include "misc/dimtypes.h"
#include <complex.h>
#include "misc/mri.h"

struct tri_poly {

	bart_dim_t D;
	bart_dim_t cdims[DIMS];
	complex float* coeff; // k-space coeff
	bart_dim_t cpdims[DIMS];
	float* cpos;	// coordinates of k-space coeff
};

extern complex double xpolygon(int N, const double pg[N][2], const double p[3]);
extern complex double kpolygon(int N, const double pg[N][2], const double q[3]);

extern complex double xtripoly(const struct tri_poly* t, const bart_dim_t C, const double p[4]);
extern complex double ktripoly(const struct tri_poly* t, const bart_dim_t C, const double p[4]);

#endif
