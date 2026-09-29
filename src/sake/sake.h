
#include "misc/dimtypes.h"
#include "misc/cppwrap.h"

extern void lrmc(float alpha, int iter, float lambda, int N, const bart_dim_t dims[__VLA(N)], _Complex float* out, const _Complex float* in);

#include "misc/cppwrap.h"
