 
#ifndef _ITER_PROX_H
#define _ITER_PROX_H

#include "misc/dimtypes.h"
#include "misc/cppwrap.h"

struct operator_p_s;
extern const struct operator_p_s* prox_leastsquares_create(int N, const bart_dim_t dims[__VLA(N)], float lambda, const _Complex float* y);
extern const struct operator_p_s* prox_weighted_leastsquares_create(int N, const bart_dim_t dims[__VLA(N)], float lambda, const _Complex float* y, bart_flags_t flags, const _Complex float* W);
extern const struct operator_p_s* prox_l2norm_create(int N, const bart_dim_t dims[__VLA(N)], float lambda);
extern const struct operator_p_s* prox_l2ball_create(int N, const bart_dim_t dims[__VLA(N)], float eps, const _Complex float* center);
extern const struct operator_p_s* prox_l2ball2_create(int N, bart_flags_t flags, const bart_dim_t dims[__VLA(N)], float eps, const _Complex float* center);
extern const struct operator_p_s* prox_zero_create(int N, const bart_dim_t dims[__VLA(N)]);
extern const struct operator_p_s* prox_lesseq_create(int N, const bart_dim_t dims[__VLA(N)], const _Complex float* b);
extern const struct operator_p_s* prox_greq_create(int N, const bart_dim_t dims[__VLA(N)], const _Complex float* b);
extern const struct operator_p_s* prox_rvc_create(int N, const bart_dim_t dims[__VLA(N)]);
extern const struct operator_p_s* prox_nonneg_create(int N, const bart_dim_t dims[__VLA(N)]);
extern const struct operator_p_s* prox_zsmax_create(int N, const bart_dim_t dims[__VLA(N)], float a);

#include "misc/cppwrap.h"
#endif
