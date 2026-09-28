 
#ifndef _ITER_PROX2_H
#define _ITER_PROX2_H

#include "misc/dimtypes.h"
#include "misc/cppwrap.h"

struct operator_p_s;
struct linop_s;
struct nlop_s;

extern const struct operator_p_s* prox_normaleq_create(const struct linop_s* op, const _Complex float* y);
extern const struct operator_p_s* prox_lineq_create(const struct linop_s* op, const _Complex float* y);
extern const struct operator_p_s* prox_nlgrad_create(const struct nlop_s* op, int steps, float stepsize, float lambda, bool grad_nlop);
extern const struct operator_p_s* prox_scale_arg_create_F(const struct operator_p_s* op, float scale);
extern const struct operator_p_s* prox_select_maps_F(int N, const long dims[__VLA(N)], unsigned long flags, const struct operator_p_s* prox);

enum norm { NORM_MAX, NORM_L2 };
extern const struct operator_p_s* op_p_auto_normalize(const struct operator_p_s* op, bart_flags_t flags, enum norm norm);
extern const struct operator_p_s* op_p_conjugate(const struct operator_p_s* op, const struct linop_s* lop);

#include "misc/cppwrap.h"

#endif

