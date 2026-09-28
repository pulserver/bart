
#ifndef _ITER_THRESH_H
#define _ITER_THRESH_H

#include "misc/dimtypes.h"
#include "misc/cppwrap.h"


struct operator_p_s;
extern const struct operator_p_s* prox_thresh_create(int D, const bart_dim_t dim[__VLA(D)], const float lambda, const bart_flags_t flags);
extern const struct operator_p_s* prox_niht_thresh_create(int D, const bart_dim_t dim[D], int k, const bart_flags_t flags);
extern void thresh_free(const struct operator_p_s* data);

extern void set_thresh_lambda(const struct operator_p_s* o, const float lambda);
extern float get_thresh_lambda(const struct operator_p_s* o);


struct linop_s;

extern const struct operator_p_s* prox_unithresh_create(int D, const struct linop_s* unitary_op, const float lambda, const bart_flags_t flags);


#include "misc/cppwrap.h"

#endif


