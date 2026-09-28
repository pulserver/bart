#include "misc/dimtypes.h"


extern const struct operator_s* operator_clip_create(int N, const bart_dim_t dims[__VLA(N)], float clipnorm, float clipval);
extern const struct operator_p_s* operator_adadelta_update_create(int N, const bart_dim_t dims[__VLA(N)], float rho, float epsilon);
extern const struct operator_p_s* operator_adam_update_create(int N, const bart_dim_t dims[__VLA(N)], float beta1, float beta2, float epsilon, bart_dim_t reset_mod);
extern const struct operator_p_s* operator_sgd_update_create(int N, const bart_dim_t dims[__VLA(N)]);

