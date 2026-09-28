#ifndef _NN_OPS_H
#define _NN_OPS_H

#include "misc/dimtypes.h"
#include "nn/layers.h"

extern const struct nlop_s* nlop_rand_mask_create(int N, const bart_dim_t dims[__VLA(N)], float p);
extern const struct nlop_s* nlop_rand_split_create(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t shared_dims_flag, float p);
extern const struct nlop_s* nlop_rand_mask_fixed_create(int N, const bart_dim_t dims[__VLA(N)], float p, bart_flags_t bat_flags);
extern const struct nlop_s* nlop_rand_split_fixed_create(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t shared_dims_flag, bart_flags_t bat_dims_flag, float p, bart_flags_t fix_flags, const _Complex float* _fix_first, float leaky_val);
extern const struct nlop_s* nlop_maxpool_create(int N, const bart_dim_t dims[__VLA(N)], const bart_dim_t pool_size[__VLA(N)]);
extern const struct nlop_s* nlop_dropout_create(int N, const bart_dim_t dims[__VLA(N)], float p, bart_flags_t shared_dims_flag);
extern const struct nlop_s* nlop_noise_create(int N, const bart_dim_t dims[__VLA(N)], float sigma, bart_flags_t shared_dims_flag, bart_flags_t shared_sigma_flag);
extern const struct nlop_s* nlop_add_noise_create(int N, const bart_dim_t dims[__VLA(N)], float sigma, bart_flags_t shared_dims_flag, bart_flags_t shared_sigma_flag);

enum norm { NORM_NONE, NORM_MAX, NORM_L2 };
extern const struct nlop_s* nlop_norm_create(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t batch_flag, enum norm norm, _Bool stop_grad);
extern const struct nlop_s* nlop_norm_max_abs_create(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t batch_flag);
extern const struct nlop_s* nlop_norm_znorm_create(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t batch_flag);

#endif
