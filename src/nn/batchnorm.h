#ifndef _BATCHNORM_H
#define _BATCHNORM_H

#include "misc/dimtypes.h"
#include "nn/layers.h"
enum NETWORK_STATUS;
extern const struct nlop_s* nlop_stats_create(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t flags);
extern const struct nlop_s* nlop_normalize_stats_create(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t flags, float epsilon);
extern const struct nlop_s* nlop_batchnorm_create(int N, const bart_dim_t dims[N], bart_flags_t flags, float epsilon, enum NETWORK_STATUS status);
extern const struct nlop_s* nlop_normalize_create(int N, const bart_dim_t dims[N], bart_flags_t flags, float epsilon);

extern const struct nlop_s* nlop_norm_avg_create(int N, const bart_dim_t dims[N], bart_flags_t flags);
extern const struct nlop_s* nlop_norm_std_create(int N, const bart_dim_t dims[N], bart_flags_t flags, float epsilon);

#endif

