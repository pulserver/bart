#ifndef _INITIALIZER_H
#define _INITIALIZER_H

#include "misc/dimtypes.h"

struct initializer_s;
typedef void (*initializer_f)(const struct initializer_s* conf, int N, const bart_dim_t dims[N], _Complex float* weights);

extern void initializer_apply(const struct initializer_s* conf, int N, const bart_dim_t dims[N], _Complex float* weights);
extern void initializer_free(const struct initializer_s* conf);
extern const struct initializer_s* initializer_clone(const struct initializer_s* x);

extern const struct initializer_s* init_reshape_create(int N, const bart_dim_t dims[N], const struct initializer_s* init);
extern const struct initializer_s* init_stack_create(int N, int stack_dim, const bart_dim_t dimsa[N], const struct initializer_s* inita, const bart_dim_t dimsb[N], const struct initializer_s* initb);
extern const struct initializer_s* init_dup_create(const struct initializer_s* inita, const struct initializer_s* initb);

extern bart_flags_t in_flag_conv(_Bool c1);
extern bart_flags_t out_flag_conv(_Bool c1);

extern bart_flags_t in_flag_conv_generic(int N, bart_flags_t conv_flag, bart_flags_t channel_flag, bart_flags_t group_flag);
extern bart_flags_t out_flag_conv_generic(int N, bart_flags_t conv_flag, bart_flags_t channel_flag, bart_flags_t group_flag);

extern const struct initializer_s* init_const_create(_Complex float val);
extern const struct initializer_s* init_xavier_create(bart_flags_t in_flags, bart_flags_t out_flags, _Bool real, _Bool uniform);
extern const struct initializer_s* init_kaiming_create(bart_flags_t in_flags, _Bool real, _Bool uniform, float leaky_val);
extern const struct initializer_s* init_array_create(int N, const bart_dim_t dims[N], const _Complex float* dat);

extern const struct initializer_s* init_std_normal_create(_Bool real, float scale, float mean);
extern const struct initializer_s* init_uniform_create(_Bool real, float scale, float mean);

extern const struct initializer_s* init_linspace_create(int dim, _Complex float min_val, _Complex float max_val, _Bool max_inc);

#endif
