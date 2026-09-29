
#ifndef _ACTIVATION_H
#define _ACTIVATION_H

#include "misc/dimtypes.h"

enum ACTIVATION { ACT_LIN, ACT_RELU, ACT_LRELU, ACT_SOFTMAX, ACT_SIGMOID, ACT_CARDIOID, ACT_SIGLOG, ACT_IGAUSSIAN };

extern const struct nlop_s* append_activation(const struct nlop_s* network, int o, enum ACTIVATION activation, bart_flags_t bflags);
extern const struct nlop_s* append_activation_bias(const struct nlop_s* network, int o, enum ACTIVATION activation, bart_flags_t bias_flags);

extern const struct nlop_s* nlop_bias_create(int N, const bart_dim_t dims[__VLA(N)], const bart_dim_t bdims[__VLA(N)]);

extern const struct nlop_s* nlop_leaky_relu_create(int N, const bart_dim_t dims[__VLA(N)], float slope_parameter);
extern const struct nlop_s* nlop_relu_create(int N, const bart_dim_t dims[__VLA(N)]);
extern const struct nlop_s* nlop_relu_bias_create(int N, const bart_dim_t dims[__VLA(N)], const bart_dim_t bdims[__VLA(N)]);

extern const struct nlop_s* nlop_softmax_create2(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], const bart_stride_t istrs[__VLA(N)], bart_flags_t flag);
extern const struct nlop_s* nlop_softmax_create(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t batch_flag);
extern const struct nlop_s* nlop_softmax_bias_create(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t batch_flag, const bart_dim_t bias_dims[__VLA(N)]);

extern const struct nlop_s* nlop_sigmoid_create(int N, const bart_dim_t dims[__VLA(N)]);
extern const struct nlop_s* nlop_sigmoid_bias_create(int N, const bart_dim_t dims[__VLA(N)], const bart_dim_t bdims[__VLA(N)]);

extern const struct nlop_s* nlop_cardioid_create(int N, const bart_dim_t dims[__VLA(N)]);
extern const struct nlop_s* nlop_siglog_create(int N, const bart_dim_t dims[__VLA(N)], float c, float r);
extern const struct nlop_s* nlop_igaussian_create(int N, const bart_dim_t dims[__VLA(N)], float sigma);

#endif // _ACTIVATION_H

