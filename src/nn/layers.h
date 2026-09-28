#ifndef _LAYERS_H
#define _LAYERS_H

#include "misc/dimtypes.h"
#include "nlops/conv.h"
#include "misc/cppwrap.h"

enum NETWORK_STATUS {STAT_TRAIN, STAT_TEST};

extern const struct nlop_s* append_convcorr_layer_generic(const struct nlop_s* network, int o, bart_flags_t conv_flag, bart_flags_t channel_flag, bart_flags_t group_flag, int N, bart_dim_t const kernel_dims[__VLA(N)], const bart_stride_t strides[__VLA2(N)], const bart_dim_t dilations[__VLA2(N)], _Bool conv, enum PADDING conv_pad);
extern const struct nlop_s* append_transposed_convcorr_layer_generic(const struct nlop_s* network, int o, bart_flags_t conv_flag, bart_flags_t channel_flag, bart_flags_t group_flag, int N, bart_dim_t const kernel_dims[__VLA(N)], const bart_stride_t strides[__VLA2(N)], const bart_dim_t dilations[__VLA2(N)], _Bool conv, enum PADDING conv_pad, _Bool adjoint);

extern const struct nlop_s* append_maxpool_layer_generic(const struct nlop_s* network, int o, int N, const bart_dim_t pool_size[__VLA(N)], enum PADDING conv_pad);

extern const struct nlop_s* append_dense_layer(const struct nlop_s* network, int o, int out_neurons);

extern const struct nlop_s* append_convcorr_layer(const struct nlop_s* network, int o, int filters, const bart_dim_t kernel_size[3], _Bool conv, enum PADDING conv_pad, _Bool channel_first, const bart_stride_t strides[3], const bart_dim_t dilations[3]);
extern const struct nlop_s* append_transposed_convcorr_layer(const struct nlop_s* network, int o, int channels, bart_dim_t const kernel_size[3], _Bool conv, _Bool adjoint, enum PADDING conv_pad, _Bool channel_first, const bart_stride_t strides[3], const bart_dim_t dilations[3]);
extern const struct nlop_s* append_maxpool_layer(const struct nlop_s* network, int o, const bart_dim_t pool_size[3], enum PADDING conv_pad, _Bool channel_first);

extern const struct nlop_s* append_padding_layer(const struct nlop_s* network, int o, bart_dim_t N, bart_dim_t pad_for[__VLA(N)], bart_dim_t pad_after[__VLA(N)], enum PADDING pad_type);

extern const struct nlop_s* append_dropout_layer(const struct nlop_s* network, int o, float p, enum NETWORK_STATUS status);
extern const struct nlop_s* append_flatten_layer(const struct nlop_s* network, int o);

extern const struct nlop_s* append_batchnorm_layer(const struct nlop_s* network, int o, bart_flags_t norm_flags, enum NETWORK_STATUS status);
extern const struct nlop_s* append_normalize_layer(const struct nlop_s* network, int o, bart_flags_t norm_flags, float epsilon);

#include "misc/cppwrap.h"

#endif
