#ifndef _NN_LAYERS_H
#define _NN_LAYERS_H

#include "misc/dimtypes.h"
#include "nn/layers.h"
#include "nn/nn.h"
#include "nn/init.h"

extern nn_t nn_append_convcorr_layer_generic(nn_t network, int o, const char* oname, const char* ker_name, bart_flags_t conv_flag, bart_flags_t channel_flag, bart_flags_t group_flag, int N, bart_dim_t const kernel_dims[__VLA2(N)], const bart_stride_t strides[__VLA2(N)], const bart_dim_t dilations[__VLA2(N)], bool conv, enum PADDING conv_pad, const struct initializer_s* init);
extern nn_t nn_append_transposed_convcorr_layer_generic(nn_t network, int o, const char* oname, const char* ker_name, bart_flags_t conv_flag, bart_flags_t channel_flag, bart_flags_t group_flag, int N, bart_dim_t const kernel_dims[__VLA(N)], const bart_stride_t strides[__VLA(N)], const bart_dim_t dilations[__VLA(N)], bool conv, enum PADDING conv_pad, bool adjoint, const struct initializer_s* init);

extern nn_t nn_append_maxpool_layer_generic(nn_t network, int o, const char* oname, int N, const bart_dim_t pool_size[__VLA(N)], enum PADDING conv_pad);

extern nn_t nn_append_convcorr_layer(nn_t network, int o, const char* oname, const char* ker_name, int filters, bart_dim_t const kernel_size[3], bool conv, enum PADDING conv_pad, bool channel_first, const bart_stride_t strides[3], const bart_dim_t dilations[3], const struct initializer_s* init);
extern nn_t nn_append_transposed_convcorr_layer(nn_t network, int o, const char* oname, const char* ker_name, int channels, bart_dim_t const kernel_size[3], bool conv, bool adjoint, enum PADDING conv_pad, bool channel_first, const bart_stride_t strides[3], const bart_dim_t dilations[3], const struct initializer_s* init);
extern nn_t nn_append_dense_layer(nn_t network, int o, const char* oname, const char* weights_name, int out_neurons, const struct initializer_s* init);
extern nn_t nn_append_batchnorm_layer(nn_t network, int o, const char* oname, const char* stat_name, bart_flags_t norm_flags, enum NETWORK_STATUS status, const struct initializer_s* init);
extern nn_t nn_append_normalize_layer(nn_t network, int o, bart_flags_t norm_flags, float epsilon);

extern nn_t nn_append_maxpool_layer(nn_t network, int o, const char* oname, const bart_dim_t pool_size[3], enum PADDING conv_pad, bool channel_first);
extern nn_t nn_append_blurpool_layer(nn_t network, int o, const char* oname, const bart_dim_t pool_size[3], enum PADDING conv_pad, bool channel_first);
extern nn_t nn_append_avgpool_layer(nn_t network, int o, const char* oname, const bart_dim_t pool_size[3], enum PADDING conv_pad, bool channel_first);
extern nn_t nn_append_upsampl_layer(nn_t network, int o, const char* oname, const bart_dim_t pool_size[3], bool channel_first);
extern nn_t nn_append_dropout_layer(nn_t network, int o, const char* oname, float p, enum NETWORK_STATUS status);
extern nn_t nn_append_flatten_layer(nn_t network, int o, const char* oname);
extern nn_t nn_append_padding_layer(nn_t network, int o, const char* oname, bart_dim_t N, bart_dim_t pad_for[__VLA(N)], bart_dim_t pad_after[__VLA(N)], enum PADDING pad_type);

#endif
