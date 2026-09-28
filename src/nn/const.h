
#ifndef _NN_CONST_H
#define _NN_CONST_H

#include "misc/dimtypes.h"
#include "nn/nn.h"

extern nn_t nn_set_input_const_F(nn_t op, int i, const char* iname, int N, const bart_dim_t dims[N], _Bool copy, const _Complex float* in);
extern nn_t nn_set_input_const_F2(nn_t op, int i, const char* iname, int N, const bart_dim_t dims[N], const bart_stride_t strs[N], _Bool copy, const _Complex float* in);
extern nn_t nn_del_out_F(nn_t op, int o, const char* oname);
extern nn_t nn_del_out_bn_F(nn_t op);
extern nn_t nn_ignore_input_F(nn_t op, int i, const char* iname, int N, const bart_dim_t dims[N], _Bool copy, const _Complex float* in);

#endif
