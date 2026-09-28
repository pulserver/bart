#ifndef _NN_ACTIVATIONS_H
#define _NN_ACTIVATIONS_H

#include "misc/dimtypes.h"
#include "nn/nn.h"
#include "nn/activation.h"

extern nn_t nn_append_activation(nn_t network, int o, const char* oname, enum ACTIVATION activation, bart_flags_t bflags);
extern nn_t nn_append_activation_bias(nn_t network, int o, const char* oname, const char* bname, enum ACTIVATION activation, bart_flags_t bflag);

#endif
