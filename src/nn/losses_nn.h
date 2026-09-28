#ifndef _NN_LOSSES_H
#define _NN_LOSSES_H

#include "misc/dimtypes.h"
#include "nn/nn.h"

extern nn_t nn_loss_mse_append(nn_t network, int o, const char* oname, bart_flags_t mean_dims);
extern nn_t nn_loss_cce_append(nn_t network, int o, const char* oname, bart_flags_t scaling_flag);
extern nn_t nn_loss_dice_append(nn_t network, int o, const char* oname, bart_flags_t label_flag, bart_flags_t mean_flag, float weighting_exponent, bool square_denominator);

#endif
