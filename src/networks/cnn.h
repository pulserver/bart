
#ifndef _NN_CNN_H
#define _NN_CNN_H

#include "misc/dimtypes.h"
#include "misc/mri.h"
#include "misc/types.h"
#include "nn/layers.h"
#include "nn/activation.h"
#include "nn/nn.h"
#include "nn/nn_ops.h"
#include "nn/init.h"

struct network_s;

typedef nn_t (*network_create_t)(const struct network_s* config, int NO, const bart_dim_t odims[NO], int NI, const bart_dim_t idims[NI], enum NETWORK_STATUS status);

typedef struct network_s {

	TYPEID* TYPEID;

	network_create_t create;
	_Bool low_mem;

	enum norm norm;
	bart_flags_t norm_batch_flag;

	_Bool debug;
	_Bool residual;
	_Bool bart_to_channel_first;

	int loopdim;

	const char* prefix;

} network_t;

extern nn_t network_create(const struct network_s* config, int NO, const bart_dim_t odims[NO], int NI, const bart_dim_t idims[NI], enum NETWORK_STATUS status);

extern _Bool network_is_diagonal(const struct network_s* config);


struct network_resnet_s {

	network_t super;

	int N;

	bart_dim_t kdims[DIMS];
	bart_dim_t dilations[DIMS];

	bart_dim_t Nl; // number of blocks

	bart_dim_t Nf; // number of filters
	bart_dim_t Kx; // filter size
	bart_dim_t Ky; // filter size
	bart_dim_t Kz; // filter size
	bart_dim_t Ng; // number groups

	bart_flags_t conv_flag;
	bart_flags_t channel_flag;
	bart_flags_t group_flag;
	bart_flags_t batch_flag;

	_Bool batch_norm;
	_Bool batch_norm_lf;
	_Bool bias;

	enum ACTIVATION activation;
	enum ACTIVATION last_activation;

	_Bool zero_init;
};
extern struct network_resnet_s network_resnet_default;

struct network_varnet_s {

	network_t super;

	bart_dim_t Kx;
	bart_dim_t Ky;
	bart_dim_t Kz;

	bart_dim_t Nf;
	bart_dim_t Nw;

	float Imax;
	float Imin;

	float init_scale_mu;
};

extern struct network_varnet_s network_varnet_default;

extern struct network_s network_mnist_default;

#endif

