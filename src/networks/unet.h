#include "misc/dimtypes.h"
#include "networks/cnn.h"


enum UNET_DOWNSAMPLING_METHOD {

	UNET_DS_STRIDED_CONV,
	UNET_DS_FFT,
	NNUNET_DS_STRIDED_CONV,
	//UNET_DS_MAX_POOL,
	//UNET_DS_AVERAGE_POOL
};

enum UNET_UPSAMPLING_METHOD {

	UNET_US_STRIDED_CONV,
	UNET_US_FFT,
	NNUNET_US_STRIDED_CONV
	//UNET_US_AVERAGE_POOL,
};

enum UNET_COMBINE_METHOD {

	UNET_COMBINE_ADD,
	UNET_COMBINE_STACK,
	UNET_COMBINE_ATTENTION_SIGMOID,
};

struct network_unet_s {

	network_t super;

	int N;
	bart_dim_t kdims[DIMS];
	bart_dim_t dilations[DIMS];

	bart_dim_t Nf; // number of filters (top-level)
	bart_dim_t Kx; // filter size
	bart_dim_t Ky; // filter size
	bart_dim_t Kz; // filter size
	bart_dim_t Ng; // number groups

	bart_flags_t conv_flag;
	bart_flags_t channel_flag;
	bart_flags_t group_flag;
	bart_flags_t batch_flag;

	bart_dim_t N_level;

	float channel_factor; //number channels on lower level
	float reduce_factor; //reduce resolution of lower level

	bart_dim_t max_channels; //maximum number of channels

	bart_dim_t Nl_highest_before; //number of layers in highest level
	bart_dim_t Nl_highest_after; //number of layers in highest level
	bart_dim_t Nl_before; //number of layers per level
	bart_dim_t Nl_after; //number of layers per level
	bart_dim_t Nl_lowest; //number of layers per level

	bool real_constraint;

	bool init_real;		//initialize weights with real numbers
	bool init_zeros_residual;	//initialize weights such that output of each level is initialized with zeros

	bool use_bn;
	bool use_instnorm;
	bool use_nnunet_last;
	bool use_bias;

	enum ACTIVATION activation;
	enum ACTIVATION activation_output;	//output of unet

	enum PADDING padding;

	enum UNET_DOWNSAMPLING_METHOD ds_method;
	enum UNET_UPSAMPLING_METHOD us_method;
	enum UNET_COMBINE_METHOD combine_method;

	bool residual;

	bool adjoint;
};

extern struct network_unet_s network_unet_default_reco;
extern struct network_unet_s network_unet_default_segm;
extern struct network_unet_s network_nnunet_default_segm;

extern nn_t network_unet_create(const struct network_s* config, int NO, const bart_dim_t odims[NO], int NI, const bart_dim_t idims[NI], enum NETWORK_STATUS status);
extern bool unet_is_diagonal(const struct network_s* config);

