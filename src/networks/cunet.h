
#include "misc/dimtypes.h"
#include "linops/someops.h"
#include "nn/activation.h"

struct nn_cunet_conf_s {

	int levels;

	enum PADDING padding;
	enum ACTIVATION activation;

	bool conditional;
	bart_dim_t num_filters;
	bart_dim_t cunits;

	bart_dim_t ksizes[3];
	bart_dim_t dilations[3];
	bart_stride_t strides[3];
};

extern struct nn_cunet_conf_s cunet_defaults;

extern const struct nn_s* cunet_create(struct nn_cunet_conf_s* conf, int N, const bart_dim_t dims[N]);
extern const struct nn_s* cunet_bart_create(struct nn_cunet_conf_s* conf, int N, const bart_dim_t bdims[N]);
