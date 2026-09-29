
#include "misc/dimtypes.h"
#include "misc/mri.h"

struct nlop_s;
struct nn_cunet_conf_s;

extern const struct nlop_s* prior_cunet(const char* cunet_weights, struct nn_cunet_conf_s* cunet_conf,
				bool real_valued, const bart_dim_t msk_dims[DIMS], complex float* mask,
				bart_dim_t img_dims[DIMS]);

extern const struct nlop_s* prior_graph(const char* graph, bool real_valued, bool gpu,
		const bart_dim_t msk_dims[DIMS], complex float* mask, bart_dim_t img_dims[DIMS]);

extern const struct nlop_s* prior_gmm(const bart_dim_t means_dims[DIMS], const complex float* means,
				const bart_dim_t weights_dims0[DIMS], const complex float *weights0,
				const bart_dim_t vars_dims0[DIMS], const complex float *vars0,
				bart_dim_t img_dims[DIMS], float *min_var);


