
#include "misc/dimtypes.h"
#include <complex.h>

#include "misc/mri.h"

struct nufft_conf_s;

struct pics_config {

	struct nufft_conf_s* nuconf;

	bool gpu;
	bool gpu_gridding;
	bool real_value_constraint;
	bool time_encoded_asl;

	bart_flags_t shared_img_flags;
	bart_flags_t motion_flags;
};

struct linop_s;

extern const struct linop_s* pics_model(const struct pics_config* conf,
				const bart_dim_t img_dims[DIMS], const bart_dim_t ksp_dims[DIMS],
				const bart_dim_t traj_dims[DIMS], const complex float* traj,
				const bart_dim_t basis_dims[DIMS], const complex float* basis,
				const bart_dim_t map_dims[DIMS], const complex float* maps,
				const bart_dim_t pat_dims[DIMS], const complex float* pattern,
				const bart_dim_t motion_dims[DIMS], complex float* motion,
				const bart_dim_t fieldmap_dims[DIMS], complex float* fieldmap,
				const bart_dim_t timemap_dims[DIMS], complex float* timemap,
				const struct linop_s** nufft_op);

