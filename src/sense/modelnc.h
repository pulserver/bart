
#include "misc/dimtypes.h"
#include <complex.h>

#include "misc/mri.h"

struct nufft_conf_s;
extern const struct linop_s* sense_nc_init(const bart_dim_t max_dims[DIMS], const bart_dim_t map_dims[DIMS], const complex float* maps,
		const bart_dim_t ksp_dims[DIMS],
		const bart_dim_t traj_dims[DIMS], const complex float* traj, const struct nufft_conf_s* conf,
		const bart_dim_t wgs_dims[DIMS], const complex float* weights,
		const bart_dim_t basis_dims[DIMS], const complex float* basis,
		const bart_dim_t fieldmap_dims[DIMS], const complex float* fieldmap,
		const bart_dim_t timemap_dims[DIMS], const complex float* timemap,
		const struct linop_s** fft_opp, bart_flags_t shared_img_dims);




