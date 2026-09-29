
#include "misc/dimtypes.h"
#include <complex.h>

struct moba_conf;
struct moba_conf_s;
extern void moba_recon(const struct moba_conf* conf, struct moba_conf_s* data,
		const bart_dim_t dims[DIMS], const bart_dim_t imgs_dims[DIMS], complex float* img, const bart_dim_t coil_dims[DIMS], complex float* sens, const bart_dim_t pat_dims[DIMS], const complex float* pattern,
		const complex float* TI, const complex float* TE_IR_MGRE, const complex float* b1, const complex float* b0,
		const bart_dim_t data_dims[DIMS], const complex float* kspace_data, const complex float* init,
		const bart_dim_t mimgs_dims[DIMS], complex float* mimg);

