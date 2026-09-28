
#ifndef _RECON_MECO_H
#define _RECON_MECO_H

#include "misc/dimtypes.h"
#include "moba/meco.h"


struct moba_conf;
struct moba_conf_s;
enum fat_spec;

void init_meco_maps(const bart_dim_t maps_dims[DIMS], complex float* maps, enum meco_model sel_model);

void meco_recon(const struct moba_conf* moba_conf, struct moba_conf_s* data,
		const long dims[DIMS],
		enum meco_model sel_model, enum fat_spec fat_spec,
		const float* scale_fB0, bool warmstart, bool out_origin_maps,
		const bart_dim_t maps_dims[DIMS], complex float* maps,
		const bart_dim_t sens_dims[DIMS], complex float* sens,
		const bart_dim_t init_dims[DIMS], const complex float* init,
		const complex float* TE,
		const bart_dim_t P_dims[DIMS], const complex float* Pin,
		const bart_dim_t Y_dims[DIMS], const complex float* Y);

#endif
