
#include "misc/dimtypes.h"
#include <complex.h>

#include "misc/mri.h"

#include "simu/signals.h"

#include "moba/meco.h"


struct linop_s;
struct nlop_s;
struct noir_model_conf_s;
struct moba_conf_s;

#ifndef MOBA_MOD
#define MOBA_MOD
struct mobamod {

	struct nlop_s* nlop;
	const struct linop_s* linop;
        const struct linop_s* linop_sobolev[24];
};
#endif

enum seq_type {
	IR_LL,
	MPL,
	TSE,
	MGRE,
	DIFF,
	IR,
	SIM,
	PHASE,
};

struct mobafit_model_config {

	enum seq_type seq;
	enum meco_model mgre_model;
	float B0;
	enum fat_spec fat_spec;
};


extern struct mobamod moba_create(const bart_dim_t dims[DIMS], const complex float* T1, const complex float* TE, const complex float* b1,
		const complex float* b0, const float* scale_fB0, enum meco_model meco_model, enum fat_spec fat_spec, float B0, const bart_dim_t psf_dims[DIMS], const complex float* psf, const bart_dim_t coil_dims[DIMS], complex float* coils, const struct noir_model_conf_s* conf, struct moba_conf_s* data,
		float scaling_M0, const complex float* fixed_maps);

const struct nlop_s* moba_get_nlop(struct mobafit_model_config* data, const bart_dim_t out_dims[DIMS], const bart_dim_t param_dims[DIMS], const bart_dim_t enc_dims[DIMS], complex float* enc);

extern const struct nlop_s* mobafit_phase_nlop(const bart_dim_t out_dims[DIMS], const complex float* sig, const bart_dim_t enc_dims[DIMS], complex float* enc);

extern void mobafit_phase_init(enum seq_type seq, const bart_dim_t coeff_dims[DIMS], complex float* init, const bart_dim_t sig_dims[DIMS], const complex float* sig, const bart_dim_t enc_dims[DIMS], complex float* enc);

