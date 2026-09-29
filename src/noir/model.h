
#include "misc/dimtypes.h"
#include <complex.h>

#include "misc/mri.h"

struct linop_s;
extern void noir_forw_coils(const struct linop_s* op, complex float* dst, const complex float* src);

struct noir_model_conf_s {

	float sobolev_os;
	bool sos;
	bool sms;
	bool rvc;
	bool noncart;
	float a;
	float b;
};

extern struct noir_model_conf_s noir_model_conf_defaults;

struct nlop_s;

struct noir_s {

	struct nlop_s* nlop;
	const struct linop_s* linop;
};

extern struct noir_s noir_create(const bart_dim_t dims[DIMS], const bart_dim_t coil_dims[DIMS], complex float* coils, const bart_dim_t pat_dims[DIMS], const complex float* psf, const struct noir_model_conf_s* conf);
