
#include <complex.h>

#include "misc/mri.h"

struct linop_s;
extern void noir_forw_coils(const struct linop_s* op, complex float* dst, const complex float* src);

struct noir_model_conf_s {

	float sobolev_os;
	unsigned int fft_flags;
	unsigned int cnstcoil_flags;
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

extern struct noir_s noir_create(const long dims[DIMS], const long pat_dims[DIMS], const complex float* psf, const struct noir_model_conf_s* conf);
