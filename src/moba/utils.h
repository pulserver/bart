
#include "misc/mri.h"

extern const struct linop_s* moba_rvc_create(int N, const long in_dims[N], unsigned long rvc);
extern const struct linop_s* moba_precond_create(int N, const long in_dims[N], const struct linop_s* linops[in_dims[COEFF_DIM]], const float scaling[in_dims[COEFF_DIM]]);

extern const struct nlop_s* moba_attach_trafo_F(const struct nlop_s* nlop, const struct linop_s* linop);
extern const struct linop_s* moba_attach_trafo_get_linop(struct nlop_s* nlop);

