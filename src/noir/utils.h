
#include "misc/dimtypes.h"
#include <complex.h>

struct linop_s;

extern void noir_calc_weights(double a, double b, const bart_dim_t dims[3], complex float* dst);
extern struct linop_s* linop_noir_weights_create(int N, const bart_dim_t img_dims[N], const bart_dim_t ksp_dims[N], const bart_dim_t ref_dims[N], bart_flags_t flags, double factor_fov, double a, double b, double c);

