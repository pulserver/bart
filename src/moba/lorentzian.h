#include "misc/dimtypes.h"

struct nlop_s;

extern const struct nlop_s* nlop_lorentzian_multi_pool_create(int N, const bart_dim_t signal_dims[N],
		const bart_dim_t param_dims[N], const bart_dim_t omega_dims[N], const _Complex float* omega);
