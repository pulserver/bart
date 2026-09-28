#include "misc/dimtypes.h"

struct nlop_s;
extern struct nlop_s* nlop_gmm_score_create(int N, const bart_dim_t score_dims[N], const bart_dim_t mean_dims[N], const _Complex float* mean, const bart_dim_t var_dims[N], const _Complex float* var, const bart_dim_t wgh_dims[N], const _Complex float* wgh);
