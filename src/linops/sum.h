#include "misc/dimtypes.h"


struct linop_s;
extern const struct linop_s* linop_avg_create(int N, const bart_dim_t imgd_dims[N], bart_flags_t flags);
extern const struct linop_s* linop_sum_create(int N, const bart_dim_t imgd_dims[N], bart_flags_t flags);
extern const struct linop_s* linop_scaled_sum_create(int N, const bart_dim_t imgd_dims[N], bart_flags_t flags);
extern const struct linop_s* linop_repmat_create(int N, const bart_dim_t odims[N], bart_flags_t flags);