#include "misc/dimtypes.h"


const struct operator_p_s* operator_project_pos_real_create(bart_dim_t N, const bart_dim_t dims[N]);
const struct operator_p_s* operator_project_mean_free_create(bart_dim_t N, const bart_dim_t dims[N], bart_flags_t bflag);
const struct operator_p_s* operator_project_sphere_create(bart_dim_t N, const bart_dim_t dims[N], bart_flags_t bflag, bool real);
const struct operator_p_s* operator_project_mean_free_sphere_create(bart_dim_t N, const bart_dim_t dims[N], bart_flags_t bflag, bool real);
const struct operator_p_s* operator_project_min_real_create(bart_dim_t N, const bart_dim_t dims[N], float min);