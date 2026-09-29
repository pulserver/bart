#include "misc/dimtypes.h"



struct nlop_s;
struct moba_conf_s;

extern float read_relax(float tr, float angle);

extern struct nlop_s* nlop_T1_phy_create(int N, const bart_dim_t out_dims[N], const bart_dim_t in_dims[N],
                const bart_dim_t TI_dims[N], const complex float* TI,  const struct moba_conf_s* config);

