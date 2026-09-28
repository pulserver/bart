#include "misc/dimtypes.h"


struct nlop_s;
struct noir_model_conf_s;
extern struct nlop_s* nlop_T2_create(int N, const bart_dim_t map_dims[N], const bart_dim_t out_dims[N], const bart_dim_t in_dims[N],
                const bart_dim_t TI_dims[N], const complex float* TI);

