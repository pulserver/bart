#include "misc/dimtypes.h"


struct nlop_s;
struct noir_model_conf_s;

enum T1_model { IRLL = 10, };


extern struct nlop_s* nlop_T1_create(int N, const bart_dim_t out_dims[N], const bart_dim_t in_dims[N],
                const bart_dim_t TI_dims[N], const _Complex float* TI, float scaling_M0);

extern const struct nlop_s* nlop_ir_create(int N, const bart_dim_t dims[N], const _Complex float* enc);

