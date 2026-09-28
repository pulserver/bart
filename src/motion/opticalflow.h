#include "misc/dimtypes.h"

extern void optical_flow(_Bool l1_reg, bart_flags_t reg_flags, float lambda, float maxnorm, _Bool l1_dc, int d, bart_flags_t flags, int N, const bart_dim_t dims[N], const _Complex float* img_static, const _Complex float* _img_moved, _Complex float* u);

extern void optical_flow_multiscale(_Bool l1_reg, bart_flags_t reg_flags, float lambda, float maxnorm, _Bool l1_dc, 
				    int levels, float sigma[levels], float factors[levels], int nwarps[levels],
				    int d, bart_flags_t flags, int N, const bart_dim_t _dims[N], const _Complex float* img_static, const _Complex float* img_moved, _Complex float* u);
