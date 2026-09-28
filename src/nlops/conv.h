#include "misc/dimtypes.h"


#ifndef _PADD_ENUMS
#define _PADD_ENUMS
enum PADDING { PAD_VALID, PAD_SAME, PAD_CYCLIC, PAD_SYMMETRIC, PAD_REFLECT, PAD_CAUSAL };
#endif

extern struct nlop_s* nlop_convcorr_geom_create(int N, bart_flags_t flags, const bart_dim_t odims[N], const bart_dim_t idims[N], const bart_dim_t kdims[N],
						enum PADDING conv_pad, _Bool conv, const bart_stride_t strides[N], const bart_dim_t dilations[N], char transpc);

