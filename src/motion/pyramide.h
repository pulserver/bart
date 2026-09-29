#include "misc/dimtypes.h"


extern void gaussian_pyramide(int levels, float factors[levels], float sigma[levels], int order,
			      int N, bart_flags_t flags, const bart_dim_t idims[N], const _Complex float* img,
			      bart_dim_t dims[levels][N], _Complex float* imgs[levels]);

extern void debug_gaussian_pyramide(int levels, float factors[levels], float sigma[levels],
				int N, bart_flags_t flags, const bart_dim_t idims[N]);

extern void upscale_displacement(int N, int d, bart_flags_t flags,
				 const bart_dim_t odims[N], _Complex float* out,
				 const bart_dim_t idims[N], const _Complex float* in);
