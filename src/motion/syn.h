#include "misc/dimtypes.h"


extern void syn(int levels, float sigma[levels], float factors[levels], int nwarps[levels],
	int d, bart_flags_t flags, int N, const bart_dim_t dims[N],
	_Complex float* disp, _Complex float* idisp,
	const _Complex float* static_img, const _Complex float* moving_img);
	