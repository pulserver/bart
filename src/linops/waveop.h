
#include "misc/dimtypes.h"
#include "misc/cppwrap.h"

#ifndef WTYPE
#define WTYPE
enum wtype { WAVELET_HAAR, WAVELET_DAU2, WAVELET_CDF44 };
#endif

extern struct linop_s* linop_wavelet_create(int N, bart_flags_t flags, const bart_dim_t dims[__VLA(N)], const bart_stride_t istr[__VLA(N)],
						enum wtype wtype, const bart_dim_t minsize[__VLA(N)], _Bool randshift);

#include "misc/cppwrap.h"

