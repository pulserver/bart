#include "misc/dimtypes.h"

#ifndef WTYPE
#define WTYPE
enum wtype { WAVELET_HAAR, WAVELET_DAU2, WAVELET_CDF44 };
#endif

extern const struct operator_p_s* prox_wavelet_thresh_create(int N, const bart_dim_t dims[N], bart_flags_t flags, bart_flags_t jflags,
				enum wtype wtype, const bart_dim_t minsize[N], float lambda, bool randshift);


extern void wavthresh_rand_state_set(const struct operator_p_s* op, unsigned long long x);


