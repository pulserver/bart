
#include "misc/cppwrap.h"

struct operator_s;
extern const struct operator_s* fft_create2(int D, const long dimensions[__VLA(D)], unsigned long flags, const long ostrides[__VLA(D)], const _Complex float* dst, const long istrides[__VLA(D)], const _Complex float* src, bool backwards);

extern void fft_cache_free(void);

extern bool use_fftw_wisdom;
extern void fft_set_num_threads(int n);

#include "misc/cppwrap.h"
