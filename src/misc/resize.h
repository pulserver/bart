
#include "misc/dimtypes.h"
#include "misc/cppwrap.h"

extern void sinc_resize(int D, const bart_dim_t out_dims[__VLA(D)], _Complex float* out, const bart_dim_t in_dims[__VLA(D)], const _Complex float* in);
extern void sinc_zeropad(int D, const bart_dim_t out_dims[__VLA(D)], _Complex float* out, const bart_dim_t in_dims[__VLA(D)], const _Complex float* in);
extern void fft_zeropad(int D, bart_flags_t flags, const bart_dim_t out_dims[__VLA(D)], _Complex float* out, const bart_dim_t in_dims[__VLA(D)], const _Complex float* in);
extern void fft_zeropadH(int D, bart_flags_t flags, const bart_dim_t out_dims[__VLA(D)], _Complex float* out, const bart_dim_t in_dims[__VLA(D)], const _Complex float* in);

#include "misc/cppwrap.h"



