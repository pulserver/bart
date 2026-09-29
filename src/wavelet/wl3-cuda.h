

#include "misc/dimtypes.h"
#include "misc/cppwrap.h"

extern void wl3_cuda_down3(const bart_dim_t dims[3], const bart_stride_t out_str[3], _Complex float* out, const bart_stride_t in_str[3], const _Complex float* in, unsigned int flen, const float filter[__VLA(flen)]);

extern void wl3_cuda_up3(const bart_dim_t dims[3], const bart_stride_t out_str[3], _Complex float* out, const bart_stride_t in_str[3],  const _Complex float* in, unsigned int flen, const float filter[__VLA(flen)]);

#include "misc/cppwrap.h"

