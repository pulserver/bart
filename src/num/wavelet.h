
#include "num/multind.h"

#include "misc/cppwrap.h"

extern void md_wavtrafo2(int D, const bart_dim_t dims[__VLA(D)], bart_flags_t flags, const bart_stride_t strs[__VLA(D)], void* ptr, md_trafo_fun_t fun, _Bool inv, _Bool nosort);
extern void md_wavtrafo(int D, const bart_dim_t dims[__VLA(D)], bart_flags_t flags, void* ptr, size_t size, md_trafo_fun_t fun, _Bool inv, _Bool nosort);
extern void md_wavtrafoz2(int D, const bart_dim_t dims[__VLA(D)], bart_flags_t flags, const bart_stride_t strs[__VLA(D)], _Complex float* x, md_trafo_fun_t fun, _Bool inv, _Bool nosort);
extern void md_wavtrafoz(int D, const bart_dim_t dims[__VLA(D)], bart_flags_t flags, _Complex float* ptr, md_trafo_fun_t fun, _Bool inv, _Bool nosort);
extern void md_cdf97z(int D, const bart_dim_t dims[__VLA(D)], bart_flags_t flags, _Complex float* data);
extern void md_icdf97z(int D, const bart_dim_t dims[__VLA(D)], bart_flags_t flags, _Complex float* data);
extern void md_cdf97z2(int D, const bart_dim_t dims[__VLA(D)], bart_flags_t flags, const bart_stride_t strs[__VLA(D)], _Complex float* data);
extern void md_icdf97z2(int D, const bart_dim_t dims[__VLA(D)], bart_flags_t flags, const bart_stride_t strs[__VLA(D)], _Complex float* data);
extern void md_resortz(int D, const bart_dim_t dims[__VLA(D)], bart_flags_t flags, _Complex float* data);
extern void md_iresortz(int D, const bart_dim_t dims[__VLA(D)], bart_flags_t flags, _Complex float* data);


#include "misc/cppwrap.h"


