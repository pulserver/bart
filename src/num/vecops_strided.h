#include "misc/dimtypes.h"


extern void activate_strided_vecops(void);
extern void deactivate_strided_vecops(void);

#ifndef NO_BLAS
extern _Bool simple_zfmac(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], _Complex float* out, const bart_stride_t istrs1[__VLA(N)], const _Complex float* in1, const bart_stride_t istrs2[__VLA(N)], const _Complex float* in2);
extern _Bool simple_zfmacc(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], _Complex float* out, const bart_stride_t istrs1[__VLA(N)], const _Complex float* in1, const bart_stride_t istrs2[__VLA(N)], const _Complex float* in2);
extern _Bool simple_fmac(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], float* out, const bart_stride_t istrs1[__VLA(N)], const float* in1, const bart_stride_t istrs2[__VLA(N)], const float* in2);

extern _Bool simple_zmul(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], _Complex float* out, const bart_stride_t istrs1[__VLA(N)], const _Complex float* in1, const bart_stride_t istrs2[__VLA(N)], const _Complex float* in2);
extern _Bool simple_zmulc(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], _Complex float* out, const bart_stride_t istrs1[__VLA(N)], const _Complex float* in1, const bart_stride_t istrs2[__VLA(N)], const _Complex float* in2);
extern _Bool simple_mul(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], float* out, const bart_stride_t istrs1[__VLA(N)], const float* in1, const bart_stride_t istrs2[__VLA(N)], const float* in2);

extern _Bool simple_zadd(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], _Complex float* out, const bart_stride_t istrs1[__VLA(N)], const _Complex float* in1, const bart_stride_t istrs2[__VLA(N)], const _Complex float* in2);
extern _Bool simple_add(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], float* out, const bart_stride_t istrs1[__VLA(N)], const float* in1, const bart_stride_t istrs2[__VLA(N)], const float* in2);

extern _Bool simple_zmax(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], _Complex float* out, const bart_stride_t istrs1[__VLA(N)], const _Complex float* in1, const bart_stride_t istrs2[__VLA(N)], const _Complex float* in2);

extern _Bool simple_fmacD(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], double* out, const bart_stride_t istrs1[__VLA(N)], const float* in1, const bart_stride_t istrs2[__VLA(N)], const float* in2);
extern _Bool simple_zfmaccD(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], _Complex double* out, const bart_stride_t istrs1[__VLA(N)], const _Complex float* in1, const bart_stride_t istrs2[__VLA(N)], const _Complex float* in2);

#else
#define simple_fmac(...) false
#define simple_zfmac(...) false
#define simple_zfmacc(...) false
#define simple_fmacc(...) false
#define simple_fmul(...) false
#define simple_fmulc(...) false
#define simple_mul(...) false
#define simple_zmul(...) false
#define simple_zmulc(...) false
#define simple_zadd(...) false
#define simple_add(...) false
#define simple_zmax(...) false
#define simple_fmacD(...) false
#define simple_zfmaccD(...) false
#endif


