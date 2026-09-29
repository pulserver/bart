#include "misc/dimtypes.h"


extern void activate_strided_vecops(void);
extern void deactivate_strided_vecops(void);

extern bool simple_zfmac(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], _Complex float* out, const bart_stride_t istrs1[__VLA(N)], const _Complex float* in1, const bart_stride_t istrs2[__VLA(N)], const _Complex float* in2);
extern bool simple_zfmacc(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], _Complex float* out, const bart_stride_t istrs1[__VLA(N)], const _Complex float* in1, const bart_stride_t istrs2[__VLA(N)], const _Complex float* in2);
extern bool simple_fmac(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], float* out, const bart_stride_t istrs1[__VLA(N)], const float* in1, const bart_stride_t istrs2[__VLA(N)], const float* in2);

extern bool simple_zmul(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], _Complex float* out, const bart_stride_t istrs1[__VLA(N)], const _Complex float* in1, const bart_stride_t istrs2[__VLA(N)], const _Complex float* in2);
extern bool simple_zmulc(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], _Complex float* out, const bart_stride_t istrs1[__VLA(N)], const _Complex float* in1, const bart_stride_t istrs2[__VLA(N)], const _Complex float* in2);
extern bool simple_mul(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], float* out, const bart_stride_t istrs1[__VLA(N)], const float* in1, const bart_stride_t istrs2[__VLA(N)], const float* in2);

extern bool simple_zadd(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], _Complex float* out, const bart_stride_t istrs1[__VLA(N)], const _Complex float* in1, const bart_stride_t istrs2[__VLA(N)], const _Complex float* in2);
extern bool simple_add(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], float* out, const bart_stride_t istrs1[__VLA(N)], const float* in1, const bart_stride_t istrs2[__VLA(N)], const float* in2);

extern bool simple_zmax(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], _Complex float* out, const bart_stride_t istrs1[__VLA(N)], const _Complex float* in1, const bart_stride_t istrs2[__VLA(N)], const _Complex float* in2);

extern bool simple_fmacD(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], double* out, const bart_stride_t istrs1[__VLA(N)], const float* in1, const bart_stride_t istrs2[__VLA(N)], const float* in2);
extern bool simple_zfmaccD(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], _Complex double* out, const bart_stride_t istrs1[__VLA(N)], const _Complex float* in1, const bart_stride_t istrs2[__VLA(N)], const _Complex float* in2);

