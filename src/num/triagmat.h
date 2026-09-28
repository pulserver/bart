
#include "misc/dimtypes.h"
#include "misc/cppwrap.h"

extern bart_dim_t upper_triag_idx(bart_dim_t i, bart_dim_t j);

extern _Complex float* hermite_to_uppertriag(int dim1, int dim2, int dimt, int N, bart_dim_t out_dims[__VLA(N)], const bart_dim_t* dims, const _Complex float* src);
extern _Complex float* uppertriag_to_hermite(int dim1, int dim2, int dimt, int N, bart_dim_t out_dims[__VLA(N)], const bart_dim_t* dims, const _Complex float* src);

extern float* symmetric_to_uppertriag(int dim1, int dim2, int dimt, int N, bart_dim_t out_dims[__VLA(N)], const bart_dim_t* dims, const float* src);
extern float* uppertriag_to_symmetric(int dim1, int dim2, int dimt, int N, bart_dim_t out_dims[__VLA(N)], const bart_dim_t* dims, const float* src);

extern void md_tenmul_upper_triag2(int dim1, int dim2, int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], float* dst, const bart_stride_t istrs[__VLA(N)], const float* src, const bart_dim_t mdims[__VLA(N)], const bart_stride_t mstrs[__VLA(N)], const float* mat);
extern void md_tenmul_upper_triag(int dim1, int dim2, int N, const bart_dim_t odims[__VLA(N)], float* dst, const bart_dim_t idims[__VLA(N)], const float* src, const bart_dim_t mdims[__VLA(N)], const float* mat);

extern void md_ztenmul_upper_triag2(int dim1, int dim2, int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], _Complex float* dst, const bart_stride_t istrs[__VLA(N)], const _Complex float* src, const bart_dim_t mdims[__VLA(N)], const bart_stride_t mstrs[__VLA(N)], const _Complex float* mat);
extern void md_ztenmul_upper_triag(int dim1, int dim2, int N, const bart_dim_t odims[__VLA(N)], _Complex float* dst, const bart_dim_t idims[__VLA(N)], const _Complex float* src, const bart_dim_t mdims[__VLA(N)], const _Complex float* mat);

#include "misc/cppwrap.h"
