#include "misc/dimtypes.h"


extern void md_positions(int N, int d, bart_flags_t flags, const bart_dim_t sdims[__VLA(N)], const bart_dim_t pdims[__VLA(N)], _Complex float* pos);

extern void md_interpolate2(int d, bart_flags_t flags, int ord, int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t istrs[__VLA(N)], _Complex float* intp, const bart_stride_t cstrs[__VLA(N)], const _Complex float* coor, const bart_dim_t gdims[__VLA(N)], const bart_stride_t gstrs[__VLA(N)], const _Complex float* grid);
extern void md_interpolateH2(int d, bart_flags_t flags, int ord, int N, const bart_dim_t gdims[__VLA(N)], const bart_stride_t gstrs[__VLA(N)], _Complex float* grid, const bart_dim_t dims[__VLA(N)], const bart_stride_t istrs[__VLA(N)], const _Complex float* intp, const bart_stride_t cstrs[__VLA(N)], const _Complex float* coor);
extern void md_interpolate_adj_coor2(int d, bart_flags_t flags, int ord, int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t cstrs[__VLA(N)], const _Complex float* coor, _Complex float* dcoor, const bart_stride_t istrs[__VLA(N)], const _Complex float* dintp, const bart_dim_t gdims[__VLA(N)], const bart_stride_t gstrs[__VLA(N)], const complex float* grid);
extern void md_interpolate_der_coor2(int d, bart_flags_t flags, int ord, int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t istrs[__VLA(N)], _Complex float* dintp, const bart_stride_t cstrs[__VLA(N)], const _Complex float* coor, const _Complex float* dcoor, const bart_dim_t gdims[__VLA(N)], const bart_stride_t gstrs[__VLA(N)], const _Complex float* grid);

extern void md_interpolate(int d, bart_flags_t flags, int ord, int N, const bart_dim_t idims[__VLA(N)], _Complex float* intp, const bart_dim_t cdims[__VLA(N)], const _Complex float* coor, const bart_dim_t gdims[__VLA(N)], const _Complex float* grid);
extern void md_interpolateH(int d, bart_flags_t flags, int ord, int N, const bart_dim_t gdims[__VLA(N)], _Complex float* grid, const bart_dim_t idims[__VLA(N)], const _Complex float* intp, const bart_dim_t cdims[__VLA(N)], const _Complex float* coor);
extern void md_interpolate_adj_coor(int d, bart_flags_t flags, int ord, int N, const bart_dim_t cdims[__VLA(N)], const _Complex float* coor, _Complex float* dcoor, const bart_dim_t idims[__VLA(N)], const _Complex float* dintp, const bart_dim_t gdims[__VLA(N)], const _Complex float* grid);
extern void md_interpolate_der_coor(int d, bart_flags_t flags, int ord, int N, const bart_dim_t idims[__VLA(N)], _Complex float* dintp, const bart_dim_t cdims[__VLA(N)], const _Complex float* coor, const _Complex float* dcoor, const bart_dim_t gdims[__VLA(N)], const _Complex float* grid);

extern void md_resample(bart_flags_t flags, int ord, int N, const bart_dim_t odims[__VLA(N)], _Complex float* dst, const bart_dim_t idims[__VLA(N)], const _Complex float* src);

struct linop_s;
extern const struct linop_s* linop_interpolate_create(int d, bart_flags_t flags, int ord, int N, const bart_dim_t idims[__VLA(N)], const bart_dim_t cdims[__VLA(N)], const _Complex float* coor, const bart_dim_t gdims[__VLA(N)]);

struct nlop_s;
extern const struct nlop_s* nlop_interpolate_create(int d, bart_flags_t flags, int ord, bool shifted_grad, int N, const bart_dim_t idims[__VLA(N)], const bart_dim_t cdims[__VLA(N)], const bart_dim_t gdims[__VLA(N)]);

