
#include "misc/dimtypes.h"
#include <complex.h>

extern void overlapandadd(int N, const bart_dim_t dims[N], const bart_dim_t blk[N], complex float* dst, complex float* src1, const bart_dim_t dim2[N], complex float* src2);
extern void overlapandsave(int N, const bart_dim_t dims[N], const bart_dim_t blk[N], complex float* dst, complex float* src1, const bart_dim_t dim2[N], complex float* src2);

extern struct conv_plan* overlapandsave_plan(int N, const bart_dim_t dims[N], const bart_dim_t blk[N], const bart_dim_t dim2[N], complex float* src2);
extern void overlapandsave_exec(struct conv_plan* plan, int N, const bart_dim_t dims[N], const bart_dim_t blk[N], complex float* dst, complex float* src1, const bart_dim_t dim2[N]);

extern void overlapandsave2(int N, bart_flags_t flags, const bart_dim_t blk[N], const bart_dim_t odims[N], complex float* dst, const bart_dim_t dims1[N], const complex float* src1, const bart_dim_t dims2[N], const complex float* src2);
extern void overlapandsave2H(int N, bart_flags_t flags, const bart_dim_t blk[N], const bart_dim_t odims[N], complex float* dst, const bart_dim_t dims1[N], const complex float* src1, const bart_dim_t dims2[N], const complex float* src2);
extern void overlapandsave2NE(int N, bart_flags_t flags, const bart_dim_t blk[N], const bart_dim_t odims[N], complex float* dst, const bart_dim_t dims1[N], complex float* src1, const bart_dim_t dims2[N], complex float* src2, const bart_dim_t mdims[N], complex float* msk);

struct vec_ops;
extern void overlapandsave2NEB(int N, bart_flags_t flags, const bart_dim_t blk[N], const bart_dim_t odims[N], complex float* dst, const bart_dim_t dims1[N], const complex float* src1, const bart_dim_t dims2[N], const complex float* src2, const bart_dim_t mdims[N], const complex float* msk);

extern void overlapandsave2HB(int N, bart_flags_t flags, const bart_dim_t blk[N], const bart_dim_t odims[N], complex float* dst, const bart_dim_t dims1[N], const complex float* src1, const bart_dim_t dims2[N], const complex float* src2, const bart_dim_t mdims[N], const complex float* msk);


