
#ifndef _GRID_H
#define _GRID_H

#include "misc/dimtypes.h"
#include "misc/cppwrap.h"


struct grid_conf_s {

	float os;
	float width;
	bool periodic;
	double beta;

	float shift[3];
};

extern int kb_size;
extern double bessel_kb_beta; // = bessel_i0(beta);
extern const struct multiplace_array_s* kb_get_table(double beta);

extern void grid(const struct grid_conf_s* conf, const bart_dim_t ksp_dims[4], const bart_stride_t trj_strs[4], const _Complex float* traj, const bart_dim_t grid_dims[4], const bart_stride_t grid_strs[4], _Complex float* grid, const bart_stride_t ksp_strs[4], const _Complex float* src);

extern void gridH(const struct grid_conf_s* conf, const bart_dim_t ksp_dims[4], const bart_stride_t trj_strs[4], const _Complex float* traj, const bart_stride_t ksp_strs[4], _Complex float* dst, const bart_dim_t grid_dims[4], const bart_stride_t grid_strs[4], const _Complex float* grid);


extern void grid2(const struct grid_conf_s* conf, int D, const bart_dim_t trj_dims[__VLA(D)], const _Complex float* traj, const bart_dim_t grid_dims[__VLA(D)], _Complex float* grid, const bart_dim_t ksp_dims[__VLA(D)], const _Complex float* src);

extern void grid2H(const struct grid_conf_s* conf, int D, const bart_dim_t trj_dims[__VLA(D)], const _Complex float* traj, const bart_dim_t ksp_dims[__VLA(D)], _Complex float* dst, const bart_dim_t grid_dims[__VLA(D)], const _Complex float* grid);


extern void grid_pointH(int ch, int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t strs[__VLA(N)], const float pos[__VLA(N)], _Complex float val[__VLA(ch)], const _Complex float* src, bool periodic, float width, int kb_size, const float kb_table[__VLA(kb_size + 1)]);
extern void grid_point(int ch, int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t strs[__VLA(N)], const float pos[__VLA(N)], _Complex float* dst, const _Complex float val[__VLA(ch)], bool periodic, float width, int kb_size, const float kb_table[__VLA(kb_size + 1)]);

extern void kb_init(double beta);
extern double calc_beta(float os, float width);
extern void kb_precompute(double beta, int n, float table[__VLA(n + 1)]);

extern void rolloff_correction(float os, float width, float beta, const bart_dim_t dim[3], _Complex float* dst);
extern void apply_rolloff_correction2(float os, float width, float beta, int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], _Complex float* dst, const bart_stride_t istrs[__VLA(N)], const _Complex float* src);
extern void apply_rolloff_correction(float os, float width, float beta, int N, const bart_dim_t dimensions[__VLA(N)], _Complex float* dst, const _Complex float* src);


#include "misc/cppwrap.h"

#endif // _GRID_H

