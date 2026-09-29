
#ifndef _PHANTOM_H
#define _PHANTOM_H

#include "misc/dimtypes.h"
#include "stl/misc.h"
#include "simu/sens.h"

enum phantom_type { SHEPPLOGAN, CIRC, TIME, SENS, GEOM, STAR, BART, BRAIN, TUBES, RAND_TUBES, NIST, SONAR, GEOMFILE, ELLIPSOID0, STL };

struct phantom_opts {

	int D;
	bool kspace;
	enum phantom_type ptype;
	bart_dim_t Nc; // number of coefficients for COEFF_DIM
	void* data;
	co_dstr_t dstr;
	sim_fun_t fun;
};

extern struct phantom_opts phantom_opts_defaults;

extern void phantom_stl_init(struct phantom_opts* popts, int D, bart_dim_t dims[D], double* model);

extern void calc_ellipsoid(int D, bart_dim_t dims[D], _Complex float* out, bool d3, bool kspace, bart_dim_t tdims[D], bart_stride_t tstrs[D], _Complex float* traj, float ax[3], bart_dim_t center[3], float rot, struct coil_opts* copts);

extern void calc_sens(const bart_dim_t dims[DIMS], complex float* sens, struct coil_opts* copts);

extern void calc_geo_phantom(const bart_dim_t dims[DIMS], complex float* out, bool ksp, int phtype, const bart_stride_t tstrs[DIMS], const _Complex float* traj, struct coil_opts* copts);

extern void calc_phantom_noncart(const bart_dim_t dims[3], complex float* out, const complex float* traj, struct coil_opts* copts);
extern void calc_geo_phantom_noncart(const bart_dim_t dims[3], complex float* out, const complex float* traj, int phtype, struct coil_opts* copts);

extern void calc_phantom(const bart_dim_t dims[DIMS], _Complex float* out, bool d3, bool ksp, const bart_stride_t tstrs[DIMS], const _Complex float* traj, struct coil_opts* copts);
extern void calc_circ(const bart_dim_t dims[DIMS], _Complex float* img, bool d3, bool ksp, const bart_stride_t tstrs[DIMS], const _Complex float* traj, struct coil_opts* copts);
extern void calc_ring(const bart_dim_t dims[DIMS], _Complex float* img, bool ksp, const bart_stride_t tstrs[DIMS], const _Complex float* traj, struct coil_opts* copts);

extern void calc_moving_circ(const bart_dim_t dims[DIMS], _Complex float* out, bool ksp, const bart_stride_t tstrs[DIMS], const _Complex float* traj, struct coil_opts* copts);
extern void calc_heart(const bart_dim_t dims[DIMS], _Complex float* out, bool ksp, const bart_stride_t tstrs[DIMS], const _Complex float* traj, struct coil_opts* copts);

extern void calc_phantom_tubes(const bart_dim_t dims[DIMS], _Complex float* out, bool kspace, bool random, float rotation_angle, int N, const bart_stride_t tstrs[DIMS], const complex float* traj, struct coil_opts* copts);


struct ellipsis_s;
extern void calc_phantom_arb(int N, const struct ellipsis_s* data /*[N]*/, const bart_dim_t dims[DIMS], _Complex float* out, bool kspace, const bart_stride_t tstrs[DIMS], const complex float* traj, float rotation_angle, struct coil_opts* copts);

extern void calc_star(const bart_dim_t dims[DIMS], complex float* out, bool kspace, const bart_stride_t tstrs[DIMS], const complex float* traj, struct coil_opts* copts);
extern void calc_star3d(const bart_dim_t dims[DIMS], complex float* out, bool kspace, const bart_stride_t tstrs[DIMS], const complex float* traj, struct coil_opts* copts);
extern void calc_bart(const bart_dim_t dims[DIMS], complex float* out, bool kspace, const bart_stride_t tstrs[DIMS], const complex float* traj, struct coil_opts* copts);
extern void calc_brain(const bart_dim_t dims[DIMS], complex float* out, bool kspace, const bart_stride_t tstrs[DIMS], const complex float* traj, struct coil_opts* copts);

extern void calc_cfl_geom(const bart_dim_t dims[DIMS], complex float* out, bool kspace, const bart_stride_t tstrs[DIMS], const complex float* traj, int D_max, bart_dim_t hdims[2][D_max], complex float* x[2], struct coil_opts* copts);
extern _Complex double* sample_signal(int D, bart_dim_t odims[D], const bart_dim_t gdims[D], const float* grid, const bart_dim_t sgdims[D], const float* sgrid, const struct phantom_opts* popts, const struct coil_opts* copts);
extern _Complex double stl_fun_k(const void* v, const bart_dim_t C, const float k1[]);

#endif

