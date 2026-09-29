
#ifndef GRID_H
#define GRID_H 1

#include "misc/dimtypes.h"
#include <complex.h>

#define VEC_DIM_S READ_DIM
#define VEC_FLAG_S (1u << VEC_DIM_S)

#ifndef DIMS
#define DIMS 16
#endif

struct grid_opts {

	bart_dim_t dims[DIMS];
	bool kspace;
	float b0[3];
	float b1[3];
	float b2[3];
	float bt;
};

extern struct grid_opts grid_opts_init;
extern struct grid_opts grid_opts_defaults;
extern struct grid_opts grid_opts_coilcoeff;

extern float* compute_grid(int D, bart_dim_t gdims[D], struct grid_opts* go, const bart_dim_t tdims[D], const complex float* traj);

#endif

