
#ifndef _SENS_H
#define _SENS_H

#include "misc/dimtypes.h"
#include "simu/grid.h"

enum coil_type { COIL_NONE, HEAD_2D_8CH, HEAD_3D_64CH };

struct coil_opts;

typedef _Complex double (*sim_fun_t)(const void*, const bart_dim_t C, const float pos[]);
typedef void (*co_dstr_t)(void* v);

struct coil_opts {

	_Bool kspace;
	enum coil_type ctype;
	bart_flags_t flags; // flags for channel selection
	bart_dim_t N; // chosen number of coil channels
	void* data;
	co_dstr_t dstr;
	sim_fun_t fun;
};

extern struct coil_opts coil_opts_defaults;
extern struct coil_opts coil_opts_pha_defaults;

extern void get_position(bart_dim_t D, float p[4], const bart_dim_t pos[D], const bart_dim_t gdims[D], const float* grid);
extern _Complex double* sample_coils(bart_dim_t D, bart_dim_t sdims[D], const bart_dim_t gdims[D], const float* grid, const struct coil_opts* copts);
extern void cnstr_coils(bart_dim_t D, struct coil_opts* copts, bool legacy_fov);
extern float* create_senstraj(bart_dim_t D, bart_dim_t gdims[D], struct grid_opts* gopts, struct coil_opts* copts);

extern complex float* sens_internal_H2D8CH(bart_dim_t D, bart_dim_t dims[D], bart_flags_t flags);
extern complex float* sens_internal_H3D64CH(bart_dim_t D, bart_dim_t dims[D], bart_flags_t flags);

extern const _Complex float sens_coeff[8][5][5];
extern const _Complex float sens64_coeff[64][5][5][5];

#endif
