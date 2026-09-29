
#ifndef _MECO_H
#define _MECO_H 1

#include "misc/dimtypes.h"
#include <complex.h>

struct linop_s;
struct nlop_s;
enum fat_spec;

#ifndef _MECO_MODEL
#define _MECO_MODEL 1
enum meco_model {
	MECO_WF,
	MECO_WFR2S,
	MECO_WF2R2S,
	MECO_R2S,
	MECO_PHASEDIFF,
	MECO_PI,
	IR_MECO_T1_R2S,
	IR_MECO_W_T1_F_T1_R2S,
};
#endif

extern int get_num_of_coeff(enum meco_model sel_model);
extern bart_flags_t get_PD_flag(enum meco_model sel_model);
extern bart_flags_t get_R2S_flag(enum meco_model sel_model);
extern bart_flags_t get_fB0_flag(enum meco_model sel_model);



extern struct nlop_s* nlop_meco_create(int N, const bart_dim_t y_dims[N], const bart_dim_t x_dims[N], const complex float* TE, enum meco_model sel_model, enum fat_spec fat_spec, float B0);

extern struct nlop_s* nlop_ir_meco_create(int N, const bart_dim_t out_dims[N], const bart_dim_t in_dims[N], const bart_dim_t TI_dims[N],
		const complex float* TI, const bart_dim_t TE_dims[N], const complex float* TE, enum meco_model meco_model, enum fat_spec fat_spec, float B0);


#endif // _MECO_H

