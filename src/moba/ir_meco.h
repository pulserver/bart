#include "misc/mri.h"

struct nlop_s;
struct noir_model_conf_s;

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
	IR_MECO_WF_fB0,
	IR_MECO_WF_R2S,
	IR_MECO_T1_R2S,
	IR_MECO_W_T1_F_T1_R2S,
};
#endif
extern const struct linop_s* ir_meco_get_fB0_trafo(struct nlop_s* op);

extern int ir_meco_get_num_of_coeff(enum meco_model sel_model);

extern const struct nlop_s* nlop_ir_meco_model_create(int N, const long map_dims[N], const long in_dims[N], const long TI_dims[N],
				const complex float* TI, const long TE_dims[N], const complex float* TE, enum meco_model meco_model, enum fat_spec fat_spec);

extern struct nlop_s* nlop_ir_meco_create(int N, const long map_dims[N], const long out_dims[N], const long in_dims[N], const long TI_dims[N],
		const complex float* TI, const long TE_dims[N], const complex float* TE, const float* scale_fB0, enum meco_model meco_model, enum fat_spec fat_spec, const float* scale);
