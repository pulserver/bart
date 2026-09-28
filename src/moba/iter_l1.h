#include "misc/dimtypes.h"



struct iter3_irgnm_conf;
struct nlop_s;
struct opt_reg_s;


#ifndef DIMS
#define DIMS 16
#endif

struct mdb_irgnm_l1_conf {

	struct iter3_irgnm_conf* c2;

	float step;
	float lower_bound;
	unsigned constrained_maps;
	bart_flags_t l2flags;
	bart_flags_t wavflags;
	bool auto_norm;
	bool no_sens_l2;

	unsigned long wav_trans_flags;
	int algo;
	float rho;
	struct opt_reg_s* ropts;
	int tvscales_N;
	complex float* tvscales;
	float l1val;

	int pusteps;
	float ratio;
};

void mdb_irgnm_l1(const struct mdb_irgnm_l1_conf* conf,
		const bart_dim_t dims[DIMS],
		struct nlop_s* nlop,
		bart_dim_t N, float* dst, float* dst_ref,
		bart_dim_t M, const float* src);

