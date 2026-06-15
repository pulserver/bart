

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
	unsigned long l2flags;
	unsigned long wavflags;
	bool auto_norm;
	bool no_sens_l2;

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
		const long dims[DIMS],
		struct nlop_s* nlop,
		long N, float* dst, float* dst_ref,
		long M, const float* src);

