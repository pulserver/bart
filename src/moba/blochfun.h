
struct nlop_s;
struct noir_model_conf_s;
struct moba_conf_s;

extern struct nlop_s* nlop_bloch_create(int N, const long out_dims[N], const long in_dims[N],
		const complex float* b1, const complex float* b0, const struct moba_conf_s* _data);

