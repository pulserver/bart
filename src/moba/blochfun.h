#include "misc/dimtypes.h"


struct nlop_s;
struct noir_model_conf_s;
struct moba_conf_s;

extern const struct linop_s* bloch_get_alpha_trafo(const struct nlop_s* op);
extern void bloch_forw_alpha(const struct linop_s* op, complex float* dst, const complex float* src);
extern void bloch_back_alpha(const struct linop_s* op, complex float* dst, const complex float* src);

extern struct nlop_s* nlop_bloch_create(int N, const bart_dim_t der_dims[N], const bart_dim_t map_dims[N], const bart_dim_t out_dims[N], const bart_dim_t in_dims[N],
		const complex float* b1, const complex float* b0, const struct moba_conf_s* _data);

