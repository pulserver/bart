#include "misc/dimtypes.h"

struct nlop_norm_inv_conf {

	bool store_nlop;
	struct iter_conjgrad_conf* iter_conf;
};

struct linop_s;

extern struct nlop_norm_inv_conf nlop_norm_inv_default;

extern const struct nlop_s* norm_inv_create(struct nlop_norm_inv_conf* conf, const struct nlop_s* normal_op);
extern const struct nlop_s* norm_inv_lambda_create(struct nlop_norm_inv_conf* conf, const struct nlop_s* normal_op, bart_flags_t lflag);
extern const struct nlop_s* norm_inv_lop_lambda_create(struct nlop_norm_inv_conf* conf, const struct linop_s* lop, bart_flags_t lflags);

extern const struct nlop_s* nlop_maxeigen_create(const struct nlop_s* normal_op);