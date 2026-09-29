#include "misc/dimtypes.h"


struct nlop_s;
extern const struct nlop_s* nlop_zexp_create(int N, const bart_dim_t dims[N]);
extern const struct nlop_s* nlop_zlog_create(int N, const bart_dim_t dims[N]);

