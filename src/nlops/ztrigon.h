#include "misc/dimtypes.h"


struct nlop_s;
extern const struct nlop_s* nlop_zsin_create(int N, const bart_dim_t dims[N]);
extern const struct nlop_s* nlop_zsinc_create(int N, const bart_dim_t dims[N]);
extern const struct nlop_s* nlop_zcos_create(int N, const bart_dim_t dims[N]);

extern const struct nlop_s* nlop_zasin_create(int N, const bart_dim_t dims[N]);
extern const struct nlop_s* nlop_zacos_create(int N, const bart_dim_t dims[N]);

