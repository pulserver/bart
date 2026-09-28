#include "misc/dimtypes.h"



struct nlop_s;
struct nlop_s* nlop_mi_metric_create(int N, const bart_dim_t dims[N], int nbins, float smin, float smax, float mmin, float mmax, bool mask);

