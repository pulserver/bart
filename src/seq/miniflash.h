#ifndef _SEQ_MINIFLASH_H
#define _SEQ_MINIFLASH_H

#include "misc/cppwrap.h"

#include "seq/event.h"

struct seq_config;

void miniflash_interface_custom(struct seq_config* seq, int nl, const long params_long[__VLA(nl)], int nd, const double params_double[__VLA(nd)]);
void miniflash_interface_custom_back(const struct seq_config* seq, int nl, long params_long[__VLA(nl)], int nd, double params_double[__VLA(nd)]);

extern double miniflash_minimum_tr(const struct seq_config* seq);
extern void miniflash_minimum_te(const struct seq_config* seq, double* min_te, double* fill_te);
extern double miniflash_total_measure_time(const struct seq_config* seq);

extern int miniflash_sample_rf_shapes(int N, struct rf_shape pulse[__VLA(N)], const struct seq_config* seq);
extern int miniflash(int N, struct seq_event ev[__VLA(N)], struct seq_state* seq_state, const struct seq_config* seq);
extern int miniflash_block(int N, struct seq_event ev[__VLA(N)], struct seq_state* seq_state, const struct seq_config* seq);

#include "misc/cppwrap.h"

#endif // _SEQ_MINIFLASH_H
