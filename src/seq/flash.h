#ifndef _SEQ_FLASH_H
#define _SEQ_FLASH_H

#include "misc/cppwrap.h"

#include "seq/event.h"

struct seq_config;

extern void flash_interface_custom_params(int reverse, struct seq_config* seq, int nl, long params_long[__VLA(nl)], int nd, double params_double[__VLA(nd)]);
extern void flash_interface_loop_dims(int reverse, struct seq_config* seq, const int D, long dims[__VLA(D)]);

extern double flash_minimum_tr(const struct seq_config* seq);
extern void flash_minimum_te(const struct seq_config* seq, double* min_te, double* fill_te);
extern double flash_total_measure_time(const struct seq_config* seq);

extern int flash_sample_rf_shapes(int N, struct rf_shape pulse[__VLA(N)], const struct seq_config* seq);
extern int flash(int N, struct seq_event ev[__VLA(N)], struct seq_state* seq_state, const struct seq_config* seq);

#include "misc/cppwrap.h"

#endif // _SEQ_FLASH_H
