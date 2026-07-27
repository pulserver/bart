#ifndef _SEQ_FLASH_H
#define _SEQ_FLASH_H

#include "misc/cppwrap.h"

#include "seq/event.h"

struct seq_config;

extern double flash_minimum_tr(const struct seq_config* seq);
extern void flash_minimum_te(const struct seq_config* seq, double* min_te, double* fill_te);
extern double flash_total_measure_time(const struct seq_config* seq);

extern int flash(int N, struct seq_event ev[__VLA(N)], struct seq_state* seq_state, const struct seq_config* seq);

#include "misc/cppwrap.h"

#endif // _SEQ_FLASH_H
