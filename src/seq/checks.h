#ifndef _SEQ_CHECKS_H
#define _SEQ_CHECKS_H

#include "misc/dimtypes.h"
#include "misc/cppwrap.h"

#include "seq/event.h"

struct seq_sys;

extern void seq_rf_count(int N, bart_dim_t calls[__VLA(N)], int E, const struct seq_event ev[__VLA(E)]);

extern bool seq_check_gradients(int N, const struct seq_event ev[__VLA(N)], const struct seq_sys* sys);

extern bool seq_check_timing(int N, const struct seq_event ev[__VLA(N)], const struct seq_sys* sys);


#include "misc/cppwrap.h"

#endif // _SEQ_CHECKS_H
