#ifndef _SEQ_CHECKS_H
#define _SEQ_CHECKS_H

#include "misc/cppwrap.h"

#include "seq/event.h"

extern void seq_rf_count(int N, long calls[__VLA(N)], int E, const struct seq_event ev[__VLA(E)]);


#include "misc/cppwrap.h"

#endif // _SEQ_CHECKS_H
