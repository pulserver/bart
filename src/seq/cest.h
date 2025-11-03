#ifndef _SEQ_CEST_H
#define _SEQ_CEST_H


#include "config.h"
#include "misc/cppwrap.h"

#include "seq/event.h"

struct seq_state;
struct seq_config;

extern long cest_offsets(const struct seq_config* seq_config);

extern int cest_block(struct seq_event ev[6], const struct seq_state* seq_state, const struct seq_config* seq);

#include "misc/cppwrap.h"

#endif