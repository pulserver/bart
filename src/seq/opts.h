#ifndef __SEQ_OPTS_H
#define __SEQ_OPTS_H

#include "misc/cppwrap.h"

struct seq_config;

enum gradient_mode { GRAD_FAST, GRAD_NORMAL, GRAD_WHISPER };

extern int read_config_from_str(struct seq_config* seq, int N, const char* buffer_in);

#include "misc/cppwrap.h"


#endif