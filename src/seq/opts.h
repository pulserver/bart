#ifndef __SEQ_OPTS_H
#define __SEQ_OPTS_H

#include "misc/cppwrap.h"

#include "misc/opts.h"

struct seq_config;

enum gradient_mode { GRAD_FAST, GRAD_NORMAL, GRAD_WHISPER };


struct seq_opts {

	float dt;
	long samples;
	double rel_shift[3];
	long raga_full_frames;
	float dist;

	enum gradient_mode gradient_mode;

	_Bool chrono;
	_Bool support;

	const char* raga_file;

	long custom_params_long[SEQ_MAX_PARAMS_LONG];
	double custom_params_double[SEQ_MAX_PARAMS_DOUBLE];
};

extern const struct seq_opts seq_opts_defaults;


extern void seq_cmdline(int* argcp, char* argv[*argcp], int m, const struct arg_s args[m],
			const char* help_str, struct seq_config* conf, struct seq_opts* seq_opts);


extern int read_config_from_str(struct seq_config* seq, int N, const char* buffer_in);

#include "misc/cppwrap.h"


#endif