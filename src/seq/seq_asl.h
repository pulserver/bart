#ifndef __SEQ_ASL_H
#define __SEQ_ASL_H

#include "misc/cppwrap.h"

#include "seq/event.h"

// Returns size of ASL BATCH_DIM so that M0 (= 0), label (= 1) and control (= 2) conditions are covered
#define ASL_BATCH_DIM_SIZE 3

enum asl_condition {

	LABEL_CONDITION,
	UNBALANCED_CONTROL_CONDITION,
};

extern const int coeff2_dim_offset;

struct seq_config;

extern double calc_asl_duration(const struct seq_config* seq);
extern int calc_num_asl_pulses(double ld, double pulse_spacing);
extern int calc_total_num_asl_pulses(const struct seq_config* seq);
extern int calc_asl_coeff2_dim(const struct seq_config* seq);

extern int asl(int N, struct seq_event ev[__VLA(N)], const struct seq_state* seq_state, const struct seq_config* seq);

#include "misc/cppwrap.h"

#endif // __SEQ_ASL_H
