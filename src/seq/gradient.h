
#ifndef _SEQ_GRADIENT_H
#define _SEQ_GRADIENT_H

#include "seq/event.h"

struct grad_limits {

	double inv_slew_rate;
	double max_amplitude;
};

struct grad_trapezoid {

	double rampup;
	double flat;
	double rampdown;

	double ampl;
};

struct grad_support {

	double time;
	double ampl;
};

extern int seq_grad_support_to_event(int E, struct seq_event ev[__VLA(E)], double start_offset, int N, const struct grad_support grad[__VLA(N)], double proj[3]);

extern int seq_event_merge_degenerated_triangles(int E, struct seq_event ev[__VLA(E)]);

extern double grad_duration(const struct grad_trapezoid* grad);
extern double grad_total_time(const struct grad_trapezoid* grad);
extern double grad_momentum(struct grad_trapezoid* grad);

extern bool grad_soft(struct grad_trapezoid* grad, double dur, double moment, struct grad_limits sys);

extern int grad_support_with_offsets(struct grad_support support[4], double moment, struct grad_support start, struct grad_support end, const struct grad_limits sys);

extern bool grad_hard(struct grad_trapezoid* grad, double moment, struct grad_limits sys);

#endif // _SEQ_GRADIENT_H

