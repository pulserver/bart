#ifndef _SEQ_KERNEL_H
#define _SEQ_KERNEL_H

#include "misc/cppwrap.h"

#include "seq/event.h"
#include "seq/config.h"

extern void seq_linearize_events(int N, struct seq_event ev[__VLA(N)], double* start_block, enum seq_block mode, double tr, double raster);

extern void seq_gradient(double m[3], double t, int N, const struct seq_event ev[__VLA(N)]);
extern void seq_slew(double m[3], double t, int N, const struct seq_event ev[__VLA(N)]);

extern void seq_compute_moment0(int M, float moments[__VLA(M)][3], double dt, int N, const struct seq_event ev[__VLA(N)]);
extern void seq_compute_moment0_offset(int M, float moments[__VLA(M)][3], double start, double dt, int N, const struct seq_event ev[__VLA(N)]);
extern void seq_compute_adc_samples(int D, const long adc_dims[__VLA(D)], _Complex float* adc, int N, const struct seq_event ev[__VLA(N)]);
extern void seq_gradients_support(int M, double gradients[__VLA(M)][6], int N, const struct seq_event ev[__VLA(N)]);

extern void seq_pulse_shapes_to_cfl(int D, const long sdims[__VLA(D)], _Complex float* shapes,
				int N, const struct rf_shape rf_shapes[__VLA(N)]);

extern void seq_pulse_shapes_from_cfl(int N, struct rf_shape rf_shapes[__VLA(N)],
					int D, const long sdims[__VLA(D)], const _Complex float* shapes);

#include "misc/cppwrap.h"

#endif // _SEQ_KERNEL_H
