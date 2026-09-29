#ifndef _SEQ_KERNEL_H
#define _SEQ_KERNEL_H

#include "misc/dimtypes.h"
#include "misc/cppwrap.h"

#include "seq/event.h"
#include "seq/config.h"

extern void seq_linearize_events(int N, struct seq_event ev[__VLA(N)], double* start_block, enum seq_block mode, double tr, double raster);

extern void seq_gradient(double m[3], double t, int N, const struct seq_event ev[__VLA(N)]);
extern void seq_slew(double m[3], double t, int N, const struct seq_event ev[__VLA(N)]);

extern void seq_compute_moment0(int M, float moments[__VLA(M)][3], double dt, int N, const struct seq_event ev[__VLA(N)]);
extern void seq_compute_moment0_offset(int M, float moments[__VLA(M)][3], double start, double dt, int N, const struct seq_event ev[__VLA(N)]);
extern void seq_compute_adc_samples(int D, const bart_dim_t adc_dims[__VLA(D)], _Complex float* adc, int N, const struct seq_event ev[__VLA(N)]);
extern void seq_gradients_support(int M, double gradients[__VLA(M)][6], int N, const struct seq_event ev[__VLA(N)]);

extern void seq_pulse_shapes_to_cfl(int D, const bart_dim_t sdims[__VLA(D)], _Complex float* shapes,
				int N, const struct rf_shape rf_shapes[__VLA(N)]);

extern void seq_pulse_shapes_from_cfl(int N, struct rf_shape rf_shapes[__VLA(N)],
					int D, const bart_dim_t sdims[__VLA(D)], const _Complex float* shapes);

extern void seq_events_to_cfl(int D, const bart_dim_t edims[__VLA(D)], _Complex float* events,
				bart_dim_t* block_pos, double start_block, int N, const struct seq_event ev[__VLA(N)]);

extern int seq_events_from_cfl(int N, struct seq_event ev[__VLA(N)], double* start_block,
				int D, const bart_dim_t edims[__VLA(D)], const _Complex float* events);

extern _Bool seq_events_is_image_block(int E, struct seq_event ev[__VLA(E)]);
extern double seq_events_cfl_find_tr(int D, const bart_dim_t edims[__VLA(D)], _Complex float* events);

#include "misc/cppwrap.h"

#endif // _SEQ_KERNEL_H
