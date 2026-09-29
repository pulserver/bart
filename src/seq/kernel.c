/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <assert.h>

#include "num/multind.h"

#include "misc/mri.h"
#include "misc/misc.h"

#include "seq/seq.h"

#include "kernel.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


/*
 * evaluate gradient/slew rate of single event at time t
 */
static void event_sample(double m[3], bool deriv, double t, const struct seq_event* ev)
{
	assert(SEQ_EVENT_GRADIENT == ev->type);

	for (int a = 0; a < 3; a++)
		m[a] = 0.;

	if (ev->start > t)
		return;

	if (ev->end < t)
		return;

	double s = ev->start;
	double e = ev->end;
	double c = ev->mid;

	for (int a = 0; a < 3; a++) {

		if (c > s) {

			double A = ev->grad.ampl[a] / (c - s);

			if (t <= c)
				m[a] = A * (deriv ? 1. : (t - s));
		}

		if (e > c) {

			double B = ev->grad.ampl[a] / (e - c);

			if (c < t)
				m[a] = B * (deriv ? 1. : (e - t));
		}
	}
}


/*
 * evaluate gradient of all events at time t
 */
void seq_gradient(double m[3], double t, int N, const struct seq_event ev[N])
{
	for (int a = 0; a < 3; a++)
		m[a] = 0.;

	for (int j = 0; j < N; j++) {

		if (SEQ_EVENT_GRADIENT != ev[j].type)
			continue;

		double m0[3];
		event_sample(m0, false, t, &ev[j]);

		for (int a = 0; a < 3; a++)
			m[a] += m0[a];
	}
}

/*
 * evaluate slew rate of all events at time t
 */
void seq_slew(double m[3], double t, int N, const struct seq_event ev[N])
{
	for (int a = 0; a < 3; a++)
		m[a] = 0.;

	for (int j = 0; j < N; j++) {

		if (SEQ_EVENT_GRADIENT != ev[j].type)
			continue;

		double m0[3];
		event_sample(m0, true, t, &ev[j]);

		for (int a = 0; a < 3; a++)
			m[a] += m0[a];
	}
}


void seq_linearize_events(int N, struct seq_event ev[__VLA(N)], double* start_block, enum seq_block mode, double tr, double raster)
{
	if ((0 >= N) || (0. > *start_block))
		return;

	double end = seq_block_end(N, ev, mode, tr, raster);

	for (int i = 0; i < N; i++) {

		ev[i].start += *start_block;
		ev[i].mid   += *start_block;
		ev[i].end   += *start_block;
	}

	*start_block += end;
}


/*
 * Compute 0th moment on a raster after RF pulse.
 * 0th moment before excitation pulse is zero.
 */
void seq_compute_moment0_offset(int M, float moments[M][3], double start, double dt, int N, const struct seq_event ev[N])
{
	for (int i = 0; i < M; i++) 
		for (int a = 0; a < 3; a++)
			moments[i][a] = 0.;

	double m_rf[3] = { };

	// last pulse
	int rf_idx = -1;

	if (0 < events_counter(SEQ_EVENT_PULSE, N, ev))
		rf_idx = events_idx(events_counter(SEQ_EVENT_PULSE, N, ev) - 1, SEQ_EVENT_PULSE, N, ev);

	double t_reset = -1.;

	if ((0 < rf_idx) && (SEQ_RF_EXCITATION == ev[rf_idx].pulse.type)) {

		t_reset = ev[rf_idx].mid;
		moment_sum(m_rf, t_reset, N, ev);
	}


	for (int p = 0; p < M; p++) {

		assert((0 <= p) && (p <= M));

		double m[3] = { };

		if (start + (p + 0.5) * dt > t_reset) {

			moment_sum(m, start + (p + 0.5) * dt, N, ev);

			for (int a = 0; a < 3; a++) 
				moments[p][a] = m[a] - m_rf[a];

		} else {

			for (int a = 0; a < 3; a++) 
				moments[p][a] = m[a];
		}
	}
}


void seq_compute_moment0(int M, float moments[M][3], double dt, int N, const struct seq_event ev[N])
{
	seq_compute_moment0_offset(M, moments, 0., dt, N, ev);
}

/*
 * Compute times and phase of adc samples. 
 */
void seq_compute_adc_samples(int D, const bart_dim_t adc_dims[D], complex float* adc, int N, const struct seq_event ev[N])
{
	md_clear(D, adc_dims, adc, CFL_SIZE);

	bart_stride_t adc_strs[D];
	md_calc_strides(D, adc_strs, adc_dims, CFL_SIZE);

	int e = 0;

	for (int i = 0; i < N; i++) {

		if (SEQ_EVENT_ADC != ev[i].type)
			continue;

		double dwell = 1.E-9 * ev[i].adc.dwell_ns /  ev[i].adc.os;

		bart_dim_t pos[DIMS] = { };
		pos[TE_DIM] = e;

		do {
			double ts = ev[i].start + (pos[1] + 0.5) * dwell;

			MD_ACCESS(D, adc_strs, (pos[READ_DIM] = 0, pos), adc) = ts;
			MD_ACCESS(D, adc_strs, (pos[READ_DIM] = 1, pos), adc) = cexpf(1.i * DEG2RAD(ev[i].adc.phase + ev[i].adc.freq * 360. * (ts - ev[i].mid)));

		} while (md_next(D, adc_dims, PHS1_FLAG, pos));

		e++;
	}

	assert(e == adc_dims[TE_DIM]);
}


void seq_gradients_support(int M, double gradients[M][6], int N, const struct seq_event ev[N])
{
	for (int i = 0; i < M; i++) 
		for (int a = 0; a < 6; a++)
			gradients[i][a] = 0.;

	int g = 0;

	for (int i = 0; i < N; i++) {

		if (SEQ_EVENT_GRADIENT != ev[i].type)
			continue;

		gradients[g][0] = ev[i].start;
		gradients[g][1] = ev[i].mid;
		gradients[g][2] = ev[i].end;
		gradients[g][3] = ev[i].grad.ampl[0];
		gradients[g][4] = ev[i].grad.ampl[1];
		gradients[g][5] = ev[i].grad.ampl[2];

		g++;
	}
}


void seq_pulse_shapes_to_cfl(int D, const bart_dim_t sdims[D], complex float* shapes, int N, const struct rf_shape rf_shapes[N])
{
	bart_stride_t sstrs[D];
	md_calc_strides(D, sstrs, sdims, CFL_SIZE);

	md_clear(D, sdims, shapes, CFL_SIZE);

	bart_dim_t pos[DIMS] = { };

	do {

		pos[PHS1_DIM] = 0;
		do {

			MD_ACCESS(D, sstrs, (pos[READ_DIM] = 0, pos), shapes) = rf_shapes[pos[TIME_DIM]].shape[pos[PHS1_DIM]];

		} while (md_next(D, sdims, PHS1_FLAG, pos));

		pos[READ_DIM] = 1;

		MD_ACCESS(D, sstrs, (pos[PHS1_DIM] = 0, pos), shapes) = rf_shapes[pos[TIME_DIM]].sar_calls;
		MD_ACCESS(D, sstrs, (pos[PHS1_DIM] = 1, pos), shapes) = rf_shapes[pos[TIME_DIM]].sar_dur;
		MD_ACCESS(D, sstrs, (pos[PHS1_DIM] = 2, pos), shapes) = rf_shapes[pos[TIME_DIM]].fa_prep;
		MD_ACCESS(D, sstrs, (pos[PHS1_DIM] = 3, pos), shapes) = rf_shapes[pos[TIME_DIM]].max;
		MD_ACCESS(D, sstrs, (pos[PHS1_DIM] = 4, pos), shapes) = rf_shapes[pos[TIME_DIM]].integral;
		MD_ACCESS(D, sstrs, (pos[PHS1_DIM] = 5, pos), shapes) = rf_shapes[pos[TIME_DIM]].samples;

	} while (md_next(D, sdims, TIME_FLAG, pos));

}

extern void seq_pulse_shapes_from_cfl(int N, struct rf_shape rf_shapes[N], int D, const bart_dim_t sdims[D], const _Complex float* shapes)
{
	bart_stride_t sstrs[D];
	md_calc_strides(D, sstrs, sdims, CFL_SIZE);

	bart_dim_t pos[DIMS] = { };

	do {

		pos[READ_DIM] = 1;
		rf_shapes[pos[TIME_DIM]].sar_calls = MD_ACCESS(D, sstrs, (pos[PHS1_DIM] = 0, pos), shapes);
		rf_shapes[pos[TIME_DIM]].sar_dur = MD_ACCESS(D, sstrs, (pos[PHS1_DIM] = 1, pos), shapes);
		rf_shapes[pos[TIME_DIM]].fa_prep = MD_ACCESS(D, sstrs, (pos[PHS1_DIM] = 2, pos), shapes);
		rf_shapes[pos[TIME_DIM]].max = MD_ACCESS(D, sstrs, (pos[PHS1_DIM] = 3, pos), shapes);
		rf_shapes[pos[TIME_DIM]].integral = MD_ACCESS(D, sstrs, (pos[PHS1_DIM] = 4, pos), shapes);
		rf_shapes[pos[TIME_DIM]].samples = (bart_dim_t)MD_ACCESS(D, sstrs, (pos[PHS1_DIM] = 5, pos), shapes);

		pos[READ_DIM] = 0;

		for (int i = 0; i < rf_shapes[pos[TIME_DIM]].samples; i++)
			rf_shapes[pos[TIME_DIM]].shape[i] = MD_ACCESS(D, sstrs, (pos[PHS1_DIM] = i, pos), shapes);

	} while (md_next(D, sdims, TIME_FLAG, pos));
}


void seq_events_to_cfl(int D, const bart_dim_t edims[D], complex float* events, bart_dim_t* block_pos, double start_block, int N, const struct seq_event ev[N])
{
	bart_stride_t estrs[D];
	md_calc_strides(D, estrs, edims, CFL_SIZE);

	bart_dim_t pos[DIMS] = { };
	pos[TIME_DIM] = *block_pos;

	do {
		if (pos[PHS1_DIM] >=  N)
			break;

		MD_ACCESS(D, estrs, (pos[READ_DIM] = 0, pos), events) = ev[pos[PHS1_DIM]].start + start_block;
		MD_ACCESS(D, estrs, (pos[READ_DIM] = 1, pos), events) = ev[pos[PHS1_DIM]].mid + start_block;
		MD_ACCESS(D, estrs, (pos[READ_DIM] = 2, pos), events) = ev[pos[PHS1_DIM]].end + start_block;
		MD_ACCESS(D, estrs, (pos[READ_DIM] = 3, pos), events) = ev[pos[PHS1_DIM]].type;

		switch (ev[pos[PHS1_DIM]].type) {

		case SEQ_EVENT_PULSE:

			MD_ACCESS(D, estrs, (pos[READ_DIM] = 4, pos), events) = ev[pos[PHS1_DIM]].pulse.shape_id;
			MD_ACCESS(D, estrs, (pos[READ_DIM] = 5, pos), events) = ev[pos[PHS1_DIM]].pulse.type;
			MD_ACCESS(D, estrs, (pos[READ_DIM] = 6, pos), events) = ev[pos[PHS1_DIM]].pulse.fa;
			MD_ACCESS(D, estrs, (pos[READ_DIM] = 7, pos), events) = ev[pos[PHS1_DIM]].pulse.freq;
			MD_ACCESS(D, estrs, (pos[READ_DIM] = 8, pos), events) = ev[pos[PHS1_DIM]].pulse.phase;
			break;

		case SEQ_EVENT_GRADIENT:

			MD_ACCESS(D, estrs, (pos[READ_DIM] = 5, pos), events) = ev[pos[PHS1_DIM]].grad.ampl[0];
			MD_ACCESS(D, estrs, (pos[READ_DIM] = 6, pos), events) = ev[pos[PHS1_DIM]].grad.ampl[1];
			MD_ACCESS(D, estrs, (pos[READ_DIM] = 7, pos), events) = ev[pos[PHS1_DIM]].grad.ampl[2];
			break;

		case SEQ_EVENT_ADC:

			MD_ACCESS(D, estrs, (pos[READ_DIM] = 4, pos), events) = ev[pos[PHS1_DIM]].adc.dwell_ns;
			MD_ACCESS(D, estrs, (pos[READ_DIM] = 5, pos), events) = ev[pos[PHS1_DIM]].adc.columns;
			MD_ACCESS(D, estrs, (pos[READ_DIM] = 6, pos), events) = ev[pos[PHS1_DIM]].adc.os;
			MD_ACCESS(D, estrs, (pos[READ_DIM] = 7, pos), events) = ev[pos[PHS1_DIM]].adc.freq;
			MD_ACCESS(D, estrs, (pos[READ_DIM] = 8, pos), events) = ev[pos[PHS1_DIM]].adc.phase;
			MD_ACCESS(D, estrs, (pos[READ_DIM] = 9, pos), events) = ev[pos[PHS1_DIM]].adc.flags;
			for (int idx = 0; idx < DIMS; idx++)
				MD_ACCESS(D, estrs, (pos[READ_DIM] = 10 + idx, pos), events) = ev[pos[PHS1_DIM]].adc.pos[idx];
			break;

		default:

		}

	} while (md_next(D, edims, PHS1_FLAG, pos));

	(*block_pos)++;

}

extern int seq_events_from_cfl(int N, struct seq_event ev[N], double* start_block, int D, const bart_dim_t edims[D], const _Complex float* events)
{
	assert(N >= edims[PHS1_DIM]);

	bart_stride_t estrs[D];
	md_calc_strides(D, estrs, edims, CFL_SIZE);

	bart_dim_t pos[DIMS] = { };
	*start_block = MD_ACCESS(D, estrs, (pos[READ_DIM] = 0, pos), events); // FIXME

	do {

		ev[pos[PHS1_DIM]].start = MD_ACCESS(D, estrs, (pos[READ_DIM] = 0, pos), events) - *start_block;
		ev[pos[PHS1_DIM]].mid = MD_ACCESS(D, estrs, (pos[READ_DIM] = 1, pos), events) - *start_block;
		ev[pos[PHS1_DIM]].end = MD_ACCESS(D, estrs, (pos[READ_DIM] = 2, pos), events) - *start_block;
		ev[pos[PHS1_DIM]].type = (enum seq_event_type)MD_ACCESS(D, estrs, (pos[READ_DIM] = 3, pos), events);

		if ((0. > ev[pos[PHS1_DIM]].start) && (0 > ev[pos[PHS1_DIM]].mid) && (0. > ev[pos[PHS1_DIM]].end))
			return pos[PHS1_DIM];

		switch (ev[pos[PHS1_DIM]].type) {

		case SEQ_EVENT_PULSE:

			ev[pos[PHS1_DIM]].pulse.shape_id = (int)MD_ACCESS(D, estrs, (pos[READ_DIM] = 4, pos), events);
			ev[pos[PHS1_DIM]].pulse.type = (enum rf_type_t)MD_ACCESS(D, estrs, (pos[READ_DIM] = 5, pos), events);
			ev[pos[PHS1_DIM]].pulse.fa = MD_ACCESS(D, estrs, (pos[READ_DIM] = 6, pos), events);
			ev[pos[PHS1_DIM]].pulse.freq = MD_ACCESS(D, estrs, (pos[READ_DIM] = 7, pos), events);
			ev[pos[PHS1_DIM]].pulse.phase = MD_ACCESS(D, estrs, (pos[READ_DIM] = 8, pos), events);
			break;

		case SEQ_EVENT_GRADIENT:

			ev[pos[PHS1_DIM]].grad.ampl[0] = MD_ACCESS(D, estrs, (pos[READ_DIM] = 5, pos), events);
			ev[pos[PHS1_DIM]].grad.ampl[1] = MD_ACCESS(D, estrs, (pos[READ_DIM] = 6, pos), events);
			ev[pos[PHS1_DIM]].grad.ampl[2] = MD_ACCESS(D, estrs, (pos[READ_DIM] = 7, pos), events);
			break;

		case SEQ_EVENT_ADC:

			ev[pos[PHS1_DIM]].adc.dwell_ns = (bart_dim_t)MD_ACCESS(D, estrs, (pos[READ_DIM] = 4, pos), events);
			ev[pos[PHS1_DIM]].adc.columns = (bart_dim_t)MD_ACCESS(D, estrs, (pos[READ_DIM] = 5, pos), events);
			ev[pos[PHS1_DIM]].adc.os = MD_ACCESS(D, estrs, (pos[READ_DIM] = 6, pos), events);
			ev[pos[PHS1_DIM]].adc.freq = MD_ACCESS(D, estrs, (pos[READ_DIM] = 7, pos), events);
			ev[pos[PHS1_DIM]].adc.phase = MD_ACCESS(D, estrs, (pos[READ_DIM] = 8, pos), events);
			ev[pos[PHS1_DIM]].adc.flags = (bart_flags_t)MD_ACCESS(D, estrs, (pos[READ_DIM] = 9, pos), events);
			for (int idx = 0; idx < DIMS; idx++)
				ev[pos[PHS1_DIM]].adc.pos[idx] = (bart_dim_t)MD_ACCESS(D, estrs, (pos[READ_DIM] = 10 + idx, pos), events);
			break;

		default:

		}

	} while (md_next(D, edims, PHS1_FLAG, pos));

	return edims[PHS1_DIM];
}


bool seq_events_is_image_block(int E, struct seq_event ev[E])
{
	int adc_idx = events_idx(0, SEQ_EVENT_ADC, E, ev);

	if (0 > adc_idx)
		return false;

	uint64_t non_image = SEQ_ADC_FLAG_ADJ | SEQ_ADC_FLAG_DUMMY;

	if (ev[adc_idx].adc.flags & non_image)
		return false;

	return true;
}


double seq_events_cfl_find_tr(int D, const bart_dim_t edims[D], _Complex float* events)
{
	bart_dim_t pos[D] = { };

	bart_stride_t strs[D];
	md_calc_strides(D, strs, edims, 1);

	double time = -1.;
	double start_block = 0.;
	bart_dim_t image1_pos = 0;

	struct seq_event ev[edims[PHS1_DIM]] = { };

	do {

		int E = seq_events_from_cfl(edims[PHS1_DIM], ev, &start_block, D, edims, events + md_calc_offset(D, strs, pos));

		if (seq_events_is_image_block(E, ev)) {

			if ((0. < time) && (1 == pos[TIME_DIM] - image1_pos))
				return start_block - time;

			time = start_block;
			image1_pos = pos[TIME_DIM];
		}


	} while (md_next(D, edims, TIME_FLAG, pos));

	return -1;
}
