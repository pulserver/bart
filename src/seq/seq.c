/* Copyright 2025-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>
#include <complex.h>

#include "num/multind.h"
#include "num/rand.h"

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/version.h"

#include "seq/adc_rf.h"
#include "seq/config.h"
#include "seq/event.h"
#include "seq/misc.h"

#include "seq/flash.h"

#include "seq.h"

#define MAX_EVENTS 2048
#define MAX_RF_PULSES 32


struct bart_seq* bart_seq_alloc(const char* driver_version)
{
	struct bart_seq* seq = NULL;
	seq = xmalloc(sizeof *seq);

	seq->bart_version = bart_version;
	seq->driver_version = driver_version;

	seq->conf = xmalloc(sizeof *seq->conf);
	seq->state = xmalloc(sizeof *seq->state);

	seq->N = MAX_EVENTS;
	seq->event = xmalloc((size_t)seq->N * sizeof *seq->event);

	seq->P = MAX_RF_PULSES;
	seq->rf_shape = xmalloc((size_t)seq->P * sizeof *seq->rf_shape);

	return seq;
}

void bart_seq_defaults(struct bart_seq* seq)
{
	memcpy(seq->conf, &seq_config_defaults_flash, sizeof *seq->conf);

	memset(seq->state, 0, sizeof *seq->state);
	memset(seq->event, 0, (size_t)seq->N * sizeof *seq->event);
	memset(seq->rf_shape, 0, (size_t)seq->P * sizeof *seq->rf_shape);
}

int bart_seq_prepare(struct bart_seq* seq)
{
	num_rand_init(0ULL); // initialize here since once called before actual sequence start

	seq->state->mode = SEQ_BLOCK_KERNEL_PREPARE;

	if (SEQ_PEMODE_CARTESIAN == seq->conf->enc.pe_mode)
		seq->state->pos[PHS1_DIM] = seq->conf->loop_dims[PHS1_DIM] - 1;

	int N = seq_block(seq->N, seq->event, seq->state, seq->conf);

	if (0 > N)
		return N;

	// FIXME: further checks ? 

	N = seq_sample_rf_shapes(MAX_RF_PULSES, seq->rf_shape, seq->conf);
	
	for (int i = 0; i < DIMS; i++)
		seq->state->pos[i] = 0;

	seq->state->mode = SEQ_BLOCK_UNDEFINED;

	return N;
}


void bart_seq_free(struct bart_seq* seq)
{
	xfree(seq->conf);
	xfree(seq->state);
	xfree(seq->event);
	xfree(seq->rf_shape);
	xfree(seq);
}

int bart_seq_version_check(const char* driver_version, const unsigned int min_bart_version[5])
{
	// FIXME check for stable event.h, seq.h, helpers.h, custom_ui.h

	unsigned int vd[5];
	if (!version_parse(vd, driver_version))
		return -1;

	const unsigned int min_driver[5] = { 0, 1, 0, 0, 0 };
	if (version_compare(vd, min_driver) < 0)
		return -1;

	unsigned int vb[5];
	if (!version_parse(vb, bart_version))
		return -1;

	if (version_compare(vb, min_bart_version) < 0)
		return -1;

	return 1;
}


int seq_sample_rf_shapes(int N, struct rf_shape pulse[N], const struct seq_config* seq)
{
	switch (seq->seq_type) {

	case SEQ_TYPE_FLASH:

		return flash_sample_rf_shapes(N, pulse, seq);

	default:

		assert(0);
	}
}



/*
 * Compute gradients on a raster. This also works
 * (i.e. yields correct 0 moment) if the abstract
 * gradients do not start and end on the raster.
 * We integrate over each interval to obtain the
 * average gradient.
 */
void seq_compute_gradients(int M, double gradients[M][3], double dt, int N, const struct seq_event ev[N])
{
	for (int i = 0; i < M; i++) 
		for (int a = 0; a < 3; a++)
			gradients[i][a] = 0.;

	for (int i = 0; i < N; i++) {

		if (SEQ_EVENT_GRADIENT != ev[i].type)
			continue;

		double s = ev[i].start;
		double e = ev[i].end;


		/*            |    /
                 *            |   /|
                 *            .../..
                 *            | /  |
                 *            |/   |
                 *            /    |
                 *       ..../|    |
                 *  |____|__/_|____|____|
                 *    0    1    2    3  
                 */

		assert(0. <= s);

		double om[3];

		for (int a = 0; a < 3; a++)
			om[a] = 0.;

		for (int p = trunc(s / dt); p <= ceil(e / dt); p++) {

			assert(0 <= p);

			double m0[3];
			moment(m0, (p + 1.) * dt, &ev[i]);

			for (int a = 0; a < 3; a++) {

				if (p < M)
					gradients[p][a] += (m0[a] - om[a]) / dt;

				om[a] = m0[a];
			}
		}
	}
}


double seq_nco_freq(const struct seq_event* ev)
{
	return (SEQ_EVENT_PULSE == ev->type) ? ev->pulse.freq : ev->adc.freq;
}

double seq_nco_phase(int set, const struct seq_event* ev)
{
	double phase_mid = (SEQ_EVENT_PULSE == ev->type) ? ev->pulse.phase : ev->adc.phase;

	if (0 == set)
		phase_mid = -1. * phase_mid;

	double time = (set) ? (ev->mid - ev->start) : (ev->end - ev->mid);

	return phase_clamp(- seq_nco_freq(ev) * 360. * time + phase_mid);
}

double seq_pulse_scaling(const struct rf_shape* pulse)
{
	return 180. / M_PI * pulse->integral;
}

double seq_pulse_norm_sum(const struct rf_shape* pulse)
{
	double dwell = pulse->sar_dur / pulse->samples;

	return (pulse->integral / dwell) / pulse->max;
}

void seq_cfl_to_sample(const struct rf_shape* pulse, int idx, float* mag, float* pha)
{
	assert(idx < pulse->samples);

	complex float val = pulse->shape[idx];

	*mag = cabs(val) / pulse->max;
	*pha = fmod(carg(val) + 2. * M_PI, 2. * M_PI);
}




double seq_block_end(int N, const struct seq_event ev[N], enum seq_block mode, double tr, double raster)
{
	if ((SEQ_BLOCK_PRE == mode) || (SEQ_BLOCK_POST == mode))
		return round_up_raster(events_end_time(N, ev, 0, 0), raster);

	return tr;
}

double seq_block_end_flat(int N, const struct seq_event ev[N], double raster)
{
	return round_up_raster(events_end_time(N, ev, 1, 1), raster);
}


double seq_block_rdt(int N, const struct seq_event ev[N], double raster)
{
	return round_up_raster(events_end_time(N, ev, 1, 0) - seq_block_end_flat(N, ev, raster), raster);
}

int seq_block(int N, struct seq_event ev[N], struct seq_state* seq_state, const struct seq_config* seq)
{

	switch (seq->seq_type) {

	case SEQ_TYPE_FLASH:

		return flash_block(N, ev, seq_state, seq);

	default:

		assert(0);
	}
}

int seq_continue(struct seq_state* seq_state, const struct seq_config* seq)
{
	return md_next_permuted(DIMS, seq->order, seq->loop_dims, SEQ_FLAGS, seq_state->pos);
}

