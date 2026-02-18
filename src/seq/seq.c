/* Copyright 2025-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>

#include "num/multind.h"
#include "num/rand.h"

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/version.h"

#include "seq/config.h"
#include "seq/event.h"
#include "seq/helpers.h"
#include "seq/misc.h"

#include "seq/adc_rf.h"
#include "seq/anglecalc.h"
#include "seq/pulse.h"
#include "seq/flash.h"
#include "seq/mag_prep.h"
#include "seq/cest.h"
#include "seq/seq_asl.h"

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
	memcpy(seq->conf, &seq_config_defaults, sizeof *seq->conf);

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

	const unsigned int min_driver[5] = { };
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
	int idx = 0;

	for (; idx < seq->geom.mb_factor; idx++) {

		if (idx >= N)
			return -1;

		pulse[idx].sar_calls = flash_ex_calls(seq);
		pulse[idx].sar_dur = seq->phys.rf_duration;
		pulse[idx].fa_prep = seq->phys.flip_angle;

		const float alpha = 0.5;

		pulse[idx].samples = lround(1.E6 * seq->phys.rf_duration);

		if (SEQ_MAX_RF_SAMPLES < pulse[idx].samples)
			return -1;

		double dwell = seq->phys.rf_duration / pulse[idx].samples;

		struct pulse_sms ps = pulse_sms_defaults;

		pulse_sms_init(&ps, seq->phys.rf_duration, seq->phys.flip_angle, 0., seq->phys.bwtp, alpha,
			seq->geom.mb_factor, idx, seq->geom.sms_distance, seq->geom.slice_thickness);

		pulse[idx].max = ps.A; // this is scaled by fa / fa_prep
		pulse[idx].integral = pulse_sms_integral(&ps);

		struct pulse* pp = CAST_UP(&ps);

		for (int j = 0; j < pulse[idx].samples; j++)
			pulse[idx].shape[j] = pulse_eval(pp, j * dwell);
	}

	if (   (SEQ_PREP_IR_NONSELECTIVE == seq->magn.mag_prep)
	    || (SEQ_PREP_IR_SELECTIVE == seq->magn.mag_prep)) {

		struct pulse_hypsec hs = pulse_hypsec_defaults;

		pulse_hypsec_init(seq->sys.gamma, &hs);

		pulse[idx].max = hs.A;
		pulse[idx].integral = pulse_hypsec_integral(&hs);
		pulse[idx].fa_prep = 180.;

		struct pulse* pp = CAST_UP(&hs);

		pulse[idx].sar_calls = seq->loop_dims[BATCH_DIM];
		pulse[idx].sar_dur = pp->duration;

		pulse[idx].samples = lround(0.5 * 1E6 * pulse[idx].sar_dur);

		if (SEQ_MAX_RF_SAMPLES < pulse[idx].samples)
			return -1;

		double dwell = pp->duration / pulse[idx].samples;

		for (int j = 0; j < pulse[idx].samples; j++)
			pulse[idx].shape[j] = pulse_eval(pp, j * dwell);

		idx++;
	}

	if (SEQ_CEST_GAUSS == seq->cest.sat_type) {

		struct pulse_gauss pg = pulse_gauss_defaults;

		pulse_gauss_init(&pg, seq->cest.gauss_pulse_duration, seq->cest.gauss_pulse_fa, 0., pulse_gauss_defaults.bwtp, pulse_gauss_defaults.alpha);

		pulse[idx].max = pg.A;
		pulse[idx].integral = pulse_gauss_integral(&pg);

		struct pulse* pp = CAST_UP(&pg);

		pulse[idx].sar_calls = seq->cest.sat_pulses * seq->loop_dims[CSHIFT_DIM];
		pulse[idx].sar_dur = seq->cest.gauss_pulse_duration;
		pulse[idx].fa_prep = seq->cest.gauss_pulse_fa;

		pulse[idx].samples = lround(1E4 * pulse[idx].sar_dur);

		if (SEQ_MAX_RF_SAMPLES < pulse[idx].samples)
			return -1;

		double dwell = pp->duration / pulse[idx].samples;

		for (int j = 0; j < pulse[idx].samples; j++)
			pulse[idx].shape[j] = pulse_eval(pp, j * dwell);

		idx++;
	}
	else if (SEQ_CEST_OC == seq->cest.sat_type) {

		pulse[idx].sar_calls = seq->cest.sat_pulses * seq->loop_dims[CSHIFT_DIM];

		struct pulse_arb arb = pulse_arb_oc_cest_sat_defaults;
		pulse_arb_init(&arb, seq->sys.gamma);
		struct pulse* pp = CAST_UP(&arb);

		pulse[idx].sar_dur = pp->duration;

		float scaling = seq->cest.oc_pulse_b1_scaling * sqrt( 1 + seq->cest.sat_pulse_pause / pulse[idx].sar_dur);
		pulse[idx].fa_prep = arb.super.flipangle / scaling;

		pulse[idx].integral = pulse_arb_integral(&arb);

		pulse[idx].samples = arb.samples;

		if (SEQ_MAX_RF_SAMPLES < pulse[idx].samples)
			return -1;

		double dwell = pp->duration / pulse[idx].samples;

		for (int j = 0; j < pulse[idx].samples; j++)
			pulse[idx].shape[j] = pulse_eval(pp, j * dwell);

		pulse[idx].max = arb.A; // default in oc_pulse{[]

		idx++;
	}

	if (SEQ_ASL_NONE != seq->asl.label_type) {

		pulse[idx].sar_calls = calc_total_num_asl_pulses(seq);
		pulse[idx].sar_dur = seq->asl.hanning.rf_duration;
		pulse[idx].fa_prep = seq->asl.hanning.flip_angle;

		const float alpha = 0.5;

		pulse[idx].samples = lround(1.E6 * seq->asl.hanning.rf_duration);

		if (SEQ_MAX_RF_SAMPLES < pulse[idx].samples)
			return -1;

		double dwell = seq->asl.hanning.rf_duration / pulse[idx].samples;

		struct pulse_sinc ps = pulse_sinc_defaults;

		pulse_sinc_init(&ps, seq->asl.hanning.rf_duration, seq->asl.hanning.flip_angle, 0., seq->phys.bwtp, alpha);

		pulse[idx].max = ps.A; // this is scaled by fa / fa_prep
		pulse[idx].integral = pulse_sinc_integral(&ps);

		struct pulse* pp = CAST_UP(&ps);

		for (int j = 0; j < pulse[idx].samples; j++)
			pulse[idx].shape[j] = pulse_eval(pp, j * dwell);

		idx++;
	}

	return idx;
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

static long get_chrono_slice(const struct seq_state* seq_state, const struct seq_config* seq)
{
	if ((SEQ_ASL_NONE != seq->asl.label_type) && (0 == seq_state->pos[COEFF_DIM]))
		return seq->asl.label_slice_index;

	if (1 < seq->geom.mb_factor)
		return seq_state->pos[PHS2_DIM] + seq_state->pos[SLICE_DIM] * seq->loop_dims[PHS2_DIM];

	return seq_state->pos[SLICE_DIM];
}

static int check_settings(const struct seq_state* seq_state, const struct seq_config* seq)
{
	if (0 > seq->loop_dims[PHS2_DIM])
		return ERROR_SETTING_DIM;

	if (SEQ_MAX_SLICES < get_slices(seq))
		return ERROR_SETTING_DIM;

	if ((SEQ_TRIGGER_OFF != seq->trigger.type) && (SEQ_CEST_NONE != seq->cest.sat_type))
		return ERROR_CEST_TRIGGER;

	if (   (SEQ_PREP_SR_SELECTIVE == seq->magn.mag_prep)
	    || (SEQ_PREP_SR_NONSELECTIVE == seq->magn.mag_prep)
	    || (SEQ_PREP_SR_ADIABATIC == seq->magn.mag_prep))
		return ERROR_MAG_PREP;

	if (   (0 < seq->magn.prep_scans)
	    && ((SEQ_PREP_OFF != seq->magn.mag_prep) || (SEQ_ASL_NONE != seq->asl.label_type)))
		return ERROR_PREP_SCANS;

	if (SEQ_ASL_NONE != seq->asl.label_type) {

		if (seq->loop_dims[SLICE_DIM] < 2)
			return ERROR_SETTING_ASL;

		if (SEQ_PREP_OFF != seq->magn.mag_prep)
			return ERROR_SETTING_ASL;

		if (SEQ_CEST_NONE != seq->cest.sat_type)
			return ERROR_SETTING_ASL;
	}


	if (SEQ_CONTEXT_BINARY != seq_state->context) {

		if (   (SEQ_PEMODE_CARTESIAN == seq->enc.pe_mode)
		    || (SEQ_PEMODE_CARTESIAN_LINEAR == seq->enc.pe_mode))
			return 1;

		if ((SEQ_PEMODE_RAGA == seq->enc.pe_mode)
		    && !check_gen_fib(seq->loop_dims[PHS1_DIM], seq->enc.tiny))
			return ERROR_SETTING_SPOKES_RAGA;

		if ((SEQ_PEMODE_RAGA == seq->enc.pe_mode)
		    && (PHS1_FLAG & seq->enc.aligned_flags))
				return ERROR_SETTING_RAGA_AL;

		if (0 == (seq->loop_dims[PHS1_DIM] % 2))
			return ERROR_SETTING_SPOKES_EVEN;
	}

	if ((1 < seq->loop_dims[TE_DIM]) && (0.5 != seq->phys.asym_echo))
		return ERROR_SETTING_ASYM_MECO;

	return 1;
}

int seq_block(int N, struct seq_event ev[N], struct seq_state* seq_state, const struct seq_config* seq)
{
	int err = check_settings(seq_state, seq);
	if (1 > err)
		return err;

	seq_state->chrono_slice = get_chrono_slice(seq_state, seq);

	if (   (SEQ_BLOCK_KERNEL_PREPARE == seq_state->mode)
	    || (SEQ_BLOCK_KERNEL_CHECK == seq_state->mode))
		return flash(N, ev, seq_state, seq);

	long zeros[DIMS] = { };
	long last_idx[DIMS];

	for (int i = 0; i < DIMS; i++)
		last_idx[i] = seq->loop_dims[i] - 1;

	// changed beahvior for sequential multislice
	unsigned long msm_flag = 0UL;

	// changed behavior for ASL
	unsigned long asl_flag = 0UL;

	if (md_check_equal_order(DIMS, seq->order, seq_loop_order_multislice, SEQ_FLAGS))
	       msm_flag = SLICE_FLAG ;
	
	if (SEQ_ASL_NONE != seq->asl.label_type)
		asl_flag = AVG_FLAG;

	if (0 == seq_state->pos[COEFF_DIM]) {

		if (md_check_equal_dims(DIMS, zeros, seq_state->pos, ~0UL)) {

			seq_state->mode = SEQ_BLOCK_PRE;

			return wait_time_to_event(ev, 0., seq->magn.init_delay);
		}

		zeros[COEFF2_DIM] = 1;

		if (md_check_equal_dims(DIMS, zeros, seq_state->pos, ~0UL)) {

			seq_state->mode = SEQ_BLOCK_KERNEL_NOISE;

			return flash(N, ev, seq_state, seq);
		}

		if (1 < seq_state->pos[COEFF2_DIM]) {

			// Skip spoke iterations for ASL
			if (   (SEQ_ASL_NONE != seq->asl.label_type)
			    && ((seq_state->pos[BATCH_DIM] == 0) || (seq_state->pos[PHS1_DIM] > 0) || (seq_state->pos[TIME_DIM] > 0)))
				md_max_dims(DIMS, COEFF2_FLAG, seq_state->pos, seq_state->pos, last_idx);

			if (md_check_equal_dims(DIMS, zeros, seq_state->pos, ~(BATCH_FLAG | COEFF2_FLAG | SLICE_FLAG | PHS2_FLAG))) {

				if ((0 < seq->magn.prep_scans) && (2 < seq_state->pos[COEFF2_DIM])) {

					seq_state->mode = SEQ_BLOCK_KERNEL_DUMMY;
					seq_state->seq_ut = 1;
					return flash(N, ev, seq_state, seq);
				}
			}

			if (md_check_equal_dims(DIMS, zeros, seq_state->pos, ~(BATCH_FLAG | msm_flag | COEFF2_FLAG | asl_flag | CSHIFT_FLAG))) {


				if ((SEQ_CEST_NONE != seq->cest.sat_type) && md_check_equal_dims(DIMS, zeros, seq_state->pos, ~(COEFF2_FLAG | CSHIFT_FLAG))) {

					int cest_with_inversion = 0;

					if (0 != mag_prep(ev, seq))
						cest_with_inversion = 1;

					seq_state->mode = SEQ_BLOCK_PRE;

					if (   (seq_state->pos[COEFF2_DIM] > 1)
					    && (seq_state->pos[COEFF2_DIM] < (seq->loop_dims[COEFF2_DIM] - (1 + cest_with_inversion))))
							return cest_block(ev, seq_state, seq);
				}

				if ((2 == seq_state->pos[COEFF2_DIM]) && (SEQ_TRIGGER_OFF != seq->trigger.type)) {

					seq_state->mode = SEQ_BLOCK_PRE;

					ev[0] = (struct seq_event){ .start = 0., .mid = 0., .end = seq->trigger.delay_time, .type = SEQ_EVENT_TRIGGER };

					return 1;
				}

				if ((2 < seq_state->pos[COEFF2_DIM]) && (SEQ_ASL_NONE != seq->asl.label_type)) {

					seq_state->mode = SEQ_BLOCK_PRE;
					return asl(N, ev, seq_state, seq);
				}

				if (seq->loop_dims[COEFF2_DIM] - 1  == seq_state->pos[COEFF2_DIM]) {

					seq_state->mode = SEQ_BLOCK_PRE;
					return mag_prep(ev, seq);
				}

				return 0;
			}

		} else if (0 < seq_state->pos[PHS1_DIM]) {

			md_max_dims(DIMS, (COEFF2_FLAG | PHS2_FLAG) &  ~msm_flag & ~asl_flag, seq_state->pos, seq_state->pos, last_idx);
		}
	}

	if (1 == seq_state->pos[COEFF_DIM]) {

		// Skip readout for ASL label slice and only acquire one M0 image at the beginning of the measurement
		if (SEQ_ASL_NONE != seq->asl.label_type) {
			
			if (seq_state->pos[SLICE_DIM] == seq->asl.label_slice_index) {

				md_max_dims(DIMS, COEFF2_FLAG, seq_state->pos, seq_state->pos, last_idx);
				return 0;
			}    

			if (seq_state->pos[BATCH_DIM] == 0 && seq_state->pos[AVG_DIM] > 0) {
				
				md_max_dims(DIMS, COEFF2_FLAG | SLICE_FLAG | TIME_FLAG | PHS1_FLAG, seq_state->pos, seq_state->pos, last_idx);
				return 0;
			}
		}

		int i = 0;

		seq_state->mode = SEQ_BLOCK_KERNEL_IMAGE;
		md_max_dims(DIMS, (COEFF2_FLAG), seq_state->pos, seq_state->pos, last_idx);

		if (seq->trigger.trigger_out && md_check_equal_dims(DIMS, (long [DIMS]){ 0 }, seq_state->pos, PHS1_FLAG))
			ev[i++] = (struct seq_event){ .start = 0., .mid = 0., .end = 1e-3, .type = SEQ_EVENT_OUTPUT, NULL };

		return flash(N - i, ev + i, seq_state, seq) + i;
	}

	if (2 == seq_state->pos[COEFF_DIM]) {

		md_max_dims(DIMS, (COEFF2_FLAG | PHS2_FLAG) & ~msm_flag & ~asl_flag, seq_state->pos, seq_state->pos, last_idx);

		if (md_check_equal_dims(DIMS, last_idx, seq_state->pos, (SEQ_FLAGS & ~(BATCH_FLAG | msm_flag | asl_flag)))
			&& (0. < seq->magn.inv_delay_time)) {

				if(SEQ_ASL_NONE != seq->asl.label_type && (seq_state->pos[BATCH_DIM] == 0 && seq_state->pos[AVG_DIM] > 0))
					return 0;

				seq_state->mode = SEQ_BLOCK_POST;

				ev[0] = (struct seq_event){ .start = 0., .mid = 0., .end = seq->magn.inv_delay_time, .type = SEQ_EVENT_WAIT, NULL };

				return 1;
		}

		return 0;
	}

	return 0;
}

int seq_continue(struct seq_state* seq_state, const struct seq_config* seq)
{
	return md_next_permuted(DIMS, seq->order, seq->loop_dims, SEQ_FLAGS, seq_state->pos);
}

