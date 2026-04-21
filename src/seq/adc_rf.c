/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <assert.h>
#include <complex.h>
#include <math.h>

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/list.h"

#include "num/multind.h"
#include "num/rand.h"

#include "noncart/traj.h"

#include "seq/event.h"
#include "seq/config.h"
#include "seq/anglecalc.h"
#include "seq/pulse.h"
#include "seq/misc.h"

#include "adc_rf.h"


#define RFSPOIL_INCREMENT_DEG	50.0


double phase_clamp(double phase)
{
	double ret = fmod(phase, 360.);
	return ret + ((ret > 180.) ? -360. : ((ret < -180.) ? 360. : 0));
}


double rf_spoiling(int D, const long pos[D], const struct seq_config* seq)
{
	double idx;

	switch (seq->phys.contrast) {

	case SEQ_CONTRAST_NO_SPOILING:

		return 0.;

	case SEQ_CONTRAST_RF_SPOILED:

		idx = 1. + md_ravel_index_permuted(DIMS, pos, SEQ_FLAGS & ~(COEFF_FLAG|COEFF2_FLAG), seq->loop_dims, seq->order);

		return fmod(0.5 * RFSPOIL_INCREMENT_DEG * (idx + pow(idx, 2.)), 360.);

	case SEQ_CONTRAST_RF_RANDOM:

		return uniform_rand() * 360. - 180.; // random seed initialized in bart_seq_prepare

	default:
		assert(0);
	}
}


int prep_rf_excitation(struct seq_event* rf_ev, double start, double rf_spoil_phase,
		const struct seq_state* seq_state, const struct seq_config* seq)
{
	if (SEQ_BLOCK_KERNEL_NOISE == seq_state->mode)
		return 0;

	rf_ev->type = SEQ_EVENT_PULSE;

	rf_ev->start = start;
	rf_ev->end = rf_ev->start + seq->phys.rf_duration;

	const double asym_pulse = 0.5;

	rf_ev->mid = rf_ev->start + (rf_ev->end - rf_ev->start) * asym_pulse;

	rf_ev->pulse.shape_id = 0;

	if (1 < seq->geom.mb_factor)
		rf_ev->pulse.shape_id = seq_state->pos[SLICE_DIM];

	if (seq->geom.mb_factor <= rf_ev->pulse.shape_id) // SLICE_DIM is only used for SMS
		return 0;	

	rf_ev->pulse.type = SEQ_RF_EXCITATION;
	rf_ev->pulse.fa = seq->phys.flip_angle;
	rf_ev->pulse.freq = seq->sys.gamma * seq->geom.shift[seq_state->chrono_slice][2] * slice_amplitude(seq);
	rf_ev->pulse.phase = phase_clamp(rf_spoil_phase);

	return 1;
}


int prep_rf_inversion(struct seq_event* rf_ev, double start, const struct seq_config* seq)
{
	if ((seq->magn.mag_prep != SEQ_PREP_IR_NONSELECTIVE) && (seq->magn.mag_prep != SEQ_PREP_IR_SELECTIVE))
		return 0;

	rf_ev->type = SEQ_EVENT_PULSE;	

	rf_ev->start = start;

	struct pulse_hypsec hs = pulse_hypsec_defaults;
	struct pulse* pp = CAST_UP(&hs);

	rf_ev->end = rf_ev->start + pp->duration;

	const double asym_pulse = 0.5;

	rf_ev->mid = rf_ev->start + (rf_ev->end - rf_ev->start) * asym_pulse;

	rf_ev->pulse.shape_id = seq->geom.mb_factor;

	rf_ev->pulse.type = SEQ_RF_REFOCUSSING;
	rf_ev->pulse.fa = 180.;
	rf_ev->pulse.freq = 0.;
	rf_ev->pulse.phase = 0.;

	return 1;
}

int prep_rf_hanning(struct seq_event* rf_ev, double start, double phase_shift, const struct seq_config* seq)
{
	if (SEQ_ASL_NONE == seq->asl.label_type)
		return 0;

	rf_ev->type = SEQ_EVENT_PULSE;

	rf_ev->start = start;
	rf_ev->end = rf_ev->start + seq->asl.hanning.rf_duration;

	const double asym_pulse = 0.5;

	rf_ev->mid = rf_ev->start + (rf_ev->end - rf_ev->start) * asym_pulse;

	rf_ev->pulse.shape_id = seq->geom.mb_factor;	

	rf_ev->pulse.type = SEQ_RF_EXCITATION;
	rf_ev->pulse.fa = seq->asl.hanning.flip_angle;
	rf_ev->pulse.freq = seq->sys.gamma * seq->geom.shift[seq->asl.label_slice_index][2] * seq->asl.ampl_grad_sli;
	rf_ev->pulse.phase = phase_clamp(phase_shift);

	return 1;
}

long inv_calls(const struct seq_config* seq)
{
	long calls = seq->loop_dims[BATCH_DIM] * ((SEQ_ASL_NONE == seq->asl.label_type) ? seq->loop_dims[CSHIFT_DIM] : 1);

	if (SEQ_ORDER_SEQ_MS == seq->enc.order)
		return calls * seq->loop_dims[SLICE_DIM];

	return calls;
}

long flash_ex_calls(const struct seq_config* seq)
{
	long dims[DIMS];
	md_select_dims(DIMS, SEQ_FLAGS & ~(COEFF_FLAG|COEFF2_FLAG), dims, seq->loop_dims);

	long incomplete_raga_spks = 0;
	if (SEQ_PEMODE_RAGA == seq->enc.pe_mode)
		incomplete_raga_spks = seq->loop_dims[PHS1_DIM] - seq->loop_dims[ITER_DIM];

	if (1 < seq->geom.mb_factor) {

		dims[SLICE_DIM] = 1;
		incomplete_raga_spks *= dims[PHS2_DIM];

	} else {

		dims[SLICE_DIM] = (SEQ_ASL_NONE != seq->asl.label_type) ? seq->loop_dims[SLICE_DIM] - 1 : seq->loop_dims[SLICE_DIM];
	}
	
	if (SEQ_ORDER_SEQ_MS == seq->enc.order)
		incomplete_raga_spks *= dims[SLICE_DIM];

	if (SEQ_ASL_NONE != seq->asl.label_type) {

		dims[AVG_DIM] = dims[AVG_DIM] * 2 + 1;
		dims[BATCH_DIM] = 1;
	}

	return md_calc_size(DIMS, dims) - incomplete_raga_spks
		+ dims[PHS2_DIM] * dims[SLICE_DIM] * seq->magn.prep_scans;
}

static long cols_to_echo(long echo, const struct seq_config* seq)
{
	return (0 == (echo % 2)) ? (seq->geom.baseres * seq->phys.asym_echo) : (seq->geom.baseres - (seq->geom.baseres * seq->phys.asym_echo));
}

double adc_time_to_echo(long echo, const struct seq_config* seq)
{
	double dc_shift = 0.;
	if (   (SEQ_PEMODE_CARTESIAN == seq->enc.pe_mode)
	    || (SEQ_PEMODE_CARTESIAN_LINEAR == seq->enc.pe_mode))
		dc_shift = 0.5 * seq->phys.dwell / seq->phys.os;

	return seq->phys.dwell * cols_to_echo(echo, seq) + dc_shift;
}


double adc_duration(const struct seq_config* seq)
{
	return round_up_raster(seq->phys.dwell * seq->geom.baseres * (0.5 + seq->phys.asym_echo), seq->sys.raster_rf);
}

static double adc_nco_freq(double proj_angle, long chrono_slice, const struct seq_config* seq)
{
	return seq->sys.gamma * 
		(seq->geom.shift[chrono_slice][0] * ro_amplitude(seq) * sin(proj_angle)
		+ seq->geom.shift[chrono_slice][1] * ro_amplitude(seq) * cos(proj_angle));
}

int prep_adc(struct seq_event* adc_ev, double start, double rf_spoil_phase,
		const struct seq_state* seq_state, const struct seq_config* seq)
{
	adc_ev->type = SEQ_EVENT_ADC;

	adc_ev->start = start;
	adc_ev->end = adc_ev->start + adc_duration(seq);

	adc_ev->adc.dwell_ns = (long)(seq->phys.dwell * 1.E9 + 0.5);
	adc_ev->adc.columns = (long)(seq->geom.baseres * (seq->phys.asym_echo + 0.5));

	adc_ev->mid = adc_ev->start + adc_time_to_echo(seq_state->pos[TE_DIM], seq);

	md_copy_dims(DIMS, adc_ev->adc.pos, seq_state->pos);

	if (SEQ_PEMODE_RAGA == seq->enc.pe_mode) {

		struct traj_conf conf;
		traj_conf_from_seq(&conf, seq);

		adc_ev->adc.pos[PHS1_DIM] = raga_increment_from_pos(seq->order, seq_state->pos,
							(SEQ_FLAGS | TE_FLAG) & ~(COEFF_FLAG | COEFF2_FLAG),
							seq->loop_dims, &conf);
	}

	if (SEQ_PEMODE_CARTESIAN == seq->enc.pe_mode)
		adc_ev->adc.pos[PHS1_DIM] = cartesian_line(seq_state->pos, seq);


	adc_ev->adc.flags = 0;
	if (SEQ_BLOCK_KERNEL_NOISE == seq_state->mode)
		adc_ev->adc.flags |= SEQ_ADC_FLAG_ADJ;

	if (SEQ_BLOCK_KERNEL_DUMMY == seq_state->mode) {

		adc_ev->adc.pos[PHS1_DIM] = seq_state->pos[COEFF2_DIM] - 3; // 3 blocks before dummy (delay, noise, ecg)
		adc_ev->adc.flags |= SEQ_ADC_FLAG_DUMMY;
	}

	long zeros[DIMS] = { 0 };
	long last_idx[DIMS];
	for (int i = 0; i < DIMS; i++)
		last_idx[i] = seq->loop_dims[i] - 1;

	if (md_check_equal_dims(DIMS, zeros, seq_state->pos, PHS1_FLAG | TE_FLAG | AVG_FLAG))
		adc_ev->adc.flags |= SEQ_ADC_FLAG_FIRSTSLI;

	if (md_check_equal_dims(DIMS, last_idx, seq_state->pos, PHS1_FLAG | TE_FLAG | AVG_FLAG | BATCH_FLAG))
		adc_ev->adc.flags |= SEQ_ADC_FLAG_PHASEFT;

	if (md_check_equal_dims(DIMS, last_idx, seq_state->pos, PHS2_FLAG | AVG_FLAG | BATCH_FLAG))
		adc_ev->adc.flags |= SEQ_ADC_FLAG_PARTFT;

	if (md_check_equal_dims(DIMS, last_idx, seq_state->pos, PHS1_FLAG | TE_FLAG | AVG_FLAG))
		adc_ev->adc.flags |= SEQ_ADC_FLAG_LASTSLI;

	if (   md_check_equal_dims(DIMS, last_idx, seq_state->pos, PHS1_FLAG | TE_FLAG | AVG_FLAG | TIME2_FLAG)
	    && md_check_equal_dims(DIMS, last_idx, seq_state->pos, PHS2_FLAG)) {

		adc_ev->adc.flags |= SEQ_ADC_FLAG_LASTCON;

		if (md_check_equal_dims(DIMS, last_idx, seq_state->pos, PHS2_FLAG | SLICE_FLAG))
			adc_ev->adc.flags |= SEQ_ADC_FLAG_LASTMEAS;
	}

	if ((SEQ_BLOCK_KERNEL_CHECK != seq_state->mode) && md_check_equal_dims(DIMS, zeros, seq_state->pos, PHS1_FLAG))
		adc_ev->adc.flags |= SEQ_ADC_FLAG_MEASTIME;


	adc_ev->adc.os = seq->phys.os;

	double proj_angle = get_rot_angle(seq_state->pos, seq);

	adc_ev->adc.freq = adc_nco_freq(proj_angle, seq_state->chrono_slice, seq);

	double delta_pe = 0.;
	if (   ((SEQ_PEMODE_CARTESIAN == seq->enc.pe_mode) || (SEQ_PEMODE_CARTESIAN_LINEAR == seq->enc.pe_mode))
	    && (0 < seq->geom.shift[seq_state->chrono_slice][1]))
		delta_pe = (360. * (seq->geom.shift[seq_state->chrono_slice][1] / seq->geom.fov)) * (- 0.5 * seq->loop_dims[PHS1_DIM] + adc_ev->adc.pos[PHS1_DIM]);

	adc_ev->adc.phase = phase_clamp(rf_spoil_phase + delta_pe);


	return 1;
}

