/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>
#include <assert.h>

#include "misc/misc.h"

#include "seq/seq.h"
#include "seq/gradient.h"
#include "seq/adc_rf.h"
#include "seq/event.h"
#include "seq/flash.h"
#include "seq/seq_asl.h"

const int coeff2_dim_offset = 3;

static double get_mean_grad(enum asl_condition condition)
{
	return LABEL_CONDITION == condition ? 1.E-3 : 0.;
}

static double calc_phase_shift(enum asl_condition condition, double gamma, double pulse_spacing, double slice_shift)
{
	return 360. * gamma * get_mean_grad(condition) * slice_shift * pulse_spacing;
}

static double calc_curr_phase_shift(int pulse_idx, const struct seq_config* seq, enum asl_condition condition)
{
	double init_phase = 0;
	double phase_shift = calc_phase_shift(condition, seq->sys.gamma, seq->asl.pulse_spacing, seq->geom.shift[seq->asl.label_slice_index][2]);

	if (UNBALANCED_CONTROL_CONDITION == condition)
		init_phase = (pulse_idx % 2 != 0) ? 180 : 0;

	return init_phase + pulse_idx * phase_shift;
}

static int prep_pcasl_grad_sli(struct grad_trapezoid* grad, const struct asl_pulse hanning, const struct seq_sys sys, double ampl)
{
	*grad = (struct grad_trapezoid){ 0 };

	if (sys.grad.max_amplitude < ampl)
		return 0;

	grad->rampup = MAX(ampl * sys.grad.inv_slew_rate, sys.coil_control_lead);
	grad->flat = 1. * hanning.rf_duration;
	grad->rampdown = grad->rampup;
	grad->ampl = ampl;

	return 1;
}

static int prep_pcasl_grad_sli_rew(struct grad_trapezoid* grad_rew, struct grad_trapezoid* grad_sli, const struct seq_sys sys, double mean_grad, double pulse_spacing)
{
	*grad_rew = (struct grad_trapezoid){ 0 };

	double grad_duration = pulse_spacing - grad_total_time(grad_sli);
	double mom_grad_rew = fabs(mean_grad * pulse_spacing - grad_momentum(grad_sli));

	if (!grad_soft(grad_rew, grad_duration, -mom_grad_rew, sys.grad))
		return 0;

	return 1;
}

static int add_pcasl_events(struct seq_event* ev, int pulse_pos, const struct seq_config* seq, enum asl_condition condition)
{
	double projSLICE[3] = { 0., 0., 1. };

	struct grad_trapezoid grad_sli;
	struct grad_trapezoid grad_rew;

	double mean_grad = get_mean_grad(condition);

	if (!prep_pcasl_grad_sli(&grad_sli, seq->asl.hanning, seq->sys, seq->asl.ampl_grad_sli))
		return ERROR_PREP_GRAD_SLI;

	if (!prep_pcasl_grad_sli_rew(&grad_rew, &grad_sli, seq->sys, mean_grad, seq->asl.pulse_spacing))
		return ERROR_PREP_GRAD_SLI_REPH;

	int i = 0;

	i += seq_grad_to_event(ev + i, 0., &grad_sli, projSLICE);

	double phase_shift = calc_curr_phase_shift(pulse_pos, seq, condition);
	i += prep_rf_hanning(ev + i, ev[i - 1].start, phase_shift, seq);

	i += seq_grad_to_event(ev + i, ev[i - 2].end, &grad_rew, projSLICE);

	return i;
}

double calc_asl_duration(const struct seq_config* seq)
{
	int cl_pairs = seq->loop_dims[AVG_DIM] * 2;
	int num_imaging_slices = seq->loop_dims[SLICE_DIM] - 1; // -1 because one slice is used for labeling
	double readout_duration = seq->phys.tr * seq->loop_dims[PHS1_DIM];

	// Calculate duration for all L/C blocks: LD + PLD + delay
	double ld = calc_num_asl_pulses(seq->asl.ld, seq->asl.pulse_spacing) * seq->asl.pulse_spacing;
	double asl_block_duration = (ld + seq->asl.pld + seq->magn.inv_delay_time) * cl_pairs;

	// Calculate duration for M0 block
	double m0_duration = (readout_duration * num_imaging_slices * seq->loop_dims[TIME_DIM]) + seq->magn.inv_delay_time;

	// Calculate duration for all imaging blocks (hence for each L/C block)
	double imaging_duration = (readout_duration * num_imaging_slices * seq->loop_dims[TIME_DIM]) * cl_pairs;

	return m0_duration + asl_block_duration + imaging_duration;
}

int calc_num_asl_pulses(double ld, double pulse_spacing)
{
	return floor(ld / pulse_spacing);
}

int calc_total_num_asl_pulses(const struct seq_config* seq)
{
	return calc_num_asl_pulses(seq->asl.ld, seq->asl.pulse_spacing) * (seq->loop_dims[AVG_DIM] * 2);
}

int calc_asl_coeff2_dim(const struct seq_config* seq)
{
	return calc_num_asl_pulses(seq->asl.ld, seq->asl.pulse_spacing) + 1; // +1 for PLD wait event
}

int asl(int N, struct seq_event ev[N], const struct seq_state* seq_state, const struct seq_config* seq)
{
	int asl_condition;

	switch (seq_state->pos[BATCH_DIM]) {

	case 0:
		return 0; // M0 image

	case 1:
		asl_condition = LABEL_CONDITION;
		break;

	case 2:
		asl_condition = UNBALANCED_CONTROL_CONDITION;
		break;

	default:
		error("Invalid ASL BATCH_DIM=%ld (valid options: 0=M0, 1=LABEL, 2=CONTROL)\n", seq_state->pos[BATCH_DIM]);
	}

	int i = 0;

	int num_hanning_pulses = calc_num_asl_pulses(seq->asl.ld, seq->asl.pulse_spacing);

	if ((seq_state->pos[COEFF2_DIM] - coeff2_dim_offset) < num_hanning_pulses) // Add hanning pulse
		i += add_pcasl_events(ev + i, ((seq_state->pos[COEFF2_DIM] - coeff2_dim_offset)), seq, asl_condition);
	else // Add PLD after labeling is finished
		i += wait_time_to_event(ev + i, 0, seq->asl.pld);

	if (0 > i)
		error("ASL condition %d not possible! - check seq_config, %d] \n", asl_condition, i);

	assert(i < N);

	return i;
}