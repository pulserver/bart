/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>

#include "num/multind.h"

#include "seq/kernel.h"
#include "seq/event.h"
#include "seq/pulse.h"
#include "seq/seq.h"
#include "seq/seq_asl.h"

#include "utest.h"

static int PCASL_HANN_PULSE = 1;

// * ---------------------------------------------------------------------- *
// * Test helper functions                                                  *
// * ---------------------------------------------------------------------- *

static bool assert_gradient_event(struct seq_event* ev, int i, double grad_amp)
{
	if (SEQ_EVENT_GRADIENT != ev[i].type) {

		debug_printf(DP_INFO, "Seq-event of gradient wrong: i = %d, expected = %d | actual = %d\n",
			     i, SEQ_EVENT_GRADIENT, ev[i].type);
		return false;
	}

	if (0 != ev[i].grad.ampl[0]) {

		debug_printf(DP_INFO, "Gx amplitude wrong: i = %d, expected = %f | actual = %f\n",
			     i, grad_amp, ev[i].grad.ampl[0]);
		return false;
	}

	if (0 != ev[i].grad.ampl[1]) {

		debug_printf(DP_INFO, "Gy amplitude wrong: i = %d, expected = %f | actual = %f\n",
			     i, grad_amp, ev[i].grad.ampl[1]);
		return false;
	}

	if (fabs(grad_amp - ev[i].grad.ampl[2]) > UT_TOL) {

		debug_printf(DP_INFO, "Gz amplitude wrong: i = %d, expected = %f | actual = %f\n",
			     i, grad_amp, ev[i].grad.ampl[2]);
		return false;
	}

	return true;
}

static bool assert_pulse_event(
	int i,
	struct seq_event* ev,
	int expected_pulse_shape_id,
	double expected_rf_duration,
	double expected_rf_freq,
	double expected_phase_shift,
	double expected_phase_shift_neg)
{
	if (SEQ_EVENT_PULSE != ev[i].type) {

		debug_printf(DP_INFO, "Seq-event of pulse wrong: i = %d, expected = %d | actual = %d\n",
			     i, SEQ_EVENT_PULSE, ev[i].type);
		return false;
	}

	if (expected_pulse_shape_id != ev[i].pulse.shape_id) {

		debug_printf(DP_INFO, "RF pulse shape ID wrong: i = %d, actual = %d | expected = %d\n",
			     i, expected_pulse_shape_id, ev[i].pulse.shape_id);
		return false;
	}

	if (fabs(expected_rf_duration - (ev[i].end - ev[i].start)) > 1e-3) {

		debug_printf(DP_INFO, "RF pulse duration wrong: i = %d, expected = %f | actual = %f\n",
			     i, expected_rf_duration, ev[i].end - ev[i].start);
		return false;
	}

	if (fabs(expected_rf_freq - ev[i].pulse.freq) > 1e-5) {

		debug_printf(DP_INFO, "RF pulse frequency wrong: i = %d, expected = %f | actual = %f\n",
			     i, expected_rf_freq, ev[i].pulse.freq);
		return false;
	}

	if (fabs(expected_phase_shift - seq_nco_phase(1, &ev[i])) > UT_TOL) {

		debug_printf(DP_INFO, "Phase shift of RF pulse wrong: i = %d, expected = %f | actual = %f\n",
			     i, expected_phase_shift, seq_nco_phase(1, &ev[i]));
		return false;
	}

	if (fabs(expected_phase_shift_neg - seq_nco_phase(0, &ev[i])) > UT_TOL) {

		debug_printf(DP_INFO, "Neg phase shift of RF pulse wrong: i = %d, expected = %f | actual = %f\n",
			     i, expected_phase_shift_neg, seq_nco_phase(0, &ev[i]));
		return false;
	}

	return true;
}

static bool assert_slice_selection(int grad_ev_index, int pulse_ev_index, struct seq_event* ev, double expected_duration)
{
	if (SEQ_EVENT_GRADIENT != ev[grad_ev_index].type) {

		debug_printf(DP_INFO, "Seq-event of slice-selection gradient wrong: i = %d, expected = %d | actual = %d\n",
			     grad_ev_index, SEQ_EVENT_GRADIENT, ev[grad_ev_index].type);
		return false;
	}

	if (SEQ_EVENT_GRADIENT != ev[grad_ev_index + 1].type) {

		debug_printf(DP_INFO, "Seq-event of slice-selection gradient wrong: i = %d, expected = %d | actual = %d\n",
			     grad_ev_index + 1, SEQ_EVENT_GRADIENT, ev[grad_ev_index + 1].type);
		return false;
	}

	if (ev[grad_ev_index].mid != ev[pulse_ev_index].start || ev[grad_ev_index + 1].start != ev[pulse_ev_index].start) {

		debug_printf(DP_INFO, "Start of RF pulse not correctly aligned with slice-selection gradient.\n");
		return false;
	}

	if (fabs(ev[grad_ev_index + 1].mid - ev[pulse_ev_index].end) > 1 || fabs(ev[grad_ev_index].end - ev[pulse_ev_index].end) > 1) {

		debug_printf(DP_INFO, "End of RF pulse not correctly aligned with slice-selection gradient.\n");
		return false;
	}

	if (expected_duration != ev[grad_ev_index + 1].mid - ev[grad_ev_index].mid) {

		debug_printf(DP_INFO, "Flat duration of slice-selection gradient wrong: expected = %f | actual = %f\n",
			     expected_duration, ev[grad_ev_index].mid - ev[grad_ev_index + 1].mid);
		return false;
	}

	return true;
}

static bool assert_wait_event(int i, struct seq_event* ev, double expected_wait_dur, double expected_start_time)
{
	if (SEQ_EVENT_WAIT != ev[i].type) {

		debug_printf(DP_INFO, "Seq-event of wait-event wrong: i = %d, expected = %d | actual = %d\n",
			     i, SEQ_EVENT_GRADIENT, ev[i].type);
		return false;
	}

	if (fabs(expected_start_time - ev[i].start) > UT_TOL) {

		debug_printf(DP_INFO, "Start time of wait-event wrong: i = %d, expected = %f | actual = %f\n",
			     i, expected_start_time, ev[i].start);
		return false;
	}

	if (fabs(expected_wait_dur - (ev[i].end - ev[i].start)) > UT_TOL) {

		debug_printf(DP_INFO, "Duration of wait-event wrong: i = %d, expected = %f | actual = %f\n",
			     i, expected_wait_dur, ev[i].end - ev[i].start);
		return false;
	}

	return true;
}

static bool assert_single_wait_event(int E, struct seq_event* ev, double expected_wait_dur)
{
	if (E != 1) {

		debug_printf(DP_INFO, "Wrong number of wait-events: expected: 1 | actual: %d \n", E);
		return false;
	}

	if (!assert_wait_event(0, ev, expected_wait_dur, 0))
		return false;

	return true;
}

static bool assert_asl_condition_event_block(
	int i,
	struct seq_event* ev,
	struct seq_config* seq,
	double expected_grad_rew_amp,
	double expected_rf_freq,
	double expected_phase_shift,
	double expected_phase_shift_neg,
	int expected_hann_pulse_shape_id)
{
	if (i != 5) {

		debug_printf(DP_INFO, "Wrong number of ASL-events: expected: 5 | actual: %d", i);
		return false;
	}

	// Assert slice selection gradient events
	if (!assert_gradient_event(ev, 0, seq->asl.ampl_grad_sli))
		return false;
	if (!assert_gradient_event(ev, 1, seq->asl.ampl_grad_sli))
		return false;

	if (0 != ev[0].start) {

		debug_printf(DP_INFO, "Start of slice-selection gradient wrong: i = %d, expected = 0 | actual = %f\n",
			     0, ev[0].start);
		return false;
	}

	// Assert hanning pulse event
	if (!assert_pulse_event(2, ev, expected_hann_pulse_shape_id, seq->asl.hanning.rf_duration, expected_rf_freq, expected_phase_shift, expected_phase_shift_neg))
		return false;
	if (!assert_slice_selection(0, 2, ev, seq->asl.hanning.rf_duration))
		return false;

	// Assert gradient rewinder events
	if (!assert_gradient_event(ev, 3, expected_grad_rew_amp))
		return false;
	if (!assert_gradient_event(ev, 4, expected_grad_rew_amp))
		return false;

	if (ev[1].end != ev[3].start) {

		debug_printf(DP_INFO, "A gap has been detected between the two consecutive gradients.\n");
		return false;
	}

	if (seq->asl.pulse_spacing != ev[4].end - ev[0].start) {

		debug_printf(DP_INFO, "Pulse spacing wrong: expected = %f | actual = %f \n",
			     seq->asl.pulse_spacing, ev[4].end - ev[0].start);
		return false;
	}

	return true;
}

static bool assert_asl_seq_events(
	struct seq_state* seq_state,
	struct seq_config* seq,
	int expected_num_loop_dims,
	double expected_grad_rew_ampl,
	double expected_rf_freq,
	double* expected_phase_shift,
	double* expected_phase_shift_neg)
{
	seq->loop_dims[COEFF2_DIM] = calc_asl_coeff2_dim(seq);
	if (seq->loop_dims[COEFF2_DIM] != expected_num_loop_dims) {

		debug_printf(DP_INFO, "Wrong number of loop dims: expected: %d | actual: %ld\n",
			     expected_num_loop_dims, seq->loop_dims[COEFF2_DIM]);
		return false;
	}

	int num_hann_pulses = calc_num_asl_pulses(seq->asl.ld, seq->asl.pulse_spacing);
	int expected_num_hann_pulses = seq->loop_dims[COEFF2_DIM] - 1;
	if (num_hann_pulses != expected_num_hann_pulses) {

		debug_printf(DP_INFO, "Wrong number of hanning pulses: expected: %d | actual: %d\n",
			     expected_num_hann_pulses, num_hann_pulses);
		return false;
	}

	int max_events = 1000;
	struct seq_event ev[max_events];

	seq->loop_dims[COEFF2_DIM] = seq->loop_dims[COEFF2_DIM] + coeff2_dim_offset;

	do
	{
		if ((seq_state->pos[COEFF2_DIM] - coeff2_dim_offset) < coeff2_dim_offset)
			continue;

		int i = asl(max_events, ev, seq_state, seq);

		// Assert PLD event
		if (((seq_state->pos[COEFF2_DIM] - coeff2_dim_offset)) == expected_num_loop_dims - 1) {

			if (!assert_single_wait_event(i, ev, seq->asl.pld))
				return false;
			return true;
		}

		if (!assert_asl_condition_event_block(i, ev, seq, expected_grad_rew_ampl,
						      expected_rf_freq,
						      expected_phase_shift[(seq_state->pos[COEFF2_DIM] - coeff2_dim_offset)],
						      expected_phase_shift_neg[(seq_state->pos[COEFF2_DIM] - coeff2_dim_offset)],
						      PCASL_HANN_PULSE))
			return false;

	} while (md_next(DIMS, seq->loop_dims, COEFF2_FLAG, seq_state->pos));

	return true;
}

static void init_expected_phase_shift(double* expected_phase_shift, double val1, double val2, int max)
{
	for (int i = 0; i < max; i++) {

		if (i % 2 == 0)
			expected_phase_shift[i] = val1;
		else
			expected_phase_shift[i] = val2;
	}
}

// * ---------------------------------------------------------------------- *
// * ASL tests								    *
// * ---------------------------------------------------------------------- *

static bool test_asl_m0_image(void)
{
	struct seq_state seq_state = { };

	struct seq_config seq = seq_config_defaults_flash;
	seq.asl.ld = 0.02;
	seq.asl.pulse_spacing = 1.15E-3;

	seq.loop_dims[COEFF2_DIM] = calc_asl_coeff2_dim(&seq);

	if (seq.loop_dims[COEFF2_DIM] != 18)
		return false;

	int i = 10;
	struct seq_event ev[i];

	i = asl(i, ev, &seq_state, &seq);

	if (0 != i)
		return false;

	return true;
}

static bool test_asl_label_condition(void)
{
	long pos[DIMS] = { 0 };
	pos[15] = 1;

	struct seq_state seq_state = { };
	for (int i = 0; i < DIMS; i++)
		seq_state.pos[i] = pos[i];

	struct seq_config seq = seq_config_defaults_flash;
	seq.asl.label_type = SEQ_ASL_PCASL;
	seq.asl.ld = 0.02;
	seq.asl.pld = 1.8;
	seq.asl.pulse_spacing = 1.15E-3;
	seq.asl.ampl_grad_sli = 10E-3;

	seq.asl.label_slice_index = 0;
	seq.geom.shift[seq.asl.label_slice_index][2] = 0;

	int expected_num_loop_dims = 18;
	double expected_grad_rew_ampl = -18.537860E-3;

	double expected_rf_freq = seq.sys.gamma * seq.geom.shift[seq.asl.label_slice_index][2] * seq.asl.ampl_grad_sli;
	double expected_phase_shift[17] = { 0 };
	return assert_asl_seq_events(&seq_state, &seq, expected_num_loop_dims, expected_grad_rew_ampl, expected_rf_freq, expected_phase_shift, expected_phase_shift);
}

static bool test_asl_label_condition_with_slice_shift(void)
{
	struct seq_state seq_state = { };
	seq_state.pos[15] = 1;

	struct seq_config seq = seq_config_defaults_flash;
	seq.asl.label_type = SEQ_ASL_PCASL;
	seq.asl.ld = 0.02;
	seq.asl.pld = 1.8;
	seq.asl.pulse_spacing = 1.15E-3;
	seq.asl.ampl_grad_sli = 10E-3;

	seq.asl.label_slice_index = 1;
	seq.geom.shift[seq.asl.label_slice_index][2] = 50E-3;

	int expected_num_loop_dims = 18;
	double expected_grad_rew_ampl = -18.537860E-3;

	double expected_rf_freq = seq.sys.gamma * seq.geom.shift[seq.asl.label_slice_index][2] * seq.asl.ampl_grad_sli;
	double expected_phase_shift[17] = { -115.900875, 45.413527, -153.272070, 8.042332, 169.356735, -29.328863, 131.985540, -66.700057, 94.614345, -104.071252, 57.243150, -141.442448, 19.871955, -178.813643, -17.499240, 143.815163, -54.870435 };
	double expected_phase_shift_neg[17] = { -115.900875, 82.784722, -78.529680, 120.155917, -41.158485, 157.527113, -3.787290, -165.101693, 33.583905, -127.730498, 70.955100, -90.359302, 108.326295, -52.988108, 145.697490, -15.616913, -176.931315 };

	return assert_asl_seq_events(&seq_state, &seq, expected_num_loop_dims, expected_grad_rew_ampl, expected_rf_freq, expected_phase_shift, expected_phase_shift_neg);
}

static bool test_asl_control_condition(void)
{
	struct seq_state seq_state = { };
	seq_state.pos[15] = 2;

	struct seq_config seq = seq_config_defaults_flash;
	seq.asl.label_type = SEQ_ASL_PCASL;
	seq.asl.ld = 0.02;
	seq.asl.pld = 1.8;
	seq.asl.pulse_spacing = 1.15E-3;
	seq.asl.ampl_grad_sli = 10E-3;

	seq.asl.label_slice_index = 0;
	seq.geom.shift[seq.asl.label_slice_index][2] = 0;

	int expected_num_loop_dims = 18;
	double expected_grad_rew_ampl = -22.933435E-3;

	double expected_rf_freq = seq.sys.gamma * seq.geom.shift[seq.asl.label_slice_index][2] * seq.asl.ampl_grad_sli;

	double expected_phase_shift[17];
	double expected_phase_shift_neg[17];
	init_expected_phase_shift(expected_phase_shift, 0, 180, expected_num_loop_dims - 1);
	init_expected_phase_shift(expected_phase_shift_neg, 0, -180, expected_num_loop_dims - 1);

	return assert_asl_seq_events(&seq_state, &seq, expected_num_loop_dims, expected_grad_rew_ampl, expected_rf_freq, expected_phase_shift, expected_phase_shift_neg);
}

static bool test_asl_control_condition_with_slice_shift(void)
{
	struct seq_state seq_state = { };
	seq_state.pos[15] = 2;

	struct seq_config seq = seq_config_defaults_flash;
	seq.enc.order = SEQ_ORDER_SEQ_ASL;
	seq.asl.label_type = SEQ_ASL_PCASL;
	seq.asl.ld = 0.02;
	seq.asl.pld = 1.8;
	seq.asl.pulse_spacing = 1.15E-3;
	seq.asl.ampl_grad_sli = 10E-3;

	seq.asl.label_slice_index = 0;
	seq.geom.shift[seq.asl.label_slice_index][2] = 50E-3;

	int expected_num_loop_dims = 18;
	double expected_grad_rew_ampl = -22.933435E-3;

	double expected_rf_freq = seq.sys.gamma * seq.geom.shift[seq.asl.label_slice_index][2] * seq.asl.ampl_grad_sli;

	double expected_phase_shift[17];
	init_expected_phase_shift(expected_phase_shift, -115.900875, 64.099125, expected_num_loop_dims - 1);

	return assert_asl_seq_events(&seq_state, &seq, expected_num_loop_dims, expected_grad_rew_ampl, expected_rf_freq, expected_phase_shift, expected_phase_shift);
}

UT_REGISTER_TEST(test_asl_m0_image);
UT_REGISTER_TEST(test_asl_label_condition);
UT_REGISTER_TEST(test_asl_label_condition_with_slice_shift);
UT_REGISTER_TEST(test_asl_control_condition);
UT_REGISTER_TEST(test_asl_control_condition_with_slice_shift);