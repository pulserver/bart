/* Copyright 2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>

#include "num/multind.h"

#include "misc/mri.h"
#include "misc/misc.h"

#include "seq/config.h"
#include "seq/event.h"
#include "seq/adc_rf.h"
#include "seq/gradient.h"
#include "seq/anglecalc.h"
#include "seq/misc.h"
#include "seq/pulse.h"
#include "seq/ui_enums.h"

#include "miniflash.h"


static double start_rf(const struct seq_config* seq)
{
	return round_up_raster(MAX(slice_amplitude(seq) * seq->sys.grad.inv_slew_rate, seq->sys.coil_control_lead), seq->sys.raster_rf);
}

static double start_adc(const struct seq_config* seq)
{
	return round_up_raster(start_rf(seq) + seq->phys.rf_duration / 2. + seq->phys.te - adc_time_to_echo(0, seq), seq->sys.raster_rf);
}

static double ro_shift(const struct seq_config* seq)
{
	double adc_start = start_adc(seq);

	return seq->sys.raster_grad - (round_up_raster(adc_start, seq->sys.raster_grad) - adc_start);
}


static double available_time_RF_SLI(const struct seq_config* seq)
{
	return seq->phys.te - seq->phys.rf_duration / 2.
		- slice_amplitude(seq) * seq->sys.grad.inv_slew_rate
		- seq->sys.grad.max_amplitude * seq->sys.grad.inv_slew_rate
		- round_up_raster(adc_time_to_echo(0, seq) - 0.99 * seq->sys.raster_rf, seq->sys.raster_rf) // round down
		- ro_shift(seq);
}

static double ro_time_to_echo(const struct seq_config* seq)
{
	return ro_shift(seq) + adc_time_to_echo(0, seq);
}

static double ro_time_after_echo(const struct seq_config* seq)
{
	return round_up_raster(adc_duration(seq) + ro_shift(seq), seq->sys.raster_grad) 
		- ro_time_to_echo(seq);
}


static double ro_momentum_to_echo(const struct seq_config* seq)
{
	double amp = ro_amplitude(seq);

	return amp *
		(0.5 * seq->sys.grad.max_amplitude * seq->sys.grad.inv_slew_rate
		+ ro_shift(seq) + adc_time_to_echo(0, seq));
}

static double ro_momentum(const struct seq_config* seq)
{
	double amp = ro_amplitude(seq);

	return amp * (seq->sys.grad.max_amplitude * seq->sys.grad.inv_slew_rate
		 + round_up_raster(adc_duration(seq) + ro_shift(seq), seq->sys.raster_grad));
}

static double ro_momentum_after_echo(const struct seq_config* seq)
{
	return ro_momentum(seq) - ro_momentum_to_echo(seq);
}

static int prep_grad_ro_deph(struct grad_trapezoid* grad, const struct seq_config* seq)
{
	if (!grad_soft(grad, available_time_RF_SLI(seq), -ro_momentum_to_echo(seq), seq->sys.grad))
		return 0;

	return 1;
}

static int prep_grad_phs1_encoding(struct grad_trapezoid* grad, int rew, const long pos[DIMS], const struct seq_config* seq)
{
	*grad = (struct grad_trapezoid){ };

	long center = 0.5 * seq->loop_dims[PHS1_DIM];

	double moment = (cartesian_line(pos, seq) - center) / (seq->sys.gamma * seq->geom.fov);

	if (rew)
		moment = -1. * moment;

	if (!grad_soft(grad, available_time_RF_SLI(seq), moment, seq->sys.grad))
		return 0;

	return 1;
}


static int prep_grad_ro(struct grad_trapezoid* grad, const struct seq_config* seq)
{
	*grad = (struct grad_trapezoid){ };

	double ampl = ro_amplitude(seq);

	if (seq->sys.grad.max_amplitude < ampl)
		return 0;

	grad->rampup = seq->sys.grad.max_amplitude * seq->sys.grad.inv_slew_rate;
	grad->flat = round_up_raster(adc_duration(seq) + ro_shift(seq), seq->sys.raster_grad);
	grad->rampdown = seq->sys.grad.max_amplitude * seq->sys.grad.inv_slew_rate;
	grad->ampl = ampl;

	return 1;
}

static double end_last_ro(int rampdown, const struct seq_config* seq)
{
	double rdt = 0.;

	if (rampdown)
		rdt = ro_amplitude(seq) * seq->sys.grad.inv_slew_rate;

	return start_rf(seq) + seq->phys.rf_duration / 2. + seq->phys.te
		+ ro_time_after_echo(seq) + rdt;
}


static int prep_grad_spoiler_read(struct grad_trapezoid* grad, const struct seq_config* seq)
{
	*grad = (struct grad_trapezoid){ };

	struct grad_limits lim = seq->sys.grad;
	lim.inv_slew_rate = seq->sys.grad.inv_slew_rate * 2;

	if (!grad_soft(grad, seq->phys.tr - end_last_ro(1, seq), ro_momentum(seq), lim))
		return 0;

	return 1;
}

static int prep_grad_spoiler_slice(struct grad_trapezoid* grad, const struct seq_config* seq)
{
	*grad = (struct grad_trapezoid){ };

	if (!grad_soft(grad, seq->phys.tr - end_last_ro(1, seq), slice_momentum_to_rephase(seq), seq->sys.grad))
		return 0;

	return 1;
}

static int prep_grad_sli(struct grad_trapezoid* grad, const struct seq_config* seq)
{
	*grad = (struct grad_trapezoid){ };

	double ampl = slice_amplitude(seq);
	double ramp = ampl * seq->sys.grad.inv_slew_rate;

	if (seq->sys.grad.max_amplitude < ampl)
		return 0;

	grad->rampup = round_up_raster(MAX(ramp, seq->sys.coil_control_lead), seq->sys.raster_rf); //round_up for start of rf pulse
	grad->flat = seq->phys.rf_duration;
	grad->rampdown = ramp;
	grad->ampl = ampl;

	return 1;
}


static int prep_grad_sli_reph(struct grad_trapezoid* grad, const struct seq_config* seq)
{
	*grad = (struct grad_trapezoid){ };

	if (!grad_soft(grad, available_time_RF_SLI(seq), -slice_momentum_to_rephase(seq), seq->sys.grad))
		return 0;

	return 1;
}


void miniflash_interface_custom(struct seq_config* seq,
				int nl, const long custom_long[__VLA(nl)],
				int nd, const double custom_double[__VLA(nd)])
{
	seq->enc.pe_mode = custom_long[SEQ_UI_IDX_LONG_PE_MODE];
	seq->phys.contrast = custom_long[SEQ_UI_IDX_LONG_CONTRAST];

	seq->phys.os = 2.;
	seq->phys.rf_duration = 1E-6 * custom_long[SEQ_UI_IDX_LONG_RF_DURATION_US];
	seq->phys.bwtp = custom_double[SEQ_UI_IDX_DOUBLE_BWTP];
}


void miniflash_interface_custom_back(const struct seq_config* seq,
				     int nl, long custom_long[__VLA(nl)],
				     int nd, double custom_double[__VLA(nd)])
{
	custom_long[SEQ_UI_IDX_LONG_PE_MODE] = seq->enc.pe_mode;;
	custom_long[SEQ_UI_IDX_LONG_CONTRAST] = seq->phys.contrast;
	custom_long[SEQ_UI_IDX_LONG_RECO] = CHECKBOX_OFF;

	custom_long[SEQ_UI_IDX_LONG_RF_DURATION_US] = lround(1.E6 * seq->phys.rf_duration);
	custom_double[SEQ_UI_IDX_DOUBLE_BWTP] = seq->phys.bwtp;
}


double miniflash_minimum_tr(const struct seq_config* seq)
{
	double mom_read = ro_momentum(seq) + ro_momentum_after_echo(seq);
	double mom_slice = slice_momentum_to_rephase(seq);


	struct grad_trapezoid grad;
	grad_hard(&grad, mom_read + mom_slice, seq->sys.grad);

	double time_gradients_after_RO = seq->sys.grad.max_amplitude * seq->sys.grad.inv_slew_rate + grad_total_time(&grad);

	// we have to check for minimum timings between rf and next adc
	double add_time_ro_rf = MAX(MAX(seq->sys.min_duration_ro_rf, seq->sys.coil_control_lead) 
					- (time_gradients_after_RO + slice_amplitude(seq) * seq->sys.grad.inv_slew_rate), 0.);

	prep_grad_ro(&grad, seq);

	double last_ro_start = start_rf(seq) + seq->phys.rf_duration + available_time_RF_SLI(seq);

	return round_up_raster(last_ro_start + grad_total_time(&grad) + time_gradients_after_RO + add_time_ro_rf, seq->sys.raster_grad);
}


void miniflash_minimum_te(const struct seq_config* seq, double* min_te, double* fill_te)
{
	double ro_deph_time = available_time_RF_SLI(seq);
	double inter_duration_READ = MAX(ro_deph_time, seq->sys.grad.max_amplitude * seq->sys.grad.inv_slew_rate);

	double ro_amp = ro_amplitude(seq); //FIXME
	double sl_amp = slice_amplitude(seq);

	double inter_duration_SLICE = available_time_RF_SLI(seq)
		+ sl_amp * seq->sys.grad.inv_slew_rate;

	double inter_duration_RF_RO = MAX(inter_duration_READ, inter_duration_SLICE);

	double time = seq->phys.rf_duration / 2. + inter_duration_RF_RO - seq->sys.grad.max_amplitude * seq->sys.grad.inv_slew_rate;

	long echo = 0;

	time += ro_amp * seq->sys.grad.inv_slew_rate; //FIXME
	time += ro_time_to_echo(seq);
	time = round_up_raster(time, seq->sys.raster_grad) - seq->sys.raster_grad;

	min_te[echo] = time;

	time += ro_time_after_echo(seq);
	time += ro_amp * seq->sys.grad.inv_slew_rate; //FIXME

	time = 0;
	fill_te[0] = seq->phys.te - min_te[0];
}

static long miniflash_ex_calls(const struct seq_config* seq)
{
	long dims[DIMS];
	md_select_dims(DIMS, SEQ_FLAGS & ~(COEFF_FLAG|COEFF2_FLAG), dims, seq->loop_dims);

	return md_calc_size(DIMS, dims);
}

double miniflash_total_measure_time(const struct seq_config* seq)
{
	return seq->phys.tr * miniflash_ex_calls(seq);
}

int miniflash_sample_rf_shapes(int N, struct rf_shape pulse[N], const struct seq_config* seq)
{
	int idx = 0;

	for (; idx < seq->geom.mb_factor; idx++) {

		if (idx >= N)
			return -1;

		pulse[idx].sar_calls = miniflash_ex_calls(seq);
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

	return idx;
}

struct miniflash_timing {

	double slice;
	double RF;
	double slice_rephaser;
	double readout_dephaser;
	double readout;
	double adc;
	double spoiler;
};


static struct miniflash_timing miniflash_compute_timing(const struct seq_config *seq)
{
	struct miniflash_timing timing;

	timing.slice = 0.;
	timing.RF = start_rf(seq);

	timing.slice_rephaser = timing.RF + seq->phys.rf_duration + seq->sys.grad.inv_slew_rate * slice_amplitude(seq);
	timing.readout_dephaser = timing.slice_rephaser;

	timing.readout = timing.readout_dephaser + available_time_RF_SLI(seq);
	timing.adc = start_adc(seq);

	timing.spoiler = end_last_ro(1, seq);

	return timing;
}


int miniflash(int N, struct seq_event ev[N], struct seq_state* seq_state, const struct seq_config* seq)
{
	struct miniflash_timing timing = miniflash_compute_timing(seq);

	int i = 0;

	double rf_spoil_phase = rf_spoiling(DIMS, seq_state->pos, seq);

	double projPHASE[3] = { 0. , 1. , 0 };
	double projREAD[3] = { 1. , 0. , 0. };
	double projSLICE[3] = { 0. , 0. , 1. };


	struct grad_trapezoid slice;

	if (!prep_grad_sli(&slice, seq))
		return ERROR_PREP_GRAD_SLI;

	i += seq_grad_to_event(ev + i, timing.slice, &slice, projSLICE);
	i += prep_rf_excitation(ev + i, timing.RF, rf_spoil_phase, seq_state, seq);

	struct grad_trapezoid slice_rephaser;

	if (!prep_grad_sli_reph(&slice_rephaser, seq))
		return ERROR_PREP_GRAD_SLI_REPH;

	if ((grad_total_time(&slice) - 1.E-9) > timing.slice_rephaser)
		return ERROR_SLI_TIMING;

	i += seq_grad_to_event(ev + i, timing.slice_rephaser, &slice_rephaser, projSLICE);

	struct grad_trapezoid readout_dephaser;

	if (!prep_grad_ro_deph(&readout_dephaser, seq))
		return ERROR_PREP_GRAD_RO_DEPH;

	i += seq_grad_to_event(ev + i, timing.readout_dephaser, &readout_dephaser, projREAD);


	struct grad_trapezoid phs1_encoding;

	if (!prep_grad_phs1_encoding(&phs1_encoding, 0, seq_state->pos, seq))
		return ERROR_PREP_GRAD_RO_DEPH;

	//check for overlapping gradients!
	if (powf(seq->sys.grad.max_amplitude, 2.) < (  powf(slice_rephaser.ampl, 2.)
						     + powf(readout_dephaser.ampl, 2.)
						     + powf(phs1_encoding.ampl, 2.)))
		return ERROR_MAX_GRAD_RO_SLI;

	i += seq_grad_to_event(ev + i, timing.readout_dephaser, &phs1_encoding, projPHASE);

	if ((timing.readout_dephaser + grad_total_time(&readout_dephaser) - 1.E-9) > timing.readout)
		return ERROR_RO_TIMING;

	if ((timing.readout_dephaser + grad_total_time(&phs1_encoding) - 1.e-3) > timing.readout)
		return ERROR_RO_TIMING;


	struct grad_trapezoid readout;

	if (!prep_grad_ro(&readout, seq))
		return ERROR_PREP_GRAD_RO_RO;

	i += seq_grad_to_event(ev + i, timing.readout, &readout, projREAD);
	i += prep_adc(ev + i, timing.adc, rf_spoil_phase, seq_state, seq);


	struct grad_trapezoid phase_rewinder;

	if (!prep_grad_phs1_encoding(&phase_rewinder, 1, seq_state->pos, seq))
		return ERROR_PREP_GRAD_SP_READ;

	i += seq_grad_to_event(ev + i, timing.spoiler, &phase_rewinder, projPHASE);

	struct grad_trapezoid spoiler_read;

	if (!prep_grad_spoiler_read(&spoiler_read, seq))
		return ERROR_PREP_GRAD_SP_READ;

	i += seq_grad_to_event(ev + i, timing.spoiler, &spoiler_read, projREAD);

	struct grad_trapezoid spoiler_slice;

	if (!prep_grad_spoiler_slice(&spoiler_slice, seq))
		return ERROR_PREP_GRAD_SP_SLICE;

	if (powf(seq->sys.grad.max_amplitude, 2.) < powf(spoiler_read.ampl, 2.) + powf(spoiler_slice.ampl, 2.))
		return ERROR_MAX_GRAD_SPOILER;

	i += seq_grad_to_event(ev + i, timing.spoiler, &spoiler_slice, projSLICE);

	if (seq_block_end_flat(i, ev, seq->sys.raster_grad) - 1E-9 > seq->phys.tr)
		return ERROR_END_FLAT_KERNEL;

	return i;
}


static int check_settings(const struct seq_state* seq_state, const struct seq_config* seq)
{
	if (   (SEQ_CONTRAST_RF_SPOILED != seq->phys.contrast)
	    && (SEQ_CONTRAST_NO_SPOILING != seq->phys.contrast))
		return ERROR_SETTING_CONTRAST;

	if (   (SEQ_PEMODE_CARTESIAN != seq->enc.pe_mode)
	    && (SEQ_PEMODE_CARTESIAN_LINEAR != seq->enc.pe_mode))
		return ERROR_SETTING_PEMODE;

	if (0.5 != seq->phys.asym_echo)
		return ERROR_SETTING_ASYM_ECHO;

	if (SEQ_ORDER_AVG_OUTER != seq->enc.order)
		return ERROR_SETTING_ORDER;

	if ((1 < seq->geom.mb_factor) && seq->enc.is3D)
		return ERROR_SETTING_DIM;

	if (1 != seq->loop_dims[PHS2_DIM])
		return ERROR_SETTING_DIM;

	if (1 != seq->loop_dims[TE_DIM])
		return ERROR_SETTING_DIM;

	if (1 != seq->loop_dims[SLICE_DIM])
		return ERROR_SETTING_DIM;

	if (SEQ_PREP_OFF != seq->magn.mag_prep)
		return ERROR_MAG_PREP;

	if (SEQ_TRIGGER_OFF != seq->trigger.type)
		return ERROR_TRIGGER;

	if (SEQ_CONTEXT_BINARY != seq_state->context) {

		if (   (SEQ_PEMODE_CARTESIAN == seq->enc.pe_mode)
		    || (SEQ_PEMODE_CARTESIAN_LINEAR == seq->enc.pe_mode))
			return 1;
	}

	return 1;
}

int miniflash_block(int N, struct seq_event ev[N], struct seq_state* seq_state, const struct seq_config* seq)
{
	int err = check_settings(seq_state, seq);

	if (1 > err)
		return err;

	seq_state->chrono_slice = seq_state->pos[SLICE_DIM];

	if (   (SEQ_BLOCK_KERNEL_PREPARE == seq_state->mode)
	    || (SEQ_BLOCK_KERNEL_CHECK == seq_state->mode))
		return miniflash(N, ev, seq_state, seq);

	seq_state->mode = SEQ_BLOCK_KERNEL_IMAGE;

	return miniflash(N, ev, seq_state, seq);
}

