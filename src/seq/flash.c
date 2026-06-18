/* Copyright 2025-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>

#include "num/multind.h"

#include "misc/mri.h"
#include "misc/misc.h"

#include "seq/config.h"
#include "seq/event.h"
#include "seq/anglecalc.h"
#include "seq/adc_rf.h"
#include "seq/gradient.h"
#include "seq/misc.h"
#include "seq/seq.h"

#include "flash.h"

// for time-optimized overlapping gradients, otherwise sqrt(2.)
#define SCALE_GRAD 0.82


int prep_grad_ro(struct grad_trapezoid* grad, long echo, const struct seq_config* seq);


static double start_rf(const struct seq_config* seq)
{
	double sli_ampl = slice_amplitude(seq);
	double min_delay = MAX(sli_ampl * seq->sys.grad.inv_slew_rate, seq->sys.coil_control_lead);

	return round_up_raster(min_delay, seq->sys.raster_rf);
}

static double start_adc(long echo, const struct seq_config* seq)
{
	return round_up_raster(start_rf(seq) + seq->phys.rf_duration / 2. + seq->phys.te + echo * seq->phys.te_delta
				- adc_time_to_echo(echo, seq), seq->sys.raster_rf);
}

static double ro_shift(long echo, const struct seq_config* seq)
{
	double adc_start = start_adc(echo, seq);

	double shift = seq->sys.raster_grad - (round_up_raster(adc_start, seq->sys.raster_grad) - adc_start);

	if (seq->sys.raster_grad <= shift)
		return 0.;

	return shift;
}


static double available_time_RF_SLI(int ro, const struct seq_config* seq)
{
	double ampl = ro ? ro_amplitude(seq) : slice_amplitude(seq);

	return seq->phys.te - seq->phys.rf_duration / 2.
		- ampl * seq->sys.grad.inv_slew_rate
		- round_up_raster(adc_time_to_echo(0, seq) - 0.99 * seq->sys.raster_rf, seq->sys.raster_rf) // round down
		- ro_shift(0, seq);
}

static double ro_time_to_echo(long echo, const struct seq_config* seq)
{
	return ro_shift(echo, seq) + adc_time_to_echo(echo, seq);
}

static double ro_time_after_echo(long echo, const struct seq_config* seq)
{
	return round_up_raster(adc_duration(seq) + ro_shift(echo, seq), seq->sys.raster_grad) 
		- ro_time_to_echo(echo, seq);
}


static double ro_momentum_to_echo(long echo, const struct seq_config* seq)
{
	double amp = ro_amplitude(seq);

	return amp *
		(0.5 * amp * seq->sys.grad.inv_slew_rate
		+ ro_shift(echo, seq) + adc_time_to_echo(echo, seq));
}

static double ro_momentum(long echo, const struct seq_config* seq)
{
	double amp = ro_amplitude(seq);

	return amp * (amp * seq->sys.grad.inv_slew_rate
		 + round_up_raster(adc_duration(seq) + ro_shift(echo, seq), seq->sys.raster_grad));
}

static double ro_momentum_after_echo(long echo, const struct seq_config* seq)
{
	return ro_momentum(echo, seq) - ro_momentum_to_echo(echo, seq);
}

static double ro_blip_angle(const long pos[DIMS], const struct seq_config* seq)
{
	if (0 < pos[TE_DIM]) {

		double angle_curr = get_rot_angle(pos, seq);
		double moment_curr = ro_momentum_to_echo(pos[TE_DIM], seq);

		long pos2[DIMS];
		md_copy_dims(DIMS, pos2, pos);
		pos2[TE_DIM] = pos[TE_DIM] - 1;

		double angle_prev = get_rot_angle(pos2, seq);
		double moment_prev = ro_momentum_after_echo(pos2[TE_DIM], seq);

		double blip_x = -fabs(moment_curr) * cos(angle_curr) - fabs(moment_prev) * cos(angle_prev);
		double blip_y = -fabs(moment_curr) * sin(angle_curr) - fabs(moment_prev) * sin(angle_prev);

		return atan2(blip_y, blip_x);
	}

	return 0.;
}

static double ro_blip_moment(const long pos[DIMS], const struct seq_config* seq)
{
	if (0 < pos[TE_DIM]) {

		double angle_curr = get_rot_angle(pos, seq);
		double moment_curr = ro_momentum_to_echo(pos[TE_DIM], seq);

		long pos2[DIMS];
		md_copy_dims(DIMS, pos2, pos);
		pos2[TE_DIM] = pos[TE_DIM] - 1;

		double angle_prev = get_rot_angle(pos2, seq);
		double moment_prev = ro_momentum_after_echo(pos2[TE_DIM], seq);

		double blip_x = -fabs(moment_curr) * cos(angle_curr) - fabs(moment_prev) * cos(angle_prev);
		double blip_y = -fabs(moment_curr) * sin(angle_curr) - fabs(moment_prev) * sin(angle_prev);

		return sqrt(pow(blip_x, 2.) + pow(blip_y, 2.));
	}

	return 0.;
}

static int prep_grad_ro_deph(struct grad_trapezoid* grad, const struct seq_config* seq)
{
	const long echo = 0;

	struct grad_limits limits = seq->sys.grad;
	limits.max_amplitude *= SCALE_GRAD;

	if (!grad_soft(grad, available_time_RF_SLI(1, seq), -ro_momentum_to_echo(echo, seq), limits))
		return 0;

	return 1;
}

static int prep_grad_phs1_encoding(struct grad_trapezoid* grad, int rew, const long pos[DIMS], const struct seq_config* seq)
{
	*grad = (struct grad_trapezoid){ 0 };

	if (!((SEQ_PEMODE_CARTESIAN == seq->enc.pe_mode) || (SEQ_PEMODE_CARTESIAN_LINEAR == seq->enc.pe_mode)))
		return 1;

	if (rew && (SEQ_CONTRAST_RF_SPOILED != seq->phys.contrast))
		return 1;

	struct grad_limits limits = seq->sys.grad;
	limits.max_amplitude *= SCALE_GRAD;

	long center = 0.5 * seq->loop_dims[PHS1_DIM];

	double moment = (cartesian_line(pos, seq) - center) / (seq->sys.gamma * seq->geom.fov);

	if (rew)
		moment = -1. * moment;

	if (!grad_soft(grad, available_time_RF_SLI(1, seq), moment, limits))
		return 0;

	// FIXME maybe similar to GSTF with fixed timing

	return 1;
}


static int prep_grad_ro_blip(struct grad_trapezoid* grad, long echo, const struct seq_config* seq)
{
	*grad = (struct grad_trapezoid){ 0 };

	if (0 < echo) {

		long pos0[DIMS] = { [TE_DIM] = echo };
		pos0[TE_DIM] = echo;

		double moment = ro_blip_moment(pos0, seq);

		grad_hard(grad, moment, seq->sys.grad);
	}

	return 1;
}


int prep_grad_ro(struct grad_trapezoid* grad, long echo, const struct seq_config* seq)
{
	*grad = (struct grad_trapezoid){ 0 };

	double ampl = ro_amplitude(seq);

	if (seq->sys.grad.max_amplitude < ampl)
		return 0;

	grad->rampup = ampl * seq->sys.grad.inv_slew_rate;
	grad->flat = round_up_raster(adc_duration(seq) + ro_shift(echo, seq), seq->sys.raster_grad);
	grad->rampdown = ampl * seq->sys.grad.inv_slew_rate;
	grad->ampl = ampl;

	return 1;
}


static double slice_momentum_to_rephase(const struct seq_config* seq)
{
	double amp = slice_amplitude(seq);
	return amp * (0.5 * seq->phys.rf_duration + 0.5 * amp * seq->sys.grad.inv_slew_rate);
}


static double end_last_ro(int rampdown, const struct seq_config* seq)
{
	double rdt = 0;
	if (rampdown)
		rdt = ro_amplitude(seq) * seq->sys.grad.inv_slew_rate;

	return start_rf(seq) + seq->phys.rf_duration / 2. + seq->phys.te
		+ seq->phys.te_delta * (seq->loop_dims[TE_DIM] - 1)
		+ ro_time_after_echo(seq->loop_dims[TE_DIM] - 1,seq) + rdt;
}


static int prep_grad_ro_reph(struct grad_trapezoid* grad, const struct seq_config* seq)
{
	*grad = (struct grad_trapezoid){ 0 };

	if (SEQ_CONTRAST_RF_SPOILED != seq->phys.contrast)
		return 1;

	struct grad_limits lim = seq->sys.grad;
	lim.inv_slew_rate = seq->sys.grad.inv_slew_rate * 2;

	if (!grad_soft(grad, seq->phys.tr - end_last_ro(1, seq),
			- ro_momentum_after_echo(seq->loop_dims[TE_DIM] - 1, seq), lim))
		return 0;

	return 1;
}

static int prep_grad_spoiler_read(struct grad_trapezoid* grad, const struct seq_config* seq)
{
	*grad = (struct grad_trapezoid){ 0 };

	if (SEQ_CONTRAST_RF_SPOILED != seq->phys.contrast)
		return 1;

	struct grad_limits lim = seq->sys.grad;
	lim.inv_slew_rate = seq->sys.grad.inv_slew_rate * 2;

	if (!grad_soft(grad, seq->phys.tr - end_last_ro(1, seq),
			ro_momentum(seq->loop_dims[TE_DIM] - 1, seq), lim))
		return 0;

	return 1;
}

static int prep_grad_spoiler_slice(struct grad_trapezoid* grad, const struct seq_config* seq)
{
	*grad = (struct grad_trapezoid){ 0 };

	if (SEQ_CONTRAST_RF_SPOILED != seq->phys.contrast)
		return 1;

	struct grad_limits lim = seq->sys.grad;
	lim.inv_slew_rate = seq->sys.grad.inv_slew_rate * 2;

	if (!grad_soft(grad, seq->phys.tr - end_last_ro(0, seq), slice_momentum_to_rephase(seq), lim))
		return 0;

	return 1;
}

static int prep_grad_sli(struct grad_trapezoid* grad, const struct seq_config* seq)
{
	*grad = (struct grad_trapezoid){ 0 };

	double ampl = slice_amplitude(seq);

	if (seq->sys.grad.max_amplitude < ampl)
		return 0;

	grad->rampup = round_up_raster(MAX(ampl * seq->sys.grad.inv_slew_rate, seq->sys.coil_control_lead), seq->sys.raster_rf); //round_up for start of rf pulse
	grad->flat = seq->phys.rf_duration;
	grad->rampdown = ampl * seq->sys.grad.inv_slew_rate;
	grad->ampl = ampl;

	return 1;
}


static int prep_grad_sli_reph(struct grad_trapezoid* grad, long pos_phs2, const struct seq_config* seq)
{
	*grad = (struct grad_trapezoid){ 0 };

	struct grad_limits limits = seq->sys.grad;
	limits.max_amplitude *= SCALE_GRAD;

	if (!grad_soft(grad, available_time_RF_SLI(0, seq), -slice_momentum_to_rephase(seq), limits))
		return 0;

	if (seq->enc.is3D) {

		// in 3d mode: slice_thickness = slab_thickness = seq->loop_dims[PHS2_DIM]) * thickness
		double pe_enc = 1. * (pos_phs2 - 0.5 * seq->loop_dims[PHS2_DIM]) / (seq->sys.gamma * seq->geom.slice_thickness * seq->geom.slab_os);
		double moment = - slice_momentum_to_rephase(seq) + pe_enc;

		if (!gradient_prepare_with_timing(grad, moment, seq))
			return 0;
	}

	return 1;
}


static int prep_grad_pe3d_rewinder(struct grad_trapezoid* grad, const long pos[DIMS], const struct seq_config* seq)
{
	*grad = (struct grad_trapezoid){ 0 };

	if (!seq->enc.is3D || (SEQ_CONTRAST_RF_SPOILED != seq->phys.contrast))
		return 1;

	struct grad_trapezoid tmp_sli_reph;
	if (!prep_grad_sli_reph(&tmp_sli_reph, pos[PHS2_DIM], seq))
		return 0;

	grad->rampup = tmp_sli_reph.rampup;
	grad->rampdown = tmp_sli_reph.rampdown;
	grad->flat = tmp_sli_reph.flat;

	// in 3d mode: slice_thickness = slab_thickness = seq->loop_dims[PHS2_DIM]) * thickness
	double pe_enc = 1. * (pos[PHS2_DIM] - 0.5 * seq->loop_dims[PHS2_DIM]) / (seq->sys.gamma * seq->geom.slice_thickness * seq->geom.slab_os);
	double moment = - slice_momentum_to_rephase(seq) - pe_enc;

	if (!gradient_prepare_with_timing(grad, moment, seq))
		return 0;

	return 1;
}


static double gradient_time_after_RO(const struct seq_config* seq)
{
	if (SEQ_CONTRAST_RF_SPOILED != seq->phys.contrast)
		return 0.;


	double mom_read = ro_momentum(seq->loop_dims[TE_DIM] - 1, seq) 
			+ ro_momentum_after_echo(seq->loop_dims[TE_DIM] - 1, seq);
	double mom_slice = slice_momentum_to_rephase(seq);


	struct grad_trapezoid grad;
	grad_hard(&grad, mom_read + mom_slice, seq->sys.grad);

	return grad_total_time(&grad); // FIXME: only approximately valid
}

double min_tr_flash(const struct seq_config* seq)
{
	double time_gradients_after_RO = ro_amplitude(seq) * seq->sys.grad.inv_slew_rate + gradient_time_after_RO(seq);

	// we have to check for minimum timings between rf and next adc
	double add_time_ro_rf = MAX(MAX(seq->sys.min_duration_ro_rf, seq->sys.coil_control_lead) 
					- (time_gradients_after_RO + slice_amplitude(seq) * seq->sys.grad.inv_slew_rate), 0.);

	struct grad_trapezoid last_ro;
	prep_grad_ro(&last_ro, seq->loop_dims[TE_DIM] - 1, seq);

	double last_ro_start = start_rf(seq) + seq->phys.rf_duration + available_time_RF_SLI(1, seq)
				+ (seq->loop_dims[TE_DIM] - 1) * seq->phys.te_delta;

	return round_up_raster(last_ro_start + grad_duration(&last_ro) + gradient_time_after_RO(seq) + add_time_ro_rf, seq->sys.raster_grad);
}


void min_te_flash(const struct seq_config* seq, double* min_te, double* fill_te)
{
	double ro_deph_time = available_time_RF_SLI(1, seq);
	double inter_duration_READ = MAX(ro_deph_time, seq->sys.grad.max_amplitude * seq->sys.grad.inv_slew_rate);

	double ro_amp = ro_amplitude(seq); //FIXME
	double sl_amp = slice_amplitude(seq);

	double inter_duration_SLICE = available_time_RF_SLI(0, seq)
		+ sl_amp * seq->sys.grad.inv_slew_rate;

	double inter_duration_RF_RO = MAX(inter_duration_READ, inter_duration_SLICE);

	double time = seq->phys.rf_duration / 2. + inter_duration_RF_RO - seq->sys.grad.max_amplitude * seq->sys.grad.inv_slew_rate;

	double blip_time = 0.;

	if (1 < seq->loop_dims[TE_DIM]) {

		struct grad_trapezoid grad;
		long pos0[DIMS] = { };
		pos0[TE_DIM] = 1;
		grad_hard(&grad, ro_blip_moment(pos0, seq), seq->sys.grad);

		blip_time = grad_total_time(&grad);
	}

	for (long echo = 0; echo < seq->loop_dims[TE_DIM]; echo++) {

		if (0 < echo)
			time += blip_time;

		time += ro_amp * seq->sys.grad.inv_slew_rate; //FIXME
		time += ro_time_to_echo(echo, seq);
		time = round_up_raster(time, seq->sys.raster_grad) - seq->sys.raster_grad;

		min_te[echo] = time;

		time += ro_time_after_echo(echo, seq);
		time += ro_amp * seq->sys.grad.inv_slew_rate; //FIXME
	}

	time = 0;
	fill_te[0] = seq->phys.te - min_te[0];

	for (long echo = 1; echo < seq->loop_dims[TE_DIM]; echo++) {

		time += fill_te[echo - 1];
		fill_te[echo] = seq->phys.te + echo * seq->phys.te_delta - min_te[echo] - time;
	}
}


struct flash_timing {

	double RF;
	double slice;
	double slice_rephaser;
	double readout_dephaser;
	double readout_blip; // actually MAX_NO_ECHOES
	double readout[SEQ_MAX_NO_ECHOES];
	double adc[SEQ_MAX_NO_ECHOES];
	double readout_rephaser;
	double spoiler_read;
	double spoiler_slice;
	double pe3d_rewinder;
};


static struct flash_timing flash_compute_timing(const struct seq_config *seq)
{
	struct flash_timing timing;

	timing.slice = 0.;
	timing.RF = start_rf(seq);
	timing.readout_dephaser = timing.RF + seq->phys.rf_duration;
	timing.slice_rephaser = timing.readout_dephaser + seq->sys.grad.inv_slew_rate * slice_amplitude(seq);

	timing.readout_blip = -1.; // calc when gradient was prepared

	for (int i = 0; i < seq->loop_dims[TE_DIM]; i++) {

		timing.readout[i] = timing.readout_dephaser + available_time_RF_SLI(1, seq)
				+ i * seq->phys.te_delta; // available_time_RF_SLI only adds first echo, but we need te[echo]

		timing.adc[i] = start_adc(i, seq);
	}

	timing.spoiler_slice = end_last_ro(0, seq);
	timing.readout_rephaser = end_last_ro(1, seq);
	timing.spoiler_read = end_last_ro(1, seq);

	timing.pe3d_rewinder = timing.readout_rephaser;

	return timing;
}


int flash(int N, struct seq_event ev[N], struct seq_state* seq_state, const struct seq_config* seq)
{
	struct flash_timing timing = flash_compute_timing(seq);

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

	if (!prep_grad_sli_reph(&slice_rephaser, seq_state->pos[PHS2_DIM], seq))
		return ERROR_PREP_GRAD_SLI_REPH;

	if ((grad_total_time(&slice) - 1.E-9) > timing.slice_rephaser)
		return ERROR_SLI_TIMING;

	i += seq_grad_to_event(ev + i, timing.slice_rephaser, &slice_rephaser, projSLICE);

	struct grad_trapezoid ro_reph;

	do {
		double proj_angle = get_rot_angle(seq_state->pos, seq);

		double projX[3] = { 0., cos(proj_angle), 0. };
		double projY[3] = { sin(proj_angle), 0., 0. };

		struct grad_trapezoid readout_dephaser;
		struct grad_trapezoid phs1_encoding;

		if (seq_state->pos[TE_DIM] == 0) {

			if (!prep_grad_ro_deph(&readout_dephaser, seq))
				return ERROR_PREP_GRAD_RO_DEPH;

			if (!prep_grad_phs1_encoding(&phs1_encoding, 0, seq_state->pos, seq))
				return ERROR_PREP_GRAD_RO_DEPH;

			//check for overlapping gradients!
			if (powf(seq->sys.grad.max_amplitude, 2.) < (  powf(slice_rephaser.ampl, 2.)
								     + powf(readout_dephaser.ampl, 2.)
								     + powf(phs1_encoding.ampl, 2.)))
				return ERROR_MAX_GRAD_RO_SLI;

			i += seq_grad_to_event(ev + i, timing.readout_dephaser, &readout_dephaser, projX);
			i += seq_grad_to_event(ev + i, timing.readout_dephaser, &readout_dephaser, projY);

			i += seq_grad_to_event(ev + i, timing.readout_dephaser, &phs1_encoding, projPHASE);
		}


		struct grad_trapezoid readout_blip;

		double blip_angle = ro_blip_angle(seq_state->pos, seq);
		double blipX[3] = { 0., cos(blip_angle), 0. };
		double blipY[3] = { sin(blip_angle), 0., 0. };

		if (!prep_grad_ro_blip(&readout_blip, seq_state->pos[TE_DIM], seq))
			return ERROR_PREP_GRAD_RO_BLIP;

		timing.readout_blip = timing.readout[seq_state->pos[TE_DIM]] - grad_total_time(&readout_blip);

		if (   (1 < seq_state->pos[TE_DIM])
		    && (timing.readout_blip < timing.adc[seq_state->pos[TE_DIM] - 1] + adc_duration(seq) + ro_amplitude(seq) * seq->sys.grad.inv_slew_rate))
			return ERROR_BLIP_TIMING;

		i += seq_grad_to_event(ev + i, timing.readout_blip, &readout_blip, blipX);
		i += seq_grad_to_event(ev + i, timing.readout_blip, &readout_blip, blipY);

		struct grad_trapezoid readout;

		if (!prep_grad_ro(&readout, seq_state->pos[TE_DIM], seq))
			return ERROR_PREP_GRAD_RO_RO;

		if ((seq_state->pos[TE_DIM] == 0) && (timing.readout_dephaser + grad_total_time(&readout_dephaser) - 1.E-9) > timing.readout[seq_state->pos[TE_DIM]])
			return ERROR_RO_TIMING;

		if ((seq_state->pos[TE_DIM] == 0) && (timing.readout_dephaser + grad_total_time(&phs1_encoding) - 1.e-3) > timing.readout[seq_state->pos[TE_DIM]])
			return ERROR_RO_TIMING;

		i += seq_grad_to_event(ev + i, timing.readout[seq_state->pos[TE_DIM]], &readout, projX);
		i += seq_grad_to_event(ev + i, timing.readout[seq_state->pos[TE_DIM]], &readout, projY);

		i += prep_adc(ev + i, timing.adc[seq_state->pos[TE_DIM]], rf_spoil_phase, seq_state, seq);

		if (seq_state->pos[TE_DIM] == seq->loop_dims[TE_DIM] - 1) {

			if (!prep_grad_ro_reph(&ro_reph, seq))
				return ERROR_PREP_GRAD_RO_REPH;

			i += seq_grad_to_event(ev + i, timing.readout_rephaser, &ro_reph, projX);
			i += seq_grad_to_event(ev + i, timing.readout_rephaser, &ro_reph, projY);
		}

	} while (md_next(DIMS, seq->loop_dims, TE_FLAG, seq_state->pos));

	struct grad_trapezoid phase_rewinder;

	if (!prep_grad_phs1_encoding(&phase_rewinder, 1, seq_state->pos, seq))
		return ERROR_PREP_GRAD_SP_READ;

	i += seq_grad_to_event(ev + i, timing.readout_rephaser, &phase_rewinder, projPHASE);

	struct grad_trapezoid spoiler_read;

	if (!prep_grad_spoiler_read(&spoiler_read, seq))
		return ERROR_PREP_GRAD_SP_READ;

	i += seq_grad_to_event(ev + i, timing.spoiler_read, &spoiler_read, projREAD); // don't project spoiler gradients, we need constant direction

	struct grad_trapezoid spoiler_slice;

	if (!prep_grad_spoiler_slice(&spoiler_slice, seq))
		return ERROR_PREP_GRAD_SP_SLICE;

	double ampl_read = fabs(ro_reph.ampl) + fabs(spoiler_read.ampl);
	if (seq->sys.grad.max_amplitude < ampl_read)
		return ERROR_MAX_GRAD_SPOILER_READ;

	if (powf(seq->sys.grad.max_amplitude, 2.) < powf(ampl_read, 2.) + powf(spoiler_slice.ampl, 2.))
		return ERROR_MAX_GRAD_SPOILER;

	i += seq_grad_to_event(ev + i, timing.spoiler_slice, &spoiler_slice, projSLICE);


	struct grad_trapezoid pe3d_rewinder;

	if (!prep_grad_pe3d_rewinder(&pe3d_rewinder, seq_state->pos, seq))
		return ERROR_PREP_GRAD_PE3D_REW;

	i += seq_grad_to_event(ev + i, timing.pe3d_rewinder, &pe3d_rewinder, projSLICE);


	if (seq_block_end_flat(i, ev, seq->sys.raster_grad) - 1E-9 > seq->phys.tr)
		return ERROR_END_FLAT_KERNEL;

	long last_idx[DIMS];
	for (int i = 0; i < DIMS; i++)
		last_idx[i] = seq->loop_dims[i] - 1;

	if (   (SEQ_BLOCK_KERNEL_IMAGE == seq_state->mode)
	    && (SEQ_PEMODE_RAGA == seq->enc.pe_mode)
	    && md_check_equal_dims(DIMS, last_idx, seq_state->pos, SEQ_FLAGS & ~(COEFF_FLAG|COEFF2_FLAG|PHS1_FLAG|ITER_FLAG))
	    && (last_idx[ITER_DIM] == seq_state->pos[PHS1_DIM]))
			seq_state->pos[PHS1_DIM] = seq->loop_dims[PHS1_DIM] - 1;

	return i;
}


