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
#include "seq/pulse.h"
#include "seq/seq.h"
#include "seq/cest.h"
#include "seq/mag_prep.h"
#include "seq/seq_asl.h"
#include "seq/ui_enums.h"

#include "flash.h"

// for time-optimized overlapping gradients, otherwise sqrt(2.)
#define SCALE_GRAD 0.82


int prep_grad_ro(struct grad_trapezoid* grad, bart_dim_t echo, const struct seq_config* seq);


static double start_rf(const struct seq_config* seq)
{
	double sli_ampl = slice_amplitude(seq);
	double min_delay = MAX(sli_ampl * seq->sys.grad.inv_slew_rate, seq->sys.coil_control_lead);

	return round_up_raster(min_delay, seq->sys.raster_rf);
}

static double start_adc(bart_dim_t echo, const struct seq_config* seq)
{
	return round_up_raster(start_rf(seq) + seq->phys.rf_duration / 2. + seq->phys.te + echo * seq->phys.te_delta
				- adc_time_to_echo(echo, seq), seq->sys.raster_rf);
}

static double ro_shift(bart_dim_t echo, const struct seq_config* seq)
{
	double adc_start = start_adc(echo, seq);

	return seq->sys.raster_grad - (round_up_raster(adc_start, seq->sys.raster_grad) - adc_start);
}


static double available_time_RF_SLI(int ro, const struct seq_config* seq)
{
	double ampl = ro ? ro_amplitude(seq) : slice_amplitude(seq);

	return seq->phys.te - seq->phys.rf_duration / 2.
		- ampl * seq->sys.grad.inv_slew_rate
		- round_up_raster(adc_time_to_echo(0, seq) - 0.99 * seq->sys.raster_rf, seq->sys.raster_rf) // round down
		- ro_shift(0, seq);
}

static double ro_time_to_echo(bart_dim_t echo, const struct seq_config* seq)
{
	return ro_shift(echo, seq) + adc_time_to_echo(echo, seq);
}

static double ro_time_after_echo(bart_dim_t echo, const struct seq_config* seq)
{
	return round_up_raster(adc_duration(seq) + ro_shift(echo, seq), seq->sys.raster_grad) 
		- ro_time_to_echo(echo, seq);
}


static double ro_momentum_to_echo(bart_dim_t echo, const struct seq_config* seq)
{
	double amp = ro_amplitude(seq);

	return amp *
		(0.5 * amp * seq->sys.grad.inv_slew_rate
		+ ro_shift(echo, seq) + adc_time_to_echo(echo, seq));
}

static double ro_momentum(bart_dim_t echo, const struct seq_config* seq)
{
	double amp = ro_amplitude(seq);

	return amp * (amp * seq->sys.grad.inv_slew_rate
		 + round_up_raster(adc_duration(seq) + ro_shift(echo, seq), seq->sys.raster_grad));
}

static double ro_momentum_after_echo(bart_dim_t echo, const struct seq_config* seq)
{
	return ro_momentum(echo, seq) - ro_momentum_to_echo(echo, seq);
}

static double ro_blip_angle(const bart_dim_t pos[DIMS], const struct seq_config* seq)
{
	if (0 < pos[TE_DIM]) {

		double angle_curr = get_rot_angle(pos, seq);
		double moment_curr = ro_momentum_to_echo(pos[TE_DIM], seq);

		bart_dim_t pos2[DIMS];
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

static double ro_blip_moment(const bart_dim_t pos[DIMS], const struct seq_config* seq)
{
	if (0 < pos[TE_DIM]) {

		double angle_curr = get_rot_angle(pos, seq);
		double moment_curr = ro_momentum_to_echo(pos[TE_DIM], seq);

		bart_dim_t pos2[DIMS];
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
	const bart_dim_t echo = 0;

	struct grad_limits limits = seq->sys.grad;
	limits.max_amplitude *= SCALE_GRAD;

	if (!grad_soft(grad, available_time_RF_SLI(1, seq), -ro_momentum_to_echo(echo, seq), limits))
		return 0;

	return 1;
}

static int prep_grad_phs1_encoding(struct grad_trapezoid* grad, int rew, const bart_dim_t pos[DIMS], const struct seq_config* seq)
{
	*grad = (struct grad_trapezoid){ 0 };

	if (!((SEQ_PEMODE_CARTESIAN == seq->enc.pe_mode) || (SEQ_PEMODE_CARTESIAN_LINEAR == seq->enc.pe_mode)))
		return 1;

	if (rew && (SEQ_CONTRAST_RF_SPOILED != seq->phys.contrast))
		return 1;

	struct grad_limits limits = seq->sys.grad;
	limits.max_amplitude *= SCALE_GRAD;

	bart_dim_t center = 0.5 * seq->loop_dims[PHS1_DIM];

	double moment = (cartesian_line(pos, seq) - center) / (seq->sys.gamma * seq->geom.fov);

	if (rew)
		moment = -1. * moment;

	if (!grad_soft(grad, available_time_RF_SLI(1, seq), moment, limits))
		return 0;

	// FIXME maybe similar to GSTF with fixed timing

	return 1;
}


static int prep_grad_ro_blip(struct grad_trapezoid* grad, bart_dim_t echo, const struct seq_config* seq)
{
	*grad = (struct grad_trapezoid){ 0 };

	if (0 < echo) {

		bart_dim_t pos0[DIMS] = { [TE_DIM] = echo };
		pos0[TE_DIM] = echo;

		double moment = ro_blip_moment(pos0, seq);

		grad_hard(grad, moment, seq->sys.grad);
	}

	return 1;
}


int prep_grad_ro(struct grad_trapezoid* grad, bart_dim_t echo, const struct seq_config* seq)
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


static int prep_grad_sli_reph(struct grad_trapezoid* grad, bart_dim_t pos_phs2, const struct seq_config* seq)
{
	*grad = (struct grad_trapezoid){ 0 };

	struct grad_limits limits = seq->sys.grad;
	limits.max_amplitude *= SCALE_GRAD;

	if (!grad_soft(grad, available_time_RF_SLI(0, seq), -slice_momentum_to_rephase(seq), limits))
		return 0;

	if (seq->enc.is3D) {

		double pe_enc = (0.5 * seq->loop_dims[PHS2_DIM] - pos_phs2) / (seq->sys.gamma * seq->geom.slice_thickness * seq->loop_dims[PHS2_DIM]);
		double moment = - slice_momentum_to_rephase(seq) + pe_enc;

		if (!gradient_prepare_with_timing(grad, moment, seq))
			return 0;
	}

	return 1;
}


static int prep_grad_pe3d_rewinder(struct grad_trapezoid* grad, const bart_dim_t pos[DIMS], const struct seq_config* seq)
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

	double pe_enc = (0.5 * seq->loop_dims[PHS2_DIM] - pos[PHS2_DIM]) / (seq->sys.gamma * seq->geom.slice_thickness * seq->loop_dims[PHS2_DIM]);
	double moment = - slice_momentum_to_rephase(seq) - pe_enc;

	if (!gradient_prepare_with_timing(grad, moment, seq))
		return 0;

	return 1;
}


static void custom_params_to_config(struct seq_config* seq, int nl, const bart_dim_t custom_long[nl], int nd, const double custom_double[nd])
{
	seq->enc.pe_mode = (enum pe_mode)custom_long[SEQ_UI_IDX_LONG_PE_MODE];
	seq->phys.contrast = (enum flash_contrast)custom_long[SEQ_UI_IDX_LONG_CONTRAST];

	seq->geom.mb_factor = 1;

	if (CHECKBOX_ON == custom_long[SEQ_UI_IDX_LONG_SMS])
		seq->geom.mb_factor = custom_long[SEQ_UI_IDX_LONG_MB_FACTOR];

	seq->phys.os = 2.;

	seq->enc.tiny = custom_long[SEQ_UI_IDX_LONG_TINY];
	seq->magn.prep_scans = custom_long[SEQ_UI_IDX_LONG_PREP_SCANS];
	seq->phys.rf_duration = 1E-6 * custom_long[SEQ_UI_IDX_LONG_RF_DURATION_US];
	seq->magn.init_delay = custom_long[SEQ_UI_IDX_LONG_INIT_DELAY];
	seq->asl.label_type = (enum asl_label_type)custom_long[SEQ_UI_IDX_LONG_ASL_MODE];
	seq->loop_dims[BATCH_DIM] = (SEQ_ASL_NONE != seq->asl.label_type) 
					? ASL_BATCH_DIM_SIZE 
					: custom_long[SEQ_UI_IDX_LONG_INVERSIONS];
	seq->magn.inv_delay_time = custom_long[SEQ_UI_IDX_LONG_INV_DELAY];
	seq->enc.aligned_flags = (bart_flags_t)custom_long[SEQ_UI_IDX_LONG_RAGA_ALIGNED_FLAGS];


	seq->phys.bwtp = custom_double[SEQ_UI_IDX_DOUBLE_BWTP];
	seq->phys.asym_echo = custom_double[SEQ_UI_IDX_DOUBLE_ASYM_ECHO];

	// CEST
	seq->cest.sat_type = (enum cest_saturation_type)custom_long[SEQ_UI_IDX_LONG_CEST_SATURATION];
	seq->cest.sat_pulses = custom_long[SEQ_UI_IDX_LONG_CEST_SAT_PULSES];
	seq->cest.sat_pulse_pause = 1.E-3 * custom_long[SEQ_UI_IDX_LONG_CEST_SAT_PULSE_PAUSE_MS];

	seq->cest.gauss_pulse_duration = 1.E-3 * custom_long[SEQ_UI_IDX_LONG_CEST_GAUSS_DURATION_MS];
	seq->cest.gauss_pulse_fa = custom_long[SEQ_UI_IDX_LONG_CEST_GAUSS_FA];
	seq->cest.oc_pulse_b1_scaling = custom_double[SEQ_UI_IDX_DOUBLE_CEST_OC_B1_SCALING];

	seq->cest.offset_type = (enum cest_offset_type)custom_long[SEQ_UI_IDX_LONG_CEST_OFFSET_TYPE];
	seq->cest.offset_first = custom_double[	SEQ_UI_IDX_DOUBLE_CEST_OFFSET_FIRST_PPM];
	seq->cest.offset_last = custom_double[SEQ_UI_IDX_DOUBLE_CEST_OFFSET_LAST_PPM];
	seq->cest.offset_increment = custom_double[SEQ_UI_IDX_DOUBLE_CEST_OFFSET_INCREMENT_PPM];
	seq->cest.offset_pause = 1E-3 * custom_long[SEQ_UI_IDX_LONG_CEST_OFFSET_PAUSE_MS]; 
	
	seq->asl.ld = 1E-3 * custom_long[SEQ_UI_IDX_LONG_ASL_LD_MS];
	seq->asl.pld = 1E-3 * custom_long[SEQ_UI_IDX_LONG_ASL_PLD_MS];
}


static void config_to_custom_params(int nl, bart_dim_t custom_long[nl], int nd, double custom_double[nd], const struct seq_config* seq)
{
	custom_long[SEQ_UI_IDX_LONG_PE_MODE] = seq->enc.pe_mode;;
	custom_long[SEQ_UI_IDX_LONG_CONTRAST] = seq->phys.contrast;
	custom_long[SEQ_UI_IDX_LONG_RECO] = CHECKBOX_OFF;

	custom_long[SEQ_UI_IDX_LONG_SMS] = CHECKBOX_OFF;
	if (1 < seq->geom.mb_factor)
		custom_long[SEQ_UI_IDX_LONG_SMS] = CHECKBOX_ON;
	custom_long[SEQ_UI_IDX_LONG_MB_FACTOR] = seq->geom.mb_factor;

	custom_long[SEQ_UI_IDX_LONG_TINY] = seq->enc.tiny;
	custom_long[SEQ_UI_IDX_LONG_PREP_SCANS] = seq->magn.prep_scans;
	custom_long[SEQ_UI_IDX_LONG_RF_DURATION_US] = llround(1.E6 * seq->phys.rf_duration);
	custom_long[SEQ_UI_IDX_LONG_INIT_DELAY] = seq->magn.init_delay;
	custom_long[SEQ_UI_IDX_LONG_INVERSIONS] = seq->loop_dims[BATCH_DIM];
	custom_long[SEQ_UI_IDX_LONG_INV_DELAY] = seq->magn.inv_delay_time;
	custom_long[SEQ_UI_IDX_LONG_RAGA_ALIGNED_FLAGS] = (bart_dim_t)seq->enc.aligned_flags;
	custom_double[SEQ_UI_IDX_DOUBLE_BWTP] = seq->phys.bwtp;
	custom_double[SEQ_UI_IDX_DOUBLE_ASYM_ECHO] = seq->phys.asym_echo;

	// CEST
	custom_long[SEQ_UI_IDX_LONG_CEST_SATURATION] =seq->cest.sat_type;
	custom_long[SEQ_UI_IDX_LONG_CEST_SAT_PULSES]= seq->cest.sat_pulses;
	custom_long[SEQ_UI_IDX_LONG_CEST_SAT_PULSE_PAUSE_MS] = llround(1.E3 * seq->cest.sat_pulse_pause); // s -> ms

	custom_long[SEQ_UI_IDX_LONG_CEST_GAUSS_DURATION_MS] = llround(1.E3 * seq->cest.gauss_pulse_duration); // s -> ms
	custom_long[SEQ_UI_IDX_LONG_CEST_GAUSS_FA] = (bart_dim_t)seq->cest.gauss_pulse_fa;
	custom_double[SEQ_UI_IDX_DOUBLE_CEST_OC_B1_SCALING] = seq->cest.oc_pulse_b1_scaling;

	custom_long[SEQ_UI_IDX_LONG_CEST_OFFSET_TYPE] = seq->cest.offset_type;
	custom_double[SEQ_UI_IDX_DOUBLE_CEST_OFFSET_FIRST_PPM] = seq->cest.offset_first;
	custom_double[SEQ_UI_IDX_DOUBLE_CEST_OFFSET_LAST_PPM] = seq->cest.offset_last;
	custom_double[SEQ_UI_IDX_DOUBLE_CEST_OFFSET_INCREMENT_PPM] = seq->cest.offset_increment;
	custom_long[SEQ_UI_IDX_LONG_CEST_OFFSET_PAUSE_MS] = llround(1.E3 * seq->cest.offset_pause);

	custom_long[SEQ_UI_IDX_LONG_ASL_MODE] = seq->asl.label_type;
	custom_long[SEQ_UI_IDX_LONG_ASL_LD_MS] = llround(1.E3 * seq->asl.ld);
	custom_long[SEQ_UI_IDX_LONG_ASL_PLD_MS] = llround(1.E3 * seq->asl.pld);
}

void flash_interface_custom_params(int reverse, struct seq_config* seq, int nl, bart_dim_t params_long[nl], int nd, double params_double[nd])
{
	if (reverse)
		config_to_custom_params(nl, params_long, nd, params_double, seq);
	else
		custom_params_to_config(seq, nl, params_long, nd, params_double);
}


static void loop_dims_to_conf(struct seq_config* seq, const int D, const bart_dim_t in_dims[D])
{
	if(SEQ_ASL_NONE != seq->asl.label_type)
		seq->enc.order = SEQ_ORDER_SEQ_ASL;

	seq_copy_order(seq);

	if (seq->enc.is3D) {

		seq->loop_dims[PHS2_DIM] = in_dims[PHS2_DIM];
		seq->loop_dims[SLICE_DIM] = in_dims[SLICE_DIM];
		seq->geom.mb_factor = 1;

		seq->geom.slice_thickness = seq->geom.slice_thickness / (seq->loop_dims[PHS2_DIM] / seq->geom.slab_os);

	} else {

		bart_dim_t total_slices = in_dims[SLICE_DIM];

		if (1 < seq->geom.mb_factor) {

			seq->loop_dims[SLICE_DIM] = seq->geom.mb_factor;
			seq->loop_dims[PHS2_DIM] = total_slices / seq->geom.mb_factor;

		} else {

			seq->loop_dims[SLICE_DIM] = total_slices;
			seq->loop_dims[PHS2_DIM] = 1;
		}

		if ((seq->loop_dims[PHS2_DIM] * seq->loop_dims[SLICE_DIM]) != total_slices)
			seq->loop_dims[PHS2_DIM] = -1; //mb groups
	}

	bart_dim_t frames = in_dims[TIME_DIM];
	seq->loop_dims[TIME_DIM] = frames;

	seq->loop_dims[BATCH_DIM] = (SEQ_ASL_NONE != seq->asl.label_type) ? ASL_BATCH_DIM_SIZE : seq->loop_dims[BATCH_DIM];

	bart_dim_t radial_views = in_dims[PHS1_DIM];

	if (SEQ_PEMODE_RAGA == seq->enc.pe_mode) {

		seq->loop_dims[TIME_DIM] = (bart_dim_t)ceil(1. * frames / radial_views);
		seq->loop_dims[ITER_DIM] = frames % radial_views;

		if (0 == seq->loop_dims[ITER_DIM])
			seq->loop_dims[ITER_DIM] = radial_views;
	}

	seq->loop_dims[TIME2_DIM] = 1;

	if (SEQ_TRIGGER_OFF != seq->trigger.type)
		seq->loop_dims[TIME2_DIM] = in_dims[TIME2_DIM];

	seq->loop_dims[AVG_DIM] = in_dims[AVG_DIM];
	seq->loop_dims[PHS1_DIM] = radial_views;
	seq->loop_dims[TE_DIM] = in_dims[TE_DIM];
	seq->loop_dims[CSHIFT_DIM] = cest_offsets(seq);

	int pre_calls = 3; // 3 additional calls for delay_meas + noise_scan + ecg trigger
	if (SEQ_CEST_NONE != seq->cest.sat_type) 
		seq->loop_dims[COEFF2_DIM] = seq->cest.sat_pulses + MAX(1, seq->magn.prep_scans) + pre_calls; // no trigger for CEST, but spoiler or mag_prep
	else if (SEQ_ASL_NONE != seq->asl.label_type)
		seq->loop_dims[COEFF2_DIM] = calc_asl_coeff2_dim(seq) + pre_calls;
	else
		seq->loop_dims[COEFF2_DIM] = MAX(1, seq->magn.prep_scans) + pre_calls;


	seq->loop_dims[COEFF_DIM] = 3; // pre-/post- and actual kernel calls
}

static void conf_to_loop_dims(const int D, bart_dim_t dims[D], struct seq_config* seq)
{
	if (seq->enc.is3D) {

		dims[PHS2_DIM] = seq->loop_dims[PHS2_DIM];
		dims[SLICE_DIM] = seq->loop_dims[SLICE_DIM];

	} else {

		dims[SLICE_DIM] = seq->loop_dims[SLICE_DIM];
		dims[PHS2_DIM] = (seq->geom.mb_factor > 1) ? seq->loop_dims[SLICE_DIM] / seq->geom.mb_factor : 1;
		if ((dims[PHS2_DIM] * seq->geom.mb_factor != seq->loop_dims[SLICE_DIM]))
			seq->loop_dims[PHS2_DIM] = -1; //mb groups
	}

	dims[PHS1_DIM] = seq->loop_dims[PHS1_DIM];

	dims[BATCH_DIM] = (SEQ_ASL_NONE != seq->asl.label_type) ? ASL_BATCH_DIM_SIZE : seq->loop_dims[BATCH_DIM];
	dims[TIME_DIM] = seq->loop_dims[TIME_DIM];

	if (SEQ_PEMODE_RAGA == seq->enc.pe_mode)
		dims[TIME_DIM] = (dims[TIME_DIM] - 1) * dims[PHS1_DIM] + seq->loop_dims[ITER_DIM];

	dims[TE_DIM] = seq->loop_dims[TE_DIM];
	dims[AVG_DIM] = seq->loop_dims[AVG_DIM];
}

void flash_interface_loop_dims(int reverse, struct seq_config* seq, const int D, bart_dim_t dims[D])
{
	if (reverse)
		conf_to_loop_dims(D, dims, seq);
	else
		loop_dims_to_conf(seq, D, dims);
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

double flash_minimum_tr(const struct seq_config* seq)
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


void flash_minimum_te(const struct seq_config* seq, double* min_te, double* fill_te)
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
		bart_dim_t pos0[DIMS] = { };
		pos0[TE_DIM] = 1;
		grad_hard(&grad, ro_blip_moment(pos0, seq), seq->sys.grad);

		blip_time = grad_total_time(&grad);
	}

	for (bart_dim_t echo = 0; echo < seq->loop_dims[TE_DIM]; echo++) {

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

	for (bart_dim_t echo = 1; echo < seq->loop_dims[TE_DIM]; echo++) {

		time += fill_te[echo - 1];
		fill_te[echo] = seq->phys.te + echo * seq->phys.te_delta - min_te[echo] - time;
	}
}

static bart_dim_t inv_calls(const struct seq_config* seq)
{
	bart_dim_t calls = seq->loop_dims[BATCH_DIM] * ((SEQ_ASL_NONE == seq->asl.label_type) ? seq->loop_dims[CSHIFT_DIM] : 1);

	if (SEQ_ORDER_SEQ_MS == seq->enc.order)
		return calls * seq->loop_dims[SLICE_DIM];

	return calls;
}

static bart_dim_t flash_ex_calls(const struct seq_config* seq)
{
	bart_dim_t dims[DIMS];
	md_select_dims(DIMS, SEQ_FLAGS & ~(COEFF_FLAG|COEFF2_FLAG), dims, seq->loop_dims);

	bart_dim_t incomplete_raga_spks = 0;
	if (SEQ_PEMODE_RAGA == seq->enc.pe_mode)
		incomplete_raga_spks = seq->loop_dims[PHS1_DIM] - seq->loop_dims[ITER_DIM];

	if ((1 < seq->geom.mb_factor) || seq->enc.is3D) {

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

	bart_dim_t factor = dims[SLICE_DIM];
	if (1 < seq->geom.mb_factor)
		factor = dims[PHS2_DIM];

	return md_calc_size(DIMS, dims) - incomplete_raga_spks
		+ factor * seq->magn.prep_scans;
}

double flash_total_measure_time(const struct seq_config* seq)
{
	double pre_duration = seq->magn.init_delay +  seq->phys.tr; // noise scan

	if (SEQ_ASL_NONE != seq->asl.label_type)
		return calc_asl_duration(seq) + pre_duration;

	struct seq_event ev[6];
	int e = mag_prep(ev, seq);

	double prep_pulse_duration = seq_block_end(e, ev, SEQ_BLOCK_PRE, seq->phys.tr, seq->sys.raster_grad);
	prep_pulse_duration += seq->magn.inv_delay_time + seq->trigger.delay_time;
	prep_pulse_duration *= inv_calls(seq);

	if (SEQ_CEST_NONE != seq->cest.sat_type) {

		double sat_time = (SEQ_CEST_GAUSS == seq->cest.sat_type) ? seq->cest.gauss_pulse_duration : 0.1;
		sat_time += seq->cest.sat_pulse_pause;
		sat_time += seq->sys.coil_control_lead * 2;
		sat_time *= seq->cest.sat_pulses;
		
		if (0 == mag_prep(ev, seq)) // spoiler after last pulse only if no inversion
			sat_time += 10.E-3; 
		prep_pulse_duration += sat_time;
		prep_pulse_duration *= cest_offsets(seq);
		prep_pulse_duration += seq->cest.offset_pause * (cest_offsets(seq) - 1);
	}

	bart_dim_t img_calls = flash_ex_calls(seq) * seq->geom.mb_factor;
	double imaging_duration = seq->phys.tr * img_calls;

	if ((SEQ_TRIGGER_OFF != seq->trigger.type) && (1 < seq->trigger.pulses))
		imaging_duration = 1. * (seq->trigger.delay_time + seq->phys.tr) * img_calls * (seq->trigger.pulses - 1);

	return pre_duration  + imaging_duration + prep_pulse_duration;
}

int flash_sample_rf_shapes(int N, struct rf_shape pulse[N], const struct seq_config* seq)
{
	int idx = 0;

	for (; idx < seq->geom.mb_factor; idx++) {

		if (idx >= N)
			return -1;

		pulse[idx].sar_calls = flash_ex_calls(seq);
		pulse[idx].sar_dur = seq->phys.rf_duration;
		pulse[idx].fa_prep = seq->phys.flip_angle;

		const float alpha = 0.5;

		pulse[idx].samples = llround(1.E6 * seq->phys.rf_duration);

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

		pulse[idx].samples = llround(0.5 * 1E6 * pulse[idx].sar_dur);

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

		pulse[idx].samples = llround(1E4 * pulse[idx].sar_dur);

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

		pulse[idx].samples = llround(1.E6 * seq->asl.hanning.rf_duration);

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

	bart_dim_t last_idx[DIMS];
	for (int i = 0; i < DIMS; i++)
		last_idx[i] = seq->loop_dims[i] - 1;

	if (   (SEQ_BLOCK_KERNEL_IMAGE == seq_state->mode)
	    && (SEQ_PEMODE_RAGA == seq->enc.pe_mode)
	    && md_check_equal_dims(DIMS, last_idx, seq_state->pos, SEQ_FLAGS & ~(COEFF_FLAG|COEFF2_FLAG|PHS1_FLAG|ITER_FLAG))
	    && (last_idx[ITER_DIM] == seq_state->pos[PHS1_DIM]))
			seq_state->pos[PHS1_DIM] = seq->loop_dims[PHS1_DIM] - 1;

	return i;
}


static bart_dim_t get_chrono_slice(const struct seq_state* seq_state, const struct seq_config* seq)
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

	if ((1 < seq->geom.mb_factor) && seq->enc.is3D)
		return ERROR_SETTING_DIM;

	if ((1 < seq->geom.mb_factor) && (SEQ_ORDER_SEQ_MS == seq->enc.order))
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

int flash_block(int N, struct seq_event ev[N], struct seq_state* seq_state, const struct seq_config* seq)
{
	int err = check_settings(seq_state, seq);
	if (1 > err)
		return err;

	seq_state->chrono_slice = get_chrono_slice(seq_state, seq);

	if (   (SEQ_BLOCK_KERNEL_PREPARE == seq_state->mode)
	    || (SEQ_BLOCK_KERNEL_CHECK == seq_state->mode))
		return flash(N, ev, seq_state, seq);

	bart_dim_t zeros[DIMS] = { };
	bart_dim_t last_idx[DIMS];

	for (int i = 0; i < DIMS; i++)
		last_idx[i] = seq->loop_dims[i] - 1;

	// changed beahvior for sequential multislice
	bart_flags_t msm_flag = 0;

	// changed behavior for ASL
	bart_flags_t asl_flag = 0;

	if (md_check_equal_order(DIMS, seq->order, seq_loop_order_multislice, SEQ_FLAGS))
	       msm_flag = SLICE_FLAG ;
	
	if (SEQ_ASL_NONE != seq->asl.label_type)
		asl_flag = AVG_FLAG;

	if (0 == seq_state->pos[COEFF_DIM]) {

		if (md_check_equal_dims(DIMS, zeros, seq_state->pos, ~UINT64_C(0))) {

			seq_state->mode = SEQ_BLOCK_PRE;

			return wait_time_to_event(ev, 0., seq->magn.init_delay);
		}

		zeros[COEFF2_DIM] = 1;

		if (md_check_equal_dims(DIMS, zeros, seq_state->pos, ~UINT64_C(0))) {

			seq_state->mode = SEQ_BLOCK_KERNEL_NOISE;

			return flash(N, ev, seq_state, seq);
		}

		if (1 < seq_state->pos[COEFF2_DIM]) {

			// Skip spoke iterations for ASL
			if (   (SEQ_ASL_NONE != seq->asl.label_type)
			    && ((seq_state->pos[BATCH_DIM] == 0) || (seq_state->pos[PHS1_DIM] > 0) || (seq_state->pos[TIME_DIM] > 0)))
				md_max_dims(DIMS, COEFF2_FLAG, seq_state->pos, seq_state->pos, last_idx);

			if (md_check_equal_dims(DIMS, zeros, seq_state->pos, ~(BATCH_FLAG | COEFF2_FLAG | SLICE_FLAG | PHS2_FLAG))) {

				if (   (0 < seq->magn.prep_scans) && (2 < seq_state->pos[COEFF2_DIM])
				    && (   ((1 == seq->geom.mb_factor) && (0 == seq_state->pos[PHS2_DIM]))
					|| ((1 <  seq->geom.mb_factor) && (0 == seq_state->pos[SLICE_DIM])))) {

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

		if (seq->trigger.trigger_out && md_check_equal_dims(DIMS, (bart_dim_t [DIMS]){ 0 }, seq_state->pos, PHS1_FLAG))
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
