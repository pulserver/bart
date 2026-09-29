/* Copyright 2025-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>
#include <stdio.h>

#include "num/multind.h"

#include "misc/mri.h"
#include "misc/misc.h"

#include "noncart/traj.h"

#include "seq/anglecalc.h"
#include "seq/config.h"
#include "seq/custom_selections.h"
#include "seq/misc.h"
#include "seq/opts.h"

#include "seq/flash.h"
#include "seq/miniflash.h"

#include "helpers.h"

int seq_raga_spokes(const struct seq_config* seq)
{
	if (SEQ_PEMODE_RAGA == seq->enc.pe_mode)
		return raga_spokes(seq->geom.baseres, seq->enc.tiny);

	return seq->loop_dims[PHS1_DIM];
}


double seq_minimum_tr(const struct seq_config* seq)
{
	switch (seq->seq_type) {

	case SEQ_TYPE_FLASH:

		return flash_minimum_tr(seq);

	case SEQ_TYPE_MINIFLASH:

		return miniflash_minimum_tr(seq);
	}

	return 0.;
}


void seq_minimum_te(const struct seq_config* seq, double* min_te, double* fill_te)
{
	switch (seq->seq_type) {

	case SEQ_TYPE_FLASH:

		flash_minimum_te(seq, min_te, fill_te);
		break;

	case SEQ_TYPE_MINIFLASH:

		miniflash_minimum_te(seq, min_te, fill_te);
		break;
	}
}




static long kernels_per_measurement(const long loop_dims[DIMS])
{
	long dims[DIMS];
	md_select_dims(DIMS, (PHS1_FLAG|TIME2_FLAG|AVG_FLAG|SLICE_FLAG|PHS2_FLAG|CSHIFT_FLAG), dims, loop_dims);

	return md_calc_size(DIMS, dims);
}

long seq_relevant_readouts_meas_time(const struct seq_config* seq)
{
	return kernels_per_measurement(seq->loop_dims) / seq->loop_dims[PHS1_DIM];
}

double seq_total_measure_time(const struct seq_config* seq)
{
	switch (seq->seq_type) {

	case SEQ_TYPE_FLASH:

		return flash_total_measure_time(seq);

	case SEQ_TYPE_MINIFLASH:

		return miniflash_total_measure_time(seq);
	}

	return 0.;
}


void seq_ui_interface_custom_params(int reverse, struct seq_config* seq, int nl, long params_long[__VLA(nl)], int nd, double params_double[__VLA(nd)])
{
	switch (seq->seq_type) {

	case SEQ_TYPE_FLASH:

		flash_interface_custom_params(reverse, seq, nl, params_long, nd, params_double);
		break;

	case SEQ_TYPE_MINIFLASH:

		if (reverse)
			miniflash_interface_custom_back(seq, nl, params_long, nd, params_double);
		else
			miniflash_interface_custom(seq, nl, params_long, nd, params_double);

		break;
	}
	
}

static void seq_init_standard_conf(struct seq_standard_conf* init_std)
{
	init_std->tr = 100E-3;
	init_std->te[0] = 5E-3;
	init_std->dwell = 10.6E-6;
	init_std->flip_angle = 8;
	init_std->fov = 256E-3;
	init_std->baseres = 128;
	init_std->slice_thickness = 5E-3;
	init_std->enc_order = SEQ_ORDER_AVG_OUTER;

	init_std->is3D = 0;
	init_std->slice_os = 1.;

	init_std->mag_prep = SEQ_PREP_OFF;
	init_std->ti = 0.;
	init_std->trigger_type = SEQ_TRIGGER_OFF;

	init_std->gamma = 42.575575E6;
	init_std->b0 = 2.893620;
	init_std->grad_max_ampl = .024;
	init_std->grad_min_rise_time = .007848885540911;
	init_std->coil_control_lead = 100.E-6;
	init_std->min_duration_ro_rf = 213.E-6;
	init_std->raster_grad = 1.E-5;
	init_std->raster_rf = 1.E-6;
	init_std->raster_dwell = 1.E-7;
}

static void seq_bart_to_standard_conf(struct seq_standard_conf* std, struct seq_config* seq)
{
	std->tr = seq->phys.tr;

	for (int i = 0; i < SEQ_MAX_NO_ECHOES; i++)
		std->te[i] = seq->phys.te + i * seq->phys.te_delta;

	std->dwell = seq->phys.dwell;
	std->flip_angle = seq->phys.flip_angle;

	std->fov = seq->geom.fov;
	std->baseres = seq->geom.baseres;
	if (seq->enc.is3D)
		std->slice_thickness = seq->geom.slice_thickness * seq->loop_dims[PHS2_DIM] / seq->geom.slab_os;
	else
		std->slice_thickness = seq->geom.slice_thickness;

	std->slice_os = seq->geom.slab_os;
	std->is3D = seq->enc.is3D;

	std->gamma = seq->sys.gamma;
	std->b0 = seq->sys.b0;
	std->grad_min_rise_time = seq->sys.grad.inv_slew_rate;
	std->grad_max_ampl = seq->sys.grad.max_amplitude;
	std->coil_control_lead = seq->sys.coil_control_lead;
	std->min_duration_ro_rf = seq->sys.min_duration_ro_rf;

	std->mag_prep = seq->magn.mag_prep;
	std->ti = 0.;

	if (SEQ_PREP_OFF != seq->magn.mag_prep)
		std->ti = seq->magn.ti;

	std->trigger_type = seq->trigger.type;
	std->trigger_delay_time = seq->trigger.delay_time;
	std->trigger_pulses = seq->trigger.pulses;
	std->trigger_out = 1;

	std->enc_order = seq->enc.order;

	for (int i = 0; i < SEQ_ACOUSTIC_RESONANCE_ENTRIES; i++) {

		std->acoustic_res_freq[i] = std->acoustic_res_freq[i];
		std->acoustic_res_bw[i] = std->acoustic_res_bw[i];
	}
}

static void seq_standard_conf_to_bart(struct seq_config* seq, struct seq_standard_conf* std)
{
	seq->phys.tr = std->tr;

	seq->phys.te = std->te[0];
	seq->phys.te_delta = std->te[1] - std->te[0];

	seq->phys.dwell = std->dwell;
	seq->phys.os = 2.;
	seq->phys.flip_angle = std->flip_angle;

	seq->geom.fov = std->fov;
	seq->geom.baseres = std->baseres;
	seq->geom.slice_thickness = std->slice_thickness;  // for 3D: changed in loop_dims_to_conf
	seq->geom.slab_os = std->slice_os;

	seq->enc.is3D = std->is3D;

	seq->sys.gamma = std->gamma;

	seq->sys.b0 = std->b0;

	seq->sys.grad.inv_slew_rate = std->grad_min_rise_time;
	seq->sys.grad.max_amplitude = std->grad_max_ampl;

	seq->sys.coil_control_lead = std->coil_control_lead;
	seq->sys.min_duration_ro_rf = std->min_duration_ro_rf;

	seq->magn.mag_prep = std->mag_prep;

	seq->magn.ti = 0;

	if (SEQ_PREP_OFF != seq->magn.mag_prep)
		seq->magn.ti = std->ti;

	seq->trigger.type = std->trigger_type;
	seq->trigger.delay_time = std->trigger_delay_time;
	seq->trigger.pulses = std->trigger_pulses;
	seq->trigger.trigger_out = 1;

	seq->enc.order = std->enc_order;
}


void seq_ui_interface_standard_conf(int reverse, struct seq_config* conf, struct seq_standard_conf* std_conf)
{
	if (2 == reverse) {

		seq_init_standard_conf(std_conf);
		return;
	}

	if (reverse)
		seq_bart_to_standard_conf(std_conf, conf);
	else
		seq_standard_conf_to_bart(conf, std_conf);
}


static void seq_init_loop_dims(const int D, long dims[D])
{
	for (int i = 0; i < D; i++)
		dims[i] = 1;
}


void seq_ui_interface_loop_dims(int reverse, struct seq_config* seq, const int D, long dims[__VLA(D)])
{
	if (2 == reverse) {

		seq_init_loop_dims(D, dims);
		return;
	}

	switch (seq->seq_type) {

	case SEQ_TYPE_FLASH:

		flash_interface_loop_dims(reverse, seq, D, dims);
		break;

	default:

		if (reverse) {

			md_copy_dims(D, dims, seq->loop_dims);

		} else {

			md_select_dims(D, (PHS1_FLAG | PHS2_FLAG | TE_FLAG | TIME_FLAG | TIME2_FLAG | SLICE_FLAG | AVG_FLAG), seq->loop_dims, dims);
			seq_copy_order(seq);
		}
	}
}


struct seq_interface_conf seq_get_interface_conf(struct seq_config* conf)
{
	struct seq_interface_conf ret = { };

	if (conf->enc.is3D)
		ret.mode |= SEQ_MODE_3D;

	if (SEQ_ASL_NONE != conf->asl.label_type)
		ret.mode |= SEQ_MODE_ASL;

	ret.tr = conf->phys.tr;
	ret.radial_views = conf->loop_dims[PHS1_DIM];
	ret.slices = get_slices(conf);
	ret.echoes = conf->loop_dims[TE_DIM];
	if (conf->enc.is3D)
		ret.slice_thickness = conf->geom.slice_thickness * conf->loop_dims[PHS2_DIM];
	else
		ret.slice_thickness = conf->geom.slice_thickness;

	ret.trigger_type = conf->trigger.type;
	ret.trigger_delay_time = conf->trigger.delay_time;
	ret.trigger_pulses = conf->trigger.pulses;

	ret.raster_grad = conf->sys.raster_grad;
	ret.raster_rf = conf->sys.raster_rf;
	ret.grad_max_ampl = conf->sys.grad.max_amplitude;

	return ret;
}


void seq_set_fov_pos(int N, int M, const float* shifts, struct seq_config* seq)
{
	long total_slices = get_slices(seq);
	assert(total_slices <= N);

	seq->geom.sms_distance = 0;

	if (1 < seq->geom.mb_factor)
		seq->geom.sms_distance = fabsf(shifts[2] - shifts[seq->loop_dims[PHS2_DIM] * M + 2]);

	for (int i = 0; i < total_slices; i++) {

		seq->geom.shift[i][0] = shifts[i * M + 0]; // RO shift
		seq->geom.shift[i][1] = shifts[i * M + 1]; // PE shift

		if (1 < seq->geom.mb_factor) {

			seq->geom.shift[i][2] = shifts[(total_slices / 2) * M + 2]
						+ (seq->geom.sms_distance / seq->loop_dims[PHS2_DIM]) 
						* (i % seq->loop_dims[PHS2_DIM] - floor(seq->loop_dims[PHS2_DIM] / 2.));

		} else {

			seq->geom.shift[i][2] = shifts[i * M + 2];
		}

		// FIXME: new interface fct to sequence
		for (int j = 0; j < 3; j++) {

			seq->geom.rot[i][j][0] = shifts[(j + 1) * total_slices * M + i * M + 0];
			seq->geom.rot[i][j][1] = shifts[(j + 1) * total_slices * M + i * M + 1];
			seq->geom.rot[i][j][2] = shifts[(j + 1) * total_slices * M + i * M + 2];
		}
	}

	if (1 == seq->geom.mb_factor)
		seq->geom.sms_distance = -0.999; // UI information
}


int seq_config_from_string(struct seq_config* seq, int N, char* buffer)
{
	return read_config_from_str(seq, N, buffer);
}


int seq_print_info_config(int N, char* info, const struct seq_config* seq)
{
	int ctr = 0;

	ctr += snprintf(info + ctr, (size_t)(N - ctr), "\n\nseq_config\nsequence type\t\t\t\t%s\nTR/TE0/deltaTE\t\t\t\t%f/%f/%f", 
			get_seqtype_str(seq->seq_type), seq->phys.tr, seq->phys.te, seq->phys.te_delta);

	ctr += snprintf(info + ctr, (size_t)(N - ctr), 
			"\ndwell/os/asym\t\t\t\t%.8f/%.2f/%.2f\ncontrast/rf duration/FA/BWTP\t\t%d (\"%s\")/%.6f/%.2f/%.2f",
			seq->phys.dwell, seq->phys.os, seq->phys.asym_echo,
			seq->phys.contrast, get_contrast_str(seq->phys.contrast),
			seq->phys.rf_duration, seq->phys.flip_angle, seq->phys.bwtp);

	ctr += snprintf(info + ctr, (size_t)(N - ctr), 
			"\nFOV/slice-th\t\t\t\t%.3f/%.3f\nBR/mb_factor/SMS dist\t\t\t%d/%d/%.3f",
			seq->geom.fov, seq->geom.slice_thickness,
			seq->geom.baseres, seq->geom.mb_factor, seq->geom.sms_distance);
	
	ctr += snprintf(info + ctr, (size_t)(N - ctr),
			"\nPE_Mode/Turns-GA/aligned flags/order\t%d (\"%s\")/%d/%ld/%d\nis3D/slab-os\t\t\t\t%d/%.2f",
			seq->enc.pe_mode, get_pemode_str(seq->enc.pe_mode), seq->enc.tiny, seq->enc.aligned_flags, seq->enc.order,
			seq->enc.is3D, seq->geom.slab_os);

	ctr += snprintf(info + ctr, (size_t)(N - ctr),
			"\nmag prep/TI/inv delay\t\t\t%d/%.3f/%.2f",
			seq->magn.mag_prep, seq->magn.ti, seq->magn.inv_delay_time);

	ctr += snprintf(info + ctr, (size_t)(N - ctr),
			"\ninit delay/prep scans\t\t\t%.2f/%ld",
			seq->magn.init_delay, seq->magn.prep_scans);

	ctr += snprintf(info + ctr, (size_t)(N - ctr),
			"\ngamma/b0/max grad/inv slew\t\t%.0f/%.3f/%.3f/%.6f\n",
			seq->sys.gamma, seq->sys.b0, seq->sys.grad.max_amplitude, seq->sys.grad.inv_slew_rate);


	ctr += snprintf(info + ctr, (size_t)(N - ctr),
			"\nloop_dims\t %ld|%ld|%ld|%ld\t\t%ld|%ld|%ld|%ld\t\t%ld|%ld|%ld|%ld\t\t%ld|%ld|%ld|%ld\t\t\n",
			seq->loop_dims[READ_DIM], seq->loop_dims[PHS1_DIM], seq->loop_dims[PHS2_DIM], seq->loop_dims[COIL_DIM],
			seq->loop_dims[MAPS_DIM], seq->loop_dims[TE_DIM], seq->loop_dims[COEFF_DIM], seq->loop_dims[COEFF2_DIM],
			seq->loop_dims[ITER_DIM], seq->loop_dims[CSHIFT_DIM], seq->loop_dims[TIME_DIM], seq->loop_dims[TIME2_DIM],
			seq->loop_dims[LEVEL_DIM], seq->loop_dims[SLICE_DIM], seq->loop_dims[AVG_DIM], seq->loop_dims[BATCH_DIM]);

	if (SEQ_TYPE_FLASH == seq->seq_type) {

		ctr += snprintf(info + ctr, (size_t)(N - ctr),
			"\nCEST sat\t\ttype=%d \t n=%ld \t\t\t (pause: %.4f)",
			seq->cest.sat_type, seq->cest.sat_pulses, seq->cest.sat_pulse_pause);
		ctr += snprintf(info + ctr, (size_t)(N - ctr),
			"\nCEST gauss\t\tdur %f\t fa %.2f \t (OC_B1: %.2f)",
			seq->cest.gauss_pulse_duration, seq->cest.gauss_pulse_fa, seq->cest.oc_pulse_b1_scaling);
		ctr += snprintf(info + ctr, (size_t)(N - ctr),
			"\nCEST offsets\t\ttype=%d \t %.2f / %.2f / %.2f \t (pause: %.2f)",
			seq->cest.offset_type, seq->cest.offset_first, seq->cest.offset_last, seq->cest.offset_increment,
			seq->cest.offset_pause);

		ctr += snprintf(info + ctr, (size_t)(N - ctr),
			"\n\nASL mode/LD/PLD\t\t%d/%.3f/%.3f\t (label sl idx: %d)",
			seq->asl.label_type, seq->asl.ld, seq->asl.pld, seq->asl.label_slice_index);
	}

	ctr += snprintf(info + ctr, (size_t)(N - ctr), "\n\nCrowthers no. of radial Spokes =\t%.2f\n\n", M_PI * seq->geom.baseres);


	ctr += snprintf(info + ctr, (size_t)(N - ctr), "\n\nbart seq ");
	struct seq_opts seq_opts = seq_opts_defaults;
	ctr += seq_cmdline_print((size_t)(N - ctr), info + ctr, seq, &seq_opts);
	ctr += snprintf(info + ctr, (size_t)(N - ctr), "\n\n");

	int slices = get_slices(seq);
	ctr += snprintf(info + ctr, (size_t)(N - ctr), "\n\nFOV shifts:\tREAD\tPHASE\tSLICE\n");
	for (int i = 0; i < slices; i++)
		ctr += snprintf(info + ctr, (size_t)(N - ctr), "\t[%d]:\t%+.4f\t%+.4f\t%+.4f\n",
				i, seq->geom.shift[i][0], seq->geom.shift[i][1], seq->geom.shift[i][2]);

	ctr += snprintf(info + ctr, (size_t)(N - ctr), "\n\nROTATION MATRIX\n");
	for (int i = 0; i < slices; i++) {

		ctr += snprintf(info + ctr, (size_t)(N - ctr), 
			"\t[%d]:\t%+.4f\t%+.4f\t%+.4f\n\t\t%+.4f\t%+.4f\t%+.4f\n\t\t%+.4f\t%+.4f\t%+.4f\n",
				i, seq->geom.rot[i][0][0], seq->geom.rot[i][0][1], seq->geom.rot[i][0][2],
				   seq->geom.rot[i][1][0], seq->geom.rot[i][1][1], seq->geom.rot[i][1][2],
				   seq->geom.rot[i][2][0], seq->geom.rot[i][2][1], seq->geom.rot[i][2][2]);

	}


	if (ctr > N)
		return -1;

	return ctr;
}



void seq_print_info_radial_views(int N, char* info, const struct seq_config* seq)
{
	(void) N;

	int ctr = 0;

	if (   (SEQ_PEMODE_CARTESIAN == seq->enc.pe_mode) 
	    || (SEQ_PEMODE_CARTESIAN_LINEAR == seq->enc.pe_mode)) {

		ctr += snprintf(info + ctr, (size_t)(N - ctr), 
			"Cartesian sequence\nRadial Views = Phase encoding lines\nMeasurements = frames.");
		return;
	}

	if (SEQ_PEMODE_RAGA != seq->enc.pe_mode)
		return;


	struct traj_conf conf;
	traj_conf_from_seq(&conf, seq);
	double angle = calc_angle_atom(&conf) * raga_increment(seq->loop_dims[PHS1_DIM], conf.tiny_gold);

	ctr += snprintf(info + ctr, (size_t)(N - ctr), 
		"Rational Approximation of Golden Angle Sampling.\nTiny-GA: %d\tProjection angle: %f (deg)\nallowed spokes: ",
		seq->enc.tiny, angle * 180. / M_PI);

	int i = 1;

	// max number of spokes
	while (2050 > gen_fibonacci(seq->enc.tiny, i)) {

		if (check_gen_fib(gen_fibonacci(seq->enc.tiny, i), seq->enc.tiny)) {

			ctr += snprintf(info + ctr, (size_t)(N - ctr), "%d\t", gen_fibonacci(seq->enc.tiny, i));
		}

		i++;
	}

	ctr += snprintf(info + ctr, (size_t)(N - ctr), "\n");
}
