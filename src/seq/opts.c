/* Copyright 2025. Insitute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <setjmp.h>

#include "misc/opts.h"
#include "misc/mri.h"
#include "misc/misc.h"

#include "seq/config.h"
#include "seq/helpers.h"

#include "opts.h"


const struct seq_opts seq_opts_defaults = {

	.dt = -1.,
	.samples = -1,
	.rel_shift = { },
	.raga_full_frames = 0,
	.dist = 1.,

	.gradient_mode = GRAD_FAST,

	.chrono = false,
	.support = false,

	.raga_file = NULL,

	.custom_params_long = { 0 },
	.custom_params_double = { 0. },
};



void seq_cmdline(int* argcp, char* argv[*argcp], int m, const struct arg_s args[m],
			const char* help_str, struct seq_config* conf, struct seq_opts* seq_opts)
{

	const struct opt_s opts[] = {

		OPT_FLOAT('d', &seq_opts->dt, "dt", "time-increment per sample (default: seq->conf->phys.tr / 1000)"),
		OPT_LONG('N', &seq_opts->samples, "samples", "Number of samples (default: 1000)"),

		OPT_DOVEC3('s', &conf->geom.shift[0], "RO:PE:SL", "FOV shift"),
		OPT_DOVEC3('S', &seq_opts->rel_shift, "RO:PE:SL", "relative FOV shift"),
		OPTL_FLOAT(0, "dist", &seq_opts->dist, "dist", "slice distance factor [1 / slice_thickness] (default: 1.)"),

		// contrast mode
		OPTL_SELECT(0, "no-spoiling", enum flash_contrast, &conf->phys.contrast,
				SEQ_CONTRAST_NO_SPOILING, "spoiling off (default: rf random)"),
		OPTL_SELECT(0, "spoiled", enum flash_contrast, &conf->phys.contrast,
				SEQ_CONTRAST_RF_SPOILED, "RF_SPOILED (inc: 50 deg, gradient on) (default: rf random)"),

		// FOV and resolution
		OPTL_DOUBLE(0, "FOV", &conf->geom.fov, "FOV", "Field Of View"),
		OPTL_PINT(0, "BR", &conf->geom.baseres, "BR", "Base Resolution"),
		OPTL_DOUBLE(0, "slice_thickness", &conf->geom.slice_thickness, "slice_thickness", "Slice thickness"),

		// basic sequence parameters
		OPTL_DOUBLE(0, "FA", &conf->phys.flip_angle, "flip angle", "Flip angle [deg]"),
		OPTL_DOUBLE(0, "TR", &conf->phys.tr, "TR", "TR"),
		OPTL_DOUBLE(0, "TE", &conf->phys.te, "TE", "TE"),
		OPTL_DOUBLE(0, "TE_delta", &conf->phys.te_delta, "TE_delta", "TE_delta"),
		OPTL_DOUBLE(0, "BWTP", &conf->phys.bwtp, "BWTP", "Bandwidth Time Product"),

		// others sequence parameters
		OPTL_DOUBLE(0, "rf_duration", &conf->phys.rf_duration, "rf_duration", "RF pulse duration"),
		OPTL_DOUBLE(0, "dwell", &conf->phys.dwell, "dwell", "Dwell time"),
		OPTL_DOUBLE(0, "os", &conf->phys.os, "os", "Oversampling factor"),

		// encoding
		OPTL_UINT(0, "pe_mode", &conf->enc.pe_mode, "pe_mode", "Phase-encoding mode"),
		OPTL_SELECT(0, "turn", enum pe_mode, &conf->enc.pe_mode, SEQ_PEMODE_TURN, "turn-based PE (default: RAGA)"),
		OPTL_SELECT(0, "mems", enum pe_mode, &conf->enc.pe_mode, SEQ_PEMODE_MEMS_HYB, "multi-echo/multi-spoke PE (default: RAGA)"),
		OPTL_SELECT(0, "raga", enum pe_mode, &conf->enc.pe_mode, SEQ_PEMODE_RAGA, "RAGA PE"),
		OPTL_ULONG(0, "raga_flags", &conf->enc.aligned_flags, "raga_aligned_flags", "RAGA aligned flags (by bitmask)"),

		OPTL_SET(0, "chrono", &seq_opts->chrono, "save gradients/moments/sampling in chronological order (RAGA)"),
		OPT_OUTFILE('R', &seq_opts->raga_file, "file", "raga indices"),

		OPTL_PINT(0, "tiny", &conf->enc.tiny, "tiny", "Tiny golden-ratio index"),

		OPTL_LONG('e', "echoes", &conf->loop_dims[TE_DIM], "echoes", "Number of echoes"),
		OPTL_LONG('r', "lines", &conf->loop_dims[PHS1_DIM], "lines", "Number of phase encoding lines"),
		OPTL_LONG('z', "partitions", &conf->loop_dims[PHS2_DIM], "partitions", "Number of partitions (3D) or SMS groups (2D)"),
		OPTL_LONG('t', "measurements", &conf->loop_dims[TIME_DIM], "measurements", "Number of measurements / frames (RAGA: total number of spokes)"),
		OPTL_LONG('f', "raga_full_frames", &seq_opts->raga_full_frames, "raga_full_frames", "Number of full frames (only RAGA)"),
		OPTL_LONG('m', "slices", &conf->loop_dims[SLICE_DIM], "slices", "Number of slices of multiband factor (SMS)"),
		OPTL_LONG('i', "inversions", &conf->loop_dims[BATCH_DIM], "inversions", "Number of inversions"),

		// order
		OPTL_SELECT(0, "sequential-multislice", enum seq_order, &conf->enc.order, SEQ_ORDER_SEQ_MS, "seq_order: sequential multislice (default: avg outer)"),
		OPTL_SELECT(0, "avg-inner", enum seq_order, &conf->enc.order, SEQ_ORDER_AVG_INNER, "seq_order: average inner (default: avg outer)"),

		// sms
		OPTL_PINT(0, "mb_factor", &conf->geom.mb_factor, "mb_factor", "Multi-band factor"),
		OPTL_DOUBLE(0, "sms_distance", &conf->geom.sms_distance, "sms_distance", "SMS slice distance"),

		// magnetization preparation
		OPTL_SELECT(0, "IR_NON", enum mag_prep, &conf->magn.mag_prep, SEQ_PREP_IR_NONSELECTIVE, "Magn. preparation: Nonselective Inversion (default: off)"),
		OPTL_DOUBLE(0, "TI", &conf->magn.ti, "TI", "Inversion time"),
		OPTL_DOUBLE(0, "init_delay", &conf->magn.init_delay, "init_delay", "Initial delay of measurement"),
		OPTL_DOUBLE(0, "inv_delay", &conf->magn.inv_delay_time, "inv_delay_time", "Inversion delay time"),

		// gradient mode
		OPTL_SELECT(0, "gradient-normal", enum gradient_mode, &seq_opts->gradient_mode, GRAD_NORMAL, "Gradient normal mode (default: fast)"),
		OPTL_SELECT(0, "gradient-whisper", enum gradient_mode, &seq_opts->gradient_mode, GRAD_WHISPER, "Gradient whispher mode (default: fast)"),

		OPTL_VECN(0, "CUSTOM_LONG", seq_opts->custom_params_long, "custom long parameters"),
		OPTL_DOVECN(0, "CUSTOM_DOUBLE", seq_opts->custom_params_double, "custom double parameters"),
		OPTL_VECN(0, "LOOP", conf->loop_dims, "sequence loop dimensions"),
	};

	cmdline(argcp, argv, m, args, help_str, ARRAY_SIZE(opts), opts);

}


static int error_catcher2(void fun(int* argcp, char* argv[*argcp], int m, const struct arg_s args[m], const char* help_str, int n, const struct opt_s opts[n]), 
					int* argcp, char* argv[*argcp], int m, const struct arg_s args[m], const char* help_str, int n, const struct opt_s opts[n])
{
	int ret = -1;

	error_jumper.initialized = true;

	if (0 == setjmp(error_jumper.buf)) {

		fun(argcp, argv, m, args, help_str, n, opts);
		ret = 0;
	}

	error_jumper.initialized = false;

	return ret;
}


int read_config_from_str(struct seq_config* seq, int N, const char* buffer_in)
{
	static const char help[] = "commandline for IDEA\n";

	const char* dummy = NULL;

	struct arg_s args[] = {

		ARG_OUTFILE(false, &dummy, "dummy"),
	};


	long custom_params_long[SEQ_MAX_PARAMS_LONG] = { 0 };
	double custom_params_double[SEQ_MAX_PARAMS_DOUBLE] = { 0. };


	const struct opt_s opts[] = {


		// FOV and resolution
		OPTL_DOUBLE(0, "FOV", &seq->geom.fov, "FOV", "Field Of View [mm]"),
		OPTL_PINT(0, "BR", &seq->geom.baseres, "BR", "Base Resolution"),
		OPTL_DOUBLE(0, "slice_thickness", &seq->geom.slice_thickness, "slice_thickness", "Slice thickness [mm]"),

		// basic sequence parameters
		OPTL_DOUBLE(0, "FA", &seq->phys.flip_angle, "flip angle", "Flip angle [deg]"),
		OPTL_DOUBLE(0, "TR", &seq->phys.tr, "TR", "TR [us]"),
		OPTL_DOUBLE(0, "TE", &seq->phys.te, "TE", "TE"),
		OPTL_DOUBLE(0, "TE_delta", &seq->phys.te_delta, "TE_delta", "TE_delta"),
		OPTL_DOUBLE(0, "BWTP", &seq->phys.bwtp, "BWTP", "Bandwidth Time Product"),

		// others sequence parameters
		OPTL_DOUBLE(0, "rf_duration", &seq->phys.rf_duration, "rf_duration", "RF pulse duration [us]"),
		OPTL_DOUBLE(0, "dwell", &seq->phys.dwell, "dwell", "Dwell time [us]"),

		// encoding
		OPTL_UINT(0, "pe_mode", &seq->enc.pe_mode, "pe_mode", "Phase-encoding mode"),
		OPTL_PINT(0, "tiny", &seq->enc.tiny, "tiny", "Tiny golden-ratio index"),


		OPTL_LONG('e', "echoes", &seq->loop_dims[TE_DIM], "echoes", "Number of echoes"),
		OPTL_LONG('r', "lines", &seq->loop_dims[PHS1_DIM], "lines", "Number of phase encoding lines"),
		OPTL_LONG('z', "partitions", &seq->loop_dims[PHS2_DIM], "partitions", "Number of partitions (3D) or SMS groups (2D)"),
		OPTL_LONG('t', "measurements", &seq->loop_dims[TIME_DIM], "measurements", "Number of measurements / frames"),
		OPTL_LONG('m', "slices", &seq->loop_dims[SLICE_DIM], "slices", "Number of slices of multiband factor (SMS)"),
		OPTL_LONG('i', "inversions", &seq->loop_dims[BATCH_DIM], "inversions", "Number of inversions"),
		OPTL_LONG(0, "physio", &seq->loop_dims[TIME2_DIM], "physio", "Number of physio phases"),
		OPTL_LONG(0, "averages", &seq->loop_dims[AVG_DIM], "averages", "Number of averages"),


		// sms
		OPTL_PINT(0, "mb_factor", &seq->geom.mb_factor, "mb_factor", "Multi-band factor"),

		// magnetization preparation
		OPTL_UINT(0, "mag_prep", &seq->magn.mag_prep, "mag_prep", "Magn. preparation [OFF, IR_SEL, IR_NON, SR_SEL, SR_NON, SR_ADIAB]"),
		OPTL_DOUBLE(0, "TI", &seq->magn.ti, "TI", "Inversion time [us]"),
		OPTL_UINT(0, "contrast", &seq->phys.contrast, "spoiling", "Spoiling [RF_RANDOM,RF_SPOILED,BALANCED,GSTF_RANDOM,GSTF_SPOILED]"),
		OPTL_DOUBLE(0, "inv_delay", &seq->magn.inv_delay_time, "inv_delay_time", "Inversion delay time (seconds)/Flat top test gradient (GIRF)"),
		OPTL_DOUBLE(0, "init_delay", &seq->magn.init_delay, "init_delay", "Initial delay of measurement (seconds)"),


		OPTL_VECN(0, "CUSTOM_LONG", custom_params_long, "custom long parameters"),
		OPTL_DOVECN(0, "CUSTOM_DOUBLE", custom_params_double, "custom double parameters"),
		OPTL_VECN(0, "LOOP", seq->loop_dims, "sequence loop dimensions"),
	};

	char* buffer = xstrdup(buffer_in);

	char *token = strtok(buffer, " \t");

	char* argv[N + 1];

	int i = 0;

	while (token != NULL) {

		argv[i++] = token;
		token = strtok(NULL, " \t");
	}

	int a = error_catcher2(cmdline, &i, argv, ARRAY_SIZE(args), args, help, ARRAY_SIZE(opts), opts);

	free(buffer);	

	if (custom_params_long[0] > 0)
		seq_ui_interface_custom_params(0, seq, SEQ_MAX_PARAMS_LONG, custom_params_long,
					       SEQ_MAX_PARAMS_DOUBLE, custom_params_double);

	if (0 > a)
		return 0;

	return 1;
}
