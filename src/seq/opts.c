/* Copyright 2025-2026. Insitute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <setjmp.h>

#include "misc/debug.h"
#include "misc/opts.h"
#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/cppmap.h"

#include "seq/config.h"
#include "seq/helpers.h"
#include "seq/misc.h"


#include "opts.h"

#define SEQ_LIST FLASH, MINIFLASH, ()

const char* seq_table[] = {
#define DENTRY(x) # x ,
	MAP(DENTRY, SEQ_LIST)
#undef  DENTRY
	NULL
};

const struct seq_opts seq_opts_defaults = {

	.dt = -1.,
	.samples = -1,
	.rel_shift = { },
	.raga_full_frames = -1,
	.dist = 1.,
	.label_slice_shift = { },

	.gradient_mode = GRAD_FAST,

	.chrono = false,
	.support = false,
	.stats = false,

	.raga_file = NULL,
	.shapes_file = NULL,

	.custom_params_long = { 0 },
	.custom_params_double = { 0. },
};


static void seq_process_options(struct seq_config* conf, struct seq_opts* seq_opts);

static bool help_func_seq(void* ptr, char c, const char* /*optarg*/)
{
	int N = 1024;
	char info[N];

	int ctr = 0;

	ctr += snprintf(info + ctr, (size_t)(N - ctr), "\nAvailable sequences are:\n");

	for (int i = 0; i < (int)ARRAY_SIZE(seq_table); i++) {

		if (NULL != seq_table[i])
			ctr += snprintf(info + ctr, (size_t)(N - ctr), "\t - %s\n", seq_table[i]);
	}

	ctr += snprintf(info + ctr, (size_t)(N - ctr), "\n");

	if ('a' == c) {

		memcpy(ptr, info, sizeof(info));
		return true;

	} else {

		printf("%s", info);
		exit(0);
	}
}

int seq_cmdline(int* argcp, char* argv[*argcp], int m, const struct arg_s args[m],
			const char* help_str, struct seq_config* conf, struct seq_opts* seq_opts,
			int len, char* buf)
{
	int off = 0;

	if (0 == len) {

		enum seq_type seq_type = 0;

		for (int i = 0; NULL != seq_table[i]; i++)
			for (int j = 0; j < *argcp; j++)
				if ((NULL != argv[j]) && (0 == strcmp(argv[j], seq_table[i])))
					seq_type = (enum seq_type)(i + 1);

		if (0 == seq_type) {

			debug_printf(DP_INFO, "No supported sequence found: Set to FLASH and move outputs by one\n");
			seq_type = SEQ_TYPE_FLASH;
			off = 1;
		}

		conf->seq_type = seq_type;

		if (SEQ_TYPE_MINIFLASH == conf->seq_type)
			memcpy(conf, &seq_config_defaults_miniflash, sizeof *conf);
	}

	const struct opt_s opts[] = {

		{ 'L', NULL, false, OPT_SPECIAL, help_func_seq, NULL, "", "(Print a list of supported sequences)" },

		OPT_DOUBLE('d', &seq_opts->dt, "dt", "time-increment per sample (default: seq->conf->phys.tr / 1000)"),
		OPT_LONG('N', &seq_opts->samples, "samples", "Number of samples (default: 1000)"),

		OPT_DOVEC3('s', &conf->geom.shift[0], "RO:PE:SL", "FOV shift"),
		OPT_DOVEC3('S', &seq_opts->rel_shift, "RO:PE:SL", "relative FOV shift"),
		OPTL_FLOAT(0, "dist", &seq_opts->dist, "dist", "slice distance factor [1 / slice_thickness] (default: 1.)"),
		OPT_DOVEC3('u', &seq_opts->label_slice_shift, "RO:PE:SL", "FOV shift of ASL label slice"),

		// contrast mode
		OPTL_UINT(0, "contrast", &conf->phys.contrast, "contrast", "(Spoiling [RF_RANDOM,RF_SPOILED,BALANCED,GSTF_RANDOM,GSTF_SPOILED])"),

		OPTL_SELECT(0, "no-spoiling", enum flash_contrast, &conf->phys.contrast,
				SEQ_CONTRAST_NO_SPOILING, "spoiling off (default: rf random)"),
		OPTL_SELECT(0, "flash", enum flash_contrast, &conf->phys.contrast,
				SEQ_CONTRAST_RF_RANDOM, "FLASH (RF_RANDOM: random phase, no gradient spoiling) (default)"),
		OPTL_SELECT(0, "random", enum flash_contrast, &conf->phys.contrast,
				SEQ_CONTRAST_RF_RANDOM, "(RF_RANDOM (phase: random, no gradient) (default))"),
		OPTL_SELECT(0, "spoiled", enum flash_contrast, &conf->phys.contrast,
				SEQ_CONTRAST_RF_SPOILED, "RF_SPOILED (inc: 50 deg, gradient on) (default: rf random)"),

		// FOV and resolution
		OPTL_DOUBLE(0, "FOV", &conf->geom.fov, "FOV", "Field Of View"),
		OPTL_PINT(0, "BR", &conf->geom.baseres, "BR", "Base Resolution"),
		OPTL_DOUBLE(0, "slice_thickness", &conf->geom.slice_thickness, "slice_thickness", "Slice thickness"),
		OPTL_DOUBLE(0, "slab_os", &conf->geom.slab_os, "slab_os", "Slab oversampling in partition dimension (default: 1.0 = no oversampling)"),

		// basic sequence parameters
		OPTL_DOUBLE(0, "FA", &conf->phys.flip_angle, "flip angle", "Flip angle [deg]"),
		OPTL_DOUBLE(0, "TR", &conf->phys.tr, "TR", "TR"),
		OPTL_DOUBLE(0, "TE", &conf->phys.te, "TE", "TE"),
		OPTL_DOUBLE(0, "TE_delta", &conf->phys.te_delta, "TE_delta", "TE_delta"),
		OPTL_DOUBLE(0, "BWTP", &conf->phys.bwtp, "BWTP", "Bandwidth Time Product"),

		// others sequence parameters
		OPTL_DOUBLE(0, "rf_duration", &conf->phys.rf_duration, "rf_duration", "RF pulse duration"),
		OPTL_DOUBLE(0, "dwell", &conf->phys.dwell, "dwell", "Dwell time"),
		OPTL_DOUBLE(0, "asym_echo", &conf->phys.asym_echo, "asym_echo", "Asymmetric echo [default = 0.5]"),
		OPTL_DOUBLE(0, "os", &conf->phys.os, "os", "Oversampling factor"),

		//dimension
		OPTL_INT(0, "is3D", &conf->enc.is3D, "is3D", "3D sequence flag"),

		// encoding
		OPTL_UINT(0, "pe_mode", &conf->enc.pe_mode, "pe_mode", "(Phase-encoding mode)"),
		OPTL_SELECT(0, "turn", enum pe_mode, &conf->enc.pe_mode, SEQ_PEMODE_TURN, "turn-based PE (default: RAGA)"),
		OPTL_SELECT(0, "mems", enum pe_mode, &conf->enc.pe_mode, SEQ_PEMODE_MEMS_HYB, "multi-echo/multi-spoke PE (default: RAGA)"),
		OPTL_SELECT(0, "raga", enum pe_mode, &conf->enc.pe_mode, SEQ_PEMODE_RAGA, "RAGA PE"),
		OPTL_SELECT(0, "cartesian", enum pe_mode, &conf->enc.pe_mode, SEQ_PEMODE_CARTESIAN, "Cartesian center-out PE (default: RAGA)"),
		OPTL_SELECT(0, "cartesian_linear", enum pe_mode, &conf->enc.pe_mode, SEQ_PEMODE_CARTESIAN_LINEAR, "Cartesian linear PE (default: RAGA)"),
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
		OPTL_LONG('a', "averages", &conf->loop_dims[AVG_DIM], "averages", "Number of averages"),

		// order
		OPTL_SELECT(0, "sequential-multislice", enum seq_order, &conf->enc.order, SEQ_ORDER_SEQ_MS, "seq_order: sequential multislice (default: avg outer)"),
		OPTL_SELECT(0, "avg-inner", enum seq_order, &conf->enc.order, SEQ_ORDER_AVG_INNER, "seq_order: average inner (default: avg outer)"),

		// sms
		OPTL_PINT(0, "mb_factor", &conf->geom.mb_factor, "mb_factor", "Multi-band factor"),
		OPTL_DOUBLE(0, "sms_distance", &conf->geom.sms_distance, "sms_distance", "SMS slice distance"),

		// magnetization preparation
		OPTL_UINT(0, "mag_prep", &conf->magn.mag_prep, "mag_prep", "Magn. preparation [OFF, IR_SEL, IR_NON, SR_SEL, SR_NON, SR_ADIAB]"),
		OPTL_SELECT(0, "IR_NON", enum mag_prep, &conf->magn.mag_prep, SEQ_PREP_IR_NONSELECTIVE, "Magn. preparation: Nonselective Inversion (default: off)"),
		OPTL_SELECT(0, "IR_SEL", enum mag_prep, &conf->magn.mag_prep, SEQ_PREP_IR_SELECTIVE, "Magn. preparation: Selective Inversion (default: off)"),
		OPTL_DOUBLE(0, "TI", &conf->magn.ti, "TI", "Inversion time"),
		OPTL_DOUBLE(0, "init_delay", &conf->magn.init_delay, "init_delay", "Initial delay of measurement"),
		OPTL_DOUBLE(0, "inv_delay", &conf->magn.inv_delay_time, "inv_delay_time", "Inversion delay time"),
		OPTL_LONG(0, "prep_scans", &conf->magn.prep_scans, "prep_scans", "Preparation scans"),

		// gradient mode
		OPTL_SELECT(0, "gradient-normal", enum gradient_mode, &seq_opts->gradient_mode, GRAD_NORMAL, "Gradient normal mode (default: fast)"),
		OPTL_SELECT(0, "gradient-whisper", enum gradient_mode, &seq_opts->gradient_mode, GRAD_WHISPER, "Gradient whispher mode (default: fast)"),

		//trigger
		OPTL_SELECT(0, "trigger", enum trigger_type, &conf->trigger.type, SEQ_TRIGGER_ECG, "Triggering (ECG)"),
		OPTL_DOUBLE(0, "trigger-delay", &conf->trigger.delay_time, "trigger.delay_time", "Trigger delay"),

		//CEST
		OPTL_SELECT(0, "cest-gauss", enum cest_saturation_type, &conf->cest.sat_type, SEQ_CEST_GAUSS, "CEST with gaussian sat. pulses"),
		OPTL_SELECT(0, "cest-oc", enum cest_saturation_type, &conf->cest.sat_type, SEQ_CEST_OC, "CEST with optimal control sat. pulses"),
		OPTL_LONG(0, "cest-sat-pulses", &conf->cest.sat_pulses, "saturation pulses", "Saturation pulses"),
		OPTL_DOUBLE(0, "cest-sat-pause", &conf->cest.sat_pulse_pause, "sat pulse pause", "Pause between saturation pulses"),

		OPTL_DOUBLE(0, "gauss-dur", &conf->cest.gauss_pulse_duration, "gauss duration", "Gauss saturation pulse duration"),
		OPTL_DOUBLE(0, "gauss-fa", &conf->cest.gauss_pulse_fa, "gauss fa", "Gauss saturation pulse flip anlge"),
		OPTL_DOUBLE(0, "oc-scale", &conf->cest.oc_pulse_b1_scaling, "oc scaling", "OC saturation pulse scaling"),

		OPTL_SELECT(0, "cest-offsets-pha", enum cest_offset_type, &conf->cest.offset_type, SEQ_CEST_OFFSET_PHANTOM, "CEST with custom offsets for phantom (default: equidistant)"),
		OPTL_SELECT(0, "cest-offsets-invivo", enum cest_offset_type, &conf->cest.offset_type, SEQ_CEST_OFFSET_INVIVO, "CEST with custom offsets for invivo (default: equidistant)"),
		OPTL_DOUBLE(0, "cest-offset-first", &conf->cest.offset_first, "cest offset first", "CEST offset first [ppm]"),
		OPTL_DOUBLE(0, "cest-offset-last", &conf->cest.offset_last, "cest offset last", "CEST offset last [ppm]"),
		OPTL_DOUBLE(0, "cest-offset-increment", &conf->cest.offset_increment, "cest offset increment", "CEST offset increment [ppm]"),
		OPTL_DOUBLE(0, "cest-offset-pause", &conf->cest.offset_pause, "cest offset pause", "CEST offset pause"),

		// ASL
		OPTL_UINT(0, "asl", &conf->asl.label_type, "asl", "ASL mode (0: NONE 1: PCASL) (default: NONE)"),
		OPTL_DOUBLE(0, "LD", &conf->asl.ld, "LD", "PCASL labeling duration"),
		OPTL_DOUBLE(0, "PLD", &conf->asl.pld, "PLD", "Post-labeling delay"),
		OPTL_INT(0, "asl_label_slice", &conf->asl.label_slice_index, "asl_label_slice", "Chronological index of labeling slice"), // mandatory in sequence

		OPTL_SET(0, "support", &seq_opts->support, "save support points of gradient"),

		OPTL_VECN(0, "CUSTOM_LONG", seq_opts->custom_params_long, "custom long parameters"),
		OPTL_DOVECN(0, "CUSTOM_DOUBLE", seq_opts->custom_params_double, "custom double parameters"),
		OPTL_VECN(0, "LOOP", conf->loop_dims, "sequence loop dimensions"),

		OPTL_SET(0, "stats", &seq_opts->stats, "Statistics / check of sequence"),

		OPT_OUTFILE('F', &seq_opts->shapes_file, "file", "RF Shapes file"),
	};


	if (0 != len)
		return cmdline_synth(NULL, len, buf, ARRAY_SIZE(opts), opts);

	char seqs[1024];
	help_func_seq(&seqs, 'a', NULL);

	char help_str2[sizeof(help_str) + sizeof(seqs) + 1];
	snprintf(help_str2, sizeof(help_str2), "%s\n%s", help_str, seqs);

	cmdline(argcp, argv, m - off, args + off, help_str2, ARRAY_SIZE(opts), opts);

	seq_process_options(conf, seq_opts);

	return 0;
}

static void seq_process_options(struct seq_config* conf, struct seq_opts* seq_opts)
{
	if (0 > seq_opts->samples)
		seq_opts->samples = (0. > seq_opts->dt) ? 1000 : (conf->phys.tr / seq_opts->dt);

	seq_opts->dt = (0 > seq_opts->dt) ? conf->phys.tr / seq_opts->samples : ceil(seq_opts->dt * 1.E6) / 1.E6;

	if (   (SEQ_PEMODE_RAGA != conf->enc.pe_mode)
	    && ! ((SEQ_PEMODE_CARTESIAN == conf->enc.pe_mode) || (SEQ_PEMODE_CARTESIAN_LINEAR == conf->enc.pe_mode)))
		seq_opts->chrono = true;

	// FIXME, this should be moved in system configurations
	switch (seq_opts->gradient_mode) {

	case GRAD_NORMAL:
		conf->sys.grad.max_amplitude = 22.E-3;
		conf->sys.grad.inv_slew_rate = 10.E-3 * sqrt(2.);
		break;

	case GRAD_WHISPER:
		conf->sys.grad.max_amplitude = 22.E-3;
		conf->sys.grad.inv_slew_rate = 20.E-3 * sqrt(2.);
		break;

	case GRAD_FAST:
		break;
	}

	if (seq_opts->custom_params_long[0] > 0)
		seq_ui_interface_custom_params(0, conf, SEQ_MAX_PARAMS_LONG, seq_opts->custom_params_long,
					SEQ_MAX_PARAMS_DOUBLE, seq_opts->custom_params_double);

	if (   (SEQ_PEMODE_RAGA == conf->enc.pe_mode)
	    && (1 == conf->loop_dims[TIME_DIM])
	    && (conf->loop_dims[TIME_DIM] < conf->loop_dims[PHS1_DIM])) {

		if (0 < seq_opts->raga_full_frames)
			conf->loop_dims[TIME_DIM] = seq_opts->raga_full_frames * conf->loop_dims[PHS1_DIM];

		if (1 == conf->loop_dims[TIME_DIM]) {

			debug_printf(DP_INFO, "Set total number of spokes to %ld (full frame for RAGA encoding)\n", conf->loop_dims[PHS1_DIM]);
			conf->loop_dims[TIME_DIM] = conf->loop_dims[PHS1_DIM];
		}
	}

	seq_ui_interface_loop_dims(0, conf, DIMS, conf->loop_dims);

	const long total_slices = get_slices(conf);

	if ((0. < fabs(seq_opts->rel_shift[0])) || (0. < fabs(seq_opts->rel_shift[1])) || (0. < fabs(seq_opts->rel_shift[2]))) {

		if ((0. < fabs(conf->geom.shift[0][0])) || (0. < fabs(conf->geom.shift[0][1])) || (0. < fabs(conf->geom.shift[0][2])))
			error("Choose either relative or absolute FOV shift");

		double slab = conf->geom.slice_thickness;
		if (conf->enc.is3D)
			slab = conf->geom.slice_thickness * conf->loop_dims[PHS2_DIM] / conf->geom.slab_os;

		for (int i = 0; i < total_slices; i++) {

			conf->geom.shift[i][0] = seq_opts->rel_shift[0] * conf->geom.fov;
			conf->geom.shift[i][1] = seq_opts->rel_shift[1] * conf->geom.fov;
			conf->geom.shift[i][2] = seq_opts->rel_shift[2] * slab;
		}
	}

	if ((1 < total_slices) && (0. < seq_opts->dist)) {

		float shift[4 * total_slices][3] = { }; // also includes 3x3 rotation matrix
		float init_shift = conf->geom.shift[0][2];

		for (int i = 0; i < total_slices; i++) {

			shift[i][0] = conf->geom.shift[0][0];
			shift[i][1] = conf->geom.shift[0][1];
			shift[i][2] = init_shift + (i - 0.5 * (total_slices - 1)) * seq_opts->dist * conf->geom.slice_thickness;
		}

		seq_set_fov_pos(total_slices, 3, &shift[0][0], conf);

		debug_printf(DP_INFO, "slice shifts:\n\t%d %f \t\n", 0, conf->geom.shift[0][2]);

		for (int i = 1; i < total_slices; i++)
			debug_printf(DP_INFO, "\t%d: %f \n", i, conf->geom.shift[i][2]);

		debug_printf(DP_INFO, "\n");
	}

	if (SEQ_ASL_NONE != conf->asl.label_type) {
		
		conf->loop_dims[SLICE_DIM] = conf->loop_dims[SLICE_DIM] + 1;  // add label slice
		conf->asl.label_slice_index = conf->loop_dims[SLICE_DIM] - 1; // set last slice as label slice
		conf->geom.shift[conf->asl.label_slice_index][2] = seq_opts->label_slice_shift[2];

		debug_printf(DP_INFO, "ASL label slice shift:\n\t%d %f \t\n", 0, conf->geom.shift[conf->asl.label_slice_index][2]);
	}

	if ((NULL != seq_opts->raga_file) && seq_opts->chrono)
		error("RAGA indices only for raga pe mode and non chronologic mode\n");
}



int seq_cmdline_print(int len, char* buf, const struct seq_config* conf, struct seq_opts* seq_opts)
{
	struct seq_config conf2;
	memcpy(&conf2, conf, sizeof(struct seq_config));

	// revert before writing cmdline seq-tool uses forward of seq_ui_interface_loop_dims
	seq_ui_interface_loop_dims(1, &conf2, DIMS, conf2.loop_dims);

	int argcp = 0; // UBSan
	return seq_cmdline(&argcp, NULL, 0, NULL, NULL, &conf2, seq_opts, len, buf);
}


static int error_catcher2(int fun(int* argcp, char* argv[*argcp], int m, const struct arg_s args[m], const char* help_str, struct seq_config* conf, struct seq_opts* seq_opts, int len, char* buf),
					int* argcp, char* argv[*argcp], int m, const struct arg_s args[m], const char* help_str, struct seq_config* conf, struct seq_opts* seq_opts, int len, char* buf)
{
	int ret = -1;

	error_jumper.initialized = true;

	if (0 == setjmp(error_jumper.buf)) {

		fun(argcp, argv, m, args, help_str, conf, seq_opts, len, buf);
		ret = 0;
	}

	error_jumper.initialized = false;

	return ret;
}


int read_config_from_str(struct seq_config* seq, int N, const char* buffer_in)
{
	static const char help[] = "commandline for IDEA\n";

	const char* seqtype = NULL;
	const char* dummy = NULL;

	struct arg_s args[] = {

		ARG_STRING(false, &seqtype, "seqtype"),
		ARG_OUTFILE(false, &dummy, "dummy"),
	};

	char* buffer = xstrdup(buffer_in);

	char *token = strtok(buffer, " \t");

	char* argv[N + 1];

	int i = 0;

	while (token != NULL) {

		argv[i++] = token;
		token = strtok(NULL, " \t");
	}

	struct seq_opts seq_opts = seq_opts_defaults;

	int a = error_catcher2(seq_cmdline, &i, argv, ARRAY_SIZE(args), args, help, seq, &seq_opts, 0, NULL);

	free(buffer);	

	if (seq_opts.custom_params_long[0] > 0)
		seq_ui_interface_custom_params(0, seq, SEQ_MAX_PARAMS_LONG, seq_opts.custom_params_long,
					       SEQ_MAX_PARAMS_DOUBLE, seq_opts.custom_params_double);

	if (0 > a)
		return 0;

	return 1;
}
