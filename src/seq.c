/* Copyright 2025-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Daniel Mackner
 */

#include <complex.h>
#include <math.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/rand.h"

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/opts.h"

#include "seq/config.h"
#include "seq/event.h"
#include "seq/helpers.h"
#include "seq/seq.h"
#include "seq/opts.h"

#include "seq/misc.h"
#include "seq/flash.h"
#include "seq/kernel.h"
#include "seq/pulseq.h"


#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char help_str[] = "Computes a GRE sequence.";


int main_seq(int argc, char* argv[argc])
{
	double start_time = timestamp();

	const char* grad_file = NULL;
	const char* mom_file = NULL;
	const char* adc_file = NULL;
	const char* seq_file = NULL;

	struct arg_s args[] = {

		ARG_OUTFILE(false, &adc_file, "0th moment (x,y,z) at sample points, sample_points, phase of adc"),
		ARG_OUTFILE(false, &grad_file, "gradients (x,y,z) per imaging block"),
		ARG_OUTFILE(false, &mom_file, "0th moment (x,y,z) per imaging block"),
		ARG_OUTFILE(false, &seq_file,  "pulseq file"),
	};

	struct bart_seq* seq = bart_seq_alloc("");
	bart_seq_defaults(seq);


	struct seq_opts seq_opts = seq_opts_defaults;

	seq_cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, 
		seq->conf, &seq_opts, 0, NULL);


	if (seq_opts.custom_params_long[0] > 0)
		seq_ui_interface_custom_params(0, seq->conf, SEQ_MAX_PARAMS_LONG, seq_opts.custom_params_long,
					SEQ_MAX_PARAMS_DOUBLE, seq_opts.custom_params_double);

	if (   (SEQ_PEMODE_RAGA == seq->conf->enc.pe_mode)
	    && (1 == seq->conf->loop_dims[TIME_DIM])
	    && (seq->conf->loop_dims[TIME_DIM] < seq->conf->loop_dims[PHS1_DIM])) {

		if (0 < seq_opts.raga_full_frames)
			seq->conf->loop_dims[TIME_DIM] = seq_opts.raga_full_frames * seq->conf->loop_dims[PHS1_DIM];

		if (1 == seq->conf->loop_dims[TIME_DIM]) {

			debug_printf(DP_INFO, "Set total number of spokes to %ld (full frame for RAGA encoding)\n", seq->conf->loop_dims[PHS1_DIM]);
			seq->conf->loop_dims[TIME_DIM] = seq->conf->loop_dims[PHS1_DIM];
		}
	}


	seq_ui_interface_loop_dims(0, seq->conf, DIMS, seq->conf->loop_dims);

	const long total_slices = get_slices(seq->conf);

	if ((0. < fabs(seq_opts.rel_shift[0])) || (0. < fabs(seq_opts.rel_shift[1])) || (0. < fabs(seq_opts.rel_shift[2]))) {

		if ((0. < fabs(seq->conf->geom.shift[0][0])) || (0. < fabs(seq->conf->geom.shift[0][1])) || (0. < fabs(seq->conf->geom.shift[0][2])))
			error("Choose either relative or absolute FOV shift");

		for (int i = 0; i < total_slices; i++) {

			seq->conf->geom.shift[i][0] = seq_opts.rel_shift[0] * seq->conf->geom.fov;
			seq->conf->geom.shift[i][1] = seq_opts.rel_shift[1] * seq->conf->geom.fov;
			seq->conf->geom.shift[i][2] = seq_opts.rel_shift[2] * seq->conf->geom.slice_thickness;
		}
	}

	if ((1 < total_slices) && (0. < seq_opts.dist)) {

		float shift[total_slices][3];
		memset(shift, 0, sizeof shift);

		for (int i = 0; i < total_slices; i++) {

			shift[i][0] = seq->conf->geom.shift[0][0];
			shift[i][1] = seq->conf->geom.shift[0][1];
			shift[i][2] = (i - 0.5 * (total_slices - 1)) * seq_opts.dist * seq->conf->geom.slice_thickness;
		}

		seq_set_fov_pos(total_slices, 3, &shift[0][0], seq->conf);

		debug_printf(DP_INFO, "slice shifts:\n\t%d %f \t\n", 0, seq->conf->geom.shift[0][2]);
		for (int i = 1; i < total_slices; i++)
			debug_printf(DP_INFO, "\t%d: %f \n", i, seq->conf->geom.shift[i][2]);
		debug_printf(DP_INFO, "\n");
	}

	if ((NULL != seq_opts.raga_file) && seq_opts.chrono)
		error("RAGA indices only for raga pe mode and non chronologic mode\n");


	debug_printf(DP_INFO, "loops: %ld \t dims: ", md_calc_size(DIMS, seq->conf->loop_dims));
	debug_print_dims(DP_INFO, DIMS, seq->conf->loop_dims);

	long kernel_dims[DIMS];
	md_select_dims(DIMS, ~(COEFF_FLAG | COEFF2_FLAG | ITER_FLAG), kernel_dims, seq->conf->loop_dims);

	debug_printf(DP_INFO, "kernels: %ld \t dims: ", md_calc_size(DIMS, kernel_dims));
	debug_print_dims(DP_INFO, DIMS, kernel_dims);

	long mdims[DIMS];
	md_select_dims(DIMS, ~TE_FLAG, mdims, kernel_dims);

	int E = 0;

	if (seq_opts.support) {

		seq->state->mode = SEQ_BLOCK_KERNEL_PREPARE;

		E = seq_block(seq->N, seq->event, seq->state, seq->conf);

		seq->state->mode = SEQ_BLOCK_UNDEFINED;

		for (int i = 0; i < DIMS; i++)
			seq->state->pos[i] = 0;

		assert(NULL == mom_file);
	}

	mdims[PHS2_DIM] *= mdims[PHS1_DIM];
	mdims[PHS1_DIM] = seq_opts.support ? events_counter(SEQ_EVENT_GRADIENT, E, seq->event) : seq_opts.samples;
	mdims[READ_DIM] = seq_opts.support ? 6 : 3;

	double g2[seq_opts.samples][mdims[READ_DIM]];
	float m0[seq_opts.samples][3];

	long mstrs[DIMS];
	md_calc_strides(DIMS, mstrs, mdims, CFL_SIZE);

	long adims[DIMS];
	md_copy_dims(DIMS, adims, kernel_dims);

	adims[PHS2_DIM] *= adims[PHS1_DIM]; // consistency with traj tool
	adims[PHS1_DIM] = lround(seq->conf->geom.baseres * seq->conf->phys.os * (0.5 + seq->conf->phys.asym_echo));
	adims[READ_DIM] = 5;

	long adc_dims[DIMS];
	md_select_dims(DIMS, (READ_FLAG | PHS1_FLAG | TE_FLAG), adc_dims, adims);

	long adc_strs[DIMS];
	md_calc_strides(DIMS, adc_strs, adc_dims, CFL_SIZE);

	long astrs[DIMS];
	md_calc_strides(DIMS, astrs, adims, CFL_SIZE);

	long ind_dims[DIMS];
	md_select_dims(DIMS, ~(READ_FLAG | PHS1_FLAG), ind_dims, adims);

	long ind_strs[DIMS];
	md_calc_strides(DIMS, ind_strs, ind_dims, CFL_SIZE);

	complex float* out_grad = NULL;
	complex float* out_mom = NULL;
	complex float* out_adc = NULL;
	complex float* out_raga = NULL;

	if (NULL != grad_file)
		out_grad = create_cfl(grad_file, DIMS, mdims);

	if (NULL != mom_file)
		out_mom = create_cfl(mom_file, DIMS, mdims);

	if (NULL != adc_file)
		out_adc = create_cfl(adc_file, DIMS, adims);

	if (NULL != seq_opts.raga_file) {

		out_raga = create_cfl(seq_opts.raga_file, DIMS, ind_dims);
		md_clear(DIMS, ind_dims, out_raga, CFL_SIZE);
	}

	struct pulseq ps;

	int prepped_rfs = bart_seq_prepare(seq);

	char radial_info[300];
	seq_print_info_radial_views(300, radial_info, seq->conf);

	if (0 > prepped_rfs) {

		if (ERROR_SETTING_SPOKES_RAGA == prepped_rfs) {

			debug_printf(DP_WARN, "%s\n", radial_info);
		}

		error("Sequence preparation failed! - check seq_config, %s [ %d ] \n", error_string(prepped_rfs), prepped_rfs);
	}

	debug_printf(DP_DEBUG1, "%s\n", radial_info);

	char config_info[7852];
	seq_print_info_config(7852, config_info, seq->conf);
	debug_printf(DP_DEBUG1, "%s\n", config_info);

	debug_printf(DP_INFO, "Nr. of RF shapes: %d\n", prepped_rfs);

	for (int i = 0; i < prepped_rfs; i++) {

		double s = seq_pulse_scaling(&seq->rf_shape[i]);
		double n = seq_pulse_norm_sum(&seq->rf_shape[i]);

		debug_printf(DP_DEBUG3, "RF pulse %d: scale = %f, sum = %f\n", i, s, n);
	}

	if (NULL != seq_file) {

		pulseq_init(&ps, seq->conf);
		pulse_shapes_to_pulseq(&ps, prepped_rfs, seq->rf_shape);
	}

	do {
		debug_print_dims(DP_DEBUG2, DIMS, seq->state->pos);

		E = seq_block(seq->N, seq->event, seq->state, seq->conf);

		if (0 < E)
			debug_printf(DP_DEBUG2, "block mode: %d ; E: %d \n", seq->state->mode, E);


		if (0 > E)
			error("Sequence execution failed! - check seq_config, %s [ %d ] \n", error_string(E), E);

		if ((SEQ_BLOCK_KERNEL_NOISE == seq->state->mode) || (0 == E)) // no noise_scan with pulseq
			goto debug_print_events;

		if (NULL != seq_file)
			events_to_pulseq(&ps, seq->state->mode, seq->conf->phys.tr, seq->conf->sys, prepped_rfs, seq->rf_shape, E, seq->event);

		if (SEQ_BLOCK_KERNEL_IMAGE != seq->state->mode)
			goto debug_print_events;

		debug_printf(DP_DEBUG2, "end of last event: %.8f \t end of calc: %.8f\n",
				events_end_time(E, seq->event, 1, 0), seq_opts.samples * seq_opts.dt);

		if (seq_opts.support)
			seq_gradients_support(seq_opts.samples, g2, E, seq->event);
		else
			seq_compute_gradients(seq_opts.samples, g2, seq_opts.dt, E, seq->event);

		seq_compute_moment0(seq_opts.samples, m0, seq_opts.dt, E, seq->event);


		long pos_save[DIMS]; // FIXME use separate function
		md_copy_dims(DIMS, pos_save, seq->state->pos);

		// revert incomplete RAGA frame handling from flash()
		if (   (SEQ_PEMODE_RAGA == seq->conf->enc.pe_mode)
		    && (seq->conf->loop_dims[TIME_DIM] - 1 == pos_save[TIME_DIM])
		    && (seq->conf->loop_dims[PHS1_DIM] - 1 == pos_save[PHS1_DIM]))
				pos_save[PHS1_DIM] = seq->conf->loop_dims[ITER_DIM] - 1;

		if (!seq_opts.chrono) {

			int adc_idx = events_idx(0, SEQ_EVENT_ADC, E, seq->event);

			if (0 > adc_idx)
				error("No ADC found - try chronologic ordering");

			if (NULL != out_raga) {

				pos_save[PHS2_DIM] = pos_save[PHS2_DIM] * seq->conf->loop_dims[PHS1_DIM] + pos_save[PHS1_DIM];
				pos_save[PHS1_DIM] = 0;
				MD_ACCESS(DIMS, ind_strs, pos_save, out_raga) = seq->event[adc_idx].adc.pos[PHS1_DIM];
			}

			md_copy_dims(DIMS, pos_save, seq->event[adc_idx].adc.pos);
		}

		pos_save[PHS2_DIM] = pos_save[PHS2_DIM] * seq->conf->loop_dims[PHS1_DIM] + pos_save[PHS1_DIM];
		pos_save[PHS1_DIM] = 0;

		do {
			if (NULL != out_grad)
				MD_ACCESS(DIMS, mstrs, pos_save, out_grad) = g2[pos_save[PHS1_DIM]][pos_save[READ_DIM]];

			if (NULL != out_mom)
				MD_ACCESS(DIMS, mstrs, pos_save, out_mom) = m0[pos_save[PHS1_DIM]][pos_save[READ_DIM]];

		} while (md_next(DIMS, mdims, (READ_FLAG | PHS1_FLAG), pos_save));

		if (NULL != out_adc) {

			complex float* adc = md_alloc(DIMS, adc_dims, CFL_SIZE);

			seq_compute_adc_samples(DIMS, adc_dims, adc, E, seq->event);

			float m0_adc[adc_dims[PHS1_DIM]][3];

			do {

				double adc_start = seq->event[events_idx(pos_save[TE_DIM], SEQ_EVENT_ADC, E, seq->event)].start;
				seq_compute_moment0_offset(adc_dims[PHS1_DIM], m0_adc, adc_start, seq->conf->phys.dwell / seq->conf->phys.os, E, seq->event);

				double scale = 1. / (seq->conf->geom.fov * seq->conf->sys.gamma);

				do {
					assert(0 == pos_save[READ_DIM]);

					MD_ACCESS(DIMS, astrs, (pos_save[READ_DIM] = 0, pos_save), out_adc) = m0_adc[pos_save[PHS1_DIM]][0] / scale;
					MD_ACCESS(DIMS, astrs, (pos_save[READ_DIM] = 1, pos_save), out_adc) = m0_adc[pos_save[PHS1_DIM]][1] / scale;
					MD_ACCESS(DIMS, astrs, (pos_save[READ_DIM] = 2, pos_save), out_adc) = m0_adc[pos_save[PHS1_DIM]][2] / scale;

					MD_ACCESS(DIMS, astrs, (pos_save[READ_DIM] = 3, pos_save), out_adc) = MD_ACCESS(DIMS, adc_strs, (pos_save[READ_DIM] = 0, pos_save), adc) + seq->state->start_block;
					MD_ACCESS(DIMS, astrs, (pos_save[READ_DIM] = 4, pos_save), out_adc) = MD_ACCESS(DIMS, adc_strs, (pos_save[READ_DIM] = 1, pos_save), adc);

					pos_save[READ_DIM] = 0;

				} while (md_next(DIMS, adims, PHS1_FLAG, pos_save));

			} while (md_next(DIMS, adims, TE_FLAG, pos_save));

			md_free(adc);
		}

debug_print_events:
		seq_linearize_events(E, seq->event, &seq->state->start_block, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad);

		for (int i = 0; i < E; i++) {

			debug_printf(DP_DEBUG3, "event[%d]:\t%.8f\t\t%.8f\t\t%.8f\t\t", i,
					seq->event[i].start, seq->event[i].mid, seq->event[i].end);

			switch (seq->event[i].type) {

			case SEQ_EVENT_GRADIENT:

				debug_printf(DP_DEBUG3, "||\t%.5f\t\t%.5f\t\t%.5f", seq->event[i].grad.ampl[0], seq->event[i].grad.ampl[1],seq->event[i].grad.ampl[2]);
				break;

			case SEQ_EVENT_PULSE:

				debug_printf(DP_DEBUG3, "|| SEQ_EVENT_PULSE \t freq: %.2f\t\t phase: %.2f", seq->event[i].pulse.freq, seq->event[i].pulse.phase);
				break;

			case SEQ_EVENT_ADC:

				debug_printf(DP_DEBUG3, "|| SEQ_EVENT_ADC \t freq: %.2f\t\t phase: %.2f", seq->event[i].adc.freq, seq->event[i].adc.phase);
				break;

			default:

			}

			debug_printf(DP_DEBUG3, "\n");
		}

		debug_printf(DP_DEBUG3, "seq_block_end_flat: %f\n", seq_block_end_flat(E, seq->event, seq->conf->sys.raster_grad));


	} while (seq_continue(seq->state, seq->conf));

	debug_printf(DP_INFO, "Sequence total duration: %.3f s (expected: %.3f)\n", seq->state->start_block, seq_total_measure_time(seq->conf));

	if (1.E-3 < fabs(seq->state->start_block - seq_total_measure_time(seq->conf)))
		debug_printf(DP_WARN, "Calculation of sequence duration invalid!\n");

	unmap_cfl(DIMS, mdims, out_grad);
	unmap_cfl(DIMS, mdims, out_mom);
	unmap_cfl(DIMS, adims, out_adc);
	unmap_cfl(DIMS, ind_dims, out_raga);

	if (NULL != seq_file) {

		FILE *fp = fopen(seq_file, "w+");

		if (NULL == fp)
			error("Opening file for .seq");

		pulseq_writef(fp, &ps);

		fclose(fp);
	}

	bart_seq_free(seq);

	double recosecs = timestamp() - start_time;

	debug_printf(DP_INFO, "Total Time: %.2f s\n", recosecs);

	return 0;
}

