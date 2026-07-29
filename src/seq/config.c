/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "misc/mri.h"

#include "num/multind.h"

#include "config.h"

const int seq_loop_order_avg_inner[DIMS] = {
	TE_DIM,
	AVG_DIM,
	PHS2_DIM,
	SLICE_DIM,
	COEFF2_DIM,
	COEFF_DIM,
	PHS1_DIM,
	TIME2_DIM,
	TIME_DIM,
	BATCH_DIM,
	CSHIFT_DIM,

	READ_DIM,
	COIL_DIM,
	MAPS_DIM,
	ITER_DIM,
	LEVEL_DIM
};


const int seq_loop_order_avg_outer[DIMS] = {
	TE_DIM,
	PHS2_DIM,
	SLICE_DIM,
	COEFF2_DIM,
	COEFF_DIM,
	PHS1_DIM,
	AVG_DIM,
	TIME2_DIM,
	TIME_DIM,
	BATCH_DIM,
	CSHIFT_DIM,

	READ_DIM,
	COIL_DIM,
	MAPS_DIM,
	ITER_DIM,
	LEVEL_DIM
};

const int seq_loop_order_multislice[DIMS] = {
	TE_DIM,
	PHS2_DIM,
	COEFF2_DIM,
	COEFF_DIM,
	PHS1_DIM,
	AVG_DIM,
	TIME2_DIM,
	TIME_DIM,
	BATCH_DIM,
	CSHIFT_DIM,

	SLICE_DIM,
	READ_DIM,
	COIL_DIM,
	MAPS_DIM,
	ITER_DIM,
	LEVEL_DIM
};

const int seq_loop_order_asl[DIMS] = {
	PHS2_DIM,
	SLICE_DIM,
	COEFF2_DIM,
	COEFF_DIM,
	PHS1_DIM,
	TIME2_DIM,
	TIME_DIM,
	BATCH_DIM,
	AVG_DIM,

	READ_DIM,
	COIL_DIM,
	MAPS_DIM,
	TE_DIM,
	ITER_DIM,
	CSHIFT_DIM,
	LEVEL_DIM
};

void seq_copy_order(struct seq_config* seq)
{
	switch (seq->enc.order) {

	case SEQ_ORDER_AVG_OUTER:
		md_copy_order(DIMS, seq->order, seq_loop_order_avg_outer);
		break;

	case SEQ_ORDER_SEQ_MS:
		md_copy_order(DIMS, seq->order, seq_loop_order_multislice);
		break;

	case SEQ_ORDER_AVG_INNER:
		md_copy_order(DIMS, seq->order, seq_loop_order_avg_inner);
		break;

	case SEQ_ORDER_SEQ_ASL:
		md_copy_order(DIMS, seq->order, seq_loop_order_asl);
		break;
	}
}


const struct seq_config seq_config_defaults_flash = {

	.seq_type = SEQ_TYPE_FLASH,

	.phys = {
		.tr = 3.11E-3,
		.te = 1.90E-3,
		.te_delta = 2.0E-3,
		.dwell = 4.E-6,
		.os = 2.,
		.asym_echo = 0.5,
		.contrast = SEQ_CONTRAST_RF_RANDOM,
		.rf_duration = 620.E-6,
		.flip_angle = 6.,
		.bwtp = 3.8,
	},

	.geom = {
		.fov = .256,
		.slice_thickness = .006,
		.slab_os = 1.,
		.shift = { [0 ... SEQ_MAX_SLICES - 1] = { 0., 0., 0. } },
		.rot = { [0 ... SEQ_MAX_SLICES - 1] = { { 1., 0., 0. }, { 0., 1., 0. }, { 0., 0., 1. } } },
		.baseres = 256,
		.mb_factor = 1,
		.sms_distance = .020,
	},

	.enc = {
		.pe_mode = SEQ_PEMODE_RAGA,
		.tiny = 1,
		.aligned_flags = 0,
		.order = SEQ_ORDER_AVG_OUTER,
		.is3D = 0,
	},

	.magn = {
		.mag_prep = SEQ_PREP_OFF,
		.ti = 0.,
		.prep_scans = 0,
		.init_delay = 0.,
		.inv_delay_time = 0.,
	},

	.trigger = {
		.type = SEQ_TRIGGER_OFF,
		.delay_time = 0.,
		.pulses = 0,
		.trigger_out = 1,
	},

	.cest = {
		.sat_type = SEQ_CEST_NONE,
		.sat_pulses = 15,
		.sat_pulse_pause = 0.005,
		.gauss_pulse_duration = 0.025,
		.gauss_pulse_fa = 360.,
		.oc_pulse_b1_scaling = 1.,
		.offset_type = SEQ_CEST_OFFSET_EQUIDISTANT,
		.offset_first = -5.,
		.offset_last = 5.,
		.offset_increment = 0.2,
		.offset_pause = 3.5,
	},

	.asl = {
		.label_type = SEQ_ASL_NONE,
		.ld = 2.,
		.pld = 1.3,
		.hanning.rf_duration = 500.E-6,
		.hanning.flip_angle = 25.,
		.pulse_spacing = 1.15E-3,
		.ampl_grad_sli = 10E-3,
		.label_slice_index = 1,
	},

	.sys = {
		.gamma = 42.575575E6,
		.b0 = 2.893620,
		.grad.inv_slew_rate = .007848885540911,
		.grad.max_amplitude = .024,
		.coil_control_lead = 100.E-6,
		.min_duration_ro_rf = 213.E-6,
		.raster_grad = 1.E-5,
		.raster_rf = 1.E-6,
		.raster_dwell = 1.E-7,
	},

	.order = { [0 ... DIMS - 1] = 1 },
	.loop_dims = { [0 ... DIMS - 1] = 1 },
};


const struct seq_config seq_config_defaults_miniflash = {

	.seq_type = SEQ_TYPE_MINIFLASH,

	.phys = {
		.tr = 10.E-3,
		.te = 3.E-3,
		.dwell = 4.E-6,
		.os = 2.,
		.asym_echo = 0.5,
		.contrast = SEQ_CONTRAST_RF_SPOILED,
		.rf_duration = 1.E-3,
		.flip_angle = 6.,
		.bwtp = 3.8,
	},

	.geom = {
		.fov = .256,
		.slice_thickness = .006,
		.slab_os = 1.,
		.shift = { [0 ... SEQ_MAX_SLICES - 1] = { 0., 0., 0. } },
		.rot = { [0 ... SEQ_MAX_SLICES - 1] = { { 1., 0., 0. }, { 0., 1., 0. }, { 0., 0., 1. } } },
		.baseres = 256,
		.mb_factor = 1,
		.sms_distance = .020,
	},

	.enc = {
		.pe_mode = SEQ_PEMODE_CARTESIAN_LINEAR,
		.tiny = 1,
		.order = SEQ_ORDER_AVG_OUTER,
	},

	.magn = {
		.mag_prep = SEQ_PREP_OFF,
	},

	.trigger = {
		.type = SEQ_TRIGGER_OFF,
		.trigger_out = 1,
	},

	.cest = {
		.sat_type = SEQ_CEST_NONE,
	},

	.asl = {
		.label_type = SEQ_ASL_NONE,
	},

	.sys = {
		.gamma = 42.575575E6,
		.b0 = 2.893620,
		.grad.inv_slew_rate = .007848885540911,
		.grad.max_amplitude = .024,
		.coil_control_lead = 100.E-6,
		.min_duration_ro_rf = 213.E-6,
		.raster_grad = 1.E-5,
		.raster_rf = 1.E-6,
		.raster_dwell = 1.E-7,
	},

	.order = { [0 ... DIMS - 1] = 1 },
	.loop_dims = { [2 ... DIMS - 1] = 1 },
	.loop_dims[READ_DIM] = 1,
	.loop_dims[PHS1_DIM] = 256,
};
