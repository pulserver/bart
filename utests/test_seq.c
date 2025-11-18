/* Copyright 2025-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>
#include <stdio.h>

#include "num/multind.h"

#include "seq/config.h"
#include "seq/event.h"
#include "seq/kernel.h"
#include "seq/flash.h"
#include "seq/seq.h"
#include "seq/helpers.h"
#include "seq/custom_ui.h"

#include "utest.h"

#define FLASH_EVENTS 14


// those are actively used in sequence
static bool test_commands_sequence(void)
{
	struct bart_seq* seq = bart_seq_alloc("");
	bart_seq_defaults(seq);

	static char seq_cmd[128];
	snprintf(seq_cmd, 128, "bart seq --pe_mode 1 --contrast 2 --tiny 13");

	if (!seq_config_from_string(seq->conf, 128, seq_cmd))
		return false;

	if ((1 != seq->conf->enc.pe_mode) || (2 != seq->conf->phys.contrast) || (13 != seq->conf->enc.tiny))
		return false;

	bart_seq_free(seq);

	return true;
}

UT_REGISTER_TEST(test_commands_sequence);


// those custom ui values are used in the sequence
static bool test_get_ui_idx(void)
{
	if (0 > seq_custom_ui_get_idx("CMD"))
		return false;

	if (0 > seq_custom_ui_get_idx("PE_MODE"))
		return false;

	if (0 > seq_custom_ui_get_idx("CONTRAST"))
		return false;

	if (0 > seq_custom_ui_get_idx("TINY"))
		return false;

	if (0 > seq_custom_ui_get_idx("RECO"))
		return false;

	if (0 > seq_custom_ui_get_idx("RAGA_ALIGNED_FLAGS"))
		return false;

	return true;
}

UT_REGISTER_TEST(test_get_ui_idx);

static int trigger_event_count(const struct seq_config* seq, const struct seq_state* seq_state)
{
	return (seq->trigger.trigger_out && (0 == seq_state->pos[PHS1_DIM])) ? 1 : 0;
}

static bool test_block_minv_init_delay(void)
{
	const enum seq_block blocks[16] = {
		SEQ_BLOCK_PRE, SEQ_BLOCK_KERNEL_NOISE, SEQ_BLOCK_PRE,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_PRE,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE
	};

	struct bart_seq* seq = bart_seq_alloc("");
	bart_seq_defaults(seq);

	seq->conf->magn.init_delay = 1.;
	seq->conf->magn.ti = 100.E-3;
	seq->conf->magn.mag_prep = SEQ_PREP_IR_NONSELECTIVE;

	seq->conf->loop_dims[BATCH_DIM] = 2;
	seq->conf->loop_dims[SLICE_DIM] = 2;
	seq->conf->loop_dims[PHS1_DIM] = 3;
	seq->conf->loop_dims[TIME_DIM] = 3;
	seq_ui_interface_loop_dims(0, seq->conf, DIMS, seq->conf->loop_dims);

	int i = 0;
	int pre_blocks = 0;

	do {
		int E = seq_block(seq->N, seq->event, seq->state, seq->conf);

		if (0 > E)
			return false;

		if (0 == E)
			continue;

		if (blocks[i] != seq->state->mode)
			return false;

		if ((SEQ_BLOCK_KERNEL_IMAGE == seq->state->mode) && (FLASH_EVENTS + trigger_event_count(seq->conf, seq->state) != E))
			return false;

		if (SEQ_BLOCK_PRE == seq->state->mode)
			pre_blocks++;

		// correct delay_meas_time
		if (   (SEQ_BLOCK_PRE == seq->state->mode)
		    && (1 == E)
		    && (seq->conf->magn.init_delay != seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad)))
			return false;

		if (   (SEQ_BLOCK_PRE == seq->state->mode)
		    && (1 < E)
		    && (1.E-4 * UT_TOL < fabs(seq->conf->magn.ti - (seq->event[E - 1].end - seq->event[E - 1].start))))
			return false;

		i++;

	} while (seq_continue(seq->state, seq->conf));

	if (3 != pre_blocks)
		return false;

	if (16 != i)
		return false;

	bart_seq_free(seq);

	return true;
}

UT_REGISTER_TEST(test_block_minv_init_delay);


static bool test_block_minv_multislice(void)
{
	const enum seq_block blocks[21] = {
		SEQ_BLOCK_KERNEL_NOISE,
		SEQ_BLOCK_PRE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_POST,
		SEQ_BLOCK_PRE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_POST,
		SEQ_BLOCK_PRE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_POST,
		SEQ_BLOCK_PRE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_POST,
	};

	const double pulse_spoiler_duration = 20.4E-3;

	struct bart_seq* seq = bart_seq_alloc("");
	bart_seq_defaults(seq);

	seq->conf->enc.order = SEQ_ORDER_SEQ_MS;
	seq->conf->magn.ti = 100.E-3;
	seq->conf->magn.mag_prep = SEQ_PREP_IR_NONSELECTIVE;
	seq->conf->magn.inv_delay_time = 100.;

	seq->conf->loop_dims[BATCH_DIM] = 2;
	seq->conf->loop_dims[SLICE_DIM] = 2;
	seq->conf->loop_dims[PHS1_DIM] = 3;
	seq->conf->loop_dims[TIME_DIM] = 3;
	seq_ui_interface_loop_dims(0, seq->conf, DIMS, seq->conf->loop_dims);

	int i = 0;
	int inversions = 0;

	do {
		int E = seq_block(seq->N, seq->event, seq->state, seq->conf);

		if (0 > E)
			return false;

		if (0 == E)
			continue;

		if (blocks[i] != seq->state->mode)
			return false;

		if ((SEQ_BLOCK_KERNEL_IMAGE == seq->state->mode) && (FLASH_EVENTS + trigger_event_count(seq->conf, seq->state) != E))
			return false;

		if (SEQ_BLOCK_PRE == seq->state->mode)
			inversions++;

		// correct ti in mag_prep block
		if (   (SEQ_BLOCK_PRE == seq->state->mode)
		    && (1 < E)
		    && (1.E-4 * UT_TOL < fabs((seq->conf->magn.ti + pulse_spoiler_duration) - seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad))))
			return false;

		i++;

	} while (seq_continue(seq->state, seq->conf));

	if (4 != inversions)
		return false;

	if (21 != i)
		return false;

	bart_seq_free(seq);

	return true;
}

UT_REGISTER_TEST(test_block_minv_multislice);


static bool test_fov_shift(void)
{
	const int slices = 3;

	float in[3] = {-27, 0, 27};
	float good[3] = {0, 0, 0};


	float gui_shift[slices][4];

	for (int i = 0; i < slices; i++) {

		gui_shift[i][0] = 0;
		gui_shift[i][1] = 0;
		gui_shift[i][2] = 1.E-3 * in[i]; // slice shift
	}

	struct bart_seq* seq = bart_seq_alloc("");
	bart_seq_defaults(seq);

	seq->conf->geom.mb_factor = 3;
	seq->conf->loop_dims[SLICE_DIM] = slices;
	seq_ui_interface_loop_dims(0, seq->conf, DIMS, seq->conf->loop_dims);

	seq_set_fov_pos(slices, 4, &gui_shift[0][0], seq->conf);

	if (1.E-2 *  UT_TOL < fabs(seq->conf->geom.sms_distance - 27.E-3))
		return false;

	for (int i = 0; i < slices; i++)
		if (0 < fabs(seq->conf->geom.shift[i][2] - good[i]))
			return false;

	bart_seq_free(seq);

	return true;
}

UT_REGISTER_TEST(test_fov_shift);

static bool test_fov_shift3x3(void)
{
	const int slices = 9;
	float in[9] = {-36, -27, -18, -9, 0, 9, 18, 27, 36};
	float good[9] = {-9, 0, 9, -9, 0, 9, -9, 0, 9};

	float gui_shift[slices][4];

	for (int i = 0; i < slices; i++) {

		gui_shift[i][0] = 0;
		gui_shift[i][1] = 0;
		gui_shift[i][2] = 1.E-3 * in[i]; // slice shift
	}

	struct bart_seq* seq = bart_seq_alloc("");
	bart_seq_defaults(seq);

	seq->conf->geom.mb_factor = 3;
	seq->conf->loop_dims[SLICE_DIM] = slices;
	seq_ui_interface_loop_dims(0, seq->conf, DIMS, seq->conf->loop_dims);

	seq_set_fov_pos(slices, 4, &gui_shift[0][0], seq->conf);

	if (1.E-2 *  UT_TOL < fabs(seq->conf->geom.sms_distance - 27.E-3))
		return false;

	for (int i = 0; i < slices; i++)
		if (1.E-2 *  UT_TOL < fabs(seq->conf->geom.shift[i][2] - 1.E-3 * good[i]))
			return false;

	bart_seq_free(seq);

	return true;
}

UT_REGISTER_TEST(test_fov_shift3x3);


static bool test_block_ecg(void)
{
	const enum seq_block blocks[17] = {
		SEQ_BLOCK_KERNEL_NOISE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_PRE, SEQ_BLOCK_PRE,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE
	};

	struct bart_seq* seq = bart_seq_alloc("");
	bart_seq_defaults(seq);

	seq->conf->magn.mag_prep = SEQ_PREP_IR_NONSELECTIVE;
	seq->conf->trigger.type = SEQ_TRIGGER_ECG;
	seq->conf->trigger.delay_time = 421.E-3;

	seq->conf->loop_dims[BATCH_DIM] = 2;
	seq->conf->loop_dims[SLICE_DIM] = 2;
	seq->conf->loop_dims[PHS1_DIM] = 3;
	seq->conf->loop_dims[TIME_DIM] = 3;
	seq_ui_interface_loop_dims(0, seq->conf, DIMS, seq->conf->loop_dims);

	int i = 0;
	int trigger_count = 0;
	do {

		int E = seq_block(seq->N, seq->event, seq->state, seq->conf);

		if (0 > E)
			return false;

		if (0 == E)
			continue;

		if (blocks[i] != seq->state->mode)
			return false;

		if ((SEQ_BLOCK_KERNEL_IMAGE == seq->state->mode) && (FLASH_EVENTS + trigger_event_count(seq->conf, seq->state) != E))
			return false;

		int trigger_idx = -1;
		if (SEQ_EVENT_TRIGGER == seq->event[0].type)
			trigger_idx = 0;

		if (-1 < trigger_idx) {
			
			if ((SEQ_BLOCK_PRE != seq->state->mode)
				|| (421.E-3 != seq->event[trigger_idx].end))
					return false;
			trigger_count++;
		}

		i++;

	} while (seq_continue(seq->state, seq->conf));

	if (2 != trigger_count)
		return false;

	if (17 != i)
		return false;

	bart_seq_free(seq);

	return true;
}

UT_REGISTER_TEST(test_block_ecg);


static bool test_block_prep(void)
{
	const enum seq_block blocks[23] = {
		SEQ_BLOCK_KERNEL_NOISE,
		SEQ_BLOCK_KERNEL_DUMMY, SEQ_BLOCK_KERNEL_DUMMY,
		SEQ_BLOCK_KERNEL_DUMMY, SEQ_BLOCK_KERNEL_DUMMY,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_POST,
		SEQ_BLOCK_KERNEL_DUMMY, SEQ_BLOCK_KERNEL_DUMMY,
		SEQ_BLOCK_KERNEL_DUMMY, SEQ_BLOCK_KERNEL_DUMMY,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_POST
	};

	struct bart_seq* seq = bart_seq_alloc("");
	bart_seq_defaults(seq);

	seq->conf->enc.order = SEQ_ORDER_AVG_OUTER;
	seq->conf->magn.prep_scans = 2;
	seq->conf->magn.inv_delay_time = 100.E-3;

	seq->conf->loop_dims[BATCH_DIM] = 2;
	seq->conf->loop_dims[SLICE_DIM] = 2;
	seq->conf->loop_dims[PHS1_DIM] = 3;
	seq->conf->loop_dims[TIME_DIM] = 3;
	seq_ui_interface_loop_dims(0, seq->conf, DIMS, seq->conf->loop_dims);

	int i = 0;

	do {

		int E = seq_block(seq->N, seq->event, seq->state, seq->conf);

		if (0 > E)
			return false;

		if (0 == E)
			continue;

		if (blocks[i] != seq->state->mode)
			return false;

		if ((SEQ_BLOCK_KERNEL_IMAGE == seq->state->mode) && (FLASH_EVENTS + trigger_event_count(seq->conf, seq->state) != E))
			return false;

		// correct delay_meas_time
		if ((SEQ_BLOCK_PRE == seq->state->mode) && (1 == E) && (seq->conf->magn.init_delay != seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad)))
			return false;

		// correct inv_delay in post block
		if ((SEQ_BLOCK_POST == seq->state->mode) && (1.E-4 * UT_TOL < fabs(seq->conf->magn.inv_delay_time - seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad))))
			return false;

		i++;

	} while (seq_continue(seq->state, seq->conf));

	bart_seq_free(seq);

	if (23 != i)
		return false;

	return true;
}

UT_REGISTER_TEST(test_block_prep);


static bool test_block_cest(void)
{
	const enum seq_block blocks[22] = {
		SEQ_BLOCK_KERNEL_NOISE,
		SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE,
		SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE,
		SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE,
		SEQ_BLOCK_KERNEL_IMAGE,

	};
	struct bart_seq* seq = bart_seq_alloc("");
	bart_seq_defaults(seq);

	seq->conf->magn.prep_scans = 0;
	seq->conf->cest.sat_pulses = 5;
	seq->conf->cest.gauss_pulse_duration = 50;
	seq->conf->cest.gauss_pulse_fa = 360;
	seq->conf->cest.offset_type = SEQ_CEST_OFFSET_EQUIDISTANT;
	seq->conf->cest.offset_first = -1;
	seq->conf->cest.offset_last = 1;
	seq->conf->cest.offset_increment = 1;
	seq->conf->cest.sat_type = SEQ_CEST_GAUSS;
	seq->conf->cest.sat_pulse_pause = 1;

	seq_ui_interface_loop_dims(0, seq->conf, DIMS, seq->conf->loop_dims);

	int i = 0;

	do
	{
		int E = seq_block(seq->N, seq->event, seq->state, seq->conf);

		if (0 > E)
			return false;

		if (0 == E)
			continue;

		if (blocks[i] != seq->state->mode)
			return false;

		if ((SEQ_BLOCK_KERNEL_IMAGE == seq->state->mode) && (FLASH_EVENTS + trigger_event_count(seq->conf, seq->state) != E))
			return false;

		// check pulse duration
		if ((SEQ_BLOCK_PRE == seq->state->mode) && (1 == E)  && (1 == i) && (1.e6 * seq->conf->cest.gauss_pulse_duration != seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad)))
			return false;

		i++;

	} while (seq_continue(seq->state, seq->conf));

	bart_seq_free(seq);

	if (22 != i)
		return false;

	return true;
}

UT_REGISTER_TEST(test_block_cest);


static bool test_block_cest_OC_non_equidistant(void)
{
	const enum seq_block blocks[8] = {
		SEQ_BLOCK_KERNEL_NOISE,
		SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE,
		SEQ_BLOCK_KERNEL_IMAGE,
	};

	long expected_offsets = 58;

	struct bart_seq* seq = bart_seq_alloc("");
	bart_seq_defaults(seq);

	seq->conf->magn.prep_scans = 0;
	seq->conf->cest.sat_pulses = 5;
	seq->conf->cest.offset_type = SEQ_CEST_OFFSET_PHANTOM;
	seq->conf->cest.sat_type = SEQ_CEST_OC;
	seq->conf->cest.sat_pulse_pause = 1;
	seq_ui_interface_loop_dims(0, seq->conf, DIMS, seq->conf->loop_dims);

	int i = 0;
	int offsets = 0;

	do
	{
		int E = seq_block(seq->N, seq->event, seq->state, seq->conf);

		if (0 > E)
			return false;

		if (0 == E)
			continue;

		if (blocks[i] != seq->state->mode)
			return false;

		if ((SEQ_BLOCK_KERNEL_IMAGE == seq->state->mode)
		    && (FLASH_EVENTS + trigger_event_count(seq->conf, seq->state) != E))
			return false;

		// check pulse duration
		if ((SEQ_BLOCK_PRE == seq->state->mode) && (1 == E)  && (1 == i) && (100.e3 != seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad)))
			return false;

		// we reset counter and repeat for each offset, but skip the noise scan (only done once)
		if (7 == i) {

			i = 1;
			offsets++;
		} else {

			i++;
		}


	} while (seq_continue(seq->state, seq->conf));

	if (expected_offsets != offsets)
		return false;

	bart_seq_free(seq);

	return true;
}

UT_REGISTER_TEST(test_block_cest_OC_non_equidistant);
