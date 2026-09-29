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
#include "seq/seq_asl.h"

#include "utest.h"

#define FLASH_EVENTS 14

#define CHECK_ID(size, arr, id) { for (int i = 0; i < size; i++) { if (arr[i].id == id) { return 1; } } return 0; }

static int is_wip_id_match_SELECTION(const struct custom_ui* ui, int id) { CHECK_ID(ui->sizes[SEQ_UI_SELECTION], ui->selections, id); }
static int is_wip_id_match_BOOL(const struct custom_ui* ui, int id) { CHECK_ID(ui->sizes[SEQ_UI_BOOL], ui->checkboxes, id); }
static int is_wip_id_match_LONG(const struct custom_ui* ui, int id) { CHECK_ID(ui->sizes[SEQ_UI_LONG], ui->longs, id); }
static int is_wip_id_match_DOUBLE(const struct custom_ui* ui, int id) { CHECK_ID(ui->sizes[SEQ_UI_DOUBLE], ui->doubles, id); }

#define IS_VALID_IDX(pcui, name, index) is_wip_id_match_##name(pcui, index)

// those are actively used in sequence
static bool test_commands_sequence(void)
{
	struct bart_seq* seq = bart_seq_alloc("");
	bart_seq_defaults(seq);

	static char seq_cmd[128];
	snprintf(seq_cmd, 128, "bart seq --pe_mode 1 --contrast 2 --tiny 13 --asl_label_slice 3");

	if (!seq_config_from_string(seq->conf, 128, seq_cmd))
		return false;

	if ((1 != seq->conf->enc.pe_mode) || (2 != seq->conf->phys.contrast) || (13 != seq->conf->enc.tiny))
		return false;

	if (3 != seq->conf->asl.label_slice_index)
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

static bool test_version_check(void)
{
	const unsigned int min_bart_version[5] = { 1, 0, 0, 390, 0 };

	struct bart_seq* seq = bart_seq_alloc("v0.1.00-1-g97f2f73");

	if (0 > bart_seq_version_check(seq->driver_version, min_bart_version))
		return false;

	bart_seq_free(seq);

	return true;
}

UT_REGISTER_TEST(test_version_check);


static bool test_init_prepare(void)
{
	struct bart_seq* bart_seq = bart_seq_alloc("");
	struct custom_ui* custom_ui = seq_custom_ui_init();

	bart_seq_defaults(bart_seq);

	struct seq_standard_conf init_std;
	seq_ui_interface_standard_conf(2, bart_seq->conf, &init_std);

	seq_ui_interface_standard_conf(0, bart_seq->conf, &init_std);

	// initialize custom UI
	bart_dim_t custom_long[SEQ_MAX_PARAMS_LONG] = { };
	double custom_double[SEQ_MAX_PARAMS_DOUBLE] = { };

	for (int i = 0; i < (custom_ui->sizes[SEQ_UI_SELECTION] + custom_ui->sizes[SEQ_UI_BOOL] + custom_ui->sizes[SEQ_UI_LONG] + custom_ui->sizes[SEQ_UI_longarr]); i++) {

		if (IS_VALID_IDX(custom_ui, SELECTION, i))
			custom_long[i] = custom_ui->selections[i - custom_ui->selections[0].id].val_default;
		else if (IS_VALID_IDX(custom_ui, BOOL, i))
			custom_long[i] = custom_ui->checkboxes[i - custom_ui->checkboxes[0].id].limit[3];
		else if (IS_VALID_IDX(custom_ui, LONG, i))
			custom_long[i] = custom_ui->longs[i - custom_ui->longs[0].id].limit[3];
		else
			custom_long[i] = custom_ui->longarr[i - custom_ui->longarr[0].id].limit[3];
	}

	for (int i = 0; i < (custom_ui->sizes[SEQ_UI_DOUBLE] + custom_ui->sizes[SEQ_UI_doublearr]); i++) {

		if (IS_VALID_IDX(custom_ui, DOUBLE, i))
			custom_double[i] = custom_ui->doubles[i -custom_ui->doubles[0].id].limit[3];
		else
			custom_double[i] = custom_ui->doublearr[i -custom_ui->doublearr[0].id].limit[3];
	}

	// get custom params from UI
	int nl = custom_ui->sizes[SEQ_UI_SELECTION] + custom_ui->sizes[SEQ_UI_BOOL] + custom_ui->sizes[SEQ_UI_LONG] + custom_ui->sizes[SEQ_UI_longarr];
	int nd = custom_ui->sizes[SEQ_UI_DOUBLE] + custom_ui->sizes[SEQ_UI_doublearr];

	seq_ui_interface_custom_params(0, bart_seq->conf, nl, custom_long, nd, custom_double);

	bart_dim_t init_dims[DIMS];
	seq_ui_interface_loop_dims(2, bart_seq->conf, DIMS, init_dims);
	seq_ui_interface_loop_dims(0, bart_seq->conf, DIMS, init_dims);

	// char config_info_tmp[7852];
	// seq_print_info_config(7852, config_info_tmp, bart_seq->conf);
	// printf("%s\n", config_info_tmp);
	// printf("test_prepare: %d\n", bart_seq_prepare(bart_seq));

	if (0 > bart_seq_prepare(bart_seq))
		return false;

	seq_custom_ui_free(custom_ui);
	bart_seq_free(bart_seq);

	return true;
}

UT_REGISTER_TEST(test_init_prepare);


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


	float gui_shift[4 * slices][3];

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

	seq_set_fov_pos(slices, 3, &gui_shift[0][0], seq->conf);

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

	float gui_shift[4 * slices][3];

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

	seq_set_fov_pos(slices, 3, &gui_shift[0][0], seq->conf);

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
	seq->conf->magn.inv_delay_time = 1.E-3;

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


static bool test_block_prep_3d(void)
{
	const enum seq_block blocks[21] = {
		SEQ_BLOCK_KERNEL_NOISE,
		SEQ_BLOCK_KERNEL_DUMMY, SEQ_BLOCK_KERNEL_DUMMY, SEQ_BLOCK_KERNEL_DUMMY,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_POST,
		SEQ_BLOCK_KERNEL_DUMMY, SEQ_BLOCK_KERNEL_DUMMY, SEQ_BLOCK_KERNEL_DUMMY,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_POST
	};

	struct bart_seq* seq = bart_seq_alloc("");
	bart_seq_defaults(seq);

	seq->conf->enc.order = SEQ_ORDER_AVG_OUTER;
	seq->conf->magn.prep_scans = 3;
	seq->conf->magn.inv_delay_time = 10.;

	seq->conf->enc.is3D = 1;
	seq->conf->loop_dims[BATCH_DIM] = 2;
	seq->conf->loop_dims[PHS2_DIM] = 2;
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

	if (21 != i)
		return false;

	return true;
}

UT_REGISTER_TEST(test_block_prep_3d);


static bool test_block_prep_sms(void)
{
	const enum seq_block blocks[51] = {
		SEQ_BLOCK_KERNEL_NOISE,
		SEQ_BLOCK_KERNEL_DUMMY, SEQ_BLOCK_KERNEL_DUMMY,
		SEQ_BLOCK_KERNEL_DUMMY, SEQ_BLOCK_KERNEL_DUMMY,
		SEQ_BLOCK_KERNEL_DUMMY, SEQ_BLOCK_KERNEL_DUMMY,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_POST,
		SEQ_BLOCK_KERNEL_DUMMY, SEQ_BLOCK_KERNEL_DUMMY,
		SEQ_BLOCK_KERNEL_DUMMY, SEQ_BLOCK_KERNEL_DUMMY,
		SEQ_BLOCK_KERNEL_DUMMY, SEQ_BLOCK_KERNEL_DUMMY,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
		SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
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
	seq->conf->loop_dims[SLICE_DIM] = 6;
	seq->conf->loop_dims[PHS1_DIM] = 3;
	seq->conf->loop_dims[TIME_DIM] = 3;
	seq->conf->geom.mb_factor = 2;
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

	if (51 != i)
		return false;

	return true;
}

UT_REGISTER_TEST(test_block_prep_sms);


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

	bart_stride_t expected_offsets = 58;

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


static bool test_block_asl(void)
{
	const int asl_events = 5;
	const int num_blocks_pre = 9;
	const int coeff2_dim_offset = 3;

	int seq_block_count = 0;
	const int expected_seq_block_count = 57;

	const enum seq_block blocks[57] = {
	    SEQ_BLOCK_KERNEL_NOISE,
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,							// M0 Image
	    SEQ_BLOCK_POST,														// Inversion delay
	    SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE,	// Label condition
	    SEQ_BLOCK_PRE, 														// PLD
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,							// Label image
	    SEQ_BLOCK_POST,														// Inversion delay
	    SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE,	// Control condition
	    SEQ_BLOCK_PRE, 														// PLD
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,							// Control image
	    SEQ_BLOCK_POST,														// Inversion delay
	    SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE,	// Label condition
	    SEQ_BLOCK_PRE, 														// PLD
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,							// Label image
	    SEQ_BLOCK_POST,														// Inversion delay
	    SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE,	// Control condition
	    SEQ_BLOCK_PRE, 														// PLD
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,							// Control image
	    SEQ_BLOCK_POST,														// Inversion delay
	};

	struct bart_seq* seq = bart_seq_alloc("");
	bart_seq_defaults(seq);

	seq->conf->enc.order = SEQ_ORDER_SEQ_ASL;
	seq->conf->magn.inv_delay_time = 4;
	seq->conf->asl.label_type = SEQ_ASL_PCASL;
	seq->conf->asl.ld = 0.01;
	seq->conf->asl.pld = 1.3;
	seq->conf->asl.label_slice_index = 1;
	seq->conf->asl.pulse_spacing = 1.15E-3;
	seq->conf->enc.pe_mode = SEQ_PEMODE_RAGA;

	seq->conf->loop_dims[PHS1_DIM] = 3;  // radial views
	seq->conf->loop_dims[PHS2_DIM] = 1;  // partitions
	seq->conf->loop_dims[SLICE_DIM] = 2; // slices
	seq->conf->loop_dims[TIME_DIM] = 3;  // RAGA: total number of spokes (3 / 3 = 1 frame)
	seq->conf->loop_dims[TE_DIM] = 1;    // echos
	seq->conf->loop_dims[AVG_DIM] = 2;   // averages
	seq_ui_interface_loop_dims(0, seq->conf, DIMS, seq->conf->loop_dims);

	int i = 0;

	do {

		int E = seq_block(seq->N, seq->event, seq->state, seq->conf);

		if (0 > E)
			return false;

		if (0 == E)
			continue;

		seq_block_count++;

		if (blocks[i] != seq->state->mode) {

			debug_printf(DP_INFO, "Block mismatch at i=%d: expected %d, got %d\n", i, blocks[i], seq->state->mode);
			return false;
		}

		int expected_flash_ev = FLASH_EVENTS + trigger_event_count(seq->conf, seq->state);
		if ((SEQ_BLOCK_KERNEL_IMAGE == seq->state->mode) && (expected_flash_ev != E)) {

			debug_printf(DP_INFO, "FLASH event count mismatch at i=%d: expected %d, got %d\n", i, expected_flash_ev, E);
			return false;
		}

		if (seq->state->pos[COEFF2_DIM] - coeff2_dim_offset < num_blocks_pre - 1) {

			if ((SEQ_BLOCK_PRE == seq->state->mode) && (asl_events != E)
			    && (seq->conf->asl.pulse_spacing != seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad))) {

				debug_printf(DP_INFO, "ASL event or pulse spacing mismatch at i=%d: expected events=%d, got=%d; expected pulse_spacing=%f, got=%f\n",
					     i, asl_events, E, seq->conf->asl.pulse_spacing, seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad));
				return false;
			}
		}
		else {

			// assert correct pld in post block
			if ((SEQ_BLOCK_PRE == seq->state->mode) && (1 != E)
			    && (seq->conf->asl.pld != seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad))) {

				debug_printf(DP_INFO, "PLD mismatch at i=%d: expected PLD=%f, got=%f\n",
					     i, seq->conf->asl.pld, seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad));
				return false;
			}
		}

		// assert correct inv_delay in post block
		if ((SEQ_BLOCK_POST == seq->state->mode)
		    && (seq->conf->magn.inv_delay_time != seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad))) {

			debug_printf(DP_INFO, "Inversion delay mismatch at i=%d: expected=%f, got=%lf\n",
				     i, seq->conf->magn.inv_delay_time, seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad));
			return false;
		}

		i++;

	} while (seq_continue(seq->state, seq->conf));

	double expected_duration = 25.286560;
	if (1.E-3 < fabs(expected_duration - seq_total_measure_time(seq->conf))) {

		debug_printf(DP_WARN, "Calculation of sequence duration invalid! Actual duration: %.3f s (expected: %.3f)\n",
			     seq_total_measure_time(seq->conf), expected_duration);
	}

	if (seq_block_count != expected_seq_block_count) {

		debug_printf(DP_WARN, "Sequence block count mismatch! Expected: %d, got %d\n", expected_seq_block_count, seq_block_count);
		return false;
	}

	bart_seq_free(seq);

	return true;
}

UT_REGISTER_TEST(test_block_asl);

static bool test_block_asl_cont(void)
{
	const int asl_events = 5;
	const int num_blocks_pre = 9;
	const int coeff2_dim_offset = 3;

	int seq_block_count = 0;
	const int expected_seq_block_count = 132;

	const enum seq_block blocks[132] = {
	    SEQ_BLOCK_KERNEL_NOISE,
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,	// M0 Image
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
	    SEQ_BLOCK_POST,														// Inversion delay
	    SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE,	// Label condition
	    SEQ_BLOCK_PRE, 														// PLD
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,	// Label image
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
	    SEQ_BLOCK_POST,														// Inversion delay
	    SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE,	// Control condition
	    SEQ_BLOCK_PRE, 														// PLD
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,	// Control image
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
	    SEQ_BLOCK_POST,														// Inversion delay
	    SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE,	// Label condition
	    SEQ_BLOCK_PRE, 														// PLD
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,	// Label image
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
	    SEQ_BLOCK_POST,														// Inversion delay
	    SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE,	// Control condition
	    SEQ_BLOCK_PRE, 														// PLD
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,	// Control image
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
	    SEQ_BLOCK_POST,														// Inversion delay
	    SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE,	// Label condition
	    SEQ_BLOCK_PRE, 														// PLD
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,	// Label image
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
	    SEQ_BLOCK_POST,														// Inversion delay
	    SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE,	// Control condition
	    SEQ_BLOCK_PRE, 														// PLD
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,	// Control image
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
	    SEQ_BLOCK_POST,														// Inversion delay
	};

	struct bart_seq* seq = bart_seq_alloc("");
	bart_seq_defaults(seq);

	seq->conf->enc.order = SEQ_ORDER_SEQ_ASL;
	seq->conf->magn.inv_delay_time = 4;
	seq->conf->asl.label_type = SEQ_ASL_PCASL;
	seq->conf->asl.ld = 0.01;
	seq->conf->asl.pld = 1.3;
	seq->conf->asl.label_slice_index = 1;
	seq->conf->asl.pulse_spacing = 1.15E-3;
	seq->conf->enc.pe_mode = SEQ_PEMODE_RAGA;

	seq->conf->loop_dims[PHS1_DIM] = 5;  // radial views (per frame)
	seq->conf->loop_dims[PHS2_DIM] = 1;  // partitions
	seq->conf->loop_dims[SLICE_DIM] = 2; // slices
	seq->conf->loop_dims[TIME_DIM] = 10; // RAGA: total number of spokes (10 / 5 = 2 frames)
	seq->conf->loop_dims[TE_DIM] = 1;    // echos
	seq->conf->loop_dims[AVG_DIM] = 3;   // averages (3 control/label pairs + 1 M0)
	seq_ui_interface_loop_dims(0, seq->conf, DIMS, seq->conf->loop_dims);

	int i = 0;

	do {

		int E = seq_block(seq->N, seq->event, seq->state, seq->conf);

		if (0 > E)
			return false;

		if (0 == E)
			continue;

		seq_block_count++;

		if (blocks[i] != seq->state->mode) {

			debug_printf(DP_INFO, "Block mismatch at i=%d: expected %d, got %d\n", i, blocks[i], seq->state->mode);
			return false;
		}

		int expected_flash_ev = FLASH_EVENTS + trigger_event_count(seq->conf, seq->state);
		if ((SEQ_BLOCK_KERNEL_IMAGE == seq->state->mode) && (expected_flash_ev != E)) {

			debug_printf(DP_INFO, "FLASH event count mismatch at i=%d: expected %d, got %d\n", i, expected_flash_ev, E);
			return false;
		}

		if (seq->state->pos[COEFF2_DIM] - coeff2_dim_offset < num_blocks_pre - 1) {

			if ((SEQ_BLOCK_PRE == seq->state->mode) && (asl_events != E)
			    && (seq->conf->asl.pulse_spacing != seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad))) {

				debug_printf(DP_INFO, "ASL event or pulse spacing mismatch at i=%d: expected events=%d, got=%d; expected pulse_spacing=%f, got=%f\n",
					     i, asl_events, E, seq->conf->asl.pulse_spacing, seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad));
				return false;
			}
		}
		else {

			// assert correct pld in post block
			if ((SEQ_BLOCK_PRE == seq->state->mode) && (1 != E)
			    && (seq->conf->asl.pld != seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad))) {

				debug_printf(DP_INFO, "PLD mismatch at i=%d: expected PLD=%f, got=%f\n",
					     i, seq->conf->asl.pld, seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad));
				return false;
			}
		}

		// assert correct inv_delay in post block
		if ((SEQ_BLOCK_POST == seq->state->mode)
		    && (seq->conf->magn.inv_delay_time != seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad))) {

			debug_printf(DP_INFO, "Inversion delay mismatch at i=%d: expected=%f, got=%lf\n",
				     i, seq->conf->magn.inv_delay_time, seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad));
			return false;
		}

		i++;

	} while (seq_continue(seq->state, seq->conf));

	double expected_duration = 36.076010;
	if (1.E-3 < fabs(expected_duration - seq_total_measure_time(seq->conf))) {

		debug_printf(DP_WARN, "Calculation of sequence duration invalid! Actual duration: %.3f s (expected: %.3f)\n",
			     seq_total_measure_time(seq->conf), expected_duration);
	}

	if (seq_block_count != expected_seq_block_count) {

		debug_printf(DP_WARN, "Sequence block count mismatch! Expected: %d, got %d\n", expected_seq_block_count, seq_block_count);
		return false;
	}

	bart_seq_free(seq);

	return true;
}

UT_REGISTER_TEST(test_block_asl_cont);

static bool test_block_asl_multi_slice(void)
{
	const int asl_events = 5;
	const int num_blocks_pre = 9;
	const int coeff2_dim_offset = 3;

	int seq_block_count = 0;
	const int expected_seq_block_count = 167;

	const enum seq_block blocks[167] = {
	    SEQ_BLOCK_KERNEL_NOISE,
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,	// M0 Image
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
	    SEQ_BLOCK_POST,														// Inversion delay
	    SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE,	// Label condition
	    SEQ_BLOCK_PRE, 														// PLD
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,	// Label image
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
	    SEQ_BLOCK_POST,														// Inversion delay
	    SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE,	// Control condition
	    SEQ_BLOCK_PRE, 														// PLD
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,	// Control image
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
	    SEQ_BLOCK_POST,														// Inversion delay
	    SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE,	// Label condition
	    SEQ_BLOCK_PRE, 														// PLD
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,	// Label image
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
	    SEQ_BLOCK_POST,														// Inversion delay
	    SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE,	// Control condition
	    SEQ_BLOCK_PRE, 														// PLD
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,	// Control image
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
	    SEQ_BLOCK_POST,														// Inversion delay
	    SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE,	// Label condition
	    SEQ_BLOCK_PRE, 														// PLD
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,	// Label image
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
	    SEQ_BLOCK_POST,														// Inversion delay
	    SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE, SEQ_BLOCK_PRE,	// Control condition
	    SEQ_BLOCK_PRE, 														// PLD
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,	// Control image
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
	    SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE, SEQ_BLOCK_KERNEL_IMAGE,
	    SEQ_BLOCK_POST,														// Inversion delay
	};

	struct bart_seq* seq = bart_seq_alloc("");
	bart_seq_defaults(seq);

	seq->conf->enc.order = SEQ_ORDER_SEQ_ASL;
	seq->conf->magn.inv_delay_time = 4;
	seq->conf->asl.label_type = SEQ_ASL_PCASL;
	seq->conf->asl.ld = 0.01;
	seq->conf->asl.pld = 1.3;
	seq->conf->asl.label_slice_index = 1;
	seq->conf->asl.pulse_spacing = 1.15E-3;
	seq->conf->enc.pe_mode = SEQ_PEMODE_RAGA;

	seq->conf->loop_dims[PHS1_DIM] = 5;  // radial views (per frame)
	seq->conf->loop_dims[PHS2_DIM] = 1;  // partitions
	seq->conf->loop_dims[SLICE_DIM] = 4; // slices (1 label slice + 3 image slices)
	seq->conf->loop_dims[TIME_DIM] = 5;  // RAGA: total number of spokes (5 / 5 = 1 frame)
	seq->conf->loop_dims[TE_DIM] = 1;    // echos
	seq->conf->loop_dims[AVG_DIM] = 3;   // averages (3 control/label pairs + 1 M0)
	seq_ui_interface_loop_dims(0, seq->conf, DIMS, seq->conf->loop_dims);

	int i = 0;

	do {

		int E = seq_block(seq->N, seq->event, seq->state, seq->conf);

		if (0 > E)
			return false;

		if (0 == E)
			continue;

		seq_block_count++;

		if (blocks[i] != seq->state->mode) {

			debug_printf(DP_INFO, "Block mismatch at i=%d: expected %d, got %d\n", i, blocks[i], seq->state->mode);
			return false;
		}

		int expected_flash_ev = FLASH_EVENTS + trigger_event_count(seq->conf, seq->state);
		if ((SEQ_BLOCK_KERNEL_IMAGE == seq->state->mode) && (expected_flash_ev != E)) {

			debug_printf(DP_INFO, "FLASH event count mismatch at i=%d: expected %d, got %d\n", i, expected_flash_ev, E);
			return false;
		}

		if (seq->state->pos[COEFF2_DIM] - coeff2_dim_offset < num_blocks_pre - 1) {

			if ((SEQ_BLOCK_PRE == seq->state->mode) && (asl_events != E)
			    && (seq->conf->asl.pulse_spacing != seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad))) {

				debug_printf(DP_INFO, "ASL event or pulse spacing mismatch at i=%d: expected events=%d, got=%d; expected pulse_spacing=%f, got=%f\n",
					     i, asl_events, E, seq->conf->asl.pulse_spacing, seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad));
				return false;
			}
		}
		else {

			// assert correct pld in post block
			if ((SEQ_BLOCK_PRE == seq->state->mode) && (1 != E)
			    && (seq->conf->asl.pld != seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad))) {

				debug_printf(DP_INFO, "PLD mismatch at i=%d: expected PLD=%f, got=%f\n",
					     i, seq->conf->asl.pld, seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad));
				return false;
			}
		}

		// assert correct inv_delay in post block
		if ((SEQ_BLOCK_POST == seq->state->mode)
		    && (seq->conf->magn.inv_delay_time != seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad))) {

			debug_printf(DP_INFO, "Inversion delay mismatch at i=%d: expected=%f, got=%lf\n",
				     i, seq->conf->magn.inv_delay_time, seq_block_end(E, seq->event, seq->state->mode, seq->conf->phys.tr, seq->conf->sys.raster_grad));
			return false;
		}

		i++;

	} while (seq_continue(seq->state, seq->conf));

	double expected_duration = 36.184860;
	if (1.E-3 < fabs(expected_duration - seq_total_measure_time(seq->conf))) {

		debug_printf(DP_WARN, "Calculation of sequence duration invalid! Actual duration: %.3f s (expected: %.3f)\n",
			     seq_total_measure_time(seq->conf), expected_duration);
	}

	if (seq_block_count != expected_seq_block_count) {

		debug_printf(DP_WARN, "Sequence block count mismatch! Expected: %d, got %d\n", expected_seq_block_count, seq_block_count);
		return false;
	}

	bart_seq_free(seq);

	return true;
}

UT_REGISTER_TEST(test_block_asl_multi_slice);