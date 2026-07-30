/* Copyright 2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "seq/helpers.h"
#include "seq/ui_enums.h"
#include "seq/custom_ui.h"

#include "seq/seq.h"

#include "utest.h"

#define CHECK_ID(size, arr, id) { for (int i = 0; i < size; i++) { if (arr[i].id == id) { return 1; } } return 0; }

static int is_wip_id_match_SELECTION(const struct custom_ui* ui, int id) { CHECK_ID(ui->sizes[SEQ_UI_SELECTION], ui->selections, id); }
static int is_wip_id_match_BOOL(const struct custom_ui* ui, int id) { CHECK_ID(ui->sizes[SEQ_UI_BOOL], ui->checkboxes, id); }
static int is_wip_id_match_LONG(const struct custom_ui* ui, int id) { CHECK_ID(ui->sizes[SEQ_UI_LONG], ui->longs, id); }
static int is_wip_id_match_DOUBLE(const struct custom_ui* ui, int id) { CHECK_ID(ui->sizes[SEQ_UI_DOUBLE], ui->doubles, id); }

#define IS_VALID_IDX(pcui, name, index) is_wip_id_match_##name(pcui, index)

static bool test_custom_ui_size(void)
{
	struct custom_ui* custom_ui = seq_custom_ui_init();

	int nl = custom_ui->sizes[SEQ_UI_SELECTION] + custom_ui->sizes[SEQ_UI_BOOL] + custom_ui->sizes[SEQ_UI_LONG] + custom_ui->sizes[SEQ_UI_longarr];
	int nd = custom_ui->sizes[SEQ_UI_DOUBLE] + custom_ui->sizes[SEQ_UI_doublearr];

	if (SEQ_MAX_PARAMS_LONG < nl)
		return false;

	if (SEQ_MAX_PARAMS_DOUBLE < nd)
		return false;

	seq_custom_ui_free(custom_ui);

	return true;
}

UT_REGISTER_TEST(test_custom_ui_size);


static bool test_custom_ui_indices(void)
{
	struct custom_ui* custom_ui = seq_custom_ui_init();

	int nl = custom_ui->sizes[SEQ_UI_SELECTION] + custom_ui->sizes[SEQ_UI_BOOL] + custom_ui->sizes[SEQ_UI_LONG] + custom_ui->sizes[SEQ_UI_longarr];
	int nd = custom_ui->sizes[SEQ_UI_DOUBLE] + custom_ui->sizes[SEQ_UI_doublearr];

	for (int i = 0; i < nl; i++) {

		if (i < custom_ui->sizes[SEQ_UI_SELECTION]) {

			if (!IS_VALID_IDX(custom_ui, SELECTION, i))
				return false;

		} else if (   (i >= custom_ui->sizes[SEQ_UI_SELECTION])
			   && (i < (custom_ui->sizes[SEQ_UI_SELECTION] + custom_ui->sizes[SEQ_UI_BOOL]))) {

			if (!IS_VALID_IDX(custom_ui, BOOL, i))
				return false;

		} else if (   (i >= (custom_ui->sizes[SEQ_UI_SELECTION] + custom_ui->sizes[SEQ_UI_BOOL]))
			   && (i <  (custom_ui->sizes[SEQ_UI_SELECTION] + custom_ui->sizes[SEQ_UI_BOOL] + custom_ui->sizes[SEQ_UI_LONG]))) {

			if (!IS_VALID_IDX(custom_ui, LONG, i))
				return false;

		} else if ((i >= (custom_ui->sizes[SEQ_UI_SELECTION] + custom_ui->sizes[SEQ_UI_BOOL] + custom_ui->sizes[SEQ_UI_LONG])) && (i < nl)) {

			if (IS_VALID_IDX(custom_ui, SELECTION, i) || IS_VALID_IDX(custom_ui, BOOL, i) || IS_VALID_IDX(custom_ui, LONG, i))
				return false;

		} else {

			return false;
		}
	}

	for (int i = 0; i < nd; i++) {

		if (i < custom_ui->sizes[SEQ_UI_DOUBLE]) {

			if (!IS_VALID_IDX(custom_ui, DOUBLE, i))
				return false;

		} else if ((i >= custom_ui->sizes[SEQ_UI_DOUBLE]) && (i < nd)) {

			if (IS_VALID_IDX(custom_ui, DOUBLE, i))
				return false;

		} else {

			return false;
		}
	}

	seq_custom_ui_free(custom_ui);

	return true;
}

UT_REGISTER_TEST(test_custom_ui_indices);


static bool test_custom_ui_limits(void)
{
	struct custom_ui* custom_ui = seq_custom_ui_init();

	int nl = custom_ui->sizes[SEQ_UI_SELECTION] + custom_ui->sizes[SEQ_UI_BOOL] + custom_ui->sizes[SEQ_UI_LONG] + custom_ui->sizes[SEQ_UI_longarr];
	int nd = custom_ui->sizes[SEQ_UI_DOUBLE] + custom_ui->sizes[SEQ_UI_doublearr];


	for (int i = 0; i < nl; i++) {

		if (IS_VALID_IDX(custom_ui, SELECTION, i)) {

			int idx = i - custom_ui->selections[0].id;
			const struct selection_opt* opts = custom_ui->selections[idx].opts;

			int found = 0;
			for (int j = 0; j < custom_ui->selections[idx].opts_size; j++)
				if (opts[j].id == custom_ui->selections[idx].val_default)
					found = 1;

			if (!found)
				return false;

		} else if (IS_VALID_IDX(custom_ui, BOOL, i)) {

			int idx = i - custom_ui->checkboxes[0].id;
			if ((1 != custom_ui->checkboxes[idx].limit[3]) && (2 != custom_ui->checkboxes[idx].limit[3]))
				return false;

		} else if (IS_VALID_IDX(custom_ui, LONG, i)) {

			int idx = i - custom_ui->longs[0].id;
			if (   (custom_ui->longs[idx].limit[3] < custom_ui->longs[idx].limit[0])
			    || (custom_ui->longs[idx].limit[3] > custom_ui->longs[idx].limit[1]))
				return false;

		} else {

			int idx = i - custom_ui->longarr[0].id;
			if (   (custom_ui->longarr[idx].limit[3] < custom_ui->longarr[idx].limit[0])
			    || (custom_ui->longarr[idx].limit[3] > custom_ui->longarr[idx].limit[1]))
				return false;
		}
	}

	for (int i = 0; i < nd; i++) {

		if (IS_VALID_IDX(custom_ui, DOUBLE, i)) {

			int idx = i - custom_ui->doubles[0].id;
			if (   (custom_ui->doubles[idx].limit[3] < custom_ui->doubles[idx].limit[0])
			    || (custom_ui->doubles[idx].limit[3] > custom_ui->doubles[idx].limit[1]))
				return false;

		} else {

			int idx = i - custom_ui->doublearr[0].id;
			if (   (custom_ui->doublearr[idx].limit[3] < custom_ui->doublearr[idx].limit[0])
			    || (custom_ui->doublearr[idx].limit[3] > custom_ui->doublearr[idx].limit[1]))
				return false;
		}
	}

	seq_custom_ui_free(custom_ui);

	return true;
}

UT_REGISTER_TEST(test_custom_ui_limits);
