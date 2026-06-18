/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <string.h>

#include "misc/misc.h"

#include "seq/config.h"
#include "seq/ui_enums.h"
#include "seq/custom_selections.h"

#include "custom_ui.h"


#ifndef ARRAY_SIZE
#define ARRAY_SIZE(x) (sizeof(x)/sizeof(x[0]))
#endif


static const struct seq_ui_selection custom_selection_defaults[] = {

	{ "seq_wip5", SEQ_UI_IDX_LONG_PE_MODE, "PE Mode", ARRAY_SIZE(pemode_opts), &pemode_opts[0], SEQ_PEMODE_TURN, "" },
	{ "seq_wip2", SEQ_UI_IDX_LONG_CONTRAST, "Contrast", ARRAY_SIZE(contrast_opts), &contrast_opts[0], SEQ_CONTRAST_RF_RANDOM, "" },
};

static const struct seq_ui_long custom_bool_defaults[] = {

	{ "seq_wip8", SEQ_UI_IDX_LONG_RECO, "Online Recon", { 0, 0, 0, 1 }, "", "" },
	{ "seq_wip12", SEQ_UI_IDX_LONG_SMS, "Simultaneous Multi-Slice", { 0, 0, 0, 1 }, "", "" },
};

static const struct seq_ui_long custom_long_defaults[] = {

	{ "seq_wip9", SEQ_UI_IDX_LONG_CMD, "BART cmd", { -1000, 2, 1, 0 }, "BART UI interface.\n1: save info to file\n2: read cmdline from file", "" },
};

static const struct seq_ui_long custom_longarr_defaults[] = {

	// tiny must be the first one
	{ "", SEQ_UI_IDX_LONG_TINY, "Turns / Tiny Golden", { 1, 50, 1, 1 }, "Tiny Golden angle approximation / Number of turns (=repetitions) of sampling pattern.", ""},
	{ "", SEQ_UI_IDX_LONG_PREP_SCANS, "Prep Scans", { 0, 2560, 1, 0 }, "Number of Prep Scans (per slice/partition)", ""},
	{ "", SEQ_UI_IDX_LONG_RF_DURATION_US, "RF pulse duration", { 20, 2560, 20, 400 }, "RF pulse duration.", "us"},
	{ "", SEQ_UI_IDX_LONG_INIT_DELAY, "Delay Measurements", { 0, 300, 1, 0 }, "Delay measurements.", "s"},
	{ "", SEQ_UI_IDX_LONG_INVERSIONS, "Inversions", { 0, 1000, 1, 1 }, "Number of IR experiments.", ""},
	{ "", SEQ_UI_IDX_LONG_INV_DELAY, "Inversion Delay", { 0, 2000, 1, 0 }, "Delay between inversions.", "s"},
	{ "", SEQ_UI_IDX_LONG_MB_FACTOR, "Multiband factor (SMS)", { 1, 5, 1, 1 }, "SMS Multiband factor", ""},
	{ "", SEQ_UI_IDX_LONG_RAGA_ALIGNED_FLAGS, "RAGA aligned flags", { 0, 65535, 1, 0 }, "Bitmask from dimension to align in RAGA sampling", ""},
	{ "", SEQ_UI_IDX_LONG_CEST_SATURATION, "CESTSaturation", { 0, 2, 1, 0 }, "0: off\n1: gauss\n2: OC", ""},
	{ "", SEQ_UI_IDX_LONG_CEST_OFFSET_TYPE, "OffsetType", { 0, 2, 1, 0 }, " 0: equidistant\n 1: custom phantom\n 2: custom invivo", ""},
	{ "", SEQ_UI_IDX_LONG_CEST_OFFSET_PAUSE_MS, "PauseOffsets", { 0, 60000, 100, 2000 }, "Pause between CEST offsets.", "ms" },
	{ "", SEQ_UI_IDX_LONG_CEST_SAT_PULSE_PAUSE_MS, "PauseSaturationPulses", { 1, 1000, 1, 5 }, "Pause between CEST saturation pulses.", "ms" },
	{ "", SEQ_UI_IDX_LONG_CEST_GAUSS_DURATION_MS, "SaturationPulseDurationGauss", { 5, 100, 1, 25 }, "Duration of one Gauss saturation pulse.", "ms" },
	{ "", SEQ_UI_IDX_LONG_CEST_GAUSS_FA, "SaturationPulseFlipAngleGauss", { 90, 10000, 1, 360. }, "Flip Angle of one Gauss saturation pulse.", "deg" },
	{ "", SEQ_UI_IDX_LONG_CEST_SAT_PULSES, "SaturationPulses", { 0, 1000, 1, 40 }, "Number of CEST saturation pulses", ""},
	{ "", SEQ_UI_IDX_LONG_ASL_MODE, "ASL", { 0, 1, 1, 0 }, "0: OFF\n1: PCASL", ""} ,
	{ "", SEQ_UI_IDX_LONG_ASL_LD_MS, "LD", { 0, 10000, 1, 0 }, "PCASL labeling duration.", "ms" },
	{ "", SEQ_UI_IDX_LONG_ASL_PLD_MS, "PLD", { 0, 10000, 1, 0 }, "ASL post-labeling delay.", "ms" },
};


static const struct seq_ui_double custom_double_defaults[] = {

};

static const struct seq_ui_double custom_doublearr_defaults[] = {

	{ "", SEQ_UI_IDX_DOUBLE_BWTP, "BWTP", { 0., 200., 0.1, 1.6 }, "RF bandwidth-time-product.", "" },
	{ "", SEQ_UI_IDX_DOUBLE_ASYM_ECHO, "Asymmetric Echo", { 0.1, 0.5, 0.01, 0.5 }, "asymmetric echo (0.5 means full echo)", "" },
	{ "", SEQ_UI_IDX_DOUBLE_CEST_OC_B1_SCALING, "SaturationScalingOC", { 0.8, 2., 0.01, 1. }, "B1rms for OC saturation pulse.", "uT" },
	{ "", SEQ_UI_IDX_DOUBLE_CEST_OFFSET_FIRST_PPM, "FirstOffset", { -10., 0., 0.01, -5. }, "First CEST offset.", "ppm" },
	{ "", SEQ_UI_IDX_DOUBLE_CEST_OFFSET_LAST_PPM, "LastOffset", { 0., 10., 0.01, 5. }, "Last CEST offset.", "ppm" },
	{ "", SEQ_UI_IDX_DOUBLE_CEST_OFFSET_INCREMENT_PPM, "OffsetIncrement", { 0.01, 1., 0.01, 1. }, "CEST offset increment.", "ppm" },
};



struct custom_ui* seq_custom_ui_init(void)
{
	struct custom_ui* ui = xmalloc(sizeof (struct custom_ui));

	ui->sizes[SELECTION] = ARRAY_SIZE(custom_selection_defaults);
	ui->sizes[BOOL] = ARRAY_SIZE(custom_bool_defaults);
	ui->sizes[LONG] = ARRAY_SIZE(custom_long_defaults);
	ui->sizes[longarr] = ARRAY_SIZE(custom_longarr_defaults);
	ui->sizes[DOUBLE] = ARRAY_SIZE(custom_double_defaults);
	ui->sizes[doublearr] = ARRAY_SIZE(custom_doublearr_defaults);

	ui->selections = xmalloc((size_t)ui->sizes[SELECTION] * (sizeof (struct seq_ui_selection)));
	ui->checkboxes = xmalloc((size_t)ui->sizes[BOOL] * (sizeof (struct seq_ui_long)));
	ui->longs = xmalloc((size_t)ui->sizes[LONG] * (sizeof (struct seq_ui_long)));
	ui->longarr = xmalloc((size_t)ui->sizes[longarr] * (sizeof (struct seq_ui_long)));
	ui->doubles = xmalloc((size_t)ui->sizes[DOUBLE] * (sizeof (struct seq_ui_double)));
	ui->doublearr = xmalloc((size_t)ui->sizes[doublearr] * (sizeof (struct seq_ui_double)));

	memcpy(ui->selections, &custom_selection_defaults, (size_t)ui->sizes[SELECTION] * (sizeof (struct seq_ui_selection)));
	memcpy(ui->checkboxes, &custom_bool_defaults, (size_t)ui->sizes[BOOL] * (sizeof (struct seq_ui_long)));
	memcpy(ui->longs, &custom_long_defaults, (size_t)ui->sizes[LONG] * (sizeof (struct seq_ui_long)));
	memcpy(ui->longarr, &custom_longarr_defaults, (size_t)ui->sizes[longarr] * (sizeof (struct seq_ui_long)));
	memcpy(ui->doubles, &custom_double_defaults, (size_t)ui->sizes[DOUBLE] * (sizeof (struct seq_ui_double)));
	memcpy(ui->doublearr, &custom_doublearr_defaults, (size_t)ui->sizes[doublearr] * (sizeof (struct seq_ui_double)));

	return ui;
}

void seq_custom_ui_free(struct custom_ui* ui)
{
	xfree(ui->selections);
	xfree(ui->checkboxes);
	xfree(ui->longs);
	xfree(ui->longarr);
	xfree(ui->doubles);
	xfree(ui->doublearr);
	xfree(ui);
}


struct lookup {
	const char* name;
	int e;
};

static struct lookup lookup_table_long[] = {

#define lut_entry(NAME) { .name = #NAME, .e = SEQ_UI_IDX_LONG_##NAME },
	SEQ_CUSTOM_UI_IDX_LONG(lut_entry)
#undef lut_entry
};

static struct lookup lookup_table_double[] = {

#define lut_entry(NAME) { .name = #NAME, .e = SEQ_UI_IDX_DOUBLE_##NAME },
	SEQ_CUSTOM_UI_IDX_DOUBLE(lut_entry)
#undef lut_entry
};

static int lookup(const char* name, int n, struct lookup table[n])
{
	for (int i = 0; i < n; i++)
		if (0 == strcmp(name, table[i].name))
			return table[i].e;
	return -1;
}


int seq_custom_ui_get_idx(const char* name)
{
	int index = lookup(name, ARRAY_SIZE(lookup_table_double), lookup_table_double);

	if (0 > index)
		return lookup(name, ARRAY_SIZE(lookup_table_long), lookup_table_long);

	return index;
}
