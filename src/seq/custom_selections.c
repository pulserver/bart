/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "seq/config.h"
#include "seq/custom_ui.h"

#include "custom_selections.h"

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(x) (sizeof(x)/sizeof(x[0]))
#endif

const struct selection_opt seqtype_opts[] = {

	{ SEQ_TYPE_FLASH, "Flash (BOOST)", },
	{ SEQ_TYPE_MINIFLASH, "Miniflash", },
};

const struct selection_opt pemode_opts[] = {

	{ SEQ_PEMODE_TURN, "1. TURN", },
	{ SEQ_PEMODE_RAGA, "2. RAGA", },
	{ SEQ_PEMODE_MEMS_HYB, "3. MEMS", },
	{ SEQ_PEMODE_CARTESIAN, "4. CARTESIAN (center-out)", },
	{ SEQ_PEMODE_CARTESIAN_LINEAR, "5. CARTESIAN (linear)", },
};


const struct selection_opt contrast_opts[] = {

	{ SEQ_CONTRAST_RF_RANDOM, "1. RF Random" },
	{ SEQ_CONTRAST_RF_SPOILED, "2. RF Spoiled" }
};


const char* get_seqtype_str(enum seq_type type)
{
	for (long unsigned int i = 0; i < ARRAY_SIZE(seqtype_opts); i++)
		if ((enum seq_type)seqtype_opts[i].id == type)
			return seqtype_opts[i].label;

	return "unknown seqtype";	
}


const char* get_pemode_str(enum pe_mode mode)
{
	for (long unsigned int i = 0; i < ARRAY_SIZE(pemode_opts); i++)
		if ((enum pe_mode)pemode_opts[i].id == mode)
			return pemode_opts[i].label;

	return "unknown pemode";	
}

const char* get_contrast_str(enum flash_contrast contrast)
{
	for (long unsigned int i = 0; i < ARRAY_SIZE(contrast_opts); i++)
		if ((enum flash_contrast)contrast_opts[i].id == contrast)
			return contrast_opts[i].label;

	return "unknown contrast";
}