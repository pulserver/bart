/* Copyright 2025-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <math.h>
#include <stdint.h>

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mri.h"

#include "num/multind.h"

#include "seq/config.h"
#include "seq/flash.h"
#include "seq/helpers.h"
#include "seq/seq.h"

#include "seq/adc_rf.h"

#include "utest.h"


static bool rf_spoiling_spoiled(void)
{
	const double ref[20] = {

		50., 150., 300., 140., 30.,
		330.,  320.,  0.,  90., 230.,
		60.,  300.,  230.,  210.,  240.,
		320.,  90.,  270.,  140.,  60.
	};

	struct seq_config seq = seq_config_defaults_flash;
	seq.phys.contrast = SEQ_CONTRAST_RF_SPOILED;

	struct seq_state seq_state = { };

	seq.loop_dims[SLICE_DIM] = 2;
	seq.loop_dims[PHS1_DIM] = 2;
	seq.loop_dims[TIME_DIM] = 10;

	seq_ui_interface_loop_dims(0, &seq, DIMS, seq.loop_dims);

	seq.loop_dims[COEFF_DIM] = 1;
	seq.loop_dims[COEFF2_DIM] = 1;

	int i = 0;

	do {
		if (UT_TOL < fabs(ref[i++] - rf_spoiling(DIMS, seq_state.pos, &seq)))
			return false;

	} while (seq_continue(&seq_state, &seq));

	return true;
}

UT_REGISTER_TEST(rf_spoiling_spoiled);



static bool test_sinc(void)
{
	struct seq_config seq = seq_config_defaults_flash;

	struct rf_shape rf_shape[10];
	int rfs = seq_sample_rf_shapes(10, rf_shape, &seq);

	float shape_mag[SEQ_MAX_RF_SAMPLES];
	float shape_pha[SEQ_MAX_RF_SAMPLES];

	for (int i =0; i < rf_shape[0].samples; i++)
		seq_cfl_to_sample(&rf_shape[0], i, &shape_mag[i], &shape_pha[i]);

	if (rfs != 1)
		return false;

	if (rf_shape[0].samples != 1E6 * seq.phys.rf_duration)
		return false;

	// expected in reference implementation
	const double good_norm = 330.154932 / 2.;

	double s = seq_pulse_scaling(&rf_shape[0]);
	double n = seq_pulse_norm_sum(&rf_shape[0]);

	if (fabs(s - seq.phys.flip_angle) > 1E-6)
		return false;

	if (fabs(n - good_norm) > 1e-6)
		return false;

	// expected in reference implementation
	double good[2] = { 0.000445, 0.046339 };

	if (   ((shape_mag[310] - 1.0) > 1e-5)
	    || (fabs(good[0] - shape_mag[473]) > 1e-5)
	    || (fabs(good[0] - shape_mag[147]) > 1e-5)
	    || (fabs(good[1] - shape_mag[518]) > 1e-5)
	    || (fabs(good[1] - shape_mag[102]) > 1e-5))
			return false;

	for (int i = 0; i < rf_shape[0].samples; i++) {

		if (((i < 294/2) || (i > 946/2)) && (fabs(shape_pha[i] - M_PI) > 1e-4))
			return false;

		if ((i > 294/2) && (i < 946/2) && (fabs(shape_pha[i]) > 1e-5))
			return  false;
	}

	return true;
}

UT_REGISTER_TEST(test_sinc);

static bool test_sms(void)
{
	struct seq_config seq = seq_config_defaults_flash;
	seq.loop_dims[SLICE_DIM] = 3;
	seq.geom.mb_factor = 3;

	struct rf_shape rf_shape[10];

	int rfs = seq_sample_rf_shapes(10, rf_shape, &seq);

	if (rfs != 3)
		return false;

	if ((rf_shape[0].samples != 1E6 * seq.phys.rf_duration)
	    || (rf_shape[1].samples != 1E6 * seq.phys.rf_duration)
	    || (rf_shape[2].samples != 1E6 * seq.phys.rf_duration))
		return false;

	// expected in reference implementation
	const double good_norm = 110.051648 / 2.;

	double s = seq_pulse_scaling(&rf_shape[0]);

	double n0 = seq_pulse_norm_sum(&rf_shape[0]);
	double n1 = seq_pulse_norm_sum(&rf_shape[1]);
	double n2 = seq_pulse_norm_sum(&rf_shape[2]);

	if (fabs(s - seq.phys.flip_angle) > 1e-6)
		return false;

	if ((fabs(n0 - good_norm) > 1e-6)
	    || (fabs(n1 - good_norm) > 1e-6)
	    || (fabs(n2 - good_norm) > 1e-6))
		return false;

	// values exported from reference implementation
	int idx[7] = { 310, 298, 322, 346, 274, 382, 238 };

	float mag[3][7] = {
		{ 1., 0.349166, 0.349166, 0.243029, 0.243031, 0.199427, 0.199427 },
		{ 0., 0.250700, 0.889000, 0.188331, 0.836786, 0.473933, 0.344311 },
		{ 0., 0.889000, 0.250700, 0.836787, 0.188333, 0.344310, 0.473932 },
	};

	float pha[3][7] = {
		{ 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 3.141593, 3.141593 },
		{ 5.176036, 5.235988, 2.094395, 5.235988, 2.094395, 2.094395, 2.094395 },
		{ 4.390638, 4.188790, 1.047197, 4.188790, 1.047198, 4.188790, 4.188790 }
	};

	for (int m = 0; m < seq.geom.mb_factor; m++) {

		float shape_mag[SEQ_MAX_RF_SAMPLES];
		float shape_pha[SEQ_MAX_RF_SAMPLES];

		for (int i =0; i < rf_shape[0].samples; i++)
			seq_cfl_to_sample(&rf_shape[m], i, &shape_mag[i], &shape_pha[i]);

		for (int i = 0; i < 7; i++) {

			if (fabsf(shape_mag[idx[i]] - mag[m][i]) > 1e-5)
				return false;

			if (fabsf(shape_pha[idx[i]] - pha[m][i]) > 1e-5)
				return false;
		}
	}

	return true;
}

UT_REGISTER_TEST(test_sms);



static bool test_oc(void)
{
	struct seq_config seq = seq_config_defaults_flash;
	seq.cest.sat_type = SEQ_CEST_OC;

	// for fa_prep
	seq.cest.oc_pulse_b1_scaling = 1.2;
	seq.cest.sat_pulse_pause = 5 * 1.E-3;

	struct rf_shape rf_shape[10];

	int rfs = seq_sample_rf_shapes(10, rf_shape, &seq);

	float shape_mag[SEQ_MAX_RF_SAMPLES];
	float shape_pha[SEQ_MAX_RF_SAMPLES];

	for (int i = 0; i < rf_shape[1].samples; i++)
		seq_cfl_to_sample(&rf_shape[1], i, &shape_mag[i], &shape_pha[i]);

	if (rfs != 2)
		return false;

	if (rf_shape[1].samples != 1000)
		return false;

	// expected in reference implementation
	const double good_norm = 693.290598;
	const double good_fa = 1482.658888;
	const double good_fa_prep = 1205.773438;
	// expected in reference implementation
	int idx[5] = { 11, 271, 500, 882, 966 };
	double good[5] = { 0.110344, 0.735732, 0.814455, 1., 0.334938 };

	double s = seq_pulse_scaling(&rf_shape[1]);
	double n = seq_pulse_norm_sum(&rf_shape[1]);

	if (fabs(s - good_fa) > 1E-6)
		return false;

	if (fabs(n - good_norm) > 1e-6)
		return false;

	if (fabs(rf_shape[1].fa_prep - good_fa_prep) > 1E-6)
		return false;

	if (   (fabs(good[0] - shape_mag[idx[0]]) > 1e-5)
	    || (fabs(good[1] - shape_mag[idx[1]]) > 1e-5)
	    || (fabs(good[2] - shape_mag[idx[2]]) > 1e-5)
	    || (fabs(good[3] - shape_mag[idx[3]]) > 1e-5)
	    || (fabs(good[4] - shape_mag[idx[4]]) > 1e-5))
			return false;

	float sum_abs_pha = 0.;

	for (int i = 0; i < rf_shape[1].samples; i++)
		sum_abs_pha += fabsf(shape_pha[i]);

	// oc pulse is real
	if (0. > sum_abs_pha)
		return false;

	return true;
}

UT_REGISTER_TEST(test_oc);


static bool test_gauss(void)
{
	struct seq_config seq = seq_config_defaults_flash;
	seq.cest.sat_type = SEQ_CEST_GAUSS;

	// for fa_prep
	seq.cest.gauss_pulse_duration = 25000 * 1E-6;
	seq.cest.gauss_pulse_fa = 344.;

	struct rf_shape rf_shape[10];

	int rfs = seq_sample_rf_shapes(10, rf_shape, &seq);

	float shape_mag[SEQ_MAX_RF_SAMPLES];
	float shape_pha[SEQ_MAX_RF_SAMPLES];

	for (int i = 0; i < rf_shape[1].samples; i++)
		seq_cfl_to_sample(&rf_shape[1], i, &shape_mag[i], &shape_pha[i]);

	if (rfs != 2)
		return false;

	if (rf_shape[1].samples != 250)
		return false;

	// expected in reference implementation
	const double good_norm = 60039.327957;
	const double good_fa = 165906.056424;
	int idx[5] = { 50, 100, 125, 150, 200 };
	double good[5] = { 0.341606, 0.903373, 1.000000, 0.903373, 0.341606 };

	double s = seq_pulse_scaling(&rf_shape[1]);
	double n = seq_pulse_norm_sum(&rf_shape[1]);



	if (fabs(s - good_fa) > 1E-6)
		return false;

	if (fabs(n - good_norm) > 1e-6)
		return false;

	if (fabs(rf_shape[1].fa_prep - seq.cest.gauss_pulse_fa) > 1E-6)
		return false;


	if (   (fabs(good[0] - shape_mag[idx[0]]) > 1e-5)
	    || (fabs(good[1] - shape_mag[idx[1]]) > 1e-5)
	    || (fabs(good[2] - shape_mag[idx[2]]) > 1e-5)
	    || (fabs(good[3] - shape_mag[idx[3]]) > 1e-5)
	    || (fabs(good[4] - shape_mag[idx[4]]) > 1e-5))
			return false;


	float sum_abs_pha = 0.;

	for (int i = 0; i < rf_shape[1].samples; i++)
		sum_abs_pha += fabsf(shape_pha[i]);

	// oc pulse is real
	if (0. > sum_abs_pha)
		return false;

	return true;
}

UT_REGISTER_TEST(test_gauss);


static bool test_hanning(void)
{
	struct seq_config seq = seq_config_defaults_flash;
	seq.asl.label_type = SEQ_ASL_PCASL;
	
	struct rf_shape rf_shape[10];
	int rfs = seq_sample_rf_shapes(10, rf_shape, &seq); 

	float shape_mag[SEQ_MAX_RF_SAMPLES];
	float shape_pha[SEQ_MAX_RF_SAMPLES];

	for (int i =0; i < rf_shape[1].samples; i++)
		seq_cfl_to_sample(&rf_shape[1], i, &shape_mag[i], &shape_pha[i]);

	if (rfs != 2) // first shape: excitation pulse, second shape: hanning pulse
		return false;

	if (rf_shape[1].samples != 1E6 * seq.asl.hanning.rf_duration)
		return false;

	// expected in reference implementation
	const double good_norm = 133.126998;
	int idx[8] = { 50, 100, 150, 200, 250, 300, 350, 400};
	double good[8] = { 0.019957, 0.041074, 0.187652, 0.704462, 1, 0.704462, 0.187653, 0.041074};

	double s = seq_pulse_scaling(&rf_shape[1]);
	double n = seq_pulse_norm_sum(&rf_shape[1]);
	
	if (fabs(s - seq.asl.hanning.flip_angle) > 1E-5)
		return false;

	if (fabs(n - good_norm) > 1e-6)
		return false;

	if (   (fabs(good[0] - shape_mag[idx[0]]) > 1e-5)
	    || (fabs(good[1] - shape_mag[idx[1]]) > 1e-5)
	    || (fabs(good[2] - shape_mag[idx[2]]) > 1e-5)
	    || (fabs(good[3] - shape_mag[idx[3]]) > 1e-5)
	    || (fabs(good[4] - shape_mag[idx[4]]) > 1e-5)
	    || (fabs(good[5] - shape_mag[idx[5]]) > 1e-5)
	    || (fabs(good[6] - shape_mag[idx[6]]) > 1e-5)
	    || (fabs(good[7] - shape_mag[idx[7]]) > 1e-5))
			return false;

	for (int i = 0; i < rf_shape[1].samples; i++) {

		if (((i < 119) || (i > 381)) && (fabs(shape_pha[i] - M_PI) > 1e-4))
			return false;

		if ((i > 118) && (i < 382) && (fabs(shape_pha[i]) > 1e-5))
			return  false;
	}

	return true;
}

UT_REGISTER_TEST(test_hanning);
