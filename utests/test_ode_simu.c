/* Copyright 2022. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author:
 *	Nick Scholand
 */

#include <math.h>

#include "seq/pulse.h"

#include "simu/bloch.h"
#include "simu/simulation.h"
#include "simu/epg.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "misc/mri.h"
#include "misc/misc.h"

#include "utest.h"




// Validate the partial derivatives estimated with the SA with
// the difference quotient method for estimating gradients

static bool test_ode_bloch_simulation_gradients(void)
{
	float e = 1.E-3;
	float tol = 1.E-4;

	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = SEQ_IRBSSFP;
	sim_data.seq.tr = 0.004;
	sim_data.seq.te = 0.002;
	sim_data.seq.rep_num = 500;
	sim_data.seq.spin_num = 1;
        sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = sim_data.seq.te;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1[0] = 1. / WATER_T1;
	sim_data.voxel.r2[0] = 1. / WATER_T2;
	sim_data.voxel.m0[0] = 1.;
	sim_data.voxel.w = 0;
	sim_data.voxel.b1 = 1.;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.sinc = pulse_sinc_defaults;
	sim_data.pulse.sinc.super.flipangle = 45.;
	sim_data.pulse.rf_end = 0.001;

	sim_data.grad = simdata_grad_defaults;
        sim_data.other = simdata_other_defaults;

	int R = sim_data.seq.rep_num;

	float mxy_ref_sig[R][3];
	float sa_r1_ref_sig[R][3];
	float sa_r2_ref_sig[R][3];
	float sa_m0_ref_sig[R][3];
	float sa_b1_ref_sig[R][3];

	bloch_simulation(&sim_data, R, &mxy_ref_sig, &sa_r1_ref_sig, &sa_r2_ref_sig, &sa_m0_ref_sig, &sa_b1_ref_sig);

	/* ------------ R1 Partial Derivative Test -------------- */

	float mxy_tmp_sig[R][3];
	float sa_r1_tmp_sig[R][3];
	float sa_r2_tmp_sig[R][3];
	float sa_m0_tmp_sig[R][3];
	float sa_b1_tmp_sig[R][3];

	struct sim_data data_r1 = sim_data;

	data_r1.voxel.r1[0] += e;

	bloch_simulation(&data_r1, R, &mxy_tmp_sig, &sa_r1_tmp_sig, &sa_r2_tmp_sig, &sa_m0_tmp_sig, &sa_b1_tmp_sig);

	for (int i = 0; i < sim_data.seq.rep_num; i++) {
		for (int j = 0; j < 3; j++) {

			float err = fabsf(e * sa_r1_ref_sig[i][j] - (mxy_tmp_sig[i][j] - mxy_ref_sig[i][j]));

			if (tol < err)
				return false;
		}
	}

	/* ------------ R2 Partial Derivative Test -------------- */

	struct sim_data data_r2 = sim_data;

	data_r2.voxel.r2[0] += e;

	bloch_simulation(&data_r2, R, &mxy_tmp_sig, &sa_r1_tmp_sig, &sa_r2_tmp_sig, &sa_m0_tmp_sig, &sa_b1_tmp_sig);

	for (int i = 0; i < sim_data.seq.rep_num; i++) {
		for (int j = 0; j < 3; j++) {

			float err = fabsf(e * sa_r2_ref_sig[i][j] - (mxy_tmp_sig[i][j] - mxy_ref_sig[i][j]));

			if (tol < err)
				return false;
		}
	}

	/* ------------ M0 Partial Derivative Test -------------- */

	struct sim_data data_m0 = sim_data;

	data_m0.voxel.m0[0] += e;

	bloch_simulation(&data_m0, R, &mxy_tmp_sig, &sa_r1_tmp_sig, &sa_r2_tmp_sig, &sa_m0_tmp_sig, &sa_b1_tmp_sig);

	for (int i = 0; i < sim_data.seq.rep_num; i++) {
		for (int j = 0; j < 3; j++) {

			float err = fabsf(e * sa_m0_ref_sig[i][j] - (mxy_tmp_sig[i][j] - mxy_ref_sig[i][j]));

			if (tol < err)
				return false;
		}
	}

	/* ------------ B1 Partial Derivative Test -------------- */

	struct sim_data data_b1 = sim_data;

	data_b1.voxel.b1 += e;

	bloch_simulation(&data_b1, R, &mxy_tmp_sig, &sa_r1_tmp_sig, &sa_r2_tmp_sig, &sa_m0_tmp_sig, &sa_b1_tmp_sig);

	for (int i = 0; i < sim_data.seq.rep_num; i++) {
		for (int j = 0; j < 3; j++) {

			float err = fabsf(e * sa_b1_ref_sig[i][j] - (mxy_tmp_sig[i][j] - mxy_ref_sig[i][j]));

			if (tol < err)
				return false;
		}
	}

	return true;
}

UT_REGISTER_TEST(test_ode_bloch_simulation_gradients);


static bool test_stm_bloch_simulation_gradients(void)
{
	float e = 1.E-3;
	float tol = 1.E-4;


	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
        sim_data.seq.type = SIM_STM;
	sim_data.seq.seq_type = SEQ_IRBSSFP;
	sim_data.seq.tr = 0.004;
	sim_data.seq.te = 0.002;
	sim_data.seq.rep_num = 500;
	sim_data.seq.spin_num = 1;
        sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = sim_data.seq.te;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1[0] = 1. / WATER_T1;
	sim_data.voxel.r2[0] = 1. / WATER_T2;
	sim_data.voxel.m0[0] = 1.;
	sim_data.voxel.w = 0.;
	sim_data.voxel.b1 = 1.;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.sinc = pulse_sinc_defaults;
	sim_data.pulse.sinc.super.flipangle = 45.;
	sim_data.pulse.rf_end = 0.001;

	sim_data.grad = simdata_grad_defaults;
        sim_data.other = simdata_other_defaults;

	int R = sim_data.seq.rep_num;

	float mxy_ref_sig[R][3];
	float sa_r1_ref_sig[R][3];
	float sa_r2_ref_sig[R][3];
	float sa_m0_ref_sig[R][3];
	float sa_b1_ref_sig[R][3];

	bloch_simulation(&sim_data, R, &mxy_ref_sig, &sa_r1_ref_sig, &sa_r2_ref_sig, &sa_m0_ref_sig, &sa_b1_ref_sig);

	/* ------------ R1 Partial Derivative Test -------------- */

	float mxy_tmp_sig[R][3];
	float sa_r1_tmp_sig[R][3];
	float sa_r2_tmp_sig[R][3];
	float sa_m0_tmp_sig[R][3];
	float sa_b1_tmp_sig[R][3];

	struct sim_data data_r1 = sim_data;

	data_r1.voxel.r1[0] += e;

	bloch_simulation(&data_r1, R, &mxy_tmp_sig, &sa_r1_tmp_sig, &sa_r2_tmp_sig, &sa_m0_tmp_sig, &sa_b1_tmp_sig);

	float err = 0;

	for (int i = 0; i < sim_data.seq.rep_num; i++) {
		for (int j = 0; j < 3; j++) {

			err = fabsf(e * sa_r1_ref_sig[i][j] - (mxy_tmp_sig[i][j] - mxy_ref_sig[i][j]));

			if (tol < err)
				return false;
		}
	}

	/* ------------ R2 Partial Derivative Test -------------- */

	struct sim_data data_r2 = sim_data;

	data_r2.voxel.r2[0] += e;

	bloch_simulation(&data_r2, R, &mxy_tmp_sig, &sa_r1_tmp_sig, &sa_r2_tmp_sig, &sa_m0_tmp_sig, &sa_b1_tmp_sig);

	for (int i = 0; i < R; i++) {

		for (int j = 0; j < 3; j++) {

			err = fabsf(e * sa_r2_ref_sig[i][j] - (mxy_tmp_sig[i][j] - mxy_ref_sig[i][j]));

			if (tol < err)
				return false;
		}
	}

	/* ------------ M0 Partial Derivative Test -------------- */

	struct sim_data data_m0 = sim_data;

	data_m0.voxel.m0[0] += e;

	bloch_simulation(&data_m0, R, &mxy_tmp_sig, &sa_r1_tmp_sig, &sa_r2_tmp_sig, &sa_m0_tmp_sig, &sa_b1_tmp_sig);

	for (int i = 0; i < sim_data.seq.rep_num; i++) {
		for (int j = 0; j < 3; j++) {

			err = fabsf(e * sa_m0_ref_sig[i][j] - (mxy_tmp_sig[i][j] - mxy_ref_sig[i][j]));

			if (tol < err)
				return false;
		}
	}

	/* ------------ B1 Partial Derivative Test -------------- */

	struct sim_data data_b1 = sim_data;

	data_b1.voxel.b1 += e;

	bloch_simulation(&data_b1, R, &mxy_tmp_sig, &sa_r1_tmp_sig, &sa_r2_tmp_sig, &sa_m0_tmp_sig, &sa_b1_tmp_sig);

	for (int i = 0; i < R; i++) {

		for (int j = 0; j < 3; j++) {

			err = fabsf(e * sa_b1_ref_sig[i][j] - (mxy_tmp_sig[i][j] - mxy_ref_sig[i][j]));

			if (tol < err)
				return false;
		}
	}

	return true;
}

UT_REGISTER_TEST(test_stm_bloch_simulation_gradients);


/* ODE Simulation
 * Compare the simulated IR bSSFP signal with the analytical model
 * Assumptions: 1. TR << T_{1,2}
 *              2. T_RF \approx 0
 *
 * References:
 * Schmitt, P. , Griswold, M. A., Jakob, P. M., Kotas, M. , Gulani, V. , Flentje, M. and Haase, A. (2004),
 * Inversion recovery TrueFISP: Quantification of T1, T2, and spin density.
 * Magn. Reson. Med., 51: 661-667. doi:10.1002/mrm.20058
 *
 * Ehses, P. , Seiberlich, N. , Ma, D. , Breuer, F. A., Jakob, P. M., Griswold, M. A. and Gulani, V. (2013),
 * IR TrueFISP with a golden‐ratio‐based radial readout: Fast quantification of T1, T2, and proton density.
 * Magn Reson Med, 69: 71-81. doi:10.1002/mrm.24225
 */
static bool test_ode_irbssfp_simulation(void)
{
	float angle = 45.;
	float repetition = 100;

	float fa = DEG2RAD(angle);

	float t1n = WATER_T1;
	float t2n = WATER_T2;
	float m0n = 1.;

	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = SEQ_IRBSSFP;
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.0015;
	sim_data.seq.rep_num = repetition;
	sim_data.seq.spin_num = 1;
        sim_data.seq.perfect_inversion = true;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = sim_data.seq.te;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1[0] = 1. / t1n;
	sim_data.voxel.r2[0] = 1. / t2n;
	sim_data.voxel.m0[0] = m0n;
	sim_data.voxel.w = 0;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.sinc = pulse_sinc_defaults;
	sim_data.pulse.sinc.super.flipangle = angle;

        // Choose close to hard-pulse approximation -> same assumptions as analytical model
	sim_data.pulse.rf_end = 0.00001;

	sim_data.grad = simdata_grad_defaults;
        sim_data.other = simdata_other_defaults;

	int R = sim_data.seq.rep_num;

	float mxy_sig[R][3];
	float sa_r1_sig[R][3];
	float sa_r2_sig[R][3];
	float sa_m0_sig[R][3];
	float sa_b1_sig[R][3];

	bloch_simulation(&sim_data, R, &mxy_sig, &sa_r1_sig, &sa_r2_sig, &sa_m0_sig, &sa_b1_sig);


	// Analytical Model
	float t1s = 1. / ((cosf(fa / 2.) * cosf(fa / 2.)) / t1n + (sinf(fa / 2.) * sinf(fa / 2.)) / t2n);
	float s0 = m0n * sinf(fa / 2.);
	float stst = m0n * sinf(fa) / ((t1n / t2n + 1.) - cosf(fa) * (t1n / t2n - 1.));
	float inv = 1. + s0 / stst;


        // Model Comparison
	float out_simu = 0.;
	float out_theory = 0.;
	float err = 0.;

	for (int z = 0; z < repetition; z++) {

                //Does NOT include phase information!
                // + data.tr through alpha/2 preparation
		out_theory = fabs(stst * (1. - inv * expf(-((float)(z + 1.) * sim_data.seq.tr) / t1s)));

		out_simu = cabsf(mxy_sig[z][0] + mxy_sig[z][1] * I);

		err = fabsf(out_simu - out_theory);

		if (1.E-3 < err)
			return false;
	}

	return true;
}

UT_REGISTER_TEST(test_ode_irbssfp_simulation);



/* ROT Simulation
 *
 * Compare the simulated IR bSSFP signal with the analytical model
 * Assumptions: 1. TR << T_{1,2}
 *              2. T_RF \approx 0
 *
 * References:
 *
 * Schmitt P, Griswold MA, Jakob PM, Kotas M, Gulani V, Flentje M, Haase A.
 * Inversion recovery TrueFISP: Quantification of T1, T2, and spin density.
 * Magn Reson Med 2004;51:661-667.
 *
 * Ehses P, Seiberlich N, Ma D, Breuer FA, Jakob PM, Griswold MA, Gulan, V.
 * IR TrueFISP with a golden‐ratio‐based radial readout: Fast quantification of T1, T2, and proton density.
 * Magn Reson Med 2013;69:71-81.
 *
 */
static bool test_rot_irbssfp_simulation(void)
{
	float angle = 45.;
	float repetition = 100;

	float fa = DEG2RAD(angle);

	float t1n = WATER_T1;
	float t2n = WATER_T2;
	float m0n = 1.;

	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;

        sim_data.seq.type = SIM_ROT;
	sim_data.seq.seq_type = SEQ_IRBSSFP;
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.0015;
	sim_data.seq.rep_num = repetition;
	sim_data.seq.spin_num = 1;
        sim_data.seq.perfect_inversion = true;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = sim_data.seq.te;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1[0] = 1. / t1n;
	sim_data.voxel.r2[0] = 1. / t2n;
	sim_data.voxel.m0[0] = m0n;
	sim_data.voxel.w = 0;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.sinc = pulse_sinc_defaults;
	sim_data.pulse.sinc.super.flipangle = angle;

        // Choose close to hard-pulse approximation -> same assumptions as analytical model
	sim_data.pulse.rf_end = 0.;

	sim_data.grad = simdata_grad_defaults;

        sim_data.other = simdata_other_defaults;
        sim_data.other.sampling_rate = 1.E4;

	int N = sim_data.seq.rep_num;

	float mxy_sig[N][3];
	float sa_r1_sig[N][3];
	float sa_r2_sig[N][3];
	float sa_m0_sig[N][3];
	float sa_b1_sig[N][3];

	bloch_simulation(&sim_data, N, &mxy_sig, &sa_r1_sig, &sa_r2_sig, &sa_m0_sig, &sa_b1_sig);


	// Analytical Model

	float t1s = 1. / ((cosf(fa / 2.) * cosf(fa / 2.)) / t1n + (sinf(fa / 2.) * sinf(fa / 2.)) / t2n);
	float s0 = m0n * sinf(fa / 2.);
	float stst = m0n * sinf(fa) / ((t1n / t2n + 1.) - cosf(fa) * (t1n / t2n - 1.));
	float inv = 1. + s0 / stst;


        // Model Comparison

	float out_simu = 0.;
	float out_theory = 0.;
	float err = 0.;

	for (int z = 0; z < repetition; z++) {

                //Does NOT include phase information!
                // + data.tr through alpha/2 preparation

		out_theory = fabs(stst * (1. - inv * expf(-((float)(z + 1.) * sim_data.seq.tr) / t1s)));

		out_simu = cabsf(mxy_sig[z][0] + mxy_sig[z][1] * I);

		err = fabsf(out_simu - out_theory);

		if (1.E-3 < err)
			return false;
	}

	return true;
}

UT_REGISTER_TEST(test_rot_irbssfp_simulation);



/* Simulation Comparison IR FLASH
 *      ODE <-> STM
 * Stages of complexity:
 *      1. Inversion
 *      2. bSSFP
 *      3. Relaxation
 *      4. Slice-selection gradient moment != 0
 *      5. Off-Resonance
 */
static bool test_stm_ode_bssfp_comparison(void)
{
	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = SEQ_IRBSSFP;
	sim_data.seq.tr = 0.0045;
	sim_data.seq.te = 0.00225;
	sim_data.seq.rep_num = 500;
	sim_data.seq.spin_num = 1;
        sim_data.seq.perfect_inversion = true;
	sim_data.seq.inversion_pulse_length = 0.01;
        sim_data.seq.inversion_spoiler = 0.005;
	sim_data.seq.prep_pulse_length = sim_data.seq.te;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1[0] = 1. / WATER_T1;
	sim_data.voxel.r2[0] = 1. / WATER_T2;
	sim_data.voxel.m0[0] = 1;
	sim_data.voxel.w = 0;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.sinc = pulse_sinc_defaults;
	sim_data.pulse.sinc.super.flipangle = 45.;
        sim_data.pulse.rf_end = 0.001;

	sim_data.grad = simdata_grad_defaults;
        sim_data.other = simdata_other_defaults;

        sim_data.grad.mom_sl = 0.25 * 2. * M_PI * 1000.;	// [rad/s]
        sim_data.voxel.w = 0.25 * 2. * M_PI * 1000.;	// [rad/s]

	struct sim_data sim_ode = sim_data;

	int R = sim_ode.seq.rep_num;

	float mxy_ode[R][3];
	float sa_r1_ode[R][3];
	float sa_r2_ode[R][3];
	float sa_m0_ode[R][3];
	float sa_b1_ode[R][3];

	bloch_simulation(&sim_ode, R, &mxy_ode, &sa_r1_ode, &sa_r2_ode, &sa_m0_ode, &sa_b1_ode);

        sim_data.seq.type = SIM_STM;

	assert(R == sim_data.seq.rep_num);

	float mxy_matexp[R][3];
	float sa_r1_matexp[R][3];
	float sa_r2_matexp[R][3];
	float sa_m0_matexp[R][3];
	float sa_b1_matexp[R][3];

	bloch_simulation(&sim_data, R, &mxy_matexp, &sa_r1_matexp, &sa_r2_matexp, &sa_m0_matexp, &sa_b1_matexp);

	float tol = 1.E-2;
	float err;

	for (int rep = 0; rep < sim_data.seq.rep_num; rep++) {

        	for (int dim = 0; dim < 3; dim++) {

			err = fabsf(mxy_matexp[rep][dim] - mxy_ode[rep][dim]);
			if (tol < err)
				return false;

			err = fabsf(sa_r1_matexp[rep][dim] - sa_r1_ode[rep][dim]);
			if (tol < err)
				return false;

			err = fabsf(sa_r2_matexp[rep][dim] - sa_r2_ode[rep][dim]);
			if (tol < err)
				return false;

			err = fabsf(sa_m0_matexp[rep][dim] - sa_m0_ode[rep][dim]);
			if (tol < err)
				return false;

			err = fabsf(sa_b1_matexp[rep][dim] - sa_b1_ode[rep][dim]);
			if (tol < err)
				return false;
		}
        }

	return true;
}

UT_REGISTER_TEST(test_stm_ode_bssfp_comparison);


/* Simulation Comparison IR FLASH
 *      ODE <-> STM
 * Stages of complexity:
 *      1. Inversion
 *      2. FLASH
 *      3. Relaxation
 *      4. Slice-selection gradient moment != 0
 *      5. Off-Resonance
 */
static bool test_stm_ode_flash_comparison(void)
{
	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = SEQ_IRFLASH;
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.0017;
	sim_data.seq.rep_num = 500;
	sim_data.seq.spin_num = 1;
        sim_data.seq.perfect_inversion = true;
	sim_data.seq.inversion_pulse_length = 0.01;
        sim_data.seq.inversion_spoiler = 0.005;
	sim_data.seq.prep_pulse_length = sim_data.seq.te;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1[0] = 1. / WATER_T1;
	sim_data.voxel.r2[0] = 1. / WATER_T2;
	sim_data.voxel.m0[0] = 1;
	sim_data.voxel.w = 0;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.sinc = pulse_sinc_defaults;
	sim_data.pulse.sinc.super.flipangle = 8.;
        sim_data.pulse.rf_end = 0.001;

	sim_data.grad = simdata_grad_defaults;
        sim_data.other = simdata_other_defaults;

        sim_data.grad.mom_sl = 0.25 * 2. * M_PI * 1000.;	// [rad/s]
        sim_data.voxel.w =  0.25 * 2. * M_PI * 1000.;	// [rad/s]

	struct sim_data sim_ode = sim_data;

	int R = sim_ode.seq.rep_num;

	float mxy_ode[R][3];
	float sa_r1_ode[R][3];
	float sa_r2_ode[R][3];
	float sa_m0_ode[R][3];
	float sa_b1_ode[R][3];

	assert(R == sim_data.seq.rep_num);

	bloch_simulation(&sim_ode, R, &mxy_ode, &sa_r1_ode, &sa_r2_ode, &sa_m0_ode, &sa_b1_ode);

        sim_data.seq.type = SIM_STM;

	float mxy_matexp[R][3];
	float sa_r1_matexp[R][3];
	float sa_r2_matexp[R][3];
	float sa_m0_matexp[R ][3];
	float sa_b1_matexp[R][3];

	bloch_simulation(&sim_data, R, &mxy_matexp, &sa_r1_matexp, &sa_r2_matexp, &sa_m0_matexp, &sa_b1_matexp);

	float tol = 1.E-2;
	float err;

	for (int rep = 0; rep < sim_data.seq.rep_num; rep++) {

        	for (int dim = 0; dim < 3; dim++) {

			err = fabsf(mxy_matexp[rep][dim] - mxy_ode[rep][dim]);
			if (tol < err)
				return false;

			err = fabsf(sa_r1_matexp[rep][dim] - sa_r1_ode[rep][dim]);
			if (tol < err)
				return false;

			err = fabsf(sa_r2_matexp[rep][dim] - sa_r2_ode[rep][dim]);
			if (tol < err)
				return false;

			err = fabsf(sa_m0_matexp[rep][dim] - sa_m0_ode[rep][dim]);
			if (tol < err)
				return false;

			err = fabsf(sa_b1_matexp[rep][dim] - sa_b1_ode[rep][dim]);
			if (tol < err)
				return false;
		}
        }

	return true;
}

UT_REGISTER_TEST(test_stm_ode_flash_comparison);



// Test off-resonance effect in ODE simulation
//      - Set off-resonance so that magnetization is rotated by 90 degree within TE
//      - for w == 0 -> Mxy = 0+1*I
//      - Goal: Mxy = 1+0*I
static bool test_ode_simu_offresonance(void)
{
	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = SEQ_FLASH;	// Does not have preparation phase
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.001;
	sim_data.seq.rep_num = 1;
	sim_data.seq.spin_num = 1;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = 0.;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1[0] = 0.;	// Turn off relaxation
	sim_data.voxel.r2[0] = 0.;	// Turn off relaxation
	sim_data.voxel.m0[0] = 1.;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.sinc = pulse_sinc_defaults;
	sim_data.pulse.sinc.super.flipangle = 90.;
	sim_data.pulse.rf_end = 1.E-8;	// Close to Hard-Pulses

        sim_data.grad = simdata_grad_defaults;
        sim_data.other = simdata_other_defaults;

	sim_data.voxel.w = 0.25 * 2. * M_PI * 1000.;	// [rad/s]

	int R = sim_data.seq.rep_num;

	float mxySig_ode[R][3];
	float saR1Sig_ode[R][3];
	float saR2Sig_ode[R][3];
	float saDensSig_ode[R][3];
	float sa_b1_ode[R][3];

	bloch_simulation(&sim_data, R, &mxySig_ode, &saR1Sig_ode, &saR2Sig_ode, &saDensSig_ode, &sa_b1_ode);

	float tol = 1.E-4;

	UT_RETURN_ASSERT(   (fabs(mxySig_ode[0][0] - 1.) < tol)
		  && (fabs(mxySig_ode[0][1] - 0.) < tol)
		  && (fabs(mxySig_ode[0][2] - 0.) < tol));
}

UT_REGISTER_TEST(test_ode_simu_offresonance);


// Test off-resonance effect in STM simulation
//      - Set off-resonance so that magnetization is rotated by 90 degree within TE
//      - for w == 0 -> Mxy = 0+1*I
//      - Goal: Mxy = 1+0*I
static bool test_stm_simu_offresonance(void)
{
	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
        sim_data.seq.type = SIM_STM;
	sim_data.seq.seq_type = SEQ_FLASH;	// Does not have preparation phase
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.001;
	sim_data.seq.rep_num = 1;
	sim_data.seq.spin_num = 1;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = 0.;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1[0] = 0.;	// Turn off relaxation
	sim_data.voxel.r2[0] = 0.;	// Turn off relaxation
	sim_data.voxel.m0[0] = 1.;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.sinc = pulse_sinc_defaults;
	sim_data.pulse.sinc.super.flipangle = 90.;
	sim_data.pulse.rf_end = 1.E-8;	// Close to Hard-Pulses

        sim_data.grad = simdata_grad_defaults;
        sim_data.other = simdata_other_defaults;

	sim_data.voxel.w =  0.25 * 2. * M_PI * 1000.;	// [rad/s]

	int R = sim_data.seq.rep_num;

	float mxySig_ode[R][3];
	float saR1Sig_ode[R][3];
	float saR2Sig_ode[R][3];
	float saDensSig_ode[R][3];
	float sa_b1_ode[R][3];

	bloch_simulation(&sim_data, R, &mxySig_ode, &saR1Sig_ode, &saR2Sig_ode, &saDensSig_ode, &sa_b1_ode);

	float tol = 1.E-4;

	UT_RETURN_ASSERT(   (fabs(mxySig_ode[0][0] - 1.) < tol)
		  && (fabs(mxySig_ode[0][1] - 0.) < tol)
		  && (fabs(mxySig_ode[0][2] - 0.) < tol));
}

UT_REGISTER_TEST(test_stm_simu_offresonance);


// Test gradient during relaxation
//      - Simulate gradient dephasing by PI/2 between RF_end and TE of first repetition
//      - This should turn the magnetization from 0,1,0 to 1,0,0
static bool test_ode_simu_gradient(void)
{
	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = SEQ_FLASH;	// Does not have preparation phase
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.001;
	sim_data.seq.rep_num = 1;
	sim_data.seq.spin_num = 1;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = 0.;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1[0] = 0.;	// Turn off relaxation
	sim_data.voxel.r2[0] = 0.;	// Turn off relaxation
	sim_data.voxel.m0[0] = 1.;
	sim_data.voxel.w = 0.;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.sinc = pulse_sinc_defaults;
	sim_data.pulse.sinc.super.flipangle = 90.;
	sim_data.pulse.rf_end = 1.E-8;	// Close to Hard-Pulses

	sim_data.grad = simdata_grad_defaults;
	sim_data.grad.mom = 0.25 * 2. * M_PI * 1000.;	// [rad/s]

        sim_data.other = simdata_other_defaults;

	int R = sim_data.seq.rep_num;

	float mxySig_ode[R][3];
	float saR1Sig_ode[R][3];
	float saR2Sig_ode[R][3];
	float saDensSig_ode[R][3];
	float sa_b1_ode[R][3];

	bloch_simulation(&sim_data, R, &mxySig_ode, &saR1Sig_ode, &saR2Sig_ode, &saDensSig_ode, &sa_b1_ode);

	float tol = 1.E-4;

	UT_RETURN_ASSERT(   (fabs(mxySig_ode[R - 1][0] - 1.) < tol)
		  && (fabs(mxySig_ode[R - 1][1] - 0.) < tol)
		  && (fabs(mxySig_ode[R - 1][2] - 0.) < tol));
}

UT_REGISTER_TEST(test_ode_simu_gradient);


// Test gradient during relaxation STM
//      - Simulate gradient dephasing by PI/2 between RF_end and TE of first repetition
//      - This should turn the magnetization from 0,1,0 to 1,0,0
static bool test_stm_simu_gradient(void)
{
	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
        sim_data.seq.type = SIM_STM;
	sim_data.seq.seq_type = SEQ_FLASH;	// Does not have preparation phase
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.001;
	sim_data.seq.rep_num = 1;
	sim_data.seq.spin_num = 1;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = 0.;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1[0] = 0.;	// Turn off relaxation
	sim_data.voxel.r2[0] = 0.;	// Turn off relaxation
	sim_data.voxel.m0[0] = 1.;
	sim_data.voxel.w = 0.;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.sinc = pulse_sinc_defaults;
	sim_data.pulse.sinc.super.flipangle = 90.;
	sim_data.pulse.rf_end = 1.E-8;	// Close to Hard-Pulses

	sim_data.grad = simdata_grad_defaults;
	sim_data.grad.mom = 0.25 * 2. * M_PI * 1000.;	// [rad/s]

        sim_data.other = simdata_other_defaults;

	int R = sim_data.seq.rep_num;

	float mxySig_ode[R][3];
	float saR1Sig_ode[R][3];
	float saR2Sig_ode[R][3];
	float saDensSig_ode[R][3];
	float sa_b1_ode[R][3];

	bloch_simulation(&sim_data, R, &mxySig_ode, &saR1Sig_ode, &saR2Sig_ode, &saDensSig_ode, &sa_b1_ode);

	float tol = 1.E-4;

	UT_RETURN_ASSERT(   (fabs(mxySig_ode[R - 1][0] - 1.) < tol)
		  && (fabs(mxySig_ode[R - 1][1] - 0.) < tol)
		  && (fabs(mxySig_ode[R - 1][2] - 0.) < tol) );
}

UT_REGISTER_TEST(test_stm_simu_gradient);


// Shaihan J Malik, Alessandro Sbrizzi, Hans Hoogduin, and Joseph V Hajnal
// Equivalence of EPG and Isochromat-based simulation of MR signals
// Proc. Intl. Soc. Mag. Reson. Med. 24 (2016), No. 3196
static void ode_fourier_modes(int N, struct sim_data* data, complex float fn[N], float angle)
{
	complex float m_plus[N];

	int t = data->seq.rep_num - 1;

	// Perform ODE simulations for isochromates

	for (int i = 0; i < N; i++) {

		struct sim_data sim_ode = *data;

		sim_ode.voxel.w = angle * i / N;

		int R = sim_ode.seq.rep_num;

		float mxySig_ode[R][3];
		float saR1Sig_ode[R][3];
		float saR2Sig_ode[R][3];
		float saDensSig_ode[R][3];
		float sa_b1_ode[R][3];

		bloch_simulation(&sim_ode, R, &mxySig_ode, &saR1Sig_ode, &saR2Sig_ode, &saDensSig_ode, &sa_b1_ode);

		// Save M+
		m_plus[i] = mxySig_ode[t][0] + mxySig_ode[t][1] * I;
	}

	// Estimate Fn based on DFT

	for (int j = 0; j < N; j++) {

		fn[j] = 0.;

		for (int m = 0; m < N; m++)
			fn[j] += m_plus[m] * cexpf(-2. * M_PI * I * (-(float)N / 2. + j) * (float)m / (float)N);

		fn[j] /= N; // Scale to compensate for Fn \prop N
	}
}

// Show relation between EPG and isochromate simulations
// Idea: use isochromates off-resonance distributions to loop through Fourier coefficients
static bool test_ode_epg_relation(void)
{
	// General simulation details

	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = SEQ_FLASH;
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.001;
	sim_data.seq.rep_num = 1;
	sim_data.seq.spin_num = 1;
	sim_data.seq.inversion_pulse_length = 0.;
        sim_data.seq.perfect_inversion = true;
	sim_data.seq.prep_pulse_length = 0.;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1[0] = 0.;	// Turn off relaxation
	sim_data.voxel.r2[0] = 0.;	// Turn off relaxation
	sim_data.voxel.m0[0] = 1.;
	sim_data.voxel.w = 0.;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.sinc = pulse_sinc_defaults;
	sim_data.pulse.sinc.super.flipangle = 8.;
	sim_data.pulse.rf_end = 1E-8;	// Close to Hard-Pulses

	sim_data.grad = simdata_grad_defaults;
        sim_data.other = simdata_other_defaults;

	// Estimate Fourier modes from ODE simulation

	int N = 10; //number of isochromates

	complex float fn[N];

	float angles[4] = { 0., 1., 2., 3. };	// [rotations/ms]

	complex float test_modes[4] = { 0. };

	for (int i = 0; i < 4; i++) {

		ode_fourier_modes(N, &sim_data, fn, angles[i] * (2. * M_PI * 1000.));	// [rad/s]

		test_modes[i] = fn[N/2+i];
	}

	// Compute F(n=0) mode with EPG

	int T = sim_data.seq.rep_num;
	int M = 2 * T;

	complex float signal[T];
	complex float states[3][M][T]; // 3 -> dims: Fn,F-n,Zn; M: k-states; T: repetition

	flash_epg_der(T, M, signal, states, NULL, NULL, sim_data.pulse.sinc.super.flipangle,
			sim_data.seq.tr, 1000000., 1000000., 1., sim_data.voxel.w, 0L);

	float tol = 1.E-4;

	// "* -1" for EPG rotations defined COUNTER clockwise, while Bloch equations rotate ation CLOCKWISE around x
	// FIXME: Redefine EPG clockwise
	UT_RETURN_ASSERT(   (fabs(cimagf(-1. * states[0][0][T - 1]) - cimagf(test_modes[0])) < tol)
		  && (fabs(cimagf(+1. * states[0][1][T - 1]) - cimagf(test_modes[1])) < tol));
}

UT_REGISTER_TEST(test_ode_epg_relation);


/* HARD PULSE Simulation
 *
 * Compare the simulated IR bSSFP signal with the analytical model
 *
 * Assumptions: 1. TR << T_{1,2}
 *              2. T_RF \approx 0
 */
static bool test_hp_irbssfp_simulation(void)
{
	float angle = 45.;
	float repetition = 100;

	float fa = DEG2RAD(angle);

	float t1n = WATER_T1;
	float t2n = WATER_T2;
	float m0n = 1.;

	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = SEQ_IRBSSFP;
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.0015;
	sim_data.seq.rep_num = repetition;
	sim_data.seq.spin_num = 1;
        sim_data.seq.perfect_inversion = true;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = sim_data.seq.te;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1[0] = 1. / t1n;
	sim_data.voxel.r2[0] = 1. / t2n;
	sim_data.voxel.m0[0] = m0n;
	sim_data.voxel.w = 0;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.sinc = pulse_sinc_defaults;
	sim_data.pulse.sinc.super.flipangle = angle;

        // Choose HARD-PULSE approximation
        // -> same assumptions as analytical model
	sim_data.pulse.rf_end = 0.;

	sim_data.grad = simdata_grad_defaults;
        sim_data.other = simdata_other_defaults;

	int R = sim_data.seq.rep_num;

	float mxy_sig[R][3];
	float sa_r1_sig[R][3];
	float sa_r2_sig[R][3];
	float sa_m0_sig[R][3];
	float sa_b1_sig[R][3];

	bloch_simulation(&sim_data, R, &mxy_sig, &sa_r1_sig, &sa_r2_sig, &sa_m0_sig, &sa_b1_sig);


	// Analytical Model
	float t1s = 1. / ((cosf(fa / 2.) * cosf(fa / 2.)) / t1n + (sinf(fa / 2.) * sinf(fa / 2.)) / t2n);
	float s0 = m0n * sinf(fa / 2.);
	float stst = m0n * sinf(fa) / ((t1n / t2n + 1.) - cosf(fa) * (t1n / t2n - 1.));
	float inv = 1. + s0 / stst;


        // Model Comparison
	float out_simu = 0.;
	float out_theory = 0.;
	float err = 0.;

	for (int z = 0; z < repetition; z++) {

                //Does NOT include phase information!
                // + data.tr through alpha/2 preparation
		out_theory = fabs(stst * (1. - inv * expf(-((float)(z + 1) * sim_data.seq.tr) / t1s)));

		out_simu = cabsf(mxy_sig[z][1] + mxy_sig[z][0] * I);

		err = fabsf(out_simu - out_theory);

		if (1.E-3 < err)
			return false;
	}

	return true;
}

UT_REGISTER_TEST(test_hp_irbssfp_simulation);


// Test off-resonance effect in HARD PULSE simulation
//      - Set off-resonance so that magnetization is rotated by 90 degree within TE
//      - for w == 0 -> Mxy = 0+1*I
//	- Rotation through off-resonance clockwise
//      - Goal: Mxy = 1+0*I
static bool test_hp_simu_offresonance(void)
{
	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = SEQ_FLASH;	// Does not have preparation phase
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.001;
	sim_data.seq.rep_num = 1;
	sim_data.seq.spin_num = 1;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = 0.;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1[0] = 0.;	// Turn off relaxation
	sim_data.voxel.r2[0] = 0.;	// Turn off relaxation
	sim_data.voxel.m0[0] = 1.;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.sinc = pulse_sinc_defaults;
	sim_data.pulse.sinc.super.flipangle = 90.;
	sim_data.pulse.rf_end = 0.;	// Here: HARD PULSE!

        sim_data.grad = simdata_grad_defaults;
        sim_data.other = simdata_other_defaults;

	sim_data.voxel.w = 0.25 * 2. * M_PI * 1000.;	// [rad/s]

	int R = sim_data.seq.rep_num;

	float mxySig_ode[R][3];
	float saR1Sig_ode[R][3];
	float saR2Sig_ode[R][3];
	float saDensSig_ode[R][3];
	float sa_b1_ode[R][3];

	bloch_simulation(&sim_data, R, &mxySig_ode, &saR1Sig_ode, &saR2Sig_ode, &saDensSig_ode, &sa_b1_ode);

	float tol = 1.E-4;

	UT_RETURN_ASSERT(   (fabs(mxySig_ode[0][0] - 1.) < tol)
		  && (fabs(mxySig_ode[0][1] - 0.) < tol)
		  && (fabs(mxySig_ode[0][2] - 0.) < tol));
}

UT_REGISTER_TEST(test_hp_simu_offresonance);


// Test gradient during relaxation in HARD PULSE simulation
//      - Simulate gradient dephasing by PI/2 between RF_end and TE of first repetition
//      - This should turn the magnetization from 0,1,0 to 1,0,0
static bool test_hp_simu_gradient(void)
{
	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = SEQ_FLASH;	// Does not have preparation phase
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.001;
	sim_data.seq.rep_num = 1;
	sim_data.seq.spin_num = 1;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = 0.;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1[0] = 0.;	// Turn off relaxation
	sim_data.voxel.r2[0] = 0.;	// Turn off relaxation
	sim_data.voxel.m0[0] = 1.;
	sim_data.voxel.w = 0.;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.sinc = pulse_sinc_defaults;
	sim_data.pulse.sinc.super.flipangle = 90.;
	sim_data.pulse.rf_end = 0.;	// Here: HARD PULSE!

	sim_data.grad = simdata_grad_defaults;
	sim_data.grad.mom = 0.25 * 2. * M_PI * 1000.;	// [rad/s]

        sim_data.other = simdata_other_defaults;

	int R = sim_data.seq.rep_num;

	float mxySig_ode[R][3];
	float saR1Sig_ode[R][3];
	float saR2Sig_ode[R][3];
	float saDensSig_ode[R][3];
	float sa_b1_ode[R][3];

	bloch_simulation(&sim_data, R, &mxySig_ode, &saR1Sig_ode, &saR2Sig_ode, &saDensSig_ode, &sa_b1_ode);

	float tol = 1.E-4;

	UT_RETURN_ASSERT(   (fabs(mxySig_ode[sim_data.seq.rep_num - 1][0] - 1.) < tol)
 		  && (fabs(mxySig_ode[sim_data.seq.rep_num - 1][1] - 0.) < tol)
		  && (fabs(mxySig_ode[sim_data.seq.rep_num - 1][2] - 0.) < tol));
}

UT_REGISTER_TEST(test_hp_simu_gradient);


// Test refocussing of z-gradient moment in ODE simulation
// Idea:        Without balanced z moments the magnetization state changes
//              for different repetitions. In the refocused case it
//              is almost unaffected.
static bool test_ode_z_gradient_refocus(void)
{
	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
        sim_data.seq.type = SIM_ODE;
	sim_data.seq.seq_type = SEQ_BSSFP;
	sim_data.seq.tr = 0.004;
	sim_data.seq.te = 0.002;
	sim_data.seq.rep_num = 2;
	sim_data.seq.spin_num = 1;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = sim_data.seq.te;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1[0] = 0.;
	sim_data.voxel.r2[0] = 0.;
	sim_data.voxel.m0[0] = 1.;
	sim_data.voxel.w = 0;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.sinc = pulse_sinc_defaults;
	sim_data.pulse.sinc.super.flipangle = 45.;

	sim_data.pulse.rf_end = 0.001;

	sim_data.grad = simdata_grad_defaults;
        sim_data.grad.mom_sl = 0.5 * 2. * M_PI * 1000.;	// [rad/s]

        sim_data.other = simdata_other_defaults;

	int R = sim_data.seq.rep_num;

	float mxy_sig[R][3];
	float sa_r1_sig[R][3];
	float sa_r2_sig[R][3];
	float sa_m0_sig[R][3];
	float sa_b1_sig[R][3];

	bloch_simulation(&sim_data, R, &mxy_sig, &sa_r1_sig, &sa_r2_sig, &sa_m0_sig, &sa_b1_sig);

        sim_data.grad.mom_sl = 0.5 * 2. * M_PI * 1000.;	// [rad/s]

        float mxy_sig2[R][3];
	float sa_r1_sig2[R][3];
	float sa_r2_sig2[R][3];
	float sa_m0_sig2[R][3];
	float sa_b1_sig2[R][3];

	bloch_simulation(&sim_data, R, &mxy_sig2, &sa_r1_sig2, &sa_r2_sig2, &sa_m0_sig2, &sa_b1_sig2);

	float tol = 1.E-2;

	UT_RETURN_ASSERT(   (fabsf(mxy_sig[0][0] - mxy_sig[1][0]) < tol)
		  && (fabsf(mxy_sig[0][1] - mxy_sig[1][1]) < tol)
		  && (fabsf(mxy_sig[0][2] - mxy_sig[1][2]) < tol));
}

UT_REGISTER_TEST(test_ode_z_gradient_refocus);


// Test refocussing of z-gradient moment in STM simulation
// Idea:        Without balanced z moments the magnetization state changes
//              for different repetitions. In the refocused case it
//              is almost unaffected.
static bool test_stm_z_gradient_refocus(void)
{
	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
        sim_data.seq.type = SIM_STM;
	sim_data.seq.seq_type = SEQ_BSSFP;
	sim_data.seq.tr = 0.004;
	sim_data.seq.te = 0.002;
	sim_data.seq.rep_num = 2;
	sim_data.seq.spin_num = 1;
	sim_data.seq.inversion_pulse_length = 0.;
	sim_data.seq.prep_pulse_length = sim_data.seq.te;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1[0] = 0.;
	sim_data.voxel.r2[0] = 0.;
	sim_data.voxel.m0[0] = 1.;
	sim_data.voxel.w = 0;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.sinc = pulse_sinc_defaults;
	sim_data.pulse.sinc.super.flipangle = 45.;

	sim_data.pulse.rf_end = 0.001;

	sim_data.grad = simdata_grad_defaults;
        sim_data.grad.mom_sl = 0.5 * 2. * M_PI * 1000.;	// [rad/s]

        sim_data.other = simdata_other_defaults;

	int R = sim_data.seq.rep_num;

	float mxy_sig[R][3];
	float sa_r1_sig[R][3];
	float sa_r2_sig[R][3];
	float sa_m0_sig[R][3];
	float sa_b1_sig[R][3];

	bloch_simulation(&sim_data, R, &mxy_sig, &sa_r1_sig, &sa_r2_sig, &sa_m0_sig, &sa_b1_sig);

	float tol = 1.E-2;

	UT_RETURN_ASSERT(   (fabsf(mxy_sig[0][0] - mxy_sig[1][0]) < tol)
		  && (fabsf(mxy_sig[0][1] - mxy_sig[1][1]) < tol)
		  && (fabsf(mxy_sig[0][2] - mxy_sig[1][2]) < tol));
}

UT_REGISTER_TEST(test_stm_z_gradient_refocus);


// Test inversion pulse
static bool test_ode_inversion(void)
{
        enum { N = 3 };              // Number of dimensions (x, y, z)
	enum { P = 4 };              // Number of parameters with estimated derivative (Mxy, R1, NULL, R2, B1)

        struct sim_data data;

        data.seq = simdata_seq_defaults;
        data.seq.seq_type = SEQ_FLASH;
        data.seq.tr = 0.001;
        data.seq.te = 0.001;
        data.seq.rep_num = 1;
        data.seq.spin_num = 1;
        data.seq.inversion_pulse_length = 0.01;
        data.seq.inversion_spoiler = 0.005;

        data.voxel = simdata_voxel_defaults;
        data.voxel.r1[0] = 0.;
        data.voxel.r2[0] = 0.;
        data.voxel.m0[0] = 1;
        data.voxel.w = 0;

        data.pulse = simdata_pulse_defaults;
	data.pulse.sinc = pulse_sinc_defaults;
        data.pulse.sinc.super.flipangle = 0.;
        data.pulse.rf_end = 0.01;

        data.grad = simdata_grad_defaults;
        data.other = simdata_other_defaults;

        float xp[P][N] = { { 0., 0., 1. }, { 0. }, { 0. }, { 0. } };

        float h = 1.E-4;
        float tol = 0.005; // >99.5% inversion efficiency

        inversion(&data, h, tol, N, P, xp, 0., 0.005);

        UT_RETURN_ASSERT(fabs(xp[0][2] + 1.) < tol);
}

UT_REGISTER_TEST(test_ode_inversion);


// Test STM matrix creation.
// Test with
//      - RF pulse: sim_data.seq.pulse_applied = true;
//      - Relaxation: sim_data.voxel.r1[0] and sim_data.voxel.r2[0] != 0;
//      - z-Gradient: sim_data.grad.gb[2] = 2. * M_PI * 1000.;
static bool test_stm_matrix_creation(void)
{
	struct sim_data sim_data;

	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.seq_type = SEQ_FLASH;
	sim_data.seq.tr = 0.003;
	sim_data.seq.te = 0.0017;
        sim_data.seq.pulse_applied = true;

	sim_data.voxel = simdata_voxel_defaults;
	sim_data.voxel.r1[0] = 10.;
	sim_data.voxel.r2[0] = 1.;
	sim_data.voxel.m0[0] = 1.;

	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.sinc = pulse_sinc_defaults;
	sim_data.pulse.sinc.super.flipangle = 90.;

        sim_data.grad = simdata_grad_defaults;
        sim_data.grad.gb[2] = 2. * M_PI * 1000.;	// [rad/s]

        sim_data.other = simdata_other_defaults;

        // Prepare RF pulse to have correct flip angle
        pulse_sinc_init(&sim_data.pulse.sinc, sim_data.pulse.rf_end - sim_data.pulse.rf_start, sim_data.pulse.sinc.super.flipangle, 
			sim_data.pulse.phase, sim_data.pulse.sinc.bwtp, sim_data.pulse.sinc.alpha);


        // Create STM for Bloch equation only

        float t0 = 0.;
        float t1 = 0.001;

        int N1 = 4;
        float out1[4][4] = { { 0. }, { 0. }, { 0. }, { 0. } };

        float m1[4] = { 0. };
        m1[2] = -1.;
        m1[3] = 1.;

        mat_exp_simu(&sim_data, 0., N1, t0, t1, out1);
        apply_sim_matrix(N1, m1, out1);


        // Create STM for Bloch + SA with dR1, dM0, dR2

        int N2 = 10;
        float out2[10][10] = {  { 0. }, { 0. }, { 0. }, { 0. },
                                { 0. }, { 0. }, { 0. }, { 0. },
                                { 0. }, { 0. } };

        float m2[10] = { 0. };
        m2[2] = -1.;
        m2[9] = 1.;

        mat_exp_simu(&sim_data, 0., N2, t0, t1, out2);
        apply_sim_matrix(N2, m2, out2);

        // Compare signal part of STM matrices estimated above by its effect on the magnetization

        float tol = 1.E-5;

	UT_RETURN_ASSERT(   (fabs(m1[0] - m2[0]) < tol)
		  && (fabs(m1[1] - m2[1]) < tol)
		  && (fabs(m1[2] - m2[2]) < tol)
		  && (fabs(m1[3] - m2[9]) < tol));
}

UT_REGISTER_TEST(test_stm_matrix_creation);


// Validate the partial derivatives of Bloch-McConnell
// equations estimated with the SA with the
// difference quotient method for estimating gradients.

static bool test_ode_bloch_mcc_simulation_gradients(void)
{
	float e = 1.E-3;
	float tol = 1.E-4;

 	struct sim_data sim_data;

	sim_data.grad = simdata_grad_defaults;
	sim_data.other = simdata_other_defaults;
	sim_data.seq = simdata_seq_defaults;
	sim_data.pulse = simdata_pulse_defaults;
	sim_data.voxel = simdata_voxel_defaults;

	sim_data.seq.seq_type = SEQ_IRBSSFP;
	sim_data.seq.type = SIM_ODE;
	sim_data.seq.tr = 0.004;
	sim_data.seq.te = 0.002;
	sim_data.seq.rep_num = 45;
	sim_data.seq.spin_num = 1;
	sim_data.seq.inversion_pulse_length = 0.01;
	sim_data.seq.prep_pulse_length = sim_data.seq.te;

	sim_data.voxel.r1[0] = 1. / WATER_T1;
	sim_data.voxel.r2[0] = 1. / WATER_T2;
	sim_data.voxel.r1[1] = 1. / WATER_T1;
	sim_data.voxel.r2[1] = 10000.;
	sim_data.voxel.m0[1] = 0.20;
	sim_data.voxel.m0[0] = 1.;
	sim_data.voxel.k[0] = 30.;

	sim_data.voxel.Om[1] = 100.;
	sim_data.voxel.P = 2;

	sim_data.pulse.sinc.super.flipangle = 45.;
	sim_data.pulse.rf_end = 0.001;

	int R = sim_data.seq.rep_num;
	float mxy_ref_sig[R][sim_data.voxel.P][3];
	float sa_r1_ref_sig[R][sim_data.voxel.P][3];
	float sa_r2_ref_sig[R][sim_data.voxel.P][3];
	float sa_m0_ref_sig[R][sim_data.voxel.P][3];
	float sa_b1_ref_sig[R][1][3];
	float sa_k_ref_sig[R][sim_data.voxel.P][3];
	float sa_om_ref_sig[R][sim_data.voxel.P][3];

	sim_data.seq.model = MODEL_BMC;

	bloch_simulation2(&sim_data, R, sim_data.voxel.P, &mxy_ref_sig, &sa_r1_ref_sig, &sa_r2_ref_sig, &sa_m0_ref_sig, &sa_b1_ref_sig, &sa_k_ref_sig, &sa_om_ref_sig);


	 /* ------------ R1 Partial Derivative Test -------------- */

	float mxy_tmp_sig[R][sim_data.voxel.P][3];
	float sa_r1_tmp_sig[R][sim_data.voxel.P][3];
	float sa_r2_tmp_sig[R][sim_data.voxel.P][3];
	float sa_m0_tmp_sig[R][sim_data.voxel.P][3];
	float sa_b1_tmp_sig[R][1][3];
	float sa_k_tmp_sig[R][sim_data.voxel.P][3];
	float sa_om_tmp_sig[R][sim_data.voxel.P][3];

	struct sim_data data_r1 = sim_data;

	data_r1.voxel.r1[0] += e;

	bloch_simulation2(&data_r1, R, sim_data.voxel.P, &mxy_tmp_sig, &sa_r1_tmp_sig, &sa_r2_tmp_sig, &sa_m0_tmp_sig, &sa_b1_tmp_sig, &sa_k_tmp_sig, &sa_om_tmp_sig);

	float err = 0;

	for (int i = 0; i < sim_data.seq.rep_num; i++) {

		for (int j = 0; j < 3; j++) {

			err = fabsf(e * sa_r1_ref_sig[i][0][j] - (mxy_tmp_sig[i][0][j] - mxy_ref_sig[i][0][j]));

			if (tol < err)
				return false;
		}
	}


	/* ------------ R2 Partial Derivative Test -------------- */

	struct sim_data data_r2 = sim_data;

	data_r2.voxel.r2[0] += e;

	bloch_simulation2(&data_r2, R, sim_data.voxel.P, &mxy_tmp_sig, &sa_r1_tmp_sig, &sa_r2_tmp_sig, &sa_m0_tmp_sig, &sa_b1_tmp_sig, &sa_k_tmp_sig, &sa_om_tmp_sig);

	for (int i = 0; i < R; i++) {

	 	for (int j = 0; j < 3; j++) {

	 		err = fabsf(e * sa_r2_ref_sig[i][0][j] - (mxy_tmp_sig[i][0][j] - mxy_ref_sig[i][0][j]));

	 		if (tol < err)
				return false;
	 	}
	}

	/* ------------ M0 Partial Derivative Test -------------- */

	struct sim_data data_m0 = sim_data;

	data_m0.voxel.m0[0] += e;
	bloch_simulation2(&data_m0, R, sim_data.voxel.P, &mxy_tmp_sig, &sa_r1_tmp_sig, &sa_r2_tmp_sig, &sa_m0_tmp_sig, &sa_b1_tmp_sig, &sa_k_tmp_sig, &sa_om_tmp_sig);

 	for (int i = 0; i < sim_data.seq.rep_num; i++) {
 		for (int j = 0; j < 3; j++) {

 			err = fabsf(e * sa_m0_ref_sig[i][0][j] - (mxy_tmp_sig[i][0][j] - mxy_ref_sig[i][0][j]));

 			if (tol < err)
				return false;
 		}
	}

	/* ------------ B1 Partial Derivative Test -------------- */

	struct sim_data data_b1 = sim_data;

	data_b1.voxel.b1 += e;

	bloch_simulation2(&data_b1, R, sim_data.voxel.P, &mxy_tmp_sig, &sa_r1_tmp_sig, &sa_r2_tmp_sig, &sa_m0_tmp_sig, &sa_b1_tmp_sig, &sa_k_tmp_sig, &sa_om_tmp_sig);

	for (int i = 0; i < R; i++) {
	 	for (int j = 0; j < 3; j++) {

	 		err = fabsf(e * sa_b1_ref_sig[i][0][j] - (mxy_tmp_sig[i][0][j] - mxy_ref_sig[i][0][j]));

	 		if (tol < err)
				return false;
	 	}
	}

	/* ------------ R1_2 Partial Derivative Test -------------- */

	struct sim_data data_r1_2 = sim_data;

	data_r1_2.voxel.r1[1] += e;

	bloch_simulation2(&data_r1_2, R, sim_data.voxel.P, &mxy_tmp_sig, &sa_r1_tmp_sig, &sa_r2_tmp_sig, &sa_m0_tmp_sig, &sa_b1_tmp_sig, &sa_k_tmp_sig, &sa_om_tmp_sig);

	for (int i = 0; i < R; i++) {
		for (int j = 0; j < 3; j++) {

	 		err = fabsf(e * sa_r1_ref_sig[i][1][j] - (mxy_tmp_sig[i][0][j] - mxy_ref_sig[i][0][j]));

	 		if (tol < err)
	 			return false;
	 	}
	 }

	/* ------------ R2_2 Partial Derivative Test -------------- */

	struct sim_data data_r2_2 = sim_data;

	data_r2_2.voxel.r2[1] += e;

	bloch_simulation2(&data_r2_2, R, sim_data.voxel.P, &mxy_tmp_sig, &sa_r1_tmp_sig, &sa_r2_tmp_sig, &sa_m0_tmp_sig, &sa_b1_tmp_sig, &sa_k_tmp_sig, &sa_om_tmp_sig);

 	for (int i = 0; i < R; i++) {
 		for (int j = 0; j < 3; j++) {

 			err = fabsf(e * sa_r2_ref_sig[i][1][j] - (mxy_tmp_sig[i][0][j] - mxy_ref_sig[i][0][j]));

 			if (tol < err)
				return false;
 		}
 	}


	/* ------------ k Partial Derivative Test -------------- */

	struct sim_data data_k = sim_data;
	data_k.voxel.k[0] += 10 * e;


	bloch_simulation2(&data_k, R, sim_data.voxel.P, &mxy_tmp_sig, &sa_r1_tmp_sig, &sa_r2_tmp_sig, &sa_m0_tmp_sig, &sa_b1_tmp_sig, &sa_k_tmp_sig, &sa_om_tmp_sig);

 	for (int i = 0; i < R; i++) {
 		for (int j = 0; j < 3; j++) {

 			err = fabsf(10 * e * sa_k_ref_sig[i][0][j] - (mxy_tmp_sig[i][0][j] - mxy_ref_sig[i][0][j]));

 			if (tol < err)
				return false;
 		}
 	}
	 /* ------------ M0 2 Partial Derivative Test -------------- */

	 struct sim_data data_m0_2 = sim_data;
	 data_m0_2.voxel.m0[1] += e;


	bloch_simulation2(&data_m0_2, R, sim_data.voxel.P, &mxy_tmp_sig, &sa_r1_tmp_sig, &sa_r2_tmp_sig, &sa_m0_tmp_sig, &sa_b1_tmp_sig, &sa_k_tmp_sig, &sa_om_tmp_sig);

	for (int i = 0; i < R; i++) {
		for (int j = 0; j < 3; j++) {

			err = fabsf(e * sa_m0_ref_sig[i][1][j] - (mxy_tmp_sig[i][0][j] - mxy_ref_sig[i][0][j]));

			if (tol < err)
				return false;
	 	}
	}

	/* ------------ Om Partial Derivative Test -------------- */

	struct sim_data data_Om = sim_data;

	data_Om.voxel.Om[1] += 100;

	bloch_simulation2(&data_Om, R, sim_data.voxel.P, &mxy_tmp_sig, &sa_r1_tmp_sig, &sa_r2_tmp_sig, &sa_m0_tmp_sig, &sa_b1_tmp_sig, &sa_k_tmp_sig, &sa_om_tmp_sig);

	for (int i = 0; i < R; i++) {
		for (int j = 0; j < 3; j++) {

			err = fabsf(100 * sa_om_ref_sig[i][0][j] - (mxy_tmp_sig[i][0][j] - mxy_ref_sig[i][0][j]));

			if (tol < err)
				return false;
		}
	}

	return true;
}

UT_REGISTER_TEST(test_ode_bloch_mcc_simulation_gradients);


// Test signal of BMC simulation 
//	- Do BMC sim with 3 identical pools without exchange
//	- Compare to 1 pool Bloch simulation

static bool test_bmc_ode_irbssfp_signal(void)
{
	float tol = 5.E-4;

 	struct sim_data sim_data;

	sim_data.grad = simdata_grad_defaults;
	sim_data.other = simdata_other_defaults;
	sim_data.seq = simdata_seq_defaults;
	sim_data.pulse = simdata_pulse_defaults;
	sim_data.voxel = simdata_voxel_defaults;

	sim_data.seq.seq_type = SEQ_IRBSSFP;
	sim_data.seq.type = SIM_ODE;
	sim_data.seq.tr = 0.004;
	sim_data.seq.te = 0.002;
	sim_data.seq.rep_num = 30;
	sim_data.seq.spin_num = 1;
	sim_data.seq.inversion_pulse_length = 0.01;
	sim_data.seq.inversion_spoiler = 0.005;

	int P = 3;

	for (int p = 0; p < P; p++) {

		sim_data.voxel.r1[p] = 1. / WATER_T1;
		sim_data.voxel.r2[p] = 1. / WATER_T2;
		sim_data.voxel.m0[p] = 1.;
		sim_data.voxel.Om[p] = 0.;

		if (p < 4)
			sim_data.voxel.k[p] = 0.;
	}

	sim_data.pulse.sinc.super.flipangle = 45.;
	sim_data.pulse.rf_end = 0.001;

	int R = sim_data.seq.rep_num;

	float mxy_sig[R][3];
	float sa_r1_sig[R][3];
	float sa_r2_sig[R][3];
	float sa_m0_sig[R][3];
	float sa_b1_sig[R][3];

	sim_data.other.ode_tol = 2.e-6;

	bloch_simulation(&sim_data, R, &mxy_sig, &sa_r1_sig, &sa_r2_sig, &sa_m0_sig, &sa_b1_sig);

	struct sim_data sim_data_pools = sim_data;
	sim_data_pools.voxel.P = P;

	sim_data_pools.seq.model = MODEL_BMC;

	float mxy_pools[R][sim_data_pools.voxel.P][3];
	float sa_r1_pools[R][sim_data_pools.voxel.P][3];
	float sa_r2_pools[R][sim_data_pools.voxel.P][3];
	float sa_m0_pools[R][sim_data_pools.voxel.P][3];
	float sa_b1_pools[R][1][3];
	float sa_k_pools[R][sim_data_pools.voxel.P][3];
	float sa_om_pools[R][sim_data_pools.voxel.P][3];

	bloch_simulation2(&sim_data_pools, R, sim_data_pools.voxel.P, &mxy_pools, &sa_r1_pools, &sa_r2_pools, &sa_m0_pools,
						&sa_b1_pools, &sa_k_pools, &sa_om_pools);

	float err = 0.;

	for (int p = 0; p < sim_data_pools.voxel.P; p++) {

		for (int r = 0; r < R; r++) {

			for (int d = 0; d < 3; d++) {

				err = fabsf(mxy_pools[r][p][d] - mxy_sig[r][d]);

				if (err > tol)
					return false;
			}
		}
	}

	return true;
}

UT_REGISTER_TEST(test_bmc_ode_irbssfp_signal);



// Test 5 pool BMC simulation
// 	- 5 pool sim with 1 water pool, 4 identical MT pools
//	- Compare signal and partial derivatives of MT pools

static bool test_ode_bmc_5pool(void)
{
	float tol = 1.E-6;

 	struct sim_data sim_data;

	sim_data.grad = simdata_grad_defaults;
	sim_data.other = simdata_other_defaults;
	sim_data.seq = simdata_seq_defaults;
	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.sinc = pulse_sinc_defaults;
	sim_data.voxel = simdata_voxel_defaults;

	sim_data.seq.seq_type = SEQ_IRBSSFP;
	sim_data.seq.type = SIM_ODE;
	sim_data.seq.tr = 0.004;
	sim_data.seq.te = 0.002;
	sim_data.seq.rep_num = 30;
	sim_data.seq.spin_num = 1;
	sim_data.seq.inversion_pulse_length = 0.01;
	sim_data.seq.inversion_spoiler = 0.005;
	sim_data.voxel.P = 5;

	sim_data.voxel.r1[0] = 1. / WATER_T1;
	sim_data.voxel.r2[0] = 10.;
	sim_data.voxel.m0[0] = 1.;
	sim_data.voxel.k[0] = 10.;
	sim_data.voxel.Om[0] = 0.;

	for (int i = 1; i < sim_data.voxel.P; i++) {

		sim_data.voxel.r1[i] = 1. / WATER_T1;
		sim_data.voxel.r2[i] = 1000.;
		sim_data.voxel.m0[i] = 1.;
		sim_data.voxel.Om[i] = -0.5 * 2. * M_PI * 3. * 42.5764; // 0.5 ppm offset

		if (sim_data.voxel.P - 1 > i)
			sim_data.voxel.k[i] = 10.;
	}

	sim_data.pulse.sinc.super.flipangle = 45.;
	sim_data.pulse.rf_end = 0.001;

	int R = sim_data.seq.rep_num;
	int P = sim_data.voxel.P;

	float m[R][sim_data.voxel.P][3];
	float sa_r1[R][sim_data.voxel.P][3];
	float sa_r2[R][sim_data.voxel.P][3];
	float sa_m0[R][sim_data.voxel.P][3];
	float sa_b1[R][1][3];
	float sa_k[R][sim_data.voxel.P][3];
	float sa_om[R][sim_data.voxel.P][3];

	sim_data.seq.model = MODEL_BMC;

	bloch_simulation2(&sim_data, R, P, &m, &sa_r1, &sa_r2, &sa_m0, &sa_b1, &sa_k, &sa_om);

	for (int r = 0; r < R; r++) {
		for (int d = 0; d < 3; d++) {
			for (int p = 2; p < P; p++) {

				float err = fabsf(m[r][1][d] - m[r][p][d]);
				if (err > tol)
					return false;

				err = fabsf(sa_r1[r][1][d] - sa_r1[r][p][d]);
				if (err > tol)
					return false;

				err = fabsf(sa_r2[r][1][d] - sa_r2[r][p][d]);
				if (err > tol)
					return false;

				err = fabsf(sa_m0[r][1][d] - sa_m0[r][p][d]);
				if (err > tol)
					return false;
			}

			for (int p = 1; p < P - 1; p++) {

				float err = fabsf(sa_k[r][0][d] - sa_k[r][p][d]);
				if (err > tol)
					return false;
			}

			for (int p = 2; p < P - 1; p++) {

				float err = fabsf(sa_om[r][1][d] - sa_om[r][p][d]);
				if (err > tol)
					return false;
			}      
		}
	}

	return true;
}

UT_REGISTER_TEST(test_ode_bmc_5pool);



// Test BMC 5 pool simulation
//	Compare with case 4 from the pulseq BMsim challenge
//	Pick 7 off-resonance values for speedup
//	See : https://github.com/pulseq-cest/BMsim_challenge
static bool test_mcconnell_CEST_ode_sim(void)
{
	struct sim_data sim_data;
	sim_data.seq = simdata_seq_defaults;
	sim_data.seq.model = MODEL_BMC;
	sim_data.pulse = simdata_pulse_defaults;
	sim_data.pulse.sinc = pulse_sinc_defaults;
	sim_data.pulse.rf_end = 0.005;

	sim_data.grad = simdata_grad_defaults;
	sim_data.other = simdata_other_defaults;
	sim_data.voxel = simdata_voxel_defaults;

	float w_larmor = 2. * M_PI * 3. * 42.5764;
	sim_data.voxel.P = 5;

	// Water pool
	sim_data.voxel.r1[0] = 1.;
	sim_data.voxel.r2[0] = 1. / 0.040;
	sim_data.voxel.m0[0] = 1.;

	// MT pool
	sim_data.voxel.r1[1] = 1.;
	sim_data.voxel.r2[1] = 1. / 4.e-5;
	sim_data.voxel.Om[1] = 3.0 * w_larmor; 
	sim_data.voxel.m0[1] = 0.1351;
	sim_data.voxel.k[0] = 30.;

	// CEST pool
	sim_data.voxel.r1[2] = 1.;
 	sim_data.voxel.r2[2] = 1. / 0.1;
	sim_data.voxel.Om[2] = -3.5 * w_larmor; 
	sim_data.voxel.m0[2] = 0.0009009;
	sim_data.voxel.k[1] = 50.;

	// Guanidine
	sim_data.voxel.r1[3] = 1.;
 	sim_data.voxel.r2[3] = 1. / 0.1;
	sim_data.voxel.Om[3] = -2. * w_larmor; 
	sim_data.voxel.m0[3] = 0.0009009;
	sim_data.voxel.k[2] = 1000.;

	// NOE
	sim_data.voxel.r1[4] = 1. / 1.3;
 	sim_data.voxel.r2[4] = 1. / 0.005;
	sim_data.voxel.Om[4] = 3. * w_larmor; 
	sim_data.voxel.m0[4] = 0.0045;
	sim_data.voxel.k[3] = 20.;

	// Reference values from group Stollberger 1
	float ref_sb[7] = { 0.604649192, 0.902927875, 0.577800915, 0.215424391, 0.576719564, 0.903064561, 0.60501979 };

	// Initialize ODE
	int P = 5 * sim_data.voxel.P;
	int N = 3 * sim_data.voxel.P;

	float xp[P][N];

	for (int p = 0; p < P; p++)
		for (int n = 0; n < N; n++)
			xp[p][n] = 0.;

 	for (int p = 0; p < sim_data.voxel.P; p++) {

		xp[0][2 + p * 3] = sim_data.voxel.m0[p];
		xp[2 + 2 * sim_data.voxel.P + p][2 + p * 3] = 1.;
	}

	float tol = 5.e-6;
	float h = 1.e-7;
	float err = 0;

	// Rectangular pulse with 3.7 uT amplitude
	sim_data.pulse.type = PULSE_REC;
	sim_data.pulse.rect = pulse_rect_defaults;
	sim_data.pulse.rect.A = 3.7 * 2. * M_PI * 42.5764;

	// Off-resonance vector
	float offset[7] = { -1.5, -1., -0.5, 0., 0.5, 1., 1.5 };

	for (int i = 0; i < 7; i++)
		offset[i] *= w_larmor;

	// Hardcode reference scan at -300 ppm for speedup
	float ref_scan = 0.99990;

	// Loop over frequency offsets, apply RF pulse and post-prep delay
	for (int i = 0; i < 7; i++) {
		
		sim_data.voxel.w = offset[i];

		rf_pulse(&sim_data, h, tol, N, P, xp, NULL);

		sim_data.voxel.w = 0;

		// Post prep delay
		relaxation2(&sim_data, h, tol, N, P, xp, 0., 0.0065, NULL, 10000.);

		err = fabsf((xp[0][2] / ref_scan) - ref_sb[i]);

		if (err > 1.E-3)
			return false;

		// reset xp
		for (int p = 0; p < P; p++)
			for (int n = 0; n < N; n++)
				xp[p][n] = 0.;

		for (int p = 0; p < sim_data.voxel.P; p++) {

			xp[0][2 + p * 3] = sim_data.voxel.m0[p];
			xp[2 + 2 * sim_data.voxel.P + p][2 + p * 3] = 1.;
		}
	}

	return true;
}

UT_REGISTER_TEST(test_mcconnell_CEST_ode_sim)


// Test BMC 5 pool CEST simulation with a 5 ms long block pulse for saturation
//	Compare with the BART submission to the BMC simulation challenge for case 4
//	See : https://github.com/pulseq-cest/BMsim_challenge
static bool test_ode_bloch_cest_simulation_with_single_precision_case4(void)
{
	// Define default sim data
	struct sim_data sim_data;
	sim_data.seq = simdata_seq_defaults;
	sim_data.pulse = simdata_pulse_defaults;
	sim_data.grad = simdata_grad_defaults;
	sim_data.other = simdata_other_defaults;
	sim_data.voxel = simdata_voxel_defaults;
	sim_data.seq.model = MODEL_BMC;
	sim_data.voxel.P = 5;

	// Define CEST parameters
	sim_data.seq.seq_type = SEQ_CEST;
	sim_data.seq.rep_num = 81; // Number of offsets
	sim_data.cest = simdata_cest_defaults;
	sim_data.cest.n_pulses = 1;
	sim_data.pulse.rf_end = 0.005;
	sim_data.cest.off_start = -2.;
	sim_data.cest.off_stop = 2.;
	sim_data.cest.t_pp = 0.0065;
	sim_data.cest.b1_amp = 3.7;
	sim_data.cest.b0 = 3.;
	sim_data.cest.ref_scan = true;
	sim_data.cest.ref_scan_ppm = -300.;
	float om_larmor = sim_data.cest.b0 * 2 * M_PI * sim_data.cest.gamma;

	// Define pool settings
	// * Water pool
	sim_data.voxel.r1[0] = 1.;
	sim_data.voxel.r2[0] = 1. / 0.040;
	sim_data.voxel.m0[0] = 1.;

	// * MT pool
	sim_data.voxel.r1[1] = 1.;
	sim_data.voxel.r2[1] = 1. / 4.0e-5;
	sim_data.voxel.Om[1] = 3.0 * om_larmor;
	sim_data.voxel.m0[1] = 0.1351;
	sim_data.voxel.k[0] = 30.;

	// * CEST pool
	sim_data.voxel.r1[2] = 1.;
	sim_data.voxel.r2[2] = 1. / 0.1;
	sim_data.voxel.Om[2] = -3.5 * om_larmor;
	sim_data.voxel.m0[2] = 0.0009009;
	sim_data.voxel.k[1] = 50.;

	// * Guanidine
	sim_data.voxel.r1[3] = 1.;
	sim_data.voxel.r2[3] = 1. / 0.1;
	sim_data.voxel.Om[3] = -2. * om_larmor;
	sim_data.voxel.m0[3] = 0.0009009;
	sim_data.voxel.k[2] = 1000.;

	// * NOE
	sim_data.voxel.r1[4] = 1. / 1.3;
	sim_data.voxel.r2[4] = 1. / 0.005;
	sim_data.voxel.Om[4] = 3. * om_larmor;
	sim_data.voxel.m0[4] = 0.0045;
	sim_data.voxel.k[3] = 20.;

	// Define ODE solver settings
	sim_data.other.ode_tol = 1.e-5;
	sim_data.other.ode_h = 1.e-7;
	sim_data.other.ode_h_min = -1;

	int R = sim_data.seq.rep_num;

	float mxy_pools[R][sim_data.voxel.P][3];
	float sa_r1_pools[R][sim_data.voxel.P][3];
	float sa_r2_pools[R][sim_data.voxel.P][3];
	float sa_m0_pools[R][sim_data.voxel.P][3];
	float sa_b1_pools[R][1][3];
	float sa_k_pools[R][sim_data.voxel.P][3];
	float sa_om_pools[R][sim_data.voxel.P][3];

	// Reference spectrum that was submitted to the BMsim challenge (BART (SP))
	static const float ref_spec[] = 
	{
		0.469944298267365, 0.454959988594055, 0.446434348821640, 0.444657504558563,
		0.449727147817612, 0.461542189121246, 0.479801505804062, 0.504008412361145,
		0.533477306365967, 0.567357420921326, 0.604647994041443, 0.644228518009186,
		0.684890627861023, 0.725368559360504, 0.764380991458893, 0.800662517547607,
		0.833008706569672, 0.860309004783630, 0.881579458713531, 0.895996987819672,
		0.902927398681641, 0.901940643787384, 0.892833411693573, 0.875633955001831,
		0.850605070590973, 0.818240761756897, 0.779253661632538, 0.734557747840881,
		0.685242176055908, 0.632540225982666, 0.577801406383514, 0.522444486618042,
		0.467925548553467, 0.415690332651138, 0.367134004831314, 0.323564022779465,
		0.286156445741653, 0.255924284458160, 0.233694598078728, 0.220071539282799,
		0.215424939990044, 0.219884172081947, 0.233327031135559, 0.255388587713242,
		0.285466194152832, 0.322739720344543, 0.366201847791672, 0.414675414562225,
		0.466857969760895, 0.521354794502258, 0.576720416545868, 0.631498098373413,
		0.684265911579132, 0.733676254749298, 0.778487384319305, 0.817609012126923,
		0.850120306015015, 0.875305652618408, 0.892664968967438, 0.901928722858429,
		0.903063952922821, 0.896271288394928, 0.881971001625061, 0.860796213150024,
		0.833568334579468, 0.801265239715576, 0.764997363090515, 0.725967884063721,
		0.685442328453064, 0.644704520702362, 0.605018734931946, 0.567599058151245,
		0.533569335937500, 0.503932297229767, 0.479547083377838, 0.461102902889252,
		0.449099659919739, 0.443846613168716, 0.445449531078339, 0.453814208507538,
		0.468657284975052
	};

	bloch_simulation2(&sim_data, R, sim_data.voxel.P, &mxy_pools, &sa_r1_pools, &sa_r2_pools,
			  &sa_m0_pools, &sa_b1_pools, &sa_k_pools, &sa_om_pools);

	for (int i = 0; i < R; i++) {

		debug_printf(DP_DEBUG3, " iter=%2.1d: offset=%2.5f, z-spectra=%.13f \n",
			     i,
			     sim_data.cest.off_start + i * (sim_data.cest.off_stop - sim_data.cest.off_start) / (R - 1),
			     mxy_pools[i][0][2]);

		float err = fabsf(mxy_pools[i][0][2] - ref_spec[i]);
		UT_RETURN_ON_FAILURE_TOL(err, 1E-5);
	}

	return true;
}

UT_REGISTER_TEST(test_ode_bloch_cest_simulation_with_single_precision_case4)

// Test BMC 5 pool CEST simulation with a 5 ms long block pulse for saturation
//	Compare with the BART submission to the BMC simulation challenge for case 4
//	See : https://github.com/pulseq-cest/BMsim_challenge
static bool test_ode_bloch_cest_simulation_with_single_precision_h_min_case4(void)
{
	// Define default sim data
	struct sim_data sim_data;
	sim_data.seq = simdata_seq_defaults;
	sim_data.pulse = simdata_pulse_defaults;
	sim_data.grad = simdata_grad_defaults;
	sim_data.other = simdata_other_defaults;
	sim_data.voxel = simdata_voxel_defaults;
	sim_data.seq.model = MODEL_BMC;
	sim_data.voxel.P = 5;

	// Define CEST parameters
	sim_data.seq.seq_type = SEQ_CEST;
	sim_data.seq.rep_num = 81; // Number of offsets
	sim_data.cest = simdata_cest_defaults;
	sim_data.cest.n_pulses = 1;
	sim_data.pulse.rf_end = 0.005;
	sim_data.cest.off_start = -2.;
	sim_data.cest.off_stop = 2.;
	sim_data.cest.t_pp = 0.0065;
	sim_data.cest.b1_amp = 3.7;
	sim_data.cest.b0 = 3.;
	sim_data.cest.ref_scan = true;
	sim_data.cest.ref_scan_ppm = -300.;
	float om_larmor = sim_data.cest.b0 * 2 * M_PI * sim_data.cest.gamma;

	// Define pool settings
	// * Water pool
	sim_data.voxel.r1[0] = 1.;
	sim_data.voxel.r2[0] = 1. / 0.040;
	sim_data.voxel.m0[0] = 1.;

	// * MT pool
	sim_data.voxel.r1[1] = 1.;
	sim_data.voxel.r2[1] = 1. / 4.0e-5;
	sim_data.voxel.Om[1] = 3.0 * om_larmor;
	sim_data.voxel.m0[1] = 0.1351;
	sim_data.voxel.k[0] = 30.;

	// * CEST pool
	sim_data.voxel.r1[2] = 1.;
	sim_data.voxel.r2[2] = 1. / 0.1;
	sim_data.voxel.Om[2] = -3.5 * om_larmor;
	sim_data.voxel.m0[2] = 0.0009009;
	sim_data.voxel.k[1] = 50.;

	// * Guanidine
	sim_data.voxel.r1[3] = 1.;
	sim_data.voxel.r2[3] = 1. / 0.1;
	sim_data.voxel.Om[3] = -2. * om_larmor;
	sim_data.voxel.m0[3] = 0.0009009;
	sim_data.voxel.k[2] = 1000.;

	// * NOE
	sim_data.voxel.r1[4] = 1. / 1.3;
	sim_data.voxel.r2[4] = 1. / 0.005;
	sim_data.voxel.Om[4] = 3. * om_larmor;
	sim_data.voxel.m0[4] = 0.0045;
	sim_data.voxel.k[3] = 20.;

	// Define ODE solver settings
	sim_data.other.ode_tol = 1.e-5;
	sim_data.other.ode_h = 1.e-7;
	sim_data.other.ode_h_min = 1.e-6;

	int R = sim_data.seq.rep_num;

	float mxy_pools[R][sim_data.voxel.P][3];
	float sa_r1_pools[R][sim_data.voxel.P][3];
	float sa_r2_pools[R][sim_data.voxel.P][3];
	float sa_m0_pools[R][sim_data.voxel.P][3];
	float sa_b1_pools[R][1][3];
	float sa_k_pools[R][sim_data.voxel.P][3];
	float sa_om_pools[R][sim_data.voxel.P][3];

	// Reference spectrum that was submitted to the BMsim challenge (BART (SP))
	static const float ref_spec[] = 
	{
		0.469944298267365, 0.454959988594055, 0.446434348821640, 0.444657504558563,
		0.449727147817612, 0.461542189121246, 0.479801505804062, 0.504008412361145,
		0.533477306365967, 0.567357420921326, 0.604647994041443, 0.644228518009186,
		0.684890627861023, 0.725368559360504, 0.764380991458893, 0.800662517547607,
		0.833008706569672, 0.860309004783630, 0.881579458713531, 0.895996987819672,
		0.902927398681641, 0.901940643787384, 0.892833411693573, 0.875633955001831,
		0.850605070590973, 0.818240761756897, 0.779253661632538, 0.734557747840881,
		0.685242176055908, 0.632540225982666, 0.577801406383514, 0.522444486618042,
		0.467925548553467, 0.415690332651138, 0.367134004831314, 0.323564022779465,
		0.286156445741653, 0.255924284458160, 0.233694598078728, 0.220071539282799,
		0.215424939990044, 0.219884172081947, 0.233327031135559, 0.255388587713242,
		0.285466194152832, 0.322739720344543, 0.366201847791672, 0.414675414562225,
		0.466857969760895, 0.521354794502258, 0.576720416545868, 0.631498098373413,
		0.684265911579132, 0.733676254749298, 0.778487384319305, 0.817609012126923,
		0.850120306015015, 0.875305652618408, 0.892664968967438, 0.901928722858429,
		0.903063952922821, 0.896271288394928, 0.881971001625061, 0.860796213150024,
		0.833568334579468, 0.801265239715576, 0.764997363090515, 0.725967884063721,
		0.685442328453064, 0.644704520702362, 0.605018734931946, 0.567599058151245,
		0.533569335937500, 0.503932297229767, 0.479547083377838, 0.461102902889252,
		0.449099659919739, 0.443846613168716, 0.445449531078339, 0.453814208507538,
		0.468657284975052
	};

	bloch_simulation2(&sim_data, R, sim_data.voxel.P, &mxy_pools, &sa_r1_pools, &sa_r2_pools,
			  &sa_m0_pools, &sa_b1_pools, &sa_k_pools, &sa_om_pools);

	for (int i = 0; i < R; i++) {

		debug_printf(DP_DEBUG3, " iter=%2.1d: offset=%2.5f, z-spectra=%.13f \n",
			     i,
			     sim_data.cest.off_start + i * (sim_data.cest.off_stop - sim_data.cest.off_start) / (R - 1),
			     mxy_pools[i][0][2]);

		float err = fabsf(mxy_pools[i][0][2] - ref_spec[i]);
		UT_RETURN_ON_FAILURE_TOL(err, 1E-5);
	}

	return true;
}

UT_REGISTER_TEST(test_ode_bloch_cest_simulation_with_single_precision_h_min_case4)

// Test BMC 5 pool CEST simulation with a 5 ms long block pulse for saturation and double precision
//	Compare with the BART submission to the BMC simulation challenge for case 4
//	See : https://github.com/pulseq-cest/BMsim_challenge
static bool test_ode_bloch_cest_simulation_with_double_precision_case4(void)
{
	// Define default sim data
	struct sim_data sim_data;
	sim_data.seq = simdata_seq_defaults;
	sim_data.pulse = simdata_pulse_defaults;
	sim_data.grad = simdata_grad_defaults;
	sim_data.other = simdata_other_defaults;
	sim_data.voxel = simdata_voxel_defaults;
	sim_data.seq.model = MODEL_BMC;
	sim_data.voxel.P = 5;

	// Define CEST parameters
	sim_data.seq.seq_type = SEQ_CEST;
	sim_data.seq.rep_num = 81; // Number of offsets
	sim_data.cest = simdata_cest_defaults;
	sim_data.cest.n_pulses = 1;
	sim_data.pulse.rf_end = 0.005;
	sim_data.cest.off_start = -2.;
	sim_data.cest.off_stop = 2.;
	sim_data.cest.t_pp = 0.0065;
	sim_data.cest.b1_amp = 3.7;
	sim_data.cest.b0 = 3.;
	sim_data.cest.ref_scan = true;
	sim_data.cest.ref_scan_ppm = -300.;
	sim_data.cest.double_precision = true;
	float om_larmor = sim_data.cest.b0 * 2 * M_PI * sim_data.cest.gamma;

	// Define pool settings
	// * Water pool
	sim_data.voxel.r1[0] = 1.;
	sim_data.voxel.r2[0] = 1. / 0.040;
	sim_data.voxel.m0[0] = 1.;

	// * MT pool
	sim_data.voxel.r1[1] = 1.;
	sim_data.voxel.r2[1] = 1. / 4.0e-5;
	sim_data.voxel.Om[1] = 3.0 * om_larmor;
	sim_data.voxel.m0[1] = 0.1351;
	sim_data.voxel.k[0] = 30.;

	// * CEST pool
	sim_data.voxel.r1[2] = 1.;
	sim_data.voxel.r2[2] = 1. / 0.1;
	sim_data.voxel.Om[2] = -3.5 * om_larmor;
	sim_data.voxel.m0[2] = 0.0009009;
	sim_data.voxel.k[1] = 50.;

	// * Guanidine
	sim_data.voxel.r1[3] = 1.;
	sim_data.voxel.r2[3] = 1. / 0.1;
	sim_data.voxel.Om[3] = -2. * om_larmor;
	sim_data.voxel.m0[3] = 0.0009009;
	sim_data.voxel.k[2] = 1000.;

	// * NOE
	sim_data.voxel.r1[4] = 1. / 1.3;
	sim_data.voxel.r2[4] = 1. / 0.005;
	sim_data.voxel.Om[4] = 3. * om_larmor;
	sim_data.voxel.m0[4] = 0.0045;
	sim_data.voxel.k[3] = 20.;

	// Define ODE solver settings
	sim_data.other.ode_tol = 1.e-7;
	sim_data.other.ode_h = 1.e-6;
	sim_data.other.ode_h_min = -1;

	int R = sim_data.seq.rep_num;

	float mxy_pools[R][sim_data.voxel.P][3];
	float sa_r1_pools[R][sim_data.voxel.P][3];
	float sa_r2_pools[R][sim_data.voxel.P][3];
	float sa_m0_pools[R][sim_data.voxel.P][3];
	float sa_b1_pools[R][1][3];
	float sa_k_pools[R][sim_data.voxel.P][3];
	float sa_om_pools[R][sim_data.voxel.P][3];

	// Reference spectrum that was submitted to the BMsim challenge (BART (DP))
	static const float ref_spec[] = 
	{
		0.469944685697556, 0.454960614442825, 0.446435451507568, 0.444658607244492,
		0.449728250503540, 0.461543649435043, 0.479803264141083, 0.504009544849396,
		0.533478856086731, 0.567358851432800, 0.604649245738983, 0.644230067729950,
		0.684891581535339, 0.725369393825531, 0.764380693435669, 0.800662517547607,
		0.833009064197540, 0.860309004783630, 0.881579577922821, 0.895998001098633,
		0.902927875518799, 0.901941180229187, 0.892833590507507, 0.875633537769318,
		0.850604474544525, 0.818240344524384, 0.779253780841827, 0.734557807445526,
		0.685241520404816, 0.632540166378021, 0.577800810337067, 0.522444486618042,
		0.467925578355789, 0.415690034627914, 0.367134034633636, 0.323563247919083,
		0.286155223846436, 0.255924671888351, 0.233694195747375, 0.220070019364357,
		0.215424269437790, 0.219884067773819, 0.233327820897102, 0.255388677120209,
		0.285465449094772, 0.322739988565445, 0.366201370954514, 0.414675146341324,
		0.466857850551605, 0.521354556083679, 0.576719462871552, 0.631497561931610,
		0.684266209602356, 0.733675658702850, 0.778487324714661, 0.817607879638672,
		0.850119590759277, 0.875304818153381, 0.892664194107056, 0.901928782463074,
		0.903064548969269, 0.896270930767059, 0.881971120834351, 0.860797226428986,
		0.833568692207336, 0.801265537738800, 0.764997363090515, 0.725969016551971,
		0.685443639755249, 0.644705116748810, 0.605019807815552, 0.567600369453430,
		0.533570230007172, 0.503934025764465, 0.479548871517181, 0.461103558540344,
		0.449100911617279, 0.443847805261612, 0.445450305938721, 0.453815102577209,
		0.468657344579697
	};

	bloch_simulation2(&sim_data, R, sim_data.voxel.P, &mxy_pools, &sa_r1_pools, &sa_r2_pools,
			  &sa_m0_pools, &sa_b1_pools, &sa_k_pools, &sa_om_pools);

	for (int i = 0; i < R; i++) {

		debug_printf(DP_DEBUG3, " iter=%2.1d: offset=%2.5f, z-spectra=%.13f \n",
			     i,
			     sim_data.cest.off_start + i * (sim_data.cest.off_stop - sim_data.cest.off_start) / (R - 1),
			     mxy_pools[i][0][2]);

		float err = fabsf(mxy_pools[i][0][2] - ref_spec[i]);
		UT_RETURN_ON_FAILURE_TOL(err, 1E-6);
	}

	return true;
}

UT_REGISTER_TEST(test_ode_bloch_cest_simulation_with_double_precision_case4)

// Test BMC 5 pool CEST simulation with a 5 ms long block pulse for saturation
//	Compare with the BART submission to the BMC simulation challenge for case 4
//	See : https://github.com/pulseq-cest/BMsim_challenge
static bool test_ode_bloch_cest_simulation_with_single_precision_case8(void)
{
	// Define default sim data
	struct sim_data sim_data;
	sim_data.seq = simdata_seq_defaults;
	sim_data.pulse = simdata_pulse_defaults;
	sim_data.grad = simdata_grad_defaults;
	sim_data.other = simdata_other_defaults;
	sim_data.voxel = simdata_voxel_defaults;
	sim_data.seq.model = MODEL_BMC;
	sim_data.voxel.P = 5;

	// Define CEST parameters
	sim_data.seq.seq_type = SEQ_CEST;
	sim_data.seq.rep_num = 81; // Number of offsets
	sim_data.cest = simdata_cest_defaults;
	sim_data.cest.n_pulses = 2;
	sim_data.pulse.rf_end = 5.e-3;
	sim_data.cest.off_start = -2.;
	sim_data.cest.off_stop = 2.;
	sim_data.cest.t_pp = 0.0065;
	sim_data.cest.t_d = 100e-6;
	sim_data.cest.b1_amp = 3.7;
	sim_data.cest.b0 = 3.;
	sim_data.cest.ref_scan = true;
	sim_data.cest.ref_scan_ppm = -300.;
	float om_larmor = sim_data.cest.b0 * 2 * M_PI * sim_data.cest.gamma;

	// Define pool settings
	// * Water pool
	sim_data.voxel.r1[0] = 1.;
	sim_data.voxel.r2[0] = 1. / 0.040;
	sim_data.voxel.m0[0] = 1.;

	// * MT pool
	sim_data.voxel.r1[1] = 1.;
	sim_data.voxel.r2[1] = 1. / 4.0e-5;
	sim_data.voxel.Om[1] = 3.0 * om_larmor;
	sim_data.voxel.m0[1] = 0.1351;
	sim_data.voxel.k[0] = 30.;

	// * CEST pool
	sim_data.voxel.r1[2] = 1.;
	sim_data.voxel.r2[2] = 1. / 0.1;
	sim_data.voxel.Om[2] = -3.5 * om_larmor;
	sim_data.voxel.m0[2] = 0.0009009;
	sim_data.voxel.k[1] = 50.;

	// * Guanidine
	sim_data.voxel.r1[3] = 1.;
	sim_data.voxel.r2[3] = 1. / 0.1;
	sim_data.voxel.Om[3] = -2. * om_larmor;
	sim_data.voxel.m0[3] = 0.0009009;
	sim_data.voxel.k[2] = 1000.;

	// * NOE
	sim_data.voxel.r1[4] = 1. / 1.3;
	sim_data.voxel.r2[4] = 1. / 0.005;
	sim_data.voxel.Om[4] = 3. * om_larmor;
	sim_data.voxel.m0[4] = 0.0045;
	sim_data.voxel.k[3] = 20.;

	// Define ODE solver settings
	sim_data.other.ode_tol = 1.e-5;
	sim_data.other.ode_h = 1.e-7;
	sim_data.other.ode_h_min = -1;

	int R = sim_data.seq.rep_num;

	float mxy_pools[R][sim_data.voxel.P][3];
	float sa_r1_pools[R][sim_data.voxel.P][3];
	float sa_r2_pools[R][sim_data.voxel.P][3];
	float sa_m0_pools[R][sim_data.voxel.P][3];
	float sa_b1_pools[R][1][3];
	float sa_k_pools[R][sim_data.voxel.P][3];
	float sa_om_pools[R][sim_data.voxel.P][3];

	// Reference spectrum that was submitted to the BMsim challenge (BART (SP))
	static const float ref_spec[81] =
	{
	       0.877631545066833, 0.861599326133728, 0.821091473102570, 0.758129656314850,
	       0.677306950092316, 0.585322320461273, 0.490283548831940, 0.400843769311905,
	       0.325269579887390, 0.270566225051880, 0.241746604442596, 0.241310924291611,
	       0.269008845090866, 0.321885615587234, 0.394608795642853, 0.480016231536865,
	       0.569832146167755, 0.655451893806458, 0.728728473186493, 0.782657861709595,
	       0.811926722526550, 0.813245415687561, 0.785484731197357, 0.729596614837646,
	       0.648358762264252, 0.545981764793396, 0.427639037370682, 0.298968881368637,
	       0.165608033537865, 0.032781865447760, -0.094990096986294, -0.214094057679176,
	       -0.321892738342285, -0.416712522506714, -0.497730165719986, -0.564810395240784,
	       -0.618309795856476, -0.658869028091431, -0.687214910984039, -0.703996300697327,
	       -0.709661483764648, -0.704363584518433, -0.687947809696198, -0.659970760345459,
	       -0.619783043861389, -0.566654384136200, -0.499933272600174, -0.419256031513214,
	       -0.324740201234818, -0.217192918062210, -0.098269999027252, 0.029411716386676,
	       0.162254586815834, 0.295751929283142, 0.424683153629303, 0.543409585952759,
	       0.646279752254486, 0.728093206882477, 0.784607946872711, 0.813001811504364,
	       0.812276363372803, 0.783510744571686, 0.729947686195374, 0.656862437725067,
	       0.571234822273254, 0.481201976537704, 0.395379275083542, 0.322068154811859,
	       0.268478840589523, 0.240002304315567, 0.239661306142807, 0.267773151397705,
	       0.321903258562088, 0.397090971469879, 0.486366331577301, 0.581469058990479,
	       0.673740625381470, 0.755044937133789, 0.818621873855591, 0.859817385673523,
	       0.876543283462524
	};

	bloch_simulation2(&sim_data, R, sim_data.voxel.P, &mxy_pools, &sa_r1_pools, &sa_r2_pools,
			  &sa_m0_pools, &sa_b1_pools, &sa_k_pools, &sa_om_pools);

	for (int i = 0; i < R; i++) {

		debug_printf(DP_DEBUG3, " iter=%2.1d: offset=%2.5f, z-spectra=%.13f \n",
			     i,
			     sim_data.cest.off_start + i * (sim_data.cest.off_stop - sim_data.cest.off_start) / (R - 1),
			     mxy_pools[i][0][2]);

		float err = fabsf(mxy_pools[i][0][2] - ref_spec[i]);
		UT_RETURN_ON_FAILURE_TOL(err, 1E-5);
	}

	return true;
}

UT_REGISTER_TEST(test_ode_bloch_cest_simulation_with_single_precision_case8)

// Test BMC 5 pool CEST simulation with a 5 ms long block pulse for saturation
//	Compare with the BART submission to the BMC simulation challenge for case 8
//	See : https://github.com/pulseq-cest/BMsim_challenge
static bool test_ode_bloch_cest_simulation_with_single_precision_h_min_case8(void)
{
	// Define default sim data
	struct sim_data sim_data;
	sim_data.seq = simdata_seq_defaults;
	sim_data.pulse = simdata_pulse_defaults;
	sim_data.grad = simdata_grad_defaults;
	sim_data.other = simdata_other_defaults;
	sim_data.voxel = simdata_voxel_defaults;
	sim_data.seq.model = MODEL_BMC;
	sim_data.voxel.P = 5;

	// Define CEST parameters
	sim_data.seq.seq_type = SEQ_CEST;
	sim_data.seq.rep_num = 81; // Number of offsets
	sim_data.cest = simdata_cest_defaults;
	sim_data.cest.n_pulses = 2;
	sim_data.pulse.rf_end = 5.e-3;
	sim_data.cest.off_start = -2.;
	sim_data.cest.off_stop = 2.;
	sim_data.cest.t_pp = 0.0065;
	sim_data.cest.t_d = 100e-6;
	sim_data.cest.b1_amp = 3.7;
	sim_data.cest.b0 = 3.;
	sim_data.cest.ref_scan = true;
	sim_data.cest.ref_scan_ppm = -300.;
	float om_larmor = sim_data.cest.b0 * 2 * M_PI * sim_data.cest.gamma;

	// Define pool settings
	// * Water pool
	sim_data.voxel.r1[0] = 1.;
	sim_data.voxel.r2[0] = 1. / 0.040;
	sim_data.voxel.m0[0] = 1.;

	// * MT pool
	sim_data.voxel.r1[1] = 1.;
	sim_data.voxel.r2[1] = 1. / 4.0e-5;
	sim_data.voxel.Om[1] = 3.0 * om_larmor;
	sim_data.voxel.m0[1] = 0.1351;
	sim_data.voxel.k[0] = 30.;

	// * CEST pool
	sim_data.voxel.r1[2] = 1.;
	sim_data.voxel.r2[2] = 1. / 0.1;
	sim_data.voxel.Om[2] = -3.5 * om_larmor;
	sim_data.voxel.m0[2] = 0.0009009;
	sim_data.voxel.k[1] = 50.;

	// * Guanidine
	sim_data.voxel.r1[3] = 1.;
	sim_data.voxel.r2[3] = 1. / 0.1;
	sim_data.voxel.Om[3] = -2. * om_larmor;
	sim_data.voxel.m0[3] = 0.0009009;
	sim_data.voxel.k[2] = 1000.;

	// * NOE
	sim_data.voxel.r1[4] = 1. / 1.3;
	sim_data.voxel.r2[4] = 1. / 0.005;
	sim_data.voxel.Om[4] = 3. * om_larmor;
	sim_data.voxel.m0[4] = 0.0045;
	sim_data.voxel.k[3] = 20.;

	// Define ODE solver settings
	sim_data.other.ode_tol = 1.e-5;
	sim_data.other.ode_h = 1.e-7;
	sim_data.other.ode_h_min = 1.e-6;

	int R = sim_data.seq.rep_num;

	float mxy_pools[R][sim_data.voxel.P][3];
	float sa_r1_pools[R][sim_data.voxel.P][3];
	float sa_r2_pools[R][sim_data.voxel.P][3];
	float sa_m0_pools[R][sim_data.voxel.P][3];
	float sa_b1_pools[R][1][3];
	float sa_k_pools[R][sim_data.voxel.P][3];
	float sa_om_pools[R][sim_data.voxel.P][3];

	// Reference spectrum that was submitted to the BMsim challenge (BART (SP))
	static const float ref_spec[81] =
	{
	       0.877631545066833, 0.861599326133728, 0.821091473102570, 0.758129656314850,
	       0.677306950092316, 0.585322320461273, 0.490283548831940, 0.400843769311905,
	       0.325269579887390, 0.270566225051880, 0.241746604442596, 0.241310924291611,
	       0.269008845090866, 0.321885615587234, 0.394608795642853, 0.480016231536865,
	       0.569832146167755, 0.655451893806458, 0.728728473186493, 0.782657861709595,
	       0.811926722526550, 0.813245415687561, 0.785484731197357, 0.729596614837646,
	       0.648358762264252, 0.545981764793396, 0.427639037370682, 0.298968881368637,
	       0.165608033537865, 0.032781865447760, -0.094990096986294, -0.214094057679176,
	       -0.321892738342285, -0.416712522506714, -0.497730165719986, -0.564810395240784,
	       -0.618309795856476, -0.658869028091431, -0.687214910984039, -0.703996300697327,
	       -0.709661483764648, -0.704363584518433, -0.687947809696198, -0.659970760345459,
	       -0.619783043861389, -0.566654384136200, -0.499933272600174, -0.419256031513214,
	       -0.324740201234818, -0.217192918062210, -0.098269999027252, 0.029411716386676,
	       0.162254586815834, 0.295751929283142, 0.424683153629303, 0.543409585952759,
	       0.646279752254486, 0.728093206882477, 0.784607946872711, 0.813001811504364,
	       0.812276363372803, 0.783510744571686, 0.729947686195374, 0.656862437725067,
	       0.571234822273254, 0.481201976537704, 0.395379275083542, 0.322068154811859,
	       0.268478840589523, 0.240002304315567, 0.239661306142807, 0.267773151397705,
	       0.321903258562088, 0.397090971469879, 0.486366331577301, 0.581469058990479,
	       0.673740625381470, 0.755044937133789, 0.818621873855591, 0.859817385673523,
	       0.876543283462524
	};

	bloch_simulation2(&sim_data, R, sim_data.voxel.P, &mxy_pools, &sa_r1_pools, &sa_r2_pools,
			  &sa_m0_pools, &sa_b1_pools, &sa_k_pools, &sa_om_pools);

	for (int i = 0; i < R; i++) {

		debug_printf(DP_DEBUG3, " iter=%2.1d: offset=%2.5f, z-spectra=%.13f \n",
			     i,
			     sim_data.cest.off_start + i * (sim_data.cest.off_stop - sim_data.cest.off_start) / (R - 1),
			     mxy_pools[i][0][2]);

		float err = fabsf(mxy_pools[i][0][2] - ref_spec[i]);
		UT_RETURN_ON_FAILURE_TOL(err, 1E-5);
	}

	return true;
}

UT_REGISTER_TEST(test_ode_bloch_cest_simulation_with_single_precision_h_min_case8)

// Test BMC 5 pool CEST simulation with a 5 ms long block pulse for saturation with double precision
//	Compare with the BART submission to the BMC simulation challenge for case 8
//	See : https://github.com/pulseq-cest/BMsim_challenge
static bool test_ode_bloch_cest_simulation_with_double_precision_case8(void)
{
	// Define default sim data
	struct sim_data sim_data;
	sim_data.seq = simdata_seq_defaults;
	sim_data.pulse = simdata_pulse_defaults;
	sim_data.grad = simdata_grad_defaults;
	sim_data.other = simdata_other_defaults;
	sim_data.voxel = simdata_voxel_defaults;
	sim_data.seq.model = MODEL_BMC;
	sim_data.voxel.P = 5;

	// Define CEST parameters
	sim_data.seq.seq_type = SEQ_CEST;
	sim_data.seq.rep_num = 81; // Number of offsets
	sim_data.cest = simdata_cest_defaults;
	sim_data.cest.n_pulses = 2;
	sim_data.pulse.rf_end = 5.e-3;
	sim_data.cest.off_start = -2.;
	sim_data.cest.off_stop = 2.;
	sim_data.cest.t_pp = 0.0065;
	sim_data.cest.t_d = 100e-6;
	sim_data.cest.b1_amp = 3.7;
	sim_data.cest.b0 = 3.;
	sim_data.cest.ref_scan = true;
	sim_data.cest.ref_scan_ppm = -300.;
	sim_data.cest.double_precision = true;
	float om_larmor = sim_data.cest.b0 * 2 * M_PI * sim_data.cest.gamma;

	// Define pool settings
	// * Water pool
	sim_data.voxel.r1[0] = 1.;
	sim_data.voxel.r2[0] = 1. / 0.040;
	sim_data.voxel.m0[0] = 1.;

	// * MT pool
	sim_data.voxel.r1[1] = 1.;
	sim_data.voxel.r2[1] = 1. / 4.0e-5;
	sim_data.voxel.Om[1] = 3.0 * om_larmor;
	sim_data.voxel.m0[1] = 0.1351;
	sim_data.voxel.k[0] = 30.;

	// * CEST pool
	sim_data.voxel.r1[2] = 1.;
	sim_data.voxel.r2[2] = 1. / 0.1;
	sim_data.voxel.Om[2] = -3.5 * om_larmor;
	sim_data.voxel.m0[2] = 0.0009009;
	sim_data.voxel.k[1] = 50.;

	// * Guanidine
	sim_data.voxel.r1[3] = 1.;
	sim_data.voxel.r2[3] = 1. / 0.1;
	sim_data.voxel.Om[3] = -2. * om_larmor;
	sim_data.voxel.m0[3] = 0.0009009;
	sim_data.voxel.k[2] = 1000.;

	// * NOE
	sim_data.voxel.r1[4] = 1. / 1.3;
	sim_data.voxel.r2[4] = 1. / 0.005;
	sim_data.voxel.Om[4] = 3. * om_larmor;
	sim_data.voxel.m0[4] = 0.0045;
	sim_data.voxel.k[3] = 20.;

	// Define ODE solver settings
	sim_data.other.ode_tol = 1.e-7;
	sim_data.other.ode_h = 1.e-6;
	sim_data.other.ode_h_min = -1;

	int R = sim_data.seq.rep_num;

	float mxy_pools[R][sim_data.voxel.P][3];
	float sa_r1_pools[R][sim_data.voxel.P][3];
	float sa_r2_pools[R][sim_data.voxel.P][3];
	float sa_m0_pools[R][sim_data.voxel.P][3];
	float sa_b1_pools[R][1][3];
	float sa_k_pools[R][sim_data.voxel.P][3];
	float sa_om_pools[R][sim_data.voxel.P][3];

	// Reference spectrum that was submitted to the BMsim challenge (BART (SP))
	static const float ref_spec[81] =
	{
		0.877631664276123, 0.861596763134003, 0.821089208126068, 0.758127629756927,
		0.677303314208984, 0.585318446159363, 0.490281611680984, 0.400841951370239,
		0.325268059968948, 0.270566076040268, 0.241747304797173, 0.241312086582184,
		0.269010007381439, 0.321887314319611, 0.394610375165939, 0.480017960071564,
		0.569832980632782, 0.655452787876129, 0.728728592395782, 0.782658934593201,
		0.811926960945129, 0.813245117664337, 0.785484254360199, 0.729596436023712,
		0.648358345031738, 0.545981168746948, 0.427638381719589, 0.298968851566315,
		0.165607020258904, 0.032780889421701, -0.094991490244865, -0.214094370603561,
		-0.321893483400345, -0.416713297367096, -0.497730553150177, -0.564810991287231,
		-0.618310511112213, -0.658869028091431, -0.687215030193329, -0.703998267650604,
		-0.709661841392517, -0.704363465309143, -0.687947452068329, -0.659971415996552,
		-0.619784593582153, -0.566654622554779, -0.499934405088425, -0.419256508350372,
		-0.324741303920746, -0.217194348573685, -0.098272234201431, 0.029409600421786,
		0.162252366542816, 0.295750439167023, 0.424681454896927, 0.543407797813416,
		0.646277070045471, 0.728091835975647, 0.784606516361237, 0.813001573085785,
		0.812276661396027, 0.783511698246002, 0.729947865009308, 0.656863927841187,
		0.571236073970795, 0.481204241514206, 0.395380407571793, 0.322070181369781,
		0.268480390310287, 0.240003883838654, 0.239661678671837, 0.267773479223251,
		0.321902662515640, 0.397089570760727, 0.486362487077713, 0.581465125083923,
		0.673738002777100, 0.755039811134338, 0.818617105484009, 0.859814107418060,
		0.876542150974274
	};

	bloch_simulation2(&sim_data, R, sim_data.voxel.P, &mxy_pools, &sa_r1_pools, &sa_r2_pools,
			  &sa_m0_pools, &sa_b1_pools, &sa_k_pools, &sa_om_pools);

	for (int i = 0; i < R; i++) {

		debug_printf(DP_DEBUG3, " iter=%2.1d: offset=%2.5f, z-spectra=%.13f \n",
			     i,
			     sim_data.cest.off_start + i * (sim_data.cest.off_stop - sim_data.cest.off_start) / (R - 1),
			     mxy_pools[i][0][2]);

		float err = fabsf(mxy_pools[i][0][2] - ref_spec[i]);
		UT_RETURN_ON_FAILURE_TOL(err, 1E-6);
	}

	return true;
}

UT_REGISTER_TEST(test_ode_bloch_cest_simulation_with_double_precision_case8)