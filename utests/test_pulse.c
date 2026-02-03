/* Copyright 2022-2023. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author:
 *	Nick Scholand
 */

#include <complex.h>
#include <math.h>

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/nested.h"

#include "num/quadrature.h"
#include "num/flpmath.h"

#include "seq/pulse.h"

#include "simu/simulation.h"

#include "utest.h"

static bool test_sinc_integral(void)
{
        struct pulse_sinc ps = pulse_sinc_defaults;

        pulse_sinc_init(&ps, 0.001, 180., 0., 4., 0.46);

        return ((M_PI - pulse_sinc_integral(&ps)) < 1E-6);
}

UT_REGISTER_TEST(test_sinc_integral);


static bool test_sinc_integral2(void)
{
        struct pulse_sinc pulse = pulse_sinc_defaults;

        pulse_sinc_init(&pulse, 0.001, 180., 0., 4., 0.46);
	struct pulse* ps = CAST_UP(&pulse);

	int N = 50;
	float samples[N + 1];

        for (int i = 0; i <= N; i++)
		samples[i] = i * ps->duration / N;

#ifdef __clang__
		float* samples2 = samples;
#endif

	NESTED(void, eval, (float out[1], int i))
	{
#ifdef __clang__
		float* samples = samples2;
#endif
		out[0] = crealf(pulse_eval(ps, samples[i]));
	};

	float integral[1];
	quadrature_simpson_ext(N, ps->duration, 1, integral, eval);

        float error = fabs(M_PI - integral[0]);

        // debug_printf(DP_WARN, "Estimated Integral: %f,\t Error: %f\n", integral, error);

        return (error < 1E-4);
}

UT_REGISTER_TEST(test_sinc_integral2);



static bool test_sinc_zeros(void)
{
        struct pulse_sinc pulse = pulse_sinc_defaults;

        pulse_sinc_init(&pulse, 0.001, 180., 0., 4., 0.46);
	struct pulse* ps = CAST_UP(&pulse);

	int O = 10;

	for (int i = 0; i < pulse.bwtp * O; i++) {

		float t = i * ps->duration / (pulse.bwtp * O);
		float zero = crealf(pulse_eval(ps, t));

		// no zero in the center
		if (i == (pulse.bwtp * O) / 2)
			continue;

		if ((0 != i % O) && (1.E-3 > fabsf(zero)))
			return false;

		if ((0 == i % O) && (1.E-3 < fabsf(zero)))
			return false;
	}

	return true;
}

UT_REGISTER_TEST(test_sinc_zeros);

static bool test_rect_integral(void)
{
	struct pulse_rect pr = pulse_rect_defaults;

	pulse_rect_init(&pr, 1., 180., 0.);

	return ((M_PI - pr.A) < 1E-6);
}


UT_REGISTER_TEST(test_rect_integral);

// Test Accuracy of on-resonant pulse
//      1. Execute pulse for various angles and durations
//      2. Compare final magnetization to nominal angle set for the pulse
static bool test_rf_pulse_ode(void)
{
	long dim[DIMS] = { [0 ... DIMS - 1] = 1 };

        enum { N = 3 };              // Number of dimensions (x, y, z)
	enum { P = 4 };              // Number of parameters with estimated derivative (Mxy, R1, R2, B1)

	dim[READ_DIM] = 10;
	dim[PHS1_DIM] = 10;

        // RF duration
	float tmin = 0.0001;
	float tmax = 0.1;

        // Nominal FA
	float amin = 0.;
	float amax = 180.;

	for (int i = 0; i < dim[READ_DIM]; i++) {
		for (int j = 0; j < dim[PHS1_DIM]; j++) {

			float trf = (tmin + i / (dim[READ_DIM] - 1.) * (tmax - tmin));
			float angle = (amin + j / (dim[PHS1_DIM] - 1.) * (amax - amin));

                        // Define sequence characteristics
			struct sim_data data;

			data.seq = simdata_seq_defaults;
			data.seq.seq_type = SEQ_IRBSSFP;
			data.seq.tr = 10.;
			data.seq.te = 5.;
			data.seq.rep_num = 1;
			data.seq.spin_num = 1;

			data.voxel = simdata_voxel_defaults;
			data.voxel.r1[0] = 0.;
			data.voxel.r2[0] = 0.;
			data.voxel.m0[0] = 1;
			data.voxel.w = 0;

			data.pulse = simdata_pulse_defaults;
			data.pulse.sinc = pulse_sinc_defaults;
			data.pulse.sinc.super.flipangle = angle;
			data.pulse.rf_end = trf;

			data.grad = simdata_grad_defaults;


                        // Prepare pulse
			pulse_sinc_init(&data.pulse.sinc, trf, angle, 0., 4., 0.46);

			float xp[4][3] = { { 0., 0., 1. }, { 0. }, { 0. }, { 0. } };

			float h = 1E-4;
			float tol = 1E-5;

                        // Run pulse
			rf_pulse(&data, h, tol, N, P, xp, NULL);

			if (1.E-3 < fabs(xp[0][0]))
				return false;

                        // Compare result to nominal FA

			float sim_angle = RAD2DEG(atan2f(xp[0][1], xp[0][2]));

			if (sim_angle < 0.)
				sim_angle += 360.;

			float delta = 180.f - fabsf(fabsf(sim_angle - data.pulse.sinc.super.flipangle) - 180.f);

			if (1E-3 < fabs(delta)) {

				debug_printf(DP_WARN, "Error for test_rf_pulse_ode\n see -> utests/test_pulse.c\n");
				return false;
			}
		}
	}

	return true;
}

UT_REGISTER_TEST(test_rf_pulse_ode);



static bool test_hypsec_rf_pulse_ode(void)
{
        enum { N = 3 };              // Number of dimensions (x, y, z)
	enum { P = 4 };              // Number of parameters with estimated derivative (Mxy, R1, R2, B1)

        struct sim_data data;

        data.seq = simdata_seq_defaults;
        data.seq.seq_type = SEQ_FLASH;
        data.seq.tr = 0.001;
        data.seq.te = 0.001;
        data.seq.rep_num = 1;
        data.seq.spin_num = 1;

        data.voxel = simdata_voxel_defaults;
        data.voxel.r1[0] = 0.;
        data.voxel.r2[0] = 0.;
        data.voxel.m0[0] = 1;
        data.voxel.w = 0;

        data.pulse = simdata_pulse_defaults;
	data.pulse.sinc = pulse_sinc_defaults;
        data.pulse.sinc.super.flipangle = 0.;      // Turn off flipangle -> do not influence inversion efficiency
        data.pulse.rf_end = 0.01;

        // Hyperbolic Secant Characteristics
        data.pulse.hs = pulse_hypsec_defaults;
        pulse_hypsec_init(GYRO, &data.pulse.hs);
        data.pulse.type = PULSE_HS;
        data.pulse.hs.super.duration = data.pulse.rf_end;

        data.grad = simdata_grad_defaults;

        float xp[P][N] = { { 0., 0., 1. }, { 0. }, { 0. }, { 0. } };

        float h = 1E-4;
        float tol = 0.005; // >99.5% inversion efficiency

        rf_pulse(&data, h, tol, N, P, xp, NULL);

        // bart_printf("%f, %f, %f\n", xp[0][0], xp[0][1], xp[0][2]);

        UT_RETURN_ASSERT(fabs(xp[0][2] + 1.) < tol);
}

UT_REGISTER_TEST(test_hypsec_rf_pulse_ode);



static bool test_pulse_clone(void)
{
	struct pulse_sinc ps = pulse_sinc_defaults;
	pulse_sinc_init(&ps, 0.001, 60., 0., 4., 0.46);

	struct pulse_rect pr = pulse_rect_defaults;
	pulse_rect_init(&pr, 0.001, 60., 0.);

	struct pulse_hypsec ph = pulse_hypsec_defaults;
	pulse_hypsec_init(GYRO, &ph);

	struct pulse_sms pm = pulse_sms_defaults;
	pulse_sms_init(&pm, 0.001, 60., 0., 4., 0.46, 3, 0, 27.e-3, 6.e-3);

	struct pulse_arb pa = pulse_arb_oc_cest_sat_defaults;
	pulse_arb_init(&pa, GYRO);

	struct pulse_gauss pg = pulse_gauss_defaults;
	pulse_gauss_init(&pg, 0.001, 60., 0., 4., 0.46);

	struct pulse* clones[6] = {
		pulse_clone(CAST_UP(&ps)),
		pulse_clone(CAST_UP(&pr)),
		pulse_clone(CAST_UP(&ph)),
		pulse_clone(CAST_UP(&pm)),
		pulse_clone(CAST_UP(&pa)),
		pulse_clone(CAST_UP(&pg)),
	};

	UT_RETURN_ON_FAILURE(NULL != CAST_MAYBE(pulse_sinc, clones[0]));
	UT_RETURN_ON_FAILURE(NULL != CAST_MAYBE(pulse_rect, clones[1]));
	UT_RETURN_ON_FAILURE(NULL != CAST_MAYBE(pulse_hypsec, clones[2]));
	UT_RETURN_ON_FAILURE(NULL != CAST_MAYBE(pulse_sms, clones[3]));
	UT_RETURN_ON_FAILURE(NULL != CAST_MAYBE(pulse_arb, clones[4]));
	UT_RETURN_ON_FAILURE(NULL != CAST_MAYBE(pulse_gauss, clones[5]));

	for (int i = 0; i < 6; i++) {

		UT_RETURN_ON_FAILURE(NULL != clones[i]);
		pulse_free(clones[i]);
	}

	return true;
}

UT_REGISTER_TEST(test_pulse_clone);


static bool test_rect_pulse_discretize(void)
{
	struct pulse_rect pr = pulse_rect_defaults;
	pulse_rect_init(&pr, 0.02, 30., 0.);

	struct pulse* ps = CAST_UP(&pr);

	enum { N = 16 };
	complex float samples[N + 1];

	pulse_discretize(ps, N, samples);

	for (int i = 0; i < N + 1; i++)
		UT_RETURN_ON_FAILURE_TOL(cabsf(samples[i] - pr.A), UT_TOL);

	UT_RETURN_ON_FAILURE_TOL(cabsf(samples[0] - pulse_eval(ps, 0.)), UT_TOL);
	UT_RETURN_ON_FAILURE_TOL(cabsf(samples[N] - pulse_eval(ps, ps->duration)), UT_TOL);

	return true;
}

UT_REGISTER_TEST(test_rect_pulse_discretize);


static bool test_sinc_pulse_discretize(void)
{
	struct pulse_sinc ps_sinc = pulse_sinc_defaults;
	pulse_sinc_init(&ps_sinc, 0.004, 20., 0., 4., 0.46);

	struct pulse* ps = CAST_UP(&ps_sinc);

	enum { N = 40 };
	complex float samples[N + 1];

	pulse_discretize(ps, N, samples);

	for (int i = 0; i < N + 1; i++) {

		float t = i * ps->duration / N;
		UT_RETURN_ON_FAILURE_TOL(cabsf(samples[i] - pulse_eval(ps, t)), UT_TOL);
	}

	UT_RETURN_ON_FAILURE_TOL(cabsf(samples[0] - pulse_eval(ps, 0.)), UT_TOL);
	UT_RETURN_ON_FAILURE_TOL(cabsf(samples[N] - pulse_eval(ps, ps->duration)), UT_TOL);

	return true;
}

UT_REGISTER_TEST(test_sinc_pulse_discretize);