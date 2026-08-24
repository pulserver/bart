/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <math.h>
#include <stdint.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "seq/gradient.h"
#include "seq/event.h"

#include "utest.h"

static struct grad_limits sys = {
	.inv_slew_rate = 1000.,
	.max_amplitude = 1.,
};


static bool test_support_triangle(void)
{
	struct grad_support grad[2] = {
		{ 0., 0. },
		{ 10., 10. }
	};

	struct seq_event ev[2];

	int e = seq_grad_support_to_event(2, ev, 0., 2, grad, (double[3]){1., 0., 0.});

	double m0[3];
	moment_sum(m0, 10, e, ev);

	if (1 != e)
		return false;

	if (50. != m0[0])
		return false;

	return true;
}

UT_REGISTER_TEST(test_support_triangle);


static bool test_support_triangle2(void)
{
	struct grad_support grad[2] = {
		{ 0., -10. },
		{ 10., 0. }
	};

	struct seq_event ev[2];

	int e = seq_grad_support_to_event(2, ev, 0., 2, grad, (double[3]){1., 0., 0.});

	double m0[3];
	moment_sum(m0, 10, e, ev);

	if (1 != e)
		return false;

	if (-50. != m0[0])
		return false;

	return true;
}

UT_REGISTER_TEST(test_support_triangle2);


static bool test_support_rect(void)
{
	struct grad_support grad[2] = {
		{ 0., 10. },
		{ 10., 10. }
	};

	struct seq_event ev[2];

	int e = seq_grad_support_to_event(2, ev, 0., 2, grad, (double[3]){1., 0., 0.});

	double m0[3];
	moment_sum(m0, 10, e, ev);

	if (2 != e)
		return false;

	if (100. != m0[0])
		return false;

	return true;
}

UT_REGISTER_TEST(test_support_rect);


static bool test_support_M(void)
{
	struct grad_support grad[3] = {
		{ 0., 10. },
		{ 10., 0. },
		{ 20., 10. }
	};

	struct seq_event ev[4];

	int e = seq_grad_support_to_event(4, ev, 0., 3, grad, (double[3]){1., 0., 0.});

	double m0[3];
	moment_sum(m0, 20, e, ev);

	if (2 != e)
		return false;

	if (100. != m0[0])
		return false;

	return true;
}

UT_REGISTER_TEST(test_support_M);


static bool test_support_posneg(void)
{
	struct grad_support grad[2] = {
		{ 0., 10. },
		{ 20., -10. }
	};

	struct seq_event ev[2];

	int e = seq_grad_support_to_event(2, ev, 0., 2, grad, (double[3]){1., 0., 0.});

	double m0[3];
	moment_sum(m0, 20, e, ev);

	if (2 != e)
		return false;

	if (0. != m0[0])
		return false;

	return true;
}

UT_REGISTER_TEST(test_support_posneg);


static bool test_softest_gradient1(void)
{
	long available_time = 3000;
	float moment = 2000;

	struct grad_trapezoid grad;

	if (!grad_soft(&grad, available_time, moment, sys))
		return false;

	if (1. != grad.ampl)
		return false;

	if (grad_total_time(&grad) != available_time)
		return false;

	return true;
}

static bool test_softest_gradient2(void)
{
	long available_time = 6000;
	float moment = 2000;

	struct grad_trapezoid grad;

	if (!grad_soft(&grad, available_time, moment, sys))
		return false;

	if (0.4 != grad.ampl)
		return false;

	if (grad_total_time(&grad) != available_time)
		return false;

	return true;
}

static bool test_hardest_gradient1(void)
{
	float moment = 2000;

	struct grad_trapezoid grad;

	if (!grad_hard(&grad, moment, sys))
		return false;

	if (1. != grad.ampl)
		return false;

	if (3000. != grad_total_time(&grad))
		return false;

	return true;
}

static bool test_hardest_gradient2(void)
{
	float moment = -2000.;

	struct grad_trapezoid grad;

	if (!grad_hard(&grad, moment, sys))
		return false;

	if (-1. != grad.ampl)
		return false;

	if (3000. != grad_total_time(&grad))
		return false;

	return true;
}

UT_REGISTER_TEST(test_softest_gradient1);
UT_REGISTER_TEST(test_softest_gradient2);
UT_REGISTER_TEST(test_hardest_gradient1);
UT_REGISTER_TEST(test_hardest_gradient2);


static bool test_grad_support_with_offsets1(void)
{
	const double end_time = 10000.;
	const double moment[6] = { 0., 100, 500, 4000.,  -5000., 7500. };

	struct grad_support start = {   0.,  1. };
	struct grad_support end =  { end_time, -1. };

	struct grad_support support[4];

	for (int i = 0; i < 6; i++) {

		int s = grad_support_with_offsets(support, moment[i], start, end, sys);

		if (3 != s)
			return false;

		struct seq_event ev[4];
		int e = seq_grad_support_to_event(4, ev, 0., s, support, (double[3]){1., 0., 0.});

		double m0[3];
		moment_sum(m0, end_time, e, ev);

		if ((4 != e) && ((0 == moment[i]) && (2 != e)))
			return false;

		if (moment[i] != m0[0])
			return false;
	}

	return true;
}

UT_REGISTER_TEST(test_grad_support_with_offsets1);


static bool test_grad_support_with_offsets_sys2(void)
{
	static struct grad_limits sys2 = {
		.inv_slew_rate = 0.008,
		.max_amplitude = 24E-3,
	};

	const double end_time = 5;
	const double moment[4] = { 0., 0.06, -0.059, 0.119 };

	struct grad_support start = {   0.,  24.E-3 };
	struct grad_support end =  { end_time, 0 };

	struct grad_support support[4];

	for (int i = 0; i < 4; i++) {

		int s = grad_support_with_offsets(support, moment[i], start, end, sys2);

		if (3 != s)
			return false;

		struct seq_event ev[4];
		int e = seq_grad_support_to_event(4, ev, 0., s, support, (double[3]){1., 0., 0.});

		double m0[3];
		moment_sum(m0, end_time, e, ev);

		if (2 != e)
			return false;

		if (1E-6 * UT_TOL < fabs(moment[i] - m0[0]))
			return false;
	}

	return true;
}

UT_REGISTER_TEST(test_grad_support_with_offsets_sys2);


static bool test_grad_support_with_offsets_3(void)
{
	const double end_time = 10000.;
	const double moment[4] = { 0., 10000, 5000, 1000. };

	struct grad_support start = {   0.,  1. };
	struct grad_support end =  { end_time, 1. };

	struct grad_support support[4];

	for (int i = 0; i < 4; i++) {

		int s = grad_support_with_offsets(support, moment[i], start, end, sys);

		if (3 != s)
			return false;

		struct seq_event ev[4];
		int e = seq_grad_support_to_event(4, ev, 0., s, support, (double[3]){1., 0., 0.});

		double m0[3];
		moment_sum(m0, end_time, e, ev);

		if ((4 != e) && ((5000. == moment[i]) && (2 != e)))
			return false;

		if (1E-6 * UT_TOL < fabs(moment[i] - m0[0]))
			return false;
	}

	return true;
}

UT_REGISTER_TEST(test_grad_support_with_offsets_3);


static bool test_grad_support_with_offsets_4(void)
{
	const double end_time = 10000.;
	const double moment[4] = { -1000., -6000, -1 };

	struct grad_support start = {   0.,  1. };
	struct grad_support end =  { end_time, 1. };

	struct grad_support support[4];

	for (int i = 0; i < 3; i++) {

		int s = grad_support_with_offsets(support, moment[i], start, end, sys);

		if (4 != s)
			return false;

		struct seq_event ev[6];
		int e = seq_grad_support_to_event(6, ev, 0., s, support, (double[3]){1., 0., 0.});

		double m0[3];
		moment_sum(m0, end_time, e, ev);

		if (4 != e)
			return false;

		if (1E-6 * UT_TOL < fabs(moment[i] - m0[0]))
			return false;
	}

	return true;
}

UT_REGISTER_TEST(test_grad_support_with_offsets_4);


static bool test_grad_support_with_offsets_5(void)
{
	const double end_time = 10000.;
	const double moment[3] = { -8000. , 8000, 7501. };

	struct grad_support start = {   0.,  1. };
	struct grad_support end =  { end_time, -1. };

	struct grad_support support[4];

	for (int i = 0; i < 3; i++) {

		int s = grad_support_with_offsets(support, moment[i], start, end, sys);

		if (4 != s)
			return false;

		struct seq_event ev[6];
		int e = seq_grad_support_to_event(6, ev, 0., s, support, (double[3]){1., 0., 0.});

		double m0[3];
		moment_sum(m0, end_time, e, ev);

		if (4 != e)
			return false;

		if (1E-6 * UT_TOL < fabs(moment[i] - m0[0]))
			return false;
	}

	return true;
}

UT_REGISTER_TEST(test_grad_support_with_offsets_5);


static bool test_grad_support_with_offsets_6(void)
{
	const double end_time = 10000.;
	const double moment[3] = { 8875. , -6000, 8126. };

	struct grad_support start = {   0.,  -0.5 };
	struct grad_support end =  { end_time, 1 };

	struct grad_support support[4];

	for (int i = 0; i < 3; i++) {

		int s = grad_support_with_offsets(support, moment[i], start, end, sys);

		if (4 != s)
			return false;

		struct seq_event ev[6];
		int e = seq_grad_support_to_event(6, ev, 0., s, support, (double[3]){1., 0., 0.});

		double m0[3];
		moment_sum(m0, end_time, e, ev);

		if (4 != e)
			return false;

		if (1E-6 * UT_TOL < fabs(moment[i] - m0[0]))
			return false;
	}

	return true;
}

UT_REGISTER_TEST(test_grad_support_with_offsets_6);
