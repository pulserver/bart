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

