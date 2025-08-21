/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2025 Moritz Blumenthal
 */

#include <complex.h>
#include <assert.h>
#include <math.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "noir/pole.h"

#include "utest.h"


static bool test_integral_line(void)
{
	float r[1][2][3] = {
		{ { 0., 0., -1000 }, { 0., 0., 1000 } },
	};

	float pos[5][3] = {
		{ 1., 1., 0. },
		{ 1., -1., 0. },
		{ -1., -1., 0. },
		{ -1., 1., 0. },
		{ 1., 1., 0. },
	};

	float dphi = integrate_phase(ARRAY_SIZE(pos), pos, ARRAY_SIZE(r), r, 1.e-6);
	return (1.e-3 > fabs(2. * M_PI - dphi));
}

UT_REGISTER_TEST(test_integral_line);

static bool test_integral_line_circle(void)
{
	float r[][2][3] = {
		{ { 0., 0., -1000 }, { 0., 0., 1000 } },
	};

	int M = 7;
	float pos[M + 1][3];

	for (int i = 0; i < M + 1; i++) {

		float t = (float)i / M;

		pos[i][0] = cosf(2. * M_PI * t);
		pos[i][1] = sinf(2. * M_PI * t);
		pos[i][2] = 0.;
	}


	float dphi = integrate_phase(ARRAY_SIZE(pos), pos, ARRAY_SIZE(r), r, 1.e-6);
	return (1.e-3 > fabs(2. * M_PI + dphi));
}

UT_REGISTER_TEST(test_integral_line_circle);


static bool test_integral_line_circle2(void)
{
	float r[][2][3] = {
		{ { 0., 0., -10000 }, { 0., 0., 0. } },
		{ { 0., 0., 0. }, { 0., 0., 10000 } },
	};

	int M = 7;
	float pos[M + 1][3];

	for (int i = 0; i < M + 1; i++) {

		float t = (float)i / M;

		pos[i][0] = cosf(2. * M_PI * t);
		pos[i][1] = sinf(2. * M_PI * t);
		pos[i][2] = 0.;
	}



	float dphi = integrate_phase(ARRAY_SIZE(pos), pos, ARRAY_SIZE(r), r, 1.e-6);
	return (1.e-3 > fabs(2. * M_PI + dphi));
}

UT_REGISTER_TEST(test_integral_line_circle2);


static bool test_integral_line_circle3(void)
{
	float r[][2][3] = {
		{ { 1000000., 0., -10000 }, { -1000000., 0., 10000 } },
	};

	int M = 7;
	float pos[M + 1][3];

	for (int i = 0; i < M + 1; i++) {

		float t = (float)i / M;

		pos[i][0] = cosf(2. * M_PI * t);
		pos[i][1] = sinf(2. * M_PI * t);
		pos[i][2] = 0.;
	}

	float dphi = integrate_phase(ARRAY_SIZE(pos), pos, ARRAY_SIZE(r), r, 1.e-6);
	return (1.e-3 > fabs(2. * M_PI + dphi));
}

UT_REGISTER_TEST(test_integral_line_circle3);

static bool test_integral_line_circle4(void)
{
	float r[][2][3] = {
		{ {  0.,  10000., 0.00 }, {  0., -10000., 0.01 } },
	};

	int M = 7;
	float pos[M + 1][3];

	for (int i = 0; i < M + 1; i++) {

		float t = (float)i / M;

		pos[i][0] = cosf(2. * M_PI * t);
		pos[i][1] = sinf(2. * M_PI * t);
		pos[i][2] = 0.;
	}

	float dphi = integrate_phase(ARRAY_SIZE(pos), pos, ARRAY_SIZE(r), r, 1.e-6);
	return (1.e-3 > fabs(dphi));
}

UT_REGISTER_TEST(test_integral_line_circle4);


static bool test_integral_line_circle5(void)
{
	float r[][2][3] = {
		{ { 100., 100., 0 }, { 0., 0., -0.1 } },
		{ { 0., 0., -0.1 }, { 0., 0., 0.1 } },
		{ { 0., 0., 0.1 }, { 0., 0., 1000 } },
	};

	int M = 7;
	float pos[M + 1][3];

	for (int i = 0; i < M + 1; i++) {

		float t = (float)i / M;

		pos[i][0] = cosf(2. * M_PI * t);
		pos[i][1] = sinf(2. * M_PI * t);
		pos[i][2] = 0.;
	}


	float dphi = integrate_phase(ARRAY_SIZE(pos), pos, ARRAY_SIZE(r), r, 1.e-6);
	return (1.e-2 > fabs(2. * M_PI + dphi));
}

UT_REGISTER_TEST(test_integral_line_circle5);


static bool test_integral_line_circle6(void)
{
	float r[][2][3] = {
		{ { 100000., 0., 0.01 }, { -100000., 0., 0.00 } },
	};

	int M = 7;
	float pos[M + 1][3];

	for (int i = 0; i < M + 1; i++) {

		float t = (float)i / M;

		pos[i][0] = cosf(2. * M_PI * t);
		pos[i][1] = sinf(2. * M_PI * t);
		pos[i][2] = 0.;
	}

	float dphi = integrate_phase(ARRAY_SIZE(pos), pos, ARRAY_SIZE(r), r, 1.e-6);
	return (1.e-3 > fabs(dphi));
}

UT_REGISTER_TEST(test_integral_line_circle6);
