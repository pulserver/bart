
#include <math.h>

#include "num/vec3.h"
#include "num/lineseg.h"

#include "utest.h"


static bool test_dist1(void)
{
	vec3_t seg1[2] = { { -1, 0, 0 }, { 1, 0, 0 }};
	vec3_t seg2[2] = { { 0, -1, 0 }, { 0, 1, 0 }};

	UT_RETURN_ASSERT(UT_TOL >= dist_of_linesegs(seg1, seg2));
}

UT_REGISTER_TEST(test_dist1);


static bool test_dist2(void)
{
	vec3_t seg1[2] = { { -1, 0, 0 }, { 1, 0, 0 }};
	vec3_t seg2[2] = { { 0, -1, 1 }, { 0, 1, 1 }};

	UT_RETURN_ASSERT(UT_TOL >= fabsf(1.f - dist_of_linesegs(seg1, seg2)));
}

UT_REGISTER_TEST(test_dist2);

static bool test_dist3(void)
{
	vec3_t seg1[2] = { { -1, 0, 0 }, { 1, 0, 0 }};
	vec3_t seg2[2] = { { 0, 0, 0 }, { 2, 0, 1 }};

	UT_RETURN_ASSERT(UT_TOL >= fabsf(0.f - dist_of_linesegs(seg1, seg2)));
}

UT_REGISTER_TEST(test_dist3);


static bool test_dist4(void)
{
	vec3_t seg1[2] = { { -1, 0, 0 }, { 1, 0, 0 }};
	vec3_t seg2[2] = { { 0, 0, 0 }, { 2, 0, 0 }};

	UT_RETURN_ASSERT(UT_TOL >= fabsf(0.f - dist_of_linesegs(seg1, seg2)));
}

UT_REGISTER_TEST(test_dist4);


static bool test_dist5(void)
{
	vec3_t seg1[2] = { { -1, 0, 0 }, { 1, 0, 0 }};
	vec3_t seg2[2] = { { 0, 1, 0 }, { 2, 1, 0 }};

	UT_RETURN_ASSERT(UT_TOL >= fabsf(1.f - dist_of_linesegs(seg1, seg2)));
}

UT_REGISTER_TEST(test_dist5);

