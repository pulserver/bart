/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Martin Uecker
 */

#include <assert.h>
#include <math.h>

#include "vec3.h"


void vec3_saxpy(vec3_t dst, const vec3_t src1, float alpha, const vec3_t src2)
{
	for (int i = 0; i < 3; i++)
		dst[i] = src1[i] + alpha * src2[i];
}

void vec3_sub(vec3_t dst, const vec3_t src1, const vec3_t src2)
{
	vec3_saxpy(dst, src1, -1., src2);
}

void vec3_add(vec3_t dst, const vec3_t src1, const vec3_t src2)
{
	vec3_saxpy(dst, src1, +1., src2);
}

void vec3_copy(vec3_t dst, const vec3_t src)
{
	for (int i = 0; i < 3; i++)
		dst[i] = src[i];
}

void vec3_clear(vec3_t dst)
{
	vec3_copy(dst, (vec3_t){ 0. });
}

float vec3_sdot(const vec3_t a, const vec3_t b)
{
	float ret = 0.;

	for (int i = 0; i < 3; i++)
		ret += a[i] * b[i];

	return ret;
}

float vec3_norm(const vec3_t x)
{
	return sqrtf(vec3_sdot(x, x));
}

float vec3_dist(const vec3_t a, const vec3_t b)
{
	vec3_t dif;
	vec3_sub(dif, a, b);
	return vec3_norm(dif);
}

void vec3_rot(vec3_t dst, const vec3_t src1, const vec3_t src2)
{
	vec3_t tmp;
	tmp[0] = src1[1] * src2[2] - src1[2] * src2[1];
	tmp[1] = src1[2] * src2[0] - src1[0] * src2[2];
	tmp[2] = src1[0] * src2[1] - src1[1] * src2[0];
	vec3_copy(dst, tmp);
}

void vec3_smul(vec3_t dst, const vec3_t src, float alpha)
{
	vec3_saxpy(dst, (vec3_t){ 0., 0., 0. }, alpha, src);
}



double vec3d_sdot(const vec3d_t x, const vec3d_t y)
{
	double r = 0;

	for (int i = 0; i < 3; i++)
		r += x[i] * y[i];

	return r;
}

double vec3d_norm(const vec3d_t x)
{
	return sqrt(vec3d_sdot(x, x));
}

void vec3d_saxpy(vec3d_t o, const vec3d_t x, double a, const vec3d_t y)
{
	for (int i = 0; i < 3; i++)
		o[i] = a * x[i] + y[i];
}

void vec3d_smul(vec3d_t o, const vec3d_t x, double a)
{
	vec3d_saxpy(o, x, a, (vec3d_t) { 0., 0., 0. });
}

void vec3d_clear(vec3d_t x)
{
	for (int i = 0; i < 3; i++)
		x[i] = 0.;
}

double vec3d_angle(const vec3d_t x, const vec3d_t y)
{
	assert(0 != vec3d_norm(x));
	assert(0 != vec3d_norm(y));

	double a = vec3d_sdot(x, y) / (vec3d_norm(x) * vec3d_norm(y));
	return acos(a);
}

void vec3d_crossproduct(vec3d_t o, const vec3d_t v0, const vec3d_t v1)
{
        o[0] = v0[1] * v1[2] - v0[2] * v1[1];
        o[1] = v0[2] * v1[0] - v0[0] * v1[2];
        o[2] = v0[0] * v1[1] - v0[1] * v1[0];
}

void vec3d_rotax(vec3d_t o, double theta, const vec3d_t ax, const vec3d_t x)
{
	if (1E-10 > fabs(theta)) {

		vec3d_copy(o, x);

	} else {

		double cp[3], cpp[3], cppp[3], tmp[3];

		vec3d_crossproduct(cp, ax, x);
		vec3d_smul(cpp, cp, cos(theta));
		vec3d_crossproduct(cppp, cpp, ax);
		vec3d_saxpy(tmp, cp, sin(theta), cppp);
		vec3d_saxpy(o, ax, vec3d_sdot(ax, x), tmp);
        }
}

void vec3d_copy(vec3d_t o, const vec3d_t x)
{
	for (int i = 0; i < 3; i++)
		o[i] = x[i];
}

void vec3d_rot(vec3d_t dst, const vec3d_t src1, const vec3d_t src2)
{
	vec3d_t tmp;
	tmp[0] = src1[1] * src2[2] - src1[2] * src2[1];
	tmp[1] = src1[2] * src2[0] - src1[0] * src2[2];
	tmp[2] = src1[0] * src2[1] - src1[1] * src2[0];
	vec3d_copy(dst, tmp);
}