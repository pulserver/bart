/* Copyright 2026. Department of Radiology. Boston Children's Hospital.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * Moritz Blumenthal
 */

#include <complex.h>

#include "num/flpmath.h"
#include "num/multind.h"

#include "laplace.h"

void md_laplace_fd_scaled(int N, const long dims[N], unsigned long flags, const float scale[N], complex float* out, const complex float* in)
{
	complex float* tmp = md_alloc_sameplace(N, dims, CFL_SIZE, in);
	md_copy(N, dims, tmp, in, CFL_SIZE);

	md_clear(N, dims, out, CFL_SIZE);

	complex float* tmp1 = md_alloc_sameplace(N, dims, CFL_SIZE, in);

	for (int i = 0; i < N; i++) {

		if (!MD_IS_SET(flags, i))
			continue;

		md_zaxpy(N, dims, out, -2. * scale[i], tmp);

		long pos[N];
		md_set_dims(N, pos, 0);

		md_circ_shift(N, dims, (pos[i] = 1, pos), tmp1, tmp, CFL_SIZE);
		md_zaxpy(N, dims, out, scale[i], tmp1);

		md_circ_shift(N, dims, (pos[i] = -1, pos), tmp1, tmp, CFL_SIZE);
		md_zaxpy(N, dims, out, scale[i], tmp1);
	}

	md_free(tmp);
	md_free(tmp1);
}

void md_laplace_fd(int N, const long dims[N], unsigned long flags, complex float* out, const complex float* in)
{
	float resolution[N];
	for (int i = 0; i < N; i++)
		resolution[i] = 1.;

	md_laplace_fd_scaled(N, dims, flags, resolution, out, in);
}


void md_laplace_fd_wrapped_phase_scaled(int N, const long dims[N], unsigned long flags, const float scale[N], complex float* out, const complex float* in)
{
	complex float* tmp = md_alloc_sameplace(N, dims, CFL_SIZE, in);
	md_zexpj(N, dims, tmp, in);

	md_clear(N, dims, out, CFL_SIZE);

	complex float* tmp1 = md_alloc_sameplace(N, dims, CFL_SIZE, in);

	for (int i = 0; i < N; i++) {

		if (!MD_IS_SET(flags, i))
			continue;

		long pos[N];
		md_set_dims(N, pos, 0);

		md_circ_shift(N, dims, (pos[i] = 1, pos), tmp1, tmp, CFL_SIZE);
		md_zmulc(N, dims, tmp1, tmp1, tmp);
		md_zarg(N, dims, tmp1, tmp1);
		md_zaxpy(N, dims, out, scale[i], tmp1);

		md_circ_shift(N, dims, (pos[i] = -1, pos), tmp1, tmp, CFL_SIZE);
		md_zmulc(N, dims, tmp1, tmp1, tmp);
		md_zarg(N, dims, tmp1, tmp1);
		md_zaxpy(N, dims, out, scale[i], tmp1);
	}

	md_free(tmp);
	md_free(tmp1);
}

void md_laplace_fd_wrapped_phase(int N, const long dims[N], unsigned long flags, complex float* out, const complex float* in)
{
	float scale[N];
	for (int i = 0; i < N; i++)
		scale[i] = 1.;

	md_laplace_fd_wrapped_phase_scaled(N, dims, flags, scale, out, in);
}



//use Lap(phi) = Im(exp(-i phi) * Lap(exp(i phi)))
void md_laplace_fd_wrapped_phase_exp_scaled(int N, const long dims[N], unsigned long flags, const float scale[N], complex float* out, const complex float* in)
{
	complex float* tmp = md_alloc_sameplace(N, dims, CFL_SIZE, in);
	md_zexpj(N, dims, tmp, in);

	md_laplace_fd_scaled(N, dims, flags, scale, out, tmp);
	md_zmulc(N, dims, out, out, tmp);
	md_zsmul(N, dims, out, out, -1.i);
	md_zreal(N, dims, out, out);

	md_free(tmp);
}

//use Lap(phi) = Im(exp(-i phi) * Lap(exp(i phi)))
void md_laplace_fd_wrapped_phase_exp(int N, const long dims[N], unsigned long flags, complex float* out, const complex float* in)
{
	float scale[N];
	for (int i = 0; i < N; i++)
		scale[i] = 1.;

	md_laplace_fd_wrapped_phase_exp_scaled(N, dims, flags, scale, out, in);
}

