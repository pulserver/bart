/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * Copyright 2026. Department of Radiology. Boston Children's Hospital.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <assert.h>
#include <math.h>

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/filter.h"
#include "num/fft.h"
#include "num/laplace.h"

#include "misc/opts.h"
#include "misc/misc.h"
#include "misc/mmio.h"

#define DIMS 16

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static void rounded_div(int D, const bart_dim_t dims[D], float bound, complex float* out, const complex float* in)
{
	bart_dim_t size = md_calc_size(D, dims);

#pragma omp parallel for
	for (bart_dim_t i = 0; i < size; i++) {

		float d = crealf(in[i]) / bound;
		out[i] = (d > 1.) ? - ceilf(d) : (d < -1.) ? - floorf(d) : 0.;
	}
}

static void unwrap(int D, const bart_dim_t dims[D], int d, float bounds,
	complex float* optr, const complex float* iptr)
{
	md_zfdiff0(D, dims, d, optr, iptr);

	rounded_div(D, dims, bounds, optr, optr);

	md_zcumsum(D, dims, MD_BIT(d), optr, optr);
	md_zsmul(D, dims, optr, optr, bounds);
	md_zadd(D, dims, optr, optr, iptr);
}

static void unwrap_lap(int D, const long dims[D], unsigned long flags, float bounds, complex float* optr, const complex float* iptr)
{
	md_zsmul(D, dims, optr, iptr, M_PI / bounds);

	md_laplace_fd_wrapped_phase(D, dims, flags, optr, optr);

	long fft_dims[D];
	md_select_dims(D, flags, fft_dims, dims);

	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	long fft_strs[D];
	md_calc_strides(D, fft_strs, fft_dims, CFL_SIZE);

	complex float* kernel = md_alloc_sameplace(D, fft_dims, CFL_SIZE, iptr);
	klaplace_fd_uncentered(D, dims, kernel);

	complex float* tmp = md_alloc_sameplace(D, dims, CFL_SIZE, iptr);
	md_zfill(D, dims, tmp, 1.);
	md_zdiv(D, fft_dims, kernel, tmp, kernel);
	md_free(tmp);

	fft(D, dims, flags, optr, optr);
	md_zmul2(D, dims, strs, optr, strs, optr, fft_strs, kernel);

	md_free(kernel);

	ifft(D, dims, flags, optr, optr);

	md_zsmul(D, dims, optr, optr, bounds / (M_PI * md_calc_size(D, fft_dims)));
}


static const char help_str[] = "Unwrap along selected dimensions.";

enum MODE { MODE_CUMSUM, MODE_LAP };

int main_unwrap(int argc, char* argv[argc])
{
	unsigned long flags = 0;
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_ULONG(true, &flags, "flags"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	float bounds = M_PI;
	enum MODE mode = MODE_CUMSUM;

	const struct opt_s opts[] = {

		OPT_FLOAT('b', &bounds, "bounds", "bounds (default: PI)"),
		OPT_SELECT('l', enum MODE, &mode, MODE_LAP, "select Laplacian-based unwrapping"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	bart_dim_t in_dims[DIMS];
	bart_dim_t out_dims[DIMS];

	complex float* in_data = load_cfl(in_file, DIMS, in_dims);

	md_copy_dims(DIMS, out_dims, in_dims);

	complex float* out_data = NULL;
	out_data = create_cfl(out_file, DIMS, out_dims);

	switch (mode) {

	case MODE_CUMSUM:

		if (1 != bitcount(flags))
			error("Cumulative sum can only be applied along one dimension, but multiple dimensions were selected.\n");

		unwrap(DIMS, in_dims, md_min_idx(flags), bounds, out_data, in_data);
		break;

	case MODE_LAP:

		unwrap_lap(DIMS, in_dims, flags, bounds, out_data, in_data);
		break;
	}

	unmap_cfl(DIMS, in_dims, in_data);
	unmap_cfl(DIMS, out_dims, out_data);

	return 0;
}

