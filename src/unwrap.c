/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <assert.h>
#include <math.h>

#include "num/flpmath.h"
#include "num/multind.h"

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


static const char help_str[] = "Unwrap along selected dimensions.";


int main_unwrap(int argc, char* argv[argc])
{
	int dim = -1;
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INT(true, &dim, "dim"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	float bounds = M_PI;

	const struct opt_s opts[] = {

		OPT_FLOAT('b', &bounds, "bounds", "bounds (default: PI)"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	bart_dim_t in_dims[DIMS];
	bart_dim_t out_dims[DIMS];

	complex float* in_data = load_cfl(in_file, DIMS, in_dims);

	md_copy_dims(DIMS, out_dims, in_dims);

	complex float* out_data = NULL;
	out_data = create_cfl(out_file, DIMS, out_dims);
	
	unwrap(DIMS, in_dims, dim, bounds, out_data, in_data);

	unmap_cfl(DIMS, in_dims, in_data);
	unmap_cfl(DIMS, out_dims, out_data);

	return 0;
}

