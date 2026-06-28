/* Copyright 2026. Department of Radiology. Boston Children's Hospital.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"



#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


static const char help_str[] = "Sorts an array by real part and returns permutation to be used with bin.";


int main_sort(int argc, char* argv[argc])
{
	const char* in_file = NULL;
	const char* out_file = NULL;


	struct arg_s args[] = {

		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	const struct opt_s opts[] = {};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long dims[DIMS];

	complex float* in_data = load_cfl(in_file, DIMS, dims);

	if (1 != bitcount(md_nontriv_dims(DIMS, dims)))
		error("Only 1D arrays can be sorted!\n");

	long N = md_calc_size(DIMS,  dims);
	int (*ord)[N] = xmalloc(sizeof(*ord));

	for (int i = 0; i < N; i++)
		(*ord)[i] = i;

	NESTED(int, cmp_float, (int a, int b))
	{
		float da = crealf(in_data[a]);
		float db = crealf(in_data[b]);

		return (da > db) - (da < db);
	};

	quicksort(N, (*ord), cmp_float);

	complex float* out_data = create_cfl(out_file, DIMS, dims);

	for (int i = 0; i < N; i++)
		out_data[i] = (*ord)[i];

	xfree(*ord);

	unmap_cfl(DIMS, dims, out_data);
	unmap_cfl(DIMS, dims, in_data);

	return 0;
}

