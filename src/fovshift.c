/* Copyright 2022-2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <stdbool.h>
#include <complex.h>
#include <stdlib.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/filter.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/opts.h"
#include "misc/misc.h"
#include "misc/mri.h"



static complex float* noncart_shift(bart_dim_t odims[DIMS], const bart_dim_t sdims[DIMS], const complex float* shift, const bart_dim_t tdims[DIMS], const complex float* tdata)
{
	md_max_dims(DIMS, ~UINT64_C(0), odims, tdims, sdims);
	md_select_dims(DIMS, ~MD_BIT(0), odims, odims);


	complex float* odata = md_alloc(DIMS, odims, CFL_SIZE);

	md_ztenmul(DIMS, odims, odata, tdims, tdata, sdims, shift);

	md_zsmul(DIMS, odims, odata, odata, +2.i * M_PI);
	md_zexp(DIMS, odims, odata, odata);

	return odata;
}


static const char help_str[] = "Shifts FOV.";


int main_fovshift(int argc, char* argv[argc])
{
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	float shift[3] = { 0., 0., 0. };
	const char* shift_file = NULL;
	const char* traj_file = NULL;
	bool pixel = false;

	const struct opt_s opts[] = {

		OPT_INFILE('t', &traj_file, "file", "k-space trajectory"),
		OPT_INFILE('S', &shift_file, "file", "FOV shift"),
		OPT_FLVEC3('s', &shift, "X:Y:Z", "FOV shift"),
		OPT_SET('p', &pixel, "interpret FOV shift in units of pixel instead of units of FoV")
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	bart_dim_t sdims[DIMS] = { 3, [ 1 ... DIMS - 1 ] = 1 };

	complex float* cshift;

	if (NULL == shift_file) {

		cshift = anon_cfl(NULL, DIMS, sdims);

		for (int i = 0; i < 3; i++)
			cshift[i] = shift[i];

	} else {

		cshift = load_cfl(shift_file, DIMS, sdims);
	}

	bart_dim_t idims[DIMS];
	complex float* idata = load_cfl(in_file, DIMS, idims);

	if (pixel) {

		if (NULL != traj_file)
			error("Shift in units of pixel only possible for Cartesian k-space!\n");

		complex float scale[3];

		for (int i = 0; i < 3; i++)
			scale[i] = 1. / idims[i];

		bart_stride_t scl_strs[DIMS] = { CFL_SIZE, [ 1 ... DIMS - 1 ] = 0 };

		md_zmul2(DIMS, sdims, MD_STRIDES(DIMS, sdims, CFL_SIZE), cshift,
				MD_STRIDES(DIMS, sdims, CFL_SIZE), cshift,
				scl_strs, scale);
	}

	bart_dim_t pdims[DIMS];
	complex float* phase;

	if (NULL != traj_file) {

		bart_dim_t tdims[DIMS];
		complex float* tdata = load_cfl(traj_file, DIMS, tdims);

		phase = noncart_shift(pdims, sdims, cshift, tdims, tdata);

		unmap_cfl(DIMS, tdims, tdata);

	} else {

		if (1 != md_nontriv_dims(DIMS, sdims))
			error("Cartesian fovshift only supports the first three dimensions.\nUse bart looping for higher dimensions.\n");

		md_select_dims(DIMS, FFT_FLAGS, pdims, idims);

		phase = md_alloc(DIMS, pdims, CFL_SIZE);

		for (int i = 0; i < 3; i++)
			shift[i] = (float)idims[i] * cshift[i];

		linear_phase(3, pdims, shift, phase);
	}

	complex float* odata = create_cfl(out_file, DIMS, idims);

	md_zmul2(DIMS, idims, MD_STRIDES(DIMS, idims, CFL_SIZE), odata,
			MD_STRIDES(DIMS, idims, CFL_SIZE), idata,
			MD_STRIDES(DIMS, pdims, CFL_SIZE), phase);

	md_free(phase);

	unmap_cfl(DIMS, idims, idata);
	unmap_cfl(DIMS, idims, odata);
	unmap_cfl(DIMS, sdims, cshift);

	return 0;
}

