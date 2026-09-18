/* Copyright 2025. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2025-2026 Martin Heide
 */

#include <complex.h>

#include "num/multind.h"
#include "num/init.h"
#include "num/flpmath.h"

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "simu/sens.h"
#include "simu/grid.h"
#include "simu/shape.h"

static const char help_str[] = "Compute coil sensitivitity maps in x-space or k-space.";

int main_coils(int argc, char* argv[argc])
{
	const char* out_file = NULL;
	const char* out_grid_file = NULL;
	const char* traj_file = NULL;
	const char* straj_file = NULL;
	const char* scoeff_file = NULL;

	struct grid_opts gopts = grid_opts_defaults;
	struct coil_opts copts = coil_opts_defaults;

	long ncoils = 0;

	bool legacy_fov = false;

	struct arg_s args[] = {

		ARG_OUTFILE(true, &out_file, "sens"),
		ARG_OUTFILE(false, &out_grid_file, "grid/traj"),
	};

	// FIXME: ulong overflow for coil # 63 in OPT_ULONG
	const struct opt_s opts[] = {

		OPT_SET('k', &copts.kspace, "Compute for k-space."),
		OPT_SET('L', &legacy_fov, "(Legacy: Coil Coefficients for FOV [-1,1].)"),
		OPT_LONG('n', &ncoils, "", "Select first n coil channels."),
		OPT_ULONG('b', &copts.flags, "", "Bitmask for selecting subset of channels.\n"),
		OPT_INFILE('t', &traj_file, "grid", "points on which sensitivity maps should be evaluated.\n"),
		OPTL_SELECT(0, "H2D8C", enum coil_type, &copts.ctype, HEAD_2D_8CH, "2D head coil 8 channels (default)."),
		OPTL_SELECT(0, "H3D64C", enum coil_type, &copts.ctype, HEAD_3D_64CH, "3D head coil 64 channels."),
		OPTL_INFILE(0, "scoord", &straj_file, "file", "k-space coordinates of coefficients of trig. polynomial model for senisitivities."),
		OPTL_INFILE(0, "scoeff", &scoeff_file, "file", "coefficients of trig. polynomial model for sensitivities."),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	if (legacy_fov) {

		gopts.b0[0] = 1;
		gopts.b1[1] = 1;
	}

	if ((NULL != straj_file && NULL == scoeff_file) || (NULL == straj_file && NULL != scoeff_file))
		error("please provide coefficients *and* k-space coordinates of trigonometric polynomial together.");

	if (0 < ncoils && 64 > ncoils)
		copts.flags = MD_BIT(ncoils) - 1;

	long strajdims[DIMS];
	float* straj = NULL;
	long scoeffdims[DIMS];
	complex float* scoeff = NULL;

	if (NULL != straj_file && NULL != scoeff_file) {

		copts.ctype = TRIG_POLY_EXT;

		complex float* strajt = load_cfl(straj_file, DIMS, strajdims);
		straj = md_alloc(DIMS, strajdims, FL_SIZE);
		md_real(DIMS, strajdims, straj, strajt);
		unmap_cfl(DIMS, strajdims, strajt);

		complex float* scoeff_ = load_cfl(scoeff_file, DIMS, scoeffdims);
		scoeff = md_alloc(DIMS, scoeffdims, CFL_SIZE);
		md_copy(DIMS, scoeffdims, scoeff, scoeff_, CFL_SIZE);
		unmap_cfl(DIMS, scoeffdims, scoeff_);
	}

	long gdims[DIMS];
	float* grid = NULL;

	cnstr_coils(DIMS, &copts, legacy_fov, strajdims, straj, scoeffdims, scoeff);
	gopts.kspace = copts.kspace;

	if (NULL != traj_file) {

		complex float* cgrid = load_cfl(traj_file, DIMS, gdims);

		grid = md_alloc(DIMS, gdims, FL_SIZE);
		md_real(DIMS, gdims, grid, cgrid);

		unmap_cfl(DIMS, gdims, cgrid);

	} else {

		grid = create_senstraj(DIMS, gdims, &gopts, &copts);
	}

	long odims[DIMS];
	complex double* sens = sample_coils(DIMS, odims, gdims, grid, &copts);
	complex float* optr = create_cfl(out_file, DIMS, odims);

	if (NULL != out_grid_file) {

		complex float* ogr = create_cfl(out_grid_file, DIMS, gdims);
		md_zcmpl_real(DIMS, gdims, ogr, grid);
		unmap_cfl(DIMS, gdims, ogr);
	}

	md_zdouble2float(DIMS, odims, optr, sens);

	md_free(grid);
	copts.dstr(&copts);
	md_free(sens);
	unmap_cfl(DIMS, odims, optr);

	return 0;
}

