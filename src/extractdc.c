/* Copyright 2026. Department of Radiology. Boston Children's Hospital.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <stdbool.h>
#include <complex.h>

#include "num/multind.h"
#include "num/init.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/filter.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/opts.h"

#include "noncart/radial.h"


static const char help_str[] = "Extract DC component from radial k-space data correcting for shifted or asymmetric trajectories.";


int main_extractdc(int argc, char* argv[argc])
{
	const char* traj_file = NULL;
	const char* ksp_file = NULL;
	const char* dc_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &traj_file, "traj"),
		ARG_INFILE(true, &ksp_file, "kspace"),
		ARG_OUTFILE(true, &dc_file, "dc"),
	};

	const struct opt_s opts[] = { };

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	bart_dim_t ksp_dims[DIMS];
	complex float* ksp = load_cfl(ksp_file, DIMS, ksp_dims);

	bart_dim_t trj_dims[DIMS];
	complex float* trj = load_cfl(traj_file, DIMS, trj_dims);

	if (!md_check_compat(DIMS, MD_BIT(0) | ~md_nontriv_dims(DIMS, trj_dims), trj_dims, ksp_dims))
		error("k-Space and trajectory are inconsistent!\n");

	bart_dim_t shift_dims[DIMS];
	md_select_dims(DIMS, ~(MD_BIT(0) | MD_BIT(1)), shift_dims, trj_dims);

	complex float* shift = md_alloc(DIMS, shift_dims, CFL_SIZE);

	traj_radial_dcshifts(DIMS, shift_dims, shift, trj_dims, trj);
	md_zsmul(DIMS, shift_dims, shift, shift, 1. / traj_radial_deltak(DIMS, trj_dims, trj));
	
	unmap_cfl(DIMS, trj_dims, trj);

	fftc(DIMS, ksp_dims, MD_BIT(1), ksp, ksp);

	bart_dim_t linphs_dims[DIMS];
	md_select_dims(DIMS, MD_BIT(1), linphs_dims, ksp_dims);

	complex float* linphs = md_alloc(DIMS, linphs_dims, CFL_SIZE);

	bart_stride_t linphs_strs[DIMS];
	md_calc_strides(DIMS, linphs_strs, linphs_dims, CFL_SIZE);

	bart_stride_t shift_strs[DIMS];
	md_calc_strides(DIMS, shift_strs, shift_dims, CFL_SIZE);

	bart_stride_t ksp_strs[DIMS];
	md_calc_strides(DIMS, ksp_strs, ksp_dims, CFL_SIZE);

	bart_dim_t red_dims[DIMS];
	md_select_dims(DIMS, MD_BIT(1) | COIL_FLAG, red_dims, ksp_dims);

	bart_dim_t pos[DIMS] = { };

	do {
		float tshift = -crealf(MD_ACCESS(DIMS, shift_strs, pos, shift));

		linear_phase(1, linphs_dims + 1, &tshift, linphs);
		
		md_zmul2(DIMS, red_dims, ksp_strs, MD_ACCESS_PTR(DIMS, ksp_strs, pos, ksp), ksp_strs, MD_ACCESS_PTR(DIMS, ksp_strs, pos, ksp), linphs_strs, linphs);

	} while (md_next(DIMS, ksp_dims, ~(MD_BIT(1) | COIL_FLAG), pos));

	md_free(linphs);

	bart_dim_t dc_dims[DIMS];
	md_select_dims(DIMS, ~(MD_BIT(0) | MD_BIT(1)), dc_dims, ksp_dims);
	
	complex float* dc = create_cfl(dc_file, DIMS, dc_dims);
	
	md_zavg(DIMS, ksp_dims, MD_BIT(1), dc, ksp);

	unmap_cfl(DIMS, dc_dims, dc);
	unmap_cfl(DIMS, ksp_dims, ksp);
	
	md_free(shift);

	return 0;
}

