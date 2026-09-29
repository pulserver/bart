/* Copyright 2026. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2026 Tobias Paul Trimmal
 *
 * References:
 *
 * Smith SM. Fast robust automated brain extraction.
 * Hum Brain Mapp. 2002;17(3):143-155.
 */

#include <assert.h>
#include <complex.h>
#include <stddef.h>
#include <stdio.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/init.h"
#include "num/morph.h"

#include "linops/linop.h"
#include "linops/grad.h"
#include "linops/sum.h"
#include "linops/fmac.h"
#include "linops/lintest.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"
#include "misc/mri.h"

#include "stl/misc.h"
#include "stl/models.h"

#include "qsm/bet.h"

#include "main.h"

#ifndef DIMS
#define DIMS 16
#endif

#define MAX_VERTS 10250
#define MAX_TRIS 30000

static const char help_str[] = "Create 3D mask from 3D brain image using BET algorithm.";
	
int main_bet(int argc, char* argv[argc])
{
	num_init();
	const char* in_file = NULL;
	const char* out_file = NULL;
	int sub = 4;
	int erode = 0;
	float bt = 0.5f;
	float resolution[3] = { 1., 1., 1.};
	int n_iter = 1000;
	const char* stl_ofile = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &in_file, "brain image (complex or magnitude)"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	bool mesh_to_mask_winding = false;

	const struct opt_s opts[] = {

		OPT_INT('s', &sub, "sub", "number of subdivisons for initialization of the polyedron - default 4 (needs to be in range of 2 - 5)"),
		OPT_INT('e', &erode, "erode", "number or erosion steps - default 0 (needs to be in range of 0 - 10)"),
		OPT_FLOAT('b', &bt, "bt", "fractional intensity threshold; between 0 and 1 - default 0.5"),
		OPTL_FLVEC3(0, "res", &resolution, "res_x:res_y:res_z", "resolution/voxel size [mm] - default 1.0:1.0:1.0"),
		OPT_INT('i', &n_iter, "n_iter", "number of iterations for the BET algorithm - default 1000"),
		OPTL_OUTFILE(0, "stl_out", &stl_ofile, "stl_outfile", "export stl mesh to file"),
		OPT_SET('w', &mesh_to_mask_winding, "use winding number method for mesh to mask conversion (default: false)"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	if ((sub < 2) || (sub > 5))
		error("sub must be in range [2, 5].\n");
	
	if ((bt < 0.f) || (bt > 1.f))
		error("bt must be in range [0, 1].\n");

	if ((erode < 0) || (erode > 10))
		error("erode must be in range [0, 10].\n");

	long in_dims[DIMS];
	complex float* in_data = load_cfl(in_file, DIMS, in_dims);

	long out_dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	out_dims[0] = in_dims[0];
	out_dims[1] = in_dims[1];
	out_dims[2] = in_dims[2];
	
	complex float* out_data = create_cfl(out_file, DIMS, out_dims);
	
	complex float* magnitude = md_alloc_sameplace(DIMS, in_dims, CFL_SIZE, in_data);
	float* interm = md_alloc_sameplace(DIMS, in_dims, FL_SIZE, in_data);
	float* out = md_alloc_sameplace(DIMS, in_dims, FL_SIZE, in_data);

	md_zabs(DIMS, in_dims, magnitude, in_data);
	md_real(DIMS, in_dims, interm, magnitude);

	float t = 0.;
	float t98 = 0.;
	float t2 = 0.;

	bet_threshold(DIMS, in_dims, interm, out, &t, &t98, &t2);

	float COG[3] = { 0., 0., 0. };
	float R = 0.;

	compute_cog(DIMS, in_dims, out, resolution, &t,
		&t98, COG, &R);

	float tm = compute_tm(DIMS, in_dims, out, resolution, COG, R);

	long dims[3];
	long o_dims[3];

        double* model = stl_internal_icosahedron(dims);
	double* out_model = stl_multiple_subdivide_model(sub,
		o_dims, dims, model);

	double shift[3] = { COG[0], COG[1], COG[2] };
	double scale[3] = { R / 2., R / 2., R / 2. };

	stl_scale_model(o_dims, out_model, scale);
	stl_shift_model(o_dims, out_model, shift);

	double verts[MAX_VERTS][3];
	int tris[MAX_TRIS][3];
	struct neighbors neigh[MAX_VERTS];
	int num_verts;
	int num_tris;

	stl_build_neighbors(MAX_VERTS, o_dims, out_model, neigh, &num_verts,
		verts, &num_tris, tris);

	run_bet(DIMS, in_dims, &verts[0][0], num_verts, neigh, out,
		resolution, COG, t2, t, tm, bt, n_iter);

	stl_update_vertices(MAX_VERTS, o_dims, out_model, verts, tris);
	stl_compute_normals(o_dims, out_model);

	float* mask = md_alloc_sameplace(DIMS, in_dims, FL_SIZE, in_data);

	md_clear(DIMS, in_dims, mask, FL_SIZE);
	
	if (mesh_to_mask_winding)
		mesh_to_mask_winding_number(DIMS, in_dims, mask, resolution, verts, tris, num_tris);
	else
		mesh_to_mask_slicewise(DIMS, in_dims, mask, resolution, verts, tris, num_tris);

	if (erode > 0) {

		complex float* mask_cmplx = md_alloc_sameplace(DIMS, in_dims, CFL_SIZE, in_data);

		md_zcmpl_real(DIMS, in_dims, mask_cmplx, mask);

		long er_mask_dims[DIMS];
		unsigned long flags = 0;

		flags = MD_SET(flags, 0);
		flags = MD_SET(flags, 1);
		flags = MD_SET(flags, 2);

		complex float* er_mask = md_structuring_element_ball(DIMS, er_mask_dims, erode, flags, in_data);
		complex float* mask_eroded = md_alloc(DIMS, in_dims, CFL_SIZE);

		md_erosion(DIMS, er_mask_dims, er_mask, in_dims, mask_eroded, mask_cmplx, CONV_TRUNCATED);
		md_real(DIMS, in_dims, mask, mask_eroded);

		md_free(mask_eroded);
		md_free(mask_cmplx);
		md_free(er_mask);
	}

	md_zcmpl_real(DIMS, out_dims, out_data, mask);

	if (NULL != stl_ofile) {

		FILE* fp = fopen(stl_ofile, "wb");
		stl_write(fp, o_dims, out_model, false);
		fclose(fp);
	}

	md_free(interm);
	md_free(magnitude);
	md_free(mask);

	unmap_cfl(DIMS, in_dims, in_data);
	unmap_cfl(DIMS, out_dims, out_data);

	return 0;
}
