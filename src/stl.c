/* Copyright 2024-2025. Uecker Lab. University Medical Center Göttingen
 * Copyright 2025-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2024 - 2026 Martin Heide
 */

#include <complex.h>

#include "num/multind.h"
#include "num/init.h"


#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "stl/models.h"
#include "stl/misc.h"

static const char help_str[] = "Read and write stl files with '.stl' or cfl fileformat.";

int main_stl(int argc, char* argv[argc])
{
	const char* out_file = NULL;
	const char* in_file = NULL;

        bool stat = false;
        bool ascii = true;
        bool no_nc = false;
	bool centerfov = false;
        float scale = 0.;
        float shift[3] = { 0., 0., 0. };
	float transform[7] = { 1., 0., 0., 0., 0., 0., 0. };
        enum stl_itype stl_choice = STL_NONE;

        struct arg_s args[] = {

		ARG_OUTFILE(false, &out_file, "output"),
	};

        struct opt_s model_opts[] = {

		OPTL_SELECT(0, "TET", enum stl_itype, &stl_choice, STL_TETRAHEDRON, "Tetrahedron."),
		OPTL_SELECT(0, "HEX", enum stl_itype, &stl_choice, STL_HEXAHEDRON, "Hexahedron (= Cube)."),
		OPTL_SELECT(0, "ICO", enum stl_itype, &stl_choice, STL_ICOSAHEDRON, "Icosahedron."),
        };

	const struct opt_s opts[] = {

		OPTL_INFILE(0, "input", &in_file, "", "Path to input file (.stl or cfl file format)."),
                OPTL_SUBOPT2(0, "model", "<tag> ", "Generic geometric structures are available.", "Internal stl model (help: bart stl --model h).\n", ARRAY_SIZE(model_opts), model_opts),
		OPTL_SET(0, "cfov", &centerfov, "Scale and move model to center of FOV."),
		OPT_FLOAT('s', &scale, "scale", "Multiplicate all coordinates of model with a scale factor."),
                OPT_FLVEC3('m', &shift, "move", "Move model by vector.\n"),
		OPTL_FLVEC7(0, "transform", &transform, "transform", "scale:move_x:move_y:move_z:rot_xy[deg]:rot_xz[deg]:rot_yz[deg]. Rotates the *centered* model.\n"),
		OPTL_SET(0, "stat", &stat, "Show statistics of model."),
		OPTL_CLEAR(0, "binary", &ascii, "Output STL files in binary format."),
		OPTL_SET(0, "no-nc", &no_nc, "(Don't recompute normal vectors with double precision.)"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

        if (NULL != in_file && STL_NONE != stl_choice)
                error("Please provide either input from filesystem or use internal stl.\n");

        if (NULL == in_file && STL_NONE == stl_choice)
                stl_choice = STL_TETRAHEDRON;

        bart_dim_t dims[DIMS];
        // for build analyzer
        double* model = NULL;

        if (NULL != in_file) {

                if (stl_fileextension(in_file)) {

			FILE *fp = fopen(in_file, "r");

			if (!fp)
				error("opening file %s\n", in_file);

			model = stl_read(fp, dims);

			fclose(fp);

                } else {

                        complex float* cmodel = load_cfl(in_file, DIMS, dims);

                        model = stl_cfl2d(dims, cmodel);

                        unmap_cfl(DIMS, dims, cmodel);
                }
        }

        if (STL_TETRAHEDRON == stl_choice)
                model = stl_internal_tetrahedron(dims);

        if (STL_HEXAHEDRON == stl_choice)
                model = stl_internal_hexahedron(dims);

        if (STL_ICOSAHEDRON == stl_choice)
                model = stl_internal_icosahedron(dims);

	if (!no_nc)
		stl_compute_normals(dims, model);

	bool btrnsf = (1. != transform[0] || 0. != transform[1] || 0. != transform[2]
			|| 0. != transform[3] || 0.!= transform[4] || 0. != transform[5]
			|| 0. != transform[6]);

	bool bscale = (0. != scale);
        bool bmove = (0. != shift[0] || 0. != shift[1] || 0. != shift[2]);

	if (bscale && btrnsf)
		error("use either scale or transform option.");

	if (bmove && btrnsf)
		error("use either move or transform option.");

        double dshift[3] = { shift[0], shift[1], shift[2] };
        double sc[3] = { scale, scale, scale };
	double rot[3] = { transform[4], transform[5], transform[6] };

	if (centerfov && (btrnsf || bscale || bmove))
		error("dont use fovnorm and scale, move or transform together.");

	if (btrnsf) {

		sc[0] = transform[0];
		sc[1] = transform[0];
		sc[2] = transform[0];

		dshift[0] = transform[1];
		dshift[1] = transform[2];
		dshift[2] = transform[3];
	}

	if (btrnsf || bmove)
                stl_shift_model(dims, model, dshift);

	if (btrnsf)
		stl_rot_model(dims, model, rot);

	if (btrnsf || bscale)
                stl_scale_model(dims, model, sc);

	if (centerfov)
		stl_center_fov(dims, model, 0.9);

        if (stat) {

                stl_stats(dims, model);


		struct triangle_stack* ts = stl_preprocess_model(dims, model);
		struct triangle* t = ts->tri;

		double smv = 0.;
		double vmv = 0.;

		for (int i = 0; i < ts->N; i++) {

			smv += t[i].sur;
			vmv += t[i].svol;
		}

		debug_printf(DP_INFO, "surface area: %f\n", smv);

		debug_printf(DP_INFO, "volume: %f\n", vmv);

		md_free(ts);
	}

	if (NULL != out_file) {

		if (stl_fileextension(out_file)) {

			FILE *fp = fopen(out_file, "w");

			if (!fp)
				error("opening file %s\n", out_file);

			stl_write(fp, dims, model, ascii);

			fclose(fp);

		} else {

			complex float* cmodel = create_cfl(out_file, 3, dims);

			stl_d2cfl(dims, cmodel, model);

			unmap_cfl(3, dims, cmodel);
		}
	}

	md_free(model);

	return 0;
}

