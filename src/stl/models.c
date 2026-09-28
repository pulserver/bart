/* Copyright 2025. University Medical Center Göttingen.
 * Copyright 2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2024 Martin Heide
 * 2026 Tobias Paul Trimmal
 */

#include <math.h>

#include "misc/debug.h"
#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/linalg.h"
#include "num/vec3.h"

#include "stl/misc.h"

#include "models.h"
#define EDGE_TABLE_SIZE 262144
#define PHI 1.6180339887498948482

// direction of normal is outward.
// we don't store the normal vector bc it will be computed in the stl_internal_* function.
// vertices listed counterclockwise when looked on from outside (right-hand rule).
//
// convert vertices in array to contiguous fortran data format and compute normal vector
static double* stl_internal_model(bart_dim_t dims[3], const double (*arr)[][3][3])
{
        if (3 != dims[0] || 4 != dims[1])
                error("dimensions do not match dimensions for stl format");

        bart_stride_t strs[3];
        md_calc_strides(3, strs, dims, DL_SIZE);

        double* model = md_alloc(3, dims, DL_SIZE);

        bart_dim_t pos[3] = { };

        do {
                // Normal vector is computed afterwards.
                if (3 > pos[1])
                        MD_ACCESS(3, strs, pos, model) = (*arr)[pos[2]][pos[1]][pos[0]];

        } while (md_next(3, dims, ~UINT64_C(0), pos));

        stl_compute_normals(dims, model);

        return model;
}

static const double stl_hexahedron[12][3][3] = {

        { { 0.45, 0.45, -0.45 }, { -0.45, 0.45, -0.45 }, { 0.45, 0.45, 0.45 } },
        { { 0.45, 0.45, 0.45 }, { -0.45, 0.45, -0.45 }, { -0.45, 0.45, 0.45 } },
        { { 0.45, -0.45, -0.45 }, { 0.45, -0.45, 0.45 }, { -0.45, -0.45, 0.45 } },
        { { 0.45, -0.45, -0.45 }, { -0.45, -0.45, 0.45 }, { -0.45, -0.45, -0.45 } },
        { { 0.45, 0.45, -0.45 }, { 0.45, 0.45, 0.45 }, { 0.45, -0.45, 0.45 } },
        { { 0.45, 0.45, -0.45 }, { 0.45, -0.45, 0.45 }, { 0.45, -0.45, -0.45 } },
        { { -0.45, 0.45, -0.45 }, { -0.45, -0.45, -0.45 }, { -0.45, 0.45, 0.45 } },
        { { -0.45, 0.45, 0.45 }, { -0.45, -0.45, -0.45 }, { -0.45, -0.45, 0.45 } },
        { { -0.45, 0.45, 0.45 }, { 0.45, -0.45, 0.45 }, { 0.45, 0.45, 0.45 } },
        { { -0.45, 0.45, 0.45 }, { -0.45, -0.45, 0.45 }, { 0.45, -0.45, 0.45 } },
        { { -0.45, -0.45, -0.45 }, { -0.45, 0.45, -0.45 }, { 0.45, 0.45, -0.45 } },
        { { -0.45, -0.45, -0.45 }, { 0.45, 0.45, -0.45 }, { 0.45, -0.45, -0.45 } },
};

double* stl_internal_hexahedron(bart_dim_t dims[3])
{
        dims[0] = 3;
        dims[1] = 4;
        dims[2] = 12;

        return stl_internal_model(dims, &stl_hexahedron);
}


static const double stl_tetrahedron[4][3][3] = {

        { { -0.45, 0.45, -0.45 }, { 0.45, 0.45, 0.45 }, { 0.45, -0.45, -0.45 } },
        { { -0.45, -0.45, 0.45 }, { 0.45, -0.45, -0.45 }, { 0.45, 0.45, 0.45 } },
        { { -0.45, -0.45, 0.45 }, { 0.45, 0.45, 0.45 }, { -0.45, 0.45, -0.45 } },
        { { -0.45, -0.45, 0.45 }, { -0.45, 0.45, -0.45 }, { 0.45, -0.45, -0.45 } },
};

double* stl_internal_tetrahedron(bart_dim_t dims[3])
{
        dims[0] = 3;
        dims[1] = 4;
        dims[2] = 4;

        return stl_internal_model(dims, &stl_tetrahedron);
}

static const double stl_icosahedron[20][3][3] = {

        { { -1, PHI, 0 }, { -PHI, 0, 1 }, { 0, 1, PHI } },
        { { -1, PHI, 0 }, { 0, 1, PHI }, { 1, PHI, 0 } },
        { { -1, PHI, 0 }, { 1, PHI, 0 }, { 0, 1, -PHI } },
        { { -1, PHI, 0 }, { 0, 1, -PHI }, { -PHI, 0, -1 } },
        { { -1, PHI, 0 }, { -PHI, 0, -1 }, { -PHI, 0, 1 } },
        { { 1, PHI, 0 }, { 0, 1, PHI }, { PHI, 0, 1 } },
        { { 0, 1, PHI }, { -PHI, 0, 1 }, { 0, -1, PHI } },
        { { -PHI, 0, 1 }, { -PHI, 0, -1 }, { -1, -PHI, 0 } },
        { { -PHI, 0, -1 }, { 0, 1, -PHI }, { 0, -1, -PHI } },
        { { 0, 1, -PHI }, { 1, PHI, 0 }, { PHI, 0, -1 } },
        { { 1, -PHI, 0 }, { PHI, 0, 1 }, { 0, -1, PHI } },
        { { 1, -PHI, 0 }, { 0, -1, PHI }, { -1, -PHI, 0 } },
        { { 1, -PHI, 0 }, { -1, -PHI, 0 }, { 0, -1, -PHI } },
        { { 1, -PHI, 0 }, { 0, -1, -PHI }, { PHI, 0, -1 } },
        { { 1, -PHI, 0 }, { PHI, 0, -1 }, { PHI, 0, 1 } },
        { { 0, -1, PHI }, { PHI, 0, 1 }, { 0, 1, PHI } },
        { { -1, -PHI, 0 }, { 0, -1, PHI }, { -PHI, 0, 1 } },
        { { 0, -1, -PHI }, { -1, -PHI, 0 }, { -PHI, 0, -1 } },
        { { PHI, 0, -1 }, { 0, -1, -PHI }, { 0, 1, -PHI } },
        { { PHI, 0, 1 }, { PHI, 0, -1 }, { 1, PHI, 0 } },
};

double* stl_internal_icosahedron(long dims[3])
{
        dims[0] = 3;
        dims[1] = 4;
        dims[2] = 20;

        double* model = stl_internal_model(dims, &stl_icosahedron);

        const double r = vec3d_norm(stl_icosahedron[0][0]);
	const double scale[3] = { 1. / r, 1. / r, 1. / r };

        stl_scale_model(dims, model, scale);

        return model;
}

struct stl_edge_entry {

	int a;
	int b;
	int mid;
	int used;
};

static inline unsigned stl_edge_hash(int a, int b)
{
	return ((unsigned)a * 73856093u) ^ ((unsigned)b * 19349663u);
}

static int stl_edge_midpoint(struct stl_edge_entry* table, long N, double verts[N][3], int* vert_count, int i, int j)
{
	if (i > j) {
		int t = i;
		i = j;
		j = t;
	}

	unsigned h = stl_edge_hash(i, j) & (EDGE_TABLE_SIZE - 1);

	while (table[h].used) {

		if (table[h].a == i && table[h].b == j)
			return table[h].mid;

		h = (h + 1) & (EDGE_TABLE_SIZE - 1);
	}

	double x = 0.5 * (verts[i][0] + verts[j][0]);
	double y = 0.5 * (verts[i][1] + verts[j][1]);
	double z = 0.5 * (verts[i][2] + verts[j][2]);

	double inv_r = 1. / sqrt(x*x + y*y + z*z);

	int idx = (*vert_count)++;

	verts[idx][0] = x*inv_r;
	verts[idx][1] = y*inv_r;
	verts[idx][2] = z*inv_r;

	table[h] = (struct stl_edge_entry){ i, j, idx, 1 };

	return idx;
}

double* stl_subdivide_model(long dims_out[3], const long dims_in[3], const double* model_in)
{
	assert(3 == dims_in[0]);
	assert(4 == dims_in[1]);

	long N = dims_in[2];

	double (*verts)[3 * N][3] = xmalloc(sizeof(*verts));
	int (*tris)[N][3] = xmalloc(sizeof(*tris));

	int nv = 0;
	int nt = 0;

	long strs_in[3];
	md_calc_strides(3, strs_in, dims_in, DL_SIZE);

	for (long i = 0; i < N; i++) {

		for (int v = 0; v < 3; v++) {

			for (int d = 0; d < 3; d++) {

				long pos[3] = { d, v, i };
				(*verts)[nv][d] = MD_ACCESS(3, strs_in, pos, model_in);
			}

			(*tris)[nt][v] = nv;
			nv++;
		}

		nt++;
	}

	struct stl_edge_entry* stl_edge_table =
		calloc(EDGE_TABLE_SIZE, sizeof(*stl_edge_table));

	double (*verts_out)[6 * N][3] = xmalloc(sizeof(*verts_out));
	int (*tris_out)[4 * N][3] = xmalloc(sizeof(*tris_out));
	memcpy(*verts_out, *verts, (unsigned long)nv * sizeof((*verts_out)[0]));

	int vert_count = nv;
	int tri_count = 0;

	for (int t = 0; t < nt; t++) {

		int i0 = (*tris)[t][0];
		int i1 = (*tris)[t][1];
		int i2 = (*tris)[t][2];

		int a = stl_edge_midpoint(stl_edge_table, 6 * N, (*verts_out), &vert_count, i0, i1);
		int b = stl_edge_midpoint(stl_edge_table, 6 * N, (*verts_out), &vert_count, i1, i2);
		int c = stl_edge_midpoint(stl_edge_table, 6 * N, (*verts_out), &vert_count, i2, i0);

		(*tris_out)[tri_count][0] = i0;
		(*tris_out)[tri_count][1] = a;
		(*tris_out)[tri_count][2] = c;
		tri_count++;

		(*tris_out)[tri_count][0] = i1;
		(*tris_out)[tri_count][1] = b;
		(*tris_out)[tri_count][2] = a;
		tri_count++;

		(*tris_out)[tri_count][0] = i2;
		(*tris_out)[tri_count][1] = c;
		(*tris_out)[tri_count][2] = b;
		tri_count++;

		(*tris_out)[tri_count][0] = a;
		(*tris_out)[tri_count][1] = b;
		(*tris_out)[tri_count][2] = c;
		tri_count++;
	}

	free(stl_edge_table);

	dims_out[0] = 3;
	dims_out[1] = 4;
	dims_out[2] = tri_count;

	double* model_out = md_alloc(3, dims_out, DL_SIZE);

	long strs_out[3];
	md_calc_strides(3, strs_out, dims_out, DL_SIZE);

	for (int t = 0; t < tri_count; t++)
		for (int v = 0; v < 3; v++)
			for (int d = 0; d < 3; d++) {

				long pos[3] = { d, v, t };
				MD_ACCESS(3, strs_out, pos, model_out) =
					(*verts_out)[ (*tris_out)[t][v] ][d];
			}

	stl_compute_normals(dims_out, model_out);

	free(verts);
	free(tris);
	free(verts_out);
	free(tris_out);

	return model_out;
}

double* stl_multiple_subdivide_model(int sub_divs, long dims_out[3], const long dims_in[3], const double *model)
{
	md_copy_dims(3, dims_out, dims_in);

        double* current_model = (double*)model;

        for (int i = 0; i < sub_divs; i++) {

		long dims[3];

		double* model = stl_subdivide_model(dims, dims_out, current_model);

		if (i > 0)
			md_free(current_model);

		md_copy_dims(3, dims_out, dims);
		current_model = model;
	}

	return current_model;
}
