/* Copyright 2025-2026. TU Graz. Institute of Biomedical Imaging
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2025 Martin Heide
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/specfun.h"
#include "num/flpmath.h"
#include "num/fft.h"

#include "stl/misc.h"
#include "stl/models.h"

#include "simu/shape.h"
#include "simu/phantom.h"

#include "utest.h"

#define UNUSED(x) (void)x


static bool test_stl_kspace(void)
{
        bool b = true;
        long stldims[3];

	struct phantom_opts popts;
	popts.kspace = true;
        double* model = stl_internal_tetrahedron(stldims);
	stl_compute_normals(stldims, model);
	phantom_stl_init(&popts, 3, stldims, model);

	float pos[3] = { 0., 0., 0. };

	complex double c = stl_fun_k(&popts, 0, pos);

	if (1E-10 < fabs(creal(c) - 0.243))
		b = false;

	pos[0] = 0.1;
	pos[1] = 0.1;
	pos[2] = 0.;

	c = stl_fun_k(&popts, 0, pos);

	if (1E-10 < fabs(creal(c) - 4. * 0.0597860454167))
		b = false;

	if (1E-10 < fabs(cimag(c)))
		b = false;

	popts.dstr(&popts);
        md_free(model);

        return b;
}

UT_REGISTER_TEST(test_stl_kspace);



static bool test_stl_kspace2(void)
{
        bool b = true;
        long stldims[3];

	struct phantom_opts popts;
	popts.kspace = true;
        double* model = stl_internal_tetrahedron(stldims);
	stl_compute_normals(stldims, model);
	phantom_stl_init(&popts, 3, stldims, model);

	float pos[3] = { 0., 0., 0. };

	for (int i = 4; i < 8; i++) {

		double x = pow(10., -i);	// close to 0.
		pos[0] = x;

		complex double c = stl_fun_k(&popts, 0, pos);

		if (1E-6 < fabs(creal(c) - 0.243))
			b = false;
	}

	popts.dstr(&popts);
        md_free(model);

        return b;
}

UT_REGISTER_TEST(test_stl_kspace2);



static bool test_kpolygon(void)
{
	const double pg[4][2] = {
		{ -0.5, -0.5 },
		{ +0.5, -0.5 },
		{ +0.5, +0.5 },
		{ -0.5, +0.5 },
	};

	double test[][3] = {
		{  0.,  0.,  0. },
		{ +0.1, 0.,  0. },
		{ -0.2, 0.,  0. },
		{ +0.1, 0.3, 0. },
		{ -0.2, 0.2, 0. },
	};

	for (int i = 0; i < 5; i++) {

		double diff1 = sinc(M_PI * test[i][0]) * sinc(M_PI * test[i][1]);
		double diff2 = 4. * creal(kpolygon(4, pg, test[i]));

		if (1.E-10 < fabs(diff1 - diff2))
			return false;
	}

	return true;
}

UT_REGISTER_TEST(test_kpolygon);



static bool test_kpolygon2(void)
{
	const double pg[4][2] = {

		{ -0.5, -0.5 },
		{ +0.5, -0.5 },
		{ +0.5, +0.5 },
		{ -0.5, +0.5 },
	};

	for (int i = 3; i < 9; i++) {

		double x = pow(10., -i);	// close to 0.
		double y = 0.;

		double pos[3] = { x, y, 0. };

		double value = 4. * kpolygon(4, pg, pos);

		if (1.E-5 < fabs(value - 1.))
			return false;
	}

	return true;
}

UT_REGISTER_TEST(test_kpolygon2);

static void dstr_none(void* v)
{
	UNUSED(v);
}

static complex double fun_dirac(const void* v, const long C, const float c[])
{
	UNUSED(v);
	UNUSED(C);

	if (0 == c[0] && 0 == c[1] && 0 == c[2])
		return 1.;

	return 0.;
}

static complex double fun_const(const void* v, const long C, const float c[])
{
	UNUSED(v);
	UNUSED(C);
	UNUSED(c);

	return 1.;
}

static bool test_phantom_sampling_dirac_nocoil_k(void)
{
	bool b = true;
	bool kspace = true;

	// sampling grid for phantom
	long gdims[DIMS];
	struct grid_opts gopts = grid_opts_defaults;
	gopts.kspace = kspace;
	gopts.dims[0] = 16;
	gopts.dims[1] = 16;
	float* grid = compute_grid(DIMS, gdims, &gopts, NULL, NULL);

	// coil config
	struct coil_opts copts = coil_opts_defaults;
	copts.ctype = COIL_NONE;
	copts.kspace = kspace;
	cnstr_coils(DIMS, &copts, false);

	// sampling grid for coils
	struct grid_opts cgopts = gopts;
	long stdims[DIMS];
	float* straj = create_senstraj(DIMS, stdims, &cgopts, &copts);

	// prepare phantom
	struct phantom_opts popts = phantom_opts_defaults;
	popts.kspace = kspace;
	popts.Nc = 1;
	popts.dstr = dstr_none;
	popts.fun = fun_dirac;

	long odims_[DIMS];
	complex double* cdout = sample_signal(DIMS, odims_, gdims, grid, stdims, straj, &popts, &copts);

	long odims[DIMS];
	md_singleton_dims(DIMS, odims);
	odims[0] = odims_[1];
	odims[1] = odims_[2];

	complex float* cdoutf = md_alloc(DIMS, odims, CFL_SIZE);

	long strsf[DIMS];
	md_calc_strides(DIMS, strsf, odims, CFL_SIZE);

	md_zdouble2float(DIMS, odims, cdoutf, cdout);

	complex float* reco = md_alloc(DIMS, odims, CFL_SIZE);

	ifftc(DIMS, odims, 3, reco, cdoutf);

	complex float* ones = md_alloc(DIMS, odims, CFL_SIZE);
	md_zfill(DIMS, odims, ones, 1);

	float err = md_znrmse(DIMS, odims, ones, reco);

	if (0 < fabs(err))
		b = false;

	md_free(reco);
	md_free(ones);
	md_free(cdoutf);
	md_free(cdout);
	copts.dstr(&copts);
	popts.dstr(&popts);
	md_free(straj);
	md_free(grid);

	return b;
}

UT_REGISTER_TEST(test_phantom_sampling_dirac_nocoil_k);


static bool test_phantom_sampling_dirac_nocoil_x(void)
{
	bool b = true;
	bool kspace = false;

	// sampling grid for phantom
	long gdims[DIMS];
	struct grid_opts gopts = grid_opts_defaults;
	gopts.kspace = kspace;
	gopts.dims[0] = 16;
	gopts.dims[1] = 16;
	float* grid = compute_grid(DIMS, gdims, &gopts, NULL, NULL);

	// coil config
	struct coil_opts copts = coil_opts_defaults;
	copts.ctype = COIL_NONE;
	copts.kspace = kspace;
	cnstr_coils(DIMS, &copts, false);

	// sampling grid for coils
	struct grid_opts cgopts = gopts;
	long stdims[DIMS];
	float* straj = create_senstraj(DIMS, stdims, &cgopts, &copts);

	// prepare phantom
	struct phantom_opts popts = phantom_opts_defaults;
	popts.kspace = kspace;
	popts.Nc = 1;
	popts.dstr = dstr_none;
	popts.fun = fun_const;

	long odims_[DIMS];
	complex double* cdout = sample_signal(DIMS, odims_, gdims, grid, stdims, straj, &popts, &copts);

	long odims[DIMS];
	md_singleton_dims(DIMS, odims);
	odims[0] = odims_[1];
	odims[1] = odims_[2];
	complex float* cdoutf = md_alloc(DIMS, odims, CFL_SIZE);
	md_zdouble2float(DIMS, odims, cdoutf, cdout);

	long strsf[DIMS];
	md_calc_strides(DIMS, strsf, odims, CFL_SIZE);
	complex float* ones = md_alloc(DIMS, odims, CFL_SIZE);
	md_zfill(DIMS, odims, ones, 1);

	float err = md_znrmse(DIMS, odims, ones, cdoutf);

	if (0 < fabs(err))
		b = false;

	md_free(ones);
	md_free(cdoutf);
	md_free(cdout);
	copts.dstr(&copts);
	popts.dstr(&popts);
	md_free(straj);
	md_free(grid);


	return b;
}

UT_REGISTER_TEST(test_phantom_sampling_dirac_nocoil_x);

static complex float* compute_HEAD_2D_8CH(int D, long dims[D], const long gdims[D], const float* grid, const long N)
{
	md_singleton_dims(D, dims);
	dims[0] = gdims[1];
	dims[1] = gdims[2];
	dims[2] = gdims[3];
	dims[COIL_DIM] = N;

	complex float* sens = md_alloc(D, dims, CFL_SIZE);

	long gstrs[D], sstrs[D], pos[D], posg[D];
	md_calc_strides(D, gstrs, gdims, FL_SIZE);
	md_calc_strides(D, sstrs, dims, CFL_SIZE);
	md_set_dims(D, pos, 0);

	float sh = 2;

	do {
		md_set_dims(D, posg, 0);
		posg[1] = pos[0];
		posg[2] = pos[1];
		posg[3] = pos[2];
		const float* g = &MD_ACCESS(D, gstrs, posg, grid);

		complex double val = 0;

		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 5; j++)
				val += sens_coeff[pos[COIL_DIM]][i][j] * cexpf(-2.i * M_PI * ((i - sh) * g[0] + (j - sh) * g[1]) / 2.);

		MD_ACCESS(D, sstrs, pos, sens) = val;

	} while(md_next(D, dims, 15UL, pos));

	return sens;
}

static bool test_phantom_sampling_const_8chcoil_x(void)
{
	bool b = true;
	bool kspace = false;

	// sampling grid for phantom
	long gdims[DIMS];
	struct grid_opts gopts = grid_opts_defaults;
	gopts.kspace = kspace;
	gopts.dims[0] = 16;
	gopts.dims[1] = 16;
	float* grid = compute_grid(DIMS, gdims, &gopts, NULL, NULL);

	// coil config
	struct coil_opts copts = coil_opts_defaults;
	copts.ctype = HEAD_2D_8CH;
	copts.kspace = kspace;
	copts.flags = 3;
	cnstr_coils(DIMS, &copts, false);

	// sampling grid for coils
	struct grid_opts cgopts = gopts;
	long stdims[DIMS];
	float* straj = create_senstraj(DIMS, stdims, &cgopts, &copts);

	// prepare phantom
	struct phantom_opts popts = phantom_opts_defaults;
	popts.kspace = kspace;
	popts.Nc = 1;
	popts.dstr = dstr_none;
	popts.fun = fun_const;
	long odims_[DIMS];
	complex double* cdout = sample_signal(DIMS, odims_, gdims, grid, stdims, straj, &popts, &copts);
	long odims[DIMS];
	md_singleton_dims(DIMS, odims);
	odims[0] = odims_[1];
	odims[1] = odims_[2];

	complex float* cdoutf = md_alloc(DIMS, odims, CFL_SIZE);
	md_zdouble2float(DIMS, odims, cdoutf, cdout);
	long sdims[DIMS];
	complex float* sens = compute_HEAD_2D_8CH(DIMS, sdims, gdims, grid, copts.N);
	float err = md_znrmse(DIMS, odims, sens, cdoutf);

	if (1E-06 < fabs(err))
		b = false;

	md_free(sens);
	md_free(cdoutf);
	md_free(cdout);
	copts.dstr(&copts);
	popts.dstr(&popts);
	md_free(straj);
	md_free(grid);

	return b;
}
UT_REGISTER_TEST(test_phantom_sampling_const_8chcoil_x);

static bool test_phantom_sampling_dirac_8chcoil_k(void)
{
	bool b = true;
	bool kspace = true;

	// sampling grid for phantom
	long gdims[DIMS];
	struct grid_opts gopts = grid_opts_defaults;
	gopts.kspace = kspace;
	// we want to sample the sens coefficients in k-space.
	// Since they are defined for an oversampled FOV, we do the same
	gopts.b0[0] = 1;
	gopts.b1[1] = 1;
	gopts.dims[0] = 64;
	gopts.dims[1] = 64;
	float* grid = compute_grid(DIMS, gdims, &gopts, NULL, NULL);

	// coil config
	struct coil_opts copts = coil_opts_defaults;
	copts.ctype = HEAD_2D_8CH;
	copts.kspace = kspace;
	copts.flags = 3;
	cnstr_coils(DIMS, &copts, false);

	// sampling grid for coils
	struct grid_opts cgopts = gopts;
	long stdims[DIMS];
	float* straj = create_senstraj(DIMS, stdims, &cgopts, &copts);

	// prepare phantom
	struct phantom_opts popts = phantom_opts_defaults;
	popts.kspace = kspace;
	popts.Nc = 1;
	popts.dstr = dstr_none;
	popts.fun = fun_dirac;

	// sample, reconstruct and extract
	long odims_[DIMS];
	complex double* cdout = sample_signal(DIMS, odims_, gdims, grid, stdims, straj, &popts, &copts);

	long odims[DIMS];
	md_copy_dims(DIMS, odims, odims_);
	odims[0] = odims_[1];
	odims[1] = odims_[2];
	odims[2] = 1;

	complex float* cdoutf = md_alloc(DIMS, odims, CFL_SIZE);
	complex float* reco = md_alloc(DIMS, odims, CFL_SIZE);

	md_zdouble2float(DIMS, odims, cdoutf, cdout);
	ifftc(DIMS, odims, 3, reco, cdoutf);

	long edims[DIMS];
	md_singleton_dims(DIMS, edims);
	edims[0] = 32;
	edims[1] = 32;
	edims[3] = 2;

	complex float* block = md_alloc(DIMS, edims, CFL_SIZE);
	long bpos[DIMS] = { 16, 16 };

	md_copy_block(DIMS, bpos, edims, block, odims, reco, CFL_SIZE);

	// compare against reference
	long sgdims[DIMS];
	struct grid_opts sgopts = grid_opts_defaults;
	sgopts.dims[0] = 32;
	sgopts.dims[1] = 32;
	sgopts.kspace = false;
	float* sgrid = compute_grid(DIMS, sgdims, &sgopts, NULL, NULL);
	long sdims[DIMS];
	complex float* sens = compute_HEAD_2D_8CH(DIMS, sdims, sgdims, sgrid, copts.N);

	float err = md_znrmse(DIMS, edims, sens, block);

	if (1E-06 < fabs(err))
		b = false;

	md_free(sgrid);
	md_free(block);
	md_free(reco);
	md_free(sens);
	md_free(cdoutf);
	md_free(cdout);
	popts.dstr(&popts);
	md_free(straj);
	copts.dstr(&copts);
	md_free(grid);

	return b;
}
UT_REGISTER_TEST(test_phantom_sampling_dirac_8chcoil_k);

static complex float* compute_HEAD_3D_64CH(int D, long dims[D], const long gdims[D], const float* grid, long N)
{
	md_singleton_dims(D, dims);
	dims[0] = gdims[1];
	dims[1] = gdims[2];
	dims[2] = gdims[3];
	dims[COIL_DIM] = N;

	complex float* sens = md_alloc(D, dims, CFL_SIZE);

	long gstrs[D], sstrs[D], pos[D], posg[D];
	md_calc_strides(D, gstrs, gdims, FL_SIZE);
	md_calc_strides(D, sstrs, dims, CFL_SIZE);
	md_set_dims(D, pos, 0);

	float sh = 2;

	do {
		md_set_dims(D, posg, 0);
		posg[1] = pos[0];
		posg[2] = pos[1];
		posg[3] = pos[2];
		const float* g = &MD_ACCESS(D, gstrs, posg, grid);

		complex double val = 0;

		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 5; j++)
				for (int k = 0; k < 5; k++)
					val += sens64_coeff[pos[COIL_DIM]][i][j][k] * cexpf(-2.i * M_PI * ((i - sh) * g[0] + (j - sh) * g[1] + (k - sh) * g[2]) / 2.);

		MD_ACCESS(D, sstrs, pos, sens) = val;

	} while(md_next(D, dims, 15UL, pos));

	return sens;
}

static bool test_phantom_sampling_const_64chcoil_x(void)
{
	bool b = true;
	bool kspace = false;

	// sampling grid for phantom
	long gdims[DIMS];
	struct grid_opts gopts = grid_opts_defaults;
	gopts.kspace = kspace;
	gopts.dims[0] = 20;
	gopts.dims[1] = 22;
	gopts.dims[2] = 24;
	gopts.b2[2] = 0.5;
	float* grid = compute_grid(DIMS, gdims, &gopts, NULL, NULL);

	// coil config
	struct coil_opts copts = coil_opts_defaults;
	copts.ctype = HEAD_3D_64CH;
	copts.kspace = kspace;
	copts.flags = 3;
	cnstr_coils(DIMS, &copts, false);

	// sampling grid for coils
	struct grid_opts cgopts = gopts;
	long stdims[DIMS];
	float* straj = create_senstraj(DIMS, stdims, &cgopts, &copts);

	// prepare phantom
	struct phantom_opts popts = phantom_opts_defaults;
	popts.kspace = kspace;
	popts.Nc = 1;
	popts.dstr = dstr_none;
	popts.fun = fun_const;

	long odims[DIMS];
	complex double* cdout = sample_signal(DIMS, odims, gdims, grid, stdims, straj, &popts, &copts);
	complex float* cdoutf = md_alloc(DIMS, odims, CFL_SIZE);

	md_zdouble2float(DIMS, odims, cdoutf, cdout);

	long sdims[DIMS];
	complex float* sens = compute_HEAD_3D_64CH(DIMS, sdims, gdims, grid, copts.N);

	float err = md_znrmse(DIMS, odims, sens, cdoutf);

	if (1E-06 < fabs(err))
		b = false;

	md_free(sens);
	md_free(cdoutf);
	md_free(cdout);
	copts.dstr(&copts);
	popts.dstr(&popts);
	md_free(straj);
	md_free(grid);

	return b;
}
UT_REGISTER_TEST(test_phantom_sampling_const_64chcoil_x);

static bool test_phantom_sampling_dirac_64chcoil_k(void)
{
	bool b = true;
	bool kspace = true;

	// sampling grid for phantom
	long gdims[DIMS];
	struct grid_opts gopts = grid_opts_defaults;
	gopts.kspace = kspace;
	// we want to sample the sens coefficients in k-space.
	// Since they are defined for an oversampled FOV, we do the same
	gopts.b0[0] = 1;
	gopts.b1[1] = 1;
	gopts.b2[2] = 1;
	gopts.dims[0] = 64;
	gopts.dims[1] = 64;
	gopts.dims[2] = 64;
	float* grid = compute_grid(DIMS, gdims, &gopts, NULL, NULL);

	// coil config
	struct coil_opts copts = coil_opts_defaults;
	copts.ctype = HEAD_3D_64CH;
	copts.kspace = kspace;
	copts.flags = 3;
	cnstr_coils(DIMS, &copts, false);

	// sampling grid for coils
	struct grid_opts cgopts = gopts;
	long stdims[DIMS];
	float* straj = create_senstraj(DIMS, stdims, &cgopts, &copts);

	// prepare phantom
	struct phantom_opts popts = phantom_opts_defaults;
	popts.kspace = kspace;
	popts.Nc = 1;
	popts.dstr = dstr_none;
	popts.fun = fun_dirac;

	// sample, reconstruct and extract
	long odims_[DIMS];
	complex double* cdout = sample_signal(DIMS, odims_, gdims, grid, stdims, straj, &popts, &copts);

	long odims[DIMS];
	md_copy_dims(DIMS, odims, odims_);
	odims[0] = gdims[1];
	odims[1] = gdims[2];
	odims[2] = gdims[3];

	complex float* cdoutf = md_alloc(DIMS, odims, CFL_SIZE);
	complex float* reco = md_alloc(DIMS, odims, CFL_SIZE);
	md_zdouble2float(DIMS, odims, cdoutf, cdout);
	ifftc(DIMS, odims, 7, reco, cdoutf);
	long edims[DIMS];
	md_singleton_dims(DIMS, edims);
	edims[0] = 32;
	edims[1] = 32;
	edims[2] = 32;
	edims[3] = copts.N;
	complex float* block = md_alloc(DIMS, edims, CFL_SIZE);
	long bpos[DIMS] = { 16, 16, 16 };
	md_copy_block(DIMS, bpos, edims, block, odims, reco, CFL_SIZE);

	// compare against reference
	long sgdims[DIMS];
	struct grid_opts sgopts = grid_opts_defaults;
	sgopts.dims[0] = 32;
	sgopts.dims[1] = 32;
	sgopts.dims[2] = 32;
	sgopts.b0[0] = 0.5;
	sgopts.b1[1] = 0.5;
	sgopts.b2[2] = 0.5;
	sgopts.kspace = false;
	float* sgrid = compute_grid(DIMS, sgdims, &sgopts, NULL, NULL);
	long sdims[DIMS];
	complex float* sens = compute_HEAD_3D_64CH(DIMS, sdims, sgdims, sgrid, copts.N);

	float err = md_znrmse(DIMS, edims, sens, block);

	if (1E-06 < fabs(err))
		b = false;

	md_free(sgrid);
	md_free(block);
	md_free(reco);
	md_free(sens);
	md_free(cdoutf);
	md_free(cdout);
	popts.dstr(&popts);
	md_free(straj);
	copts.dstr(&copts);
	md_free(grid);

	return b;
}
UT_REGISTER_TEST(test_phantom_sampling_dirac_64chcoil_k);
