/* Copyright 2023-2026. Institute of Biomedical Imaging. TU Graz.
 * Copyright 2026. Department of Radiology. Boston Children's Hospital.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2023 Nick Scholand <scholand@tugraz.at>
 * 2026 Moritz Blumenthal
 */

#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/cppmap.h"

#ifndef DIMS
#define DIMS 16
#endif

static void md_zrmax2(int D, const long dim[D], const long ostr[D], complex float* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2)
{
	float* tmp1 = md_alloc_sameplace(D, dim, FL_SIZE, iptr1);
	float* tmp2 = md_alloc_sameplace(D, dim, FL_SIZE, iptr2);

	md_real2(D, dim, MD_STRIDES(D, dim, FL_SIZE), tmp1, istr1, iptr1);
	md_real2(D, dim, MD_STRIDES(D, dim, FL_SIZE), tmp2, istr2, iptr2);

	md_max(D, dim, tmp1, tmp1, tmp2);

	md_clear2(D, dim, ostr, optr, CFL_SIZE);
	md_zcmpl_real2(D, dim, ostr, optr, MD_STRIDES(D, dim, FL_SIZE), tmp1);

	md_free(tmp1);
	md_free(tmp2);
}

static void md_zrmin2(int D, const long dim[D], const long ostr[D], complex float* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2)
{
	float* tmp1 = md_alloc_sameplace(D, dim, FL_SIZE, iptr1);
	float* tmp2 = md_alloc_sameplace(D, dim, FL_SIZE, iptr2);

	md_real2(D, dim, MD_STRIDES(D, dim, FL_SIZE), tmp1, istr1, iptr1);
	md_real2(D, dim, MD_STRIDES(D, dim, FL_SIZE), tmp2, istr2, iptr2);

	md_min(D, dim, tmp1, tmp1, tmp2);

	md_clear2(D, dim, ostr, optr, CFL_SIZE);
	md_zcmpl_real2(D, dim, ostr, optr, MD_STRIDES(D, dim, FL_SIZE), tmp1);

	md_free(tmp1);
	md_free(tmp2);
}



#define FUNC_LIST zabs, zacosr, zarg, zatanr, zconj, zcos, zcosh, zexp, zimag, zlog, zphsr, zreal, zround, zsin, zsinh, zsqrt, zsetnanzero, ()
#define FUNC_LIST_3OP zmul, zrmul, zmulc, zdiv, zadd, zsub, zpow, zatan2r, zrmax, zrmin, zlessequal, zgreatequal, ()

static const char help_str[] = "Perform function evaluation on array.";


typedef void (*z2op)(int D, const long dims[D], const long ostrs[D], complex float* optr, const long istrs[D], const complex float* iptr);
typedef void (*z3op)(int D, const long dims[D], const long ostrs[D], complex float* optr, const long istrs1[D], const complex float* iptr1, const long istrs2[D], const complex float* iptr2);

struct {

	z2op func;
	const char* name;

} calc_table[] = {

#define DENTRY(x) { md_ ## x ## 2, # x },
	MAP(DENTRY, FUNC_LIST)
#undef  DENTRY
	{ NULL, NULL }
};


struct {

	z3op func;
	const char* name;

} calc_table_scalar[] = {

#define DENTRY(x) { md_ ## x ## 2, # x },
	MAP(DENTRY, FUNC_LIST_3OP)
#undef  DENTRY
	{ NULL, NULL }
};

static bool help_func_calc(void* /*ptr*/, char /*c*/, const char* /*optarg*/)
{
	printf( "Available functions are:\n");
	printf( "\nOne argument:");

	for (int i = 0; i < (int)ARRAY_SIZE(calc_table); i++) {

		if (0 == i % 6)
			printf("\n");

		if (NULL != calc_table[i].name)
			printf("%s\t", calc_table[i].name);
	}

	printf("\n\nTwo arguments:");

	for (int i = 0; i < (int)ARRAY_SIZE(calc_table_scalar); i++) {

		if (0 == i % 6)
			printf("\n");

		if (NULL != calc_table_scalar[i].name)
			printf("%s\t", calc_table_scalar[i].name);
	}

	printf("\n");

	exit(0);
}

int main_calc(int argc, char* argv[argc])
{
	const char* in_file = NULL;
	const char* in_file2 = NULL;
	const char* out_file = NULL;

	const char* func_name = NULL;

	struct arg_s args[] = {

		ARG_STRING(true, &func_name, "func"),
		ARG_INFILE(true, &in_file, "input1"),
		ARG_INFILE(false, &in_file2, "input2"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	complex float val = NAN;

	const struct opt_s opts[] = {

		{ 'L', NULL, false, OPT_SPECIAL, help_func_calc, NULL, "", "Print a list of all supported functions" },
		OPT_CFL('v', &val, "scalar", "scalar value for functions that require a scalar argument"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	// Find function pointer before accessing memory

	z2op fun_2op = NULL;
	z3op fun_3op = NULL;

	for (int i = 0; NULL != calc_table[i].name; i++)
		if (0 == strcmp(func_name, calc_table[i].name))
			fun_2op = calc_table[i].func;

	for (int i = 0; NULL != calc_table_scalar[i].name; i++)
		if (0 == strcmp(func_name, calc_table_scalar[i].name))
			fun_3op = calc_table_scalar[i].func;

	if (NULL == fun_2op && NULL == fun_3op)
		error("Not supported function \"%s\" was called!\n", func_name);


	// Execute found function
	long in1_dims[DIMS];
	complex float* idata = load_cfl(in_file, DIMS, in1_dims);

	long in2_dims[DIMS];
	md_singleton_dims(DIMS, in2_dims);
	complex float* idata2 = NULL;

	if (NULL != in_file2)
		idata2 = load_cfl(in_file2, DIMS, in2_dims);

	if (!safe_isnanf(crealf(val))) {

		if (NULL != idata2)
			error("Cannot use scalar argument and input second file!\n");

		idata2 = anon_cfl(NULL, DIMS, in2_dims);
		md_zfill(DIMS, in2_dims, idata2, val);
	}

	if (!md_check_compat(DIMS, ~0UL, in1_dims, in2_dims))
		error("Input files have incompatible dimensions!\n");

	long max_dims[DIMS];
	md_max_dims(DIMS, ~0UL, max_dims, in1_dims, in2_dims);

	complex float* odata = create_cfl(out_file, DIMS, max_dims);

	long istrs1[DIMS];
	long istrs2[DIMS];
	long ostrs[DIMS];

	md_calc_strides(DIMS, istrs1, in1_dims, CFL_SIZE);
	md_calc_strides(DIMS, istrs2, in2_dims, CFL_SIZE);
	md_calc_strides(DIMS, ostrs, max_dims, CFL_SIZE);

	if (NULL != fun_2op)
		fun_2op(DIMS, max_dims, ostrs, odata, istrs1, idata);

	if (NULL != fun_3op)
		fun_3op(DIMS, max_dims, ostrs, odata, istrs1, idata, istrs2, idata2);

	unmap_cfl(DIMS, in1_dims, idata);
	unmap_cfl(DIMS, in2_dims, idata2);
	unmap_cfl(DIMS, max_dims, odata);

	return 0;
}


