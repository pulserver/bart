#include <complex.h>

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/vecops_strided.h"
#include "num/rand.h"
#include "num/init.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "utest.h"


#define UNUSED(x) (void)x


static bool test_optimized_md_zfmac2_flags(bart_flags_t out_flag, bart_flags_t in1_flag, bart_flags_t in2_flag, bool optimization_expected, float err_tol)
{
#ifndef USE_CUDA
	UNUSED(out_flag);
	UNUSED(in1_flag);
	UNUSED(in2_flag);
	UNUSED(optimization_expected);
	UNUSED(err_val);
	return true;
#else

	enum {D = 5};
	bart_dim_t dims[D] = {3, 5, 2, 4, 4};

	size_t size = CFL_SIZE;

	bart_dim_t odims[D];
	bart_dim_t idims1[D];
	bart_dim_t idims2[D];
	md_select_dims(D, out_flag, odims, dims);
	md_select_dims(D, in1_flag, idims1, dims);
	md_select_dims(D, in2_flag, idims2, dims);

	bart_stride_t ostr[D];
	bart_stride_t istr1[D];
	bart_stride_t istr2[D];
	md_calc_strides(D, ostr, odims, size);
	md_calc_strides(D, istr1, idims1, size);
	md_calc_strides(D, istr2, idims2, size);

	complex float* optr1 = md_alloc_gpu(D, odims, size);
	complex float* optr2 = md_alloc_gpu(D, odims, size);
	complex float* iptr1 = md_alloc_gpu(D, idims1, size);
	complex float* iptr2 = md_alloc_gpu(D, idims2, size);

	md_gaussian_rand(D, odims, optr1);
	md_gaussian_rand(D, idims1, iptr1);
	md_gaussian_rand(D, idims2, iptr2);
	md_copy(D, odims, optr2, optr1, size);

	deactivate_strided_vecops();
	md_zfmac2(D, dims, ostr, optr1, istr1, iptr1, istr2, iptr2);
	activate_strided_vecops();
	bool result = (optimization_expected == simple_zfmac(D, dims, ostr, optr2, istr1, iptr1, istr2, iptr2));
	result &= (!optimization_expected) || (err_tol > md_znrmse(D, odims, optr1, optr2));
	debug_printf(result ? DP_DEBUG1 : DP_INFO, "%e\n", md_znrmse(D, odims, optr1, optr2));
	md_free(optr1);
	md_free(optr2);
	md_free(iptr1);
	md_free(iptr2);

	return result;
#endif
}

static bool test_optimized_md_zfmac2_dot(void) { UT_RETURN_ASSERT(test_optimized_md_zfmac2_flags(UINT64_C(0), UINT64_C(1), UINT64_C(1), true, 1.1e-5)); }
static bool test_optimized_md_zfmac2_dot2(void) { UT_RETURN_ASSERT(test_optimized_md_zfmac2_flags(UINT64_C(2), UINT64_C(3), UINT64_C(3), true, 1.e-6)); }
static bool test_optimized_md_zfmac2_gemv(void) { UT_RETURN_ASSERT(test_optimized_md_zfmac2_flags(UINT64_C(1), UINT64_C(3), UINT64_C(2), true, 3.e-6)); }
static bool test_optimized_md_zfmac2_gemv2(void) { UT_RETURN_ASSERT(test_optimized_md_zfmac2_flags(UINT64_C(2), UINT64_C(1), UINT64_C(3), true, 1.5e-6)); }
static bool test_optimized_md_zfmac2_gemv3(void) { UT_RETURN_ASSERT(test_optimized_md_zfmac2_flags(UINT64_C(14), UINT64_C(13), UINT64_C(7), true, 1.e-6)); }
static bool test_optimized_md_zfmac2_gemm(void) { UT_RETURN_ASSERT(test_optimized_md_zfmac2_flags(UINT64_C(3), UINT64_C(6), UINT64_C(5), true, 2.e-6)); }
static bool test_optimized_md_zfmac2_gemm2(void) { UT_RETURN_ASSERT(test_optimized_md_zfmac2_flags(UINT64_C(11), UINT64_C(14), UINT64_C(13), true, 1.e-6));}
static bool test_optimized_md_zfmac2_ger(void) { UT_RETURN_ASSERT(test_optimized_md_zfmac2_flags(UINT64_C(3), UINT64_C(1), UINT64_C(2), true, 2.e-6)); }
static bool test_optimized_md_zfmac2_ger2(void) { UT_RETURN_ASSERT(test_optimized_md_zfmac2_flags(UINT64_C(7), UINT64_C(5), UINT64_C(6), true, 1.e-6)); }
static bool test_optimized_md_zfmac2_axpy(void) { UT_RETURN_ASSERT(test_optimized_md_zfmac2_flags(UINT64_C(1), UINT64_C(1), UINT64_C(0), true, 2.e-6)); }
static bool test_optimized_md_zfmac2_axpy2(void) { UT_RETURN_ASSERT(test_optimized_md_zfmac2_flags(UINT64_C(3), UINT64_C(2), UINT64_C(3), true, 1.e-6));}
//static bool test_optimized_md_zfmac2_dot_transp(void) { UT_RETURN_ASSERT(test_optimized_md_zfmac2_flags(5ul, 7ul, 7ul, true, 1.e-6)); }

UT_GPU_REGISTER_TEST(test_optimized_md_zfmac2_dot);
UT_GPU_REGISTER_TEST(test_optimized_md_zfmac2_dot2);
UT_GPU_REGISTER_TEST(test_optimized_md_zfmac2_gemv);
UT_GPU_REGISTER_TEST(test_optimized_md_zfmac2_gemv2);
UT_GPU_REGISTER_TEST(test_optimized_md_zfmac2_gemv3);
UT_GPU_REGISTER_TEST(test_optimized_md_zfmac2_gemm);
UT_GPU_REGISTER_TEST(test_optimized_md_zfmac2_gemm2);
UT_GPU_REGISTER_TEST(test_optimized_md_zfmac2_ger);
UT_GPU_REGISTER_TEST(test_optimized_md_zfmac2_ger2);
UT_GPU_REGISTER_TEST(test_optimized_md_zfmac2_axpy);
UT_GPU_REGISTER_TEST(test_optimized_md_zfmac2_axpy2);
//UT _GPU_REGISTER_TEST(test_optimized_md_zfmac2_dot_transp); only activated for large arrays


static bool test_optimized_md_zfmacc2_flags(bart_flags_t out_flag, bart_flags_t in1_flag, bart_flags_t in2_flag, bool optimization_expected, float err_tol)
{
#ifndef USE_CUDA
	UNUSED(out_flag);
	UNUSED(in1_flag);
	UNUSED(in2_flag);
	UNUSED(optimization_expected);
	UNUSED(err_val);
	return true;
#else

	enum {D = 5};
	bart_dim_t dims[D] = {3, 5, 2, 4, 4};

	size_t size = CFL_SIZE;

	bart_dim_t odims[D];
	bart_dim_t idims1[D];
	bart_dim_t idims2[D];
	md_select_dims(D, out_flag, odims, dims);
	md_select_dims(D, in1_flag, idims1, dims);
	md_select_dims(D, in2_flag, idims2, dims);

	bart_stride_t ostr[D];
	bart_stride_t istr1[D];
	bart_stride_t istr2[D];
	md_calc_strides(D, ostr, odims, size);
	md_calc_strides(D, istr1, idims1, size);
	md_calc_strides(D, istr2, idims2, size);

	complex float* optr1 = md_alloc_gpu(D, odims, size);
	complex float* optr2 = md_alloc_gpu(D, odims, size);
	complex float* iptr1 = md_alloc_gpu(D, idims1, size);
	complex float* iptr2 = md_alloc_gpu(D, idims2, size);

	md_gaussian_rand(D, odims, optr1);
	md_gaussian_rand(D, idims1, iptr1);
	md_gaussian_rand(D, idims2, iptr2);
	md_copy(D, odims, optr2, optr1, size);

	deactivate_strided_vecops();
	md_zfmacc2(D, dims, ostr, optr1, istr1, iptr1, istr2, iptr2);
	activate_strided_vecops();
	bool result = (optimization_expected == simple_zfmacc(D, dims, ostr, optr2, istr1, iptr1, istr2, iptr2));
	result &= (!optimization_expected) || (err_tol > md_znrmse(D, odims, optr1, optr2));
	if (!result)
		debug_printf(DP_INFO, "%.10f\n", md_znrmse(D, odims, optr1, optr2));
	md_free(optr1);
	md_free(optr2);
	md_free(iptr1);
	md_free(iptr2);

	return result;
#endif
}

static bool test_optimized_md_zfmacc2_dot(void) { UT_RETURN_ASSERT(test_optimized_md_zfmacc2_flags(UINT64_C(0), UINT64_C(1), UINT64_C(1), true, 5.e-6)); }
static bool test_optimized_md_zfmacc2_dot2(void) { UT_RETURN_ASSERT(test_optimized_md_zfmacc2_flags(UINT64_C(2), UINT64_C(3), UINT64_C(3), true, 5.e-6)); }
static bool test_optimized_md_zfmacc2_gemv(void) { UT_RETURN_ASSERT(test_optimized_md_zfmacc2_flags(UINT64_C(1), UINT64_C(3), UINT64_C(2), true, 2.e-6)); }
static bool test_optimized_md_zfmacc2_gemv2(void) { UT_RETURN_ASSERT(test_optimized_md_zfmacc2_flags(UINT64_C(2), UINT64_C(1), UINT64_C(3), true, 2.e-6)); }
static bool test_optimized_md_zfmacc2_gemv3(void) { UT_RETURN_ASSERT(test_optimized_md_zfmacc2_flags(UINT64_C(14), UINT64_C(13), UINT64_C(7), true, 1.e-6)); }
static bool test_optimized_md_zfmacc2_gemm(void) { UT_RETURN_ASSERT(test_optimized_md_zfmacc2_flags(UINT64_C(3), UINT64_C(6), UINT64_C(5), true, 2.e-6)); }
static bool test_optimized_md_zfmacc2_gemm2(void) { UT_RETURN_ASSERT(test_optimized_md_zfmacc2_flags(UINT64_C(11), UINT64_C(14), UINT64_C(13), true, 1.e-6));}
static bool test_optimized_md_zfmacc2_ger(void) { UT_RETURN_ASSERT(test_optimized_md_zfmacc2_flags(UINT64_C(3), UINT64_C(1), UINT64_C(2), true, 2.e-6)); }
static bool test_optimized_md_zfmacc2_ger2(void) { UT_RETURN_ASSERT(test_optimized_md_zfmacc2_flags(UINT64_C(7), UINT64_C(5), UINT64_C(6), true, 1.e-6)); }
static bool test_optimized_md_zfmacc2_axpy(void) { UT_RETURN_ASSERT(test_optimized_md_zfmacc2_flags(UINT64_C(1), UINT64_C(1), UINT64_C(0), true, 5.e-6)); }
static bool test_optimized_md_zfmacc2_axpy2(void) { UT_RETURN_ASSERT(test_optimized_md_zfmacc2_flags(UINT64_C(3), UINT64_C(2), UINT64_C(3), true, 1.e-6));}
//static bool test_optimized_md_zfmacc2_dot_transp(void) { UT_RETURN_ASSERT(test_optimized_md_zfmacc2_flags(5ul, 7ul, 7ul, true, 1.e-6)); }

UT_GPU_REGISTER_TEST(test_optimized_md_zfmacc2_dot);
UT_GPU_REGISTER_TEST(test_optimized_md_zfmacc2_dot2);
UT_GPU_REGISTER_TEST(test_optimized_md_zfmacc2_gemv);
UT_GPU_REGISTER_TEST(test_optimized_md_zfmacc2_gemv2);
UT_GPU_REGISTER_TEST(test_optimized_md_zfmacc2_gemv3);
UT_GPU_REGISTER_TEST(test_optimized_md_zfmacc2_gemm);
UT_GPU_REGISTER_TEST(test_optimized_md_zfmacc2_gemm2);
UT_GPU_REGISTER_TEST(test_optimized_md_zfmacc2_ger);
UT_GPU_REGISTER_TEST(test_optimized_md_zfmacc2_ger2);
UT_GPU_REGISTER_TEST(test_optimized_md_zfmacc2_axpy);
UT_GPU_REGISTER_TEST(test_optimized_md_zfmacc2_axpy2);
//UT _GPU_REGISTER_TEST(test_optimized_md_zfmacc2_dot_transp); only activated for large arrays

static bool test_optimized_md_fmac2_flags(bart_flags_t out_flag, bart_flags_t in1_flag, bart_flags_t in2_flag, bool optimization_expected, float err_tol)
{
#ifndef USE_CUDA
	UNUSED(out_flag);
	UNUSED(in1_flag);
	UNUSED(in2_flag);
	UNUSED(optimization_expected);
	UNUSED(err_val);
	return true;
#else

	enum {D = 5};
	bart_dim_t dims[D] = {3, 5, 2, 4, 4};

	size_t size = FL_SIZE;

	bart_dim_t odims[D];
	bart_dim_t idims1[D];
	bart_dim_t idims2[D];
	md_select_dims(D, out_flag, odims, dims);
	md_select_dims(D, in1_flag, idims1, dims);
	md_select_dims(D, in2_flag, idims2, dims);

	bart_stride_t ostr[D];
	bart_stride_t istr1[D];
	bart_stride_t istr2[D];
	md_calc_strides(D, ostr, odims, size);
	md_calc_strides(D, istr1, idims1, size);
	md_calc_strides(D, istr2, idims2, size);

	float* optr1 = md_alloc_gpu(D, odims, CFL_SIZE);
	float* optr2 = md_alloc_gpu(D, odims, CFL_SIZE);
	float* iptr1 = md_alloc_gpu(D, idims1, CFL_SIZE);
	float* iptr2 = md_alloc_gpu(D, idims2, CFL_SIZE);

	md_gaussian_rand(D, odims, (complex float*)optr1);
	md_gaussian_rand(D, idims1, (complex float*)iptr1);
	md_gaussian_rand(D, idims2, (complex float*)iptr2);
	md_copy(D, odims, optr2, optr1, size);

	deactivate_strided_vecops();
	md_fmac2(D, dims, ostr, optr1, istr1, iptr1, istr2, iptr2);
	activate_strided_vecops();
	bool result = (optimization_expected == simple_fmac(D, dims, ostr, optr2, istr1, iptr1, istr2, iptr2));
	result &= (!optimization_expected) || (err_tol > md_nrmse(D, odims, optr1, optr2));
	if (!result)
		debug_printf(DP_INFO, "%.10f\n", md_nrmse(D, odims, optr1, optr2));
	md_free(optr1);
	md_free(optr2);
	md_free(iptr1);
	md_free(iptr2);

	return result;
#endif
}

static bool test_optimized_md_fmac2_dot(void) { UT_RETURN_ASSERT(test_optimized_md_fmac2_flags(UINT64_C(0), UINT64_C(1), UINT64_C(1), true, 2.e-5)); }
static bool test_optimized_md_fmac2_dot2(void) { UT_RETURN_ASSERT(test_optimized_md_fmac2_flags(UINT64_C(2), UINT64_C(3), UINT64_C(3), true, 2.e-6)); }
static bool test_optimized_md_fmac2_gemv(void) { UT_RETURN_ASSERT(test_optimized_md_fmac2_flags(UINT64_C(1), UINT64_C(3), UINT64_C(2), true, 3.e-6)); }
static bool test_optimized_md_fmac2_gemv2(void) { UT_RETURN_ASSERT(test_optimized_md_fmac2_flags(UINT64_C(2), UINT64_C(1), UINT64_C(3), true, 2.e-6)); }
static bool test_optimized_md_fmac2_gemv3(void) { UT_RETURN_ASSERT(test_optimized_md_fmac2_flags(UINT64_C(14), UINT64_C(13), UINT64_C(7), true, 1.e-6)); }
static bool test_optimized_md_fmac2_gemm(void) { UT_RETURN_ASSERT(test_optimized_md_fmac2_flags(UINT64_C(3), UINT64_C(6), UINT64_C(5), true, 2.e-6)); }
static bool test_optimized_md_fmac2_gemm2(void) { UT_RETURN_ASSERT(test_optimized_md_fmac2_flags(UINT64_C(11), UINT64_C(14), UINT64_C(13), true, 1.e-6));}
static bool test_optimized_md_fmac2_ger(void) { UT_RETURN_ASSERT(test_optimized_md_fmac2_flags(UINT64_C(3), UINT64_C(1), UINT64_C(2), true, 2.e-6)); }
static bool test_optimized_md_fmac2_ger2(void) { UT_RETURN_ASSERT(test_optimized_md_fmac2_flags(UINT64_C(7), UINT64_C(5), UINT64_C(6), true, 1.e-6)); }
static bool test_optimized_md_fmac2_axpy(void) { UT_RETURN_ASSERT(test_optimized_md_fmac2_flags(UINT64_C(1), UINT64_C(1), UINT64_C(0), true, 2.e-6)); }
static bool test_optimized_md_fmac2_axpy2(void) { UT_RETURN_ASSERT(test_optimized_md_fmac2_flags(UINT64_C(3), UINT64_C(2), UINT64_C(3), true, 1.e-6));}

UT_GPU_REGISTER_TEST(test_optimized_md_fmac2_dot);
UT_GPU_REGISTER_TEST(test_optimized_md_fmac2_dot2);
UT_GPU_REGISTER_TEST(test_optimized_md_fmac2_gemv);
UT_GPU_REGISTER_TEST(test_optimized_md_fmac2_gemv2);
UT_GPU_REGISTER_TEST(test_optimized_md_fmac2_gemv3);
UT_GPU_REGISTER_TEST(test_optimized_md_fmac2_gemm);
UT_GPU_REGISTER_TEST(test_optimized_md_fmac2_gemm2);
UT_GPU_REGISTER_TEST(test_optimized_md_fmac2_ger);
UT_GPU_REGISTER_TEST(test_optimized_md_fmac2_ger2);
UT_GPU_REGISTER_TEST(test_optimized_md_fmac2_axpy);
UT_GPU_REGISTER_TEST(test_optimized_md_fmac2_axpy2);

static bool test_optimized_md_zmul2_flags(bart_flags_t out_flag, bart_flags_t in1_flag, bart_flags_t in2_flag, bool optimization_expected, float err_tol)
{
#ifndef USE_CUDA
	UNUSED(out_flag);
	UNUSED(in1_flag);
	UNUSED(in2_flag);
	UNUSED(optimization_expected);
	UNUSED(err_val);
	return true;
#else

	enum {D = 5};
	bart_dim_t dims[D] = {3, 5, 2, 4, 4};

	size_t size = CFL_SIZE;

	bart_dim_t odims[D];
	bart_dim_t idims1[D];
	bart_dim_t idims2[D];
	md_select_dims(D, out_flag, odims, dims);
	md_select_dims(D, in1_flag, idims1, dims);
	md_select_dims(D, in2_flag, idims2, dims);

	bart_stride_t ostr[D];
	bart_stride_t istr1[D];
	bart_stride_t istr2[D];
	md_calc_strides(D, ostr, odims, size);
	md_calc_strides(D, istr1, idims1, size);
	md_calc_strides(D, istr2, idims2, size);

	complex float* optr1 = md_alloc_gpu(D, odims, size);
	complex float* optr2 = md_alloc_gpu(D, odims, size);
	complex float* iptr1 = md_alloc_gpu(D, idims1, size);
	complex float* iptr2 = md_alloc_gpu(D, idims2, size);

	md_gaussian_rand(D, idims1, iptr1);
	md_gaussian_rand(D, idims2, iptr2);

	deactivate_strided_vecops();
	md_zmul2(D, dims, ostr, optr1, istr1, iptr1, istr2, iptr2);
	activate_strided_vecops();
	bool result = (optimization_expected == simple_zmul(D, dims, ostr, optr2, istr1, iptr1, istr2, iptr2));
	debug_printf(DP_DEBUG1, "%d %.10f\n", optimization_expected, md_znrmse(D, odims, optr1, optr2));
	result &= (!optimization_expected) || (err_tol > md_znrmse(D, odims, optr1, optr2));
	debug_printf(result ? DP_DEBUG1 : DP_INFO, "%e\n", md_znrmse(D, odims, optr1, optr2));
	md_free(optr1);
	md_free(optr2);
	md_free(iptr1);
	md_free(iptr2);

	return result;
#endif
}

static bool test_optimized_md_zmul2_smul(void) { UT_RETURN_ASSERT(test_optimized_md_zmul2_flags(~UINT64_C(0), UINT64_C(1), UINT64_C(0), true, 1.e-7)); }
static bool test_optimized_md_zmul2_smul2(void) { UT_RETURN_ASSERT(test_optimized_md_zmul2_flags(~UINT64_C(0), UINT64_C(2), UINT64_C(3), true, 1.e-7)); } // also dgmm on gpu
static bool test_optimized_md_zmul2_dgmm(void) { UT_RETURN_ASSERT(test_optimized_md_zmul2_flags(~UINT64_C(0), UINT64_C(1), UINT64_C(3), true, 1.e-7)); } // only on gpu
static bool test_optimized_md_zmul2_ger(void) { UT_RETURN_ASSERT(test_optimized_md_zmul2_flags(~UINT64_C(0), UINT64_C(1), UINT64_C(2), true, 2.e-7)); }
static bool test_optimized_md_zmul2_ger2(void) { UT_RETURN_ASSERT(test_optimized_md_zmul2_flags(~UINT64_C(0), UINT64_C(5), UINT64_C(6), true, 1.e-7)); }

UT_GPU_REGISTER_TEST(test_optimized_md_zmul2_smul);
UT_GPU_REGISTER_TEST(test_optimized_md_zmul2_smul2);
UT_GPU_REGISTER_TEST(test_optimized_md_zmul2_dgmm);
UT_GPU_REGISTER_TEST(test_optimized_md_zmul2_ger);
UT_GPU_REGISTER_TEST(test_optimized_md_zmul2_ger2);

static bool test_optimized_md_zmulc2_flags(bart_flags_t out_flag, bart_flags_t in1_flag, bart_flags_t in2_flag, bool optimization_expected, float err_tol)
{
#ifndef USE_CUDA
	UNUSED(out_flag);
	UNUSED(in1_flag);
	UNUSED(in2_flag);
	UNUSED(optimization_expected);
	UNUSED(err_val);
	return true;
#else

	enum {D = 5};
	bart_dim_t dims[D] = {3, 5, 2, 4, 4};

	size_t size = CFL_SIZE;

	bart_dim_t odims[D];
	bart_dim_t idims1[D];
	bart_dim_t idims2[D];
	md_select_dims(D, out_flag, odims, dims);
	md_select_dims(D, in1_flag, idims1, dims);
	md_select_dims(D, in2_flag, idims2, dims);

	bart_stride_t ostr[D];
	bart_stride_t istr1[D];
	bart_stride_t istr2[D];
	md_calc_strides(D, ostr, odims, size);
	md_calc_strides(D, istr1, idims1, size);
	md_calc_strides(D, istr2, idims2, size);

	complex float* optr1 = md_alloc_gpu(D, odims, size);
	complex float* optr2 = md_alloc_gpu(D, odims, size);
	complex float* iptr1 = md_alloc_gpu(D, idims1, size);
	complex float* iptr2 = md_alloc_gpu(D, idims2, size);

	md_gaussian_rand(D, idims1, iptr1);
	md_gaussian_rand(D, idims2, iptr2);

	deactivate_strided_vecops();
	md_zmulc2(D, dims, ostr, optr1, istr1, iptr1, istr2, iptr2);
	activate_strided_vecops();
	bool result = (optimization_expected == simple_zmulc(D, dims, ostr, optr2, istr1, iptr1, istr2, iptr2));
	debug_printf(DP_DEBUG1, "%d %.10f\n", optimization_expected, md_znrmse(D, odims, optr1, optr2));
	result &= (!optimization_expected) || (err_tol > md_znrmse(D, odims, optr1, optr2));
	debug_printf(result ? DP_DEBUG1 : DP_INFO, "%e\n", md_znrmse(D, odims, optr1, optr2));
	md_free(optr1);
	md_free(optr2);
	md_free(iptr1);
	md_free(iptr2);

	return result;
#endif
}

static bool test_optimized_md_zmulc2_smul(void) { UT_RETURN_ASSERT(test_optimized_md_zmulc2_flags(~UINT64_C(0), UINT64_C(1), UINT64_C(0), true, 1.e-7)); }
static bool test_optimized_md_zmulc2_smul2(void) { UT_RETURN_ASSERT(test_optimized_md_zmulc2_flags(~UINT64_C(0), UINT64_C(2), UINT64_C(3), true, 1.e-7)); } // also dgmm on gpu
static bool test_optimized_md_zmulc2_dgmm(void) { UT_RETURN_ASSERT(test_optimized_md_zmulc2_flags(~UINT64_C(0), UINT64_C(1), UINT64_C(3), true, 1.e-7)); } // only on gpu
static bool test_optimized_md_zmulc2_ger(void) { UT_RETURN_ASSERT(test_optimized_md_zmulc2_flags(~UINT64_C(0), UINT64_C(1), UINT64_C(2), true, 2.e-7)); }
static bool test_optimized_md_zmulc2_ger2(void) { UT_RETURN_ASSERT(test_optimized_md_zmulc2_flags(~UINT64_C(0), UINT64_C(5), UINT64_C(6), true, 1.e-7)); }

UT_GPU_REGISTER_TEST(test_optimized_md_zmulc2_smul);
UT_GPU_REGISTER_TEST(test_optimized_md_zmulc2_smul2);
UT_GPU_REGISTER_TEST(test_optimized_md_zmulc2_dgmm);
UT_GPU_REGISTER_TEST(test_optimized_md_zmulc2_ger);
UT_GPU_REGISTER_TEST(test_optimized_md_zmulc2_ger2);

static bool test_optimized_md_mul2_flags(bart_flags_t out_flag, bart_flags_t in1_flag, bart_flags_t in2_flag, bool optimization_expected, float err_tol)
{
#ifndef USE_CUDA
	UNUSED(out_flag);
	UNUSED(in1_flag);
	UNUSED(in2_flag);
	UNUSED(optimization_expected);
	UNUSED(err_val);
	return true;
#else

	enum {D = 5};
	bart_dim_t dims[D] = {3, 5, 2, 4, 4};

	size_t size = FL_SIZE;

	bart_dim_t odims[D];
	bart_dim_t idims1[D];
	bart_dim_t idims2[D];
	md_select_dims(D, out_flag, odims, dims);
	md_select_dims(D, in1_flag, idims1, dims);
	md_select_dims(D, in2_flag, idims2, dims);

	bart_stride_t ostr[D];
	bart_stride_t istr1[D];
	bart_stride_t istr2[D];
	md_calc_strides(D, ostr, odims, size);
	md_calc_strides(D, istr1, idims1, size);
	md_calc_strides(D, istr2, idims2, size);

	float* optr1 = md_alloc_gpu(D, odims, CFL_SIZE);
	float* optr2 = md_alloc_gpu(D, odims, CFL_SIZE);
	float* iptr1 = md_alloc_gpu(D, idims1, CFL_SIZE);
	float* iptr2 = md_alloc_gpu(D, idims2, CFL_SIZE);

	md_gaussian_rand(D, idims1, (complex float*)iptr1);
	md_gaussian_rand(D, idims2, (complex float*)iptr2);
	md_clear(D, odims, optr1, size);
	md_clear(D, odims, optr2, size);

	deactivate_strided_vecops();
	md_mul2(D, dims, ostr, optr1, istr1, iptr1, istr2, iptr2);
	activate_strided_vecops();
	bool result = (optimization_expected == simple_mul(D, dims, ostr, optr2, istr1, iptr1, istr2, iptr2));
	debug_printf(DP_DEBUG1, "%d %.10f %.10f\n", optimization_expected, md_nrmse(D, odims, optr1, optr2), md_rms(D, odims, optr1));
	result &= (!optimization_expected) || (err_tol > md_nrmse(D, odims, optr1, optr2));
	debug_printf(result ? DP_DEBUG1 : DP_INFO, "%e\n", md_nrmse(D, odims, optr1, optr2));
	md_free(optr1);
	md_free(optr2);
	md_free(iptr1);
	md_free(iptr2);

	return result;
#endif
}

static bool test_optimized_md_mul2_smul(void) { UT_RETURN_ASSERT(test_optimized_md_mul2_flags(~UINT64_C(0), UINT64_C(1), UINT64_C(0), true, 1.e-8)); }
static bool test_optimized_md_mul2_smul2(void) { UT_RETURN_ASSERT(test_optimized_md_mul2_flags(~UINT64_C(0), UINT64_C(2), UINT64_C(3), true, 1.e-8)); } // also dgmm on gpu
static bool test_optimized_md_mul2_dgmm(void) { UT_RETURN_ASSERT(test_optimized_md_mul2_flags(~UINT64_C(0), UINT64_C(1), UINT64_C(3), true, 1.e-8)); } // only on gpu
static bool test_optimized_md_mul2_ger(void) { UT_RETURN_ASSERT(test_optimized_md_mul2_flags(~UINT64_C(0), UINT64_C(1), UINT64_C(2), true, 1.e-8)); }
static bool test_optimized_md_mul2_ger2(void) { UT_RETURN_ASSERT(test_optimized_md_mul2_flags(~UINT64_C(0), UINT64_C(5), UINT64_C(6), true, 1.e-8)); }

UT_GPU_REGISTER_TEST(test_optimized_md_mul2_smul);
UT_GPU_REGISTER_TEST(test_optimized_md_mul2_smul2);
UT_GPU_REGISTER_TEST(test_optimized_md_mul2_dgmm);
UT_GPU_REGISTER_TEST(test_optimized_md_mul2_ger);
UT_GPU_REGISTER_TEST(test_optimized_md_mul2_ger2);

static bool test_optimized_md_zadd(bart_flags_t out_flag, bart_flags_t in1_flag, bart_flags_t in2_flag, bool in1_same, bool in2_same, bool optimization_expected, float err_tol)
{
#ifndef USE_CUDA
	UNUSED(out_flag);
	UNUSED(in1_flag);
	UNUSED(in2_flag);
	UNUSED(optimization_expected);
	UNUSED(err_val);
	UNUSED(in1_same);
	UNUSED(in2_same);
	return true;
#else

	enum {D = 5};
	bart_dim_t dims[D] = {3, 32, 7, 13, 3};
	md_select_dims(D, out_flag | in1_flag | in2_flag, dims, dims);

	size_t size = CFL_SIZE;

	bart_dim_t odims[D];
	bart_dim_t idims1[D];
	bart_dim_t idims2[D];
	md_select_dims(D, out_flag, odims, dims);
	md_select_dims(D, in1_flag, idims1, dims);
	md_select_dims(D, in2_flag, idims2, dims);

	bart_stride_t ostr[D];
	bart_stride_t istr1[D];
	bart_stride_t istr2[D];
	md_calc_strides(D, ostr, odims, size);
	md_calc_strides(D, istr1, idims1, size);
	md_calc_strides(D, istr2, idims2, size);

	complex float* optr1 = md_alloc_gpu(D, odims, size);
	complex float* optr2 = md_alloc_gpu(D, odims, size);
	complex float* iptr1 = md_alloc_gpu(D, idims1, size);
	complex float* iptr2 = md_alloc_gpu(D, idims2, size);

	md_gaussian_rand(D, idims1, iptr1);
	md_gaussian_rand(D, idims2, iptr2);
	md_gaussian_rand(D, odims, optr1);
	md_copy(D, odims, optr2, optr1, size);

	assert(!in1_same || (md_calc_size(D, odims) == md_calc_size(D, idims1)));
	assert(!in2_same || (md_calc_size(D, odims) == md_calc_size(D, idims2)));

	deactivate_strided_vecops();
	md_zadd2(D, dims, ostr, optr1, istr1, !in1_same ? iptr1 : optr1, istr2, !in2_same ? iptr2 : optr1);
	activate_strided_vecops();
	bool result = (optimization_expected == simple_zadd(D, dims, ostr, optr2, istr1, !in1_same ? iptr1 : optr2, istr2, !in2_same ? iptr2 : optr2));
	debug_printf(DP_DEBUG1, "%d %.10f\n", optimization_expected, md_znrmse(D, odims, optr1, optr2));
	result &= (!optimization_expected) || (err_tol > md_znrmse(D, odims, optr1, optr2));
	debug_printf(result ? DP_DEBUG1 : DP_INFO, "%e\n", md_znrmse(D, odims, optr1, optr2));
	md_free(optr1);
	md_free(optr2);
	md_free(iptr1);
	md_free(iptr2);

	return result;
#endif
}

static bool test_optimized_md_zadd2_reduce_inner1(void) { UT_RETURN_ASSERT(test_optimized_md_zadd(~(UINT64_C(1)+UINT64_C(4)), ~(UINT64_C(1)+UINT64_C(4)), ~UINT64_C(0), true, false, true, 1.e-6)); }
static bool test_optimized_md_zadd2_reduce_inner2(void) { UT_RETURN_ASSERT(test_optimized_md_zadd(~(UINT64_C(1)+UINT64_C(4)), ~(UINT64_C(1)+UINT64_C(4)), ~UINT64_C(0), false, false, false, 1.e-6)); }
static bool test_optimized_md_zadd2_reduce_inner4(void) { UT_RETURN_ASSERT(test_optimized_md_zadd(~(UINT64_C(1)+UINT64_C(2)), ~UINT64_C(4), ~(UINT64_C(1) + UINT64_C(2)), false, true, true, 1.e-6)); }
static bool test_optimized_md_zadd2_reduce_inner5(void) { UT_RETURN_ASSERT(test_optimized_md_zadd(UINT64_C(0), ~UINT64_C(4), UINT64_C(0), false, true, true, 3.e-6)); }
UT_GPU_REGISTER_TEST(test_optimized_md_zadd2_reduce_inner1);
UT_GPU_REGISTER_TEST(test_optimized_md_zadd2_reduce_inner2);
UT_GPU_REGISTER_TEST(test_optimized_md_zadd2_reduce_inner4);
UT_GPU_REGISTER_TEST(test_optimized_md_zadd2_reduce_inner5);

static bool test_optimized_md_zadd2_reduce_outer1(void) { UT_RETURN_ASSERT(test_optimized_md_zadd(~(UINT64_C(4)), ~(UINT64_C(4)), ~UINT64_C(0), true, false, true, 1.e-6)); }
static bool test_optimized_md_zadd2_reduce_outer2(void) { UT_RETURN_ASSERT(test_optimized_md_zadd(~(UINT64_C(2)), ~(UINT64_C(2)+UINT64_C(4)), ~UINT64_C(0), false, false, false, 1.e-6)); }
static bool test_optimized_md_zadd2_reduce_outer4(void) { UT_RETURN_ASSERT(test_optimized_md_zadd(~(UINT64_C(4)), ~(UINT64_C(8)), ~(UINT64_C(4)), false, true, true, 1.e-6)); }
UT_GPU_REGISTER_TEST(test_optimized_md_zadd2_reduce_outer1);
UT_GPU_REGISTER_TEST(test_optimized_md_zadd2_reduce_outer2);
UT_GPU_REGISTER_TEST(test_optimized_md_zadd2_reduce_outer4);

static bool test_optimized_md_add(bart_flags_t out_flag, bart_flags_t in1_flag, bart_flags_t in2_flag, bool in1_same, bool in2_same, bool optimization_expected, float err_tol)
{
#ifndef USE_CUDA
	UNUSED(out_flag);
	UNUSED(in1_flag);
	UNUSED(in2_flag);
	UNUSED(optimization_expected);
	UNUSED(err_val);
	UNUSED(in1_same);
	UNUSED(in2_same);
	return true;
#else


	enum {D = 5};
	bart_dim_t dims[D] = {3, 32, 7, 13, 3};
	md_select_dims(D, out_flag | in1_flag | in2_flag, dims, dims);

	size_t size = FL_SIZE;

	bart_dim_t odims[D];
	bart_dim_t idims1[D];
	bart_dim_t idims2[D];
	md_select_dims(D, out_flag, odims, dims);
	md_select_dims(D, in1_flag, idims1, dims);
	md_select_dims(D, in2_flag, idims2, dims);

	bart_stride_t ostr[D];
	bart_stride_t istr1[D];
	bart_stride_t istr2[D];
	md_calc_strides(D, ostr, odims, size);
	md_calc_strides(D, istr1, idims1, size);
	md_calc_strides(D, istr2, idims2, size);

	float* optr1 = md_alloc_gpu(D, odims, 2 * size);
	float* optr2 = md_alloc_gpu(D, odims, 2 * size);
	float* iptr1 = md_alloc_gpu(D, idims1, 2 * size);
	float* iptr2 = md_alloc_gpu(D, idims2, 2 * size);

	md_gaussian_rand(D, idims1, (complex float*)iptr1);
	md_gaussian_rand(D, idims2, (complex float*)iptr2);
	md_gaussian_rand(D, odims, (complex float*)optr1);
	md_copy(D, odims, optr2, optr1, size);

	assert(!in1_same || (md_calc_size(D, odims) == md_calc_size(D, idims1)));
	assert(!in2_same || (md_calc_size(D, odims) == md_calc_size(D, idims2)));

	deactivate_strided_vecops();
	md_add2(D, dims, ostr, optr1, istr1, !in1_same ? iptr1 : optr1, istr2, !in2_same ? iptr2 : optr1);
	activate_strided_vecops();
	bool result = (optimization_expected == simple_add(D, dims, ostr, optr2, istr1, !in1_same ? iptr1 : optr2, istr2, !in2_same ? iptr2 : optr2));
	debug_printf(DP_DEBUG1, "%d %d %.10f\n", result, optimization_expected, md_nrmse(D, odims, optr1, optr2));
	result &= (!optimization_expected) || (err_tol > md_nrmse(D, odims, optr1, optr2));
	debug_printf(result ? DP_DEBUG1 : DP_INFO, "%e\n", md_nrmse(D, odims, optr1, optr2));
	md_free(optr1);
	md_free(optr2);
	md_free(iptr1);
	md_free(iptr2);

	return result;
#endif
}

static bool test_optimized_md_add2_reduce_inner1(void) { UT_RETURN_ASSERT(test_optimized_md_add(~(UINT64_C(1)+UINT64_C(4)), ~(UINT64_C(1)+UINT64_C(4)), ~UINT64_C(0), true, false, true, 1.e-6)); }
static bool test_optimized_md_add2_reduce_inner2(void) { UT_RETURN_ASSERT(test_optimized_md_add(~(UINT64_C(1)+UINT64_C(4)), ~(UINT64_C(1)+UINT64_C(4)), ~UINT64_C(0), false, false, false, 1.e-6)); }
static bool test_optimized_md_add2_reduce_inner4(void) { UT_RETURN_ASSERT(test_optimized_md_add(~(UINT64_C(1)+UINT64_C(2)), ~UINT64_C(4), ~(UINT64_C(1) + UINT64_C(2)), false, true, true, 1.e-6)); }
static bool test_optimized_md_add2_reduce_inner5(void) { UT_RETURN_ASSERT(test_optimized_md_add(UINT64_C(0), ~UINT64_C(4), UINT64_C(0), false, true, true, 3.e-5)); }
UT_GPU_REGISTER_TEST(test_optimized_md_add2_reduce_inner1);
UT_GPU_REGISTER_TEST(test_optimized_md_add2_reduce_inner2);
UT_GPU_REGISTER_TEST(test_optimized_md_add2_reduce_inner4);
UT_GPU_REGISTER_TEST(test_optimized_md_add2_reduce_inner5);

static bool test_optimized_md_add2_reduce_outer1(void) { UT_RETURN_ASSERT(test_optimized_md_add(~(UINT64_C(4)), ~(UINT64_C(4)), ~UINT64_C(0), true, false, true, 1.e-6)); }
static bool test_optimized_md_add2_reduce_outer2(void) { UT_RETURN_ASSERT(test_optimized_md_add(~(UINT64_C(2)), ~(UINT64_C(2)+UINT64_C(4)), ~UINT64_C(0), false, false, false, 1.e-6)); }
static bool test_optimized_md_add2_reduce_outer4(void) { UT_RETURN_ASSERT(test_optimized_md_add(~(UINT64_C(4)), ~(UINT64_C(8)), ~(UINT64_C(4)), false, true, true, 1.e-6)); }
UT_GPU_REGISTER_TEST(test_optimized_md_add2_reduce_outer1);
UT_GPU_REGISTER_TEST(test_optimized_md_add2_reduce_outer2);
UT_GPU_REGISTER_TEST(test_optimized_md_add2_reduce_outer4);

static bool test_optimized_md_zmax(bart_flags_t out_flag, bart_flags_t in1_flag, bart_flags_t in2_flag, bool in1_same, bool in2_same, bool optimization_expected, float err_tol)
{
#ifndef USE_CUDA
	UNUSED(out_flag);
	UNUSED(in1_flag);
	UNUSED(in2_flag);
	UNUSED(optimization_expected);
	UNUSED(err_val);
	UNUSED(in1_same);
	UNUSED(in2_same);
	return true;
#else

	enum {D = 5};
	bart_dim_t dims[D] = {3, 32, 7, 13, 3};
	md_select_dims(D, out_flag | in1_flag | in2_flag, dims, dims);

	size_t size = CFL_SIZE;

	bart_dim_t odims[D];
	bart_dim_t idims1[D];
	bart_dim_t idims2[D];
	md_select_dims(D, out_flag, odims, dims);
	md_select_dims(D, in1_flag, idims1, dims);
	md_select_dims(D, in2_flag, idims2, dims);

	bart_stride_t ostr[D];
	bart_stride_t istr1[D];
	bart_stride_t istr2[D];
	md_calc_strides(D, ostr, odims, size);
	md_calc_strides(D, istr1, idims1, size);
	md_calc_strides(D, istr2, idims2, size);

	complex float* optr1 = md_alloc_gpu(D, odims, size);
	complex float* optr2 = md_alloc_gpu(D, odims, size);
	complex float* iptr1 = md_alloc_gpu(D, idims1, size);
	complex float* iptr2 = md_alloc_gpu(D, idims2, size);

	md_gaussian_rand(D, idims1, iptr1);
	md_gaussian_rand(D, idims2, iptr2);
	md_gaussian_rand(D, odims, optr1);
	md_copy(D, odims, optr2, optr1, size);

	assert(!in1_same || (md_calc_size(D, odims) == md_calc_size(D, idims1)));
	assert(!in2_same || (md_calc_size(D, odims) == md_calc_size(D, idims2)));

	deactivate_strided_vecops();
	md_zmax2(D, dims, ostr, optr1, istr1, !in1_same ? iptr1 : optr1, istr2, !in2_same ? iptr2 : optr1);
	activate_strided_vecops();
	bool result = (optimization_expected == simple_zmax(D, dims, ostr, optr2, istr1, !in1_same ? iptr1 : optr2, istr2, !in2_same ? iptr2 : optr2));
	debug_printf(DP_DEBUG1, "%d %.10f\n", optimization_expected, md_znrmse(D, odims, optr1, optr2));
	result &= (!optimization_expected) || (err_tol > md_znrmse(D, odims, optr1, optr2));
	debug_printf(result ? DP_DEBUG1 : DP_INFO, "%e\n", md_znrmse(D, odims, optr1, optr2));
	md_free(optr1);
	md_free(optr2);
	md_free(iptr1);
	md_free(iptr2);

	return result;
#endif
}

static bool test_optimized_md_zmax2_reduce_inner1(void) { UT_RETURN_ASSERT(test_optimized_md_zmax(~(UINT64_C(1)+UINT64_C(4)), ~(UINT64_C(1)+UINT64_C(4)), ~UINT64_C(0), true, false, true, 1.e-8)); }
static bool test_optimized_md_zmax2_reduce_inner2(void) { UT_RETURN_ASSERT(test_optimized_md_zmax(~(UINT64_C(1)+UINT64_C(4)), ~(UINT64_C(1)+UINT64_C(4)), ~UINT64_C(0), false, false, false, 1.e-8)); }
static bool test_optimized_md_zmax2_reduce_inner4(void) { UT_RETURN_ASSERT(test_optimized_md_zmax(~(UINT64_C(1)+UINT64_C(2)), ~UINT64_C(4), ~(UINT64_C(1) + UINT64_C(2)), false, true, true, 1.e-8)); }
static bool test_optimized_md_zmax2_reduce_inner5(void) { UT_RETURN_ASSERT(test_optimized_md_zmax(UINT64_C(0), ~UINT64_C(4), UINT64_C(0), false, true, true, 2.e-8)); }
UT_GPU_REGISTER_TEST(test_optimized_md_zmax2_reduce_inner1);
UT_GPU_REGISTER_TEST(test_optimized_md_zmax2_reduce_inner2);
UT_GPU_REGISTER_TEST(test_optimized_md_zmax2_reduce_inner4);
UT_GPU_REGISTER_TEST(test_optimized_md_zmax2_reduce_inner5);

static bool test_optimized_md_zmax2_reduce_outer1(void) { UT_RETURN_ASSERT(test_optimized_md_zmax(~(UINT64_C(4)), ~(UINT64_C(4)), ~UINT64_C(0), true, false, true, 1.e-8)); }
static bool test_optimized_md_zmax2_reduce_outer2(void) { UT_RETURN_ASSERT(test_optimized_md_zmax(~(UINT64_C(2)), ~(UINT64_C(2)+UINT64_C(4)), ~UINT64_C(0), false, false, false, 1.e-8)); }
static bool test_optimized_md_zmax2_reduce_outer4(void) { UT_RETURN_ASSERT(test_optimized_md_zmax(~(UINT64_C(4)), ~(UINT64_C(8)), ~(UINT64_C(4)), false, true, true, 1.e-8)); }
UT_GPU_REGISTER_TEST(test_optimized_md_zmax2_reduce_outer1);
UT_GPU_REGISTER_TEST(test_optimized_md_zmax2_reduce_outer2);
UT_GPU_REGISTER_TEST(test_optimized_md_zmax2_reduce_outer4);
