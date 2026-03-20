
#include <complex.h>
#include <math.h>

#include "misc/debug.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/laplace.h"
#include "num/filter.h"
#include "num/fft.h"

#include "utest.h"



static bool test_laplace_fd(void)
{
	enum { N = 1 };
	long dims_in[N] = { 127 };

	complex float* in = md_alloc(N, dims_in, CFL_SIZE);

	float scale = 2 * M_PI / (float)dims_in[0];

	for (int i = 0; i < dims_in[0]; i++)
		in[i] = cos(scale * i);

	complex float* out = md_alloc(N, dims_in, CFL_SIZE);
	md_laplace_fd(N, dims_in, ~0UL, out, in);

	for (int i = 0; i < dims_in[0]; i++)
		in[i] = -1. * scale * scale * cos(scale * i);

	float err = md_znrmse(N, dims_in, out, in);
	md_free(in);
	md_free(out);

	UT_RETURN_ASSERT(0.0005 > err);
}

UT_REGISTER_TEST(test_laplace_fd);



static bool test_laplace_fd_wrapped_phase(void)
{
	enum { N = 1 };
	long dims_in[N] = { 256 };

	complex float* in = md_alloc(N, dims_in, CFL_SIZE);

	float scale = 2 * M_PI / (float)dims_in[0];

	for (int i = 0; i < dims_in[0]; i++) {

		in[i] = 4 * cos(scale * i);
		in[i] = cargf(cexpf(1.i * in[i]));
	}

	complex float* out = md_alloc(N, dims_in, CFL_SIZE);
	md_laplace_fd_wrapped_phase(N, dims_in, ~0UL, out, in);

	for (int i = 0; i < dims_in[0]; i++)
		in[i] = -4. * scale * scale * cos(scale * i);

	float err = md_znrmse(N, dims_in, out, in);
	md_free(in);
	md_free(out);

	UT_RETURN_ASSERT(0.0005 > err);
}

UT_REGISTER_TEST(test_laplace_fd_wrapped_phase);



static bool test_laplace_fd_wrapped_phase_exp(void)
{
	enum { N = 1 };
	long dims_in[N] = { 256 };

	complex float* in = md_alloc(N, dims_in, CFL_SIZE);

	float scale = 2 * M_PI / (float)dims_in[0];

	for (int i = 0; i < dims_in[0]; i++) {

		in[i] = 4 * cos(scale * i);
		in[i] = cargf(cexpf(1.i * in[i]));
	}

	complex float* out = md_alloc(N, dims_in, CFL_SIZE);
	md_laplace_fd_wrapped_phase_exp(N, dims_in, ~0UL, out, in);

	for (int i = 0; i < dims_in[0]; i++)
		in[i] = -4. * scale * scale * cos(scale * i);

	float err = md_znrmse(N, dims_in, out, in);
	md_free(in);
	md_free(out);

	UT_RETURN_ASSERT(0.003 > err);
}

UT_REGISTER_TEST(test_laplace_fd_wrapped_phase_exp);


static bool test_klaplace_filter(void)
{
	enum { N = 1 };
	long dims_in[N] = { 127 };

	complex float* in = md_alloc(N, dims_in, CFL_SIZE);

	float scale = 2 * M_PI / (float)dims_in[0];

	for (int i = 0; i < dims_in[0]; i++)
		in[i] = cos(scale * i);

	complex float* out = md_alloc(N, dims_in, CFL_SIZE);
	fftuc(N, dims_in, 1, out, in);

	complex float* filter = md_alloc(N, dims_in, CFL_SIZE);
	klaplace(N, dims_in, 1UL, filter);

	// WARNING: compensate wrong scaling of klaplace:
	md_zsmul(N, dims_in, filter, filter, -powf(2 * M_PI, 2));

	md_zmul(N, dims_in, out, out, filter);
	md_free(filter);

	ifftuc(N, dims_in, 1, out, out);

	for (int i = 0; i < dims_in[0]; i++)
		in[i] = -1. * scale * scale * cos(scale * i);

	float err = md_znrmse(N, dims_in, out, in);
	md_free(in);
	md_free(out);

	UT_RETURN_ASSERT(0.0005 > err);
}

UT_REGISTER_TEST(test_klaplace_filter);


static bool test_klaplace_fd_filter(void)
{
	enum { N = 1 };
	long dims_in[N] = { 127 };

	complex float* in = md_alloc(N, dims_in, CFL_SIZE);

	float scale = 2 * M_PI / (float)dims_in[0];

	for (int i = 0; i < dims_in[0]; i++)
		in[i] = cos(scale * i);

	complex float* out = md_alloc(N, dims_in, CFL_SIZE);
	fftu(N, dims_in, 1, out, in);

	complex float* filter = md_alloc(N, dims_in, CFL_SIZE);
	klaplace_fd_uncentered(N, dims_in, filter);

	md_zmul(N, dims_in, out, out, filter);
	md_free(filter);

	ifftu(N, dims_in, 1, out, out);

	for (int i = 0; i < dims_in[0]; i++)
		in[i] = -1. * scale * scale * cos(scale * i);

	float err = md_znrmse(N, dims_in, out, in);
	md_free(in);
	md_free(out);

	UT_RETURN_ASSERT(0.0005 > err);
}

UT_REGISTER_TEST(test_klaplace_fd_filter);


