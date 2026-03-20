
#include <complex.h>
#include <math.h>

#include "misc/debug.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/laplace.h"

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

