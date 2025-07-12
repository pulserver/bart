/* Copyright 2018-2021. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2021-2023. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>

#include "misc/debug.h"
#include "misc/list.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "seq/pulse.h"

#include "simu/bloch.h"
#include "simu/simulation.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/const.h"
#include "nsimu/nlop_seq.h"
#include "nlops/nltest.h"
#include "nlops/someops.h"

#include "utest.h"

static const struct nlop_s* nlop_set_input_real(const struct nlop_s* nlop, int i)
{
	const struct iovec_s* io = nlop_generic_domain(nlop, i);
	return nlop_prepend_FF(nlop_from_linop_F(linop_zreal_create(io->N, io->dims)), nlop, i);
}

static const struct nlop_s* nlop_set_output_real(const struct nlop_s* nlop, int i)
{
	const struct iovec_s* io = nlop_generic_codomain(nlop, i);
	return nlop_append_FF(nlop, i, nlop_from_linop_F(linop_zreal_create(io->N, io->dims)));
}




static bool test_nlop_pulse(void)
{
	struct pulse_sinc ps = pulse_sinc_defaults;
	pulse_sinc_init(&ps, 0.001, 90., 0., 4., ps.alpha);

	long dims[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	struct sim_config_s sim = sim_config_default_cpu;
	sim_config_set_dims(&sim, N, dims, 1);

	float r1 = 0.;
	float r2 = 0.;
	float b1 = 1.;
	float b0 = 0.;

	complex float mag[] = { 0., 0., 1. };
	complex float par[] = { r1, r2, b1, b0 };

	const struct nlop_s* nlop = nlop_pulse_create(sim, CAST_UP(&ps), 0., (float[3]) { 0., 0., 0. });
	nlop = nlop_set_input_real(nlop, 0);
	nlop = nlop_set_input_real(nlop, 1);
	nlop = nlop_set_input_const_F(nlop, 1, N, sim.pdims, false, par);

	complex float out[3];
	nlop_generic_apply_unchecked(nlop, 2, (void*[2]) { out, mag });
	nlop_free(nlop);

	complex float ref[] = { 0., 1., 0. };

	UT_RETURN_ASSERT(1.e-5 > md_zrmse(N, sim.mdims, ref, out));
}

UT_REGISTER_TEST(test_nlop_pulse);


static bool test_nlop_hard_pulse(void)
{
	struct pulse_sinc ps = pulse_sinc_defaults;
	pulse_sinc_init(&ps, 0.001, 45., 0., 4., ps.alpha);

	long dims[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	struct sim_config_s sim = sim_config_default_cpu;
	sim_config_set_dims(&sim, N, dims, 1);

	float r1 = 0.;
	float r2 = 0.;
	float b1 = 1.25;
	float b0 = 0.;

	complex float par[] = { r1, r2, b1, b0 };

	const struct nlop_s* nlop1 = nlop_pulse_create(sim, CAST_UP(&ps), M_PI / 4., (float[3]) { 0., 0., 0. });
	const struct nlop_s* nlop2 = nlop_hard_pulse_create(sim, true, DEG2RAD(45), M_PI / 4.);

	nlop1 = nlop_set_input_real(nlop1, 0);
	nlop1 = nlop_set_input_const_F(nlop1, 1, N, sim.pdims, false, par);

	nlop2 = nlop_set_input_real(nlop2, 0);
	nlop2 = nlop_set_input_const_F(nlop2, 1, N, sim.pdims, false, par);

	bool ok = true;

	for (int i = 0; i < 10; i++)
		ok = ok && compare_nlops(nlop1, nlop2, true, true, true, 1.e-4);

	nlop_free(nlop1);
	nlop_free(nlop2);

	UT_RETURN_ASSERT(ok);
}

UT_REGISTER_TEST(test_nlop_hard_pulse);

static bool test_nlop_hard_pulse2(void)
{
	long dims[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	struct sim_config_s sim = sim_config_default_cpu;
	sim_config_set_dims(&sim, N, dims, 1);

	float r1 = 0.;
	float r2 = 0.;
	float b1 = 1.;
	float b0 = 0.;

	complex float par[] = { r1, r2, b1, b0 };

	const struct nlop_s* nlop1 = nlop_hard_pulse_create(sim, false, DEG2RAD(45), M_PI / 4.);
	const struct nlop_s* nlop2 = nlop_hard_pulse_create(sim, true, DEG2RAD(45), M_PI / 4.);

	nlop1 = nlop_set_input_real(nlop1, 0);
	nlop1 = nlop_set_input_const_F(nlop1, 1, N, sim.pdims, false, par);

	nlop2 = nlop_set_input_real(nlop2, 0);
	nlop2 = nlop_set_input_const_F(nlop2, 1, N, sim.pdims, false, par);

	bool ok = true;

	for (int i = 0; i < 10; i++)
		ok = ok && compare_nlops(nlop1, nlop2, true, true, true, 1.e-4);

	nlop_free(nlop1);
	nlop_free(nlop2);

	UT_RETURN_ASSERT(ok);
}

UT_REGISTER_TEST(test_nlop_hard_pulse2);

static bool test_nlop_inv_pulse(void)
{
	struct pulse_hypsec ps = pulse_hypsec_defaults;
	pulse_hypsec_init(GYRO, &ps);

	long dims[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	struct sim_config_s sim = sim_config_default_cpu;
	sim_config_set_dims(&sim, N, dims, 1);

	float r1 = 0.;
	float r2 = 0.;
	float b1 = 1.;
	float b0 = 0.;

	complex float mag[] = { 0., 0., 1. };
	complex float par[] = { r1, r2, b1, b0 };

	const struct nlop_s* nlop = nlop_pulse_create(sim, CAST_UP(&ps), 0., (float[3]) { 0., 0., 0. });
	nlop = nlop_set_input_real(nlop, 0);
	nlop = nlop_set_input_real(nlop, 1);
	nlop = nlop_set_input_const_F(nlop, 1, N, sim.pdims, false, par);

	complex float out[3];
	nlop_generic_apply_unchecked(nlop, 2, (void*[2]) { out, mag });
	nlop_free(nlop);

	UT_RETURN_ASSERT(1.e-2 > cabsf((-1) - out[2]));
}

UT_REGISTER_TEST(test_nlop_inv_pulse);


static bool test_nlop_phase_wrap(void)
{
	struct pulse_sinc ps = pulse_sinc_defaults;
	pulse_sinc_init(&ps, 0.001, 90., 0., 4., ps.alpha);

	long dims[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	struct sim_config_s sim = sim_config_default_cpu;
	sim_config_set_dims(&sim, N, dims, 1);

	const struct nlop_s* nlop1 = nlop_pulse_create(sim, CAST_UP(&ps), M_PI / 2., (float[3]) { 0., 0., 0. });
	nlop1 = nlop_set_input_real(nlop1, 0);
	nlop1 = nlop_set_input_real(nlop1, 1);

	const struct nlop_s* nlop2 = nlop_pulse_create(sim, CAST_UP(&ps), 0., (float[3]) { 0., 0., 0. });
	nlop2 = nlop_phase_wrap_F(sim, nlop2, M_PI / 2.);
	nlop2 = nlop_set_input_real(nlop2, 0);
	nlop2 = nlop_set_input_real(nlop2, 1);

	complex float scl[4] = { 1., 1., 1., 1. };
	nlop1 = nlop_prepend_FF(nlop_from_linop_F(linop_cdiag_create(sim.N, sim.pdims, MD_BIT(sim.PI_DIM), scl)), nlop1, 1);
	nlop2 = nlop_prepend_FF(nlop_from_linop_F(linop_cdiag_create(sim.N, sim.pdims, MD_BIT(sim.PI_DIM), scl)), nlop2, 1);

	bool ok = true;

	for (int i = 0; i < 10; i++)
		ok = ok && compare_nlops(nlop1, nlop2, true, true, true, 1.e-4);

	nlop_free(nlop1);
	nlop_free(nlop2);

	UT_RETURN_ASSERT(ok);
}

UT_REGISTER_TEST(test_nlop_phase_wrap);


static bool test_nlop_pulse_der(void)
{
	struct pulse_sinc ps = pulse_sinc_defaults;
	pulse_sinc_init(&ps, 0.001, 45., 0., 4., ps.alpha);

	long dims[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	struct sim_config_s sim = sim_config_default_cpu;
	sim_config_set_dims(&sim, N, dims, 1);
	sim.tol = 1.e-6;

	float r1 = 1.;
	float r2 = 10.;
	float b1 = 1.;
	float b0 = 0.;

	complex float mag[] = { 0., 1., 1. };
	complex float par[] = { r1, r2, b1, b0 };

	const struct nlop_s* nlop = nlop_pulse_create(sim, CAST_UP(&ps), 0., (float[3]) { 0., 0., 0. });
	nlop = nlop_set_input_real(nlop, 0);
	nlop = nlop_set_input_real(nlop, 1);

	const struct nlop_s* nlop_m = nlop_set_input_const(nlop, 1, N, sim.pdims, false, par);
	float errm = nlop_test_affine_at(nlop_m, mag);
	nlop_free(nlop_m);

	long idims[N];
	md_transpose_dims(N, sim.MI_DIM, sim.MO_DIM, idims, sim.mdims);

	const struct nlop_s* nlop_p = nlop_set_input_const(nlop, 0, N, idims, false, mag);
	float errp = nlop_test_derivative_at(nlop_p, par);
	nlop_free(nlop_p);

	nlop_free(nlop);

	UT_RETURN_ON_FAILURE(0.2 > errp);
	UT_RETURN_ON_FAILURE(1.e-5 > errm);
	return true;
}

UT_REGISTER_TEST(test_nlop_pulse_der);


static bool test_nlop_pulse_der_phase(void)
{
	struct pulse_sinc ps = pulse_sinc_defaults;
	pulse_sinc_init(&ps, 0.001, 45., 0., 4., ps.alpha);

	long dims[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	struct sim_config_s sim = sim_config_default_cpu;
	sim_config_set_dims(&sim, N, dims, 1);
	sim.tol = 1.e-6;

	float r1 = 1.;
	float r2 = 10.;
	float b1 = 1.;
	float b0 = 0.;

	complex float mag[] = { 0., 1., 1. };
	complex float par[] = { r1, r2, b1, b0 };

	const struct nlop_s* nlop = nlop_pulse_create(sim, CAST_UP(&ps), M_PI / 2, (float[3]) { 0., 0., 0. });
	nlop = nlop_set_input_real(nlop, 0);
	nlop = nlop_set_input_real(nlop, 1);

	const struct nlop_s* nlop_m = nlop_set_input_const(nlop, 1, N, sim.pdims, false, par);
	float errm = nlop_test_affine_at(nlop_m, mag);
	nlop_free(nlop_m);

	long idims[N];
	md_transpose_dims(N, sim.MI_DIM, sim.MO_DIM, idims, sim.mdims);

	const struct nlop_s* nlop_p = nlop_set_input_const(nlop, 0, N, idims, false, mag);
	float errp = nlop_test_derivative_at(nlop_p, par);
	nlop_free(nlop_p);

	nlop_free(nlop);

	UT_RETURN_ON_FAILURE(0.2 > errp);
	UT_RETURN_ON_FAILURE(1.e-5 > errm);
	return true;
}

UT_REGISTER_TEST(test_nlop_pulse_der_phase);




static bool test_nlop_pulse_stm(void)
{
	struct pulse_sinc ps = pulse_sinc_defaults;
	pulse_sinc_init(&ps, 0.001, 45., 0., 4., ps.alpha);

	long dims[] = { 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	struct sim_config_s sim = sim_config_default_cpu;
	sim_config_set_dims(&sim, N, dims, 1);
	sim.tol = 1.e-6;

	const struct nlop_s* nlop = nlop_pulse_create(sim, CAST_UP(&ps), 0., (float[3]) { 0., 0., 0. });
	nlop = nlop_set_input_real(nlop, 0);
	nlop = nlop_set_input_real(nlop, 1);

	const struct nlop_s* nlop_stm = nlop_pulse_create(sim, CAST_UP(&ps), 0., (float[3]) { 0., 0., 0. });
	struct stm_s* stm = stm_create(sim, nlop_stm);
	nlop_free(nlop_stm);

	nlop_stm = nlop_stm_create(stm);
	nlop_stm = nlop_set_input_real(nlop_stm, 0);
	nlop_stm = nlop_set_input_real(nlop_stm, 1);
	stm_free(stm);

	bool ok = true;

	for (int i = 0; i < 10; i++)
		ok = ok && compare_nlops(nlop, nlop_stm, true, true, true, 2.e-5);

	nlop_free(nlop);
	nlop_free(nlop_stm);

	UT_RETURN_ASSERT(ok);
}

UT_REGISTER_TEST(test_nlop_pulse_stm);


static bool test_nlop_flash_cmp(void)
{
	struct pulse_sinc ps = pulse_sinc_defaults;
	pulse_sinc_init(&ps, 0.001, 6., 0., 4., ps.alpha);

	long dims[] = { 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	struct sim_config_s sim = sim_config_default_cpu;
	sim_config_set_dims(&sim, N, dims, 1);

	const struct nlop_s* nlop1 = nlop_seq_from_blocks_create_F(sim, flash_ops_create(sim, 5, 0.0028, 0.0018, CAST_UP(&ps)));
	const struct nlop_s* nlop2 = nlop_seq_from_blocks_jac_create_F(sim, flash_ops_create(sim, 5, 0.0028, 0.0018, CAST_UP(&ps)));

	nlop1 = nlop_set_input_real(nlop1, 0);
	nlop1 = nlop_set_input_real(nlop1, 1);
	nlop1 = nlop_set_output_real(nlop1, 0);

	nlop2 = nlop_set_input_real(nlop2, 0);
	nlop2 = nlop_set_input_real(nlop2, 1);
	nlop2 = nlop_set_output_real(nlop2, 0);

	bool ok = true;

	for (int i = 0; i < 10; i++)
		ok = ok && compare_nlops(nlop1, nlop2, true, true, true, 1.e-5);

	nlop_free(nlop1);
	nlop_free(nlop2);

	UT_RETURN_ASSERT(ok);
}

UT_REGISTER_TEST(test_nlop_flash_cmp);


static bool test_nlop_flash_adc(void)
{
	long dims[] = { 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	struct sim_config_s sim = sim_config_default_cpu;
	sim_config_set_dims(&sim, N, dims, 1);

	const struct nlop_s* nlop = nlop_adc_create(sim, -1, MD_BIT(sim.MO_DIM), 0.);

	float err = nlop_test_derivatives(nlop);
	nlop_free(nlop);

	UT_RETURN_ASSERT(err <0.01);
}

UT_REGISTER_TEST(test_nlop_flash_adc);



static bool test_nlop_pulse_order(void)
{
	struct pulse_sinc ps = pulse_sinc_defaults;
	pulse_sinc_init(&ps, 0.001, 90., 0., 4., ps.alpha);

	long dims[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	struct sim_config_s sim1 = sim_config_default_cpu;
	struct sim_config_s sim2 = sim_config_default_gpu;
	sim_config_set_dims(&sim1, N, dims, 5);
	sim_config_set_dims(&sim2, N, dims, 5);

	float r1 = 0.;
	float r2 = 0.;
	float b1 = 1.;
	float b0 = 0.;

	complex float par[] = { r1, r2, b1, b0 };

	const struct nlop_s* nlop1 = nlop_seq_from_blocks_jac_create_F(sim1, flash_ops_create(sim1, 5, 0.0028, 0.0018, CAST_UP(&ps)));
	const struct nlop_s* nlop2 = nlop_seq_from_blocks_jac_create_F(sim2, flash_ops_create(sim2, 5, 0.0028, 0.0018, CAST_UP(&ps)));

	nlop1 = nlop_del_out_F(nlop1, 0);
	nlop1 = sim_nlop_set_init(sim1, nlop1);
	nlop1 = nlop_set_input_const_F(nlop1, 0, sim1.N, nlop_domain(nlop1)->dims, true, par);

	nlop2 = nlop_del_out_F(nlop2, 0);
	nlop2 = sim_nlop_set_init(sim2, nlop2);
	nlop2 = nlop_set_input_const_F(nlop2, 0, sim2.N, nlop_domain(nlop2)->dims, true, par);

	bool ok = compare_nlops(nlop1, nlop2, true, true, true, 1.e-5);

	nlop_free(nlop1);
	nlop_free(nlop2);

	UT_RETURN_ASSERT(ok);
}

UT_REGISTER_TEST(test_nlop_pulse_order);



