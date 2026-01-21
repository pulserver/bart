/* Copyright 2018-2021. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2021-2023. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <math.h>

#include "misc/debug.h"
#include "misc/list.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/mri.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/rand.h"

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
#include "nsimu/nlop_seq_flash.h"
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

static float flash_signal(complex float* sig_ref, const struct flash_config_s config, float* adc_phase) {

	long pixels = config.npixels * config.npixels;
	float flip_rad = DEG2RAD(config.flip_angle);

	float mz = config.m0;
	mz *= config.inv ? (1. - 2. * expf(-config.TI * config.r1)) : 1.;

	for (int e = 0; e < config.excitations; e++) {

		float phase = (NULL == adc_phase) ? 0. : DEG2RAD(adc_phase[e]);
		complex float s = mz * sinf(flip_rad) * cexpf(-config.TE * config.r2) * cexpf(-phase * 1.i);

		for (int j = 0; j < pixels; j++)
			sig_ref[e * pixels + j] = s;

		mz = mz * cosf(flip_rad) * expf(-config.TR * config.r1) + config.m0 * (1. - expf(-config.TR * config.r1));
	}

	return mz;
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

static bool test_nlop_hard_pulse3(void)
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

	nlop1 = sim_nlop_set_init(sim, nlop1);
	nlop2 = sim_nlop_set_init(sim, nlop2);

	complex float omag[3];

	nlop_generic_apply_unchecked(nlop1, 2, (void* [2]) { omag, par });
	nlop_generic_apply_unchecked(nlop2, 2, (void* [2]) { omag, par });

	complex float dpar[] = { 0, 0, 1, 0 };

	complex float dmag1[3];
	complex float dmag2[3];

	linop_forward_unchecked(nlop_get_derivative(nlop1, 0, 0), dmag1, dpar);
	linop_forward_unchecked(nlop_get_derivative(nlop2, 0, 0), dmag2, dpar);

	nlop_free(nlop1);
	nlop_free(nlop2);
	
	float tol = 1.e-4;
	UT_RETURN_ON_FAILURE(tol > cabsf(dmag1[0] - dmag2[0]));
	UT_RETURN_ON_FAILURE(tol > cabsf(dmag1[1] - dmag2[1]));
	UT_RETURN_ON_FAILURE(tol > cabsf(dmag1[2] - dmag2[2]));

	return true;
}

UT_REGISTER_TEST(test_nlop_hard_pulse3);


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
	long dims[] = { 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	struct sim_config_s sim = sim_config_default_cpu;
	sim_config_set_dims(&sim, N, dims, 1);

	struct flash_config_s config = flash_config_default;
	config.excitations = 5;
	config.TE = 0.0018;
	config.TR = 0.0028;
	config.flip_angle = 6.;

	const struct nlop_s* nlop1 = nlop_seq_from_blocks_create_F(sim, flash_ops_create(sim, config));
	const struct nlop_s* nlop2 = nlop_seq_from_blocks_jac_create_F(sim, flash_ops_create(sim, config));

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
	long dims[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	struct sim_config_s sim1 = sim_config_default_cpu;
	struct sim_config_s sim2 = sim_config_default_gpu;
	sim_config_set_dims(&sim1, N, dims, 5);
	sim_config_set_dims(&sim2, N, dims, 5);

	struct flash_config_s config = flash_config_default;
	config.excitations = 5;
	config.r1 = 0.;
	config.r2 = 0.;
	config.b1 = 1.;
	config.b0 = 0.;
	config.flip_angle = 90.;

	complex float par[] = { config.r1, config.r2, config.b1, config.b0 };

	const struct nlop_s* nlop1 = nlop_seq_from_blocks_jac_create_F(sim1, flash_ops_create(sim1, config));
	const struct nlop_s* nlop2 = nlop_seq_from_blocks_jac_create_F(sim2, flash_ops_create(sim2, config));

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

static bool test_nlop_ir_flash_no_excitation(void)
{
	long dims[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	struct sim_config_s sim = sim_config_default_cpu;
	sim_config_set_dims(&sim, N, dims, 1);

	struct flash_config_s config = flash_config_default;
	config.nparams = 4;
	config.excitations = 10;
	config.flip_angle = 0; // No excitation
	config.TI = 0.1;

	const struct nlop_s* nlop = nlop_seq_from_blocks_jac_create_F(sim, ir_flash_ops_create(sim, config));

	complex float mag_in[] = { 0., 0., 1. };
	complex float par[] = { config.r1, config.r2, config.b1, config.b0 };

	long sdims[N];
	md_copy_dims(N, sdims, nlop_generic_codomain(nlop, 1)->dims);

	float mz = 1. - 2. * expf(-(config.TI + config.TR * config.excitations) * config.r1);
	const complex float mag_out_ref[] = { 0., 0., mz };

	complex float mag_out[3];
	complex float sig[md_calc_size(N, sdims)];

	nlop_generic_apply_unchecked(nlop, 4, (void* [4]) { mag_out, sig, mag_in, par });
	nlop_free(nlop);

	UT_RETURN_ON_FAILURE_TOL(md_zrmse(N, sim.mdims, mag_out_ref, mag_out), 1.e-5);

	// Assert that the signal is close enough to zero
	// accounting for floating-point rounding errors for all time points
	for (int i = 0; i < md_calc_size(N, sdims); i++)
		UT_RETURN_ON_FAILURE_TOL(cabsf(sig[i]), 1.e-7 );

	return true;
}

UT_REGISTER_TEST(test_nlop_ir_flash_no_excitation);

static bool test_nlop_ir_flash(void)
{
	long dims[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	struct sim_config_s sim = sim_config_default_cpu;
	sim_config_set_dims(&sim, N, dims, 1);

	struct flash_config_s config = flash_config_default;

	const struct nlop_s* nlop = nlop_seq_from_blocks_jac_create_F(sim, ir_flash_ops_create(sim, config));

	complex float mag_in[] = { 0., 0., 1. };
	complex float par[] = { config.r1, config.r2, config.b1, config.b0 };

	long sdims[N];
	md_copy_dims(N, sdims, nlop_generic_codomain(nlop, 1)->dims);

	// Calculate reference signal for IR-FLASH sequence
	complex float sig_ref[md_calc_size(N, sdims)];
	float mz = flash_signal(sig_ref, config, NULL);

	complex float mag_out_ref[] = { 0., 0., mz };

	complex float mag_out[3];
	complex float sig[md_calc_size(N, sdims)];

	nlop_generic_apply_unchecked(nlop, 4, (void* [4]) { mag_out, sig, mag_in, par });

	nlop_free(nlop);

	UT_RETURN_ON_FAILURE_TOL(md_zrmse(N, sdims, sig_ref, sig), 1.e-6);
	UT_RETURN_ON_FAILURE_TOL(md_zrmse(N, sim.mdims, mag_out_ref, mag_out), 1.e-5);

	return true;
}

UT_REGISTER_TEST(test_nlop_ir_flash);

static bool test_nlop_flash(void)
{
	long dims[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	struct sim_config_s sim = sim_config_default_cpu;
	sim_config_set_dims(&sim, N, dims, 1);

	struct flash_config_s config = flash_config_default;
	config.inv = false;
	config.excitations = 100;

	const struct nlop_s* nlop = nlop_seq_from_blocks_jac_create_F(sim, flash_ops_create(sim, config));

	complex float mag_in[] = { 0., 0., 1. };
	complex float par[] = { config.r1, config.r2, config.b1, config.b0 };

	long sdims[N];
	md_copy_dims(N, sdims, nlop_generic_codomain(nlop, 1)->dims);

	// Calculate reference signal for FLASH sequence
	complex float sig_ref[md_calc_size(N, sdims)];
	float mz = flash_signal(sig_ref, config, NULL);

	complex float mag_out_ref[] = { 0., 0., mz };

	complex float mag_out[3];
	complex float sig[md_calc_size(N, sdims)];

	nlop_generic_apply_unchecked(nlop, 4, (void* [4]) { mag_out, sig, mag_in, par });

	nlop_free(nlop);

	UT_RETURN_ON_FAILURE_TOL(md_zrmse(N, sdims, sig_ref, sig), 1.e-6);
	UT_RETURN_ON_FAILURE_TOL(md_zrmse(N, sim.mdims, mag_out_ref, mag_out), 1.e-6);

	return true;
}

UT_REGISTER_TEST(test_nlop_flash);

static bool test_nlop_phy_create(void)
{
	struct flash_config_s config = flash_config_default;
	config.inv = true;
	config.npixels = 16;
	config.excitations = 10;
	long num_tot_pixels = config.npixels * config.npixels;

	long dims[] = { config.npixels, config.npixels, 1, 1, 1, config.excitations, config.nparams, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	long map_dims[N];
	md_select_dims(N, ~TE_FLAG & ~COEFF_FLAG, map_dims, dims);

	long out_dims[N];
	md_select_dims(N, ~COEFF_FLAG, out_dims, dims);

	long sdims[N];
	md_copy_dims(N, sdims, out_dims);
	complex float sig[md_calc_size(N, sdims)];

	long pdims[N];
	md_select_dims(N, ~TE_FLAG, pdims, dims);
	complex float par[num_tot_pixels * config.nparams];
	for (int i = 0; i < num_tot_pixels; i++) {
		par[i + 0 * num_tot_pixels] = config.m0;
		par[i + 1 * num_tot_pixels] = config.r1;
		par[i + 2 * num_tot_pixels] = config.r2;
		par[i + 3 * num_tot_pixels] = config.b1;
		par[i + 4 * num_tot_pixels] = config.b0;
	}

	// In: (M0, R1, R2, B0, B1), out: signal for each excitation
	const struct nlop_s* nlop = nlop_phy_create(N, map_dims, out_dims, config, sim_config_default_gpu);
	nlop_apply(nlop, N, sdims, sig, N, pdims, par);

	nlop_free(nlop);

	// Calculate reference signal for IR-FLASH sequence
	complex float sig_ref[md_calc_size(N, sdims)];
	flash_signal(sig_ref, config, NULL);

	UT_RETURN_ON_FAILURE_TOL(md_zrmse(N, sdims, sig_ref, sig), 1.e-6);

	return true;
}

UT_REGISTER_TEST(test_nlop_phy_create);

static bool test_nlop_flash_sim_pulse(void)
{
	long dims[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	struct sim_config_s sim = sim_config_default_cpu;
	sim_config_set_dims(&sim, N, dims, 1);

	struct flash_config_s config = flash_config_default;
	config.flip_angle = 6;
	config.TI = 0.;

	// Set R1 and R2 to zero to avoid relaxation effects during the simulated pulse
	config.r1 = 0.; 
	config.r2 = 0.;

	const struct nlop_s* nlop_hard_pulse = nlop_seq_from_blocks_jac_create_F(sim, flash_ops_create(sim, config));

	sim.hard_pulse_sim = false;
	const struct nlop_s* nlop_sim_pulse = nlop_seq_from_blocks_jac_create_F(sim, flash_ops_create(sim, config));

	complex float imag[] = { 0., 0., 1. };
	complex float par[] = { config.r1, config.r2, config.b1, config.b0 };

	long sdims[N];
	md_copy_dims(N, sdims, nlop_generic_codomain(nlop_hard_pulse, 1)->dims);

	complex float omag_hard_pulse[3];
	complex float omag_sim_pulse[3];
	complex float sig_hard_pulse[md_calc_size(N, sdims)];
	complex float sig_sim_pulse[md_calc_size(N, sdims)];

	nlop_generic_apply_unchecked(nlop_hard_pulse, 4, (void* [4]) { omag_hard_pulse, sig_hard_pulse, imag, par });
	nlop_generic_apply_unchecked(nlop_sim_pulse, 4, (void* [4]) { omag_sim_pulse, sig_sim_pulse, imag, par });

	nlop_free(nlop_hard_pulse);
	nlop_free(nlop_sim_pulse);

	float tol = 1.e-4;
	UT_RETURN_ON_FAILURE_TOL(md_zrmse(N, sdims, sig_hard_pulse, sig_sim_pulse), tol);
	UT_RETURN_ON_FAILURE_TOL(cabsf(omag_hard_pulse[0] - omag_sim_pulse[0]), tol);
	UT_RETURN_ON_FAILURE_TOL(cabsf(omag_hard_pulse[1] - omag_sim_pulse[1]), tol);
	UT_RETURN_ON_FAILURE_TOL(cabsf(omag_hard_pulse[2] - omag_sim_pulse[2]), tol);

	return true;
}

UT_REGISTER_TEST(test_nlop_flash_sim_pulse);

static bool test_nlop_ir_flash_sim_pulse(void)
{
	long dims[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	struct sim_config_s sim = sim_config_default_cpu;
	sim_config_set_dims(&sim, N, dims, 1);

	struct flash_config_s config = flash_config_default;
	config.flip_angle = 6;
	config.TI = 0;

	// Set R1 and R2 to zero to avoid relaxation effects during the simulated pulse
	config.r1 = 0.; 
	config.r2 = 0.;

	const struct nlop_s* nlop_hard_pulse = nlop_seq_from_blocks_jac_create_F(sim, ir_flash_ops_create(sim, config));

	sim.hard_pulse_sim = false;
	const struct nlop_s* nlop_sim_pulse = nlop_seq_from_blocks_jac_create_F(sim, ir_flash_ops_create(sim, config));

	complex float imag[] = { 0., 0., 1. };
	complex float par[] = { config.r1, config.r2, config.b1, config.b0 };

	long sdims[N];
	md_copy_dims(N, sdims, nlop_generic_codomain(nlop_hard_pulse, 1)->dims);

	complex float omag_hard_pulse[3];
	complex float omag_sim_pulse[3];
	complex float sig_hard_pulse[md_calc_size(N, sdims)];
	complex float sig_sim_pulse[md_calc_size(N, sdims)];

	nlop_generic_apply_unchecked(nlop_hard_pulse, 4, (void* [4]) { omag_hard_pulse, sig_hard_pulse, imag, par });
	nlop_generic_apply_unchecked(nlop_sim_pulse, 4, (void* [4]) { omag_sim_pulse, sig_sim_pulse, imag, par });

	nlop_free(nlop_hard_pulse);
	nlop_free(nlop_sim_pulse);
	
	float tol = 1.e-2;
	UT_RETURN_ON_FAILURE_TOL(md_zrmse(N, sdims, sig_hard_pulse, sig_sim_pulse), tol);
	UT_RETURN_ON_FAILURE_TOL(cabsf(omag_hard_pulse[0] - omag_sim_pulse[0]), tol);
	UT_RETURN_ON_FAILURE_TOL(cabsf(omag_hard_pulse[1] - omag_sim_pulse[1]), tol);
	UT_RETURN_ON_FAILURE_TOL(cabsf(omag_hard_pulse[2] - omag_sim_pulse[2]), tol);
	
	return true;
}

UT_REGISTER_TEST(test_nlop_ir_flash_sim_pulse);

static bool test_nlop_phy_create_flash_sim_pulse(void)
{
	struct sim_config_s sim = sim_config_default_gpu;
	sim.hard_pulse_sim = false;

	struct flash_config_s config = flash_config_default;
	config.inv = false;
	config.npixels = 2;
	config.excitations = 10;
	config.flip_angle = 10;
	config.m0 = 1;
	config.b1 = 1;
	config.b0 = 0;
	config.TI = 0;
	
	// Set R1 and R2 to zero to avoid relaxation effects during the simulated pulse
	config.r1 = 0;
	config.r2 = 0;

	long num_tot_pixels = config.npixels * config.npixels;

	long dims[] = { config.npixels, config.npixels, 1, 1, 1, config.excitations, config.nparams, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	long mdims[N];
	md_select_dims(N, ~TE_FLAG & ~COEFF_FLAG, mdims, dims);

	long odims[N];
	md_select_dims(N, ~COEFF_FLAG, odims, dims);

	long sdims[N];
	md_copy_dims(N, sdims, odims);
	complex float sig[md_calc_size(N, sdims)];

	long pdims[N];
	md_select_dims(N, ~TE_FLAG, pdims, dims);

	complex float par[num_tot_pixels * config.nparams];
	for (int i = 0; i < num_tot_pixels; i++) {

		par[i + 0 * num_tot_pixels] = config.m0;
		par[i + 1 * num_tot_pixels] = config.r1;
		par[i + 2 * num_tot_pixels] = config.r2;
		par[i + 3 * num_tot_pixels] = config.b1;
		par[i + 4 * num_tot_pixels] = config.b0;
	}

	// In: (M0, R1, R2, B0, B1), out: signal for each excitation per pixel
	const struct nlop_s* nlop = nlop_phy_create(N, mdims, odims, config, sim);
	nlop_apply(nlop, N, sdims, sig, N, pdims, par);
	nlop_free(nlop);

	// Calculate reference signal for FLASH sequence
	complex float sig_ref[md_calc_size(N, sdims)];
	flash_signal(sig_ref, config, NULL);

	UT_RETURN_ON_FAILURE_TOL(md_zrmse(N, sdims, sig_ref, sig), 1.e-5);

	return true;
}

UT_REGISTER_TEST(test_nlop_phy_create_flash_sim_pulse);

static bool test_nlop_phy_create_flash_hard_pulse(void)
{
	struct sim_config_s sim = sim_config_default_gpu;
	sim.hard_pulse_sim = true;

	struct flash_config_s config = flash_config_default;
	config.inv = false;
	config.npixels = 1;
	config.excitations = 100;
	config.flip_angle = 6;
	config.TE = 1.90E-3;
	config.TR = 0.05;
	config.TI = 0;

	long num_tot_pixels = config.npixels * config.npixels;

	long dims[] = { config.npixels, config.npixels, 1, 1, 1, config.excitations, config.nparams, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	long mdims[N];
	md_select_dims(N, ~TE_FLAG & ~COEFF_FLAG, mdims, dims);

	long odims[N];
	md_select_dims(N, ~COEFF_FLAG, odims, dims);

	long sdims[N];
	md_copy_dims(N, sdims, odims);
	complex float sig[md_calc_size(N, sdims)];

	long pdims[N];
	md_select_dims(N, ~TE_FLAG, pdims, dims);

	complex float par[num_tot_pixels * config.nparams];
	for (int i = 0; i < num_tot_pixels; i++) {

		par[i + 0 * num_tot_pixels] = config.m0;
		par[i + 1 * num_tot_pixels] = config.r1;
		par[i + 2 * num_tot_pixels] = config.r2;
		par[i + 3 * num_tot_pixels] = config.b1;
		par[i + 4 * num_tot_pixels] = config.b0;
	}

	// In: (M0, R1, R2, B0, B1), out: signal for each excitation per pixel
	const struct nlop_s* nlop = nlop_phy_create(N, mdims, odims, config, sim);
	nlop_apply(nlop, N, sdims, sig, N, pdims, par);
	nlop_free(nlop);

	// Calculate reference signal for FLASH sequence
	complex float sig_ref[md_calc_size(N, sdims)];
	flash_signal(sig_ref, config, NULL);

	UT_RETURN_ON_FAILURE_TOL(md_zrmse(N, sdims, sig_ref, sig), 1.e-5);

	return true;
}

UT_REGISTER_TEST(test_nlop_phy_create_flash_hard_pulse);

static bool test_nlop_phy_create_ir_flash_sim_pulse(void)
{
	struct sim_config_s sim = sim_config_default_gpu;
	sim.hard_pulse_sim = false;

	struct flash_config_s config = flash_config_default;
	config.inv = true;
	config.npixels = 10;
	config.excitations = 10;
	config.flip_angle = 10;
	config.m0 = 1;
	config.b1 = 1;
	config.b0 = 0;
	config.TI = 0;
	
	// Set R1 and R2 to zero to avoid relaxation effects during the simulated pulse
	config.r1 = 0;
	config.r2 = 0;

	long num_tot_pixels = config.npixels * config.npixels;

	long dims[] = { config.npixels, config.npixels, 1, 1, 1, config.excitations, config.nparams, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	long mdims[N];
	md_select_dims(N, ~TE_FLAG & ~COEFF_FLAG, mdims, dims);

	long odims[N];
	md_select_dims(N, ~COEFF_FLAG, odims, dims);

	long sdims[N];
	md_copy_dims(N, sdims, odims);
	complex float sig[md_calc_size(N, sdims)];

	long pdims[N];
	md_select_dims(N, ~TE_FLAG, pdims, dims);

	complex float par[num_tot_pixels * config.nparams];
	for (int i = 0; i < num_tot_pixels; i++) {

		par[i + 0 * num_tot_pixels] = config.m0;
		par[i + 1 * num_tot_pixels] = config.r1;
		par[i + 2 * num_tot_pixels] = config.r2;
		par[i + 3 * num_tot_pixels] = config.b1;
		par[i + 4 * num_tot_pixels] = config.b0;
	}

	// In: (M0, R1, R2, B0, B1), out: signal for each excitation per pixel
	const struct nlop_s* nlop = nlop_phy_create(N, mdims, odims, config, sim);
	nlop_apply(nlop, N, sdims, sig, N, pdims, par);
	nlop_free(nlop);

	// Calculate reference signal for IR-FLASH sequence
	complex float sig_ref[md_calc_size(N, sdims)];
	flash_signal(sig_ref, config, NULL);

	UT_RETURN_ON_FAILURE_TOL(md_zrmse(N, sdims, sig_ref, sig), 1.e-3);

	return true;
}

UT_REGISTER_TEST(test_nlop_phy_create_ir_flash_sim_pulse);

static bool test_nlop_pulse_shape_create(void)
{
	struct seq_config seq = seq_config_defaults;
	seq.phys.rf_duration = 0.001;
	seq.phys.flip_angle = 90.;
	struct rf_shape rf_shapes[1];
	seq_sample_rf_shapes(1, rf_shapes, &seq);

	long dims[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	struct sim_config_s sim = sim_config_default_cpu;
	sim_config_set_dims(&sim, N, dims, 1);

	float r1 = 1.;
	float r2 = 0.;
	float b1 = 1.;
	float b0 = 0.;

	complex float mag[] = { 0., 0., 1. };
	complex float par[] = { r1, r2, b1, b0 };

	const struct nlop_s* nlop = nlop_pulse_shape_create(sim, &rf_shapes[0], 0., (float[3]) { 0., 0., 0. });
	
	nlop = nlop_set_input_real(nlop, 0);
	nlop = nlop_set_input_real(nlop, 1);
	nlop = nlop_set_input_const_F(nlop, 1, N, sim.pdims, false, par);

	complex float out[3];
	nlop_generic_apply_unchecked(nlop, 2, (void*[2]) { out, mag });
	nlop_free(nlop);

	complex float ref[] = { 0., 1., 0. };

	UT_RETURN_ASSERT_TOL(md_zrmse(N, sim.mdims, ref, out), 1.e-3);
}

UT_REGISTER_TEST(test_nlop_pulse_shape_create);

static bool test_nlop_pulse_shape_create2(void)
{
	struct seq_config seq = seq_config_defaults;
	struct rf_shape rf_shapes[1];
	seq_sample_rf_shapes(1, rf_shapes, &seq);

	struct pulse_sms ps = pulse_sms_defaults;
	pulse_sms_init(&ps, seq.phys.rf_duration, seq.phys.flip_angle, 0., seq.phys.bwtp, 0.5, seq.geom.mb_factor, 0, seq.geom.sms_distance, seq.geom.slice_thickness);

	long dims[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	struct sim_config_s sim = sim_config_default_cpu;
	sim_config_set_dims(&sim, N, dims, 1);

	float r1 = 1.;
	float r2 = 0.;
	float b1 = 1.;
	float b0 = 0.;

	complex float par[] = { r1, r2, b1, b0 };

	float grad[3] = { 0. };

	const struct nlop_s* nlop1 = nlop_pulse_create(sim, CAST_UP(&ps), 0., grad);
	const struct nlop_s* nlop2 = nlop_pulse_shape_create(sim, &rf_shapes[0], 0., grad);

	nlop1 = nlop_set_input_real(nlop1, 0);
	nlop1 = nlop_set_input_const_F(nlop1, 1, N, sim.pdims, false, par);

	nlop2 = nlop_set_input_real(nlop2, 0);
	nlop2 = nlop_set_input_const_F(nlop2, 1, N, sim.pdims, false, par);

	bool ok = true;

	for (int i = 0; i < 10; i++)
		ok = ok && compare_nlops(nlop1, nlop2, true, true, true, 1.e-3);

	nlop_free(nlop1);
	nlop_free(nlop2);

	UT_RETURN_ASSERT(ok);
}

UT_REGISTER_TEST(test_nlop_pulse_shape_create2);

static bool test_nlop_pulse_shape_create3(void)
{
	struct seq_config seq = seq_config_defaults;
	struct rf_shape rf_shapes[1];
	seq_sample_rf_shapes(1, rf_shapes, &seq);

	struct pulse_sms ps = pulse_sms_defaults;
	pulse_sms_init(&ps, seq.phys.rf_duration, seq.phys.flip_angle, 0., seq.phys.bwtp, 0.5, seq.geom.mb_factor, 0, seq.geom.sms_distance, seq.geom.slice_thickness);

	long dims[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	struct sim_config_s sim = sim_config_default_cpu;
	sim_config_set_dims(&sim, N, dims, 1);

	float r1 = 0.;
	float r2 = 0.;
	float b1 = 1.25;
	float b0 = 0.;

	complex float par[] = { r1, r2, b1, b0 };

	const struct nlop_s* nlop1 = nlop_pulse_create(sim, CAST_UP(&ps), 0, (float[3]) { 0., 0., 0. });
	const struct nlop_s* nlop2 = nlop_pulse_shape_create(sim, &rf_shapes[0], 0., (float[3]) { 0., 0., 0. });
	nlop1 = sim_nlop_set_init(sim, nlop1);
	nlop2 = sim_nlop_set_init(sim, nlop2);

	complex float omag[3];

	nlop_generic_apply_unchecked(nlop1, 2, (void* [2]) { omag, par });
	nlop_generic_apply_unchecked(nlop2, 2, (void* [2]) { omag, par });

	complex float dpar[] = { 0, 0, 1, 0 };

	complex float dmag1[3];
	complex float dmag2[3];

	linop_forward_unchecked(nlop_get_derivative(nlop1, 0, 0), dmag1, dpar);
	linop_forward_unchecked(nlop_get_derivative(nlop2, 0, 0), dmag2, dpar);

	nlop_free(nlop1);
	nlop_free(nlop2);

	float tol = 1.e-3;
	UT_RETURN_ON_FAILURE_TOL(cabsf(dmag1[0] - dmag2[0]), tol);
	UT_RETURN_ON_FAILURE_TOL(cabsf(dmag1[1] - dmag2[1]), tol);
	UT_RETURN_ON_FAILURE_TOL(cabsf(dmag1[2] - dmag2[2]), tol);

	return true;
}

UT_REGISTER_TEST(test_nlop_pulse_shape_create3);

static bool test_pulse_shape_create_inv_pulse(void)
{
	struct seq_config seq = seq_config_defaults;
	seq.magn.mag_prep = SEQ_PREP_IR_NONSELECTIVE;
	struct rf_shape rf_shapes[2];
	seq_sample_rf_shapes(2, rf_shapes, &seq);

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

	const struct nlop_s* nlop1 = nlop_pulse_shape_create(sim, &rf_shapes[1], 0., (float[3]) { 0., 0., 0. });
	const struct nlop_s* nlop2 = nlop_pulse_create(sim, CAST_UP(&ps), 0., (float[3]) { 0., 0., 0. });

	nlop1 = nlop_set_input_real(nlop1, 0);
	nlop1 = nlop_set_input_real(nlop1, 1);
	nlop1 = nlop_set_input_const_F(nlop1, 1, N, sim.pdims, false, par);

	nlop2 = nlop_set_input_real(nlop2, 0);
	nlop2 = nlop_set_input_real(nlop2, 1);
	nlop2 = nlop_set_input_const_F(nlop2, 1, N, sim.pdims, false, par);

	complex float out1[3];
	complex float out2[3];

	nlop_generic_apply_unchecked(nlop1, 2, (void*[2]) { out1, mag });
	nlop_generic_apply_unchecked(nlop2, 2, (void*[2]) { out2, mag });

	nlop_free(nlop1);
	nlop_free(nlop2);

	// Only check magnetization in z-direction (since magnetization in xy-plane will be spoiled for inversion pulse)
	UT_RETURN_ASSERT_TOL(cabsf(out1[2] - out2[2]), 1.e-4);

	return true;
}

UT_REGISTER_TEST(test_pulse_shape_create_inv_pulse);
