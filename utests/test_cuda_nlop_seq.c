/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <math.h>

#include "misc/debug.h"
#include "misc/mri.h"

#include "misc/misc.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"

#include "simu/bloch.h"
#include "seq/pulse.h"
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

#include "utest.h"

static bool test_cuda_nlop_pulse(void)
{
	struct pulse_sinc ps = pulse_sinc_defaults;
	pulse_sinc_init(&ps, 0.001, 90., 0., 4., ps.alpha);

	bart_dim_t dims[16] = { [0 ... 15] = 1 };
	int N = ARRAY_SIZE(dims);

	struct sim_config_s sim = sim_config_default_cpu;
	sim_config_set_dims(&sim, N, dims, 1);

	float grad[3] = { 0., 0., 0. };

	const struct nlop_s* nlop_cpu = nlop_pulse_create(sim, CAST_UP(&ps), 0.1, grad);
	const struct nlop_s* nlop_gpu = nlop_gpu_wrapper_F(nlop_pulse_create(sim, CAST_UP(&ps), 0.1, grad));

	bool ok = true;

	for (int i = 0; i < 10; i++)
		ok = ok && compare_nlops(nlop_cpu, nlop_gpu, true, true, true, 1.e-5);

	nlop_free(nlop_cpu);
	nlop_free(nlop_gpu);

	return ok;
}

UT_GPU_REGISTER_TEST(test_cuda_nlop_pulse);

static bool test_cuda_nlop_hypsec(void)
{
	struct pulse_hypsec ps = pulse_hypsec_defaults;
	pulse_hypsec_init(GYRO, &ps);

	bart_dim_t dims[16] = { [0 ... 15] = 1 };
	int N = ARRAY_SIZE(dims);

	struct sim_config_s sim = sim_config_default_cpu;
	sim_config_set_dims(&sim, N, dims, 1);

	float grad[3] = { 0., 0., 0. };

	const struct nlop_s* nlop_cpu = nlop_pulse_create(sim, CAST_UP(&ps), 0.1, grad);
	const struct nlop_s* nlop_gpu = nlop_gpu_wrapper_F(nlop_pulse_create(sim, CAST_UP(&ps), 0.1, grad));

	bool ok = true;

	for (int i = 0; i < 10; i++)
		ok = ok && compare_nlops(nlop_cpu, nlop_gpu, true, true, true, 1.e-4);

	nlop_free(nlop_cpu);
	nlop_free(nlop_gpu);

	return ok;
}

UT_GPU_REGISTER_TEST(test_cuda_nlop_hypsec);

static bool test_cuda_flash_ops_create(void)
{
	struct flash_config_s config = flash_config_default;
	config.npixels = 1;

	bart_dim_t sim_dims[] = { [0 ... DIMS - 1] = 1 };
	int N = ARRAY_SIZE(sim_dims);

	struct sim_config_s sim_cpu = sim_config_default_cpu;
	sim_cpu.hard_pulse_sim = false;
	sim_config_set_dims(&sim_cpu, N, sim_dims, 1);

	struct sim_config_s sim_gpu = sim_config_default_gpu;
	sim_gpu.hard_pulse_sim = false;
	sim_config_set_dims(&sim_gpu, N, sim_dims, 1);

	const struct nlop_s* nlop_cpu = nlop_seq_from_blocks_jac_create_F(sim_cpu, flash_ops_create(sim_cpu, config));
	const struct nlop_s* nlop_gpu = nlop_gpu_wrapper_F(nlop_seq_from_blocks_jac_create_F(sim_gpu, flash_ops_create(sim_gpu, config)));

	complex float imag[] = { 0., 0., 1. };
	complex float par[] = { config.r1, config.r2, config.b1, config.b0 };

	bart_dim_t dims[] = { config.npixels, config.npixels, 1, 1, 1, config.excitations, config.nparams, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	N = ARRAY_SIZE(dims);

	bart_dim_t mdims[N];
	md_select_dims(N, ~TE_FLAG & ~COEFF_FLAG, mdims, dims);

	bart_dim_t sdims[N];
	md_select_dims(N, ~COEFF_FLAG, sdims, dims);

	complex float omag_cpu[3];
	complex float omag_gpu[3];
	complex float sig_cpu[md_calc_size(N, sdims)];
	complex float sig_gpu[md_calc_size(N, sdims)];

	nlop_generic_apply_unchecked(nlop_cpu, 4, (void* [4]) { omag_cpu, sig_cpu, imag, par });
	nlop_generic_apply_unchecked(nlop_gpu, 4, (void* [4]) { omag_gpu, sig_gpu, imag, par });

	nlop_free(nlop_cpu);
	nlop_free(nlop_gpu);

	UT_RETURN_ON_FAILURE_TOL(md_zrmse(N, sdims, sig_cpu, sig_gpu), 1.e-4 );
	UT_RETURN_ON_FAILURE_TOL(md_zrmse(N, sim_cpu.mdims, omag_cpu, omag_gpu), 1.e-5);

	return true;
}

UT_GPU_REGISTER_TEST(test_cuda_flash_ops_create);

static bool test_cuda_ir_flash_ops_create(void)
{
	struct flash_config_s config = flash_config_default;
	config.npixels = 1;

	bart_dim_t sim_dims[] = { [0 ... DIMS - 1] = 1 };
	int N = ARRAY_SIZE(sim_dims);

	struct sim_config_s sim_cpu = sim_config_default_cpu;
	sim_cpu.hard_pulse_sim = false;
	sim_config_set_dims(&sim_cpu, N, sim_dims, 1);

	struct sim_config_s sim_gpu = sim_config_default_gpu;
	sim_gpu.hard_pulse_sim = false;
	sim_config_set_dims(&sim_gpu, N, sim_dims, 1);

	const struct nlop_s* nlop_cpu = nlop_seq_from_blocks_jac_create_F(sim_cpu, ir_flash_ops_create(sim_cpu, config));
	const struct nlop_s* nlop_gpu = nlop_gpu_wrapper_F(nlop_seq_from_blocks_jac_create_F(sim_gpu, ir_flash_ops_create(sim_gpu, config)));

	complex float imag[] = { 0., 0., 1. };
	complex float par[] = { config.r1, config.r2, config.b1, config.b0 };

	bart_dim_t dims[] = { config.npixels, config.npixels, 1, 1, 1, config.excitations, config.nparams, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	N = ARRAY_SIZE(dims);

	bart_dim_t mdims[N];
	md_select_dims(N, ~TE_FLAG & ~COEFF_FLAG, mdims, dims);

	bart_dim_t sdims[N];
	md_select_dims(N, ~COEFF_FLAG, sdims, dims);

	complex float omag_cpu[3];
	complex float omag_gpu[3];
	complex float sig_cpu[md_calc_size(N, sdims)];
	complex float sig_gpu[md_calc_size(N, sdims)];

	nlop_generic_apply_unchecked(nlop_cpu, 4, (void* [4]) { omag_cpu, sig_cpu, imag, par });
	nlop_generic_apply_unchecked(nlop_gpu, 4, (void* [4]) { omag_gpu, sig_gpu, imag, par });

	nlop_free(nlop_cpu);
	nlop_free(nlop_gpu);

	UT_RETURN_ON_FAILURE_TOL(md_zrmse(N, sdims, sig_cpu, sig_gpu), 1.e-4 );
	UT_RETURN_ON_FAILURE_TOL(md_zrmse(N, sim_cpu.mdims, omag_cpu, omag_gpu), 1.e-5);

	return true;
}

UT_GPU_REGISTER_TEST(test_cuda_ir_flash_ops_create);

static bool test_cuda_nlop_phy_create(void)
{
	struct sim_config_s sim_cpu = sim_config_default_cpu;
	struct sim_config_s sim_gpu = sim_config_default_gpu;
	sim_cpu.hard_pulse_sim = false;
	sim_gpu.hard_pulse_sim = false;

	struct flash_config_s config = flash_config_default;
	config.npixels = 12;

	bart_dim_t num_tot_pixels = config.npixels * config.npixels;

	bart_dim_t dims[] = { config.npixels, config.npixels, 1, 1, 1, config.excitations, config.nparams, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	bart_dim_t mdims[N];
	md_select_dims(N, ~TE_FLAG & ~COEFF_FLAG, mdims, dims);

	bart_dim_t odims[N];
	md_select_dims(N, ~COEFF_FLAG, odims, dims);

	bart_dim_t sdims[N];
	md_copy_dims(N, sdims, odims);

	bart_dim_t pdims[N];
	md_select_dims(N, ~TE_FLAG, pdims, dims);

	complex float par[num_tot_pixels * config.nparams];
	for (int i = 0; i < num_tot_pixels; i++) {

		par[i + 0 * num_tot_pixels] = config.m0;
		par[i + 1 * num_tot_pixels] = config.r1;
		par[i + 2 * num_tot_pixels] = config.r2;
		par[i + 3 * num_tot_pixels] = config.b1;
		par[i + 4 * num_tot_pixels] = config.b0;
	}

	const struct nlop_s* nlop_cpu = nlop_phy_create(N, mdims, odims, config, sim_cpu);
	const struct nlop_s* nlop_gpu = nlop_gpu_wrapper_F(nlop_phy_create(N, mdims, odims, config, sim_gpu));
	
	complex float sig_cpu[md_calc_size(N, sdims)];
	complex float sig_gpu[md_calc_size(N, sdims)];

	nlop_apply(nlop_cpu, N, sdims, sig_cpu, N, pdims, par);
	nlop_apply(nlop_gpu, N, sdims, sig_gpu, N, pdims, par);

	nlop_free(nlop_cpu);
	nlop_free(nlop_gpu);

	UT_RETURN_ON_FAILURE_TOL(md_zrmse(N, sdims, sig_cpu, sig_gpu), 1.e-4 );

	return true;
}

UT_GPU_REGISTER_TEST(test_cuda_nlop_phy_create);

static bool test_cuda_flash_nlop_phy_create_sim_pulses_der(void)
{
	struct sim_config_s sim_cpu = sim_config_default_cpu;
	struct sim_config_s sim_gpu = sim_config_default_gpu;
	sim_cpu.hard_pulse_sim = false;
	sim_gpu.hard_pulse_sim = false;

	struct flash_config_s config = flash_config_default;
	config.npixels = 12;
	config.inv = false;

	bart_dim_t num_tot_pixels = config.npixels * config.npixels;

	bart_dim_t dims[] = { config.npixels, config.npixels, 1, 1, 1, config.excitations, config.nparams, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	bart_dim_t mdims[N];
	md_select_dims(N, ~TE_FLAG & ~COEFF_FLAG, mdims, dims);

	bart_dim_t odims[N];
	md_select_dims(N, ~COEFF_FLAG, odims, dims);

	bart_dim_t sdims[N];
	md_copy_dims(N, sdims, odims);

	bart_dim_t pdims[N];
	md_select_dims(N, ~TE_FLAG, pdims, dims);

	complex float par[num_tot_pixels * config.nparams];
	for (int i = 0; i < num_tot_pixels; i++) {

		par[i + 0 * num_tot_pixels] = config.m0;
		par[i + 1 * num_tot_pixels] = config.r1;
		par[i + 2 * num_tot_pixels] = config.r2;
		par[i + 3 * num_tot_pixels] = config.b1;
		par[i + 4 * num_tot_pixels] = config.b0;
	}

	const struct nlop_s* nlop_cpu = nlop_phy_create(N, mdims, odims, config, sim_cpu);
	const struct nlop_s* nlop_gpu = nlop_gpu_wrapper_F(nlop_phy_create(N, mdims, odims, config, sim_gpu));
	
	complex float sig_cpu[md_calc_size(N, sdims)];
	complex float sig_gpu[md_calc_size(N, sdims)];

	nlop_apply(nlop_cpu, N, sdims, sig_cpu, N, pdims, par);
	nlop_apply(nlop_gpu, N, sdims, sig_gpu, N, pdims, par);

	// Simulate derivatives and adjoints
	complex float dsig_cpu[md_calc_size(N, sdims)];
	complex float dsig_gpu[md_calc_size(N, sdims)];

	complex float dpar[md_calc_size(N, pdims)];
	md_gaussian_rand(N, pdims, dpar);

	nlop_derivative(nlop_cpu, N, sdims, dsig_cpu, N, pdims, dpar);
	nlop_derivative(nlop_gpu, N, sdims, dsig_gpu, N, pdims, dpar);

	complex float dpar_adj_cpu[md_calc_size(N, pdims)];
	complex float dpar_adj_gpu[md_calc_size(N, pdims)];
	complex float dsig_adj[md_calc_size(N, sdims)];

	md_gaussian_rand(N, sdims, dsig_adj);

	nlop_adjoint(nlop_cpu, N, pdims, dpar_adj_cpu, N, sdims, dsig_adj);
	nlop_adjoint(nlop_gpu, N, pdims, dpar_adj_gpu, N, sdims, dsig_adj);

	nlop_free(nlop_cpu);
	nlop_free(nlop_gpu);

	// Assert
	UT_RETURN_ON_FAILURE_TOL(md_zrmse(N, sdims, sig_cpu, sig_gpu), 1.e-4 );
	UT_RETURN_ON_FAILURE_TOL(md_zrmse(N, sdims, dsig_cpu, dsig_gpu), 1.e-3);
	UT_RETURN_ON_FAILURE_TOL(md_zrmse(1, (bart_dim_t[]) { 5 }, dpar_adj_cpu, dpar_adj_gpu), 1.e-3);

	return true;
}

UT_GPU_REGISTER_TEST(test_cuda_flash_nlop_phy_create_sim_pulses_der);

static bool test_cuda_flash_nlop_phy_create_hp_pulses_der(void)
{
	struct flash_config_s config = flash_config_default;
	config.npixels = 12;
	config.inv = false;

	bart_dim_t num_tot_pixels = config.npixels * config.npixels;

	bart_dim_t dims[] = { config.npixels, config.npixels, 1, 1, 1, config.excitations, config.nparams, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	bart_dim_t mdims[N];
	md_select_dims(N, ~TE_FLAG & ~COEFF_FLAG, mdims, dims);

	bart_dim_t odims[N];
	md_select_dims(N, ~COEFF_FLAG, odims, dims);

	bart_dim_t sdims[N];
	md_copy_dims(N, sdims, odims);

	bart_dim_t pdims[N];
	md_select_dims(N, ~TE_FLAG, pdims, dims);

	complex float par[num_tot_pixels * config.nparams];
	for (int i = 0; i < num_tot_pixels; i++) {

		par[i + 0 * num_tot_pixels] = config.m0;
		par[i + 1 * num_tot_pixels] = config.r1;
		par[i + 2 * num_tot_pixels] = config.r2;
		par[i + 3 * num_tot_pixels] = config.b1;
		par[i + 4 * num_tot_pixels] = config.b0;
	}

	const struct nlop_s* nlop_cpu = nlop_phy_create(N, mdims, odims, config, sim_config_default_cpu);
	const struct nlop_s* nlop_gpu = nlop_gpu_wrapper_F(nlop_phy_create(N, mdims, odims, config, sim_config_default_gpu));
	
	complex float sig_cpu[md_calc_size(N, sdims)];
	complex float sig_gpu[md_calc_size(N, sdims)];

	nlop_apply(nlop_cpu, N, sdims, sig_cpu, N, pdims, par);
	nlop_apply(nlop_gpu, N, sdims, sig_gpu, N, pdims, par);

	// Simulate derivatives and adjoints
	complex float dsig_cpu[md_calc_size(N, sdims)];
	complex float dsig_gpu[md_calc_size(N, sdims)];

	complex float dpar[md_calc_size(N, pdims)];
	md_gaussian_rand(N, pdims, dpar);

	nlop_derivative(nlop_cpu, N, sdims, dsig_cpu, N, pdims, dpar);
	nlop_derivative(nlop_gpu, N, sdims, dsig_gpu, N, pdims, dpar);

	complex float dpar_adj_cpu[md_calc_size(N, pdims)];
	complex float dpar_adj_gpu[md_calc_size(N, pdims)];
	complex float dsig_adj[md_calc_size(N, sdims)];

	md_gaussian_rand(N, sdims, dsig_adj);

	nlop_adjoint(nlop_cpu, N, pdims, dpar_adj_cpu, N, sdims, dsig_adj);
	nlop_adjoint(nlop_gpu, N, pdims, dpar_adj_gpu, N, sdims, dsig_adj);

	nlop_free(nlop_cpu);
	nlop_free(nlop_gpu);

	// Assert
	UT_RETURN_ON_FAILURE_TOL(md_zrmse(N, sdims, sig_cpu, sig_gpu), 1.e-4 );
	UT_RETURN_ON_FAILURE_TOL(md_zrmse(N, sdims, dsig_cpu, dsig_gpu), 1.e-3);
	UT_RETURN_ON_FAILURE_TOL(md_zrmse(1, (bart_dim_t[]) { 5 }, dpar_adj_cpu, dpar_adj_gpu), 1.e-3);

	return true;
}

UT_GPU_REGISTER_TEST(test_cuda_flash_nlop_phy_create_hp_pulses_der);

static bool test_cuda_ir_flash_nlop_phy_create_sim_pulses_der(void)
{
	struct sim_config_s sim_cpu = sim_config_default_cpu;
	struct sim_config_s sim_gpu = sim_config_default_gpu;
	sim_cpu.hard_pulse_sim = false;
	sim_gpu.hard_pulse_sim = false;

	struct flash_config_s config = flash_config_default;
	config.npixels = 12;

	bart_dim_t num_tot_pixels = config.npixels * config.npixels;

	bart_dim_t dims[] = { config.npixels, config.npixels, 1, 1, 1, config.excitations, config.nparams, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	bart_dim_t mdims[N];
	md_select_dims(N, ~TE_FLAG & ~COEFF_FLAG, mdims, dims);

	bart_dim_t odims[N];
	md_select_dims(N, ~COEFF_FLAG, odims, dims);

	bart_dim_t sdims[N];
	md_copy_dims(N, sdims, odims);

	bart_dim_t pdims[N];
	md_select_dims(N, ~TE_FLAG, pdims, dims);

	complex float par[num_tot_pixels * config.nparams];
	for (int i = 0; i < num_tot_pixels; i++) {

		par[i + 0 * num_tot_pixels] = config.m0;
		par[i + 1 * num_tot_pixels] = config.r1;
		par[i + 2 * num_tot_pixels] = config.r2;
		par[i + 3 * num_tot_pixels] = config.b1;
		par[i + 4 * num_tot_pixels] = config.b0;
	}

	const struct nlop_s* nlop_cpu = nlop_phy_create(N, mdims, odims, config, sim_cpu);
	const struct nlop_s* nlop_gpu = nlop_gpu_wrapper_F(nlop_phy_create(N, mdims, odims, config, sim_gpu));
	
	complex float sig_cpu[md_calc_size(N, sdims)];
	complex float sig_gpu[md_calc_size(N, sdims)];

	nlop_apply(nlop_cpu, N, sdims, sig_cpu, N, pdims, par);
	nlop_apply(nlop_gpu, N, sdims, sig_gpu, N, pdims, par);

	// Simulate derivatives and adjoints
	complex float dsig_cpu[md_calc_size(N, sdims)];
	complex float dsig_gpu[md_calc_size(N, sdims)];

	complex float dpar[md_calc_size(N, pdims)];
	md_gaussian_rand(N, pdims, dpar);

	nlop_derivative(nlop_cpu, N, sdims, dsig_cpu, N, pdims, dpar);
	nlop_derivative(nlop_gpu, N, sdims, dsig_gpu, N, pdims, dpar);

	complex float dpar_adj_cpu[md_calc_size(N, pdims)];
	complex float dpar_adj_gpu[md_calc_size(N, pdims)];
	complex float dsig_adj[md_calc_size(N, sdims)];

	md_gaussian_rand(N, sdims, dsig_adj);

	nlop_adjoint(nlop_cpu, N, pdims, dpar_adj_cpu, N, sdims, dsig_adj);
	nlop_adjoint(nlop_gpu, N, pdims, dpar_adj_gpu, N, sdims, dsig_adj);

	nlop_free(nlop_cpu);
	nlop_free(nlop_gpu);

	// Assert
	UT_RETURN_ON_FAILURE_TOL(md_zrmse(N, sdims, sig_cpu, sig_gpu), 1.e-4);
	UT_RETURN_ON_FAILURE_TOL(md_zrmse(N, sdims, dsig_cpu, dsig_gpu), 1.e-3);
	UT_RETURN_ON_FAILURE_TOL(md_zrmse(1, (bart_dim_t[]) { 5 }, dpar_adj_cpu, dpar_adj_gpu), 1.e-3);

	return true;
}

UT_GPU_REGISTER_TEST(test_cuda_ir_flash_nlop_phy_create_sim_pulses_der);

static bool test_cuda_ir_flash_nlop_phy_create_hard_pulse_der(void)
{
	struct flash_config_s config = flash_config_default;
	config.npixels = 12;

	bart_dim_t num_tot_pixels = config.npixels * config.npixels;

	bart_dim_t dims[] = { config.npixels, config.npixels, 1, 1, 1, config.excitations, config.nparams, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int N = ARRAY_SIZE(dims);

	bart_dim_t mdims[N];
	md_select_dims(N, ~TE_FLAG & ~COEFF_FLAG, mdims, dims);

	bart_dim_t odims[N];
	md_select_dims(N, ~COEFF_FLAG, odims, dims);

	bart_dim_t sdims[N];
	md_copy_dims(N, sdims, odims);

	bart_dim_t pdims[N];
	md_select_dims(N, ~TE_FLAG, pdims, dims);

	complex float par[num_tot_pixels * config.nparams];
	for (int i = 0; i < num_tot_pixels; i++) {

		par[i + 0 * num_tot_pixels] = config.m0;
		par[i + 1 * num_tot_pixels] = config.r1;
		par[i + 2 * num_tot_pixels] = config.r2;
		par[i + 3 * num_tot_pixels] = config.b1;
		par[i + 4 * num_tot_pixels] = config.b0;
	}

	const struct nlop_s* nlop_cpu = nlop_phy_create(N, mdims, odims, config, sim_config_default_cpu);
	const struct nlop_s* nlop_gpu = nlop_gpu_wrapper_F(nlop_phy_create(N, mdims, odims, config, sim_config_default_gpu));
	
	complex float sig_cpu[md_calc_size(N, sdims)];
	complex float sig_gpu[md_calc_size(N, sdims)];

	nlop_apply(nlop_cpu, N, sdims, sig_cpu, N, pdims, par);
	nlop_apply(nlop_gpu, N, sdims, sig_gpu, N, pdims, par);

	// Simulate derivatives and adjoints
	complex float dsig_cpu[md_calc_size(N, sdims)];
	complex float dsig_gpu[md_calc_size(N, sdims)];

	complex float dpar[md_calc_size(N, pdims)];
	md_gaussian_rand(N, pdims, dpar);

	nlop_derivative(nlop_cpu, N, sdims, dsig_cpu, N, pdims, dpar);
	nlop_derivative(nlop_gpu, N, sdims, dsig_gpu, N, pdims, dpar);

	complex float dpar_adj_cpu[md_calc_size(N, pdims)];
	complex float dpar_adj_gpu[md_calc_size(N, pdims)];
	complex float dsig_adj[md_calc_size(N, sdims)];

	md_gaussian_rand(N, sdims, dsig_adj);

	nlop_adjoint(nlop_cpu, N, pdims, dpar_adj_cpu, N, sdims, dsig_adj);
	nlop_adjoint(nlop_gpu, N, pdims, dpar_adj_gpu, N, sdims, dsig_adj);

	nlop_free(nlop_cpu);
	nlop_free(nlop_gpu);

	// Assert
	UT_RETURN_ON_FAILURE_TOL(md_zrmse(N, sdims, sig_cpu, sig_gpu), 1.e-4 );
	UT_RETURN_ON_FAILURE_TOL(md_zrmse(N, sdims, dsig_cpu, dsig_gpu), 1.e-3);
	UT_RETURN_ON_FAILURE_TOL(md_zrmse(1, (bart_dim_t[]) { 5 }, dpar_adj_cpu, dpar_adj_gpu), 1.e-3);

	return true;
}

UT_GPU_REGISTER_TEST(test_cuda_ir_flash_nlop_phy_create_hard_pulse_der);