/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <assert.h>
#include <complex.h>
#include <stdio.h>
#include <math.h>

#include "misc/mri.h"
#include "misc/types.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/shrdptr.h"
#include "misc/list.h"

#include "num/iovec.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/multiplace.h"
#include "num/ops.h"
#include "num/ode.h"
#include "num/loop.h"

#include "seq/pulse.h"

#include "simu/bloch.h"
#include "simu/simulation.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/sum.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/const.h"
#include "nlops/zexp.h"
#include "nlops/ztrigon.h"
#include "nlops/someops.h"
#include "nlops/tenmul.h"
#include "nlops/stack.h"
#include "nlops/nlop_jacobian.h"
#include "nlops/snlop.h"
#include "nlops/smath.h"
#include "moba/moba.h"


#ifdef USE_CUDA
#include "num/gpuops.h"
#include "simu/gpu_bloch.h"
#endif

#include "nlop_seq.h"
#include "nlop_seq_flash.h"

struct flash_config_s flash_config_default = {

	.N = DIMS,
	.npixels = 1,
	.nparams = 5, // M0, R1, R2, B1, B0
	.excitations = 100,
	.inv = true,

	.m0 = 1.,
	.r1 = 1.,
	.r2 = 0.,
	.b1 = 1.,
	.b0 = 0.,		

	.TI = 0.1,
	.TE = 0.0018,
	.TR = 0.05,		
	.flip_angle = 30.,

	.rf_duration = 0.001,
};

//in: mag_in, par
//out: mag_out, mag_read
struct list_s* flash_kern_ops_create(struct sim_config_s sim, struct flash_config_s config)
{
	list_t ret = list_create();

	if (sim.hard_pulse_sim)
		list_append(ret, (struct nlop_s*)nlop_hard_pulse_create(sim, true, DEG2RAD(config.flip_angle), 0));
	else {

		float grad[3] = { 0., 0., 0. }; // None-selective

		struct pulse_sinc ps = pulse_sinc_defaults;
		pulse_sinc_init(&ps, config.rf_duration, config.flip_angle, 0., ps.bwtp, ps.alpha);

		list_append(ret, (struct nlop_s*)nlop_pulse_create(sim, CAST_UP(&ps), 0., grad));
	}

	list_append(ret, (struct nlop_s*)nlop_relax_create(sim, config.TE, (float[3]){ 0., 0., 0. }));
	list_append(ret, (struct nlop_s*)nlop_adc_create(sim, -1, MD_BIT(sim.MO_DIM), 0));
	list_append(ret, (struct nlop_s*)nlop_relax_create(sim, config.TR - config.TE, (float[3]){ 0., 0., 0. }));
	list_append(ret, (struct nlop_s*)nlop_spoile_create(sim));

	return ret;
}

//in: mag_in, par
//out: mag_out, mag_read
struct list_s* flash_ops_create(struct sim_config_s sim, struct flash_config_s config)
{
	list_t ret = list_create();

	for (int i = 0; i < config.excitations; i++)
		list_merge(ret, flash_kern_ops_create(sim, config), true);

	return ret;
}

//in: mag_in, par
//out: mag_out, mag_read
struct list_s* ir_flash_ops_create(struct sim_config_s sim, struct flash_config_s config)
{
	list_t ret = list_create();

	if (sim.hard_pulse_sim)
		list_append(ret, (struct nlop_s*)nlop_hard_pulse_create(sim, false, DEG2RAD(180), 0));
	else {
		float grad[3] = { 0., 0., 0. }; // None-selective

		struct pulse_hypsec ps = pulse_hypsec_defaults;
		pulse_hypsec_init(GYRO, &ps);

		list_append(ret, (struct nlop_s*)nlop_pulse_create(sim, CAST_UP(&ps), 0., grad));
		list_append(ret, (struct nlop_s*)nlop_spoile_create(sim));
	}

	list_append(ret, (struct nlop_s*)nlop_relax_create(sim, config.TI, (float[3]) { 0., 0., 0. }));

	for (int i = 0; i < config.excitations; i++)
		list_merge(ret, flash_kern_ops_create(sim, config), true);

	return ret;
}

struct nlop_s* nlop_phy_create(int N, const long map_dims[N], const long out_dims[N], struct flash_config_s config, struct sim_config_s sim)
{
	sim_config_set_dims(&sim, N, map_dims, 1);

	long in_dims[N];
	md_copy_dims(N, in_dims, map_dims);
	in_dims[COEFF_DIM] = config.nparams;

	debug_printf(DP_DEBUG2, "mdims: ");
	debug_print_dims(DP_DEBUG2, DIMS, map_dims);
	debug_printf(DP_DEBUG2, "odims: ");
	debug_print_dims(DP_DEBUG2, DIMS, out_dims);
	debug_printf(DP_DEBUG2, "idims: ");
	debug_print_dims(DP_DEBUG2, DIMS, in_dims);
	debug_printf(DP_DEBUG2, "Use hard pulse simulation: %s\n", sim.hard_pulse_sim ? "true" : "false");
	debug_printf(DP_DEBUG2, "Simulate: %s\n", config.inv ? "IR-FLASH" : "FLASH");

	/*
	inputs: 2
	[ 16  16   1   1   1   1   1   1   1   1   1   1   1   1   1   3 ] -> Input magnetization: Mx, My, Mz
	[ 16  16   1   1   1   1   1   1   1   1   1   1   1   1   1   4 ] -> Input parameters: R1, R2, B1, B0
	outputs: 2
	[ 16  16   1   1   1   1   1   1   1   1   1   1   1   1   3   1 ] -> Output magnetization: Mx, My, Mz
	[ 16  16   1   1   1  10   1   1   1   1   1   1   1   1   1   1 ] -> Output signal for NR (= 10) excitations
	*/
	const struct nlop_s* ret;
	if(config.inv)
		ret = nlop_seq_from_blocks_jac_create_F(sim, ir_flash_ops_create(sim, config));
	else
		ret = nlop_seq_from_blocks_jac_create_F(sim, flash_ops_create(sim, config));

	/*
	Remove input magnetization (=> set to (0 0 1)), only keep parameters
	inputs: 1
	[ 16  16   1   1   1   1   1   1   1   1   1   1   1   1   1   4 ]
	outputs: 2
	[ 16  16   1   1   1   1   1   1   1   1   1   1   1   1   3   1 ]
	[ 16  16   1   1   1  10   1   1   1   1   1   1   1   1   1   1 ]
	*/
	ret = sim_nlop_set_init(sim, ret);

	const struct iovec_s* dom = nlop_generic_domain(ret, 0);
	ret = nlop_prepend_FF(nlop_from_linop_F(linop_zreal_create(dom->N, dom->dims)), ret, 0);

	/*
	Remove output magnetization, we are only interested in the output signal
	inputs: 1
	[ 16  16   1   1   1   1   1   1   1   1   1   1   1   1   1   4 ]
	outputs: 1
	[ 16  16   1   1   1  10   1   1   1   1   1   1   1   1   1   1 ]
	*/
	ret = nlop_del_out_F(ret, 0);

	assert(md_check_equal_dims(N, out_dims, nlop_codomain(ret)->dims, ~0UL));

	// out_dims = [ 16  16  1  1  1  10  1  1  1  1  1  1  1  1  1  1 ]
	// map_dims = [ 16  16  1  1  1   1  1  1  1  1  1  1  1  1  1  1 ]
	ret = nlop_prepend_FF(ret, nlop_tenmul_create(N, out_dims, out_dims, map_dims), 0); // in pars, m0; out: sig * m0

	ret = nlop_stack_inputs_F(ret, 1, 0, sim.PI_DIM); // input: [m0, pars] stacked

	// Move parameters to COEFF_DIM
	/*
	inputs: 1
	[ 16  16   1   1   1   1   5   1   1   1   1   1   1   1   1   1 ]
	outputs: 1
	[ 16  16   1   1   1  10   1   1   1   1   1   1   1   1   1   1 ]
	*/
	ret = nlop_chain_FF(nlop_from_linop_F(linop_transpose_create(N, sim.PI_DIM, COEFF_DIM, in_dims)), ret);

	assert(md_check_equal_dims(N, in_dims, nlop_domain(ret)->dims, ~0UL));

	nlop_debug(DP_DEBUG2, ret);

	return (struct nlop_s*)ret;
}