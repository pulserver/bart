/* Copyright 2022-2026. TU Graz. Institute of Biomedical Imaging.
 * Copyright 2026. Department of Radiology. Boston Children's Hospital.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author:
 *	Nick Scholand
 *	Markus Huemer
 *	Moritz Blumenthal
 */

#include <complex.h>
#include <math.h>
#include <assert.h>

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/cast.h"
#include "nlops/snlop.h"
#include "nlops/smath.h"
#include "nlops/cast.h"

#include "linops/someops.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "noir/model.h"
#include "noir/utils.h"

#include "moba/blochfun.h"
#include "moba/T1phyfun.h"
#include "moba/meco.h"
#include "moba/moba.h"
#include "moba/lorentzian.h"
#include "moba/exp.h"
#include "moba/T1fun.h"
#include "moba/utils.h"

#include "simu/signals.h"

#include "model_moba.h"


struct mobamod moba_create(const long dims[DIMS], const complex float* TI, const complex float* TE, const complex float* b1,
		const complex float* b0, const float* scale_fB0, enum meco_model meco_model, enum fat_spec fat_spec, const long psf_dims[DIMS], const complex float* psf, const long coil_dims[DIMS], complex float* coil, const struct noir_model_conf_s* conf, struct moba_conf_s* data,
		float scaling_M0, const complex float* fixed_maps)
{
	long data_dims[DIMS];
	md_select_dims(DIMS, ~COEFF_FLAG, data_dims, dims);

	struct noir_model_conf_s mconf = *conf;
	mconf.sobolev_os = data->other.sobolev_os;

	struct noir_s nlinv = noir_create(data_dims, coil_dims, coil, psf_dims, psf, &mconf);
	struct mobamod ret;

	// FIXME: unify them more
	long out_dims[DIMS];
	long in_dims[DIMS];
	long map_dims[DIMS];
	long TI_dims[DIMS];
	long TE_dims[DIMS];

	md_select_dims(DIMS, FFT_FLAGS|SLICE_FLAG|TE_FLAG|CSHIFT_FLAG|TIME_FLAG|TIME2_FLAG, out_dims, dims);
	md_select_dims(DIMS, FFT_FLAGS|SLICE_FLAG|COEFF_FLAG|TIME_FLAG|TIME2_FLAG, in_dims, dims);
	md_select_dims(DIMS, TE_FLAG|TIME_FLAG|TIME2_FLAG, TI_dims, dims);
	md_select_dims(DIMS, CSHIFT_FLAG|TIME_FLAG|TIME2_FLAG, TE_dims, dims);
	md_select_dims(DIMS, ~COEFF_FLAG, map_dims, in_dims);

	long out_dims2[DIMS];
	long in_dims2[DIMS];
	long map_dims2[DIMS];

	float fov = data->other.fov_reduction_factor;

	for (int i = 0; i < DIMS; i++) {

		out_dims2[i] = (1 < out_dims[i] && 3 > i) ? out_dims[i] * fov : out_dims[i];
		in_dims2[i] = (1 < in_dims[i] && 3 > i) ? in_dims[i] * fov : in_dims[i];
		map_dims2[i] = (1 < map_dims[i] && 3 > i) ? map_dims[i] * fov : map_dims[i];
	}

	struct nlop_s* model = NULL;

	int NC = in_dims[COEFF_DIM];
	assert((int)ARRAY_SIZE(ret.linop_sobolev) >= NC);

	for (int i = 0; i < (int)ARRAY_SIZE(ret.linop_sobolev); i++)
		ret.linop_sobolev[i] = NULL;

	unsigned long sobolev_trafo_flags = mconf.sos ? FFT_FLAGS | SLICE_FLAG : FFT_FLAGS;

	switch (data->model) {

	case MDB_MGRE:

		if (MECO_PI == meco_model) {

			model = nlop_from_linop_F(linop_identity_create(DIMS, out_dims2));
			break;
		}

		if (0. != scale_fB0[0])
			ret.linop_sobolev[NC - 1] = linop_chain_FF(linop_noir_weights_create(DIMS, map_dims, map_dims, NULL, sobolev_trafo_flags, data->other.sobolev_os, scale_fB0[0], scale_fB0[1], 1), linop_zreal_create(DIMS, map_dims));

		model = nlop_meco_create(DIMS, out_dims2, in_dims2, TI/*TI is used as TE*/, meco_model, fat_spec);

		break;

	case MDB_T1:

		model = nlop_T1_create(DIMS, out_dims2, in_dims2, TI_dims, TI, scaling_M0);

		break;

	case MDB_T2:

		complex float* enc = md_alloc_sameplace(DIMS, TI_dims, CFL_SIZE, TI);

		md_zsmul(DIMS, TI_dims, enc, TI, -1.);
		model = nlop_exp_create(DIMS, out_dims2, enc);

		md_free(enc);

		break;

	case MDB_T1_PHY:

		if (0. != data->other.b1_sobolev_a)
			ret.linop_sobolev[2] = linop_chain_FF(linop_noir_weights_create(DIMS, map_dims, map_dims, NULL, sobolev_trafo_flags, data->other.sobolev_os, data->other.b1_sobolev_a, data->other.b1_sobolev_b, 1.), linop_zreal_create(DIMS, map_dims));

		model = nlop_T1_phy_create(DIMS, out_dims2, in_dims2, TI_dims, TI, data);
		break;

	case MDB_IR_MGRE:

		if (0. != scale_fB0[0])
			ret.linop_sobolev[NC - 1] = linop_chain_FF(linop_noir_weights_create(DIMS, map_dims, map_dims, NULL, sobolev_trafo_flags, data->other.sobolev_os, scale_fB0[0], scale_fB0[1], 1), linop_zreal_create(DIMS, map_dims));

		model = nlop_ir_meco_create(DIMS, out_dims2, in_dims2, TI_dims, TI, TE_dims, TE, meco_model, fat_spec);
		break;

	case MDB_BLOCH:

		if (0. != data->other.b1_sobolev_a)
			ret.linop_sobolev[3] = linop_chain_FF(linop_noir_weights_create(DIMS, map_dims, map_dims, NULL, sobolev_trafo_flags, data->other.sobolev_os, data->other.b1_sobolev_a, data->other.b1_sobolev_b, 1.), linop_zreal_create(DIMS, map_dims));

		// Turn off matching of T2 for IR FLASH

		if (SEQ_IRFLASH == data->sim.seq.seq_type)
			data->other.scale[2] = 0.;

		complex float* b1_2 = NULL;
		complex float* b0_2 = NULL;

		if (NULL != b1) {

			b1_2 = md_alloc_sameplace(DIMS, map_dims2, CFL_SIZE, b1);
			md_resize_center(DIMS, map_dims2, b1_2, map_dims, b1, CFL_SIZE);
		}

		if (NULL != b0) {

			b0_2 = md_alloc_sameplace(DIMS, map_dims2, CFL_SIZE, b0);
			md_resize_center(DIMS, map_dims2, b0_2, map_dims, b0, CFL_SIZE);
		}

		model = nlop_bloch_create(DIMS, out_dims2, in_dims2, b1_2, b0_2, data);

		md_free(b1_2);
		md_free(b0_2);

		break;
	}

	if (!md_check_equal_dims(DIMS, out_dims, out_dims2, ~0UL))
		model = nlop_chain_FF(model, nlop_from_linop_F(linop_resize_center_create(DIMS, out_dims, out_dims2)));

	if (!md_check_equal_dims(DIMS, in_dims, in_dims2, ~0UL))
		model = nlop_chain_FF(nlop_from_linop_F(linop_resize_center_create(DIMS, in_dims2, in_dims)), model);

	for (int i = 0; i < NC; i++)
		debug_printf(DP_DEBUG2, "FP Scale[%d]=%f\n", i, crealf(data->other.scale[i]));

	model = nlop_chain_FF(moba_precond_create(DIMS, in_dims, ret.linop_sobolev, data->other.scale, data->other.initval, fixed_maps), model);

	debug_printf(DP_INFO, "Physics-");
	nlop_debug(DP_INFO, model);
	debug_printf(DP_INFO, "Encoding-");
	nlop_debug(DP_INFO, nlinv.nlop);

	const struct nlop_s* b = nlinv.nlop;

	// Turn off coil derivative
	if (data->other.no_sens_deriv && !data->other.fixed_coil)
		b = nlop_no_der_F(b, 0, 1);

	b = nlop_prepend_FF(model, b, 0);

	ret.nlop = nlop_flatten_F(b);
	ret.linop = nlinv.linop;

	return ret;
}

const struct nlop_s* moba_get_nlop(struct mobafit_model_config* config, const long out_dims[DIMS], const long param_dims[DIMS], const long enc_dims[DIMS], complex float* enc)
{
	const struct nlop_s* nlop = NULL;
	int n_params = param_dims[COEFF_DIM];

	assert(md_check_compat(DIMS, ~0UL, param_dims, out_dims));

	long dims[DIMS];
	md_copy_dims(DIMS, dims, out_dims);
	dims[COEFF_DIM] = enc_dims[COEFF_DIM];

	if (config->seq == TSE)
		md_zsmul(DIMS, enc_dims, enc, enc, -1.);

	switch (config->seq) {

	case IR:

		if (n_params  != 3)
			error("Number of parameters (%d) does not match IR model (M0, R1, c)\n", n_params);

		nlop = nlop_ir_create(DIMS, out_dims, enc);
		break;

	case IR_LL:

		if (n_params  != 3)
			error("Number of parameters (%d) does not match IR-LL model (Mss, M0, R1s)\n", n_params);

		nlop = nlop_T1_create(DIMS, out_dims, param_dims, enc_dims, enc, 1.);
		break;

	case MGRE:

		nlop = nlop_meco_create(DIMS, out_dims, param_dims, enc, config->mgre_model, FAT_SPEC_1);
		break;

	case TSE:

		if (n_params  != 2)
			error("Number of parameters (%d) does not match TSE model\n", n_params);

		nlop = nlop_exp_create(DIMS, dims, enc);
		break;

	case DIFF:


		if (n_params  != enc_dims[COEFF_DIM] + 1)
			error("Number of parameters (%d) does not match diffusion model \n", n_params);

		nlop = nlop_exp_create(DIMS, dims, enc);
		break;

	case MPL:
		// M0 exists once, every pool has 3 parameters, we need at least one pool >3 parameters
		if ((n_params < 4) || ((n_params - 1) % 3 != 0))
			error("Number of parameters (%d) does not match MPL model\n", n_params);

		nlop = nlop_lorentzian_multi_pool_create(DIMS, out_dims, param_dims, enc_dims, enc);
		break;

	default:
		assert(0);
	}

	return nlop;
}


// Simple phase evolution. Not integrated moba_get_nlop becaus it requires the signal itself.
const struct nlop_s* mobafit_phase_nlop(const long out_dims[DIMS], const complex float* sig, const long enc_dims[DIMS], complex float* enc)
{
	long map_dims[DIMS];
	md_select_dims(DIMS, ~TE_FLAG, map_dims, out_dims);

	arg_t args[2] = { snlop_input(DIMS, map_dims, "Phi0"), snlop_input(DIMS, map_dims, "fB0") };

	arg_t TE = snlop_const(DIMS, enc_dims, enc, "TE");

	arg_t out = snlop_mul_F(snlop_real(args[1]), TE, 0);
	out = snlop_add_F(out, snlop_real(args[0]));
	out = snlop_scale_F(out, 2.i * M_PI);
	out = snlop_exp_F(out);

	complex float* mag = md_alloc_sameplace(DIMS, out_dims, CFL_SIZE, sig);
	md_zabs(DIMS, out_dims, mag, sig);

	out = snlop_mul_F(out, snlop_const(DIMS, map_dims, mag, "mag"), 0);

	md_free(mag);

	const struct nlop_s* ret = nlop_from_snlop_F(snlop_from_arg(out),
							1, (arg_t[1]){ out },
							2, args);

	return nlop_stack_inputs_F(ret, 0, 1, COEFF_DIM);
}

static void mobafit_phase_average(int N, const long dims[N], complex float* fB0, const complex float* sig, const long TE_dims[N], complex float* TE)
{
	assert(TE_DIM < N);

	long pos[N];
	md_set_dims(N, pos, 0);
	pos[TE_DIM] = -1;

	complex float* dTE = md_alloc_sameplace(N, TE_dims, CFL_SIZE, TE);
	md_circ_shift(N, TE_dims, pos, dTE, TE, CFL_SIZE);
	md_zsub(N, TE_dims, dTE, dTE, TE);

	complex float* dsig = md_alloc_sameplace(N, dims, CFL_SIZE, sig);
	md_circ_shift(N, dims, pos, dsig, sig, CFL_SIZE);
	md_zmulc(N, dims, dsig, dsig, sig);

	long TE_strs[N];
	long sig_strs[N];

	md_calc_strides(N, TE_strs, TE_dims, CFL_SIZE);
	md_calc_strides(N, sig_strs, dims, CFL_SIZE);

	long sTE_dims[N];
	md_select_dims(N, ~TE_FLAG, sTE_dims, TE_dims);
	pos[TE_DIM] = 0;

	complex float* dTE_tmp = md_alloc_sameplace(N, sTE_dims, CFL_SIZE, TE);
	md_copy_block(N, pos, sTE_dims, dTE_tmp, TE_dims, dTE, CFL_SIZE);
	float nrm = md_znorm(N, sTE_dims, dTE_tmp);

	for (pos[TE_DIM] = 0; pos[TE_DIM] < TE_dims[TE_DIM]; pos[TE_DIM]++) {

		md_zsub2(N, sTE_dims, MD_STRIDES(N, sTE_dims, CFL_SIZE), dTE_tmp, TE_strs, MD_ACCESS_PTR(N, TE_strs, pos, dTE), TE_strs, dTE);
		float nrmse = md_znorm(N, sTE_dims, dTE_tmp) / nrm;

		if (1.e-5 < nrmse)
			break;
	}

	md_free(dTE_tmp);

	long avg_dims[N];
	md_copy_dims(N, avg_dims, dims);
	avg_dims[TE_DIM] = pos[TE_DIM];

	debug_printf(DP_INFO, "Use first %ld echos to initialize fB0.\n", pos[TE_DIM] + 1);

	long fB0_dims[N];
	long fB0_strs[N];
	md_select_dims(N, ~TE_FLAG, fB0_dims, dims);
	md_calc_strides(N, fB0_strs, fB0_dims, CFL_SIZE);

	md_clear(N, fB0_dims, fB0, CFL_SIZE);
	md_zadd2(N, avg_dims, fB0_strs, fB0, fB0_strs, fB0, sig_strs, dsig);

	md_zarg(N, fB0_dims, fB0, fB0);

	md_zspow(N, TE_dims, dTE, dTE, -1.);
	md_zsmul(N, TE_dims, dTE, dTE, 1. / (2. * M_PI));

	md_zmul2(N, fB0_dims, fB0_strs, fB0, fB0_strs, fB0, TE_strs, dTE);

	md_free(dTE);
	md_free(dsig);
}


void mobafit_phase_init(enum seq_type seq, const long coeff_dims[DIMS], complex float* init, const long sig_dims[DIMS], const complex float* sig, const long enc_dims[DIMS], complex float* enc)
{
	if (PHASE != seq && MGRE != seq)
		error("Phase initialization only available for phase contrast and MGRE models");

	long map_dims[DIMS];
	md_select_dims(DIMS, ~TE_FLAG, map_dims, sig_dims);

	complex float* fB0 = md_alloc_sameplace(DIMS, map_dims, CFL_SIZE, sig);
	mobafit_phase_average(DIMS, sig_dims, fB0, sig, enc_dims, enc);

	long pos[DIMS] = { 0 };
	pos[COEFF_DIM] = coeff_dims[COEFF_DIM] - 1; // fB0 is always the last coefficient
	md_copy_block(DIMS, pos,coeff_dims, init, map_dims, fB0, CFL_SIZE);

	// also init phase at TE=0
	if (PHASE == seq) {

		long sig_strs[DIMS];
		long coeff_strs[DIMS];
		long map_strs[DIMS];

		md_calc_strides(DIMS, sig_strs, sig_dims, CFL_SIZE);
		md_calc_strides(DIMS, coeff_strs, coeff_dims, CFL_SIZE);
		md_calc_strides(DIMS, map_strs, map_dims, CFL_SIZE);

		md_zarg2(DIMS, map_dims, coeff_strs, init, sig_strs, sig);
		md_zsmul2(DIMS, map_dims, coeff_strs, init, coeff_strs, init, 1. / (2. * M_PI));

		long enc_strs[DIMS];
		md_calc_strides(DIMS, enc_strs, enc_dims, CFL_SIZE);

		md_zsmul(DIMS, map_dims, fB0, fB0, -1.);
		md_zfmac2(DIMS, map_dims, coeff_strs, init, map_strs, fB0, enc_strs, enc);
	}

	md_free(fB0);
}



