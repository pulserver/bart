/* Copyright 2020-2021. Uecker Lab, University Medical Center Goettingen.
 * Copyright 2022-2025. Institute of Biomedical Imaging. TU Graz.
 * Copyright 2026. Department of Radiology. Boston Children's Hospital.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2019-2020 Martin Uecker
 * 2019-2020 Zhengguo Tan
 * 2026 Moritz Blumenthal
 */

#include <complex.h>
#include <math.h>
#include <stdio.h>

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/types.h"
#include "misc/version.h"

#include "num/gpuops.h"
#include "num/filter.h"
#include "num/flpmath.h"
#include "num/multind.h"
#include "num/multiplace.h"
#include "num/iovec.h"

#include "simu/signals.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/cast.h"
#include "nlops/snlop.h"
#include "nlops/smath.h"
#include "nlops/nlop_jacobian.h"

#include "noir/model.h"
#include "noir/utils.h"

#include "moba/utils.h"

#include "meco.h"


struct meco_old_phasecontrast_s {

	nlop_data_t super;

	struct multiplace_array_s* TE;
};

DEF_TYPEID(meco_old_phasecontrast_s);

// ************************************************************* //
//  Model: rho .* exp(i 2\pi fB0 TE) reproducing wrong scaling from old version
// ************************************************************* //
static void meco_fun_phasediff(const nlop_data_t* _data, int N, const long y_dims[N], _Complex float* dst, const long x_dims[N], const _Complex float* src, const long ddims[N], _Complex float* jac)
{
	struct meco_old_phasecontrast_s* data = CAST_DOWN(meco_old_phasecontrast_s, _data);

	long map_dims[N];
	long TE_dims[N];

	md_select_dims(N, TE_FLAG, TE_dims, y_dims);
	md_select_dims(N, ~COEFF_FLAG, map_dims, x_dims);

	long pos[N];
	md_set_dims(N, pos, 0);

	complex float* tmp_exp = md_alloc_sameplace(N, y_dims, CFL_SIZE, dst);

	complex float* rho = md_alloc_sameplace(N, map_dims, CFL_SIZE, dst);
	complex float* fB0 = md_alloc_sameplace(N, map_dims, CFL_SIZE, dst);

	md_copy_block(N, (pos[COEFF_DIM] = 0, pos), map_dims, rho, x_dims, src, CFL_SIZE);
	md_copy_block(N, (pos[COEFF_DIM] = 1, pos), map_dims, fB0, x_dims, src, CFL_SIZE);

	//exp (i 2\pi fB0 TE) reproducing wrong scaling from old version
	md_zsmul(N, map_dims, fB0, fB0, 1. + 2.i * M_PI);

	md_ztenmul(N, y_dims, tmp_exp, TE_dims, multiplace_read(data->TE, dst), map_dims, fB0);
	md_zexp(N, y_dims, tmp_exp, tmp_exp);

	md_ztenmul(N, y_dims, dst, y_dims, tmp_exp, map_dims, rho);


	if (NULL != jac) {

		md_copy_block(N, (pos[COEFF_DIM] = 0, pos), ddims, jac, y_dims, tmp_exp, CFL_SIZE);

		complex float* tmp_eco = md_alloc_sameplace(N, y_dims, CFL_SIZE, dst);

		md_ztenmul(N, y_dims, tmp_eco, y_dims, dst, TE_dims, multiplace_read(data->TE, dst));
		md_zsmul(N, y_dims, tmp_eco, tmp_eco, 2.i * M_PI);

		md_copy_block(N, (pos[COEFF_DIM] = 1, pos), ddims, jac, y_dims, tmp_eco, CFL_SIZE);

		md_free(tmp_eco);
	}

	md_free(tmp_exp);

	md_free(rho);
	md_free(fB0);
}

static void meco_del(const nlop_data_t* _data)
{
	struct meco_old_phasecontrast_s* data = CAST_DOWN(meco_old_phasecontrast_s, _data);

	multiplace_free(data->TE);

	xfree(data);
}

static struct nlop_s* nlop_meco_old_phase_constrast_create(const int N, const long x_dims[N], const long TE_dims[N], const complex float* TE)
{
	PTR_ALLOC(struct meco_old_phasecontrast_s, data);
	SET_TYPEID(meco_old_phasecontrast_s, data);

	data->TE = multiplace_move(N, TE_dims, CFL_SIZE, TE);

	long y_dims[N];
	md_max_dims(N, ~0UL, y_dims, x_dims, TE_dims);
	md_select_dims(N, ~COEFF_FLAG, y_dims, y_dims);

	long ddims[N];
	assert(md_check_compat(N, ~0UL, y_dims, x_dims));
	md_max_dims(N, ~0UL, ddims, y_dims, x_dims);

	return nlop_zblock_diag_create(CAST_UP(PTR_PASS(data)), N, y_dims, x_dims, ddims, meco_fun_phasediff, meco_del);
}




int get_num_of_coeff(enum meco_model sel_model)
{
	switch (sel_model) {
	case MECO_WF: 			return 3;
	case MECO_WFR2S:		return 4;
	case MECO_WF2R2S:		return 5;
	case MECO_R2S:			return 3;
	case MECO_PHASEDIFF:		return 2;
	case IR_MECO_T1_R2S:		return 5; // ir + meco, water T1, R2*, fB0
	case IR_MECO_W_T1_F_T1_R2S:	return 8; // ir + meco, water T1, fat T1, R2*, fB0
	default:
		assert(0);
	}
}

unsigned long get_PD_flag(enum meco_model sel_model)
{
	switch (sel_model) {
	case MECO_WF:		return MD_BIT(0) | MD_BIT(1);
	case MECO_WFR2S:	return MD_BIT(0) | MD_BIT(1);
	case MECO_WF2R2S:	return MD_BIT(0) | MD_BIT(2);
	case MECO_R2S:		return MD_BIT(0);
	case MECO_PHASEDIFF:	return MD_BIT(0);
	default:
		assert(0);
	}
}

unsigned long get_R2S_flag(enum meco_model sel_model)
{
	switch (sel_model) {

	case MECO_WF:			return 0;
	case MECO_WFR2S:		return MD_BIT(2);
	case MECO_WF2R2S:		return MD_BIT(1) | MD_BIT(3);
	case MECO_R2S:			return MD_BIT(1);
	case MECO_PHASEDIFF:		return 0;
	case IR_MECO_T1_R2S:		return MD_BIT(3);
	case IR_MECO_W_T1_F_T1_R2S:	return MD_BIT(6);
	default:
		assert(0);
	}
}

static unsigned long get_R1S_flag(enum meco_model sel_model)
{
	switch (sel_model) {
	case IR_MECO_T1_R2S:		return MD_BIT(2);
	case IR_MECO_W_T1_F_T1_R2S:	return MD_BIT(5);
	default:			return 0;
	}
}

unsigned long get_fB0_flag(enum meco_model sel_model)
{
	// fB0 is always in the last position
	return MD_BIT(get_num_of_coeff(sel_model) - 1);
}

static void calc_fat_modu(int N, const long dims[N], complex float* dst, const complex float* TE, enum fat_spec fat_spec)
{
	assert(1 == bitcount(md_nontriv_dims(N, dims)));
	md_clear(N, dims, dst, CFL_SIZE);

	for (int i = 0; i < md_calc_size(N, dims); i++) {

		assert(0. == cimagf(TE[i]));

		dst[i] = calc_fat_modulation(3.0, crealf(TE[i]) * 1.E-3, fat_spec); // FIXME: TE in SI units instead ms
	}
}







// F .* zm
static arg_t fat_spectrum(arg_t F, int N, const long TE_dims[N], const complex float* TE, enum fat_spec fat_spec)
{
	complex float* cshift = md_alloc(N, TE_dims, CFL_SIZE);

	calc_fat_modu(N, TE_dims, cshift, TE, fat_spec);

	arg_t arg_cshift = snlop_const(N, TE_dims, cshift, "cshift");

	md_free(cshift);

	arg_t out = snlop_mul(F, arg_cshift, 0);
	snlop_del_arg(arg_cshift);

	return out;
}

static arg_t B0_modulation(arg_t M0, arg_t fB0, int N, const long TE_dims[N], const complex float* TE)
{
	arg_t arg_TE = snlop_const(N, TE_dims, TE, "TE");
	arg_TE = snlop_scale_F(arg_TE, 2.i * M_PI);

	arg_t tmp = snlop_mul(fB0, arg_TE, 0);
	snlop_del_arg(arg_TE);
	tmp = snlop_exp_F(tmp);

	arg_t out = snlop_mul(M0, tmp, 0);
	snlop_del_arg(tmp);

	return out;
}

static arg_t T2s_decay(arg_t M0, arg_t R2s, int N, const long TE_dims[N], const complex float* TE)
{
	arg_t arg_TE = snlop_const(N, TE_dims, TE, "TE");
	arg_TE = snlop_scale_F(arg_TE, -1.);

	arg_t tmp = snlop_mul(R2s, arg_TE, 0);
	snlop_del_arg(arg_TE);
	tmp = snlop_exp_F(tmp);

	arg_t out = snlop_mul(M0, tmp, 0);
	snlop_del_arg(tmp);

	return out;
}

static arg_t inversion_recovery(arg_t MS, arg_t M0, arg_t R1s, int N, const long TI_dims[N], const complex float* TI)
{
	arg_t arg_TI = snlop_const(N, TI_dims, TI, "TI");
	arg_TI = snlop_scale_F(arg_TI, -1.);

	arg_t tmp1 = snlop_mul(R1s, arg_TI, 0);
	snlop_del_arg(arg_TI);
	tmp1 = snlop_exp_F(tmp1);

	arg_t tmp2 = snlop_mul_F(snlop_add(MS, M0), tmp1, 0);

	arg_t out = snlop_sub(MS, tmp2);
	snlop_del_arg(tmp2);

	return out;
}





const struct nlop_s* nlop_ir_meco_model_create(int N, const long map_dims[N], const long in_dims[N], const long TI_dims[N],
				const complex float* TI, const long TE_dims[N], const complex float* TE, enum meco_model meco_model, enum fat_spec fat_spec)
{
	assert((MECO_PI != meco_model) || (get_num_of_coeff(meco_model) == in_dims[COEFF_DIM]));

	arg_t args [in_dims[COEFF_DIM]];
	arg_t out = NULL;

	arg_t tmp[6] = { NULL };

	switch (meco_model) {

	case MECO_WF:

		debug_printf(DP_DEBUG1, "MODEL: W, F, fB0\n");
		args[0] = snlop_input(N, map_dims, "W");
		args[1] = snlop_input(N, map_dims, "F");
		args[2] = snlop_input(N, map_dims, "fB0");

		tmp[0] = fat_spectrum(args[1], N, TE_dims, TE, fat_spec);
		tmp[1] = snlop_add(tmp[0], args[0]);

		out = B0_modulation(tmp[1], args[2], N, TE_dims, TE);
		break;

	case MECO_WFR2S:

		debug_printf(DP_DEBUG1, "MODEL: W, F, R2*, fB0\n");
		args[0] = snlop_input(N, map_dims, "W");
		args[1] = snlop_input(N, map_dims, "F");
		args[2] = snlop_input(N, map_dims, "R2s");
		args[3] = snlop_input(N, map_dims, "fB0");

		tmp[0] = fat_spectrum(args[1], N, TE_dims, TE, fat_spec);
		tmp[1] = snlop_add(tmp[0], args[0]);
		tmp[2] = B0_modulation(tmp[1], args[3], N, TE_dims, TE);

		out = T2s_decay(tmp[2], args[2], N, TE_dims, TE);
		break;

	case MECO_WF2R2S:

		debug_printf(DP_DEBUG1, "MODEL: W, R2*W, F, R2*F, fB0\n");
		args[0] = snlop_input(N, map_dims, "W");
		args[1] = snlop_input(N, map_dims, "R2sW");
		args[2] = snlop_input(N, map_dims, "F");
		args[3] = snlop_input(N, map_dims, "R2sF");
		args[4] = snlop_input(N, map_dims, "fB0");

		tmp[0] = fat_spectrum(args[2], N, TE_dims, TE, fat_spec);
		tmp[1] = T2s_decay(tmp[0], args[3], N, TE_dims, TE);

		tmp[3] = T2s_decay(args[0], args[1], N, TE_dims, TE);

		tmp[4] = snlop_add(tmp[1], tmp[3]);
		tmp[5] = snlop_add(tmp[3], args[4]);

		out = B0_modulation(tmp[5], args[4], N, TE_dims, TE);
		break;

	case MECO_R2S:

		debug_printf(DP_DEBUG1, "MODEL: rho, R2*, fB0\n");
		args[0] = snlop_input(N, map_dims, "rho");
		args[1] = snlop_input(N, map_dims, "R2s");
		args[2] = snlop_input(N, map_dims, "fB0");

		tmp[0] = B0_modulation(args[0], args[2], N, TE_dims, TE);
		out = T2s_decay(tmp[0], args[1], N, TE_dims, TE);
		break;

	case MECO_PHASEDIFF:

		debug_printf(DP_DEBUG1, "MODEL: rho, fB0\n");
		args[0] = snlop_input(N, map_dims, "rho");
		args[1] = snlop_input(N, map_dims, "fB0");

		out = B0_modulation(args[0], args[1], N, TE_dims, TE);
		break;

	case IR_MECO_T1_R2S:

		debug_printf(DP_DEBUG1, "MODEL: Ms, M0, R1*, R2*, fB0\n");
		args[0] = snlop_input(N, map_dims, "Ms");
		args[1] = snlop_input(N, map_dims, "M0");
		args[2] = snlop_input(N, map_dims, "R1s");
		args[3] = snlop_input(N, map_dims, "R2s");
		args[4] = snlop_input(N, map_dims, "fB0");

		tmp[0] = inversion_recovery(args[0], args[1], args[2], N, TI_dims, TI);
		tmp[1] = B0_modulation(tmp[0], args[4], N, TE_dims, TE);
		out = T2s_decay(tmp[1], args[3], N, TE_dims, TE);
		break;

	case IR_MECO_W_T1_F_T1_R2S:

		args[0] = snlop_input(N, map_dims, "Ms_w");
		args[1] = snlop_input(N, map_dims, "M0_w");
		args[2] = snlop_input(N, map_dims, "R1s_w");
		args[3] = snlop_input(N, map_dims, "Ms_f");
		args[4] = snlop_input(N, map_dims, "M0_f");
		args[5] = snlop_input(N, map_dims, "R1s_f");
		args[6] = snlop_input(N, map_dims, "R2s");
		args[7] = snlop_input(N, map_dims, "fB0");

		tmp[0] = inversion_recovery(args[0], args[1], args[2], N, TI_dims, TI);
		tmp[1] = inversion_recovery(args[3], args[4], args[5], N, TI_dims, TI);
		tmp[2] = fat_spectrum(tmp[1], N, TE_dims, TE, fat_spec);

		tmp[3] = snlop_add(tmp[0], tmp[2]);
		tmp[4] = B0_modulation(tmp[3], args[7], N, TE_dims, TE);
		out = T2s_decay(tmp[4], args[6], N, TE_dims, TE);
		break;
	default:
		error("invalid model");
	}

	for (int i = 0; i < (int)ARRAY_SIZE(tmp); i++)
		snlop_del_arg(tmp[i]);

	const struct nlop_s* ret = nlop_from_snlop_F(snlop_from_arg(out),
							1, (arg_t[1]){ out },
							in_dims[COEFF_DIM], args);

	// Stack inputs for the final result
	for (int i = 0; i < in_dims[COEFF_DIM] - 1; i++)
		ret = nlop_stack_inputs_F(ret, 0, 1, COEFF_DIM);

	// precompute jacobian wrapper, such that derivative is simple ztenmul
	// This requires the model to be holomorphic, so we impose real constraints afterwards
	ret = nlop_zprecomp_jacobian_F(ret);


	if ((MECO_PHASEDIFF == meco_model) && use_compat_to_version("v0.9.00")) {

		nlop_free(ret);
		ret = nlop_meco_old_phase_constrast_create(N, in_dims, TE_dims, TE);
	}


	unsigned long real_constraint_flag = get_R2S_flag(meco_model) | get_R1S_flag(meco_model) | get_fB0_flag(meco_model);

	float scales[in_dims[COEFF_DIM]];
	const struct linop_s* lop_rvcs[in_dims[COEFF_DIM]];

	for (int i = 0; i < in_dims[COEFF_DIM]; i++) {

		if (MD_IS_SET(real_constraint_flag, i))
			lop_rvcs[i] = linop_zreal_create(N, map_dims);
		else
			lop_rvcs[i] = NULL;

		scales[i] = 1.;
	}

	const struct linop_s* rvc = moba_precond_create(N, in_dims, lop_rvcs, scales);

	for (int i = 0; i < in_dims[COEFF_DIM]; i++)
		linop_free(lop_rvcs[i]);

	return nlop_chain_FF(nlop_from_linop_F(rvc), ret);
}

struct nlop_s* nlop_ir_meco_create(int N, const long map_dims[N], const long /*out_dims*/[N], const long in_dims[N], const long TI_dims[N],
				const complex float* TI, const long TE_dims[N], const complex float* TE, const float* scale_fB0, enum meco_model meco_model, enum fat_spec fat_spec, const float* scale)
{
	const struct nlop_s* model = nlop_ir_meco_model_create(N, map_dims, in_dims, TI_dims, TI, TE_dims, TE, meco_model, fat_spec);

	const struct linop_s* prec[in_dims[COEFF_DIM]];

	const struct linop_s* linop_fB0 = NULL;

	if (0. == scale_fB0[0]) {

		debug_printf(DP_DEBUG2, " identity weight on fB0\n");

		linop_fB0 = linop_identity_create(N, map_dims);

	} else {

		debug_printf(DP_DEBUG2, " sobolev weight on fB0\n");

		linop_fB0 = linop_noir_weights_create(N, map_dims, map_dims, map_dims, FFT_FLAGS, 1., scale_fB0[0], scale_fB0[1], 1);
	}

	linop_fB0 = linop_chain_FF(linop_fB0, linop_zreal_create(N, map_dims));

	for (int i = 0; i < in_dims[COEFF_DIM]; i++)
		prec[i] = NULL;

	prec[in_dims[COEFF_DIM] - 1] = linop_fB0;


	const struct linop_s* precond = moba_precond_create(N, in_dims, prec, scale);

	const struct nlop_s* ret = nlop_chain_FF(nlop_from_linop_F(precond), model);
	ret = moba_attach_trafo_F(ret, linop_fB0);

	for(int i = 0; i < in_dims[COEFF_DIM]; i++)
		if (NULL != prec[i])
			linop_free(prec[i]);


	return (struct nlop_s*)ret;
}

const struct linop_s* ir_meco_get_fB0_trafo(struct nlop_s* op)
{
	return moba_attach_trafo_get_linop(op);
}

const struct linop_s* meco_get_fB0_trafo(struct nlop_s* op)
{
	return moba_attach_trafo_get_linop(op);
}


struct nlop_s* nlop_meco_create(const int N, const long y_dims[N], const long x_dims[N], const complex float* TE, enum meco_model sel_model, enum fat_spec fat_spec, const float* scale_fB0)
{
	long map_dims[N];
	md_select_dims(N, ~COEFF_FLAG, map_dims, x_dims);

	long TE_dims[N];
	md_select_dims(N, TE_FLAG, TE_dims, y_dims);

	float scale[x_dims[COEFF_DIM]];
	for (long i = 0; i < x_dims[COEFF_DIM]; i++)
		scale[i] = 1.;

	const struct nlop_s* ret = nlop_ir_meco_create(N, map_dims, /*out_dims*/NULL, x_dims, /*TI_dims*/NULL,
							/*TI*/NULL, TE_dims, TE, scale_fB0, sel_model, fat_spec, /*scale*/scale);

	assert(md_check_equal_dims(N, y_dims, nlop_codomain(ret)->dims, ~0UL));

	return (struct nlop_s*)ret;
}



