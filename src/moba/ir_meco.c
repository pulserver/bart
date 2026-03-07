/* Copyright 2024-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Xiaoqing Wang, Nick Scholand, Martin Uecker, Moritz Blumenthal
 */

#include <complex.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "misc/mri.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/multiplace.h"
#include "num/iovec.h"

#include "simu/signals.h"

#include "noir/utils.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/cast.h"
#include "nlops/snlop.h"
#include "nlops/smath.h"
#include "nlops/nlop_jacobian.h"

#include "moba/utils.h"

#include "ir_meco.h"

int ir_meco_get_num_of_coeff(enum meco_model sel_model)
{
	int ncoeff = 0;

	switch (sel_model) {

	case IR_MECO_WF_fB0:		ncoeff = 3; break; // meco, water, fat, fB0
	case IR_MECO_WF_R2S:		ncoeff = 4; break; // meco, water, fat, R2*, fB0
	case IR_MECO_T1_R2S:		ncoeff = 5; break; // ir + meco, water T1, R2*, fB0
	case IR_MECO_W_T1_F_T1_R2S:	ncoeff = 8; break; // ir + meco, water T1, fat T1, R2*, fB0
	default: error("invalid model");
	}

	return ncoeff;
}

// Calculate Model:
// Water = (Ms_w - (Ms_w + M0_w) * exp(-TI_k.*R1s_w))  // only for IR
// Fat = (Ms_f - (Ms_f + M0_f) * exp(-TI_k.*R1s_f)) // only for meco
// Water-fat model: (Water + Fat * z_m) * exp(i * 2pi * f_B0 * TE_m) // for IR meco

// Water-fat, R2* model: (Water + Fat * z_m) * exp(i * 2pi * f_B0 * TE_m) * exp(i * 2pi * f_B0 * TE_m)// for IR meco

static void calc_fat_modu(int N, const long dims[N], complex float* dst, const complex float* TE, enum fat_spec fat_spec)
{
	assert(1 == bitcount(md_nontriv_dims(N, dims)));
	md_clear(N, dims, dst, CFL_SIZE);

	for (int i = 0; i < md_calc_size(N, dims); i++)
		dst[i] = calc_fat_modulation(3.0, crealf(TE[i]) * 1.E-3, fat_spec); // FIXME: TE in SI units instead ms
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
	assert(ir_meco_get_num_of_coeff(meco_model) == in_dims[COEFF_DIM]);

	arg_t args [in_dims[COEFF_DIM]];
	arg_t out = NULL;

	arg_t tmp[5] = { NULL };

	switch (meco_model) {

	case IR_MECO_WF_fB0:

		debug_printf(DP_DEBUG1, "MODEL: W, F, fB0\n");
		args[0] = snlop_input(N, map_dims, "W");
		args[1] = snlop_input(N, map_dims, "F");
		args[2] = snlop_input(N, map_dims, "fB0");

		tmp[0] = fat_spectrum(args[1], N, TE_dims, TE, fat_spec);
		tmp[1] = snlop_add(tmp[0], args[0]);

		out = B0_modulation(tmp[1], args[2], N, TE_dims, TE);
		break;

	case IR_MECO_WF_R2S:

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

	ret = nlop_zprecomp_jacobian_F(ret);

	return ret;
}


struct nlop_s* nlop_ir_meco_create(int N, const long map_dims[N], const long /*out_dims*/[N], const long in_dims[N], const long TI_dims[N],
				const complex float* TI, const long TE_dims[N], const complex float* TE, const float* scale_fB0, enum meco_model meco_model, enum fat_spec fat_spec, const float* scale)
{
	const struct nlop_s* model = nlop_ir_meco_model_create(N, map_dims, in_dims, TI_dims, TI, TE_dims, TE, meco_model, fat_spec);

	const struct linop_s* prec[in_dims[COEFF_DIM]];

	for (int i = 0; i < in_dims[COEFF_DIM]; i++)
		prec[i] = NULL;


	// weight on alpha
	long w_dims[N];
	md_select_dims(N, FFT_FLAGS, w_dims, map_dims);

	complex float* weights = md_alloc(N, w_dims, CFL_SIZE);
	noir_calc_weights(scale_fB0[0], scale_fB0[1], w_dims, weights);

	const struct linop_s* linop_fB0 = linop_cdiag_create(N, map_dims, FFT_FLAGS, weights);
	linop_fB0 = linop_chain_FF(linop_fB0, linop_ifftc_create(N, map_dims, FFT_FLAGS)); // IFFT(W.* \hat{x_{k}})
	linop_fB0 = linop_chain_FF(linop_fB0, linop_zreal_create(N, map_dims)); // IFFT(W.* \hat{x_{k}})

	md_free(weights);

	if (3 == in_dims[COEFF_DIM]) { // W, F, fB0

		prec[2] = linop_fB0;

	} else if (4 == in_dims[COEFF_DIM]) { // W, F, R2*, fB0

		prec[2] = linop_zreal_create(N, map_dims);
		prec[3] = linop_fB0;

	} else if (5 == in_dims[COEFF_DIM]) { // Ms, M0, R1*, R2*, fB0

		prec[2] = linop_zreal_create(N, map_dims);
		prec[3] = linop_zreal_create(N, map_dims);
		prec[4] = linop_fB0;

	} else if (8 == in_dims[COEFF_DIM]) { // Ms_w, M0_w, R1*_w, Ms_f, M0_f, R1*_f, R2*, fB0

		prec[2] = linop_zreal_create(N, map_dims);
		prec[5] = linop_zreal_create(N, map_dims);
		prec[6] = linop_zreal_create(N, map_dims);
		prec[7] = linop_fB0;

	} else {

		assert(0);
	}

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

