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

#include "num/gpuops.h"
#include "num/filter.h"
#include "num/flpmath.h"
#include "num/multind.h"
#include "num/multiplace.h"

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


struct meco_s {

	nlop_data_t super;

	int N;
	long model;

	const long* y_dims;
	const long* x_dims;
	const long* der_dims;
	const long* map_dims;
	const long* TE_dims;

	const long* y_strs;
	const long* x_strs;
	const long* der_strs;
	const long* map_strs;
	const long* TE_strs;

	// Parameter maps
	complex float* der_x;
	struct multiplace_array_s* TE;
	struct multiplace_array_s* cshift;

	const struct linop_s* linop_fB0;
};

DEF_TYPEID(meco_s);


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

	case MECO_WF:		return 0;
	case MECO_WFR2S:	return MD_BIT(2);
	case MECO_WF2R2S:	return MD_BIT(1) | MD_BIT(3);
	case MECO_R2S:		return MD_BIT(1);
	case MECO_PHASEDIFF:	return 0;
	default:
		assert(0);
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


const struct linop_s* meco_get_fB0_trafo(struct nlop_s* op)
{
	const nlop_data_t* _data = nlop_get_data(op);
	struct meco_s* data = CAST_DOWN(meco_s, _data);
	return data->linop_fB0;
}

// ************************************************************* //
//  Model: (W + F cshift) .* exp(i 2\pi fB0 TE)
// ************************************************************* //
static void meco_fun_wf(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct meco_s* data = CAST_DOWN(meco_s, _data);

	if (NULL == data->der_x)
		data->der_x = md_alloc_sameplace(data->N, data->der_dims, CFL_SIZE, dst);

	long x_pos[data->N];

	for (int i = 0; i < data->N; i++)
		x_pos[i] = 0;

	enum { PIND_W = 0, PIND_F = 1, PIND_FB0 = 2 };

	complex float* tmp_exp = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);
	complex float* tmp_eco = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);

	// =============================== //
	//  forward operator
	// =============================== //

	// F
	x_pos[COEFF_DIM] = PIND_F;

	complex float* F = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, F, data->x_dims, src, CFL_SIZE);

	// dst = F .* cshift
	md_zmul2(data->N, data->y_dims, data->y_strs, dst, data->map_strs, F, data->TE_strs, multiplace_read(data->cshift, dst));


	// W
	x_pos[COEFF_DIM] = PIND_W;

	complex float* W = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, W, data->x_dims, src, CFL_SIZE);

	// dst = W + F .* cshift
	md_zadd2(data->N, data->y_dims, data->y_strs, dst, data->y_strs, dst, data->map_strs, W);


	// fB0
	x_pos[COEFF_DIM] = PIND_FB0;

	complex float* fB0 = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, fB0, data->x_dims, src, CFL_SIZE);

	linop_forward_unchecked(data->linop_fB0, fB0, fB0);

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_exp, data->map_strs, fB0, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->y_dims, tmp_exp, tmp_exp, 2.i * M_PI);

	// tmp_exp = exp(1i*2*pi * fB0 .* TE)
	md_zexp(data->N, data->y_dims, tmp_exp, tmp_exp);

	// dst = dst .* tmp_exp
	md_zmul(data->N, data->y_dims, dst, dst, tmp_exp);


	// =============================== //
	//  partial derivative operator
	// =============================== //
	// der_W
	x_pos[COEFF_DIM] = PIND_W;
	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_exp, CFL_SIZE);

	// der_F
	x_pos[COEFF_DIM] = PIND_F;

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, tmp_exp, data->TE_strs, multiplace_read(data->cshift, dst));

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);

	// der_fB0
	x_pos[COEFF_DIM] = PIND_FB0;

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, dst, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->y_dims, tmp_eco, tmp_eco, 2.i * M_PI);

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);

	md_free(tmp_exp);
	md_free(tmp_eco);
	md_free(W);
	md_free(F);
	md_free(fB0);
}


// ************************************************************* //
//  Model: (W + F cshift) .* exp(- R2s TE) .* exp(i 2\pi fB0 TE)
// ************************************************************* //
static void meco_fun_wfr2s(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct meco_s* data = CAST_DOWN(meco_s, _data);

	if (NULL == data->der_x)
		data->der_x = md_alloc_sameplace(data->N, data->der_dims, CFL_SIZE, dst);

	long x_pos[data->N];

	for (int i = 0; i < data->N; i++)
		x_pos[i] = 0;


	enum { PIND_W = 0, PIND_F = 1, PIND_R2S = 2, PIND_FB0 = 3 };

	complex float* tmp_exp = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);
	complex float* tmp_eco = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);


	// =============================== //
	//  forward operator
	// =============================== //

	// F
	x_pos[COEFF_DIM] = PIND_F;

	complex float* F = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, F, data->x_dims, src, CFL_SIZE);

	// dst = F .* cshift
	md_zmul2(data->N, data->y_dims, data->y_strs, dst, data->map_strs, F, data->TE_strs, multiplace_read(data->cshift, dst));


	// W
	x_pos[COEFF_DIM] = PIND_W;

	complex float* W = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, W, data->x_dims, src, CFL_SIZE);

	// dst = W + F .* cshift
	md_zadd2(data->N, data->y_dims, data->y_strs, dst, data->y_strs, dst, data->map_strs, W);


	// R2s and fB0
	x_pos[COEFF_DIM] = PIND_R2S;

	complex float* R2s = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, R2s, data->x_dims, src, CFL_SIZE);

	md_zsmul(data->N, data->map_dims, R2s, R2s, -1.);


	x_pos[COEFF_DIM] = PIND_FB0;

	complex float* fB0 = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, fB0, data->x_dims, src, CFL_SIZE);

	linop_forward_unchecked(data->linop_fB0, fB0, fB0);

	md_zaxpy2(data->N, data->map_dims, data->map_strs, R2s, 2.i * M_PI, data->map_strs, fB0);
	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_exp, data->map_strs, R2s, data->TE_strs, multiplace_read(data->TE, dst));

	// tmp_exp = exp(z TE)
	md_zexp(data->N, data->y_dims, tmp_exp, tmp_exp);

	// dst = dst .* tmp_exp
	md_zmul(data->N, data->y_dims, dst, dst, tmp_exp);


	// =============================== //
	//  partial derivative operator
	// =============================== //
	// der_W
	x_pos[COEFF_DIM] = PIND_W;

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_exp, CFL_SIZE);

	// der_F
	x_pos[COEFF_DIM] = PIND_F;

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, tmp_exp, data->TE_strs, multiplace_read(data->cshift, dst));

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);

	// der_R2s
	x_pos[COEFF_DIM] = PIND_R2S;

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, dst, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->y_dims, tmp_eco, tmp_eco, -1.);

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);

	// der_fB0
	x_pos[COEFF_DIM] = PIND_FB0;

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, dst, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->y_dims, tmp_eco, tmp_eco, 2.i * M_PI);

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);

	md_free(tmp_exp);
	md_free(tmp_eco);
	md_free(W);
	md_free(F);
	md_free(R2s);
	md_free(fB0);
}


// ************************************************************* //
//  Model: (W exp(- R2s_W TE) + F cshift exp(- R2s_F TE)) .* exp(i 2\pi fB0 TE)
// ************************************************************* //
static void meco_fun_wf2r2s(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct meco_s* data = CAST_DOWN(meco_s, _data);

	if (NULL == data->der_x)
		data->der_x = md_alloc_sameplace(data->N, data->der_dims, CFL_SIZE, dst);

	long x_pos[data->N];

	for (int i = 0; i < data->N; i++)
		x_pos[i] = 0;


	enum { PIND_W = 0, PIND_R2SW = 1, PIND_F = 2, PIND_R2SF = 3, PIND_FB0 = 4 };

	complex float* tmp_exp_R2sW = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);
	complex float* tmp_exp_R2sF = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);
	complex float* tmp_exp_fB0  = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);
	complex float* tmp_eco      = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);

	// =============================== //
	//  forward operator
	// =============================== //

	// W
	x_pos[COEFF_DIM] = PIND_W;

	complex float* W = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, W, data->x_dims, src, CFL_SIZE);


	// R2sW
	x_pos[COEFF_DIM] = PIND_R2SW;

	complex float* R2sW = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, R2sW, data->x_dims, src, CFL_SIZE);

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_exp_R2sW, data->map_strs, R2sW, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->y_dims, tmp_exp_R2sW, tmp_exp_R2sW, -1.);


	// F
	x_pos[COEFF_DIM] = PIND_F;

	complex float* F = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, F, data->x_dims, src, CFL_SIZE);

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->map_strs, F, data->TE_strs, multiplace_read(data->cshift, dst));


	// R2sF
	x_pos[COEFF_DIM] = PIND_R2SF;

	complex float* R2sF = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
	md_copy_block(data->N, x_pos, data->map_dims, R2sF, data->x_dims, src, CFL_SIZE);

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_exp_R2sF, data->map_strs, R2sF, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->y_dims, tmp_exp_R2sF, tmp_exp_R2sF, -1.);


	// fB0
	x_pos[COEFF_DIM] = PIND_FB0;

	complex float* fB0 = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
	md_copy_block(data->N, x_pos, data->map_dims, fB0, data->x_dims, src, CFL_SIZE);

	linop_forward_unchecked(data->linop_fB0, fB0, fB0);

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_exp_fB0, data->map_strs, fB0, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->y_dims, tmp_exp_fB0, tmp_exp_fB0, 2.i * M_PI);

	// tmp_exp_R2sW = exp(- R2sW TE)
	md_zexp(data->N, data->y_dims, tmp_exp_R2sW, tmp_exp_R2sW);

	// tmp_exp_R2sF = exp(- R2sF TE)
	md_zexp(data->N, data->y_dims, tmp_exp_R2sF, tmp_exp_R2sF);

	// tmp_exp_fB0 = exp(i 2\pi fB0 TE)
	md_zexp(data->N, data->y_dims, tmp_exp_fB0, tmp_exp_fB0);

	// tmp_eco = W exp(- R2s_W TE) + F cshift exp(- R2s_F TE)
	md_zmul(data->N, data->y_dims, tmp_eco, tmp_eco, tmp_exp_R2sF);
	md_zfmac2(data->N, data->y_dims, data->y_strs, tmp_eco, data->map_strs, W, data->y_strs, tmp_exp_R2sW);

	// dst = tmp_eco .* tmp_exp_fB0
	md_zmul(data->N, data->y_dims, dst, tmp_eco, tmp_exp_fB0);


	// =============================== //
	//  partial derivative operator
	// =============================== //
	// der_W
	x_pos[COEFF_DIM] = PIND_W;
	md_zmul(data->N, data->y_dims, tmp_eco, tmp_exp_fB0, tmp_exp_R2sW);

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);

	// der_R2sW
	x_pos[COEFF_DIM] = PIND_R2SW;
	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->map_strs, W, data->y_strs, tmp_eco);

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, tmp_eco, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->y_dims, tmp_eco, tmp_eco, -1.);

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);

	// der_F
	x_pos[COEFF_DIM] = PIND_F;
	md_zmul(data->N, data->y_dims, tmp_eco, tmp_exp_fB0, tmp_exp_R2sF);

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, tmp_eco, data->TE_strs, multiplace_read(data->cshift, dst));

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);

	// der_R2sF
	x_pos[COEFF_DIM] = PIND_R2SF;
	md_zmul(data->N, data->y_dims, tmp_eco, tmp_exp_fB0, tmp_exp_R2sF);
	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->map_strs, F, data->y_strs, tmp_eco);


	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, tmp_eco, data->TE_strs, multiplace_read(data->cshift, dst));
	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, tmp_eco, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->y_dims, tmp_eco, tmp_eco, -1.);

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);

	// der_fB0
	x_pos[COEFF_DIM] = PIND_FB0;

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, dst, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->y_dims, tmp_eco, tmp_eco, 2.i * M_PI);

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);

	md_free(tmp_exp_fB0);
	md_free(tmp_exp_R2sW);
	md_free(tmp_exp_R2sF);
	md_free(tmp_eco);
	md_free(W);
	md_free(R2sW);
	md_free(F);
	md_free(R2sF);
	md_free(fB0);
}


// ************************************************************* //
//  Model: rho .* exp(- R2s TE) .* exp(i 2\pi fB0 TE)
// ************************************************************* //
static void meco_fun_r2s(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct meco_s* data = CAST_DOWN(meco_s, _data);

	if (NULL == data->der_x)
		data->der_x = md_alloc_sameplace(data->N, data->der_dims, CFL_SIZE, dst);

	long x_pos[data->N];

	for (int i = 0; i < data->N; i++)
		x_pos[i] = 0;


	enum { PIND_RHO = 0, PIND_R2S = 1, PIND_FB0 = 2 };

	complex float* tmp_exp = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);
	complex float* tmp_eco = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);

	// =============================== //
	//  forward operator
	// =============================== //

	// R2s and fB0
	x_pos[COEFF_DIM] = PIND_R2S;

	complex float* R2s = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, R2s, data->x_dims, src, CFL_SIZE);

	md_zsmul(data->N, data->map_dims, R2s, R2s, -1.);


	x_pos[COEFF_DIM] = PIND_FB0;

	complex float* fB0 = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, fB0, data->x_dims, src, CFL_SIZE);

	linop_forward_unchecked(data->linop_fB0, fB0, fB0);

	md_zaxpy2(data->N, data->map_dims, data->map_strs, R2s, 2.i * M_PI, data->map_strs, fB0);

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_exp, data->map_strs, R2s, data->TE_strs, multiplace_read(data->TE, dst));

	// tmp_exp = exp(z TE)
	md_zexp(data->N, data->y_dims, tmp_exp, tmp_exp);


	// rho
	x_pos[COEFF_DIM] = PIND_RHO;

	complex float* rho = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, rho, data->x_dims, src, CFL_SIZE);

	// dst = tmp_exp .* rho
	md_zmul2(data->N, data->y_dims, data->y_strs, dst, data->y_strs, tmp_exp, data->map_strs, rho);


	// =============================== //
	//  partial derivative operator
	// =============================== //
	// der_rho
	x_pos[COEFF_DIM] = PIND_RHO;

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_exp, CFL_SIZE);


	// der_R2s
	x_pos[COEFF_DIM] = PIND_R2S;

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, dst, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->y_dims, tmp_eco, tmp_eco, -1.);

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);


	// der_fB0
	x_pos[COEFF_DIM] = PIND_FB0;

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, dst, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->y_dims, tmp_eco, tmp_eco, 2.i * M_PI);

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);

	md_free(tmp_exp);
	md_free(tmp_eco);
	md_free(rho);
	md_free(R2s);
	md_free(fB0);
}


// ************************************************************* //
//  Model: rho .* exp(i 2\pi fB0 TE)
// ************************************************************* //
static void meco_fun_phasediff(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct meco_s* data = CAST_DOWN(meco_s, _data);

	if (NULL == data->der_x)
		data->der_x = md_alloc_sameplace(data->N, data->der_dims, CFL_SIZE, dst);

	long x_pos[data->N];

	for (int i = 0; i < data->N; i++)
		x_pos[i] = 0;


	enum { PIND_RHO = 0, PIND_FB0 = 1 };

	complex float* tmp_exp = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);
	complex float* tmp_eco = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);

	// =============================== //
	//  forward operator
	// =============================== //

	// R2s and fB0
	x_pos[COEFF_DIM] = PIND_FB0;

	complex float* fB0 = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, fB0, data->x_dims, src, CFL_SIZE);

	linop_forward_unchecked(data->linop_fB0, fB0, fB0);

	md_zaxpy2(data->N, data->map_dims, data->map_strs, fB0, 2.i * M_PI, data->map_strs, fB0);
	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_exp, data->map_strs, fB0, data->TE_strs, multiplace_read(data->TE, dst));


	// tmp_exp = exp(z TE)
	md_zexp(data->N, data->y_dims, tmp_exp, tmp_exp);


	// rho
	x_pos[COEFF_DIM] = PIND_RHO;

	complex float* rho = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, rho, data->x_dims, src, CFL_SIZE);

	// dst = tmp_exp .* rho
	md_zmul2(data->N, data->y_dims, data->y_strs, dst, data->y_strs, tmp_exp, data->map_strs, rho);


	// =============================== //
	//  partial derivative operator
	// =============================== //
	// der_rho
	x_pos[COEFF_DIM] = PIND_RHO;

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_exp, CFL_SIZE);

	// der_fB0
	x_pos[COEFF_DIM] = PIND_FB0;

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, dst, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->y_dims, tmp_eco, tmp_eco, 2.i * M_PI);

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);

	md_free(tmp_exp);
	md_free(tmp_eco);
	md_free(rho);
	md_free(fB0);
}


static void meco_der(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	struct meco_s* data = CAST_DOWN(meco_s, _data);

	long x_pos[data->N];

	for (int i = 0; i < data->N; i++)
		x_pos[i] = 0;


	complex float* tmp_map = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
	complex float* tmp_exp = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);

	md_clear(data->N, data->y_dims, dst, CFL_SIZE);

	for (long pind = 0; pind < data->x_dims[COEFF_DIM]; pind++) {

		x_pos[COEFF_DIM] = pind;

		md_copy_block(data->N, x_pos, data->map_dims, tmp_map, data->x_dims, src, CFL_SIZE);
		md_copy_block(data->N, x_pos, data->y_dims, tmp_exp, data->der_dims, data->der_x, CFL_SIZE);

		if (pind == data->x_dims[COEFF_DIM] - 1)
			linop_forward_unchecked(data->linop_fB0, tmp_map, tmp_map);

		md_zfmac2(data->N, data->y_dims, data->y_strs, dst, data->map_strs, tmp_map, data->y_strs, tmp_exp);
	}

	md_free(tmp_map);
	md_free(tmp_exp);
}

static void meco_adj(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	struct meco_s* data = CAST_DOWN(meco_s, _data);

	long x_pos[data->N];

	for (int i = 0; i < data->N; i++)
		x_pos[i] = 0;


	complex float* tmp_map = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
	complex float* tmp_exp = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);

	md_clear(data->N, data->x_dims, dst, CFL_SIZE);

	for (long pind = 0; pind < data->x_dims[COEFF_DIM]; pind++) {

		x_pos[COEFF_DIM] = pind;

		md_copy_block(data->N, x_pos, data->map_dims, tmp_map, data->x_dims, dst, CFL_SIZE);
		md_copy_block(data->N, x_pos, data->y_dims, tmp_exp, data->der_dims, data->der_x, CFL_SIZE);

		md_zfmacc2(data->N, data->y_dims, data->map_strs, tmp_map, data->y_strs, src, data->y_strs, tmp_exp);

		md_copy_block(data->N, x_pos, data->x_dims, dst, data->map_dims, tmp_map, CFL_SIZE);
	}


	// real constraint
	unsigned long R2S_flag = get_R2S_flag(data->model);
	unsigned long fB0_flag = get_fB0_flag(data->model);

	for (long pind = 0; pind < data->x_dims[COEFF_DIM]; pind++) {

		if (   MD_IS_SET(R2S_flag, pind)
		    || MD_IS_SET(fB0_flag, pind)) {

			x_pos[COEFF_DIM] = pind;

			md_copy_block(data->N, x_pos, data->map_dims, tmp_map, data->x_dims, dst, CFL_SIZE);
#if 1
			md_zreal(data->N, data->map_dims, tmp_map, tmp_map);
#endif
			if (MD_IS_SET(fB0_flag, pind))
				linop_adjoint_unchecked(data->linop_fB0, tmp_map, tmp_map);

			md_copy_block(data->N, x_pos, data->x_dims, dst, data->map_dims, tmp_map, CFL_SIZE);
		}
	}

	md_free(tmp_map);
	md_free(tmp_exp);
}

static void meco_del(const nlop_data_t* _data)
{
	struct meco_s* data = CAST_DOWN(meco_s, _data);

	multiplace_free(data->TE);
	multiplace_free(data->cshift);

	md_free(data->der_x);

	xfree(data->y_dims);
	xfree(data->x_dims);
	xfree(data->der_dims);
	xfree(data->map_dims);
	xfree(data->TE_dims);

	xfree(data->y_strs);
	xfree(data->x_strs);
	xfree(data->der_strs);
	xfree(data->map_strs);
	xfree(data->TE_strs);

	linop_free(data->linop_fB0);

	xfree(data);
}


struct nlop_s* nlop_meco_create(const int N, const long y_dims[N], const long x_dims[N], const complex float* TE, enum meco_model sel_model, enum fat_spec fat_spec, const float* scale_fB0)
{
	PTR_ALLOC(struct meco_s, data);
	SET_TYPEID(meco_s, data);


	PTR_ALLOC(long[N], nydims);
	md_copy_dims(N, *nydims, y_dims);
	data->y_dims = *PTR_PASS(nydims);

	assert(x_dims[COEFF_DIM] == get_num_of_coeff(sel_model));
	data->model = sel_model;

	PTR_ALLOC(long[N], nxdims);
	md_copy_dims(N, *nxdims, x_dims);
	data->x_dims = *PTR_PASS(nxdims);

	PTR_ALLOC(long[N], nderdims);
	md_merge_dims(N, *nderdims, y_dims, x_dims);
	data->der_dims = *PTR_PASS(nderdims);

	long map_dims[N];
	md_select_dims(N, ~COEFF_FLAG, map_dims, x_dims);
	PTR_ALLOC(long[N], n1dims);
	md_copy_dims(N, *n1dims, map_dims);
	data->map_dims = *PTR_PASS(n1dims);

	long TE_dims[N];
	md_select_dims(N, TE_FLAG, TE_dims, y_dims);
	PTR_ALLOC(long[N], ntedims);
	md_copy_dims(N, *ntedims, TE_dims);
	data->TE_dims = *PTR_PASS(ntedims);

	long scaling_dims[N];
	md_select_dims(N, COEFF_FLAG, scaling_dims, x_dims);


	PTR_ALLOC(long[N], nystr);
	md_calc_strides(N, *nystr, y_dims, CFL_SIZE);
	data->y_strs = *PTR_PASS(nystr);

	PTR_ALLOC(long[N], nxstr);
	md_calc_strides(N, *nxstr, x_dims, CFL_SIZE);
	data->x_strs = *PTR_PASS(nxstr);

	PTR_ALLOC(long[N], nderstr);
	md_calc_strides(N, *nderstr, data->der_dims, CFL_SIZE);
	data->der_strs = *PTR_PASS(nderstr);

	PTR_ALLOC(long[N], n1str);
	md_calc_strides(N, *n1str, map_dims, CFL_SIZE);
	data->map_strs = *PTR_PASS(n1str);

	PTR_ALLOC(long[N], ntestr);
	md_calc_strides(N, *ntestr, TE_dims, CFL_SIZE);
	data->TE_strs = *PTR_PASS(ntestr);

	data->N = N;
	data->der_x = NULL;

	// echo times
	data->TE = multiplace_move(N, TE_dims, CFL_SIZE, TE);


	// calculate cshift
	complex float* cshift = md_alloc(N, TE_dims, CFL_SIZE);

	calc_fat_modu(N, TE_dims, cshift, TE, fat_spec);

	data->cshift = multiplace_move_F(N, TE_dims, CFL_SIZE, cshift);

	if (0. == scale_fB0[0]) {

		debug_printf(DP_DEBUG2, " identity weight on fB0\n");

		data->linop_fB0 = linop_identity_create(N, data->map_dims);

	} else {

		debug_printf(DP_DEBUG2, " sobolev weight on fB0\n");

		data->linop_fB0 = linop_noir_weights_create(N, map_dims, map_dims, map_dims, FFT_FLAGS, 1., scale_fB0[0], scale_fB0[1], 1);
	}

	nlop_fun_t meco_funs[] = {

		[MECO_WF] = meco_fun_wf,
		[MECO_WFR2S] = meco_fun_wfr2s,
		[MECO_WF2R2S] = meco_fun_wf2r2s,
		[MECO_R2S] = meco_fun_r2s,
		[MECO_PHASEDIFF] = meco_fun_phasediff,
	};

	return nlop_create(N, y_dims, N, x_dims, CAST_UP(PTR_PASS(data)), meco_funs[sel_model], meco_der, meco_adj, NULL, NULL, meco_del);
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

	arg_t tmp[5] = { NULL };

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




