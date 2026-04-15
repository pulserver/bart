/* Copyright 2022. Institute of Medical Engineering. Graz University of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <assert.h>
#include <math.h>

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/version.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "iter/iter3.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"

#include "noir/model.h"

#include "moba/model_moba.h"
#include "moba/blochfun.h"
#include "moba/T1phyfun.h"
#include "moba/meco.h"
#include "moba/iter_l1.h"
#include "moba/moba.h"
#include "moba/exp.h"
#include "moba/recon_meco.h"

#include "recon.h"


static void post_process(enum mdb_t mode, struct moba_conf_s* data, const long imgs_dims[DIMS], const struct linop_s* op[], complex float* img)
{
	long pos[DIMS] = { 0L };

	// Project B1 map back into image space

	long map_dims[DIMS];
	md_select_dims(DIMS, ~COEFF_FLAG, map_dims, imgs_dims);

	long img_strs[DIMS];
	md_calc_strides(DIMS, img_strs, imgs_dims, CFL_SIZE);

	complex float* tmp = md_alloc_sameplace(DIMS, map_dims, CFL_SIZE, img);

	for (pos[COEFF_DIM] = 0; pos[COEFF_DIM] < imgs_dims[COEFF_DIM]; pos[COEFF_DIM]++) {

		if (NULL == op[pos[COEFF_DIM]])
			continue;

		md_copy_block(DIMS, pos, map_dims, tmp, imgs_dims, img, CFL_SIZE);

		linop_forward_unchecked(op[pos[COEFF_DIM]], tmp, tmp);

		md_copy_block(DIMS, pos, imgs_dims, img, map_dims, tmp, CFL_SIZE);
	}

	switch (mode) {


	// Reparameterized Look-Locker Model
	// Estimate effective flip angle from R1'
	// FIXME: Move to separate function which can be tested with a unit test

	case MDB_T1_PHY:

		float r1p_nom = read_relax(data->sim.seq.tr, DEG2RAD(CAST_UP(&data->sim.pulse.sinc)->flipangle));

		md_set_dims(DIMS, pos, 0);

		pos[COEFF_DIM] = 2;

		long map_size = md_calc_size(DIMS, map_dims);

		md_copy_block(DIMS, pos, map_dims, tmp, imgs_dims, img, CFL_SIZE);

		md_zreal(DIMS, map_dims, tmp, tmp);

		md_zsmul(DIMS, map_dims, tmp, tmp, data->other.scale[2]);

		complex float* offset = md_alloc_sameplace(DIMS, map_dims, CFL_SIZE, img);

		md_zfill(DIMS, map_dims, offset, r1p_nom);
		md_zadd(DIMS, map_dims, tmp, tmp, offset);

		md_zsmul(DIMS, map_dims, tmp, tmp, -data->sim.seq.tr);    // Same scaling set in T1phyfun.c

		md_smin(1, MD_DIMS(2 * map_size), (float*)tmp, (float*)tmp, 0.);

		md_zexp(DIMS, map_dims, tmp, tmp);

		md_zacosr(DIMS, map_dims, tmp, tmp);

		md_zsmul(DIMS, map_dims, tmp, tmp, 180. / M_PI);        // output the effective flip angle map (in degree!)

		md_copy_block(DIMS, pos, imgs_dims, img, map_dims, tmp, CFL_SIZE);

		md_free(offset);

		break;

	case MDB_IR_MGRE:

		if (use_compat_to_version("v1.0.00")) {

			complex float* map_B0 = MD_ACCESS_PTR(DIMS, img_strs, (pos[COEFF_DIM] = imgs_dims[COEFF_DIM] - 1, pos), img);
			md_zsmul2(DIMS, map_dims, img_strs, map_B0, img_strs, map_B0, 1000.);

			if (3 != imgs_dims[COEFF_DIM]) {

				complex float* map_R2s = MD_ACCESS_PTR(DIMS, img_strs, (pos[COEFF_DIM] = imgs_dims[COEFF_DIM] - 2, pos), img);
				md_zsmul2(DIMS, map_dims, img_strs, map_R2s, img_strs, map_R2s, 1000.);
			}
		}

		break;

	case MDB_BLOCH:

		complex float* map_B1 = MD_ACCESS_PTR(DIMS, img_strs, (pos[COEFF_DIM] = 3, pos), img);

		// Rescale due to scaling in moba.c
		md_zsadd2(DIMS, map_dims, img_strs, map_B1, img_strs, map_B1, 1. / (data->other.scale[3] ?: 1.));

		break;

	default:
	}

	md_free(tmp);
}


static unsigned long get_constrained_maps(enum mdb_t mode, enum meco_model mgre_mode)
{
	switch (mode) {

	case MDB_T1: return MD_BIT(2);				// T1 map
	case MDB_T2: return MD_BIT(1);				// T2 map
	case MDB_BLOCH:	return MD_BIT(0) | MD_BIT(2);		// T1 and T2 map
	case MDB_T1_PHY: return MD_BIT(1);			// T1 map
	case MDB_MGRE:
	case MDB_IR_MGRE:
		switch (mgre_mode) {

		case MECO_WF: return 0UL;
		case MECO_WFR2S: return MD_BIT(2);			// R2*
		case MECO_WF2R2S: return MD_BIT(1) | MD_BIT(3);		// R2*W and R2*F
		case MECO_R2S: return MD_BIT(1);			// R2*
		case MECO_PHASEDIFF: return 0UL;
		case MECO_PI: return 0UL;
		case IR_MECO_T1_R2S: return MD_BIT(2) | MD_BIT(3);	// R1* and R2*
		case IR_MECO_W_T1_F_T1_R2S: return MD_BIT(2) | MD_BIT(5) | MD_BIT(6);	// R1*W, R1*F and R2*
		}
	}

	assert(0);
}

static void set_regu_flags(struct mdb_irgnm_l1_conf* conf2, const struct moba_conf* conf, struct moba_conf_s* data, long ncoeffs, const struct linop_s* lop_sobolev[ncoeffs])
{
/*
	Simple rules for model independent default regularization flags:
	1.) Only active maps (with non-zero scale) are regularized
	2.) Maps with a Sobolev operator get L2 regularization
	3.) Remaining active maps get wavelet / L2 depending on conf.opt_reg
*/

	unsigned long act_flags = 0UL;
	unsigned long sob_flags = 0UL;

	for (int i = 0; i < ncoeffs; i++) {

		sob_flags |= (NULL != lop_sobolev[i] && 0. != data->other.scale[i]) ? MD_BIT(i) : 0UL;
		act_flags |= (0. != data->other.scale[i]) ? MD_BIT(i) : 0UL;
	}

	unsigned long l2flags = conf->l2para;
	unsigned long wavflags = ~0UL;

	switch (conf->opt_reg) {

	case 1:
		l2flags = (~0UL == l2flags) ? sob_flags : l2flags;
		wavflags = act_flags & ~sob_flags;
		break;
	case 2:
		l2flags = (~0UL == l2flags) ? act_flags : l2flags;
		wavflags = 0UL;
		break;
	default:
		error("Invalid regularization option!\n");
	}

	wavflags &= MD_BIT(ncoeffs - conf->not_wav_maps) - 1;

	conf2->l2flags = l2flags;
	conf2->wavflags = wavflags;

	conf2->tvscales_N = data->other.tvscales_N;
	conf2->tvscales = data->other.tvscales;
}


static void recon(const struct moba_conf* conf, struct moba_conf_s* data,
                const long dims[DIMS],
		const long imgs_dims[DIMS], complex float* img,
		const long coil_dims[DIMS], complex float* sens,
		const long pat_dims[DIMS], const complex float* pattern,
		const complex float* TI,
		const complex float* TE_IR_MGRE,
		const complex float* b1,
		const complex float* b0,
		const long data_dims[DIMS], const complex float* kspace_data)
{

	struct noir_model_conf_s mconf = noir_model_conf_defaults;
	mconf.rvc = false;
	mconf.noncart = conf->noncartesian;
	mconf.a = conf->sobolev_a;
	mconf.b = conf->sobolev_b;
	mconf.sms = conf->sms;
	mconf.sos = conf->sos;

	struct mobamod nl = { };

	switch (conf->mode) {

	case MDB_MGRE:

		assert(0); // done in meco_recon (recon_meco.c)

	case MDB_T1:
	case MDB_T2:
	case MDB_T1_PHY:
	case MDB_BLOCH:
	case MDB_IR_MGRE:

		nl = moba_create(dims, TI, TE_IR_MGRE, b1, b0, conf->scale_fB0, conf->mgre_model, conf->fat_spec, pat_dims, pattern, coil_dims, (data->other.fixed_coil) ? sens : NULL, &mconf, data, conf->scaling_M0);
		break;
	}

	long map_dims[DIMS];

	md_copy_dims(DIMS, map_dims, imgs_dims);
	map_dims[COEFF_DIM] = 1;
	long pos[DIMS] = { 0L };

	if (MDB_IR_MGRE == conf->mode && use_compat_to_version("v1.0.00")) {

		// Used in fetal paper

		md_set_dims(DIMS, pos, 0);

		pos[COEFF_DIM] = imgs_dims[COEFF_DIM] - 1; // FIXME: fB0 is always in the last

		complex float* tmp = md_alloc(DIMS, map_dims, CFL_SIZE);

		md_copy_block(DIMS, pos, map_dims, tmp, imgs_dims, img, CFL_SIZE);

		linop_adjoint_unchecked(nl.linop_sobolev[pos[COEFF_DIM]], tmp, tmp);

		md_copy_block(DIMS, pos, imgs_dims, img, map_dims, tmp, CFL_SIZE);
	}

	long skip = md_calc_size(DIMS, imgs_dims);
	long size = skip + (!data->other.fixed_coil ? md_calc_size(DIMS, coil_dims) : 0);
	long data_size = md_calc_size(DIMS, data_dims);

	long d1[1] = { size };
	// variable which is optimized by the IRGNM
	complex float* x = md_alloc_sameplace(1, d1, CFL_SIZE, kspace_data);
	complex float* x_ref = md_alloc_sameplace(1, d1, CFL_SIZE, kspace_data);

	md_copy(DIMS, imgs_dims, x, img, CFL_SIZE);

	if (!data->other.fixed_coil)
		md_copy(DIMS, coil_dims, x + skip, sens, CFL_SIZE);

	//reference
	md_zsmul(1, MD_DIMS(size), x_ref, x, conf->damping);

	struct iter3_irgnm_conf irgnm_conf = iter3_irgnm_defaults;

	irgnm_conf.iter = conf->iter;
	irgnm_conf.alpha = conf->alpha;
	irgnm_conf.redu = conf->redu;

	if (conf->alpha_min_exp_decay)
		irgnm_conf.alpha_min = conf->alpha_min;
	else
		irgnm_conf.alpha_min0 = conf->alpha_min;

	irgnm_conf.cgtol = conf->tolerance;

	if (MDB_T1 == conf->mode)
		if ((2 == conf->opt_reg) || (!conf->auto_norm))
			irgnm_conf.cgtol = 1e-3;

	irgnm_conf.cgiter = conf->inner_iter;
	irgnm_conf.nlinv_legacy = true;

	struct mdb_irgnm_l1_conf conf2 = {

		.c2 = &irgnm_conf,
		.step = conf->step,
		.lower_bound = conf->lower_bound,
		.constrained_maps = (~0UL != conf->constrained_maps) ? conf->constrained_maps : get_constrained_maps(conf->mode, conf->mgre_model),
		.auto_norm = conf->auto_norm,
		.no_sens_l2 = data->other.no_sens_l2,
		.wav_trans_flags = FFT_FLAGS | (conf->sos ? SLICE_FLAG : 0),
		.algo = conf->algo,
		.rho = conf->rho,
		.ropts = conf->ropts,
		.l1val = conf->l1val,
		.pusteps = conf->pusteps,
		.ratio = conf->ratio,
	};

	set_regu_flags(&conf2, conf, data, imgs_dims[COEFF_DIM], nl.linop_sobolev);

	long irgnm_conf_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|SLICE_FLAG|MAPS_FLAG|COEFF_FLAG|TIME_FLAG|TIME2_FLAG, irgnm_conf_dims, imgs_dims);

	irgnm_conf_dims[COIL_DIM] = coil_dims[COIL_DIM];

	mdb_irgnm_l1(&conf2,
			irgnm_conf_dims,
			nl.nlop,
			size * 2, (float*)x, (float*)x_ref,
			data_size * 2, (const float*)kspace_data);

	md_copy(DIMS, imgs_dims, img, x, CFL_SIZE);

	if (NULL != sens && !data->other.fixed_coil) {

		if (data->other.export_ksp_coils) {

			md_copy(DIMS, coil_dims, sens, x + skip, CFL_SIZE);

		} else {

			noir_forw_coils(nl.linop, x + skip, x + skip);
			md_copy(DIMS, coil_dims, sens, x + skip, CFL_SIZE);
		}
	}

	if (!conf->out_origin_maps)
		post_process(conf->mode, data, imgs_dims, nl.linop_sobolev, img);

	// Clean up
	for (int i = 0; i < (int)ARRAY_SIZE(nl.linop_sobolev); i++)
		linop_free(nl.linop_sobolev[i]);

	nlop_free(nl.nlop);

	md_free(x);
	md_free(x_ref);
}


void moba_recon(const struct moba_conf* conf, struct moba_conf_s* data, const long dims[DIMS], const long imgs_dims[DIMS], complex float* img, const long coil_dims[DIMS], complex float* sens, const long pat_dims[DIMS], const complex float* pattern, const complex float* TI, const complex float* TE, const complex float* b1, const complex float* b0, const long data_dims[DIMS], const complex float* kspace_data, const complex float* init)
{
	switch (conf->mode) {

	case MDB_T1:
	case MDB_T1_PHY:
	case MDB_T2:
	case MDB_BLOCH:
	case MDB_IR_MGRE:

		recon(conf, data, dims, imgs_dims, img, coil_dims, sens, pat_dims, pattern, TI, TE, b1, b0, data_dims, kspace_data);
		break;

	case MDB_MGRE:

		meco_recon(conf, data, dims, conf->mgre_model, conf->fat_spec, conf->scale_fB0, true, conf->out_origin_maps, imgs_dims, img, coil_dims, sens, imgs_dims, init, TI, pat_dims, pattern, data_dims, kspace_data);
		break;

	default:
		assert(0);
	}
}

