/* Copyright 2022. Institute of Medical Engineering. Graz University of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <stdbool.h>
#include <assert.h>
#include <math.h>

#include "misc/debug.h"
#include "misc/misc.h"
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


static void post_process(enum mdb_t mode, const struct linop_s* op, struct moba_conf_s* data, const long dims[DIMS], complex float* img)
{
	long imgs_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|MAPS_FLAG|CSHIFT_FLAG|COEFF_FLAG|TIME_FLAG|TIME2_FLAG, imgs_dims, dims);

	long pos[DIMS] = { 0L };

	// Project B1 map back into image space

	long map_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|TIME_FLAG|TIME2_FLAG, map_dims, dims);

	complex float* tmp = md_alloc_sameplace(DIMS, map_dims, CFL_SIZE, img);

	switch (mode) {

	case MDB_BLOCH:

		assert(NULL != data);

		pos[COEFF_DIM] = 3;

		md_copy_block(DIMS, pos, map_dims, tmp, imgs_dims, img, CFL_SIZE);
		linop_forward_unchecked(op, tmp, tmp);
		md_copy_block(DIMS, pos, imgs_dims, img, map_dims, tmp, CFL_SIZE);

		break;

	// Reparameterized Look-Locker Model
	// Estimate effective flip angle from R1'
	// FIXME: Move to separate function which can be tested with a unit test

	case MDB_T1_PHY:

		float r1p_nom = read_relax(data->sim.seq.tr, DEG2RAD(CAST_UP(&data->sim.pulse.sinc)->flipangle));

		md_set_dims(DIMS, pos, 0);

		pos[COEFF_DIM] = 2;

		long map_size = md_calc_size(DIMS, map_dims);

		md_copy_block(DIMS, pos, map_dims, tmp, imgs_dims, img, CFL_SIZE);

		linop_forward_unchecked(op, tmp, tmp);

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

	// IR multi-echo gradient echo model

	case MDB_IR_MGRE:

		// Rescale R2* from ms to s
		if (3 != imgs_dims[COEFF_DIM]) {

			md_set_dims(DIMS, pos, 0);
			pos[COEFF_DIM] = imgs_dims[COEFF_DIM] - 2;

			md_copy_block(DIMS, pos, map_dims, tmp, imgs_dims, img, CFL_SIZE);

			// TE is provided in ms, therefore R2s*1000 transforms: [1/ms] -> [1/s]
			if (use_compat_to_version("v1.0.00"))
				md_zsmul(DIMS, map_dims, tmp, tmp, 1000.);

			md_copy_block(DIMS, pos, imgs_dims, img, map_dims, tmp, CFL_SIZE);
		}

		// Transform and rescale B0 to SI units in image space
		md_set_dims(DIMS, pos, 0);
		pos[COEFF_DIM] = imgs_dims[COEFF_DIM] - 1;

		md_copy_block(DIMS, pos, map_dims, tmp, imgs_dims, img, CFL_SIZE);

		linop_forward_unchecked(op, tmp, tmp);

		// TE is provided in ms, therefore B0*1000 transforms: [1/ms] -> [1/s]
		if (use_compat_to_version("v1.0.00"))
			md_zsmul(DIMS, map_dims, tmp, tmp, 1000.);

		md_copy_block(DIMS, pos, imgs_dims, img, map_dims, tmp, CFL_SIZE);
		break;

	default:
	}

	md_free(tmp);
}


static void set_bloch_conf(enum mdb_t mode, struct mdb_irgnm_l1_conf* conf2, const struct moba_conf* conf, struct moba_conf_s* data, const long img_dims[DIMS])
{
	// T2 estimation turned off for IR FLASH Simulation

	switch (mode) {

	case MDB_BLOCH:

		assert(NULL != data);

		switch (data->sim.seq.seq_type) {

		case SEQ_IRFLASH:

			conf2->l2flags = (0 != data->other.scale[3]) ? ((0 == conf->l2para) ? 8 : conf->l2para) : 0;
			conf2->constrained_maps = (-1 == conf->constrained_maps) ? 1 : conf->constrained_maps;	// only R1 map: bitmask (1 0 0 0) = 1
			conf2->not_wav_maps = (0 == conf->not_wav_maps) ? 2 : conf->not_wav_maps; // no wavelet for T2 and B1 map
			break;

		case SEQ_IRBSSFP:

			conf2->l2flags = (0 == conf->l2para) ? 0 : conf->l2para;
			conf2->constrained_maps = (-1 == conf->constrained_maps) ? 5 : conf->constrained_maps;	// only T1 and T2: bitmask(1 0 1 0) = 5
			conf2->not_wav_maps = (0 == conf->not_wav_maps) ? 1 : conf->not_wav_maps; // no wavelet for B1 map
			break;

		default:
	}

		break;

	// No Wavelet penalty on flip angle map

	case MDB_T1_PHY:

		conf2->l2flags = (0 == conf->l2para) ? 4 : conf->l2para;
		conf2->constrained_maps = (-1 == conf->constrained_maps) ? 2 : conf->constrained_maps;    // only R1 map: bitmask (0 1 0) = 2
		conf2->not_wav_maps = (0 == conf->not_wav_maps) ? 1 : conf->not_wav_maps;	// no wavelet for R1' map

		break;

	// No Wavelet penalty on B0 map

	case MDB_IR_MGRE:

		switch (img_dims[COEFF_DIM]) {

		case 3:
			conf2->constrained_maps = (-1 == conf->constrained_maps) ? 0 : conf->constrained_maps;     // (W, F, B0): bitmask(0 0 0) = 0
			conf2->not_wav_maps = (0 == conf->not_wav_maps) ? 1 : conf->not_wav_maps;
			conf2->l2flags = (0 == conf->l2para) ? 4 : conf->l2para;	// (W, F, B0): bitmask(0 0 1) = 4
			break;
		case 4:
			conf2->constrained_maps = (-1 == conf->constrained_maps) ? 4 : conf->constrained_maps;     // (W, F, R2s, B0): bitmask(0 0 1 0) = 4
			conf2->not_wav_maps = (0 == conf->not_wav_maps) ? 1 : conf->not_wav_maps;
			conf2->l2flags = (0 == conf->l2para) ? 8 : conf->l2para;	// (W, F, R2s, B0): bitmask(0 0 0 1) = 8
			break;
		case 5:
			conf2->constrained_maps = (-1 == conf->constrained_maps) ? 12 : conf->constrained_maps;     // (Ms_w, M0_w, R1s_w, R2s, B0): bitmask(0 0 1 1 0) = 12
			conf2->not_wav_maps = (0 == conf->not_wav_maps) ? 1 : conf->not_wav_maps;
			conf2->l2flags = (0 == conf->l2para) ? 16 : conf->l2para;	// (Ms_w, M0_w, R1s_w, R2s, B0): bitmask(0 0 0 0 1) = 16
			break;
		default:
			conf2->constrained_maps = (-1 == conf->constrained_maps) ? 100 : conf->constrained_maps;     // (Ms_w, M0_w, R1s_w, Ms_f, M0_f, R1s_f, R2s, B0): bitmask(0 0 1 0 0 1 1 0) = 100
			conf2->not_wav_maps = (0 == conf->not_wav_maps) ? 1 : conf->not_wav_maps;
			conf2->l2flags = (0 == conf->l2para) ? 128 : conf->l2para;	// (Ms_w, M0_w, R1s_w, Ms_f, M0_f, R1s_f, R2s, B0): bitmask(0 0 0 0 0 0 0 1) = 128
			break;
		}

		break;

	default:
	}

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
	unsigned long fft_flags = FFT_FLAGS;

	if (conf->sms)
		fft_flags |= SLICE_FLAG;

	long img1_dims[DIMS];
	md_select_dims(DIMS, fft_flags, img1_dims, dims);


	struct noir_model_conf_s mconf = noir_model_conf_defaults;
	mconf.rvc = false;
	mconf.noncart = conf->noncartesian;
	mconf.fft_flags = fft_flags;
	mconf.a = conf->sobolev_a;
	mconf.b = conf->sobolev_b;
	mconf.cnstcoil_flags = TE_FLAG | CSHIFT_FLAG;

	struct mobamod nl = { };

	switch (conf->mode) {

	case MDB_MGRE:

		assert(0); // done in meco_recon (recon_meco.c)

	case MDB_T1:
	case MDB_T2:
	case MDB_T1_PHY:
	case MDB_BLOCH:
	case MDB_IR_MGRE:

		nl = moba_create(dims, TI, TE_IR_MGRE, b1, b0, conf->scale_fB0, conf->mgre_model, conf->fat_spec, pat_dims, pattern, &mconf, data, conf->scaling_M0);
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

		linop_adjoint_unchecked(nl.linop_alpha, tmp, tmp);

		md_copy_block(DIMS, pos, imgs_dims, img, map_dims, tmp, CFL_SIZE);
	}

	long skip = md_calc_size(DIMS, imgs_dims);
	long size = skip + md_calc_size(DIMS, coil_dims);
	long data_size = md_calc_size(DIMS, data_dims);

	long d1[1] = { size };
	// variable which is optimized by the IRGNM
	complex float* x = md_alloc_sameplace(1, d1, CFL_SIZE, kspace_data);
	complex float* x_ref = md_alloc_sameplace(1, d1, CFL_SIZE, kspace_data);

	md_copy(DIMS, imgs_dims, x, img, CFL_SIZE);
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
		.opt_reg = conf->opt_reg,
		.step = conf->step,
		.lower_bound = conf->lower_bound,
		.l2flags = (0 == conf->l2para) ? ((1 == conf->opt_reg) ? (0UL) : ~(0UL)) : conf->l2para,
		.constrained_maps = conf->constrained_maps,
		.auto_norm = conf->auto_norm,
		.no_sens_l2 = data->other.no_sens_l2,
		.not_wav_maps = (0 == conf->not_wav_maps) ? 0 : conf->not_wav_maps,
		.algo = conf->algo,
		.rho = conf->rho,
		.ropts = conf->ropts,
		.l1val = conf->l1val,
		.pusteps = conf->pusteps,
		.ratio = conf->ratio,
	};

	set_bloch_conf(conf->mode, &conf2, conf, data, imgs_dims);

	// Always constrain last parameter map as default
	if (-1 == conf2.constrained_maps)
		conf2.constrained_maps = (1UL << (dims[COEFF_DIM] - 1));

	assert(0 <= conf2.constrained_maps);

	long irgnm_conf_dims[DIMS];
	md_select_dims(DIMS, fft_flags|MAPS_FLAG|COEFF_FLAG|TIME_FLAG|TIME2_FLAG, irgnm_conf_dims, imgs_dims);

	irgnm_conf_dims[COIL_DIM] = coil_dims[COIL_DIM];

	debug_printf(DP_INFO, "imgs_dims:\n\t");
	debug_print_dims(DP_INFO, DIMS, irgnm_conf_dims);


	mdb_irgnm_l1(&conf2,
			irgnm_conf_dims,
			nl.nlop,
			size * 2, (float*)x, (float*)x_ref,
			data_size * 2, (const float*)kspace_data);

	md_copy(DIMS, imgs_dims, img, x, CFL_SIZE);

	if (NULL != sens) {

		if (data->other.export_ksp_coils) {

			md_copy(DIMS, coil_dims, sens, x + skip, CFL_SIZE);

		} else {

			noir_forw_coils(nl.linop, x + skip, x + skip);
			md_copy(DIMS, coil_dims, sens, x + skip, CFL_SIZE);
		}
	}

	if (!conf->out_origin_maps)
		post_process(conf->mode, nl.linop_alpha, data, dims, img);

	// Clean up


	if ((MDB_T1_PHY == conf->mode) || (MDB_BLOCH == conf->mode) || (MDB_IR_MGRE == conf->mode))
		linop_free(nl.linop_alpha);

	nlop_free(nl.nlop);

	md_free(x);
	md_free(x_ref);
}


void moba_recon(const struct moba_conf* conf, struct moba_conf_s* data, const long dims[DIMS], complex float* img, complex float* sens, const long pat_dims[DIMS], const complex float* pattern, const complex float* TI, const complex float* TE, const complex float* b1, const complex float* b0, const complex float* kspace_data, const complex float* init)
{
	long imgs_dims[DIMS];
	long coil_dims[DIMS];
	long data_dims[DIMS];

	unsigned long fft_flags = FFT_FLAGS;

	if (conf->sms)
		fft_flags |= SLICE_FLAG;

	md_select_dims(DIMS, fft_flags|MAPS_FLAG|COEFF_FLAG|TIME_FLAG|TIME2_FLAG, imgs_dims, dims);
	md_select_dims(DIMS, fft_flags|COIL_FLAG|MAPS_FLAG|TIME_FLAG|TIME2_FLAG, coil_dims, dims);
	md_select_dims(DIMS, fft_flags|COIL_FLAG|TE_FLAG|MAPS_FLAG|CSHIFT_FLAG|TIME_FLAG|TIME2_FLAG, data_dims, dims);

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

