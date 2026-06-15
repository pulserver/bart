/* Copyright 2013. The Regents of the University of California.
 * Copyright 2017-2022. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2022-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 *
 * Uecker M, Hohage T, Block KT, Frahm J. Image reconstruction by regularized nonlinear
 * inversion – Joint estimation of coil sensitivities and image content.
 * Magn Reson Med 2008; 60:674-682.
 */

#include <complex.h>
#include <math.h>
#include <assert.h>

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/fmac.h"

#include "nlops/nlop.h"
#include "nlops/tenmul.h"
#include "nlops/chain.h"
#include "nlops/cast.h"

#include "num/fft.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/filter.h"

#include "noir/utils.h"
#include "num/ops.h"

#include "model.h"



struct noir_model_conf_s noir_model_conf_defaults = {

	.sobolev_os = 1.f,
	.fft_flags = FFT_FLAGS,
	.cnstcoil_flags = 0u,
	.rvc = false,
	.noncart = false,
	.a = 220.,
	.b = 32.,
};


static void noir_linop_del(const void* _data)
{
	linop_free(_data);
}

struct noir_s noir_create(const long dims[DIMS], const long pat_dims[DIMS], const complex float* psf, const struct noir_model_conf_s* conf)
{

	long data_dims[DIMS];
	long data_red_dims[DIMS];
	long coil_dims[DIMS];
	long imgs_dims[DIMS];

	md_select_dims(DIMS, ~conf->cnstcoil_flags, coil_dims, dims);
	md_select_dims(DIMS, ~COIL_FLAG, imgs_dims, dims);
	md_select_dims(DIMS, ~MAPS_FLAG, data_red_dims, dims);

	md_copy_dims(DIMS, data_dims, data_red_dims);
	md_copy_dims(3, data_dims, pat_dims);
	assert(md_check_compat(DIMS, md_nontriv_dims(DIMS, data_dims), data_dims, pat_dims));

	const struct linop_s* lop_fft = linop_fft_create(DIMS, data_dims, conf->fft_flags);

	long fft_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, fft_dims, data_dims);

	complex float* fft_mod = md_alloc(DIMS, fft_dims, CFL_SIZE);
	md_zfill(DIMS, fft_dims, fft_mod, 1.);
	fftscale(DIMS, fft_dims, FFT_FLAGS, fft_mod, fft_mod);
	fftmod(DIMS, fft_dims, FFT_FLAGS, fft_mod, fft_mod);

	lop_fft = linop_chain_FF(linop_cdiag_create(DIMS, data_dims, FFT_FLAGS, fft_mod), lop_fft);
	md_free(fft_mod);

	if (!md_check_equal_dims(DIMS, data_red_dims, data_dims, ~0UL))
		lop_fft = linop_chain_FF(linop_resize_center_create(DIMS, data_dims, data_red_dims), lop_fft);

	const struct linop_s* lop_pattern = linop_cdiag_create(DIMS, data_dims, md_nontriv_dims(DIMS, pat_dims), psf);

	const struct operator_s* ops[3] = {

		lop_fft->forward,
		conf->noncart ? lop_pattern->forward : lop_pattern->normal,
		lop_fft->adjoint
	};

	const struct operator_s* op_frw = operator_chainN(3, ops);
	const struct operator_s* op_adj = operator_identity_create(DIMS, data_red_dims);

	// This is an asymmetric operator,
	// forward maps coil images to gridded coil images
	// adjoint is just identity
	const struct linop_s* trafo = linop_from_ops(op_frw, op_adj, op_frw, NULL);

	linop_free(lop_fft);
	linop_free(lop_pattern);
	operator_free(op_frw);
	operator_free(op_adj);

	const struct nlop_s* nlw1 = nlop_tenmul_create(DIMS, data_red_dims, imgs_dims, coil_dims);

	const struct linop_s* weights = linop_noir_weights_create(DIMS, coil_dims, coil_dims, NULL, FFT_FLAGS, conf->sobolev_os, conf->a, conf->b, 1.);
	const struct nlop_s* nlw2 = nlop_from_linop(weights);
	const struct nlop_s* nl = nlop_chain2_FF(nlw2, 0, nlw1, 1);

	if (conf->rvc) {

		const struct nlop_s* nlop_zreal = nlop_from_linop_F(linop_zreal_create(DIMS, imgs_dims));
		nl = nlop_chain2_swap_FF(nlop_zreal, 0, nl, 0);
	}

	const struct nlop_s* nl2 = nlop_chain2_FF(nl, 0, nlop_from_linop_F(trafo), 0);

	struct nlop_s* nlop = (struct nlop_s*)nlop_attach(nl2, (void*)weights, noir_linop_del);
	nlop_free(nl2);

	return (struct noir_s){ .nlop = nlop, .linop = weights };
}

void noir_forw_coils(const struct linop_s* op, complex float* dst, const complex float* src)
{
	linop_forward_unchecked(op, dst, src);
}

