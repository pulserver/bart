/* Copyright 2013. The Regents of the University of California.
 * Copyright 2017-2022. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2022-2025. Institute of Biomedical Imaging. TU Graz.
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
#include <stdbool.h>
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

#include "model.h"



struct noir_model_conf_s noir_model_conf_defaults = {

	.fft_flags = FFT_FLAGS,
	.cnstcoil_flags = 0u,
	.ptrn_flags = ~(COIL_FLAG|MAPS_FLAG),
	.rvc = false,
	.noncart = false,
	.a = 220.,
	.b = 32.,
};


static void noir_linop_del(const void* _data)
{
	linop_free(_data);
}

struct noir_s noir_create(const long dims[DIMS], const complex float* psf, const struct noir_model_conf_s* conf)
{

	long data_dims[DIMS];
	long coil_dims[DIMS];
	long imgs_dims[DIMS];


	md_select_dims(DIMS, ~conf->cnstcoil_flags, coil_dims, dims);
	md_select_dims(DIMS, ~COIL_FLAG, imgs_dims, dims);
	md_select_dims(DIMS, ~MAPS_FLAG, data_dims, dims);

	long wght_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, wght_dims, dims);

	long ptrn_dims[DIMS];
	md_select_dims(DIMS, conf->ptrn_flags, ptrn_dims, dims);


	complex float* wghts = md_alloc(DIMS, wght_dims, CFL_SIZE);

	noir_calc_weights(conf->a, conf->b, dims, wghts);
	fftmod(DIMS, wght_dims, FFT_FLAGS, wghts, wghts);
	fftscale(DIMS, wght_dims, FFT_FLAGS, wghts, wghts);

	const struct linop_s* lop_wghts = linop_cdiag_create(DIMS, coil_dims, FFT_FLAGS, wghts);
	const struct linop_s* lop_wghts_ifft = linop_ifft_create(DIMS, coil_dims, FFT_FLAGS);

	md_free(wghts);

	const struct linop_s* weights = linop_chain_FF(lop_wghts, lop_wghts_ifft);



	const struct linop_s* lop_fft = linop_fft_create(DIMS, data_dims, conf->fft_flags);


	complex float* ptr = md_alloc_sameplace(DIMS, ptrn_dims, CFL_SIZE, psf);

	md_copy(DIMS, ptrn_dims, ptr, psf, CFL_SIZE);
	fftmod(DIMS, ptrn_dims, conf->fft_flags, ptr, ptr);

	const struct linop_s* lop_pattern = linop_fmac_create(DIMS, data_dims, 0, 0, ~conf->ptrn_flags, ptr);
	md_free(ptr);

	if (conf->noncart) {

		complex float* adj_ptr = md_alloc(DIMS, ptrn_dims, CFL_SIZE);

		md_zfill(DIMS, ptrn_dims, adj_ptr, 1.);

		fftmod(DIMS, ptrn_dims, conf->fft_flags, adj_ptr, adj_ptr);

		const struct linop_s* lop_adj_pattern = linop_fmac_create(DIMS, data_dims, 0, 0, ~conf->ptrn_flags, adj_ptr);

		md_free(adj_ptr);

		const struct linop_s* lop_tmp = linop_from_ops(lop_pattern->forward, lop_adj_pattern->adjoint, NULL, NULL);

		linop_free(lop_adj_pattern);
		linop_free(lop_pattern);

		lop_pattern = lop_tmp;
	}

	long fft_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, fft_dims, dims);

	complex float* fft_mod = md_alloc(DIMS, fft_dims, CFL_SIZE);
	md_zfill(DIMS, fft_dims, fft_mod, 1.);
	fftscale(DIMS, fft_dims, FFT_FLAGS, fft_mod, fft_mod);

	const struct linop_s* lop_fftmod = linop_cdiag_create(DIMS, data_dims, FFT_FLAGS, fft_mod);
	md_free(fft_mod);

	const struct linop_s* lop_fft2 = linop_chain(lop_fftmod, lop_fft);
	linop_free(lop_fftmod);
	linop_free(lop_fft);

	const struct linop_s* frw = linop_chain_FF(lop_fft2, lop_pattern);


	const struct nlop_s* nlw1 = nlop_tenmul_create(DIMS, data_dims, imgs_dims, coil_dims);
	const struct nlop_s* nlw2 = nlop_from_linop(weights);
	const struct nlop_s* nl = nlop_chain2_FF(nlw2, 0, nlw1, 1);

	if (conf->rvc) {

		const struct nlop_s* nlop_zreal = nlop_from_linop_F(linop_zreal_create(DIMS, imgs_dims));
		nl = nlop_chain2_swap_FF(nlop_zreal, 0, nl, 0);
	}

	const struct nlop_s* nl2 = nlop_chain2_FF(nl, 0, nlop_from_linop_F(frw), 0);

	struct nlop_s* nlop = (struct nlop_s*)nlop_attach(nl2, (void*)weights, noir_linop_del);
	nlop_free(nl2);

	return (struct noir_s){ .nlop = nlop, .linop = weights };
}

void noir_forw_coils(const struct linop_s* op, complex float* dst, const complex float* src)
{
	linop_forward_unchecked(op, dst, src);
}

