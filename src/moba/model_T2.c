/* Copyright 2018-2020. Uecker Lab, University Medical Center Goettingen.
 * Copyright 2022-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018-2020 Martin Uecker
 * 2018-2020 Xiaoqing Wang
 */

#include <stdlib.h>
#include <assert.h>

#include "misc/misc.h"
#include "misc/mri.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/cast.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "noir/model.h"

#include "moba/T2fun.h"

#include "model_T2.h"




struct mobamod T2_create(const bart_dim_t dims[DIMS], const complex float* mask, const complex float* TI, const complex float* psf, const struct noir_model_conf_s* conf)
{
	bart_dim_t data_dims[DIMS];
	md_select_dims(DIMS, ~COEFF_FLAG, data_dims, dims);

	struct noir_s nlinv = noir_create(data_dims, mask, psf, conf);
	struct mobamod ret;

	bart_dim_t map_dims[DIMS];
	bart_dim_t out_dims[DIMS];
	bart_dim_t in_dims[DIMS];
	bart_dim_t TI_dims[DIMS];

	md_select_dims(DIMS, conf->fft_flags, map_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|TE_FLAG, out_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|COEFF_FLAG, in_dims, dims);
	md_select_dims(DIMS, TE_FLAG, TI_dims, dims);

	// chain T2 model
	struct nlop_s* T2 = nlop_T2_create(DIMS, map_dims, out_dims, in_dims, TI_dims, TI);


	const struct nlop_s* b = nlinv.nlop;
	const struct nlop_s* c = nlop_chain2_FF(T2, 0, b, 0);

	nlinv.nlop = nlop_permute_inputs_F(c, 2, (const int[2]){ 1, 0 });

	ret.nlop = nlop_flatten_F(nlinv.nlop);
	ret.linop = nlinv.linop;

	return ret;
}


