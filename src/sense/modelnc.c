/* Copyright 2013-2019. The Regents of the University of California.
 * Copyright 2015-2023. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2021-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Pruessmann KP, Weiger M, Boernert P, Boesiger P. Advances in sensitivity
 * encoding with arbitrary k-space trajectories.
 * Magn Reson Med 2001; 46:638-651.
 */

#include "misc/types.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "linops/linop.h"

#include "noncart/nufft.h"

#include "sense/model.h"

#include "modelnc.h"


const struct linop_s* sense_nc_init(const bart_dim_t max_dims[DIMS], const bart_dim_t map_dims[DIMS], const complex float* maps,
		const bart_dim_t ksp_dims[DIMS],
		const bart_dim_t traj_dims[DIMS], const complex float* traj, const struct nufft_conf_s *_conf,
		const bart_dim_t wgs_dims[DIMS], const complex float* weights,
		const bart_dim_t basis_dims[DIMS], const complex float* basis,
		const struct linop_s** fft_opp, bart_flags_t shared_img_dims)
{
	auto conf = *_conf;
	bart_dim_t coilim_dims[DIMS];
	bart_dim_t img_dims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, coilim_dims, max_dims);
	md_select_dims(DIMS, ~COIL_FLAG & ~shared_img_dims, img_dims, max_dims);

	bart_dim_t ksp_dims2[DIMS];
	md_copy_dims(DIMS, ksp_dims2, ksp_dims);
	ksp_dims2[COEFF_DIM] = max_dims[COEFF_DIM];

	debug_print_dims(DP_INFO, DIMS, ksp_dims2);
	debug_print_dims(DP_INFO, DIMS, coilim_dims);

	bart_stride_t map_strs[DIMS];
	md_calc_strides(DIMS, map_strs, map_dims, CFL_SIZE);


	const struct linop_s* nufft_op = nufft_create2(DIMS, ksp_dims2, coilim_dims,
						traj_dims, traj,
						(weights ? wgs_dims : NULL), weights,
						(basis ? basis_dims : NULL), basis, conf);

	const struct linop_s* maps_op = maps2_create(coilim_dims, map_dims, img_dims, maps);
	const struct linop_s* lop = linop_chain(maps_op, nufft_op);
	linop_free(maps_op);

	if (NULL != fft_opp)
		*fft_opp = linop_clone(nufft_op);

	linop_free(nufft_op);

	return lop;
}


