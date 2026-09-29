/* Copyright 2022-2025. Institute of Biomedical Imaging. TU Graz.
 * Copyright 2022. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author:
 *	Nick Scholand, Martin Juschitz
 */

#include <complex.h>
#include <math.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "noir/model.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/filter.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/cast.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "simu/bloch.h"

#include "moba/moba.h"
#include "moba/utils.h"

#include "noir/utils.h"

#include "blochfun.h"

#define round(x)	((int) ((x) + .5))



struct blochfun_s {

	nlop_data_t super;

	int N;

	const bart_dim_t* der_dims;
	const bart_dim_t* map_dims;
	const bart_dim_t* in_dims;
	const bart_dim_t* out_dims;

	const bart_stride_t* der_strs;
	const bart_stride_t* map_strs;
	const bart_stride_t* in_strs;
	const bart_stride_t* out_strs;

	//derivatives
	complex float* derivatives;

	const complex float* b1;
	const complex float* b0;

	const struct moba_conf_s* moba_data;
};

DEF_TYPEID(blochfun_s);


static void bloch_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	double starttime = timestamp();

	debug_printf(DP_DEBUG2, "Started Forward Calculation\n");

	struct blochfun_s* data = CAST_DOWN(blochfun_s, _data);

	// Forward model is always on CPU
	complex float* r1scale = md_alloc(data->N, data->map_dims, CFL_SIZE);
	complex float* r2scale = md_alloc(data->N, data->map_dims, CFL_SIZE);
	complex float* m0scale = md_alloc(data->N, data->map_dims, CFL_SIZE);
	complex float* b1scale = md_alloc(data->N, data->map_dims, CFL_SIZE);

	bart_dim_t pool_dims[DIMS];
	md_copy_dims(DIMS, pool_dims, data->map_dims);
	pool_dims[ITER_DIM] = data->moba_data->sim.voxel.P - 1;

	bart_stride_t pool_strs[DIMS];
	md_calc_strides(data->N, pool_strs, pool_dims, CFL_SIZE);

	bart_dim_t pool_out_dims[DIMS];
	md_copy_dims(DIMS, pool_out_dims, data->out_dims);
	pool_out_dims[ITER_DIM] = data->moba_data->sim.voxel.P - 1;

	bart_stride_t pool_out_strs[DIMS];
	md_calc_strides(data->N, pool_out_strs, pool_out_dims, CFL_SIZE);

	// FIXME: Keep all r1, r2, m0 scales in one variable
	complex float* r1_poolscale = md_alloc(data->N, pool_dims, CFL_SIZE);
	complex float* r2_poolscale = md_alloc(data->N, pool_dims, CFL_SIZE);
	complex float* kscale = md_alloc(data->N, pool_dims, CFL_SIZE);
	complex float* omscale = md_alloc(data->N, pool_dims, CFL_SIZE);
	complex float* m0_poolscale = md_alloc(data->N, pool_dims, CFL_SIZE);

	bart_dim_t pos[data->N];
	md_set_dims(data->N, pos, 0);

	//-------------------------------------------------------------------
	// Copy necessary files from GPU to CPU
	//-------------------------------------------------------------------


	pos[COEFF_DIM] = 0;// R1
	md_copy_block(data->N, pos, data->map_dims, r1scale, data->in_dims, src, CFL_SIZE);

	pos[COEFF_DIM] = 1;// M0
	md_copy_block(data->N, pos, data->map_dims, m0scale, data->in_dims, src, CFL_SIZE);

	pos[COEFF_DIM] = 2;// R2
	md_copy_block(data->N, pos, data->map_dims, r2scale, data->in_dims, src, CFL_SIZE);

	pos[COEFF_DIM] = 3;// B1
	md_copy_block(data->N, pos, data->map_dims, b1scale, data->in_dims, src, CFL_SIZE);

	for (int p = 0; p < data->moba_data->sim.voxel.P - 1; p++) {

		pos[ITER_DIM] = p;
		pos[COEFF_DIM] = 4 + p; // R1
		md_copy_block(data->N, pos, pool_dims, r1_poolscale, data->in_dims, src, CFL_SIZE);

		pos[COEFF_DIM] = 4 + (data->moba_data->sim.voxel.P - 1) + p; // R2
		md_copy_block(data->N, pos, pool_dims, r2_poolscale, data->in_dims, src, CFL_SIZE);

		pos[COEFF_DIM] = 4 + 2 * (data->moba_data->sim.voxel.P - 1) + p; // k
		md_copy_block(data->N, pos, pool_dims, kscale, data->in_dims, src, CFL_SIZE);

		pos[COEFF_DIM] = 4 + 3 * (data->moba_data->sim.voxel.P - 1) + p; // M0
		md_copy_block(data->N, pos, pool_dims, m0_poolscale, data->in_dims, src, CFL_SIZE);

		pos[COEFF_DIM] = 4 + 4 * (data->moba_data->sim.voxel.P - 1) + p; // Om
		md_copy_block(data->N, pos, pool_dims, omscale, data->in_dims, src, CFL_SIZE);
	}



	//Allocate Output CPU memory
	complex float* sig_cpu = md_calloc(data->N, data->out_dims, CFL_SIZE);
	complex float* dr1_cpu = md_calloc(data->N, data->out_dims, CFL_SIZE);
	complex float* dr2_cpu = md_calloc(data->N, data->out_dims, CFL_SIZE);
	complex float* dm0_cpu = md_calloc(data->N, data->out_dims, CFL_SIZE);
	complex float* db1_cpu = md_calloc(data->N, data->out_dims, CFL_SIZE);

	complex float* dr1_pools_cpu = md_calloc(data->N, pool_out_dims, CFL_SIZE);
	complex float* dr2_pools_cpu = md_calloc(data->N, pool_out_dims, CFL_SIZE);
	complex float* dk_cpu = md_calloc(data->N, pool_out_dims, CFL_SIZE);
	complex float* dm0_pools_cpu = md_calloc(data->N, pool_out_dims, CFL_SIZE);
	complex float* dom_cpu = md_calloc(data->N, pool_out_dims, CFL_SIZE);

	// debug_sim(&data->moba_data.sim);

#pragma omp parallel for collapse(3)
	for (int x = 0; x < data->map_dims[0]; x++) {
		for (int y = 0; y < data->map_dims[1]; y++) {
			for (int z = 0; z < data->map_dims[2]; z++) {

				//Calculate correct spatial position
				bart_dim_t spa_pos[DIMS];

				md_copy_dims(DIMS, spa_pos, data->map_dims);

				spa_pos[0] = x;
				spa_pos[1] = y;
				spa_pos[2] = z;

				bart_dim_t spa_ind = md_calc_offset(data->N, data->map_strs, spa_pos) / (bart_stride_t)CFL_SIZE;

				bart_dim_t spa_pos_pools[DIMS];

				md_copy_dims(DIMS, spa_pos_pools, pool_dims);
				spa_pos_pools[0] = x;
				spa_pos_pools[1] = y;
				spa_pos_pools[2] = z;
				bart_dim_t spa_ind_pools;

				//-------------------------------------------------------------------
				// Define simulation parameter
				//-------------------------------------------------------------------

				// Extract external B1 value from input

				float b1s = 1.;

				if (NULL != data->b1) {

					b1s = crealf(data->b1[spa_ind]);

					if (safe_isnanf(b1s))
						b1s = 0.;
				}

				// Copy simulation data

				struct sim_data sim_data = data->moba_data->sim;


				sim_data.seq.rep_num = data->out_dims[TE_DIM];

				// FIXME: Why set scaling? Maybe change to boolean?
				sim_data.voxel.r1[0] = crealf(r1scale[spa_ind]);
				sim_data.voxel.r2[0] = crealf(r2scale[spa_ind]);
				if (sim_data.voxel.P > 1) {

					sim_data.voxel.m0[0] = crealf(m0scale[spa_ind]);
					sim_data.voxel.b1 = crealf(b1scale[spa_ind]);
				} else {

					sim_data.voxel.m0[0] = 1.;
					sim_data.voxel.b1 = b1s * (1. + crealf(b1scale[spa_ind]));
				}

				for (int p = 0; p < sim_data.voxel.P - 1; p++) {

					spa_pos_pools[ITER_DIM] = p;
					spa_ind_pools = md_calc_offset(data->N, pool_strs, spa_pos_pools) / (bart_stride_t)CFL_SIZE;

					//FIXME: Switch to Boolean?
					sim_data.voxel.r1[p + 1] =  crealf(r1_poolscale[spa_ind_pools]);
					sim_data.voxel.r2[p + 1] =  crealf(r2_poolscale[spa_ind_pools]);
					sim_data.voxel.k[p]	 =  crealf(kscale[spa_ind_pools]);
					sim_data.voxel.m0[p + 1] =  crealf(m0_poolscale[spa_ind_pools]);
					sim_data.voxel.Om[p + 1] =  crealf(omscale[spa_ind_pools]);
				}

				// Extract external B0 value from input

				float b0s = 0.;

				if (NULL != data->b0) {

					b0s = crealf(data->b0[spa_ind]);

					if (safe_isnanf(b0s))
						b0s = 0.;
				}

				sim_data.voxel.w = b0s;

				//debug_printf(DP_INFO, "\tR1:%f, R1_2:%f, R1_3:%f,R1_4:%f,R1_5:%f \n", sim_data.voxel.r1[0], sim_data.voxel.r1[1], sim_data.voxel.r1[2], sim_data.voxel.r1[3], sim_data.voxel.r1[4]);
				//debug_printf(DP_INFO, "\tR2:%f, R2_2:%f, R2_3:%f, R2_4:%f, R2_5:%f \n", sim_data.voxel.r2[0], sim_data.voxel.r2[1], sim_data.voxel.r2[2], sim_data.voxel.r2[3], sim_data.voxel.r2[4]);
				//debug_printf(DP_INFO, "\tM0:%f, M0_2:%f, M0_3:%f, M0_4:%f,M0_5:%f\n", sim_data.voxel.m0[0],sim_data.voxel.m0[1],sim_data.voxel.m0[2],sim_data.voxel.m0[3], sim_data.voxel.m0[4]);
				//debug_printf(DP_INFO, "\tB1:%f\n", sim_data.voxel.b1);
				//debug_printf(DP_INFO, "\tk:%f, k2:%f, k3:%f, k4:%f\n", sim_data.voxel.k[0], sim_data.voxel.k[1], sim_data.voxel.k[2],sim_data.voxel.k[3]);
				//debug_printf(DP_INFO, "\tOm:%f, Om2:%f, Om3:%f, Om4:%f\n\n", sim_data.voxel.Om[1], sim_data.voxel.Om[2], sim_data.voxel.Om[3], sim_data.voxel.Om[4]);

				//-------------------------------------------------------------------
				// Run simulation
				//-------------------------------------------------------------------

				float m[sim_data.seq.rep_num][3];
				float sa_r1[sim_data.seq.rep_num][3];
				float sa_r2[sim_data.seq.rep_num][3];
				float sa_m0[sim_data.seq.rep_num][3];
				float sa_b1[sim_data.seq.rep_num][3];

				float m_p[sim_data.seq.rep_num][sim_data.voxel.P][3];
 				float sa_r1_p[sim_data.seq.rep_num][sim_data.voxel.P][3];
				float sa_r2_p[sim_data.seq.rep_num][sim_data.voxel.P][3];
				float sa_m0_p[sim_data.seq.rep_num][sim_data.voxel.P][3];
				float sa_b1_p[sim_data.seq.rep_num][1][3];
				float sa_k_p[sim_data.seq.rep_num][sim_data.voxel.P][3];
				float sa_om_p[sim_data.seq.rep_num][sim_data.voxel.P][3];

				switch (sim_data.seq.model) {

				case MODEL_BLOCH:

					bloch_simulation(&sim_data, sim_data.seq.rep_num, &m, &sa_r1, &sa_r2, &sa_m0, &sa_b1);
					break;

				case MODEL_BMC:

					bloch_simulation2(&sim_data, sim_data.seq.rep_num, sim_data.voxel.P, &m_p, &sa_r1_p, &sa_r2_p, &sa_m0_p, &sa_b1_p, &sa_k_p,  &sa_om_p);
					break;
				}

				//-------------------------------------------------------------------
				// Copy simulation output to storage on CPU
				//-------------------------------------------------------------------

				bart_dim_t curr_pos[DIMS];
				md_copy_dims(DIMS, curr_pos, spa_pos);

				bart_dim_t curr_pos_pools[DIMS];
				md_copy_dims(DIMS, curr_pos_pools, spa_pos_pools);

				bart_dim_t position = 0;

				for (int j = 0; j < sim_data.seq.rep_num; j++) {

					curr_pos[TE_DIM] = j;
					curr_pos_pools[TE_DIM] = j;

					position = md_calc_offset(data->N, data->out_strs, curr_pos) / (bart_stride_t)CFL_SIZE;

					float a = 1.;

					assert(0. != CAST_UP(&sim_data.pulse.sinc)->flipangle);

					// Scaling signal close to 1
					//	-> Comparable to Look-Locker model (difference relaxation factor: expf(-sim_data.voxel.r2 * sim_data.seq.te))
					//	-> divided by nom slice thickness to keep multi spin simulation signal around 1
					// 	-> nom slice thickness [m] * 1000 -> [mm], because relative to default slice thickness of single spin of 0.001 m
					if ((SEQ_FLASH == sim_data.seq.seq_type) || (SEQ_IRFLASH == sim_data.seq.seq_type))
						a = 1. / sinf(CAST_UP(&sim_data.pulse.sinc)->flipangle * M_PI / 180.) / (sim_data.seq.nom_slice_thickness * 1000.);

					else if ((SEQ_BSSFP == sim_data.seq.seq_type) || (SEQ_IRBSSFP == sim_data.seq.seq_type))
						a = 1. / sinf(CAST_UP(&sim_data.pulse.sinc)->flipangle / 2. * M_PI / 180.) / (sim_data.seq.nom_slice_thickness * 1000.);

					// complex m0scale[spa_ind] adds scaling and phase to the signal
					// M = M_x + i M_y	and S = S_x + i S_y
					switch (sim_data.seq.model) {

					case MODEL_BLOCH:

						dr1_cpu[position] = a * m0scale[spa_ind] * (sa_r1[j][0] + sa_r1[j][1] * 1.i);
						dm0_cpu[position] = a * 		   (sa_m0[j][0] + sa_m0[j][1] * 1.i);
						dr2_cpu[position] = a * m0scale[spa_ind] * (sa_r2[j][0] + sa_r2[j][1] * 1.i);
						db1_cpu[position] = a * m0scale[spa_ind] * (sa_b1[j][0] + sa_b1[j][1] * 1.i);
						sig_cpu[position] = a * m0scale[spa_ind] * (    m[j][0] +     m[j][1] * 1.i);
						break;

					case MODEL_BMC:

						if (SEQ_CEST == sim_data.seq.seq_type) {

							dr1_cpu[position] = (sa_r1_p[j][0][2]);
							dm0_cpu[position] = (sa_m0_p[j][0][2]);
							dr2_cpu[position] = (sa_r2_p[j][0][2]);
							db1_cpu[position] = (sa_b1_p[j][0][2]);
							sig_cpu[position] = (    m_p[0][0][2]);
						} else {

							dr1_cpu[position] = (sa_r1_p[j][0][0] + sa_r1_p[j][0][1] * 1.i);
							dm0_cpu[position] = (sa_m0_p[j][0][0] + sa_m0_p[j][0][1] * 1.i);
							dr2_cpu[position] = (sa_r2_p[j][0][0] + sa_r2_p[j][0][1] * 1.i);
							db1_cpu[position] = (sa_b1_p[j][0][0] + sa_b1_p[j][0][1] * 1.i);
							sig_cpu[position] = (    m_p[0][0][0] +     m_p[j][0][1] * 1.i);
						}
						break;
					}

					for (int p = 0; p < sim_data.voxel.P - 1; p++) {

						curr_pos_pools[ITER_DIM] = p;
						position = md_calc_offset(data->N, pool_out_strs, curr_pos_pools) / (bart_stride_t)CFL_SIZE;

						if (SEQ_CEST == sim_data.seq.seq_type) {

							dr1_pools_cpu[position] = sa_r1_p[j][p + 1][2];
							dr2_pools_cpu[position] = sa_r2_p[j][p + 1][2];
							dk_cpu[position]	=  sa_k_p[j][p + 0][2];
							dm0_pools_cpu[position] = sa_m0_p[j][p + 1][2];
							dom_cpu[position]	= sa_om_p[j][p + 0][2];
						} else {

							dr1_pools_cpu[position] = (sa_r1_p[j][p + 1][0] + sa_r1_p[j][p + 1][1] * 1.i);
							dr2_pools_cpu[position] = (sa_r2_p[j][p + 1][0] + sa_r2_p[j][p + 1][1] * 1.i);
							dk_cpu[position] 	= ( sa_k_p[j][p + 0][0] +  sa_k_p[j][p + 0][1] * 1.i);
							dm0_pools_cpu[position] = (sa_m0_p[j][p + 1][0] + sa_m0_p[j][p + 1][1] * 1.i);
							dom_cpu[position] 	= (sa_om_p[j][p + 0][0] + sa_om_p[j][p + 0][1] * 1.i);
						}
					}
				}
			}
		}
	}

	md_free(r1scale);
	md_free(r2scale);
	md_free(m0scale);
	md_free(b1scale);
	md_free(r1_poolscale);
	md_free(r2_poolscale);
	md_free(kscale);
	md_free(omscale);
	md_free(m0_poolscale);

	debug_printf(DP_DEBUG3, "Copy data\n");

	//-------------------------------------------------------------------
	// Collect data of signal (potentially on GPU)
	//-------------------------------------------------------------------

	md_copy(data->N, data->out_dims, dst, sig_cpu, CFL_SIZE);

	md_free(sig_cpu);

	//-------------------------------------------------------------------
	// Collect data of derivatives in single array
	//-------------------------------------------------------------------

	if (NULL == data->derivatives)
		data->derivatives = md_alloc_sameplace(data->N, data->der_dims, CFL_SIZE, dst);

	md_clear(data->N, data->der_dims, data->derivatives, CFL_SIZE);

	md_set_dims(data->N, pos, 0);

	pos[COEFF_DIM] = 0; // R1
	md_copy_block(data->N, pos, data->der_dims, data->derivatives, data->out_dims, dr1_cpu, CFL_SIZE);

	pos[COEFF_DIM] = 1; // M0
	md_copy_block(data->N, pos, data->der_dims, data->derivatives, data->out_dims, dm0_cpu, CFL_SIZE);

	pos[COEFF_DIM] = 2; // R2
	md_copy_block(data->N, pos, data->der_dims, data->derivatives, data->out_dims, dr2_cpu, CFL_SIZE);

	pos[COEFF_DIM] = 3; // B1
	md_copy_block(data->N, pos, data->der_dims, data->derivatives, data->out_dims, db1_cpu, CFL_SIZE);

	for (int p = 0; p < data->moba_data->sim.voxel.P - 1; p++) {

		pos[ITER_DIM] = p;
		pos[COEFF_DIM] = 4 + 0 * (data->moba_data->sim.voxel.P - 1) + p;// R1
		md_copy_block(data->N, pos, data->der_dims, data->derivatives, pool_out_dims, dr1_pools_cpu, CFL_SIZE);

		pos[COEFF_DIM] = 4 + 1 * (data->moba_data->sim.voxel.P - 1) + p;// R2
		md_copy_block(data->N, pos, data->der_dims, data->derivatives, pool_out_dims, dr2_pools_cpu, CFL_SIZE);

		pos[COEFF_DIM] = 4 + 2 * (data->moba_data->sim.voxel.P - 1) + p;// k
		md_copy_block(data->N, pos, data->der_dims, data->derivatives, pool_out_dims, dk_cpu, CFL_SIZE);

		pos[COEFF_DIM] = 4 + 3 * (data->moba_data->sim.voxel.P - 1) + p;// M0
		md_copy_block(data->N, pos, data->der_dims, data->derivatives, pool_out_dims, dm0_pools_cpu, CFL_SIZE);

		pos[COEFF_DIM] = 4 + 4 * (data->moba_data->sim.voxel.P - 1) + p;// Om
		md_copy_block(data->N, pos, data->der_dims, data->derivatives, pool_out_dims, dom_cpu, CFL_SIZE);
	}

	md_free(dr1_cpu);
	md_free(dr2_cpu);
	md_free(dm0_cpu);
	md_free(db1_cpu);
	md_free(dr1_pools_cpu);
	md_free(dr2_pools_cpu);
	md_free(dm0_pools_cpu);
	md_free(dk_cpu);
	md_free(dom_cpu);

	double totaltime = timestamp() - starttime;

	debug_printf(DP_DEBUG2, "Time = %.2f s\n", totaltime);
}


static void bloch_der(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	struct blochfun_s* data = CAST_DOWN(blochfun_s, _data);


	md_ztenmul(data->N, data->out_dims, dst, data->in_dims, src, data->der_dims, data->derivatives);
}

static void bloch_adj(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	struct blochfun_s* data = CAST_DOWN(blochfun_s, _data);


	md_ztenmulc(data->N, data->in_dims, dst, data->out_dims, src, data->der_dims, data->derivatives);
}


static void bloch_del(const nlop_data_t* _data)
{
	struct blochfun_s* data = CAST_DOWN(blochfun_s, _data);

	md_free(data->derivatives);

	xfree(data->der_dims);
	xfree(data->map_dims);
	xfree(data->in_dims);
	xfree(data->out_dims);

	xfree(data->der_strs);
	xfree(data->map_strs);
	xfree(data->in_strs);
	xfree(data->out_strs);

	md_free(data->b1);
	md_free(data->b0);

	xfree(data);
}


struct nlop_s* nlop_bloch_create(int N, const bart_dim_t out_dims[N], const bart_dim_t in_dims[N],
			const complex float* b1, const complex float* b0, const struct moba_conf_s* config)
{
	PTR_ALLOC(struct blochfun_s, data);
	SET_TYPEID(blochfun_s, data);

	bart_dim_t der_dims[N];
	md_max_dims(N, ~UINT64_C(0), der_dims, out_dims, in_dims);

	bart_dim_t map_dims[N];
	md_select_dims(N, ~COEFF_FLAG, map_dims, in_dims);

	PTR_ALLOC(bart_dim_t[N], derdims);
	md_copy_dims(N, *derdims, der_dims);
	data->der_dims = *PTR_PASS(derdims);

	PTR_ALLOC(bart_dim_t[N], allstr);
	md_calc_strides(N, *allstr, der_dims, CFL_SIZE);
	data->der_strs = *PTR_PASS(allstr);

	PTR_ALLOC(bart_dim_t[N], ndims);
	md_copy_dims(N, *ndims, map_dims);
	data->map_dims = *PTR_PASS(ndims);

	PTR_ALLOC(bart_dim_t[N], nmstr);
	md_calc_strides(N, *nmstr, map_dims, CFL_SIZE);
	data->map_strs = *PTR_PASS(nmstr);

	PTR_ALLOC(bart_dim_t[N], nodims);
	md_copy_dims(N, *nodims, out_dims);
	data->out_dims = *PTR_PASS(nodims);

	PTR_ALLOC(bart_dim_t[N], nostr);
	md_calc_strides(N, *nostr, out_dims, CFL_SIZE);
	data->out_strs = *PTR_PASS(nostr);

	PTR_ALLOC(bart_dim_t[N], nidims);
	md_copy_dims(N, *nidims, in_dims);
	data->in_dims = *PTR_PASS(nidims);

	PTR_ALLOC(bart_dim_t[N], nistr);
	md_calc_strides(N, *nistr, in_dims, CFL_SIZE);
	data->in_strs = *PTR_PASS(nistr);

	data->N = N;

	data->derivatives = NULL;


	data->moba_data = config;

	data->b1 = NULL;
	data->b0 = NULL;

	if (NULL != b1) {

		data->b1 = md_alloc(N, map_dims, CFL_SIZE);
		md_copy(N, map_dims, (void*)data->b1, b1, CFL_SIZE);
	}

	if (NULL != b0) {

		data->b0 = md_alloc(N, map_dims, CFL_SIZE);
		md_copy(N, map_dims, (void*)data->b0, b0, CFL_SIZE);
	}

	const struct nlop_s* ret = nlop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), bloch_fun, bloch_der, bloch_adj, NULL, NULL, bloch_del);

	bart_flags_t rvc = MD_BIT(0) | MD_BIT(2) | MD_BIT(3); // R1, R2, B1
	for (int p = 0; p < config->sim.voxel.P - 1; p++) {

		rvc |= MD_BIT(4 + 0 * (config->sim.voxel.P - 1) + p); // R1_2
		rvc |= MD_BIT(4 + 1 * (config->sim.voxel.P - 1) + p); // R2_2
		rvc |= MD_BIT(4 + 2 * (config->sim.voxel.P - 1) + p); // k
		rvc |= MD_BIT(4 + 4 * (config->sim.voxel.P - 1) + p); // Om
	}

	return nlop_chain_FF(nlop_from_linop_F(moba_rvc_create(N, in_dims, rvc)), ret);
}

