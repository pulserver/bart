/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <assert.h>
#include <complex.h>
#include <stdio.h>
#include <math.h>

#include "misc/mri.h"
#include "misc/types.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/shrdptr.h"
#include "misc/list.h"

#include "num/iovec.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/multiplace.h"
#include "num/ops.h"
#include "num/ode.h"
#include "num/loop.h"

#include "seq/pulse.h"
#include "seq/event.h"

#include "simu/bloch.h"
#include "simu/simulation.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/sum.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/const.h"
#include "nlops/zexp.h"
#include "nlops/ztrigon.h"
#include "nlops/someops.h"
#include "nlops/tenmul.h"
#include "nlops/stack.h"
#include "nlops/nlop_jacobian.h"


#ifdef USE_CUDA
#include "num/gpuops.h"
#include "simu/gpu_bloch.h"
#endif

#include "nlop_seq.h"

struct sim_config_s sim_config_default_cpu = {

	.N = DIMS,
	.mdims = { [0 ... DIMS - 1] = 1 },
	.pdims = { [0 ... DIMS - 1] = 1 },

	.MO_DIM = 2,
	.MI_DIM = 3,
	.PI_DIM = 3,
	.spatial_flags = MD_BIT(11) | MD_BIT(12) | MD_BIT(13),
	.voxel_size = { 0., 0., 0. },

	.tol = 1.e-5,
};

struct sim_config_s sim_config_default_gpu = {

	.N = DIMS,
	.mdims = { [0 ... DIMS - 1] = 1 },
	.pdims = { [0 ... DIMS - 1] = 1 },

	.MO_DIM = 14,
	.MI_DIM = 15,
	.PI_DIM = 15,
	.spatial_flags = MD_BIT(11) | MD_BIT(12) | MD_BIT(13),
	.voxel_size = { 0., 0., 0. },

	.tol = 1.e-5,
};

void sim_config_set_dims(struct sim_config_s* sim, int N, const long dims[N], int Nspins)
{
	assert(DIMS >= N);
	sim->N = N;

	assert(1 == dims[sim->MI_DIM]);
	assert(1 == dims[sim->MO_DIM]);
	assert(1 == dims[sim->PI_DIM]);
	assert(N > sim->PI_DIM);
	assert(N > sim->MI_DIM);
	assert(N > sim->MO_DIM);

	md_copy_dims(N, sim->mdims, dims);
	sim->mdims[sim->MO_DIM] = 3; // Mx, My, Mz

	md_copy_dims(N, sim->pdims, dims);
	sim->pdims[sim->PI_DIM] = 4; // R1, R2, B1, B0

	md_select_dims(N, ~sim->spatial_flags, sim->pdims, sim->pdims);

	if (1 < Nspins) {

		sim->mdims[md_max_idx(sim->spatial_flags)] = Nspins;
	}
}

extern void sim_config_debug(int dl, struct sim_config_s* sim)
{
	debug_printf(dl, "mdims: ");
	debug_print_dims(dl, sim->N, sim->mdims);
	debug_printf(dl, "pdims: ");
	debug_print_dims(dl, sim->N, sim->pdims);
}


struct nlop_seq_data_s;


typedef void (*nlop_seq_fun_t)(const struct nlop_seq_data_s* d, int N,
	const long modims[N], complex float* omag,
	const long midims[N], const complex float* imag,
	const long pdims[N], const complex float* pars,
	const long dmmdims[N], complex float* dmm,
	const long dmpdims[N], complex float* dmp
);

typedef void (*nlop_seq_sig_fun_t)(const struct nlop_seq_data_s* d, int N,
	const long modims[N], complex float* omag,
	int S, const long sdims[S][N], complex float* signal[S],
	const long midims[N], const complex float* imag,
	const long pdims[N], const complex float* pars,
	const long dmmdims[N], complex float* dmm,
	const long dmpdims[N], complex float* dmp,
	const long dsmdims[S][N], complex float* dsm[S],
	const long dspdims[S][N], complex float* dsp[S]
);

typedef void (*nlop_seq_free_t)(const struct nlop_seq_data_s* d);

typedef struct nlop_seq_data_s {

	INTERFACE(nlop_data_t);

	TYPEID* TYPEID;

	struct sim_config_s sim;

	nlop_seq_fun_t fun;
	nlop_seq_sig_fun_t fun_sig;
	nlop_seq_free_t free;

} nlop_seq_data_t;

DEF_TYPEID(nlop_seq_data_s);

static void nlop_seq_fun(const nlop_data_t* data, int N, int OO, const long odims[OO][N], complex float* arr_dst[OO], int II, const long idims[II][N], const complex float* arr_src[II], const long ddims[OO][II][N], complex float* jac[OO][II])
{
	auto d = CAST_DOWN(nlop_seq_data_s, data);

	assert(2 == II);

	assert(N == d->sim.N);
	assert(md_check_equal_dims(N, d->sim.mdims, odims[0], ~0UL));
	assert(md_check_equal_dims(N, d->sim.pdims, idims[1], ~0UL));

	if (NULL != d->fun) {

		assert(0 == d->fun_sig);

		d->fun(d, N,
			odims[0], arr_dst[0],
			idims[0], arr_src[0], idims[1], arr_src[1],
			ddims[0][0], jac[0][0], ddims[0][1], jac[0][1]);

		return;
	}

	if (NULL != d->fun_sig) {

		assert(0 == d->fun);

		int S = OO - 1;

		complex float** signal = arr_dst + 1;

		complex float* dsm[S ?: 1];
		complex float* dsp[S ?: 1];

		long dsmdims[S ?: 1][N];
		long dspdims[S ?: 1][N];

		for (int i = 0; i < S; i++) {

			dsm[i] = jac[i + 1][0];
			dsp[i] = jac[i + 1][1];

			md_copy_dims(N, dsmdims[i], ddims[i + 1][0]);
			md_copy_dims(N, dspdims[i], ddims[i + 1][1]);
		}

		d->fun_sig(d, N, odims[0], arr_dst[0], S, odims + 1, signal, idims[0], arr_src[0], idims[1], arr_src[1],
				ddims[0][0], jac[0][0], ddims[0][1], jac[0][1], dsmdims, dsm, dspdims, dsp);

		return;
	}

	assert(0);
}

static void nlop_seq_free(const nlop_data_t* data)
{
	auto d = CAST_DOWN(nlop_seq_data_s, data);

	d->free(d);
}

static const struct nlop_s* nlop_seq_create(const struct sim_config_s* sim, nlop_seq_data_t* data, nlop_seq_fun_t fun, nlop_seq_free_t free)
{
	SET_TYPEID(nlop_seq_data_s, data);

	data->sim = *sim;
	data->fun = fun;
	data->fun_sig = NULL;
	data->free = free;

	int OO = 1;
	int II = 2;
	int N = sim->N;

	long odims[1][N];
	md_copy_dims(N, odims[0], sim->mdims);

	long idims[II][N];
	md_transpose_dims(N, sim->MI_DIM, sim->MO_DIM, idims[0], sim->mdims);
	md_copy_dims(N, idims[1], sim->pdims);

	unsigned long diag_flags[OO][II];
	for (int i = 0; i < OO; i++)
		for (int j = 0; j < II; j++)
			diag_flags[i][j] = 0UL;

	return nlop_zblock_diag_generic_create(CAST_UP(data), N, OO, odims, II, idims, diag_flags, nlop_seq_fun, nlop_seq_free);
}

static const struct nlop_s* nlop_sig_seq_create(const struct sim_config_s* sim, nlop_seq_data_t* data, nlop_seq_sig_fun_t fun, nlop_seq_free_t free, int S, int N, long sig_dims[S][N])
{
	SET_TYPEID(nlop_seq_data_s, data);

	data->sim = *sim;
	data->fun = NULL;
	data->fun_sig = fun;
	data->free = free;

	int OO = 1 + S;
	int II = 2;

	long odims[OO][N];
	md_copy_dims(N, odims[0], sim->mdims);

	for (int i = 0; i < S; i++)
		md_copy_dims(N, odims[i + 1], sig_dims[i]);

	long idims[II][N];
	md_transpose_dims(N, sim->MI_DIM, sim->MO_DIM, idims[0], sim->mdims);
	md_copy_dims(N, idims[1], sim->pdims);

	unsigned long diag_flags[OO][II];
	for (int i = 0; i < OO; i++)
		for (int j = 0; j < II; j++)
			diag_flags[i][j] = 0UL;

	return nlop_zblock_diag_generic_create(CAST_UP(data), N, OO, odims, II, idims, diag_flags, nlop_seq_fun, nlop_seq_free);
}

static void nlop_seq_apply(const struct nlop_s* op, int N,
	const long modims[N], complex float* omag,
	int S, const long sdims[S][N], complex float* signal[S],
	const long midims[N], const complex float* imag,
	const long pdims[N], const complex float* pars,
	const long dmmdims[N], complex float* dmm,
	const long dmpdims[N], complex float* dmp,
	const long dsmdims[S][N], complex float* dsm[S],
	const long dspdims[S][N], complex float* dsp[S])
{
	int OO = 1 + S;
	int II = 2;

	long odims[OO][N];
	long idims[II][N];
	long ddims[OO][II][N];

	complex float* dst_arr[OO];
	const complex float* src_arr[II];
	complex float* jac[OO][II];

	md_copy_dims(N, odims[0], modims);
	dst_arr[0] = omag;

	md_copy_dims(N, idims[0], midims);
	md_copy_dims(N, idims[1], pdims);
	src_arr[0] = imag;
	src_arr[1] = pars;

	md_copy_dims(N, ddims[0][0], dmmdims);
	md_copy_dims(N, ddims[0][1], dmpdims);
	jac[0][0] = dmm;
	jac[0][1] = dmp;

	for (int s = 0; s < S; s++) {

		md_copy_dims(N, odims[s + 1], sdims[s]);
		md_copy_dims(N, ddims[s + 1][0], dsmdims[s]);
		md_copy_dims(N, ddims[s + 1][1], dspdims[s]);

		dst_arr[s + 1] = signal[s];
		jac[s + 1][0] = dsm[s];
		jac[s + 1][1] = dsp[s];
	}

	nlop_zblock_diag_apply(op, N, OO, odims, dst_arr, II, idims, src_arr, ddims, jac);
}



static void pars_compute_grad(const struct sim_config_s* sim, float grad[3], int N, const long gdims[N], complex float* grad_pars)
{
	assert(3 == bitcount(sim->spatial_flags));

	unsigned long flags = 0;
	complex float grad2[N];
	float offset = 0;

	for (int i = 0, j = 0; i < N; i++) {

		grad2[i] = 0.;

		if (!MD_IS_SET(sim->spatial_flags, i))
			continue;

		if (1 < gdims[i] && 0. != grad[j]) {

			flags = MD_SET(flags, i);
			grad2[i] = grad[j] * sim->voxel_size[j] / (gdims[i] - 1.);
			offset -= ((int)(gdims[i] / 2.)) * grad2[i];
		}

		j++;
	}

	md_clear(N, gdims, grad_pars, CFL_SIZE);

	if (0 != flags) {

		md_zgradient(N, gdims, grad_pars, grad2);
		md_zsadd(N, gdims, grad_pars, grad_pars, offset);
	}
}

static void pars_add_grad(const struct sim_config_s* sim, float grad[3], int N,
				const long epdims[N], complex float* epars,
				const long pdims[N], const complex float* pars)
{
	md_copy2(N, epdims, MD_STRIDES(N, epdims, CFL_SIZE), epars, MD_STRIDES(N, pdims, CFL_SIZE), pars, CFL_SIZE);

	long gdims[N];
	md_select_dims(N, sim->spatial_flags, gdims, epdims);

	complex float* grad_pars = md_alloc_sameplace(N, gdims, CFL_SIZE, epars);
	pars_compute_grad(sim, grad, N, gdims, grad_pars);

	long pstrs[N];
	md_calc_strides(N, pstrs, epdims, CFL_SIZE);
	complex float* B0map = epars + B0_IDX * (pstrs[sim->PI_DIM] / (long)CFL_SIZE);

	long map_dims[N];
	md_select_dims(N, ~MD_BIT(sim->PI_DIM), map_dims, epdims);

	md_zadd2(N, map_dims, pstrs, B0map, pstrs, B0map, MD_STRIDES(N, gdims, CFL_SIZE), grad_pars);

	md_free(grad_pars);
}




struct pulse_s {

	INTERFACE(nlop_seq_data_t);

	struct pulse* pulse;
	float phase;
	struct rf_shape* shape;

	float grad[3];

	float h;
};

DEF_TYPEID(pulse_s);

static complex float lerp_pulse_shape(const struct rf_shape* shape, float t)
{
	// Normalize time to [0, 1]
	float t_norm = t / shape->sar_dur;

	// Convert to continuous index in [0, samples-1]
	float x = t_norm * (shape->samples - 1);

	// Get neighboring integer indices and clamp to valid range
	int x0 = (int)floorf(x);
	int x1 = x0 + 1;
	
	x0 = MAX(0, MIN(shape->samples - 1, x0));
	x1 = MAX(0, MIN(shape->samples - 1, x1));
	
	// Linear interpolation weight (fractional part)
	float alpha = x - (float)x0;
	
	complex float y0 = shape->shape[x0];
	complex float y1 = shape->shape[x1];

	// Linear interpolation
	return (1. - alpha) * y0 + alpha * y1;
}

static void simulate(float h, float tol, float r1, float r2, float B0, float B1, bool dini, bool dpars, int P, float state[1 + P][3], const struct pulse* ps, const struct rf_shape* shape, float phase)
{
	NESTED(void, call_fun, (float* out, float t, const float* in))
	{
		complex float pulse_sample = 0.;
		if (ps != NULL)
			pulse_sample = pulse_eval(ps, t);
		else if (shape != NULL)
			pulse_sample = lerp_pulse_shape(shape, t);
		
		complex float p = cexpf(1.i * phase) * pulse_sample;
		p = conjf(p);
		float gb[3] = { B1 * crealf(p), B1 * cimagf(p), B0 };

		bloch_ode(out, in, r1, r2, gb);
	};

	NESTED(void, call_pdy2, (float* out, float t, const float* in))
	{
		complex float pulse_sample = 0.;
		if (ps != NULL)
			pulse_sample = pulse_eval(ps, t);
		else if (shape != NULL)
			pulse_sample = lerp_pulse_shape(shape, t);
		
		complex float p = cexpf(1.i * phase) * pulse_sample;
		p = conjf(p);
		float gb[3] = { B1 * crealf(p), B1 * cimagf(p), B0 };

		bloch_pdy((float(*)[3])out, in, r1, r2, gb);
	};

	NESTED(void, call_pdp2, (float* out, float t, const float* in))
	{
		complex float pulse_sample = 0.;
		if (ps != NULL)
			pulse_sample = pulse_eval(ps, t);
		else if (shape != NULL)
			pulse_sample = lerp_pulse_shape(shape, t);
		
		complex float p = cexpf(1.i * phase) * pulse_sample;
		p = conjf(p);
		float gb[3] = { B1 * crealf(p), B1 * cimagf(p), B0 };

		if (dini)
			for (int i = 0; i < 3 * 3; i++)
				out[i] = 0.;

		if (dpars)
			bloch_b1b0_pdp((float(*)[3])out + (dini ? 3 : 0), in, r1, r2, gb, conj(p));
	};

	float dur = ps ? ps->duration : shape->sar_dur;

	if (dini || dpars)
		// Solve with sensitivity analysis
		ode_direct_sa(h, tol, 3, P, state, 0, dur, call_fun, call_pdy2, call_pdp2);
	else
		// Solve without sensitivity analysis (only bloch equation)
		ode_interval(h, tol, 3, state[0], 0, dur, call_fun);
}

static void seq_check_dims(struct sim_config_s* conf, int N, int OO, const long odims[OO][N], int II, const long idims[II][N], const long /*ddims*/[OO][II][N])
{
	assert(1 == OO);
	assert(2 == II);

	assert(conf->MO_DIM < N);
	assert(conf->MI_DIM < N);
	assert(conf->PI_DIM < N);

	assert(odims[0][conf->MO_DIM] == idims[0][conf->MI_DIM]);
	assert(md_check_equal_dims(N, odims[0], idims[0], ~(MD_BIT(conf->MO_DIM) | MD_BIT(conf->MI_DIM))));
	assert(1 == odims[0][conf->MI_DIM]);
	assert(1 == odims[0][conf->PI_DIM]);
	assert(1 == idims[0][conf->MO_DIM]);
	assert(1 == idims[1][conf->MO_DIM]);
	assert(conf->PI_DIM == conf->MI_DIM || 1 == idims[0][conf->PI_DIM]);
}

static void init_id_matrix2(int a, int b, int N, const long dims[N], const long strs[N], complex float* mat)
{
	assert(dims[a] == dims[b]);
	long M = dims[a];

	complex float diag[M][M];
	for (long i = 0; i < M; i++)
		for (long j = 0; j < M; j++)
			diag[i][j] = (i == j) ? 1. : 0.;

	long strs2[N];
	md_singleton_strides(N, strs2);
	strs2[a] = CFL_SIZE;
	strs2[b] = M * strs2[a];

	if (strs[a] > strs[b])
		SWAP(strs2[a], strs2[b]);

	md_copy2(N, dims, strs, mat, strs2, diag, CFL_SIZE);
}

/**
 * mdims = { N, 1, M }
 * pdims = { 1, P, M }
 * dmdims = { N, N, M }
 * dpdims = { N, P, M }
 * M - Batch size
 * N - State size
 * P - Parameter size
**/
//
static void pulse_sim_vec(const long mdims[3], const long mstrs[3], complex float* mag,
			  const long pdims[3], const long pstrs[3], const complex float* par,
			  const long dmdims[3], const long dmstrs[3], complex float* dmag,
			  const long dpdims[3], const long dpstrs[3], complex float* dpar,
			  struct pulse* pulse, struct rf_shape* shape, float phase, float h, float tol)
{
	assert(1 == mdims[1]);
	assert(1 == pdims[0]);

	assert(NULL == dmag || dmstrs[1] == dmstrs[0] * dmdims[0]);
	assert(NULL == dpar || dpstrs[1] == dpstrs[0] * dpdims[0]);

	assert(NULL == dmag || mdims[0] == dmdims[0]);
	assert(NULL == dpar || mdims[0] == dpdims[0]);

	if (NULL != dmag)
		init_id_matrix2(0, 1, 3, dmdims, dmstrs, dmag);

	if (NULL != dpar)
		md_zfill2(3, dpdims, dpstrs, dpar, 0);

	long M = mdims[2];

#ifdef USE_CUDA
	if (cuda_ondevice(mag)) {

		complex float* dpulse = NULL;
		long Np;
		float duration;

		if (NULL != shape) {

			Np = shape->samples;
			duration = shape->sar_dur;

			dpulse = md_alloc_sameplace(1, MD_DIMS(Np), CFL_SIZE, mag);
			md_copy(1, MD_DIMS(Np), dpulse, shape->shape, CFL_SIZE);
		}
		else if (NULL != pulse) {

			Np = 100000;
			duration = pulse->duration;

			dpulse = md_alloc_sameplace(1, MD_DIMS(Np), CFL_SIZE, mag);
			pulse_discretize(pulse, Np - 1, dpulse);
		}
		else
			error("Either pulse or shape must be provided for GPU simulation\n");

		md_zsmul(1, MD_DIMS(Np), dpulse, dpulse, cexpf(1.i * phase));

		assert(dmstrs[1] == dmstrs[0] * dmdims[0]);
		assert(dpstrs[1] == dpstrs[0] * dpdims[0]);

		cuda_ode_interval_bloch_sa(M, mstrs[0] / (long)CFL_SIZE, mstrs[2] / (long)CFL_SIZE, mag,
					   dmstrs[0] / (long)CFL_SIZE, dmstrs[2] / (long)CFL_SIZE, dmag,
					   dpstrs[0] / (long)CFL_SIZE, dpstrs[2] / (long)CFL_SIZE, dpar,
					   pstrs[1] / (long)CFL_SIZE, pstrs[2] / (long)CFL_SIZE, par,
					   Np, duration, dpulse, h, tol, 0, duration);

		md_free(dpulse);
	}
	else
#endif
	{
		int P = 0;

		if (NULL != dmag)
			P += dmdims[1];

		if (NULL != dpar)
			P += dpdims[1];

#pragma		omp parallel for
		for (long i = 0; i < M; i++) {

			float state[1 + P][3];

			complex float* tmag = mag + i * mstrs[2] / (long)CFL_SIZE;
			const complex float* tpar = par + i * pstrs[2] / (long)CFL_SIZE;

			complex float* tdpar = (NULL == dpar) ? NULL : dpar + i * dpstrs[2] / (long)CFL_SIZE;
			complex float* tdmag = (NULL == dmag) ? NULL : dmag + i * dmstrs[2] / (long)CFL_SIZE;

			float r1 = crealf(tpar[R1_IDX * pstrs[1] / (long)CFL_SIZE]);
			float r2 = crealf(tpar[R2_IDX * pstrs[1] / (long)CFL_SIZE]);
			float B1 = crealf(tpar[B1_IDX * pstrs[1] / (long)CFL_SIZE]);
			float B0 = crealf(tpar[B0_IDX * pstrs[1] / (long)CFL_SIZE]);

			for (int j = 0; j < mdims[0]; j++)
				state[0][j] = crealf(tmag[j * mstrs[0] / (long)CFL_SIZE]);

			int p = 1;

			if (NULL != tdmag) {

				for (int k = 0; k < dmdims[1]; k++, p++)
					for (int j = 0; j < dmdims[0]; j++)
						state[p][j] = crealf(tdmag[(j * dmstrs[0] +  k * dmstrs[1]) / (long)CFL_SIZE]);
			}

			if (NULL != tdpar) {

				for (int k = 0; k < dpdims[1]; k++, p++)
					for (int j = 0; j < dpdims[0]; j++)
						state[p][j] = crealf(tdpar[(j * dpstrs[0] +  k * dpstrs[1]) / (long)CFL_SIZE]);
			}


			simulate(h, tol, r1, r2, B0, B1, dmag != NULL, dpar!= NULL, P, state, pulse, shape, phase);


			for (int j = 0; j < mdims[0]; j++)
				tmag[j * mstrs[0] / (long)CFL_SIZE] = state[0][j];

			p = 1;

			if (NULL != tdmag) {

				for (int k = 0; k < dmdims[1]; k++, p++)
					for (int j = 0; j < dmdims[0]; j++)
						tdmag[(j * dmstrs[0] +  k * dmstrs[1]) / (long)CFL_SIZE] = state[p][j];
			}

			if (NULL != tdpar) {

				for (int k = 0; k < dpdims[1]; k++, p++)
					for (int j = 0; j < dpdims[0]; j++)
						tdpar[(j * dpstrs[0] +  k * dpstrs[1]) / (long)CFL_SIZE] = state[p][j];
			}
		}
	}
}

static void pulse_fun(const struct nlop_seq_data_s* data, int N,
	const long modims[N], complex float* omag, 	// output magnetization
	const long midims[N], const complex float* imag,// input magnetization
	const long pdims[N], const complex float* pars, // parameters (R1, R2 B1, B0)
	const long dmmdims[N], complex float* dmm, 	// derivative of omag w.r.t. imag
	const long dmpdims[N], complex float* dmp) 	// derivative of omag w.r.t. pars
{
	auto d = CAST_DOWN(pulse_s, data);

	long epdims[N];		// expanded parameter dims
	long edmpdims[N];	// expanded derivative parameter dims

	md_copy_dims(N, epdims, pdims);
	md_copy_dims(N, edmpdims, dmpdims);

	md_max_dims(N, ~(MD_BIT(data->sim.MI_DIM) | MD_BIT(data->sim.MO_DIM) | MD_BIT(data->sim.PI_DIM)), epdims, epdims, modims);
	md_max_dims(N, ~(MD_BIT(data->sim.MI_DIM) | MD_BIT(data->sim.MO_DIM) | MD_BIT(data->sim.PI_DIM)), edmpdims, edmpdims, modims);

	complex float* epars = md_alloc_sameplace(N, epdims, CFL_SIZE, pars);
	pars_add_grad(&(data->sim), d->grad, N, epdims, epars, pdims, pars); // Add gradient to B0 so that B0 + Gz (for slice selection)

	complex float* edmp = NULL;

	if (NULL != dmp) {

		edmp = md_alloc_sameplace(N, edmpdims, CFL_SIZE, dmp);
		md_clear(N, edmpdims, edmp, CFL_SIZE);
	}

	long mostrs[N];
	long mistrs[N];
	long pstrs[N];
	long dmmstrs[N];
	long dmpstrs[N];

	md_calc_strides(N, mostrs, modims, CFL_SIZE);
	md_calc_strides(N, mistrs, midims, CFL_SIZE);
	md_calc_strides(N, pstrs, epdims, CFL_SIZE);
	md_calc_strides(N, dmmstrs, dmmdims, CFL_SIZE);
	md_calc_strides(N, dmpstrs, edmpdims, CFL_SIZE);

	md_transpose(N, data->sim.MO_DIM, data->sim.MI_DIM, modims, omag, midims, imag, CFL_SIZE); // Copy imag to omag (and transpose M0 and MI)

	unsigned long bflags = md_nontriv_dims(N, modims); // Batch flags

	bflags &= ~MD_BIT(data->sim.MO_DIM);
	bflags &= ~MD_BIT(data->sim.MI_DIM);

	int bidx = md_min_idx(bflags);

	// reduced dims and strides (prep for simulation)
	long rmdims[3] = { modims[data->sim.MO_DIM], 1, (-1 == bidx) ? 1 : modims[bidx] };
	long rmstrs[3] = { mostrs[data->sim.MO_DIM], 0, (-1 == bidx) ? 0 : mostrs[bidx] };
	long rpdims[3] = { 1, epdims[data->sim.PI_DIM], (-1 == bidx) ? 1 : modims[bidx] };
	long rpstrs[3] = { 0, pstrs[data->sim.PI_DIM], (-1 == bidx) ? 0 : pstrs[bidx] };
	long dmdims[3] = { dmmdims[data->sim.MO_DIM], dmmdims[data->sim.MI_DIM], (-1 == bidx) ? 1 : modims[bidx] };
	long dmstrs[3] = { dmmstrs[data->sim.MO_DIM], dmmstrs[data->sim.MI_DIM], (-1 == bidx) ? 0 : dmmstrs[bidx] };
	long dpdims[3] = { edmpdims[data->sim.MO_DIM], edmpdims[data->sim.PI_DIM], (-1 == bidx) ? 1 : modims[bidx] };
	long dpstrs[3] = { dmpstrs[data->sim.MO_DIM], dmpstrs[data->sim.PI_DIM], (-1 == bidx) ? 0 : dmpstrs[bidx] };

	if (-1 != bidx)
		bflags = MD_CLEAR(bflags, bidx);

	for (int i = 0; i < N; i++) {

		if (    MD_IS_SET(bflags, i)
		    && mostrs[i] == rmdims[2] * rmstrs[2]
		    && pstrs[i] == rpdims[2] * rpstrs[2]
		    && dmmstrs[i] == dmdims[2] * dmstrs[2]
		    && dmpstrs[i] == dpdims[2] * dpstrs[2])
			{
				bflags = MD_CLEAR(bflags, i);

				rmdims[2] *= modims[i];
				rpdims[2] *= modims[i];
				dmdims[2] *= modims[i];
				dpdims[2] *= modims[i];
			}
	}


	long pos[N];
	md_set_dims(N, pos, 0);

	unsigned long lflags = bflags;


	do {
		complex float* tmag = MD_ACCESS_PTR(N, mostrs, pos, omag);
		const complex float* tpar = MD_ACCESS_PTR(N, pstrs, pos, epars);
		complex float* tdmag = MD_ACCESS_PTR(N, dmmstrs, pos, dmm);
		complex float* tdpar = MD_ACCESS_PTR(N, dmpstrs, pos, edmp);

		pulse_sim_vec(rmdims, rmstrs, tmag, rpdims, rpstrs, tpar,
			dmdims, dmstrs, tdmag, dpdims, dpstrs, tdpar,
			d->pulse, d->shape, d->phase, d->h, data->sim.tol);


	} while (md_next(N, modims, lflags, pos));

	md_free(epars);

	if (NULL != edmp) {

		md_zsum(N, edmpdims, ~md_nontriv_dims(N, dmpdims), dmp, edmp);
		md_free(edmp);
	}
}

static void nlop_pulse_free(const struct nlop_seq_data_s* _data)
{
	auto data = CAST_DOWN(pulse_s, _data);

	if(data->pulse != NULL)
		pulse_free(data->pulse);
	xfree(data);
}

const struct nlop_s* nlop_pulse_create(struct sim_config_s sim, const struct pulse* pulse, float phase, float grad[3])
{
	PTR_ALLOC(struct pulse_s, data);
	SET_TYPEID(pulse_s, data);

	data->pulse = pulse_clone(pulse);
	data->phase = phase;
	data->shape = NULL;
	data->h = pulse->duration / 100.;

	for (int i = 0; i < 3; i++)
		data->grad[i] = grad[i];

	return nlop_seq_create(&sim, CAST_UP(PTR_PASS(data)), pulse_fun, nlop_pulse_free);
}

const struct nlop_s* nlop_pulse_shape_create(struct sim_config_s sim, struct rf_shape* shape, float phase, float grad[3])
{
	PTR_ALLOC(struct pulse_s, data);
	SET_TYPEID(pulse_s, data);

	data->pulse = NULL;
	data->phase = phase;
	data->shape = shape;
	data->h = shape->sar_dur / 100.;

	for (int i = 0; i < 3; i++)
		data->grad[i] = grad[i];

	return nlop_seq_create(&sim, CAST_UP(PTR_PASS(data)), pulse_fun, nlop_pulse_free);
}


const struct nlop_s* nlop_phase_wrap_F(struct sim_config_s sim, const struct nlop_s* nlop, float phase)
{
	list_t ret = list_create();

	list_append(ret, (struct nlop_s*)nlop_rotz_create(sim, -phase));
	list_append(ret, (struct nlop_s*)nlop);
	list_append(ret, (struct nlop_s*)nlop_rotz_create(sim, phase));

	return nlop_simu_jacobian_chain_create(sim, ret);
}


struct adc_s {

	INTERFACE(nlop_seq_data_t);

	long index;
	const long* wgh_dims;
	struct multiplace_array_s* wgh;
};

DEF_TYPEID(adc_s);

static void adc_fun(const struct nlop_seq_data_s* data, int N,
	const long modims[N], complex float* omag,
	int S, const long sdims[S][N], complex float* signal[S],
	const long midims[N], const complex float* imag,
	const long /*pdims*/[N], const complex float* /*pars*/,
	const long dmmdims[N], complex float* dmm,
	const long dmpdims[N], complex float* dmp,
	const long dsmdims[S][N], complex float* dsm[S],
	const long dspdims[S][N], complex float* dsp[S])
{
	auto d = CAST_DOWN(adc_s, data);

	assert(1 == S);

	if (NULL != dmm)
		init_id_matrix2(data->sim.MO_DIM, data->sim.MI_DIM, N, dmmdims, MD_STRIDES(N, dmmdims, CFL_SIZE), dmm);

	if (NULL != dmp)
		md_clear(N, dmpdims, dmp, CFL_SIZE);

	if (NULL != dsm[0])
		md_copy2(N, dsmdims[0], MD_STRIDES(N, dsmdims[0], CFL_SIZE), dsm[0], MD_STRIDES(N, d->wgh_dims, CFL_SIZE), multiplace_read(d->wgh, dsm[0]), CFL_SIZE);

	if (NULL != dsp[0])
		md_clear(N, dspdims[0], dsp[0], CFL_SIZE);

	md_transpose(N, data->sim.MO_DIM, data->sim.MI_DIM, modims, omag, midims, imag, CFL_SIZE);

	long max_dims[N];
	assert(md_check_compat(N, ~0UL, d->wgh_dims, midims));
	md_max_dims(N, ~0UL, max_dims, d->wgh_dims, midims);
	md_ztenmul2(N, max_dims, MD_STRIDES(N, sdims[0], CFL_SIZE), signal[0], MD_STRIDES(N, midims, CFL_SIZE), imag, MD_STRIDES(N, d->wgh_dims, CFL_SIZE), multiplace_read(d->wgh, imag));
}

static void nlop_adc_free(const struct nlop_seq_data_s* _data)
{
	auto data = CAST_DOWN(adc_s, _data);

	multiplace_free(data->wgh);
	xfree(data->wgh_dims);

	xfree(data);
}

const struct nlop_s* nlop_adc_create(struct sim_config_s sim, long index, unsigned long sflags, float phase)
{
	PTR_ALLOC(struct adc_s, data);
	SET_TYPEID(adc_s, data);

	assert(3 == sim.mdims[sim.MO_DIM]);

	data->index = index;
	complex float id[3][3] = { { 1., 0., 0. }, { 0., 1., 0. }, { 0., 0., 1. } };
	complex float acc[3] = { cexpf((M_PI_2 - phase) * 1.i), cexpf(-phase * 1.i), 0. };

	long wgh_dims[sim.N];
	md_singleton_dims(sim.N, wgh_dims);
	wgh_dims[sim.MO_DIM] = sim.mdims[sim.MO_DIM];
	wgh_dims[sim.MI_DIM] = sim.mdims[sim.MO_DIM];
	md_select_dims(sim.N, ~sflags, wgh_dims, wgh_dims);
	//assert(3 == md_calc_size(sim.N, wgh_dims));
	assert(!MD_IS_SET(sflags, sim.MI_DIM));
	data->wgh_dims = ARR_CLONE(long[sim.N], wgh_dims);
	data->wgh = multiplace_move(sim.N, wgh_dims, CFL_SIZE, MD_IS_SET(sflags, sim.MO_DIM) ? acc : &(id[0][0]));

	long signal_dims[1][sim.N];
	md_select_dims(sim.N, ~sflags, signal_dims[0], sim.mdims);

	return nlop_sig_seq_create(&sim, CAST_UP(PTR_PASS(data)), adc_fun, nlop_adc_free, 1, sim.N, signal_dims);
}

struct rot_s {

	INTERFACE(nlop_seq_data_t);
	struct multiplace_array_s* wgh;
};

DEF_TYPEID(rot_s);

static void rot_fun(const struct nlop_seq_data_s* data, int N,
	const long modims[N], complex float* omag,
	const long midims[N], const complex float* imag,
	const long /*pdims*/[N], const complex float* /*pars*/,
	const long dmmdims[N], complex float* dmm,
	const long dmpdims[N], complex float* dmp)
{
	auto d = CAST_DOWN(rot_s, data);

	long mat_dims[N];
	md_select_dims(N, MD_BIT(data->sim.MO_DIM) | MD_BIT(data->sim.MI_DIM), mat_dims, dmmdims);

	md_ztenmul(N, modims, omag, midims, imag, mat_dims, multiplace_read(d->wgh, omag));

	if (NULL != dmm)
		md_copy2(N, dmmdims, MD_STRIDES(N, dmmdims, CFL_SIZE), dmm, MD_STRIDES(N, mat_dims, CFL_SIZE), multiplace_read(d->wgh, dmm), CFL_SIZE);

	if (NULL != dmp)
		md_clear(N, dmpdims, dmp, CFL_SIZE);
}

static void nlop_rot_free(const struct nlop_seq_data_s* _data)
{
	auto data = CAST_DOWN(rot_s, _data);
	multiplace_free(data->wgh);

	xfree(data);
}

static const struct nlop_s* nlop_seq_mat_create(struct sim_config_s sim, float mat[3][3])
{
	PTR_ALLOC(struct rot_s, data);
	SET_TYPEID(rot_s, data);

	long wdims[sim.N];
	long wstrs[sim.N];

	md_singleton_dims(sim.N, wdims);
	wdims[sim.MO_DIM] = sim.mdims[sim.MO_DIM];
	wdims[sim.MI_DIM] = sim.mdims[sim.MO_DIM];
	md_calc_strides(sim.N, wstrs, wdims, CFL_SIZE);

	complex float mat2[9];
	long pos[sim.N];
	md_set_dims(sim.N, pos, 0);

	for (int o = 0; o < 3; o++) {

		for (int i = 0; i < 3; i++) {

			pos[sim.MO_DIM] = o;
			pos[sim.MI_DIM] = i;
			MD_ACCESS(sim.N, wstrs, pos, mat2) = mat[o][i];
		}
	}

	data->wgh = multiplace_move(sim.N, wdims, CFL_SIZE, mat2);

	return nlop_seq_create(&sim, CAST_UP(PTR_PASS(data)), rot_fun, nlop_rot_free);
}

const struct nlop_s* nlop_rotx_create(struct sim_config_s sim, float angle)
{
	assert(3 == sim.mdims[sim.MO_DIM]);

	complex float ca = cosf(angle);
	complex float sa = sinf(angle);

	float ret[3][3] = {
		{ 1., 0., 0.	},
		{ 0., ca, sa	},
		{ 0., -sa, ca	},
	};

	return nlop_seq_mat_create(sim, ret);
}

const struct nlop_s* nlop_roty_create(struct sim_config_s sim, float angle)
{
	assert(3 ==sim.mdims[sim.MO_DIM]);

	complex float ca = cosf(angle);
	complex float sa = sinf(angle);

	float ret[3][3] = {
		{ca, 0., -sa	},
		{0., 1., 0.	},
		{sa, 0., ca	},
	};

	return nlop_seq_mat_create(sim, ret);
}

const struct nlop_s* nlop_rotz_create(struct sim_config_s sim, float angle)
{
	assert(3 == sim.mdims[sim.MO_DIM]);

	complex float ca = cosf(angle);
	complex float sa = sinf(angle);

	float ret[3][3] = {
		{ ca, sa, 0.	},
		{ -sa, ca, 0.	},
		{ 0., 0., 1.	}
	};

	return nlop_seq_mat_create(sim, ret);
}


static complex float* extract_map(struct sim_config_s sim, int map_idx, int N, long map_dims[N], const long pdims[N], const complex float* pars)
{
	md_select_dims(N, ~MD_BIT(sim.PI_DIM), map_dims, pdims);

	long pos[N];
	md_set_dims(N, pos, 0);
	pos[sim.PI_DIM] = map_idx;

	complex float* map = md_alloc_sameplace(N, map_dims, CFL_SIZE, pars);
	md_slice(N, MD_BIT(sim.PI_DIM), pos, pdims, map, pars, CFL_SIZE);

	return map;
}

static complex float* affine_map(struct sim_config_s sim, int N, long adims[N], const long pdims[N], const complex float* pars)
{
	md_copy_dims(N, adims, pdims);

	adims[sim.MO_DIM] = sim.mdims[sim.MO_DIM];
	adims[sim.MI_DIM] = sim.mdims[sim.MO_DIM] + 1;

	complex float* aff = md_alloc_sameplace(N, adims, CFL_SIZE, pars);
	md_clear(N, adims, aff, CFL_SIZE);
	return aff;
}

static void affine_set(struct sim_config_s sim, int oidx, int iidx, int N, long aff_dims[N], complex float* aff, long map_dims[N], const complex float* map)
{
	assert(md_check_equal_dims(N, aff_dims, map_dims, ~(MD_BIT(sim.MO_DIM) | MD_BIT(sim.MI_DIM))));

	long pos[N];
	md_set_dims(N, pos, 0);
	pos[sim.MO_DIM] = oidx;
	pos[sim.MI_DIM] = iidx;

	md_copy_block(N, pos, aff_dims, aff, map_dims, map, CFL_SIZE);
}

static void affine_mul(struct sim_config_s sim, int i, int N, const long modims[N], complex float* omag, long aff_dims[N], const complex float* aff, const long midims[N], const complex float* imag)
{
	long mostrs[N];
	md_calc_strides(N, mostrs, modims, CFL_SIZE);

	long pos[N];
	md_set_dims(N, pos, 0);
	pos[sim.PI_DIM] = i;
	assert(i < modims[sim.PI_DIM]);
	omag = &(MD_ACCESS(N, mostrs, pos, omag));

	long tmodims[N];
	md_select_dims(N, ~MD_BIT(sim.MI_DIM), tmodims, modims);

	long aff_strs[N];
	md_calc_strides(N, aff_strs, aff_dims, CFL_SIZE);

	md_set_dims(N, pos, 0);
	pos[sim.PI_DIM] = aff_dims[sim.PI_DIM] - 1;
	const complex float* offset = &MD_ACCESS(N, aff_strs, pos, aff);

	md_copy2(N, tmodims, mostrs, omag, aff_strs, offset, CFL_SIZE);

	long lin_dims[N];
	md_copy_dims(N, lin_dims, aff_dims);
	lin_dims[sim.MI_DIM]--;
	md_max_dims(N, ~(MD_BIT(sim.MO_DIM) | MD_BIT(sim.MI_DIM)) , lin_dims, modims, lin_dims);

	mostrs[sim.MI_DIM] = 0;
	md_zfmac2(N, lin_dims, mostrs, omag, aff_strs, aff, MD_STRIDES(N, midims, CFL_SIZE), imag);
}




struct relax_s {

	INTERFACE(nlop_seq_data_t);

	float t;

	float grad[3];
};

DEF_TYPEID(relax_s);


static void relax_fun(const struct nlop_seq_data_s* data, int N,
	const long modims[N], complex float* omag,
	const long midims[N], const complex float* imag,
	const long pdims[N], const complex float* pars,
	const long dmmdims[N], complex float* dmm,
	const long dmpdims[N], complex float* dmp)
{
	auto d = CAST_DOWN(relax_s, data);

	long aff_dims[N];
	complex float* affine = affine_map(data->sim, N, aff_dims, pdims, pars);

	long map_dims[N];
	complex float* R2map = extract_map(data->sim, R2_IDX, N, map_dims, pdims, pars);
	md_zsmul(N, map_dims, R2map, R2map, -d->t);
	md_zexp(N, map_dims, R2map, R2map);

	complex float* R1map = extract_map(data->sim, R1_IDX, N, map_dims, pdims, pars);
	md_zsmul(N, map_dims, R1map, R1map, -d->t);
	md_zexp(N, map_dims, R1map, R1map);

	complex float* offset = md_alloc_sameplace(N, map_dims, CFL_SIZE, pars);
	md_zfill(N, map_dims, offset, 1.);
	md_zsub(N, map_dims, offset, offset, R1map);

	affine_set(data->sim, 0, 0, N, aff_dims, affine, map_dims, R2map);
	affine_set(data->sim, 1, 1, N, aff_dims, affine, map_dims, R2map);
	affine_set(data->sim, 2, 2, N, aff_dims, affine, map_dims, R1map);
	affine_set(data->sim, 2, 3, N, aff_dims, affine, map_dims, offset);

	affine_mul(data->sim, 0, N, modims, omag, aff_dims, affine, midims, imag);

	if (NULL != dmm)
		md_copy2(N, dmmdims, MD_STRIDES(N, dmmdims, CFL_SIZE), dmm, MD_STRIDES(N, aff_dims, CFL_SIZE), affine, CFL_SIZE);

	if (NULL != dmp) {

		md_clear(N, dmpdims, dmp, CFL_SIZE);

		md_clear(N, aff_dims, affine, CFL_SIZE);
		md_zsmul(N, map_dims, R2map, R2map, -d->t);
		affine_set(data->sim, 0, 0, N, aff_dims, affine, map_dims, R2map);
		affine_set(data->sim, 1, 1, N, aff_dims, affine, map_dims, R2map);
		affine_mul(data->sim, R2_IDX, N, dmpdims, dmp, aff_dims, affine, midims, imag);

		md_clear(N, aff_dims, affine, CFL_SIZE);
		md_zsmul(N, map_dims, R1map, R1map, -d->t);
		affine_set(data->sim, 2, 2, N, aff_dims, affine, map_dims, R1map);
		md_zsmul(N, map_dims, R1map, R1map, -1.);
		affine_set(data->sim, 2, 3, N, aff_dims, affine, map_dims, R1map);
		affine_mul(data->sim, R1_IDX, N, dmpdims, dmp, aff_dims, affine, midims, imag);
	}

	md_free(affine);
	md_free(offset);

	md_free(R1map);
	md_free(R2map);
}

static void relax_phase_fun(const struct nlop_seq_data_s* data, int N,
	const long modims[N], complex float* omag,
	const long midims[N], const complex float* imag,
	const long pdims[N], const complex float* pars,
	const long dmmdims[N], complex float* dmm,
	const long dmpdims[N], complex float* dmp)
{
	auto d = CAST_DOWN(relax_s, data);

	long map_dims[N];
	complex float* tB0map = extract_map(data->sim, B0_IDX, N, map_dims, pdims, pars);

	long gdims[N];
	md_select_dims(N, data->sim.spatial_flags, gdims, modims);
	complex float* grad = md_alloc_sameplace(N, gdims, CFL_SIZE, pars);

	pars_compute_grad(&data->sim, d->grad, N, gdims, grad);

	long emap_dims[N];
	md_max_dims(N, ~0UL, emap_dims, map_dims, gdims);
	complex float* B0map = md_alloc_sameplace(N, emap_dims, CFL_SIZE, pars);
	md_zadd2(N, emap_dims, MD_STRIDES(N, emap_dims, CFL_SIZE), B0map, MD_STRIDES(N, map_dims, CFL_SIZE), tB0map, MD_STRIDES(N, gdims, CFL_SIZE), grad);
	md_free(grad);
	md_free(tB0map);

	long aff_dims[N];
	complex float* affine = affine_map(data->sim, N, aff_dims, emap_dims, pars);

	md_zsmul(N, emap_dims, B0map, B0map, d->t);

	complex float* ca = md_alloc_sameplace(N, emap_dims, CFL_SIZE, pars);
	complex float* sa = md_alloc_sameplace(N, emap_dims, CFL_SIZE, pars);
	complex float* mca = md_alloc_sameplace(N, emap_dims, CFL_SIZE, pars);
	complex float* msa = md_alloc_sameplace(N, emap_dims, CFL_SIZE, pars);
	complex float* one = md_alloc_sameplace(N, emap_dims, CFL_SIZE, pars);

	md_zfill(N, emap_dims, one, 1.);
	md_zcos(N, emap_dims, ca, B0map);
	md_zsin(N, emap_dims, sa, B0map);
	md_zsmul(N, emap_dims, mca, ca, -1.);
	md_zsmul(N, emap_dims, msa, sa, -1.);

	affine_set(data->sim, 0, 0, N, aff_dims, affine, emap_dims, ca);
	affine_set(data->sim, 1, 1, N, aff_dims, affine, emap_dims, ca);
	affine_set(data->sim, 2, 2, N, aff_dims, affine, emap_dims, one);
	affine_set(data->sim, 0, 1, N, aff_dims, affine, emap_dims, sa);
	affine_set(data->sim, 1, 0, N, aff_dims, affine, emap_dims, msa);

	affine_mul(data->sim, 0, N, modims, omag, aff_dims, affine, midims, imag);

	if (NULL != dmm)
		md_copy2(N, dmmdims, MD_STRIDES(N, dmmdims, CFL_SIZE), dmm, MD_STRIDES(N, aff_dims, CFL_SIZE), affine, CFL_SIZE);

	if (NULL != dmp) {

		md_clear(N, dmpdims, dmp, CFL_SIZE);

		md_clear(N, aff_dims, affine, CFL_SIZE);
		affine_set(data->sim, 0, 0, N, aff_dims, affine, emap_dims, msa);
		affine_set(data->sim, 1, 1, N, aff_dims, affine, emap_dims, msa);
		affine_set(data->sim, 0, 1, N, aff_dims, affine, emap_dims, ca);
		affine_set(data->sim, 1, 0, N, aff_dims, affine, emap_dims, mca);
		md_zsmul(N, aff_dims, affine, affine, d->t);

		affine_mul(data->sim, B0_IDX, N, dmpdims, dmp, aff_dims, affine, midims, imag);
	}

	md_free(affine);
	md_free(ca);
	md_free(sa);
	md_free(mca);
	md_free(msa);
	md_free(one);
	md_free(B0map);
}

static void nlop_relax_free(const struct nlop_seq_data_s* _data)
{
	auto data = CAST_DOWN(relax_s, _data);
	free(data);
}


static const struct nlop_s* nlop_relax_phase_create(struct sim_config_s sim, float t, float grad[3])
{
	PTR_ALLOC(struct relax_s, data);
	SET_TYPEID(relax_s, data);

	data->t = t;
	for (int i = 0; i < 3; i++)
		data->grad[i] = grad[i];

	return nlop_seq_create(&sim, CAST_UP(PTR_PASS(data)), relax_phase_fun, nlop_relax_free);
}

/**
 * Create an nlop computing free relaxation
 *
 * @param sim
 * @param t time of relaxation
 * @param grad additional gradient
 *
 * Input tensors:
 * mag [Mx My Mz] stacked along MI_DIM		example: [ 3 1 X Y 1 1 1 Z ]
 * pars [R1 R2 B0 B1] stacked along PI_DIM	example: [ 1 4 X Y 1 1 1 1 ]
 *
 * Output tensors:
 * mag [Mx My Mz] stacked along MO_DIM		example: [ 3 1 X Y 1 1 1 Z ]
 */
const struct nlop_s* nlop_relax_create(struct sim_config_s sim, float t, float grad[3])

{
	PTR_ALLOC(struct relax_s, data);
	SET_TYPEID(relax_s, data);

	data->t = t;

	list_t nlops = list_create();
	list_append(nlops, (struct nlop_s*)nlop_seq_create(&sim, CAST_UP(PTR_PASS(data)), relax_fun, nlop_relax_free));
	list_append(nlops, (struct nlop_s*)nlop_relax_phase_create(sim, t, grad));

	return nlop_simu_jacobian_chain_create(sim, nlops);
}

static const struct nlop_s* nlop_hard_pulse_no_b1create(struct sim_config_s sim, float angle)
{
	assert(3 == sim.mdims[sim.MO_DIM]);

	complex float ca = cosf(angle);
	complex float sa = sinf(angle);

	float rot_x[3][3] = {
		{ 1., 0., 0.	},
		{ 0., ca, sa	},
		{ 0., -sa, ca	},
	};

	return nlop_seq_mat_create(sim, rot_x);
}


struct hardpulse_s {

	INTERFACE(nlop_seq_data_t);

	float angle;

};

DEF_TYPEID(hardpulse_s);


static void hardpulse_fun(const struct nlop_seq_data_s* data, int N,
	const long modims[N], complex float* omag,
	const long midims[N], const complex float* imag,
	const long pdims[N], const complex float* pars,
	const long dmmdims[N], complex float* dmm,
	const long dmpdims[N], complex float* dmp)
{
	auto d = CAST_DOWN(hardpulse_s, data);

	long map_dims[N];
	complex float* B1map = extract_map(data->sim, B1_IDX, N, map_dims, pdims, pars);

	long aff_dims[N];
	complex float* affine = affine_map(data->sim, N, aff_dims, pdims, pars);

	md_zsmul(N, map_dims, B1map, B1map, d->angle);

	complex float* ca = md_alloc_sameplace(N, map_dims, CFL_SIZE, pars);
	complex float* sa = md_alloc_sameplace(N, map_dims, CFL_SIZE, pars);
	complex float* mca = md_alloc_sameplace(N, map_dims, CFL_SIZE, pars);
	complex float* msa = md_alloc_sameplace(N, map_dims, CFL_SIZE, pars);
	complex float* one = md_alloc_sameplace(N, map_dims, CFL_SIZE, pars);

	md_zfill(N, map_dims, one, 1.);
	md_zcos(N, map_dims, ca, B1map);
	md_zsin(N, map_dims, sa, B1map);
	md_zsmul(N, map_dims, mca, ca, -1.);
	md_zsmul(N, map_dims, msa, sa, -1.);

	affine_set(data->sim, 0, 0, N, aff_dims, affine, map_dims, one);
	affine_set(data->sim, 1, 1, N, aff_dims, affine, map_dims, ca);
	affine_set(data->sim, 2, 2, N, aff_dims, affine, map_dims, ca);
	affine_set(data->sim, 1, 2, N, aff_dims, affine, map_dims, sa);
	affine_set(data->sim, 2, 1, N, aff_dims, affine, map_dims, msa);

	affine_mul(data->sim, 0, N, modims, omag, aff_dims, affine, midims, imag);

	if (NULL != dmm)
		md_copy2(N, dmmdims, MD_STRIDES(N, dmmdims, CFL_SIZE), dmm, MD_STRIDES(N, aff_dims, CFL_SIZE), affine, CFL_SIZE);

	if (NULL != dmp) {

		md_clear(N, dmpdims, dmp, CFL_SIZE);

		md_clear(N, aff_dims, affine, CFL_SIZE);
		affine_set(data->sim, 1, 1, N, aff_dims, affine, map_dims, msa);
		affine_set(data->sim, 2, 2, N, aff_dims, affine, map_dims, msa);
		affine_set(data->sim, 1, 2, N, aff_dims, affine, map_dims, ca);
		affine_set(data->sim, 2, 1, N, aff_dims, affine, map_dims, mca);
		md_zsmul(N, aff_dims, affine, affine, d->angle);

		affine_mul(data->sim, B1_IDX, N, dmpdims, dmp, aff_dims, affine, midims, imag);
	}

	md_free(affine);
	md_free(ca);
	md_free(sa);
	md_free(mca);
	md_free(msa);
	md_free(one);
	md_free(B1map);
}

static void nlop_harpulse_free(const struct nlop_seq_data_s* _data)
{
	auto data = CAST_DOWN(hardpulse_s, _data);
	free(data);
}


static const struct nlop_s* nlop_hard_pulse_b1create(struct sim_config_s sim, float angle)
{
	PTR_ALLOC(struct hardpulse_s, data);
	SET_TYPEID(hardpulse_s, data);

	data->angle = angle;

	return nlop_seq_create(&sim, CAST_UP(PTR_PASS(data)), hardpulse_fun, nlop_harpulse_free);
}

const struct nlop_s* nlop_hard_pulse_create(struct sim_config_s sim, bool b1, float angle, float phase)
{
	const struct nlop_s* rot_x = b1 ? nlop_hard_pulse_b1create(sim, angle) : nlop_hard_pulse_no_b1create(sim, angle);

	return nlop_phase_wrap_F(sim, rot_x, phase);
}

const struct nlop_s* nlop_spoile_create(struct sim_config_s sim)
{
	assert(3 == sim.mdims[sim.MO_DIM]);

	float spoile[3][3] = {
		{ 0., 0., 0. },
		{ 0., 0., 0. },
		{ 0., 0., 1. } };

	return nlop_seq_mat_create(sim, spoile);
}


struct simu_chain_s {

	INTERFACE(nlop_seq_data_t);

	list_t nlops;
};

DEF_TYPEID(simu_chain_s);

static void simu_jac_mul(int N, int MO_DIM, int MI_DIM, const long cdims[N], complex float* C, const long adims[N], complex float* A, const long bdims[N], complex float* B, bool add)
{
	if (C == A) {

		complex float* tA = md_alloc_sameplace(N, adims, CFL_SIZE, A);

		md_copy(N, adims, tA, A, CFL_SIZE);
		simu_jac_mul(N, MO_DIM, MI_DIM, cdims, C, adims, tA, bdims, B, add);

		md_free(tA);
		return;
	}

	if (C == B) {

		complex float* tB = md_alloc_sameplace(N, bdims, CFL_SIZE, B);

		md_copy(N, bdims, tB, B, CFL_SIZE);
		simu_jac_mul(N, MO_DIM, MI_DIM, cdims, C, adims, A, bdims, tB, add);

		md_free(tB);
		return;
	}

	long adims2[N + 1];
	long bdims2[N + 1];
	long cdims2[N + 1];

	long astrs[N + 1];
	long bstrs[N + 1];
	long cstrs[N + 1];

	md_copy_dims(N, adims2, adims);
	md_copy_dims(N, bdims2, bdims);
	md_copy_dims(N, cdims2, cdims);

	adims2[N + 0] = 1;
	bdims2[N + 0] = 1;
	cdims2[N + 0] = 1;

	md_calc_strides(N + 1, astrs, adims2, CFL_SIZE);
	md_calc_strides(N + 1, bstrs, bdims2, CFL_SIZE);
	md_calc_strides(N + 1, cstrs, cdims2, CFL_SIZE);

	SWAP(astrs[MI_DIM], astrs[N + 0]);
	SWAP(bstrs[MO_DIM], bstrs[N + 0]);
	SWAP(adims2[MI_DIM], adims2[N + 0]);
	SWAP(bdims2[MO_DIM], bdims2[N + 0]);

	assert(md_check_compat(N, ~0UL, adims2, bdims2));
	assert(md_check_compat(N, ~0UL, cdims2, bdims2));

	long max_dims[N + 1];
	md_max_dims(N + 1, ~0UL, max_dims, adims2, bdims2);
	md_max_dims(N + 1, ~0UL, max_dims, max_dims, cdims2);

	(add ? md_zfmac2 : md_ztenmul2)(N + 1, max_dims, cstrs, C, astrs, A, bstrs, B);
}


static void simu_chain_jac_fun(const struct nlop_seq_data_s* data, int N,
	const long modims[N], complex float* omag,
	int S, const long sdims[S][N], complex float* signal[S],
	const long midims[N], const complex float* imag,
	const long pdims[N], const complex float* pars,
	const long dmmdims[N], complex float* dmm,
	const long dmpdims[N], complex float* dmp,
	const long dsmdims[S][N], complex float* dsm[S],
	const long dspdims[S][N], complex float* dsp[S])
{
	auto d = CAST_DOWN(simu_chain_s, data);

	complex float* tdmm = dmm ?: md_alloc_sameplace(N, dmmdims, CFL_SIZE, omag);
	complex float* tdmp = dmp ?: md_alloc_sameplace(N, dmpdims, CFL_SIZE, omag);
	complex float* tmag = md_alloc_sameplace(N, midims, CFL_SIZE, omag);

	md_transpose(N, data->sim.MI_DIM, data->sim.MO_DIM, modims, omag, midims, imag, CFL_SIZE);

	init_id_matrix2(data->sim.MO_DIM, data->sim.MI_DIM, N, dmmdims, MD_STRIDES(N, dmmdims, CFL_SIZE), tdmm);
	md_clear(N, dmpdims, tdmp, CFL_SIZE);

	complex float* ttdmm = md_alloc_sameplace(N, dmmdims, CFL_SIZE, omag);
	complex float* ttdmp = md_alloc_sameplace(N, dmpdims, CFL_SIZE, omag);



	for (int j = 0; j < list_count(d->nlops); j++) {

		const struct nlop_s* op = list_get_item(d->nlops, j);

		int ST = nlop_get_nr_out_args(op) - 1;

		md_transpose(N, data->sim.MO_DIM, data->sim.MI_DIM, midims, tmag, modims, omag, CFL_SIZE);

		complex float* tdsm[ST?:1];
		for (int s = 0; s < ST; s++)
			tdsm[s] = dsm[s] ?: md_alloc_sameplace(N, dsmdims[s], CFL_SIZE, omag);

		nlop_seq_apply(op, N, modims, omag, ST, sdims, signal, midims, tmag, pdims, pars,
			dmmdims, ttdmm, dmpdims, ttdmp,
			dsmdims, tdsm, dspdims, dsp);

		for (int s = 0; s < ST; s++) {

			if (NULL != dsp[s])
				simu_jac_mul(N, data->sim.MO_DIM, data->sim.MI_DIM, dspdims[s], dsp[s], dsmdims[s], tdsm[s], dmpdims, tdmp, true);

			if (NULL != dsm[s])
				simu_jac_mul(N, data->sim.MO_DIM, data->sim.MI_DIM, dsmdims[s], dsm[s], dsmdims[s], dsm[s], dmmdims, tdmm, false);
		}

		if (NULL != dmm)
			simu_jac_mul(N, data->sim.MO_DIM, data->sim.MI_DIM, dmmdims, dmm, dmmdims, ttdmm, dmmdims, dmm, false);

		if (NULL != dmp) {

			simu_jac_mul(N, data->sim.MO_DIM, data->sim.MI_DIM, dmpdims, dmp, dmmdims, ttdmm, dmpdims, dmp, false);
			md_zadd(N, dmpdims, dmp, dmp, ttdmp);
		}

		for (int s = 0; s < ST; s++)
			if (tdsm[s] != dsm[s])
				md_free(tdsm[s]);


		sdims += ST;
		dsmdims += ST;
		dspdims += ST;

		signal += ST;
		dsm += ST;
		dsp += ST;
	}

	md_free(tmag);

	md_free(ttdmm);
	md_free(ttdmp);

	if (tdmm != dmm)
		md_free(tdmm);

	if (tdmp != dmp)
		md_free(tdmp);
}

static void simu_chain_free(const nlop_seq_data_t* data)
{
	auto d = CAST_DOWN(simu_chain_s, data);

	while (0 != list_count(d->nlops))
		nlop_free(list_pop(d->nlops));

	list_free(d->nlops);

	free(d);
}

const struct nlop_s* nlop_simu_jacobian_chain_create(struct sim_config_s sim, list_t nlops)
{
	PTR_ALLOC(struct simu_chain_s, data);
	SET_TYPEID(simu_chain_s, data);

	int S = 0;

	for (int i = 0; i < list_count(nlops); i++) {

		const struct nlop_s* nlop = list_get_item(nlops, i);
		S += nlop_get_nr_out_args(nlop) - 1;
	}

	data->nlops = nlops;

	long sdims[S ?: 1][sim.N];
	int s = 0;

	for (int i = 0; i < list_count(nlops); i++) {

		const struct nlop_s* nlop = list_get_item(nlops, i);
		int ST = nlop_get_nr_out_args(nlop) - 1;
		for (int ss = 0; ss < ST; ss++)
			md_copy_dims(sim.N, sdims[s++], nlop_generic_codomain(nlop, ss + 1)->dims);
	}

	return nlop_sig_seq_create(&sim, CAST_UP(PTR_PASS(data)), simu_chain_jac_fun, simu_chain_free, S, sim.N, sdims);
}


struct simu_stack_signal_s {

	INTERFACE(nlop_seq_data_t);

	int stackidx;
	const struct nlop_s* nlop;
};

DEF_TYPEID(simu_stack_signal_s);

static void simu_stack_fun(const struct nlop_seq_data_s* data, int N,
	const long modims[N], complex float* omag,
	int S, const long sdims[S][N], complex float* signal[S],
	const long midims[N], const complex float* imag,
	const long pdims[N], const complex float* pars,
	const long dmmdims[N], complex float* dmm,
	const long dmpdims[N], complex float* dmp,
	const long dsmdims[S][N], complex float* dsm[S],
	const long dspdims[S][N], complex float* dsp[S])
{
	(void)modims; (void)midims; (void)pdims; (void)dmmdims; (void)dmpdims;

	auto d = CAST_DOWN(simu_stack_signal_s, data);

	int OO = nlop_get_nr_out_args(d->nlop);
	int II = nlop_get_nr_in_args(d->nlop);

	assert(1 < OO);
	assert(2 == II);

	long odims[OO][N];
	long idims[II][N];
	long ddims[OO][II][N];

	nlop_zblock_diag_get_dims(d->nlop, N, OO, odims, II, idims, ddims);

	complex float* arr_dst[OO];
	complex float* arr_jac[OO][II];
	const complex float* arr_src[II];

	arr_src[0] = imag;
	arr_src[1] = pars;

	arr_dst[0] = omag;

	arr_jac[0][0] = dmm;
	arr_jac[0][1] = dmp;


	for (int o = 1; o < OO; o++) {

		//FIXME: depending on strides, we could directly write to dst

		arr_dst[o] = md_alloc_sameplace(N, odims[o], CFL_SIZE, omag);
		arr_jac[o][0] = (NULL == dsm[0]) ? NULL : md_alloc_sameplace(N, ddims[o][0], CFL_SIZE, omag);
		arr_jac[o][1] = (NULL == dsp[0]) ? NULL : md_alloc_sameplace(N, ddims[o][1], CFL_SIZE, omag);
	}

	nlop_zblock_diag_apply(d->nlop, N, OO, odims, arr_dst, II, idims, arr_src, ddims, arr_jac);

	long pos[N];
	md_set_dims(N, pos, 0);

	for(int o = 1; o < OO; o++) {

		md_copy_block(N, pos, sdims[0], signal[0], odims[o], arr_dst[o], CFL_SIZE);
		md_free(arr_dst[o]);

		if (NULL != dsm[0]) {

			md_copy_block(N, pos, dsmdims[0], dsm[0], ddims[o][0], arr_jac[o][0], CFL_SIZE);
			md_free(arr_jac[o][0]);
		}

		if (NULL != dsp[0]) {

			md_copy_block(N, pos, dspdims[0], dsp[0], ddims[o][1], arr_jac[o][1], CFL_SIZE);
			md_free(arr_jac[o][1]);
		}

		pos[d->stackidx] += odims[o][d->stackidx];
	}
}

static void simu_stack_free(const nlop_seq_data_t* data)
{
	auto d = CAST_DOWN(simu_stack_signal_s, data);

	nlop_free(d->nlop);

	free(d);
}

const struct nlop_s* nlop_simu_stack_create(struct sim_config_s sim, const struct nlop_s* nlop, int stack_dim)
{
	PTR_ALLOC(struct simu_stack_signal_s, data);
	SET_TYPEID(simu_stack_signal_s, data);

	data->nlop = nlop;
	data->stackidx = stack_dim;

	int OO = nlop_get_nr_out_args(nlop);
	int N = nlop_generic_codomain(nlop, 0)->N;

	assert(1 < OO);

	long sdims[1][N];
	md_copy_dims(N, sdims[0], nlop_generic_codomain(nlop, 1)->dims);
	for (int o = 2; o < OO; o++)
		sdims[0][stack_dim] += nlop_generic_codomain(nlop, o)->dims[stack_dim];

	return nlop_sig_seq_create(&sim, CAST_UP(PTR_PASS(data)), simu_stack_fun, simu_stack_free, 1, sim.N, sdims);
}



struct stm_s {

	INTERFACE(nlop_data_t);

	struct sim_config_s sim;

	const struct nlop_s* nlop;
	complex float* pars;

	int S;
	complex float** mag;
	complex float** jac;

	struct shared_obj_s sptr;
};

DEF_TYPEID(stm_s);

static void stm_clear(struct stm_s* x)
{
	md_free(x->pars);
	x->pars = NULL;

	for (int i = 0; i < x->S + 1; i++) {

		md_free(x->mag[i]);
		md_free(x->jac[i]);

		x->mag[i] = NULL;
		x->jac[i] = NULL;
	}

}

static void stm_update(struct stm_s* x, int N, const long pdims[N], const complex float* pars)
{
	if (NULL != x->pars && 0. != md_zrmse(N, pdims, x->pars, pars))
		stm_clear(x);

	if (NULL != x->pars)
		return;

	x->pars = md_alloc_sameplace(N, pdims, CFL_SIZE, pars);
	md_copy(N, pdims, x->pars, pars, CFL_SIZE);

	assert(NULL == x->mag[x->S]);
	assert(NULL == x->jac[x->S]);

	long mdims[N];
	md_copy_dims(N, mdims, nlop_generic_domain(x->nlop, 0)->dims);

	complex float* mag = md_alloc_sameplace(N, mdims, CFL_SIZE, pars);
	md_clear(N, mdims, mag, CFL_SIZE);

	long odims[1][N];
	long idims[2][N];
	long ddims[1][2][N];

	nlop_zblock_diag_get_dims(x->nlop, N, 1, odims, 2, idims, ddims);

	complex float* dst[1];
	const complex float* src[2];
	complex float* jac[1][2];

	x->mag[x->S] = md_alloc_sameplace(N, mdims, CFL_SIZE, pars);
	x->jac[x->S] = md_alloc_sameplace(N, ddims[0][1], CFL_SIZE, pars);

	dst[0] = x->mag[x->S];

	src[0] = mag;
	src[1] = pars;

	jac[0][0] = NULL;
	jac[0][1] = x->jac[x->S];

	nlop_zblock_diag_apply(x->nlop, N, 1, odims, dst, 2, idims, src, ddims, jac);

	for (int i = 0; i < x->S; i++) {

		x->mag[i] = md_alloc_sameplace(N, mdims, CFL_SIZE, pars);
		x->jac[i] = md_alloc_sameplace(N, ddims[0][1], CFL_SIZE, pars);

		complex float vec[x->S];
		for (int j = 0; j < x->S; j++)
			vec[j] = (i == j) ? 1. : 0.;

		long vdims[N];
		md_select_dims(N, MD_BIT(x->sim.MI_DIM), vdims, idims[0]);
		md_copy2(N, idims[0], MD_STRIDES(N, idims[0], CFL_SIZE), mag, MD_STRIDES(N, vdims, CFL_SIZE), vec, CFL_SIZE);

		dst[0] = x->mag[i];
		jac[0][1] = x->jac[i];

		nlop_zblock_diag_apply(x->nlop, N, 1, odims, dst, 2, idims, src, ddims, jac);

		md_zsub(N, odims[0], x->mag[i], x->mag[i], x->mag[x->S]);
		md_zsub(N, ddims[0][1], x->jac[i], x->jac[i], x->jac[x->S]);
	}

	md_free(mag);
}

static void stm_del(const struct shared_obj_s* sptr)
{
	struct stm_s* x = CONTAINER_OF(sptr, struct stm_s, sptr);

	stm_clear(x);

	nlop_free(x->nlop);
	xfree(x->mag);
	xfree(x->jac);

	xfree(x);
}

void stm_free(struct stm_s* x)
{
	shared_obj_destroy(&x->sptr);
}

struct stm_s* stm_create(struct sim_config_s sim, const struct nlop_s* nlop)
{
	PTR_ALLOC(struct stm_s, x);
	SET_TYPEID(stm_s, x);

	x->sim = sim;
	x->nlop = nlop_clone(nlop);
	x->pars = NULL;

	x->S = nlop_generic_codomain(nlop, 0)->dims[x->sim.MO_DIM];

	x->mag = *TYPE_ALLOC(complex float*[x->S + 1]);
	x->jac = *TYPE_ALLOC(complex float*[x->S + 1]);

	for (int i = 0; i < x->S + 1; i++) {

		x->mag[i] = NULL;
		x->jac[i] = NULL;
	}

	shared_obj_init(&x->sptr, stm_del);

	return PTR_PASS(x);
}


static void stm_nlop_free(const nlop_data_t* x)
{
	stm_free(CAST_DOWN(stm_s, x));
}

static void stm_fun(const nlop_data_t* data, int N, int OO, const long odims[OO][N], complex float* arr_dst[OO], int II, const long idims[II][N], const complex float* arr_src[II], const long ddims[OO][II][N], complex float* jac[OO][II])
{
	auto d = CAST_DOWN(stm_s, data);

	seq_check_dims(&d->sim, N, OO, odims, II, idims, ddims);

	complex float* dst = arr_dst[0];
	const complex float* src = arr_src[0];
	const complex float* par = arr_src[1];

	stm_update(d, N, idims[1], par);

	long pos[N];
	md_set_dims(N, pos, 0);

	md_copy(N, odims[0], dst, d->mag[d->S], CFL_SIZE);
	if (NULL != jac[0][1])
		md_copy(N, ddims[0][1], jac[0][1], d->jac[d->S], CFL_SIZE);

	long sdims[N];
	md_select_dims(N, ~MD_BIT(d->sim.MI_DIM), sdims, idims[0]);

	for (pos[d->sim.MI_DIM] = 0; pos[d->sim.MI_DIM] < idims[0][d->sim.MI_DIM]; pos[d->sim.MI_DIM]++) {

		complex float* slice = md_alloc_sameplace(N, sdims, CFL_SIZE, src);
		md_copy_block(N, pos, sdims, slice, idims[0], src, CFL_SIZE);

		md_zfmac2(N, odims[0], MD_STRIDES(N, odims[0], CFL_SIZE), dst, MD_STRIDES(N, odims[0], CFL_SIZE), d->mag[pos[d->sim.MI_DIM]], MD_STRIDES(N, sdims, CFL_SIZE), slice);

		if (NULL != jac[0][1])
			md_zfmac2(N, ddims[0][1], MD_STRIDES(N, ddims[0][1], CFL_SIZE), jac[0][1], MD_STRIDES(N, ddims[0][1], CFL_SIZE), d->jac[pos[d->sim.MI_DIM]], MD_STRIDES(N, sdims, CFL_SIZE), slice);

		md_free(slice);

		if (NULL != jac[0][0])
			md_copy_block(N, pos, ddims[0][0], jac[0][0], odims[0], d->mag[pos[d->sim.MI_DIM]], CFL_SIZE);
	}
}

struct nlop_s* nlop_stm_create(struct stm_s* x)
{
	shared_obj_ref(&(x->sptr));

	int N = nlop_generic_codomain(x->nlop, 0)->N;

	int OO = 1;
	int II = 2;

	long odims[OO][N];
	long idims[II][N];
	long ddims[OO][II][N];

	nlop_zblock_diag_get_dims(x->nlop, N, OO, odims, II, idims, ddims);

	unsigned long diag_flags[OO][II];
	diag_flags[0][0] = ~md_nontriv_dims(N, ddims[0][0]);
	diag_flags[0][1] = ~md_nontriv_dims(N, ddims[0][1]);


	return nlop_zblock_diag_generic_create(CAST_UP(x), N, OO, odims, II, idims, diag_flags, stm_fun, stm_nlop_free);
}



//in: mag_in, par
//out: mag_out, mag_read
const struct nlop_s* nlop_seq_from_blocks_jac_create_F(struct sim_config_s sim, struct list_s* nlops)
{
	const struct nlop_s* ret = nlop_simu_jacobian_chain_create(sim, nlops);

#if 1
	ret = nlop_simu_stack_create(sim, ret, TE_DIM);
#else
	ret = nlop_stack_outputs_generic_F(ret, OO - 1, index, TE_DIM);
#endif

	const struct linop_s* lop = linop_repmat_create(sim.N, sim.pdims, sim.spatial_flags);
	ret = nlop_prepend_FF(nlop_from_linop_F(lop), ret, 1);

	return ret;
}
#if 0

//in: mag_in, par
//out: mag_out, mag_read
const struct nlop_s* nlop_seq_from_blocks_jac_create_F(struct sim_config_s sim, struct list_s* nlops)
{
	long OO = 0;
	bool ordered = true;

	for (int i = 0; i < list_count(nlops); i++) {

		const struct nlop_s* nlop = list_get_item(nlops, i);
		assert(nlop_is_zblock_diag(nlop));

		struct adc_s* d = CAST_MAYBE(adc_s, nlop_zblock_diag_get_data(nlop));

		if (NULL != d) {

			OO++;
			ordered = ordered && (0 <= d->index);
		}
	}

	int index[OO];
	for (int i = 0; i < OO; i++)
		index[i] = i + 1;

	debug_printf(DP_WARN, "%d\n", ordered);

	for (int i = 0, j = 1; ordered && i < list_count(nlops); i++) {

		const struct nlop_s* nlop = list_get_item(nlops, i);
		assert(nlop_is_zblock_diag(nlop));

		struct adc_s* d = CAST_MAYBE(adc_s, nlop_zblock_diag_get_data(nlop));

		if (NULL != d)
			index[d->index] = j++;
	}

	const struct nlop_s* ret = nlop_simu_jacobian_chain_create(sim, nlops);

#if 1
	ret = nlop_simu_stack_create(ret, OO, index, TE_DIM);
#else
	ret = nlop_stack_outputs_generic_F(ret, OO - 1, index, TE_DIM);
#endif

	const struct linop_s* lop = linop_repmat_create(sim.N, sim.pdims, sim.spatial_flags);
	ret = nlop_prepend_FF(nlop_from_linop_F(lop), ret, 1);

	return ret;
}

#endif

//in: mag_in, par
//out: mag_out, mag_read
const struct nlop_s* nlop_seq_from_blocks_create_F(struct sim_config_s sim, struct list_s* nlops)
{
	const struct nlop_s* ret = list_pop(nlops);

	auto cod = nlop_generic_codomain(ret, 0);

	int N = cod->N;
	long mdims[N];
	long midims[N];
	md_copy_dims(N, mdims, cod->dims);
	md_transpose_dims(N, sim.MI_DIM, sim.MO_DIM, midims, mdims);

	ret = nlop_reshape_in_F(ret, 0, N, midims);

	while (0 < list_count(nlops)) {

		const struct nlop_s* nlop = list_pop(nlops);

		int OO = nlop_get_nr_out_args(nlop);
		nlop = nlop_reshape_in_F(nlop, 0, N, mdims);

		ret = nlop_chain2_swap_FF(ret, 0, nlop, 0);
		for (int i = 1; i < OO; i++)
			ret = nlop_shift_output_F(ret, nlop_get_nr_out_args(ret) - 1, 1);

		if (2 < nlop_get_nr_in_args(ret))
			ret = nlop_dup_F(ret, 1, 2);
	}

	list_free(nlops);

	int OO = nlop_get_nr_out_args(ret);

	int index[OO - 1];
	for (int i = 0; i < OO - 1; i++)
		index[i] = i + 1;

	ret = nlop_stack_outputs_generic_F(ret, OO - 1, index, TE_DIM);

	return ret;
}


const struct nlop_s* sim_nlop_set_init(struct sim_config_s sim, const struct nlop_s* nlop)
{
	complex float init[3] = { 0., 0., 1 };

	auto dom = nlop_generic_domain(nlop, 0);

	long idims[dom->N];
	md_select_dims(dom->N, MD_BIT(sim.MO_DIM) | MD_BIT(sim.MI_DIM), idims, dom->dims);

	return nlop_set_input_const_F2(nlop, 0, dom->N, dom->dims, MD_STRIDES(dom->N, idims, CFL_SIZE), true, init);
}

