/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2025 Moritz Blumenthal
 */

#include <assert.h>
#include <complex.h>
#include <math.h>

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "num/lineseg.h"
#include "num/multind.h"
#include "num/morph.h"
#include "num/flpmath.h"
#include "num/loop.h"
#include "num/rand.h"
#include "num/vec3.h"
#include "num/ode.h"

#ifdef USE_GPU
#include "num/gpuops.h"
#endif

#include "pole.h"

struct pole_config_s pole_config_default = {

	.diameter = 0.05,
	.closing = -1.,
	.thresh = 0.5,
	.segments = 10,
	.avg_flag = COIL_FLAG,
	.normal = -1,
	.espirit = false,
	.tol = 1.e-4,
	.osx = 1.,
};

static void get_circle_coords(struct pole_config_s* conf, bart_dim_t pos[3], int index, int normal, int diameter, bool twoD)
{
	float angle = 2. * M_PI * index / conf->segments;

	for (int i = 0; i < 3; i++)
		pos[i] = 0;

	float e1[3][3] = { { 0., 1., 0. }, { 0., 0., 1. }, { 1., 0., 0. } };
	float e2[3][3] = { { 0., 0., 1. }, { 1., 0., 0. }, { 0., 1., 0. } };

	float fpos[3] = { diameter / 2., diameter / 2., diameter / 2. };

	vec3_saxpy(fpos, fpos, 0.5 * diameter * cosf(angle), e1[normal]);
	vec3_saxpy(fpos, fpos, 0.5 * diameter * sinf(angle), e2[normal]);

	for (int i = 0; i < 3; i++)
		pos[i] = MAX(0, MIN(diameter, llroundf(fpos[i])));

	if (twoD)
		pos[normal] = 0;
}

static int get_diameter(struct pole_config_s conf, int normal, const long dims[3], bool print)
{
	if (conf.espirit) {

		debug_printf(DP_DEBUG1, "Circle diameter set to 1pixel (ESPIRiT mode).\n");
		return 1;
	}

	long tdims[3];
	for (int i = 0; i < 3; i++)
		tdims[i] = dims[i];

	tdims[0] = lroundf(tdims[0] / conf.osx);

	if (print && (tdims[(normal + 1) % 3] != tdims[(normal + 2) % 3]))
		debug_printf(DP_DEBUG1, "Non-square dimensions detected (%ld, %ld, %ld): ", tdims[0], tdims[1], tdims[2]);

	int diameter = roundf(ceil(conf.diameter * MAX(tdims[(normal + 1) % 3], tdims[(normal + 2) % 3])));

	if (print)
		debug_printf(DP_DEBUG1, "Circle diameter set to %d (%.3f * %ld).\n", diameter, conf.diameter,  MAX(tdims[(normal + 1) % 3], tdims[(normal + 2) % 3]));

	return diameter;
}

static void compute_curl_map_normal(struct pole_config_s conf, int N, const bart_dim_t dims[N], complex float* curl_map, const complex float* sens, int normal)
{
	assert(2 <= bitcount(md_nontriv_dims(MIN(N, 3), dims)));
	assert(0 < conf.diameter);
	assert(0 <= normal && normal < 3);

	bart_dim_t pos1[N];
	bart_dim_t pos2[N];

	md_set_dims(N, pos1, 0);
	md_set_dims(N, pos2, 0);

	int diameter = get_diameter(conf, normal, dims, true);

	bool twoD = false;

	bart_dim_t odims[N];
	md_copy_dims(N, odims, dims);

	for (int i = 0; i < 3; i++) {

		if (1 != dims[i]) {

			odims[i] -= diameter;

		} else {

			assert(i == normal);
			twoD = true;
		}

		assert(0 < odims[i]);
	}

	complex float* tmp_angle = md_alloc_sameplace(N, odims, CFL_SIZE, sens);
	complex float* angle = md_alloc_sameplace(N, odims, CFL_SIZE, sens);
	md_clear(N, odims, angle, CFL_SIZE);

	complex float* prod = md_alloc_sameplace(N, odims, CFL_SIZE, sens);
	md_zfill(N, odims, prod, 1.);

	complex float* sens1 = md_alloc_sameplace(N, odims, CFL_SIZE, sens);
	complex float* sens2 = md_alloc_sameplace(N, odims, CFL_SIZE, sens);

	get_circle_coords(&conf, pos1, 0, normal, diameter, twoD);
	md_copy_block(N, pos1, odims, sens1, dims, sens, CFL_SIZE);

	for (int i = 0; i < conf.segments; i++) {

		get_circle_coords(&conf, pos1, i + 0, normal, diameter, twoD);
		get_circle_coords(&conf, pos2, i + 1, normal, diameter, twoD);

		if (md_check_equal_dims(N, pos1, pos2, MD_BIT(3) - 1))
			continue;

		//prod stays 1, where sens is not zero as x / 0. is set to 0 in md_zdiv
		md_zmul(N, odims, prod, prod, sens1);
		md_zdiv(N, odims, prod, prod, sens1);

		md_copy_block(N, pos2, odims, sens2, dims, sens, CFL_SIZE);

		md_zmulc(N, odims, tmp_angle, sens1, sens2);
		md_zphsr(N, odims, tmp_angle, tmp_angle); // to handle 0.+0.i case
		md_zarg(N, odims, tmp_angle, tmp_angle);
		md_zadd(N, odims, angle, angle, tmp_angle);

		SWAP(sens1, sens2);
	}

	md_free(sens1);
	md_free(sens2);

	md_free(tmp_angle);
	md_zsmul(N, odims, angle, angle, -1. / (2. * M_PI));

	md_zmul(N, odims, angle, angle, prod);
	md_free(prod);

	md_resize_center(N, dims, curl_map, odims, angle, CFL_SIZE);
	md_free(angle);
}


void compute_curl_map(struct pole_config_s conf, int N, const bart_dim_t curl_dims[N], int dim, complex float* curl_map, const bart_dim_t sens_dims[N], const complex float* sens)
{
	assert(dim < N);
	assert(md_check_compat(N, MD_BIT(dim), curl_dims, sens_dims));

	if (1 == curl_dims[dim]) {

		int normal = conf.normal;

		for (int i = 0; i < 3; i++)
			if (1 == sens_dims[i])
				normal = i;

		assert(-1 != normal && normal < 3);

		compute_curl_map_normal(conf, N, sens_dims, curl_map, sens, normal);

	} else {

		for (int i = 0; i < curl_dims[dim]; i++)
			compute_curl_map_normal(conf, N, sens_dims, curl_map + md_calc_size(N, sens_dims) * i, sens, i);
	}
}

void compute_curl_weighting(struct pole_config_s conf, int N, const bart_dim_t curl_dims[N], int dim, complex float* wgh_map, const bart_dim_t col_dims[N], const complex float* sens)
{
	assert(dim < N);
	assert(md_check_compat(N, MD_BIT(dim), curl_dims, col_dims));

	complex float* wgh = md_alloc_sameplace(N, col_dims, CFL_SIZE, wgh_map);

	md_clear(N, col_dims, wgh, CFL_SIZE);
	md_zss(N, col_dims, 0, wgh, sens);

	bart_dim_t rdims[N];
	md_select_dims(N, ~conf.avg_flag, rdims, col_dims);

	complex float* tmp = md_alloc_sameplace(N, rdims, CFL_SIZE, wgh_map);
	md_zss(N, col_dims, conf.avg_flag, tmp, sens);

	complex float* one = md_alloc_sameplace(N, rdims, CFL_SIZE, wgh_map);
	md_zfill(N, rdims, one, 1.);

	md_zdiv(N, rdims, tmp, one, tmp);
	md_free(one);

	md_zmul2(N, col_dims, MD_STRIDES(N, col_dims, CFL_SIZE), wgh, MD_STRIDES(N, col_dims, CFL_SIZE), wgh, MD_STRIDES(N, rdims, CFL_SIZE), tmp);
	md_free(tmp);

	md_copy2(N, curl_dims, MD_STRIDES(N, curl_dims, CFL_SIZE), wgh_map, MD_STRIDES(N, col_dims, CFL_SIZE), wgh, CFL_SIZE);
	md_free(wgh);
}


void average_curl_map(int N, const bart_dim_t pmap_dims[N], complex float* red_curl_map, const bart_dim_t curl_dims[N], int dim, complex float* curl_map, complex float* wgh_map)
{
	bart_dim_t tmp_dims[N];
	md_select_dims(N, MD_BIT(dim) | md_nontriv_dims(N, pmap_dims), tmp_dims, curl_dims);

	complex float* tmp = md_alloc_sameplace(N, tmp_dims, CFL_SIZE, curl_map);

	if (NULL != wgh_map)
		md_ztenmul(N, tmp_dims, tmp, curl_dims, curl_map, curl_dims, wgh_map);
	else
		md_zavg(N, curl_dims, ~md_nontriv_dims(N, tmp_dims), tmp, curl_map);

	if (1 < curl_dims[dim]) {

		md_zabs(N, tmp_dims, tmp, tmp);
		md_reduce_zmax(N, tmp_dims, MD_BIT(dim), red_curl_map, tmp);

	} else {

		md_copy(N, tmp_dims, red_curl_map, tmp, CFL_SIZE);
	}

	md_free(tmp);
}


static struct lseg_s extract_phase_poles_2d_sign(struct pole_config_s conf, int N, const bart_dim_t dims[N], const complex float* curl_map, bool pos)
{
	assert((3 == bitcount(md_nontriv_dims(3, dims))) || (2 == bitcount(md_nontriv_dims(3, dims))));
	assert(1 == md_calc_size(N - 3, dims + 3));

	int normal = conf.normal;

	for (int i = 0; i < 3; i++)
		if (1 == dims[i])
			normal = i;

	assert((0 <= normal) && (3 > normal));

	complex float* binary = md_alloc_sameplace(3, dims, CFL_SIZE, curl_map);

	if (pos)
		md_zsgreatequal(3, dims, binary, curl_map, conf.thresh);
	else
		md_zslessequal(3, dims, binary, curl_map, -conf.thresh);


	struct lseg_s ret = { .N = 0, .pos = NULL };

	if (0. == md_znorm(3, dims, binary)) {

		md_free(binary);

		return ret;
	}

	if (conf.closing != 0. && !conf.espirit) {

		int dmin = llroundf(ceilf(((-1 == conf.closing) ? conf.diameter / 2. : conf.closing) * MAX(dims[(normal + 1) % 3], dims[(normal + 2) % 3])));
		bart_dim_t mdims[3];
		complex float* mask = md_structuring_element_cube(3, mdims, dmin, md_nontriv_dims(3, dims), curl_map);

		md_closing(3, mdims, mask, dims, binary, binary, CONV_TRUNCATED);
		md_free(mask);
	}

	bart_dim_t sdims[3];
	complex float* strc = md_structuring_element_cube(3, sdims, 1, md_nontriv_dims(3, dims), curl_map);

	complex float* labels = md_alloc_sameplace(3, dims, CFL_SIZE, curl_map);

	bart_dim_t nlabel = md_label(3, dims, labels, binary, sdims, strc);

	md_free(binary);
	md_free(strc);

	float com[nlabel][3];
	md_center_of_mass(nlabel, 3, com, dims, labels, NULL);
	md_free(labels);

	ret.N = 0;
	ret.pos = xmalloc(sizeof(vec3_t[nlabel][2]));

	int diameter = conf.espirit ? 1 : roundf(ceil(conf.diameter * MAX(dims[(normal + 1) % 3], dims[(normal + 2) % 3])));
	float offset = diameter / 2 - diameter / 2.;

	for (int i = 0; i < nlabel; i++) {

		for (int j = 0; j < 3; j++)
			com[i][j] = (com[i][j] + ((1 != dims[j]) ? offset : 0.) - dims[j] / 2) / (float)dims[j];

		debug_printf(DP_DEBUG1, "Found%s pole at %f %f %f.\n", pos ? "" : " conjugate", com[i][0], com[i][1], com[i][2]);

		vec3_copy(ret.pos[ret.N][0], com[i]);
		vec3_copy(ret.pos[ret.N][1], com[i]);

		ret.pos[ret.N][0][normal] = pos ? -1. : 1.;
		ret.pos[ret.N][1][normal] = pos ? 1. : -1.;

		ret.N++;
	}

	return ret;
}


struct lseg_s extract_phase_poles_2D(struct pole_config_s conf, int N, const bart_dim_t dims[N], const complex float* curl_map)
{
	struct lseg_s pos = extract_phase_poles_2d_sign(conf, N, dims, curl_map, true);
	struct lseg_s neg = extract_phase_poles_2d_sign(conf, N, dims, curl_map, false);

	struct lseg_s ret = { .N = pos.N + neg.N, .pos = NULL };

	if (0 == ret.N)
		return ret;

	ret.pos = xmalloc(sizeof(vec3_t[ret.N][2]));

	for (int i = 0; i < pos.N; i++) {

		vec3_copy(ret.pos[i][0], pos.pos[i][0]);
		vec3_copy(ret.pos[i][1], pos.pos[i][1]);
	}

	for (int i = 0; i < neg.N; i++) {

		vec3_copy(ret.pos[i + pos.N][0], neg.pos[i][0]);
		vec3_copy(ret.pos[i + pos.N][1], neg.pos[i][1]);
	}

	xfree(pos.pos);
	xfree(neg.pos);

	return ret;
}


static void get_coord_transform(vec3_t evec[3], const vec3_t r1, const vec3_t r2)
{
	vec3_copy(evec[2], r2);
	vec3_sub(evec[2], evec[2], r1);
	vec3_smul(evec[2], evec[2],  1. / vec3_norm(evec[2]));

	evec[0][0] = 1.;
	evec[0][1] = 0;
	evec[0][2] = 0.;

	if (0.8 < fabsf(vec3_sdot(evec[0], evec[2]))) {

		evec[0][0] = 0.;
		evec[0][1] = 1.;
		evec[0][2] = 0.;
	}

	vec3_saxpy(evec[0], evec[0], -vec3_sdot(evec[0], evec[2]), evec[2]);
	vec3_smul(evec[0], evec[0], 1. / vec3_norm(evec[0]));

	vec3_rot(evec[1], evec[2], evec[0]);
}



void sample_phase_pole_2D(int N, const bart_dim_t dims[N], complex float* dst, int D, const float r[D][2][3])
{
	assert(2 == bitcount(md_nontriv_dims(3, dims)));

	vec3_t evec[D][3];
	vec3_t center[D];
	float L[D];

	for (int i = 0; i < D; i++) {

		vec3_t r1;
		vec3_t r2;

		vec3_copy(r1, r[i][0]);
		vec3_copy(r2, r[i][1]);

		vec3_t diff;
		vec3_sub(diff, r2, r1);

		if (0. == vec3_norm(diff)) {

			for (int j = 0; j < 3; j++) {

				if (1 == dims[j]) {

					r1[j] = 0.5;
					r2[j] = -0.5;
				}
			}

			vec3_sub(diff, r2, r1);
			assert(0. != vec3_norm(diff));
		}

		get_coord_transform(evec[i], r1, r2);

		L[i] = vec3_norm(diff) / 2.;

		vec3_add(center[i], r1, r2);
		vec3_smul(center[i], center[i], 0.5);

		for (int j = 0; j < 3; j++)
			center[i][j] += (dims[j] / 2) / (float)dims[j];
	}

	const bart_dim_t* dimsp = dims;
	float* centerp0 = &center[0][0];
	float* evecp0 = &evec[0][0][0];
	float* Lp = L;


	NESTED(complex float, pole_kernel, (const bart_dim_t pos[]))
	{
		complex float ret = 1.;

		float (*centerp)[D][3] = (float (*)[D][3])centerp0;
		float (*evecp)[D][3][3] = (float (*)[D][3][3])evecp0;

		for (int i = 0; i < D; i++) {

			vec3_t fpos;

			for (int j = 0; j < 3; j++)
				fpos[j] = pos[j] / (float)dimsp[j] - (*centerp)[i][j];

			if (Lp[i] < fabsf(vec3_sdot((*evecp)[i][2], fpos)))
				continue;

			complex float val = vec3_sdot(fpos, (*evecp)[i][0]) + vec3_sdot(fpos, (*evecp)[i][1]) * 1.i;
			complex float mag = cabsf(val);

			val = (0. == mag) ? 1. : val / mag;
			ret *= val;
		}

		return ret;
	};

	md_parallel_zsample(N, dims, dst, pole_kernel);
}


bool phase_pole_correction(struct pole_config_s conf, int N, const bart_dim_t pmap_dims[N], complex float* phase, const bart_dim_t sens_dims[N], const complex float* sens)
{
	bart_dim_t curl_dims[N];
	md_copy_dims(N, curl_dims, sens_dims);

	int normal = conf.normal;

	for (int i = 0; i < 3; i++)
		if (1 == curl_dims[i])
			normal = i;

	assert((-1 <= normal) && (3 > normal));

	if (-1 == normal)
		curl_dims[ITER_DIM] = 3;

	complex float* curl_map = md_alloc_sameplace(N, curl_dims, CFL_SIZE, sens);
	complex float* curl_wgh = md_alloc_sameplace(N, curl_dims, CFL_SIZE, sens);

	assert(ITER_DIM < N);

	compute_curl_map(conf, N, curl_dims, ITER_DIM, curl_map, sens_dims, sens);
	compute_curl_weighting(conf, N, curl_dims, ITER_DIM, curl_wgh, sens_dims, sens);

	bart_dim_t rcurl_map_dims[N];
	md_select_dims(N, ~(conf.avg_flag | MD_BIT(ITER_DIM)), rcurl_map_dims, curl_dims);

	complex float* red_curl_map = md_alloc_sameplace(N, rcurl_map_dims, CFL_SIZE, sens);
	average_curl_map(N, rcurl_map_dims, red_curl_map, curl_dims, ITER_DIM, curl_map, curl_wgh);

	struct lseg_s pos;
	if (3 == curl_dims[ITER_DIM])
		pos = extract_phase_poles_3D(conf, N, rcurl_map_dims, red_curl_map, sens_dims, sens);
	else
		pos = extract_phase_poles_2D(conf, N, rcurl_map_dims, red_curl_map);

	md_zfill(N, pmap_dims, phase, 1.);

	if (0 < pos.N) {

		if (3 == curl_dims[ITER_DIM])
			sample_phase_pole_3D(N, pmap_dims, phase, pos.N, pos.pos, conf.tol);
		else
			sample_phase_pole_2D(N, pmap_dims, phase, pos.N, pos.pos);
	}

	xfree(pos.pos);

	md_free(red_curl_map);
	md_free(curl_map);
	md_free(curl_wgh);

	return 0 < pos.N;
}


bool phase_pole_correction_loop(struct pole_config_s conf, int N, bart_flags_t lflags, const bart_dim_t pmap_dims[N], complex float* phase, const bart_dim_t sens_dims[N], const complex float* sens)
{
	bool ret = false;

	bart_dim_t npmap_dims[N];
	bart_dim_t nsens_dims[N];

	md_select_dims(N, ~lflags, npmap_dims, pmap_dims);
	md_select_dims(N, ~lflags, nsens_dims, sens_dims);

	bart_stride_t pmap_strs[N];
	bart_stride_t sens_strs[N];

	md_calc_strides(N, pmap_strs, pmap_dims, CFL_SIZE);
	md_calc_strides(N, sens_strs, sens_dims, CFL_SIZE);

	bart_dim_t pos[N];
	md_set_dims(N, pos, 0);

	do {
		//FIXME: this should probybly be moved to src/noir/recon2.c
		if (0 != pos[MAPS_DIM])
			continue;

		ret = phase_pole_correction(conf, N, npmap_dims, &MD_ACCESS(N, pmap_strs, pos, phase), nsens_dims, &MD_ACCESS(N, sens_strs, pos, sens)) || ret;

	} while (md_next(N, pmap_dims, lflags, pos));

	return ret;
}


void phase_pole_normalize(int N, const bart_dim_t pdims[N], complex float* phase, const bart_dim_t idims[N], const complex float* image)
{
	complex float* timage = md_alloc_sameplace(N, idims, CFL_SIZE, image);

	md_ztenmul(N, idims, timage, pdims, phase, pdims, image);

	bart_dim_t tpdims[N];
	md_select_dims(N, ~UINT64_C(7), tpdims, pdims);

	complex float* dot = md_alloc_sameplace(N, tpdims, CFL_SIZE, image);

	md_ztenmulc(N, tpdims, dot, idims, image, idims, timage);
	md_zphsr(N, tpdims, dot, dot);

	md_zmul2(N, pdims, MD_STRIDES(N, pdims, CFL_SIZE), phase, MD_STRIDES(N, pdims, CFL_SIZE), phase, MD_STRIDES(N, tpdims, CFL_SIZE), dot);

	md_free(timage);
	md_free(dot);
}



static float get_min_dist(const vec3_t pos, int N, const vec3_t r[N][2])
{
	assert(1 <= N);

	float dist = dist_to_lineseg(pos, r[0]);

	for (int i = 1; i < N; i++)
		dist = MIN(dist, dist_to_lineseg(pos, r[i]));

	return dist;
}

/**
*	(1 / sqrt(1 + r2 / d2) - 1) / r2
*	define y = r2 / d2
*	1 / d2 * (1 / sqrt(1 + y) - 1) / y
**/
static float taylor(float r2, float d2)
{
	float y = r2 / d2;

	if (y > 1.e-4)
		return (1. / sqrtf(1. + r2 / d2) - 1) / r2;
	else
		return (-0.5 + 3. / 8. * y - 5. / 16. * y * y) / d2;
}


//https://books.physics.oregonstate.edu/GSF/wire.html
//compute phase due to a line segment from (0, 0, -L) to (0, 0, L)
// computes -1 / (2r^2) * ((z-L)/sqrt(r^2 + (z-L)^2) - (z+L)/sqrt(r^2 + (z+L)^2))
static float get_grad_wire_zaxis(float r, float z, float L)
{
	float r2 = powf(r, 2);
	float dp = powf(z + L, 2);
	float dm = powf(z - L, 2);

	if (z >= -L && z <= L) {

		float t1 = (z + L) / sqrtf(r2 + dp) / (2. * r2);
		float t2 = (z - L) / sqrtf(r2 + dm) / (2. * r2);

		return t1 - t2;
	} else {

		float sgn =  z > L ? 1 : -1;

		// we use taylor expansion for r << 1 as the zero order term cancels
		float t1 = sgn * 0.5 * taylor(r2, dp);
		float t2 = sgn * 0.5 * taylor(r2, dm);

		return t1 - t2;
	}
}

static void get_grad_line(vec3_t grad, const vec3_t pos, const vec3_t r2, const vec3_t r1)
{
	vec3_t cen;
	vec3_add(cen, r1, r2);
	vec3_smul(cen, cen, 0.5);

	vec3_t Lvec;
	vec3_sub(Lvec, r2, cen);

	float L = vec3_norm(Lvec);

	vec3_t ez;
	vec3_smul(ez, Lvec, 1. / L);

	vec3_t rer;	//r*\hat{e}_r; don't devide by r as r might be 0
	vec3_sub(rer, pos, cen);

	float z = vec3_sdot(rer, ez);

	vec3_saxpy(rer, rer, -z, ez);

	float r = vec3_norm(rer);

	vec3_rot(grad, ez, rer); //grad = r*\hat{e}_\phi; don't devide by r as r might be 0

	vec3_smul(grad, grad, get_grad_wire_zaxis(r, z, L));
}

static void get_grad(vec3_t grad, const vec3_t pos, int N, const vec3_t r[N][2])
{
	grad[0] = 0.;
	grad[1] = 0.;
	grad[2] = 0.;

	for (int i = 0; i < N; i++) {

		vec3_t tgrad;
		get_grad_line(tgrad, pos, r[i][0], r[i][1]);

		vec3_add(grad, grad, tgrad);
	}
}

float integrate_phase(int M, const float pos[M][3], int N, const float r[N][2][3], float tol)
{
	const float* ptr_pos = &pos[0][0];
	const float* ptr_r = &r[0][0][0];

	NESTED(void, eval, (float out[1], float t, const float /*in*/[1]))
	{
		vec3_t cpos;
		vec3_t dcpos;
		vec3_t grad;

		int index = (int)(t * (M - 1));
		float t_index = t * (M - 1) - index;

		vec3_sub(dcpos, ptr_pos + 3 * (MIN(M - 1, index + 1)), ptr_pos + 3 * index);
		vec3_saxpy(cpos, ptr_pos + 3 * index, t_index, dcpos);
		vec3_smul(dcpos, dcpos, (M - 1));

		get_grad(grad, cpos, N, (const float(*)[2][3])ptr_r);

		out[0] = vec3_sdot(grad, dcpos);
	};

	float phi[] = { 0. };
	ode_interval(0.5 / (M - 1), -1., tol, 1, phi, 0., 1., eval);

	return phi[0];
}


static void search_start(long pos[3], const long dims[3], int D, const float r[D][2][3])
{
	for (int i = 0; i < 3; i++)
		pos[i] = dims[i] / 2;

	float fpos[3];
	for (int i = 0; i < 3; i++)
		fpos[i] = pos[i];

	struct bart_rand_state* rstate = rand_state_create(123);

	while (2. > get_min_dist(fpos, D, r)) {

		md_unravel_index(3, pos, 7ul, dims, rand_range_state(rstate, md_calc_size(3, dims)));
		for (int i = 0; i < 3; i++)
			fpos[i] = pos[i];
	}
}

static long get_integration_order(const long dims[3], long* sint, long* eint, complex float* visited, long start_idx, int D, const float r[D][2][3])
{
	double time = -timestamp();
	debug_printf(DP_DEBUG1, "Start finding integration path ... ");

	long qmax = 0; // queue of points to be processed
	long qmin = 0;

	md_clear(3, dims, visited, CFL_SIZE);

	long idx = start_idx;
	visited[idx] = 1.;

	do {
		long pos[3];
		md_unravel_index(3, pos, ~0UL, dims, idx);

		for (int i = 0; i < 6; i++) {

			long pos2[3];
			md_copy_dims(3, pos2, pos);

			pos2[i / 2] += (0 == i % 2) ? 1 : -1;

			if ((0 > pos2[i / 2]) || (dims[i / 2] == pos2[i / 2]))
				continue;

			long idx2 = md_ravel_index(3, pos2, ~0UL, dims);

			if (0. != visited[idx2])
				continue;

			vec3_t path[2];
			for(int i = 0; i < 3; i++) {

				path[0][i] = pos[i];
				path[1][i] = pos2[i];
			}

			bool stop = false;

			for (int i = 0; i < D; i++)
				stop = stop || (dist_of_linesegs_smaller(path, r[i], 0.3));

			if (stop)
				continue;

			visited[idx2] = 1.;

			sint[qmax] = idx;
 			eint[qmax++] = idx2;
		}

		if (qmin == qmax)
			break;
		else
			idx = eint[qmin++];

	} while (true);

	debug_printf(DP_DEBUG1, "done (took %es)\n", timestamp() + time);

	return qmax;
}

static void integrate_path(long qmax, const long dims[3], float* dphi, const long* sint, const long* eint, int D, const float r[D][2][3], float tol)
{
	debug_printf(DP_DEBUG1, "Start integrating phases ... ");
	double time = -timestamp();

#pragma omp parallel for
	for (long i = 0; i < qmax; i++) {

		long idx = sint[i];
		long idx2 = eint[i];

		long pos[3];
		long pos2[3];

		md_unravel_index(3, pos, ~0UL, dims, idx);
		md_unravel_index(3, pos2, ~0UL, dims, idx2);

		float fpos[2][3];

		for (int i = 0; i < 3; i++) {

			fpos[0][i] = pos[i];
			fpos[1][i] = pos2[i];
		}

		dphi[i] = integrate_phase(2, fpos, D, r, tol);

		if (!safe_isfinite(dphi[i]))
			dphi[i] = 0.;
	}

	debug_printf(DP_DEBUG1, "done (%ld paths, took %es)\n", qmax, timestamp() + time);
}


static void grid_to_fov(const long dims[3], vec3_t pos)
{
	for (int i = 0; i < 3; i++)
		pos[i] = (pos[i] - dims[i] / 2) / (float)dims[i];
}

static void fov_to_grid(const long dims[3], vec3_t pos)
{
	for (int i = 0; i < 3; i++)
		pos[i] = pos[i] * dims[i] + dims[i] / 2;
}


void sample_phase_pole_3D(int N, const long dims[N], complex float* dst, int D, const float r_fov[D][2][3], float tol)
{
#ifdef USE_GPU
	if (cuda_ondevice(dst)) {

		complex float* tmp = md_alloc(N, dims, CFL_SIZE);
		sample_phase_pole_3D(N, dims, tmp, D, r_fov, tol);
		md_copy(N, dims, dst, tmp, CFL_SIZE);
		return;
	}
#endif

	assert(3 <= N);
	assert(1 == md_calc_size(N - 3, dims + 3));

	long* eint = md_alloc(1, MD_DIMS(md_calc_size(3, dims)), sizeof(long));
	long* sint = md_alloc(1, MD_DIMS(md_calc_size(3, dims)), sizeof(long));
	md_clear(N, dims, eint, sizeof(long));
	md_clear(N, dims, sint, sizeof(long));

	float r[D][2][3];

	for (int i = 0; i < D; i++) {

		vec3_copy(r[i][0], r_fov[i][0]);
		vec3_copy(r[i][1], r_fov[i][1]);

		fov_to_grid(dims, r[i][0]);
		fov_to_grid(dims, r[i][1]);
	}

	// we don't want to integrate through singularities
	// so we start with a point away from singularities
	long pos[3];
	search_start(pos, dims, D, r);

	debug_printf(DP_DEBUG1, "Found start position for integraion: (%ld, %ld, %ld)\n", pos[0], pos[1], pos[2]);

	long idx = md_ravel_index(3, pos, ~0UL, dims);
	long qmax = get_integration_order(dims, sint, eint, dst, idx, D, r);

	float* dphi = md_alloc(1, MD_DIMS(qmax), FL_SIZE);

	integrate_path(qmax, dims, dphi, sint, eint, D, r, tol);

	md_clear(N, dims, dst, CFL_SIZE);

	dst[sint[0]] = 1.;

	for (long i = 0; i < qmax; i++)
		dst[eint[i]] = cexpf(-1.I * dphi[i]) * dst[sint[i]];

	md_free(eint);
	md_free(sint);
	md_free(dphi);
}





static float phase_pole_3D_estimate_polarity(int N, const long dims[N], const complex float* col, float r[2][3], float radius, int segments)
{
	vec3_t cen;
	vec3_add(cen, r[0], r[1]);
	vec3_smul(cen, cen, 0.5);

	vec3_t r1;
	vec3_t r2;

	for (int i = 0; i < 3; i++) {

		r1[i] = r[0][i];
		r2[i] = r[1][i];
	}

	vec3_t evec[3];
	get_coord_transform(evec, r2, r1);

	vec3_t evecT[3];
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			evecT[i][j] = evec[j][i];

	long pos1[N];
	long pos2[N];

	md_set_dims(N, pos1, 0);
	md_set_dims(N, pos2, 0);

	long sdims[N];
	md_select_dims(N, ~7UL, sdims, dims);

	complex float* tmp1 = md_alloc(N, sdims, CFL_SIZE);
	complex float* tmp2 = md_alloc(N, sdims, CFL_SIZE);
	complex float* dst = md_alloc(N, sdims, CFL_SIZE);

	md_clear(N, sdims, dst, CFL_SIZE);

	int M = segments;
	for (int i = 0; i < M; i++) {

		vec3_t pos_seg1 = { radius * cos (2. * M_PI * ((i + 0) % M) / M), radius * sin(2. * M_PI * ((i + 0) % M) / M), 0. };
		vec3_t pos_seg2 = { radius * cos (2. * M_PI * ((i + 1) % M) / M), radius * sin(2. * M_PI * ((i + 1) % M) / M), 0. };

		for (int j = 0; j < 3; j++) {

			pos1[j] = MAX(MIN(dims[j] - 1, lroundf(vec3_sdot(pos_seg1, evecT[j]) + cen[j])), 0);
			pos2[j] = MAX(MIN(dims[j] - 1, lroundf(vec3_sdot(pos_seg2, evecT[j]) + cen[j])), 0);
		}

		if (md_check_equal_dims(3, pos1, pos2, 7UL))
			continue;

		md_slice(N, 7UL, pos1, dims, tmp1, col, CFL_SIZE);
		md_slice(N, 7UL, pos2, dims, tmp2, col, CFL_SIZE);

		md_zmulc(N, sdims, tmp1, tmp1, tmp2);
		md_zphsr(N, sdims, tmp1, tmp1); // to handle 0.+0.i case
		md_zarg(N, sdims, tmp1, tmp1);
		md_zadd(N, sdims, dst, dst, tmp1);
	}

	md_free(tmp1);
	md_free(tmp2);

	bool okay = true;
	int ret = 0;

	for (int i = 0; i < md_calc_size(N, sdims); i++) {

		int tmp = lroundf(dst[i] / (2. * M_PI));

		if (0 == tmp)
			continue;

		if (0 == ret)
			ret = tmp;
		else
			okay = okay && (ret == tmp);
	}

	md_free(dst);

	if (!okay) {

		debug_printf(DP_DEBUG1, "Phase pole polarity is not consistent across coils!\n");
		return 0.;
	}

	return (float)ret;
}

static void phase_pole_3D_fix_polarity(int N, const long dims[N], const complex float* col, struct lseg_s* seg, float radius, int segments)
{
	line_segments_sort(seg);

	for (int i = 0; i < seg->N; i++) {

		fov_to_grid(dims, seg->pos[i][0]);
		fov_to_grid(dims, seg->pos[i][1]);
	}

	int i = 0;
	int D = 0;
	float sum = 0.;

	while (i < seg->N) {

		while ((0 == D) || ((i + D < seg->N) && (0. == vec3_dist(seg->pos[i + D][0], seg->pos[i + D - 1][1])))) {

			float tmp = phase_pole_3D_estimate_polarity(N, dims, col, seg->pos[i + D], radius, segments);

			if (tmp * sum < -0.5)
				debug_printf(DP_DEBUG1, "Phase pole polarity is not consistent across segments\n");

			sum += tmp;
			D++;
		}

		if (0 > sum) {

			debug_printf(DP_DEBUG1, "Phase pole polarity (%e) is negative, flipping ...\n", sum / D);
			line_segments_revert(D, seg->pos + i);
		} else {

			debug_printf(DP_DEBUG1, "Phase pole polarity (%e) is positive, nothing to do ...\n", sum / D);
		}

		i += D;
		D = 0;
		sum = 0.;
	}

	for (int i = 0; i < seg->N; i++) {

		grid_to_fov(dims, seg->pos[i][0]);
		grid_to_fov(dims, seg->pos[i][1]);
	}
}


struct lseg_s extract_phase_poles_3D(struct pole_config_s conf, int N, const long dims[N], const _Complex float* curl_map, const long sens_dims[N], const _Complex float* sens)
{
	complex float* tmp = md_alloc_sameplace(N, dims, CFL_SIZE, curl_map);

	md_zabs(N, dims, tmp, curl_map);
	md_zsgreatequal(N, dims, tmp, tmp, conf.thresh);

	float radius = MAX(get_diameter(conf, 0, dims, false), get_diameter(conf, 1, dims, false)) / 2.;

	struct lseg_s ret = md_trace_binary_mask(3, dims, tmp, radius);

	md_free(tmp);

	phase_pole_3D_fix_polarity(N, sens_dims, sens, &ret, radius, conf.segments);
	line_segments_connect(&ret);
	line_segments_extend_bounds(&ret);

	return ret;
}
