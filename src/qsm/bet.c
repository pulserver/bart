/* Copyright 2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2026 Tobias Paul Trimmal
 *
 * References:
 *
 * Smith SM. Fast robust automated brain extraction.
 * Hum Brain Mapp. 2002;17:143-155.
 *
 * Jacobson, Alec & Kavan, Ladislav & Sorkine-Hornung, Olga.
 * Robust inside-outside segmentation using generalized winding numbers.
 * ACM Transactions on Graphics. 2013;32:1-12. 10.1145/2461912.2461916.
 */

#include <assert.h>
#include <complex.h>
#include <math.h>

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/loop.h"
#include "num/vec3.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"


#include "stl/misc.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "bet.h"

#define EPS 1e-6f
#define EPS_NORMAL 1e-12f

static int cmp_float(const void* a, const void* b)
{
	float fa = *(const float*)a;
	float fb = *(const float*)b;

	return (fa > fb) - (fa < fb);
}


void bet_threshold(int N, bart_dim_t dims[N], float* img,
	float* new_img, float* t, float* t98, float* t2)
{
	bart_dim_t n = md_calc_size(3, dims);

	assert(n > 0);

	float (*tmp)[n] = xmalloc(sizeof(*tmp));
	memcpy(*tmp, img, sizeof(*tmp));

	int k2 = (int)(0.98f * (n - 1));
	int k98 = (int)(0.02f * (n - 1));

	*t2  = quickselect(*tmp, n, k2);
	*t98 = quickselect(*tmp, n, k98);

	*t = *t2 + 0.1f * (*t98 - *t2);

	for (bart_dim_t i = 0; i < n; i++) {

		if (*t > img[i])
			new_img[i] = 0.0f;
		else
			new_img[i] = img[i];
	}

	xfree(tmp);
}


void compute_cog(int N, bart_dim_t dims[N], const float* img, const float res[3], float* t,
	float* t98, float COG[3], float* R_out)
{
	assert(N > 3);
	assert(1 < md_calc_size(3, dims));

	bart_dim_t nx = dims[0];
	bart_dim_t ny = dims[1];
	bart_dim_t nz = dims[2];

	float sum_w = 0.;
	float sum_x = 0.;
	float sum_y = 0.;
	float sum_z = 0.;
	bart_dim_t number = 0;
	float w = 0.;

	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			for (int k = 0; k < nz; k++) {

				bart_dim_t index = k * ny * nx + j * nx + i;

				assert(index >= 0);

				float value = img[index];

				if (*t >= value)
					continue;

				number++;

				if (*t98 <= value)
					w = *t98;
				else
					w = value;

				sum_w += w;
				sum_x += w * i * res[0];
				sum_y += w * j * res[1];
				sum_z += w * k * res[2];
			}
		}
	}

	double V = (double)number * (double)res[0] * (double)res[1] * (double)res[2];

	double R = cbrt((3.0 * V) / (4.0 * M_PI));

	COG[0] = sum_x / sum_w;
	COG[1] = sum_y / sum_w;
	COG[2] = sum_z / sum_w;

	*R_out = (float)R;
}

float compute_tm(int N, bart_dim_t dims[N], const float* image, const float voxel_size[3],
	const float COG[3], float R)
{
	bart_dim_t nx = dims[0];
	bart_dim_t ny = dims[1];
	bart_dim_t nz = dims[2];

	bart_dim_t max_vals = nx * ny * nz;

	assert(max_vals > 0);

	float* vals = xmalloc(sizeof(float[max_vals]));

	bart_dim_t n = 0;

	for (int ix = 0; ix < nx; ix++) {
		for (int iy = 0; iy < ny; iy++) {
			for (int iz = 0; iz < nz; iz++) {

				float x = ix * voxel_size[0] - COG[0];
				float y = iy * voxel_size[1] - COG[1];
				float z = iz * voxel_size[2] - COG[2];

				if (x * x + y * y + z * z <= R * R) {

					vals[n] = image[(iz) * ny * nx + (iy) * nx + (ix)];
					n++;
				}
			}
		}
	}

	qsort(vals, (size_t)n, sizeof(float), cmp_float);

	float tm = vals[n / 2];

	free(vals);

	return tm;
}


static void compute_normal(int i, const double* verts, const struct neighbors* neigh,
	const float COG[3], float n_hat[3])
{
	const double* v = &verts[3 * i];
	const struct neighbors* nb = &neigh[i];

	float n[3] = { 0., 0., 0. };

	for (int k = 0; k < nb->n; k++) {

		int i1 = nb->v[k];
		int i2 = nb->v[(k + 1) % nb->n];

		float v1[3];
		float v2[3];

		for (int d = 0; d < 3; d++) {

			v1[d] = verts[3 * i1 + d] - v[d];
			v2[d] = verts[3 * i2 + d] - v[d];
		}

		n[0] += v1[1] * v2[2] - v1[2] * v2[1];
		n[1] += v1[2] * v2[0] - v1[0] * v2[2];
		n[2] += v1[0] * v2[1] - v1[1] * v2[0];
	}

	float norm = vec3_norm(n);

	if (EPS_NORMAL > norm)
		norm = 1.f;

	for (int d = 0; d < 3; d++)
		n_hat[d] = n[d] / norm;

	float dot = 0.f;

	for (int d = 0; d < 3; d++)
		dot += n_hat[d] * (v[d] - COG[d]);

	if (0.f > dot)
		for (int d = 0; d < 3; d++)
			n_hat[d] = -n_hat[d];
}

static float mean_edge_length(const double* verts, int nv, const struct neighbors* neigh)
{
	double sum = 0.0;
	int count = 0;

	for (int i = 0; i < nv; i++) {

		for (int k = 0; k < neigh[i].n; k++) {

			int j = neigh[i].v[k];

			float dx = verts[3 * j + 0] - verts[3 * i + 0];
			float dy = verts[3 * j + 1] - verts[3 * i + 1];
			float dz = verts[3 * j + 2] - verts[3 * i + 2];

			sum += sqrtf(dx * dx + dy * dy + dz * dz);
			count++;
		}
	}

	return (float)(sum / count);
}

static float sample_image_nn(int N, bart_dim_t dims[N], const float* image,
	const float voxel_size[3], const float x[3])
{
	int nx = dims[0];
	int ny = dims[1];
	int nz = dims[2];

	int ix = lroundf(x[0] / voxel_size[0]);
	int iy = lroundf(x[1] / voxel_size[1]);
	int iz = lroundf(x[2] / voxel_size[2]);

	if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz)
		return 0.f;

	return image[(iz) * ny * nx + (iy) * nx + (ix)];
}

static void sample_ray(int N, bart_dim_t dims[N], const float v[3], const float n_hat[3],
	const float* image, const float voxel_size[3], float* Imin, float* Imax)
{
	const float d1 = 20.f;
	const float d2 = 10.f;
	const int steps = 20;

	*Imin = 1e30f;
	*Imax = -1e30f;

	for (int s = 0; s < steps; s++) {

		float d = d1 * s / (steps - 1);
		float x[3];

		for (int k = 0; k < 3; k++)
			x[k] = v[k] - d * n_hat[k];

		float intensity = sample_image_nn(DIMS, dims, image, voxel_size, x);

		if (*Imin > intensity)
			*Imin = intensity;

		if (d <= d2 && intensity > *Imax)
			*Imax = intensity;
	}
}


static void update_vertex(int N, bart_dim_t dims[N], int i, double* verts, const struct neighbors* neigh,
	const float* image, const float voxel_size[3], const float COG[3], float t2, float t,
	float tm, float bt, float l, float du[3])
{
	float v[3] = {
		verts[3 * i + 0],
		verts[3 * i + 1],
		verts[3 * i + 2]
	};

	float n_hat[3];
	compute_normal(i, verts, neigh, COG, n_hat);

	float mean[3] = { 0, 0, 0 };

	for (int k = 0; k < neigh[i].n; k++) {

		int j = neigh[i].v[k];

		for (int d = 0; d < 3; d++)
			mean[d] += verts[3 * j + d];
	}

	for (int d = 0; d < 3; d++)
		mean[d] /= neigh[i].n;

	float s[3];

	for (int d = 0; d < 3; d++)
		s[d] = mean[d] - v[d];

	float sn = s[0] * n_hat[0] + s[1] * n_hat[1] + s[2] * n_hat[2];

	float s_n[3], s_t[3];

	for (int d = 0; d < 3; d++) {

		s_n[d] = sn * n_hat[d];
		s_t[d] = s[d] - s_n[d];
	}

	float u1[3] = {
		0.5f * s_t[0],
		0.5f * s_t[1],
		0.5f * s_t[2]
	};

	float u2[3] = { 0, 0, 0 };
	float sn_norm = fabsf(sn);

	if (sn_norm > EPS) {

		float r = (l * l) / (2.f * sn_norm);
		float E = 0.5f * (1.f / 3.33f + 1.f / 10.f);
		float F = 6.f / (1.f / 3.33f - 1.f / 10.f);
		float f2 = 0.05f * (1.f - tanhf(F * (1.f / r - E)));

		for (int d = 0; d < 3; d++)
			u2[d] = f2 * s_n[d];
	}

	float Imin;
	float Imax;

	sample_ray(DIMS, dims, v, n_hat, image, voxel_size, &Imin, &Imax);

	Imin = fmaxf(t2, fminf(Imin, tm));
	Imax = fminf(tm, fmaxf(Imax, t));

	float f3 = 2.f * (Imin - (t2 + bt * (Imax - t2))) / (Imax - t2 + EPS);

	float u3[3];

	for (int d = 0; d < 3; d++)
		u3[d] = 0.05f * f3 * l * n_hat[d];

	for (int d = 0; d < 3; d++)
		du[d] = u1[d] + u2[d] + u3[d];

}

static void bet_iteration(int N, bart_dim_t dims[N], double* verts, int nv, const struct neighbors* neigh,
	const float* image, const float voxel_size[3], const float COG[3],
	float t2, float t, float tm, float bt, float l)
{
	for (int i = 0; i < nv; i++) {

		float du[3];
		update_vertex(N, dims, i, verts, neigh, image,
			voxel_size, COG, t2, t, tm, bt, l, du);

		for (int d = 0; d < 3; d++)
			verts[3 * i + d] += du[d];
	}
}

void run_bet(int N, bart_dim_t dims[N], double* verts, int nv, const struct neighbors* neigh,
	const float* image, const float voxel_size[3], const float COG[3],
	float t2, float t, float tm, float bt, int n_iter)
{
	float l = mean_edge_length(verts, nv, neigh);

	for (int it = 0; it < n_iter; it++)
		bet_iteration(N, dims, verts, nv, neigh, image, voxel_size, COG, t2, t, tm, bt, l);

}

static int intersect_triangle_z(const double v0[3], const double v1[3], const double v2[3], float z, struct Segment2* seg)
{
	float t[3];
	const double* v[3] = { v0, v1, v2 };
	int n = 0;

	for (int i = 0; i < 3; i++) {

		const double* p0 = v[i];
		const double* p1 = v[(i + 1) % 3];

		double z0 = p0[2] - z;
		double z1 = p1[2] - z;

		if ((z0 > 0 && z1 < 0) || (z0 < 0 && z1 > 0)) {

			float alpha = z0 / (z0 - z1);
			t[n++] = alpha;
		}
	}

	if (2 != n)
		return 0;

	for (int i = 0; i < 3; i++) {

		const double* p0 = v[i];
		const double* p1 = v[(i + 1) % 3];

		double z0 = p0[2] - z;
		double z1 = p1[2] - z;

		if ((z0 > 0 && z1 < 0) || (z0 < 0 && z1 > 0)) {

			double alpha = z0 / (z0 - z1);

			seg->a.x = p0[0] + alpha * (p1[0] - p0[0]);
			seg->a.y = p0[1] + alpha * (p1[1] - p0[1]);
			break;
		}
	}

	for (int i = 0; i < 3; i++) {

		const double* p0 = v[i];
		const double* p1 = v[(i + 1) % 3];

		double z0 = p0[2] - z;
		double z1 = p1[2] - z;

		if ((z0 > 0 && z1 < 0) || (z0 < 0 && z1 > 0)) {

			double alpha = z0 / (z0 - z1);

			if (fabs(alpha - t[0]) > EPS) {

				seg->b.x = p0[0] + alpha * (p1[0] - p0[0]);
				seg->b.y = p0[1] + alpha * (p1[1] - p0[1]);
				return 1;
			}
		}
	}

	return 0;
}

static int ray_intersects_segment_2d(float px, float py, const float n[2], const struct Segment2* seg)
{
	float sx = seg->b.x - seg->a.x;
	float sy = seg->b.y - seg->a.y;
	float den = n[0] * sy - n[1] * sx;

	if (fabsf(den) <= EPS)
		return 0;

	float ax = seg->a.x - px;
	float ay = seg->a.y - py;

	float t = (ax * sy - ay * sx) / den;
	float s = (ax * n[1] - ay * n[0]) / den;

	if ((s < 0.f) || (s >= 1.f))
		return 0;

	if (t > EPS)
		return 1;

	if (t < -EPS)
		return 2;

	return 0;
}

void mesh_to_mask_slicewise(int N, bart_dim_t dims[N], float* mask, float resolution[3], const double (*verts)[3], const int (*tris)[3], int ntris)
{
	int nx = dims[0];
	int ny = dims[1];
	int nz = dims[2];

	float vx = resolution[0];
	float vy = resolution[1];
	float vz = resolution[2];

	struct Segment2 (*segments)[ntris] = xmalloc(sizeof(*segments));

	bart_dim_t count_inconsistent_pixel = 0;

	for (int k = 0; k < nz; k++) {

		float z = (k + 0.5f) * vz;
		int nseg = 0;

		for (int t = 0; t < ntris; t++) {

			const double* v0 = verts[tris[t][0]];
			const double* v1 = verts[tris[t][1]];
			const double* v2 = verts[tris[t][2]];

			nseg += intersect_triangle_z(v0, v1, v2, z, &(*segments)[nseg]);
		}

		if (0 == nseg)
			continue;

		float n[4][2] = { {0.f, 1.f}, {1.f, 0.f}, { sqrtf(2.f), sqrtf(2.f) }, { sqrtf(2.f), -sqrtf(2.f) } };

		for (int j = 0; j < ny; j++) {

			for (int i = 0; i < nx; i++) {

				float x = (i + 0.5f) * vx;
				float y = (j + 0.5f) * vy;

				int crossings_pos[4] = { };
				int crossings_neg[4] = { };

				for (int s = 0; s < nseg; s++) {

					for (int d = 0; d < 4; d++) {

						int cross = ray_intersects_segment_2d(x, y, n[d], &(*segments)[s]);

						crossings_pos[d] += (1 == cross);
						crossings_neg[d] += (2 == cross);
					}
				}

				int count_odd = 0;

				for (int d = 0; d < 4; d++)
					count_odd += (crossings_pos[d] % 2) + (crossings_neg[d] % 2);

				if (2 < count_odd && count_odd < 6)
					count_inconsistent_pixel++;

				mask[i + nx * j + nx * ny * k] = (count_odd > 4) ? 1.0f : 0.0f;
			}
		}
	}

	if (0 < count_inconsistent_pixel)
		debug_printf(DP_WARN, "Warning: %" PRId64 " of %" PRId64 " pixels have inconsistent ray crossings.\n", count_inconsistent_pixel, (bart_dim_t)(nx * ny * nz));

	xfree(segments);
}

static double compute_solid_angle(const double p[3], const double v0[3], const double v1[3], const double v2[3])
{
	float a[3], b[3], c[3];

	for (int d = 0; d < 3; d++) {

		a[d] = v0[d] - p[d];
		b[d] = v1[d] - p[d];
		c[d] = v2[d] - p[d];
	}

	float len_a = vec3_norm(a);
	float len_b = vec3_norm(a);
	float len_c = vec3_norm(a);

	float dot_ab = vec3_sdot(a, b);
	float dot_bc = vec3_sdot(b, c);
	float dot_ca = vec3_sdot(c, a);

	float det_abc =  c[0] * (a[1] * b[2] - a[2] * b[1])
		+ c[1] * (a[2] * b[0] - a[0] * b[2])
		+ c[2] * (a[0] * b[1] - a[1] * b[0]);

	float denom = len_a * len_b * len_c + dot_ab * len_c + dot_bc * len_a + dot_ca * len_b;

	return 2 * atan2(det_abc, denom);
}


void mesh_to_mask_winding_number(int N, bart_dim_t dims[N], float* mask, float resolution[3],
	const double (*verts)[3], const int (*tris)[3], int ntris)
{
	for (int d = 3; d < N; d++)
		assert(dims[d] == 1);

	const float factor = 1.0 / (4.0 * M_PI);

	bart_stride_t strs[N];
	md_calc_strides(N, strs, dims, sizeof(float));

#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
	for (int k = 0; k < dims[2]; k++)
	for (int j = 0; j < dims[1]; j++)
	for (int i = 0; i < dims[0]; i++) {

		double center[3] = { (i + 0.5) * resolution[0], (j + 0.5) * resolution[1], (k + 0.5) * resolution[2] };

		float solid_angle_sum = 0.0;

		for (int t = 0; t < ntris; t++) {

			const double* v0 = verts[tris[t][0]];
			const double* v1 = verts[tris[t][1]];
			const double* v2 = verts[tris[t][2]];

			solid_angle_sum += compute_solid_angle(center, v0, v1, v2);
		}

		bart_dim_t pos[N] = { };

		pos[0] = i;
		pos[1] = j;
		pos[2] = k;

		MD_ACCESS(N, strs, pos, mask) = (solid_angle_sum * factor > 0.5) ? 1.f : 0.f;
	}
}

