/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2025 Moritz Blumenthal
 */

 #include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <math.h>

#include "misc/misc.h"
#include "misc/list.h"
#include "misc/egraph.h"
#include "misc/debug.h"

#include "num/vec3.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/morph.h"

#include "lineseg.h"


float dist_to_lineseg(const vec3_t pos, const vec3_t seg[2])
{
	vec3_t n;
	vec3_sub(n, seg[1], seg[0]);
	float tmax = vec3_norm(n);
	vec3_smul(n, n, 1. / tmax);

	vec3_t dif;
	vec3_sub(dif, pos, seg[0]);
	vec3_saxpy(dif, dif, -vec3_sdot(dif, n) / vec3_norm(n), n);

	float dist = vec3_norm(dif);

	vec3_add(dif, dif, pos); // closest point on line
	vec3_sub(dif, dif, seg[0]);
	float t = vec3_sdot(dif, n);

	if ((0 <= t) && (t <= tmax))
		return dist;
	else
		return MIN(vec3_dist(pos, seg[0]), vec3_dist(pos, seg[1]));
}

float dist_of_linesegs(const vec3_t seg1[2], const vec3_t seg2[2])
{
	vec3_t u;
	vec3_t v;
	vec3_t w;

	vec3_sub(u, seg1[1], seg1[0]);
	vec3_sub(v, seg2[1], seg2[0]);
	vec3_sub(w, seg1[0], seg2[0]);

/*
	w = r0 - r1
	r0 + u * t
	r1 + v * s

	d(t, s)^2 = || r0 - r1 + u * t - v * s||^2
		  = || w + u * t - v * s||^2

	A = (-u v)

	A^tA = ( u*u u*v ) = ( a b )
	       ( u*v v*v )   ( b c )

	(A^tA)^(-1)A
*/
	float a = vec3_sdot(u, u);
	float b = -vec3_sdot(u, v);
	float c = vec3_sdot(v, v);

	float D = a * c - b * b;

	if (1.e-5 < fabsf(D)) {

		float t1 = -vec3_sdot(u, w);
		float s1 = vec3_sdot(v, w);

		float t = c * t1 - b * s1;
		float s = a * s1 - b * t1;

		t /= D;
		s /= D;

		if ((0 <= t) && (t <= 1) && (0 <= s) && (s <= 1)) {

			vec3_saxpy(u, seg1[0], t, u);
			vec3_saxpy(v, seg2[0], s, v);

			return vec3_dist(u, v);
		}
	}

	return MIN(MIN(dist_to_lineseg(seg1[0], seg2), dist_to_lineseg(seg1[1], seg2)),
		   MIN(dist_to_lineseg(seg2[0], seg1), dist_to_lineseg(seg2[1], seg1)));
}

bool dist_of_linesegs_smaller(const float seg1[2][3], const float seg2[2][3], float tol)
{
	float r1 = vec3_dist(seg1[0], seg1[1]);
	float r2 = vec3_dist(seg2[0], seg2[1]);

	vec3_t cen1;
	vec3_t cen2;

	vec3_add(cen1, seg1[0], seg1[1]);
	vec3_add(cen2, seg2[0], seg2[1]);

	if (0.5 * vec3_dist(cen1, cen2) - r1 - r2 > tol)
		return false;

	return tol > dist_of_linesegs(seg1, seg2);
}



static void douglas_peucker_rec(int N, vec3_t pos[N], bool active[N], long start, long end, float tol)
{
	long max_index = start;
	float max_dist = 0.;

	vec3_t seg[2];
	vec3_copy(seg[0], pos[start]);
	vec3_copy(seg[1], pos[end]);

	for (long i = start + 1; i < end; i++) {

		float dist = dist_to_lineseg(pos[i], seg);

		if (dist > max_dist) {

			max_index = i;
			max_dist = dist;
		}
	}

	if (max_dist > tol) {

		active[max_index] = true;

		douglas_peucker_rec(N, pos, active, start, max_index, tol);
		douglas_peucker_rec(N, pos, active, max_index, end, tol);
	}
}

long douglas_peucker(long N, vec3_t pos[N], float tol)
{
	bool active[N];
	for (long i = 0; i < N; i++)
		active[i] = false;

	active[0] = true;
	active[N - 1] = true;
	douglas_peucker_rec(N, pos, active, 0, N - 1, tol);

	long count = 0;
	for (long i = 0; i < N; i++)
		if (active[i])
			vec3_copy(pos[count++], pos[i]);

	return count;
}



static bool con26(const vec3_t pos1, const vec3_t pos2)
{
	return (1.001 >= fabsf(pos1[0] - pos2[0])) && (1.001 >= fabsf(pos1[1] - pos2[1])) && (1.001 >= fabsf(pos1[2] - pos2[2]));
}


static long md_mask_count(int N, const long dims[N], complex float* mask)
{
	long count = 0;

	long size = md_calc_size(N, dims);

	for (long i = 0; i < size; i++)
		if (0. != crealf(mask[i]))
			count++;

	return count;
}

static long md_mask_to_pos(int N, const long dims[N], unsigned long flags, long max, int M, float pos[max][M], const complex float* mask)
{
	assert(M == bitcount(flags));

	long count = 0;

	long size = md_calc_size(N, dims);

	for (int i = 0; i < size; i++)  {

		if (0. == lroundf(crealf(mask[i])))
			continue;

		long lpos[N];
		md_unravel_index(N, lpos, flags, dims, i);

		for (int j = 0; j < M; j++)
			pos[count][j] = lpos[j];

		count++;
	}

	return count;
}

static egraph_t pos_to_graph(int N, vec3_t pos[N])
{
	egraph_t nodes = egraph_create();
	for (long i = 0; i < N; i++) {

		enode_t node = enode_create(NULL, pos[i]);
		egraph_add_node(nodes, node);

		for (long j = 0; j < i; j++) {
			if (con26(pos[j], pos[i])) {

				enode_add_dependency(list_get_item(nodes, j), list_get_item(nodes, i));
				enode_add_dependency(list_get_item(nodes, i), list_get_item(nodes, j));
			}
		}
	}

	return nodes;
}

static void lspline_to_segments(int N, vec3_t seg[N - 1][2], const vec3_t pos[N])
{
	assert(1 <= N);

	for (int i = 0; i < N - 1; i++) {

		vec3_copy(seg[i][0], pos[i]);
		vec3_copy(seg[i][1], pos[i + 1]);
	}
}

static void grid_to_fov(const long dims[3], int N, vec3_t pos[N])
{
	for (int i = 0; i < N; i++) {

		pos[i][0] = (pos[i][0] - dims[0] / 2) / (float)dims[0];
		pos[i][1] = (pos[i][1] - dims[1] / 2) / (float)dims[1];
		pos[i][2] = (pos[i][2] - dims[2] / 2) / (float)dims[2];
	}
}


static void md_mark_endpoints(int N, const long dims[N], complex float* mask)
{
	complex float* tmask = md_alloc(N, dims, CFL_SIZE);
	md_copy(N, dims, tmask, mask, CFL_SIZE);

	long count = md_mask_count(N, dims, tmask);

	vec3_t (*pos)[count?:1] = xmalloc(sizeof(vec3_t[count?:1]));
	md_mask_to_pos(N, dims, 7UL, count, 3, (*pos), tmask);

	egraph_t nodes = pos_to_graph(count, (*pos));
	list_t components = egraph_split_connected_components(nodes);

	complex float* keep = md_alloc(N, dims, CFL_SIZE);
	md_clear(N, dims, keep, CFL_SIZE);

	while (0 < list_count(components)) {

		egraph_t comp = list_pop(components);

		enode_t rand = list_get_item(comp, 0);
		enode_t start = egraph_find_most_distant(comp, rand);
		enode_t end = egraph_find_most_distant(comp, start);

		long pos[3];
		for (int j = 0; j < 3; j++)
			pos[j] = lroundf(((float*)enode_get_data(start))[j]);

		long idx = md_ravel_index(3, pos, ~0UL, dims);
		keep[idx] = 1.;

		for (int j = 0; j < 3; j++)
			pos[j] = lroundf(((float*)enode_get_data(end))[j]);

		idx = md_ravel_index(3, pos, ~0UL, dims);
		keep[idx] = 1.;

		egraph_free(comp);
	}

	list_free(components);
	md_free(tmask);

	md_copy(N, dims, mask, keep, CFL_SIZE);
	md_free(keep);
}

struct lseg_s md_trace_binary_mask(int N, const long dims[N], complex float* mask, float tol)
{
	complex float* tmask = md_alloc(N, dims, CFL_SIZE);

	complex float* keep = md_alloc(N, dims, CFL_SIZE);
	md_copy(N, dims, keep, mask, CFL_SIZE);
	md_mark_endpoints(N, dims, keep);

	md_thinning_3D(N, dims, tmask, mask, keep);
	md_free(keep);

	long count = md_mask_count(N, dims, tmask);

	vec3_t (*pos)[count?:1] = xmalloc(sizeof(vec3_t[count?:1]));
	md_mask_to_pos(N, dims, 7UL, count, 3, (*pos), tmask);

	md_free(tmask);

	if (0 == count) {

		xfree(pos);

		debug_printf(DP_DEBUG1, "No points in the mask.\n");
		struct lseg_s ret = { .N = 0, .pos = NULL };
		return ret;
	}

	debug_printf(DP_DEBUG1, "Found %ld points in the mask.\n", count);

	list_t nodes = pos_to_graph(count, (*pos));
	list_t components = egraph_split_connected_components(nodes);

	vec3_t (*seg)[count?:1][2] = xmalloc(sizeof(vec3_t[count?:1][2]));

	long count2 = 0;

	while (0 != list_count(components)) {

		egraph_t nodes = list_pop(components);

		if (3 > list_count(nodes)) {

			debug_printf(DP_DEBUG1, "Skipping path with %d elements\n", list_count(nodes));
			egraph_free(nodes);

			continue;
		}

		list_t end_nodes = list_create();

		for (long j = 0; j < list_count(nodes); j++)
			if (   (1 == list_count(enode_get_iedges(list_get_item(nodes, j))))
			    || (1 == list_count(enode_get_oedges(list_get_item(nodes, j)))))
				list_append(end_nodes, list_get_item(nodes, j));

		while (2 > list_count(end_nodes)) {

			// 0 == n_end_nodes => circle like o
			// 1 == n_end_nodes => circle like o-

			enode_t node = (1 == list_count(end_nodes)) ? list_pop(end_nodes) : list_get_item(nodes, 0);

			enode_t remove = egraph_find_most_distant(nodes, node);
			enode_free(list_get_first_item(nodes, remove, NULL, true));

			for (long j = 0; j < list_count(nodes); j++)
				if (   (1 == list_count(enode_get_iedges(list_get_item(nodes, j))))
				    || (1 == list_count(enode_get_oedges(list_get_item(nodes, j)))))
					list_append(end_nodes, list_get_item(nodes, j));
		}

		enode_t dst;
		enode_t src;

		egraph_longest_distance(&dst, &src, nodes, end_nodes);

		list_t path = egraph_shortest_path(nodes, dst, src);
		long pathlen = list_count(path);

		vec3_t (*pos2)[pathlen] = xmalloc(sizeof(vec3_t[pathlen]));
		for(int i = 0; i < pathlen; i++) {

			enode_t node = list_get_item(path, i);
			vec3_copy((*pos2)[i], enode_get_data(node));
		}

		list_free(path);

		pathlen = douglas_peucker(pathlen, (*pos2), tol);
		grid_to_fov(dims, pathlen, (*pos2));

		debug_printf(DP_DEBUG1, "Found path with %ld segments\n", pathlen);

		lspline_to_segments(pathlen, (*seg) + count2, (*pos2));

		xfree(pos2);
		count2 += pathlen - 1;
	}

	list_free(components);

	struct lseg_s ret = { .N = 0, .pos = NULL };

	if (0 == count2) {

		xfree(*seg);
		return ret;
	}

	xfree(pos);

	ret.N = count2;
	ret.pos = realloc(*seg, sizeof(vec3_t[count2][2]));
	return ret;
}


void line_segments_revert(int N, vec3_t seg[N][2])
{
	vec3_t tmp;
	for (int i = 0; i < N; i++) {

		vec3_copy(tmp, seg[i][0]);
		vec3_copy(seg[i][0], seg[i][1]);
		vec3_copy(seg[i][1], tmp);
	}
}

static void line_segments_connections(int N, int idx_pre[N], int idx_post[N], int id[N], vec3_t seg[N][2])
{
	for (int i = 0; i < N; i++) {

		idx_pre[i] = -1;
		idx_post[i] = -1;
		id[i] = -1;
	}

	for (int i = 0; i < N; i++) {

		for (int j = 0; j < N; j++) {

			if (i == j)
				continue;

			if (vec3_dist(seg[i][1], seg[j][0]) < 1.e-6)
				idx_post[i] = j;

			if (vec3_dist(seg[i][0], seg[j][1]) < 1.e-6)
				idx_pre[i] = j;
		}
	}

	int count = 0;
	for (int i = 0; i < N; i++) {

		if (-1 != id[i])
			continue;

		id[i] = count++;

		int j = i;
		while ((-1 != idx_post[j]) && (-1 == id[idx_post[j]])) {

			id[idx_post[j]] = id[j];
			j = idx_post[j];
		}

		j = i;
		while ((-1 != idx_pre[j]) && (-1 == id[idx_pre[j]])) {

			id[idx_pre[j]] = id[j];
			j = idx_pre[j];
		}
	}
}

static void reorder_seg(int N, int ord[N], vec3_t seg[N][2])
{
	vec3_t tseg[N][2];

	for (int i = 0; i < N; i++) {

		vec3_copy(tseg[i][0], seg[ord[i]][0]);
		vec3_copy(tseg[i][1], seg[ord[i]][1]);
	}

	for (int i = 0; i < N; i++) {

		vec3_copy(seg[i][0], tseg[i][0]);
		vec3_copy(seg[i][1], tseg[i][1]);
	}
}

void line_segments_sort(struct lseg_s* seg)
{
	int N = seg->N;

	int idx_pre[N];
	int idx_post[N];
	int idx[N];

	line_segments_connections(N, idx_pre, idx_post, idx, seg->pos);

	__block const int* idx_p = idx;

	int ord[N];

	for (int i = 0; i < N; i++)
		ord[i] = i;

	NESTED(int, cmp_seg, (int a, int b))
	{
		int da = idx_p[a];
		int db = idx_p[b];

		return (da > db) - (da < db);;
	};

	quicksort(N, ord, cmp_seg);

	reorder_seg(N, ord, seg->pos);

	line_segments_connections(N, idx_pre, idx_post, idx, seg->pos);

	int i = 0;
	int start = 0;
	int Nseg = 1;

	while (i < N) {

		while (i + Nseg < N) {

			if (idx[i] == idx[i + Nseg]) {

				if (-1 == idx_pre[i + Nseg])
					start = i + Nseg;

				Nseg++;
			} else {

				break;
			}
		}

		ord[i] = start;

		for (int j = i + 1; j < i + Nseg; j++)
			ord[j] = idx_post[ord[j - 1]];

		i += Nseg;
		start = i;
		Nseg = 1;
	}

	reorder_seg(N, ord, seg->pos);
}

static float dist_to_bounds(const vec3_t pos)
{
	float dist = 1.;

	for (int i = 0; i < 3; i++) {

		assert(fabsf(pos[i]) <= 0.5);
		dist = MIN(dist, 0.5 - fabsf(pos[i]));
	}

	return dist;
}

static void closest_bounds(vec3_t bound, const vec3_t pos)
{
	float dist = dist_to_bounds(pos);

	for (int i = 0; i < 3; i++) {

		if (dist == (0.5 - fabsf(pos[i]))) {

			bound[i] = pos[i] < 0. ? -0.5 : 0.5;

		} else {

			bound[i] = pos[i];
		}
	}
}

static bool in_bounds(const vec3_t pos)
{
	for (int i = 0; i < 3; i++)
		if (fabsf(pos[i]) > 0.5)
			return false;

	return true;
}

static void lseg_add(struct lseg_s* seg, vec3_t nseg[2])
{
	if (0. == vec3_dist(nseg[0], nseg[1]))
		return;

	seg->N++;
	seg->pos = realloc(seg->pos, sizeof(float[seg->N][2][3]));

	vec3_copy(seg->pos[seg->N - 1][0], nseg[0]);
	vec3_copy(seg->pos[seg->N - 1][1], nseg[1]);

}


void line_segments_connect(struct lseg_s* seg)
{
	line_segments_sort(seg);

	int N = seg->N;

	int idx_pre[N];
	int idx_post[N];
	int idx[N];

	line_segments_connections(N, idx_pre, idx_post, idx, seg->pos);

	for (int i = 0; i < N; i++) {

		if (-1 != idx_post[i])
			continue;

		if (!in_bounds(seg->pos[i][1]))
			continue;

		int n = -1;

		float dist = -1.;

		for (int j = 0; j < N; j++) {

			if (i == j)
				continue;

			if (-1 != idx_pre[j])
				continue;

			if (!in_bounds(seg->pos[j][0]))
				continue;

			float d = vec3_dist(seg->pos[i][1], seg->pos[j][0]);

			if (dist_to_bounds(seg->pos[i][1]) < d || dist_to_bounds(seg->pos[j][0]) < d)
				continue;

			if ((-1. == dist || d < dist)) {

				dist = d;
				n = j;
			}
		}

		vec3_t nseg[2];
		vec3_copy(nseg[0], seg->pos[i][1]);

		if (-1 != n)
			vec3_copy(nseg[1], seg->pos[n][0]);
		else
			closest_bounds(nseg[1], seg->pos[i][1]);

		lseg_add(seg, nseg);
	}

	N = seg->N;

	int idx_pre2[N];
	int idx_post2[N];
	int idx2[N];

	line_segments_connections(N, idx_pre2, idx_post2, idx2, seg->pos);

	for (int i = 0; i < N; i++) {

		if (-1 != idx_pre2[i])
			continue;

		vec3_t nseg[2];
		vec3_copy(nseg[1], seg->pos[i][0]);
		closest_bounds(nseg[0], nseg[1]);

		lseg_add(seg, nseg);
	}

	line_segments_sort(seg);
}

void line_segments_extend_bounds(struct lseg_s* seg)
{
	int N = seg->N;

	int idx_pre[N];
	int idx_post[N];
	int idx[N];

	line_segments_connections(N, idx_pre, idx_post, idx, seg->pos);

	for (int i = 0; i < N; i++) {

		if (-1 != idx_post[i])
			continue;

		vec3_t nseg[2];
		vec3_copy(nseg[0], seg->pos[i][1]);

		for (int j = 0; j < 3; j++)
			nseg[1][j] = (0.5 == fabsf(nseg[0][j])) ? nseg[0][j] * 100 : nseg[0][j];

		lseg_add(seg, nseg);
	}

	for (int i = 0; i < N; i++) {

		if (-1 != idx_pre[i])
			continue;

		vec3_t nseg[2];
		vec3_copy(nseg[1], seg->pos[i][0]);

		for (int j = 0; j < 3; j++)
			nseg[0][j] = (0.5 == fabsf(nseg[1][j])) ? nseg[1][j] * 100 : nseg[1][j];

		lseg_add(seg, nseg);
	}

	line_segments_sort(seg);
}

