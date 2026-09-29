
#ifndef LINSEG_H
#define LINSEG_H
#include "misc/dimtypes.h"

struct lseg_s {

	int N;
	float (*pos)[2][3];
};

extern float dist_to_lineseg(const float pos[3], const float seg[2][3]);
extern float dist_of_linesegs(const float seg1[2][3], const float seg2[2][3]);

extern _Bool dist_of_linesegs_smaller(const float seg1[2][3], const float seg2[2][3], float tol);

extern bart_dim_t douglas_peucker(bart_dim_t N, float pos[N][3], float tol);


extern struct lseg_s md_trace_binary_mask(int N, const bart_dim_t dims[N], _Complex float* mask, float tol);

extern void line_segments_revert(int N, float seg[N][2][3]);
extern void line_segments_sort(struct lseg_s* seg);
extern void line_segments_connect(struct lseg_s* seg);
extern void line_segments_extend_bounds(struct lseg_s* seg);

#endif // LINSEG_H
