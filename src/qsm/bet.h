#include <stdbool.h>
#include "stl/misc.h"

struct Point2 {
    float x;
    float y;
};

struct Segment2{
	struct Point2 a;
	struct Point2 b;
};

extern void threshold(int N, long dims[N], float* img, float* new_img, float* t, float* t98, float* t2);
extern void compute_cog(int N, long dims[N], const float* img, const float res[3], float* t, float* t98, float COG[3], float* R_out);
extern float compute_tm(int N, long dims[N], const float* image, const float voxel_size[3], const float COG[3], float R);
extern void run_bet(int N, long dims[N], double* verts, int nv, const struct neighbors* neigh,
	const float* image, const float voxel_size[3], const float COG[3], float t2, float t, float tm, float bt, int n_iter);
extern void mesh_to_mask_slicewise(int N, long dims[N], float* mask, float resolution[3], const double (*verts)[3], const int (*tris)[3], int ntris);
extern void mesh_to_mask_winding_number(int N, long dims[N], float* mask, float resolution[3], const double (*verts)[3], const int (*tris)[3], int ntris);
