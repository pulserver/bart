#include "misc/dimtypes.h"



extern void compose_displacement(int N, int d, bart_flags_t flags, const bart_dim_t dims[__VLA(N)], _Complex float* composed, const _Complex float* d1, const _Complex float* d2);
extern void invert_displacement(int N, int d, bart_flags_t flags, const bart_dim_t dims[__VLA(N)], _Complex float* inv_disp, const _Complex float* disp);

struct linop_s;
extern const struct linop_s* linop_interpolate_displacement_create(int d, bart_flags_t flags, int ord, int N, const bart_dim_t idims[__VLA(N)], const bart_dim_t mdims[__VLA(N)], const _Complex float* motion, const bart_dim_t gdims[__VLA(N)]);


