#include "misc/dimtypes.h"


extern void traj_radial_angles(int N, const bart_dim_t adims[__VLA(N)], float* angles, const bart_dim_t tdims[__VLA(N)], const _Complex float* traj);
extern float traj_radial_dcshift(int N, const bart_dim_t tdims[__VLA(N)], const _Complex float* traj);
extern float traj_radial_deltak(int N, const bart_dim_t tdims[__VLA(N)], const _Complex float* traj);

extern void traj_radial_direction(int N, const bart_dim_t ddims[__VLA(N)], _Complex float* dir, const bart_dim_t tdims[__VLA(N)], const _Complex float* traj);

extern _Bool traj_radial_same_dk(int N, const bart_dim_t tdims[__VLA(N)], const _Complex float* traj);
extern _Bool traj_radial_through_center(int N, const bart_dim_t tdims[__VLA(N)], const _Complex float* traj);
extern _Bool traj_is_radial(int N, const bart_dim_t tdims[__VLA(N)], const _Complex float* traj);
