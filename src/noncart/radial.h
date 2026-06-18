
extern void traj_radial_angles(int N, const long adims[__VLA(N)], float* angles, const long tdims[__VLA(N)], const _Complex float* traj);
extern void traj_radial_dcshifts(int N, const long sdims[__VLA(N)], _Complex float* shifts, const long tdims[__VLA(N)], const _Complex float* traj);

extern float traj_radial_dcshift(int N, const long tdims[__VLA(N)], const _Complex float* traj);
extern float traj_radial_deltak(int N, const long tdims[__VLA(N)], const _Complex float* traj);

extern void traj_radial_direction(int N, const long ddims[__VLA(N)], _Complex float* dir, const long tdims[__VLA(N)], const _Complex float* traj);

extern bool traj_radial_same_dk(int N, const long tdims[__VLA(N)], const _Complex float* traj);
extern bool traj_radial_through_center(int N, const long tdims[__VLA(N)], const _Complex float* traj);
extern bool traj_is_radial(int N, const long tdims[__VLA(N)], const _Complex float* traj);

