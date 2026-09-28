#include "misc/dimtypes.h"
#include "misc/cppwrap.h"

struct linop_s;
struct grid_conf_s;

extern struct linop_s* linop_kb_rolloff_create(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t flags, struct grid_conf_s* conf);
extern struct linop_s* linop_interpolate_kb_create(int N, bart_flags_t flags, const bart_dim_t ksp_dims[__VLA(N)], const bart_dim_t grd_dims[__VLA(N)], const bart_dim_t trj_dims[__VLA(N)], const _Complex float* traj, struct grid_conf_s* conf);

extern struct linop_s* nufft_create_chain(int N,
			     const bart_dim_t ksp_dims[N],
			     const bart_dim_t cim_dims[N],
			     const bart_dim_t traj_dims[N],
			     const _Complex float* traj,
			     const bart_dim_t wgh_dims[N],
			     const _Complex float* weights,
			     const bart_dim_t bas_dims[N],
			     const _Complex float* basis,
			     struct grid_conf_s* conf);

#include "misc/cppwrap.h"

