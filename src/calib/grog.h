#include "misc/dimtypes.h"


extern void grog_calib(int D, const bart_dim_t lnG_dims[D], complex float* lnG, const bart_dim_t tdims[D], const complex float* traj, const bart_dim_t ddims[D], const complex float* data);

extern void grog_grid(int D, const bart_dim_t tdims[D], const complex float* traj_shift, const bart_dim_t ddims[D], complex float* data_grid, const complex float* data, const bart_dim_t lnG_dims[D], complex float* lnG);

