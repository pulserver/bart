
#include "misc/dimtypes.h"
#include "misc/mri.h"

extern void walsh(const bart_dim_t bsize[3], const bart_dim_t dims[DIMS], _Complex float* sens, const bart_dim_t caldims[DIMS], const _Complex float* data);
extern void phase_normalization(const bart_dim_t bsize[3], const bart_dim_t dims[DIMS], _Complex float* sens, const bart_dim_t caldims[DIMS], const _Complex float* data);
