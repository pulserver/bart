
#include "misc/mri.h"

extern void walsh(const long bsize[3], const long dims[DIMS], _Complex float* sens, const long caldims[DIMS], const _Complex float* data);
extern void phase_normalization(const long bsize[3], const long dims[DIMS], _Complex float* sens, const long caldims[DIMS], const _Complex float* data);
