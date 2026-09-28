
#ifndef _MRI2_H
#define _MRI2_H

#include "misc/cppwrap.h"

#ifndef DIMS
#define DIMS		16
#endif

extern void estimate_pattern(int D, const bart_dim_t dims[__VLA(D)], bart_flags_t flags, _Complex float* pattern, const _Complex float* kspace_data);
extern _Complex float* extract_calib(bart_dim_t caldims[DIMS], const bart_dim_t calsize[3], const bart_dim_t in_dims[DIMS], const _Complex float* in_data, _Bool fixed);
extern _Complex float* extract_calib2(bart_dim_t caldims[DIMS], const bart_dim_t calsize[3], const bart_dim_t in_dims[DIMS], const bart_stride_t in_strs[DIMS], const _Complex float* in_data, _Bool fixed);
extern void data_consistency(const bart_dim_t dims[DIMS], _Complex float* dst, const _Complex float* pattern, const _Complex float* kspace1, const _Complex float* kspace2);
extern void calib_geom(bart_dim_t caldims[DIMS], bart_dim_t calpos[DIMS], const bart_dim_t calsize[3], const bart_dim_t in_dims[DIMS], const _Complex float* in_data);

extern void estimate_im_dims(int N, bart_flags_t flags, bart_dim_t dims[__VLA(N)], const bart_dim_t tdims[__VLA(N)], const _Complex float* traj);
extern void estimate_fast_sq_im_dims(int N, bart_dim_t dims[3], const bart_dim_t tdims[__VLA(N)], const _Complex float* traj);

#include "misc/cppwrap.h"

#endif	// _MRI2_H

