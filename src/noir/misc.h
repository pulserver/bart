#include "misc/dimtypes.h"


#ifndef DIMS
#define DIMS 16
#endif

extern void scale_psf_k(const bart_dim_t pat_dims[DIMS],
			_Complex float* pattern,
			const bart_dim_t ksp_dims[DIMS],
			_Complex float* kspace_data,
			const bart_dim_t trj_dims[DIMS],
			_Complex float* traj);

extern void postprocess(const bart_dim_t dims[DIMS], bool normalize,
			const bart_stride_t sens_strs[DIMS], const complex float* sens,
			const bart_stride_t img_strs[DIMS], const complex float* img,
			const bart_dim_t img_output_dims[DIMS], const bart_stride_t img_output_strs[DIMS], complex float* img_output);

extern void postprocess2(bool normalize,
		  const bart_dim_t sens_dims[DIMS], const complex float* sens,
		  const bart_dim_t img_dims[DIMS], const complex float* img,
		  const bart_dim_t img_output_dims[DIMS], complex float* img_output);

