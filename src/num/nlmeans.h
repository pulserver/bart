
#include "misc/dimtypes.h"
#include <complex.h>

extern void md_znlmeans2(int D, const bart_dim_t dims[D], bart_flags_t flags,
		const bart_stride_t ostrs[D], complex float* optr,
		const bart_stride_t istrs[D], const complex float* iptr,
		bart_dim_t patch_size, bart_dim_t patch_dist, float h, float a);

extern void md_znlmeans(int D, const bart_dim_t dims[D], bart_flags_t flags,
		complex float* optr, const complex float* iptr,
		bart_dim_t patch_size, bart_dim_t patch_dist, float h, float a);

extern void md_znlmeans_distance2(int D, const bart_dim_t idims[D], int xD,
		const bart_dim_t odims[xD], bart_flags_t flags,
		const bart_stride_t ostrs[xD], complex float* optr,
		const bart_stride_t istrs[D], const complex float* iptr);

extern void md_znlmeans_distance(int D, const bart_dim_t idims[D], int xD,
		const bart_dim_t odims[xD], bart_flags_t flags,
		complex float* optr, const complex float* iptr);

