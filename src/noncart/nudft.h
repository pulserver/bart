
#include "misc/dimtypes.h"
#include <complex.h>

extern void nudft_forward2(int N, bart_flags_t flags,
			const bart_dim_t odims[N], const bart_stride_t ostrs[N], complex float* out,
			const bart_dim_t idims[N], const bart_stride_t istrs[N], const complex float* in,
			const bart_dim_t tdims[N], const bart_stride_t tstrs[N], const complex float* traj,
			const complex float* fieldmap,
			const bart_stride_t tmstrs[N], const complex float* timemap);

extern void nudft_forward(int N, bart_flags_t flags,
			const bart_dim_t odims[N], complex float* out,
			const bart_dim_t idims[N], const complex float* in,
			const bart_dim_t tdims[N], const complex float* traj,
			const complex float* fieldmap,
			const bart_dim_t tmdims[N], const complex float* timemap);

extern void nudft_adjoint2(int N, bart_flags_t flags,
			const bart_dim_t odims[N], const bart_stride_t ostrs[N], complex float* out,
			const bart_dim_t idims[N], const bart_stride_t istrs[N], const complex float* in,
			const bart_dim_t tdims[N], const bart_stride_t tstrs[N], const complex float* traj,
			const complex float* fieldmap,
			const bart_stride_t tmstrs[N], const complex float* timemap);

extern void nudft_adjoint(int N, bart_flags_t flags,
			const bart_dim_t odims[N], complex float* out,
			const bart_dim_t idims[N], const complex float* in,
			const bart_dim_t tdims[N], const complex float* traj,
			const complex float* fieldmap,
			const bart_dim_t tmdims[N], const complex float* timemap);

struct linop_s;
extern struct linop_s* nudft_create2(int N, bart_flags_t flags,
					const bart_dim_t odims[N], const bart_stride_t ostrs[N],
					const bart_dim_t idims[N], const bart_stride_t istrs[N],
					const bart_dim_t tdims[N], const complex float* traj,
					const bart_dim_t fmdims[N], const complex float* fieldmap,
					const bart_dim_t tmdims[N], const complex float* timemap);

extern struct linop_s* nudft_create(int N, bart_flags_t flags,
					const bart_dim_t odims[N], const bart_dim_t idims[N],
					const bart_dim_t tdims[N], const complex float* traj,
					const bart_dim_t fmdims[N], const complex float* fieldmap,
					const bart_dim_t tmdims[N], const complex float* timemap);

