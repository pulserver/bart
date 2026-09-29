
#include "misc/dimtypes.h"
#include <complex.h>

#include "misc/mri.h"


extern void ssa_fary(	const bart_dim_t kernel_dims[3],
			const bart_dim_t cal_dims[DIMS],
			const bart_dim_t A_dims[2],
			const complex float* A,
			complex float* U,
			float* S_square,
			complex float* back,
			const int rank,
			const bart_dim_t group);

extern void ssa_fary_econ(	const bart_dim_t kernel_dims[3],
				const bart_dim_t cal_dims[DIMS],
				const bart_dim_t A_dims[2],
				complex float* A,
				complex float* U,
				float* S_square,
				complex float* back,
				const int rank,
				const bart_dim_t group);
