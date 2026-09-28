
#include <stdlib.h>

#include "misc/cppwrap.h"

extern void md_shuffle2(int N, const bart_dim_t dims[__VLA(N)], const bart_dim_t factors[__VLA(N)],
		const bart_stride_t ostrs[__VLA(N)], void* out, const bart_stride_t istrs[__VLA(N)], const void* in, size_t size);

extern void md_shuffle(int N, const bart_dim_t dims[__VLA(N)], const bart_dim_t factors[__VLA(N)],
		void* out, const void* in, size_t size);

extern void md_decompose2(int N, const bart_dim_t factors[__VLA(N)],
		const bart_dim_t odims[__VLA(N + 1)], const bart_stride_t ostrs[__VLA(N + 1)], void* out,
		const bart_dim_t idims[__VLA(N)], const bart_stride_t istrs[__VLA(N)], const void* in, size_t size);

extern void md_decompose(int N, const bart_dim_t factors[__VLA(N)], const bart_dim_t odims[__VLA(N + 1)],
		void* out, const bart_dim_t idims[__VLA(N)], const void* in, size_t size);

extern void md_recompose2(int N, const bart_dim_t factors[__VLA(N)],
		const bart_dim_t odims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], void* out,
		const bart_dim_t idims[__VLA(N + 1)], const bart_stride_t istrs[__VLA(N + 1)], const void* in, size_t size);

extern void md_recompose(int N, const bart_dim_t factors[__VLA(N)], const bart_dim_t odims[__VLA(N)],
		void* out, const bart_dim_t idims[__VLA(N + 1)], const void* in, size_t size);

#include "misc/cppwrap.h"
