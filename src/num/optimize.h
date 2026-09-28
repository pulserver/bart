
#include "misc/dimtypes.h"
#include <stdlib.h>

extern void merge_dims(int D, int N, bart_dim_t dims[N], bart_stride_t (*ostrs[D])[N]);
extern int remove_empty_dims(int D, int N, bart_dim_t dims[N], bart_stride_t (*ostrs[D])[N]);

extern int simplify_dims(int D, int N, bart_dim_t dims[N], bart_stride_t (*strs[D])[N]);
extern int optimize_dims(int D, int N, bart_dim_t dims[N], bart_stride_t (*strs[D])[N]);
extern int optimize_dims_gpu(int D, int N, bart_dim_t dims[N], bart_stride_t (*strs[D])[N]);
extern int min_blockdim(int D, int N, const bart_dim_t dims[N], bart_stride_t (*strs[D])[N], size_t size[D]);
extern bart_flags_t dims_parallel(int D, bart_flags_t io, int N, const bart_dim_t dims[N], bart_stride_t (*strs[D])[N], size_t size[D]);
extern bart_flags_t parallelizable(int D, unsigned int io, int N, const bart_dim_t dims[N], const bart_stride_t (*strs[D])[N], size_t size[D]);

struct vec_ops;

struct nary_opt_data_s {

	bart_dim_t size;
	const struct vec_ops* ops;
};



typedef void CLOSURE_TYPE(md_nary_opt_fun_t)(struct nary_opt_data_s* data, void* ptr[]);

extern void optimized_nop(int N, bart_flags_t io, int D, const bart_dim_t dim[D], const bart_stride_t (*nstr[N])[D?:1], void* const nptr[N], size_t sizes[N], md_nary_opt_fun_t too);

