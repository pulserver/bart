#include "misc/dimtypes.h"
#include "misc/cppwrap.h"

#include <stdint.h>
#include <stddef.h>

extern void init_mpi(int* argc, char*** argv);
extern void deinit_mpi(void);
extern void abort_mpi(int err_code);
extern void mpi_signoff_proc(_Bool signof);

extern int mpi_get_rank(void);
extern int mpi_get_num_procs(void);
extern _Bool mpi_is_main_proc(void);

extern void mpi_sync(void);

extern void mpi_sync_val(void* pval, bart_dim_t size);
extern void mpi_bcast(void* ptr, bart_dim_t size, int root);
extern void mpi_bcast_selected(_Bool tag, void* ptr, bart_dim_t size, int root);
extern void mpi_bcast2(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t strs[__VLA(N)], void* ptr, bart_dim_t size, int root);
extern void mpi_copy(void* dst, bart_dim_t size, const void* src, int sender_rank, int recv_rank);
extern void mpi_copy2(int N, const bart_dim_t dim[__VLA(N)], const bart_stride_t ostr[__VLA(N)], void* optr, const bart_stride_t istr[__VLA(N)], const void* iptr, bart_dim_t size, int sender_rank, int recv_rank);

extern void mpi_scatter_batch(void* dst, bart_dim_t count, const void* src, size_t type_size);
extern void mpi_gather_batch(void* dst, bart_dim_t count, const void* src, size_t type_size);

extern void mpi_reduce_sum(int N, const bart_dim_t dims[__VLA(N)], float* optr, float* rptr);
extern void mpi_reduce_zsum(int N, const bart_dim_t dims[__VLA(N)], _Complex float* optr, _Complex float* rptr);
extern void mpi_reduce_sumD(int N, const bart_dim_t dims[__VLA(N)], double* optr, double* rptr);
extern void mpi_reduce_zsumD(int N, const bart_dim_t dims[__VLA(N)], _Complex double* optr, _Complex double* rptr);

extern void* mpi_reduction_sum_buffer_create(const void* ptr);
extern void mpi_reduction_sum_buffer(float* optr, float* rptr);
extern void mpi_reduction_sumD_buffer(double* optr, double* rptr);

extern void  mpi_reduce_sum_vector(bart_dim_t N, float ptr[__VLA(N)]);
extern void  mpi_reduce_zsum_vector(bart_dim_t N, _Complex float ptr[__VLA(N)]);

extern void mpi_reduce_land(bart_dim_t N, _Bool vec[__VLA(N)]);

#include "misc/cppwrap.h"

