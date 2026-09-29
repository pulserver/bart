
#include "misc/dimtypes.h"
#include "misc/cppwrap.h"

extern void casorati_dims(int N, bart_dim_t odim[2], const bart_dim_t dimk[__VLA(N)], const bart_dim_t dims[__VLA(N)]);
extern void casorati_matrix(int N, const bart_dim_t dimk[__VLA(N)], const bart_dim_t odim[2], _Complex float* optr, const bart_dim_t dim[__VLA(N)], const bart_stride_t str[__VLA(N)], const _Complex float* iptr);
extern void casorati_matrixH(int N, const bart_dim_t dimk[__VLA(N)], const bart_dim_t dim[__VLA(N)], const bart_stride_t str[__VLA(N)], _Complex float* optr, const bart_dim_t odim[2], const _Complex float* iptr);


extern void basorati_dims(int N, bart_dim_t odim[2], const bart_dim_t dimk[__VLA(N)], const bart_dim_t dims[__VLA(N)]);
extern void basorati_matrix(int N, const bart_dim_t dimk[__VLA(N)], const bart_dim_t odim[2], _Complex float* optr, const bart_dim_t dim[__VLA(N)], const bart_stride_t str[__VLA(N)], const _Complex float* iptr);
extern void basorati_matrixH(int N, const bart_dim_t dimk[__VLA(N)], const bart_dim_t dim[__VLA(N)], const bart_stride_t str[__VLA(N)], _Complex float* optr, const bart_dim_t odim[2], const _Complex float* iptr);

#include "misc/cppwrap.h"
