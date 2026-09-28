#include "misc/dimtypes.h"
#include "misc/cppwrap.h"

void cuda_positions(int N, int d, bart_flags_t flags, const bart_dim_t sdims[__VLA(N)], const bart_dim_t pdims[__VLA(N)], _Complex float* pos);

void cuda_interpolate2(int ord, int M, 
			const bart_dim_t intp_dims[__VLA(M)], const bart_stride_t intp_strs[__VLA(M)], _Complex float* intp,
							const bart_stride_t coor_strs[__VLA(M)], bart_stride_t coor_dir_dim_str, const _Complex float* coor,
			const bart_dim_t grid_dims[__VLA(M)], const bart_stride_t grid_strs[__VLA(M)], const _Complex float* grid);

void cuda_interpolateH2(int ord, int M, 
			const bart_dim_t grid_dims[__VLA(M)], const bart_stride_t grid_strs[__VLA(M)], _Complex float* grid,
			const bart_dim_t intp_dims[__VLA(M)], const bart_stride_t intp_strs[__VLA(M)], const _Complex float* intp,
							const bart_stride_t coor_strs[__VLA(M)], bart_stride_t coor_dir_dim_str, const _Complex float* coor);

void cuda_interpolate_adj_coor2(int ord, int M, 
			const bart_dim_t intp_dims[__VLA(M)], const bart_stride_t intp_strs[__VLA(M)], const _Complex float* dintp,
							const bart_stride_t coor_strs[__VLA(M)], bart_stride_t coor_dir_dim_str, const _Complex float* coor, _Complex float* dcoor,
			const bart_dim_t grid_dims[__VLA(M)], const bart_stride_t grid_strs[__VLA(M)], const _Complex float* grid);

void cuda_interpolate_der_coor2(int ord, int M, 
			const bart_dim_t intp_dims[__VLA(M)], const bart_stride_t intp_strs[__VLA(M)], _Complex float* dintp,
							const bart_stride_t coor_strs[__VLA(M)], bart_stride_t coor_dir_dim_str, const _Complex float* coor, const _Complex float* dcoor,
			const bart_dim_t grid_dims[__VLA(M)], const bart_stride_t grid_strs[__VLA(M)], const _Complex float* grid);

#include "misc/cppwrap.h"

