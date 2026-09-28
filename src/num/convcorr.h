#include "misc/dimtypes.h"

extern _Bool simple_zconvcorr(	int N, const bart_dim_t dims[__VLA(N)],
				const bart_stride_t ostrs[__VLA(N)], _Complex float* optr,
				const bart_stride_t istrs1[__VLA(N)], const _Complex float* iptr1,
				const bart_stride_t istrs2[__VLA(N)], const _Complex float* iptr2);

typedef _Bool zconvcorr_fwd_algo_f(	int N,
					bart_dim_t odims[__VLA(N)], bart_stride_t ostrs[__VLA(N)], _Complex float* out,
					bart_dim_t idims[__VLA(N)], bart_stride_t istrs[__VLA(N)], const _Complex float* in,
					bart_dim_t kdims[__VLA(N)], bart_stride_t kstrs[__VLA(N)], const _Complex float* krn,
					bart_flags_t flags, const bart_dim_t dilation[__VLA2(N)], const bart_stride_t strides[__VLA2(N)], _Bool conv);
typedef _Bool zconvcorr_bwd_in_algo_f(	int N,
					bart_dim_t odims[__VLA(N)], bart_stride_t ostrs[__VLA(N)], const _Complex float* out,
					bart_dim_t idims[__VLA(N)], bart_stride_t istrs[__VLA(N)], _Complex float* in,
					bart_dim_t kdims[__VLA(N)], bart_stride_t kstrs[__VLA(N)], const _Complex float* krn,
					bart_flags_t flags, const bart_dim_t dilation[__VLA2(N)], const bart_stride_t strides[__VLA2(N)], _Bool conv);
typedef _Bool zconvcorr_bwd_krn_algo_f(	int N,
					bart_dim_t odims[__VLA(N)], bart_stride_t ostrs[__VLA(N)], const _Complex float* out,
					bart_dim_t idims[__VLA(N)], bart_stride_t istrs[__VLA(N)], const _Complex float* in,
					bart_dim_t kdims[__VLA(N)], bart_stride_t kstrs[__VLA(N)], _Complex float* krn,
					bart_flags_t flags, const bart_dim_t dilation[__VLA2(N)], const bart_stride_t strides[__VLA2(N)], _Bool conv);

zconvcorr_fwd_algo_f zconvcorr_fwd_im2col_cf_cpu;
zconvcorr_bwd_in_algo_f zconvcorr_bwd_in_im2col_cf_cpu;
zconvcorr_bwd_krn_algo_f zconvcorr_bwd_krn_im2col_cf_cpu;

zconvcorr_fwd_algo_f zconvcorr_fwd_im2col_cf_gpu;
zconvcorr_bwd_in_algo_f zconvcorr_bwd_in_im2col_cf_gpu;
zconvcorr_bwd_krn_algo_f zconvcorr_bwd_krn_im2col_cf_gpu;

_Bool test_zconvcorr_fwd(	int N,
				bart_dim_t odims[__VLA(N)], bart_stride_t ostrs[__VLA(N)],
				bart_dim_t idims[__VLA(N)], bart_stride_t istrs[__VLA(N)],
				bart_dim_t kdims[__VLA(N)], bart_stride_t kstrs[__VLA(N)],
				bart_flags_t flags, const bart_dim_t dilation[__VLA2(N)], const bart_stride_t strides[__VLA2(N)], _Bool conv,
				float max_rmse, _Bool gpu, bart_dim_t min_no_algos);
_Bool test_zconvcorr_bwd_in(	int N,
				bart_dim_t odims[__VLA(N)], bart_stride_t ostrs[__VLA(N)],
				bart_dim_t idims[__VLA(N)], bart_stride_t istrs[__VLA(N)],
				bart_dim_t kdims[__VLA(N)], bart_stride_t kstrs[__VLA(N)],
				bart_flags_t flags, const bart_dim_t dilation[__VLA2(N)], const bart_stride_t strides[__VLA2(N)], _Bool conv,
				float max_rmse, _Bool gpu, bart_dim_t min_no_algos);
_Bool test_zconvcorr_bwd_krn(	int N,
				bart_dim_t odims[__VLA(N)], bart_stride_t ostrs[__VLA(N)],
				bart_dim_t idims[__VLA(N)], bart_stride_t istrs[__VLA(N)],
				bart_dim_t kdims[__VLA(N)], bart_stride_t kstrs[__VLA(N)],
				bart_flags_t flags, const bart_dim_t dilation[__VLA2(N)], const bart_stride_t strides[__VLA2(N)], _Bool conv,
				float max_rmse, _Bool gpu, bart_dim_t min_no_algos);
