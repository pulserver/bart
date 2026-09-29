extern bool simple_zconvcorr(	int N, const long dims[__VLA(N)],
				const long ostrs[__VLA(N)], _Complex float* optr,
				const long istrs1[__VLA(N)], const _Complex float* iptr1,
				const long istrs2[__VLA(N)], const _Complex float* iptr2);

typedef bool zconvcorr_fwd_algo_f(	int N,
					long odims[__VLA(N)], long ostrs[__VLA(N)], _Complex float* out,
					long idims[__VLA(N)], long istrs[__VLA(N)], const _Complex float* in,
					long kdims[__VLA(N)], long kstrs[__VLA(N)], const _Complex float* krn,
					unsigned long flags, const long dilation[__VLA2(N)], const long strides[__VLA2(N)], bool conv);
typedef bool zconvcorr_bwd_in_algo_f(	int N,
					long odims[__VLA(N)], long ostrs[__VLA(N)], const _Complex float* out,
					long idims[__VLA(N)], long istrs[__VLA(N)], _Complex float* in,
					long kdims[__VLA(N)], long kstrs[__VLA(N)], const _Complex float* krn,
					unsigned long flags, const long dilation[__VLA2(N)], const long strides[__VLA2(N)], bool conv);
typedef bool zconvcorr_bwd_krn_algo_f(	int N,
					long odims[__VLA(N)], long ostrs[__VLA(N)], const _Complex float* out,
					long idims[__VLA(N)], long istrs[__VLA(N)], const _Complex float* in,
					long kdims[__VLA(N)], long kstrs[__VLA(N)], _Complex float* krn,
					unsigned long flags, const long dilation[__VLA2(N)], const long strides[__VLA2(N)], bool conv);

zconvcorr_fwd_algo_f zconvcorr_fwd_im2col_cf_cpu;
zconvcorr_bwd_in_algo_f zconvcorr_bwd_in_im2col_cf_cpu;
zconvcorr_bwd_krn_algo_f zconvcorr_bwd_krn_im2col_cf_cpu;

zconvcorr_fwd_algo_f zconvcorr_fwd_im2col_cf_gpu;
zconvcorr_bwd_in_algo_f zconvcorr_bwd_in_im2col_cf_gpu;
zconvcorr_bwd_krn_algo_f zconvcorr_bwd_krn_im2col_cf_gpu;

bool test_zconvcorr_fwd(	int N,
				long odims[__VLA(N)], long ostrs[__VLA(N)],
				long idims[__VLA(N)], long istrs[__VLA(N)],
				long kdims[__VLA(N)], long kstrs[__VLA(N)],
				unsigned long flags, const long dilation[__VLA2(N)], const long strides[__VLA2(N)], bool conv,
				float max_rmse, bool gpu, long min_no_algos);
bool test_zconvcorr_bwd_in(	int N,
				long odims[__VLA(N)], long ostrs[__VLA(N)],
				long idims[__VLA(N)], long istrs[__VLA(N)],
				long kdims[__VLA(N)], long kstrs[__VLA(N)],
				unsigned long flags, const long dilation[__VLA2(N)], const long strides[__VLA2(N)], bool conv,
				float max_rmse, bool gpu, long min_no_algos);
bool test_zconvcorr_bwd_krn(	int N,
				long odims[__VLA(N)], long ostrs[__VLA(N)],
				long idims[__VLA(N)], long istrs[__VLA(N)],
				long kdims[__VLA(N)], long kstrs[__VLA(N)],
				unsigned long flags, const long dilation[__VLA2(N)], const long strides[__VLA2(N)], bool conv,
				float max_rmse, bool gpu, long min_no_algos);

