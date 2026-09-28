void cudnn_init(void);
void cudnn_deinit(void);

extern _Bool zconvcorr_fwd_cudnn(	int N,
					bart_dim_t odims[N], bart_stride_t ostrs[N], _Complex float* out,
					bart_dim_t idims[N], bart_stride_t istrs[N], const _Complex float* in,
					bart_dim_t kdims[N], bart_stride_t kstrs[N], const _Complex float* krn,
					bart_flags_t flags, const bart_dim_t dilation[N], const bart_stride_t strides[N], _Bool conv);

extern _Bool zconvcorr_bwd_in_cudnn(	int N,
					bart_dim_t odims[N], bart_stride_t ostrs[N], const _Complex float* out,
					bart_dim_t idims[N], bart_stride_t istrs[N], _Complex float* in,
					bart_dim_t kdims[N], bart_stride_t kstrs[N], const _Complex float* krn,
					bart_flags_t flags, const bart_dim_t dilation[N], const bart_stride_t strides[N], _Bool conv);

extern _Bool zconvcorr_bwd_krn_cudnn(	int N,
					bart_dim_t odims[N], bart_stride_t ostrs[N], const _Complex float* out,
					bart_dim_t idims[N], bart_stride_t istrs[N], const _Complex float* in,
					bart_dim_t kdims[N], bart_stride_t kstrs[N], _Complex float* krn,
					bart_flags_t flags, const bart_dim_t dilation[N], const bart_stride_t strides[N], _Bool conv);