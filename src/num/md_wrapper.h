void zfmac_gpu_batched_loop(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], _Complex float* optr, const bart_stride_t istr1[__VLA(N)], const _Complex float* iptr1, const bart_stride_t istr2[__VLA(N)], const _Complex float* iptr2);
void zfmacc_gpu_batched_loop(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], _Complex float* optr, const bart_stride_t istr1[__VLA(N)], const _Complex float* iptr1, const bart_stride_t istr2[__VLA(N)], const _Complex float* iptr2);

void add_gpu_unfold(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], float* optr, const bart_stride_t istr1[__VLA(N)], const float* iptr1, const bart_stride_t istr2[__VLA(N)], const float* iptr2);
void zadd_gpu_unfold(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], _Complex float* optr, const bart_stride_t istr1[__VLA(N)], const _Complex float* iptr1, const bart_stride_t istr2[__VLA(N)], const _Complex float* iptr2);

void mul_gpu_unfold(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], float* optr, const bart_stride_t istr1[__VLA(N)], const float* iptr1, const bart_stride_t istr2[__VLA(N)], const float* iptr2);
void zmul_gpu_unfold(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], _Complex float* optr, const bart_stride_t istr1[__VLA(N)], const _Complex float* iptr1, const bart_stride_t istr2[__VLA(N)], const _Complex float* iptr2);
void zmulc_gpu_unfold(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], _Complex float* optr, const bart_stride_t istr1[__VLA(N)], const _Complex float* iptr1, const bart_stride_t istr2[__VLA(N)], const _Complex float* iptr2);

void fmac_gpu_unfold(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], float* optr, const bart_stride_t istr1[__VLA(N)], const float* iptr1, const bart_stride_t istr2[__VLA(N)], const float* iptr2);
void zfmac_gpu_unfold(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], _Complex float* optr, const bart_stride_t istr1[__VLA(N)], const _Complex float* iptr1, const bart_stride_t istr2[__VLA(N)], const _Complex float* iptr2);
void zfmacc_gpu_unfold(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], _Complex float* optr, const bart_stride_t istr1[__VLA(N)], const _Complex float* iptr1, const bart_stride_t istr2[__VLA(N)], const _Complex float* iptr2);

void fmacD_dot(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], double* optr, const bart_stride_t istr1[__VLA(N)], const float* iptr1, const bart_stride_t istr2[__VLA(N)], const float* iptr2);
void zfmaccD_dot(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostr[__VLA(N)], _Complex double* optr, const bart_stride_t istr1[__VLA(N)], const _Complex float* iptr1, const bart_stride_t istr2[__VLA(N)], const _Complex float* iptr2);
