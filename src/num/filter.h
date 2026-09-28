
#include <complex.h>

#include "misc/cppwrap.h"

extern float median_float(int N, const float ar[N]);
extern complex float median_complex_float(int N, const complex float ar[N]);
extern void weiszfeld(int iter, int N, int D, float out[D], const float in[N][D]);


extern void md_medianz2(int D, int M, const bart_dim_t dim[D], const bart_stride_t ostr[D], complex float* optr, const bart_stride_t istr[D], const complex float* iptr);
extern void md_medianz(int D, int M, const bart_dim_t dim[D], complex float* optr, const complex float* iptr);

extern void md_geometric_medianz2(int D, int M, const bart_dim_t dim[D], const bart_stride_t ostr[D], complex float* optr, const bart_stride_t istr[D], const complex float* iptr);
extern void md_geometric_medianz(int D, int M, const bart_dim_t dim[D], complex float* optr, const complex float* iptr);

extern void md_moving_avgz2(int D, int M, const bart_dim_t dim[D], const bart_stride_t ostr[D], complex float* optr, const bart_stride_t istr[D], const complex float* iptr);
extern void md_moving_avgz(int D, int M, const bart_dim_t dim[D], complex float* optr, const complex float* iptr);

extern void linear_phase(int N, const bart_dim_t dims[__VLA(N)], const float pos[__VLA(N)], _Complex float* out);
extern void centered_gradient(int N, const bart_dim_t dims[__VLA(N)], const _Complex float grad[__VLA(N)], _Complex float* out);
extern void klaplace(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t flags, _Complex float* out);
void klaplace_scaled(int N, const bart_dim_t dims[N], bart_flags_t flags, const float sc[N], complex float* out);

extern void md_zhamming(int D, const bart_dim_t dims[__VLA(D)], const bart_flags_t flags, complex float* optr, const complex float* iptr);
extern void md_zhamming2(int D, const bart_dim_t dims[__VLA(D)], const bart_flags_t flags, const bart_stride_t ostr[__VLA(D)], complex float* optr, const bart_stride_t istr[__VLA(D)], const complex float* iptr);

extern void md_zhann(int D, const bart_dim_t dims[__VLA(D)], const bart_flags_t flags, complex float* optr, const complex float* iptr);
extern void md_zhann2(int D, const bart_dim_t dims[__VLA(D)], const bart_flags_t flags, const bart_stride_t ostr[__VLA(D)], complex float* optr, const bart_stride_t istr[__VLA(D)], const complex float* iptr);

#include "misc/cppwrap.h"

