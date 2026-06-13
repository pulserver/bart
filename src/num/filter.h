
#include <complex.h>

#include "misc/cppwrap.h"

#include "misc/nested.h"

extern float median_float(int N, const float ar[N]);
extern complex float median_complex_float(int N, const complex float ar[N]);
extern void weiszfeld(int iter, int N, int D, float out[D], const float in[N][D]);


extern void md_medianz2(int D, int M, const long dim[D], const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr);
extern void md_medianz(int D, int M, const long dim[D], complex float* optr, const complex float* iptr);

extern void md_geometric_medianz2(int D, int M, const long dim[D], const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr);
extern void md_geometric_medianz(int D, int M, const long dim[D], complex float* optr, const complex float* iptr);

extern void md_moving_avgz2(int D, int M, const long dim[D], const long ostr[D], complex float* optr, const long istr[D], const complex float* iptr);
extern void md_moving_avgz(int D, int M, const long dim[D], complex float* optr, const complex float* iptr);

extern void linear_phase(int N, const long dims[__VLA(N)], const float pos[__VLA(N)], _Complex float* out);
extern void centered_gradient(int N, const long dims[__VLA(N)], const _Complex float grad[__VLA(N)], _Complex float* out);
extern void klaplace(int N, const long dims[__VLA(N)], unsigned long flags, _Complex float* out);
extern void klaplace_scaled(int N, const long dims[N], unsigned long flags, const float sc[N], complex float* out);

extern void md_zhamming(int D, const long dims[__VLA(D)], const unsigned long flags, complex float* optr, const complex float* iptr);
extern void md_zhamming2(int D, const long dims[__VLA(D)], const unsigned long flags, const long ostr[__VLA(D)], complex float* optr, const long istr[__VLA(D)], const complex float* iptr);

extern void md_zhann(int D, const long dims[__VLA(D)], const unsigned long flags, complex float* optr, const complex float* iptr);
extern void md_zhann2(int D, const long dims[__VLA(D)], const unsigned long flags, const long ostr[__VLA(D)], complex float* optr, const long istr[__VLA(D)], const complex float* iptr);

typedef CLOSURE_TYPE(complex float, (const long pos[], const float kpos[])) sample_filter_fun;

extern void md_zsample_filter(int D, const long dims[__VLA(D)], unsigned long flags, const float resolution[__VLA2(D)], _Complex float* z, sample_filter_fun fun, _Bool centered);

extern void klaplace_fd_scaled_uncentered(int N, const long dims[__VLA(N)], const float scale[__VLA(N)], complex float* z);
extern void klaplace_fd_uncentered(int N, const long dims[__VLA(N)], complex float* z);

#include "misc/cppwrap.h"

