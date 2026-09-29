
#include "misc/dimtypes.h"
#include <complex.h>

extern const float wavelet_haar[2][2][2];
extern const float wavelet_dau2[2][2][4];
extern const float wavelet_cdf44[2][2][10];

// layer 1

extern void fwt1(int N, int d, const bart_dim_t dims[N], const bart_stride_t ostr[N], complex float* low, complex float* hgh, const bart_stride_t istr[N], const complex float* in, const bart_dim_t flen, const float filter[2][2][flen]);
extern void iwt1(int N, int d, const bart_dim_t dims[N], const bart_stride_t ostr[N], complex float* out, const bart_stride_t istr[N], const complex float* low, const complex float* hgh, const bart_dim_t flen, const float filter[2][2][flen]);

// layer 2

extern void fwtN(int N, bart_flags_t flags, const bart_dim_t shifts[N], const bart_dim_t dims[N], const bart_stride_t ostr[2 * N], complex float* out, const bart_stride_t istr[N], const complex float* in, const bart_dim_t flen, const float filter[2][2][flen]);
extern void iwtN(int N, bart_flags_t flags, const bart_dim_t shifts[N], const bart_dim_t dims[N], const bart_stride_t ostr[N], complex float* out, const bart_stride_t istr[2 * N], const complex float* in, const bart_dim_t flen, const float filter[2][2][flen]);

extern void wavelet_dims(int N, bart_flags_t flags, bart_dim_t odims[2 * N], const bart_dim_t dims[N], const bart_dim_t flen);

// layer 3

extern void fwt(int N, bart_flags_t flags, const bart_dim_t shifts[N], const bart_dim_t odims[N], complex float* out, const bart_dim_t idims[N], const complex float* in, const bart_dim_t minsize[N], const bart_dim_t flen, const float filter[2][2][flen]);
extern void iwt(int N, bart_flags_t flags, const bart_dim_t shifts[N], const bart_dim_t odims[N], complex float* out, const bart_dim_t idims[N], const complex float* in, const bart_dim_t minsize[N], const bart_dim_t flen, const float filter[2][2][flen]);

extern int wavelet_num_levels(int N, bart_flags_t flags, const bart_dim_t dims[N], const bart_dim_t min[N], const bart_dim_t flen);
extern bart_dim_t wavelet_coeffs(int N, bart_flags_t flags, const bart_dim_t dims[N], const bart_dim_t min[N], const bart_dim_t flen);

extern void wavelet_coeffs2(int N, bart_flags_t flags, bart_dim_t odims[N], const bart_dim_t dims[N], const bart_dim_t min[N], const bart_dim_t flen);

extern void fwt2(int N, bart_flags_t flags, const bart_dim_t shifts[N], const bart_dim_t odims[N], const bart_stride_t ostr[N], complex float* out, const bart_dim_t idims[N], const bart_stride_t istr[N], const complex float* in, const bart_dim_t minsize[N], bart_dim_t flen, const float filter[2][2][flen]);
extern void iwt2(int N, bart_flags_t flags, const bart_dim_t shifts[N], const bart_dim_t odims[N], const bart_stride_t ostr[N], complex float* out, const bart_dim_t idims[N], const bart_stride_t istr[N], const complex float* in, const bart_dim_t minsize[N], const bart_dim_t flen, const float filter[2][2][flen]);

extern void wavelet_thresh(int N, float lambda, bart_flags_t flags, bart_flags_t jflags, const bart_dim_t shifts[N], const bart_dim_t dims[N], complex float* out, const complex float* in, const bart_dim_t minsize[N], bart_dim_t flen, const float filter[2][2][flen]);


