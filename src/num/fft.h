
#ifndef _FFT_H
#define _FFT_H

#include "misc/dimtypes.h"
#include "misc/cppwrap.h"

// similar to fftshift but modulates in the transform domain
extern void fftmod(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t flags, _Complex float* dst, const _Complex float* src);
extern void fftmod2(int D, const bart_dim_t dimensions[__VLA(D)], bart_flags_t flags, const bart_stride_t ostrides[__VLA(D)], _Complex float* dst, const bart_stride_t istrides[__VLA(D)], const _Complex float* src);

// fftmod for ifft
extern void ifftmod(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t flags, _Complex float* dst, const _Complex float* src);
extern void ifftmod2(int D, const bart_dim_t dimensions[__VLA(D)], bart_flags_t flags, const bart_stride_t ostrides[__VLA(D)], _Complex float* dst, const bart_stride_t istrides[__VLA(D)], const _Complex float* src);

// apply scaling necessary for unitarity
extern void fftscale(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t flags, _Complex float* dst, const _Complex float* src);
extern void fftscale2(int D, const bart_dim_t dimensions[__VLA(D)], bart_flags_t flags, const bart_stride_t ostrides[__VLA(D)], _Complex float* dst, const bart_stride_t istrides[__VLA(D)], const _Complex float* src);

// fftshift
extern void fftshift(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t flags, _Complex float* dst, const _Complex float* src);
extern void fftshift2(int D, const bart_dim_t dimensions[__VLA(D)], bart_flags_t flags, const bart_stride_t ostrides[__VLA(D)], _Complex float* dst, const bart_stride_t istrides[__VLA(D)], const _Complex float* src);

// ifftshift
extern void ifftshift(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t flags, _Complex float* dst, const _Complex float* src);
extern void ifftshift2(int D, const bart_dim_t dimensions[__VLA(D)], bart_flags_t flags, const bart_stride_t ostrides[__VLA(D)], _Complex float* dst, const bart_stride_t istrides[__VLA(D)], const _Complex float* src);



// FFT
extern void fft(int D, const bart_dim_t dimensions[__VLA(D)], bart_flags_t flags, _Complex float* dst, const _Complex float* src);
extern void ifft(int D, const bart_dim_t dimensions[__VLA(D)], bart_flags_t flags, _Complex float* dst, const _Complex float* src);
extern void fft2(int D, const bart_dim_t dimensions[__VLA(D)], bart_flags_t flags, const bart_stride_t ostrides[__VLA(D)], _Complex float* dst, const bart_stride_t istrides[__VLA(D)], const _Complex float* src);
extern void ifft2(int D, const bart_dim_t dimensions[__VLA(D)], bart_flags_t flags, const bart_stride_t ostrides[__VLA(D)], _Complex float* dst, const bart_stride_t istrides[__VLA(D)], const _Complex float* src);

// centered
extern void fftc(int D, const bart_dim_t dimensions[__VLA(D)], bart_flags_t flags, _Complex float* dst, const _Complex float* src);
extern void ifftc(int D, const bart_dim_t dimensions[__VLA(D)], bart_flags_t flags, _Complex float* dst, const _Complex float* src);
extern void fftc2(int D, const bart_dim_t dimensions[__VLA(D)], bart_flags_t flags, const bart_stride_t ostrides[__VLA(D)], _Complex float* dst, const bart_stride_t istrides[__VLA(D)], const _Complex float* src);
extern void ifftc2(int D, const bart_dim_t dimensions[__VLA(D)], bart_flags_t flags, const bart_stride_t ostrides[__VLA(D)], _Complex float* dst, const bart_stride_t istrides[__VLA(D)], const _Complex float* src);

// unitary
extern void fftu(int D, const bart_dim_t dimensions[__VLA(D)], bart_flags_t flags, _Complex float* dst, const _Complex float* src);
extern void ifftu(int D, const bart_dim_t dimensions[__VLA(D)], bart_flags_t flags, _Complex float* dst, const _Complex float* src);
extern void fftu2(int D, const bart_dim_t dimensions[__VLA(D)], bart_flags_t flags, const bart_stride_t ostrides[__VLA(D)], _Complex float* dst, const bart_stride_t istrides[__VLA(D)], const _Complex float* src);
extern void ifftu2(int D, const bart_dim_t dimensions[__VLA(D)], bart_flags_t flags, const bart_stride_t ostrides[__VLA(D)], _Complex float* dst, const bart_stride_t istrides[__VLA(D)], const _Complex float* src);

// unitary and centered
extern void fftuc(int D, const bart_dim_t dimensions[__VLA(D)], bart_flags_t flags, _Complex float* dst, const _Complex float* src);
extern void ifftuc(int D, const bart_dim_t dimensions[__VLA(D)], bart_flags_t flags, _Complex float* dst, const _Complex float* src);
extern void fftuc2(int D, const bart_dim_t dimensions[__VLA(D)], bart_flags_t flags, const bart_stride_t ostrides[__VLA(D)], _Complex float* dst, const bart_stride_t istrides[__VLA(D)], const _Complex float* src);
extern void ifftuc2(int D, const bart_dim_t dimensions[__VLA(D)], bart_flags_t flags, const bart_stride_t ostrides[__VLA(D)], _Complex float* dst, const bart_stride_t istrides[__VLA(D)], const _Complex float* src);

#include "misc/cppwrap.h"

#endif

