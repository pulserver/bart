
#ifndef _MD_FFT_H
#define _MD_FFT_H	1

#include "misc/dimtypes.h"

#define MD_FFT_FORWARD 0u
#define MD_FFT_INVERSE (~0u)


extern void md_fft2(int N, const bart_dim_t dims[N],
	bart_flags_t flags, bart_flags_t dirs,
	const bart_stride_t ostr[N], complex float* dst,
	const bart_stride_t istr[N], const complex float* in);

extern void md_fft(int N, const bart_dim_t dims[N],
		bart_flags_t flags, bart_flags_t dirs,
		complex float* dst, const complex float* in);


#endif		// _MD_FFT_H

