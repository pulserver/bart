#ifndef _NUM_GPU_COMPAT_FFT_H
#define _NUM_GPU_COMPAT_FFT_H

#ifdef USE_GPU

#ifdef USE_HIP
#include <hipfft/hipfft.h>
#define cufftHandle hipfftHandle
#define cufftResult_t hipfftResult_t
#define cufftComplex hipfftComplex
#define CUFFT_SUCCESS HIPFFT_SUCCESS
#define CUFFT_INVALID_PLAN HIPFFT_INVALID_PLAN
#define CUFFT_ALLOC_FAILED HIPFFT_ALLOC_FAILED
#define CUFFT_INVALID_TYPE HIPFFT_INVALID_TYPE
#define CUFFT_INVALID_VALUE HIPFFT_INVALID_VALUE
#define CUFFT_INTERNAL_ERROR HIPFFT_INTERNAL_ERROR
#define CUFFT_EXEC_FAILED HIPFFT_EXEC_FAILED
#define CUFFT_SETUP_FAILED HIPFFT_SETUP_FAILED
#define CUFFT_INVALID_SIZE HIPFFT_INVALID_SIZE
#define CUFFT_UNALIGNED_DATA HIPFFT_UNALIGNED_DATA
#define CUFFT_INVALID_DEVICE HIPFFT_INVALID_DEVICE
#define CUFFT_NO_WORKSPACE HIPFFT_NO_WORKSPACE
#define CUFFT_NOT_IMPLEMENTED HIPFFT_NOT_IMPLEMENTED
#define CUFFT_NOT_SUPPORTED HIPFFT_NOT_SUPPORTED
#define CUFFT_C2C HIPFFT_C2C
#define CUFFT_FORWARD HIPFFT_FORWARD
#define CUFFT_INVERSE HIPFFT_BACKWARD
#define cufftCreate hipfftCreate
#define cufftSetAutoAllocation hipfftSetAutoAllocation
#define cufftMakePlanMany hipfftMakePlanMany
#define cufftDestroy hipfftDestroy
#define cufftSetStream hipfftSetStream
#define cufftSetWorkArea hipfftSetWorkArea
#define cufftExecC2C hipfftExecC2C
#else
#include <cufft.h>
#endif

#endif

#endif