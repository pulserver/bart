#ifndef _NUM_GPU_COMPAT_BLAS_H
#define _NUM_GPU_COMPAT_BLAS_H

#ifdef USE_GPU

#ifdef USE_HIP
#if __has_include(<hipblas/hipblas.h>)
#include <hipblas/hipblas.h>
#elif __has_include(<hipblas.h>)
#include <hipblas.h>
#else
#error "hipblas header not found"
#endif

#define cublasStatus_t hipblasStatus_t
#define cublasHandle_t hipblasHandle_t
#define cublasOperation_t hipblasOperation_t
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED
#ifndef HIPBLAS_STATUS_LICENSE_ERROR
#define HIPBLAS_STATUS_LICENSE_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
#endif
#define CUBLAS_STATUS_LICENSE_ERROR HIPBLAS_STATUS_LICENSE_ERROR
#define CUBLAS_POINTER_MODE_HOST HIPBLAS_POINTER_MODE_HOST
#define CUBLAS_POINTER_MODE_DEVICE HIPBLAS_POINTER_MODE_DEVICE
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_OP_C HIPBLAS_OP_C
#define CUBLAS_SIDE_LEFT HIPBLAS_SIDE_LEFT
#define CUBLAS_SIDE_RIGHT HIPBLAS_SIDE_RIGHT
#define cublasCreate hipblasCreate
#define cublasSetPointerMode hipblasSetPointerMode
#define cublasSetStream hipblasSetStream
#define cublasDestroy hipblasDestroy
#define cublasSasum hipblasSasum
#define cublasSaxpy hipblasSaxpy
#define cublasSswap hipblasSswap
#define cublasCgemm hipblasCgemm
#define cublasCgemv hipblasCgemv
#define cublasCgeru hipblasCgeru
#define cublasCaxpy hipblasCaxpy
#define cublasCscal hipblasCscal
#define cublasCdotu hipblasCdotu
#define cublasSgemm hipblasSgemm
#define cublasSgemv hipblasSgemv
#define cublasSger hipblasSger
#define cublasSscal hipblasSscal
#define cublasSdot hipblasSdot
#define cublasCdgmm hipblasCdgmm
#define cublasSdgmm hipblasSdgmm
#define cublasCgeam hipblasCgeam
#define cublasSgeam hipblasSgeam
#else
#include <cublas_v2.h>
#endif

#endif

#endif