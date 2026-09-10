#ifndef _NUM_GPU_COMPAT_RUNTIME_H
#define _NUM_GPU_COMPAT_RUNTIME_H

#ifdef USE_GPU

#ifdef USE_HIP
#if __has_include(<hip/hip_runtime.h>)
#include <hip/hip_runtime.h>
#else
#include <hip/hip_runtime_api.h>
#endif

#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaStream_t hipStream_t
#define cudaGetErrorString hipGetErrorString
#define cudaMemGetInfo hipMemGetInfo
#define cudaSetDevice hipSetDevice
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaStreamLegacy hipStreamDefault
#define cudaStreamCreate hipStreamCreate
#define cudaErrorDevicesUnavailable hipErrorNoDevice
#define cudaGetLastError hipGetLastError
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamDestroy hipStreamDestroy
#define cudaDeviceReset hipDeviceReset
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDevAttrConcurrentManagedAccess hipDeviceAttributeConcurrentManagedAccess
#define cudaMallocManaged hipMallocManaged
#define cudaMemAttachGlobal hipMemAttachGlobal
#define cudaMemLocation hipMemLocation
#define cudaMemLocationTypeDevice hipMemLocationTypeDevice
#define cudaMemAdvise hipMemAdvise
#define cudaMemAdviseSetAccessedBy hipMemAdviseSetAccessedBy
#define cudaMemAdviseSetPreferredLocation hipMemAdviseSetPreferredLocation
#define cudaMemPrefetchAsync hipMemPrefetchAsync
#define cudaMalloc hipMalloc
#define cudaErrorMemoryAllocation hipErrorMemoryAllocation
#define cudaMallocHost(ptr, size) hipHostMalloc((ptr), (size), 0)
#define cudaFreeHost hipHostFree
#define cudaFree hipFree
#define cudaPointerAttributes hipPointerAttribute_t
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaMemoryTypeUnregistered hipMemoryTypeUnregistered
#define cudaMemoryTypeHost hipMemoryTypeHost
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyDefault hipMemcpyDefault

#define cudaFuncAttributes hipFuncAttributes
#define cudaFuncGetAttributes(attr, func) hipFuncGetAttributes((attr), (const void*)(func))
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties

#else

#include <cuda_runtime_api.h>

#endif

#endif

#endif