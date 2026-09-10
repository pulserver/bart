#ifndef _NUM_GPU_COMPAT_COMPLEX_H
#define _NUM_GPU_COMPAT_COMPLEX_H

#ifdef USE_GPU

#ifdef USE_HIP
#include <hip/hip_complex.h>
#define cuComplex hipComplex
#define cuFloatComplex hipFloatComplex
#define cuDoubleComplex hipDoubleComplex
#define make_cuFloatComplex make_hipFloatComplex
#define make_cuDoubleComplex make_hipDoubleComplex
#define cuCaddf hipCaddf
#define cuCmulf hipCmulf
#define cuCdivf hipCdivf
#define cuConjf hipConjf
#define cuCrealf hipCrealf
#define cuCimagf hipCimagf
#define cuCabsf hipCabsf
#define cuCsubf hipCsubf
#define cuCreal hipCreal
#define cuCimag hipCimag
#define cuCadd hipCadd
#define cuCmul hipCmul
#define cuComplexFloatToDouble hipComplexFloatToDouble
#define cuComplexDoubleToFloat hipComplexDoubleToFloat
#else
#include <cuComplex.h>
#endif

#endif

#endif