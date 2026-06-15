
#include "misc/cppwrap.h"

extern void casorati_dims(int N, long odim[2], const long dimk[__VLA(N)], const long dims[__VLA(N)]);
extern void casorati_matrix(int N, const long dimk[__VLA(N)], const long odim[2], _Complex float* optr, const long dim[__VLA(N)], const long str[__VLA(N)], const _Complex float* iptr);
extern void casorati_matrixH(int N, const long dimk[__VLA(N)], const long dim[__VLA(N)], const long str[__VLA(N)], _Complex float* optr, const long odim[2], const _Complex float* iptr);


extern void basorati_dims(int N, long odim[2], const long dimk[__VLA(N)], const long dims[__VLA(N)]);
extern void basorati_matrix(int N, const long dimk[__VLA(N)], const long odim[2], _Complex float* optr, const long dim[__VLA(N)], const long str[__VLA(N)], const _Complex float* iptr);
extern void basorati_matrixH(int N, const long dimk[__VLA(N)], const long dim[__VLA(N)], const long str[__VLA(N)], _Complex float* optr, const long odim[2], const _Complex float* iptr);

#include "misc/cppwrap.h"
