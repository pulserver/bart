
#include <complex.h>

#include "misc/cppwrap.h"
#include "misc/nested.h"

typedef CLOSURE_TYPE(float, (const long pos[])) sample_fun_t;
typedef CLOSURE_TYPE(complex float, (const long pos[])) zsample_fun_t;
typedef CLOSURE_TYPE(complex double, (const long pos[])) zzsample_fun_t;


extern void md_sample(int N, const long dims[__VLA(N)], float* z, sample_fun_t fun);
#define md_sample(N, dims, z, fun) md_sample(N, dims, z, CLOSURE(sample_fun_t, fun))
extern void md_parallel_sample(int N, const long dims[__VLA(N)], float* z, sample_fun_t fun);
#define md_parallel_sample(N, dims, z, fun) md_parallel_sample(N, dims, z, CLOSURE(sample_fun_t, fun))

extern void md_zsample(int N, const long dims[__VLA(N)], complex float* z, zsample_fun_t fun);
#define md_zsample(N, dims, z, fun) md_zsample(N, dims, z, CLOSURE(zsample_fun_t, fun))
extern void md_parallel_zsample(int N, const long dims[__VLA(N)], complex float* z, zsample_fun_t fun);
#define md_parallel_zsample(N, dims, z, fun) md_parallel_zsample(N, dims, z, CLOSURE(zsample_fun_t, fun))

extern void md_zzsample(int N, const long dims[__VLA(N)], complex double* z, zzsample_fun_t fun);
#define md_zzsample(N, dims, z, fun) md_zzsample(N, dims, z, CLOSURE(zzsample_fun_t, fun))
extern void md_parallel_zzsample(int N, const long dims[__VLA(N)], complex double* z, zzsample_fun_t fun);
#define md_parallel_zzsample(N, dims, z, fun) md_parallel_zzsample(N, dims, z, CLOSURE(zzsample_fun_t, fun))

extern void md_zgradient(int N, const long dims[__VLA(N)], complex float* out, const complex float grad[__VLA(N)]);

typedef complex float (*map_fun_t)(complex float arg);


extern void md_zmap(int N, const long dims[__VLA(N)], complex float* out, const complex float* in, map_fun_t fun);


#include "misc/cppwrap.h"

