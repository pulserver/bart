
#include <stddef.h>

#include "misc/cppwrap.h"
#include "misc/types.h"

struct vptr_fun_data_s;
typedef void (*vptr_fun_free_t)(struct vptr_fun_data_s* data);
typedef void (*vptr_fun_t)(struct vptr_fun_data_s* data, int N, int D, const bart_dim_t* dims[__VLA(N)], const bart_stride_t* strs[__VLA(N)], void* args[__VLA(N)]);
typedef struct vptr_fun_data_s { TYPEID* TYPEID; vptr_fun_free_t del; } vptr_fun_data_t;

extern void exec_vptr_fun_internal(vptr_fun_t fun, vptr_fun_data_t* data, int N, int D, bart_flags_t lflags, const bart_dim_t* dims[__VLA(N)], const bart_stride_t* strs[__VLA(N)], void* ptr[__VLA(N)], size_t sizes[__VLA(N)], bool resolve);
extern void exec_vptr_fun_gen(vptr_fun_t fun, vptr_fun_data_t* data, int N, int D, bart_flags_t lflags, bart_flags_t wflags, bart_flags_t rflags, const bart_dim_t* dims[__VLA(N)], const bart_stride_t* strs[__VLA(N)], void* ptr[__VLA(N)], size_t sizes[__VLA(N)], bool resolve);
extern void exec_vptr_fun(vptr_fun_t fun, vptr_fun_data_t* data, int N, int D, bart_flags_t lflags, bart_flags_t wflags, bart_flags_t rflags, const bart_dim_t* dims[__VLA(N)], const bart_stride_t* strs[__VLA(N)], float* cptr[__VLA(N)]);
extern void exec_vptr_zfun(vptr_fun_t fun, vptr_fun_data_t* data, int N, int D, bart_flags_t lflags, bart_flags_t wflags, bart_flags_t rflags, const bart_dim_t* dims[__VLA(N)], const bart_stride_t* strs[__VLA(N)], _Complex float* cptr[__VLA(N)]);

#include "misc/cppwrap.h"

