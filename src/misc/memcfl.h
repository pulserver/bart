
#include "misc/dimtypes.h"
#include <complex.h>

extern void memcfl_register(const char* name, int D, const bart_dim_t dims[D], complex float* data, bool managed);
extern bool memcfl_exists(const char* name);
extern complex float* memcfl_create(const char* name, int D, const bart_dim_t dims[D]);
extern complex float* memcfl_load(const char* name, int D, bart_dim_t dims[D]);
extern bool memcfl_unmap(const complex float* p);
extern void memcfl_unlink(const char* name);
extern const char** memcfl_list_all(void);

