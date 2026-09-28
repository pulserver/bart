
#include "misc/dimtypes.h"
#include <complex.h>

struct linop_s;
extern struct linop_s* linop_fmac_create(int N, const bart_dim_t dims[N], bart_flags_t oflags, bart_flags_t iflags, bart_flags_t flags, const complex float* tensor);
extern struct linop_s* linop_fmac_dims_create(int N, const bart_dim_t odims[N], const bart_dim_t idims[N], const bart_dim_t tdims[N], const complex float* tensor);

extern void linop_fmac_set_tensor(const struct linop_s* lop, int N, const bart_dim_t tdims[N], const complex float* tensor);
extern void linop_fmac_set_tensor_F(const struct linop_s* lop, int N, const bart_dim_t tdims[N], const complex float* tensor);
extern void linop_fmac_set_tensor_ref(const struct linop_s* lop, int N, const bart_dim_t tdims[N], const complex float* tensor);
