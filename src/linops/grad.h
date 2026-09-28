
#include "misc/dimtypes.h"
#include "misc/cppwrap.h"

extern struct linop_s* linop_grad_forward_create(bart_dim_t N, const bart_dim_t dims[__VLA(N)], int d, bart_flags_t flags);
extern struct linop_s* linop_grad_backward_create(bart_dim_t N, const bart_dim_t dims[__VLA(N)], int d, bart_flags_t flags);
extern struct linop_s* linop_grad_zentral_create(bart_dim_t N, const bart_dim_t dims[__VLA(N)], int d, bart_flags_t flags);

extern struct linop_s* linop_grad_create(bart_dim_t N, const bart_dim_t dims[__VLA(N)], int d, bart_flags_t flags);

#include "misc/cppwrap.h"

