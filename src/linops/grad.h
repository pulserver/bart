
#include "misc/cppwrap.h"

extern struct linop_s* linop_grad_forward_create(long N, const long dims[__VLA(N)], int d, unsigned long flags);
extern struct linop_s* linop_grad_backward_create(long N, const long dims[__VLA(N)], int d, unsigned long flags);
extern struct linop_s* linop_grad_zentral_create(long N, const long dims[__VLA(N)], int d, unsigned long flags);

extern struct linop_s* linop_grad_create(long N, const long dims[__VLA(N)], int d, unsigned long flags);

extern struct linop_s* linop_symmetrize_create(long N, const long dims[__VLA(N)], unsigned long flags);

extern struct linop_s* linop_scaled_laplace_create(long N, const long dims[__VLA(N)], unsigned long flags, const float scaling[__VLA(N)]);
extern struct linop_s* linop_laplace_create(long N, const long dims[__VLA(N)], unsigned long flags);

#include "misc/cppwrap.h"

