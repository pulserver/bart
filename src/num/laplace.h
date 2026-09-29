#include "misc/dimtypes.h"

extern void md_laplace_fd_scaled(int N, const bart_dim_t dims[N], bart_flags_t flags, const float scale[N], _Complex float* out, const _Complex float* in);
extern void md_laplace_fd(int N, const bart_dim_t dims[N], bart_flags_t flags, _Complex float* out, const _Complex float* in);

extern void md_laplace_fd_wrapped_phase_scaled(int N, const bart_dim_t dims[N], bart_flags_t flags, const float scale[N], _Complex float* out, const _Complex float* in);
extern void md_laplace_fd_wrapped_phase(int N, const bart_dim_t dims[N], bart_flags_t flags, _Complex float* out, const _Complex float* in);

extern void md_laplace_fd_wrapped_phase_exp_scaled(int N, const bart_dim_t dims[N], bart_flags_t flags, const float scale[N], _Complex float* out, const _Complex float* in);
extern void md_laplace_fd_wrapped_phase_exp(int N, const bart_dim_t dims[N], bart_flags_t flags, _Complex float* out, const _Complex float* in);
