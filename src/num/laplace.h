
extern void md_laplace_fd_scaled(int N, const long dims[N], unsigned long flags, const float scale[N], _Complex float* out, const _Complex float* in);
extern void md_laplace_fd(int N, const long dims[N], unsigned long flags, _Complex float* out, const _Complex float* in);

extern void md_laplace_fd_wrapped_phase_scaled(int N, const long dims[N], unsigned long flags, const float scale[N], _Complex float* out, const _Complex float* in);
extern void md_laplace_fd_wrapped_phase(int N, const long dims[N], unsigned long flags, _Complex float* out, const _Complex float* in);

extern void md_laplace_fd_wrapped_phase_exp_scaled(int N, const long dims[N], unsigned long flags, const float scale[N], _Complex float* out, const _Complex float* in);
extern void md_laplace_fd_wrapped_phase_exp(int N, const long dims[N], unsigned long flags, _Complex float* out, const _Complex float* in);
