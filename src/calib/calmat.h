
#include "misc/dimtypes.h"
#include "misc/cppwrap.h"

#include <complex.h>

extern complex float* calibration_matrix(bart_dim_t calmat_dims[2], const bart_dim_t kdims[3], const bart_dim_t calreg_dims[4], const complex float* data);
extern complex float* calibration_matrix_mask(bart_dim_t calmat_dims[2], const bart_dim_t kdims[3], const complex float* mask, const bart_dim_t calreg_dims[4], const complex float* data);
extern complex float* calibration_matrix_mask2(bart_dim_t calmat_dims[2], const bart_dim_t kdims[3], const complex float* msk, const bart_dim_t calreg_dims[4], const complex float* data);

#ifndef __cplusplus
extern void covariance_function(const bart_dim_t kdims[3], int N, complex float cov[static N][N], const bart_dim_t calreg_dims[4], const complex float* data);
extern void covariance_function_fft(const bart_dim_t kdims[3], int N, complex float cov[static N][N], const bart_dim_t calreg_dims[4], const complex float* data);
extern void calmat_svd(const bart_dim_t kdims[3], int N, complex float cov[static N][N], float* S, const bart_dim_t calreg_dims[4], const complex float* data);
#endif

#include "misc/cppwrap.h"

