#include "misc/dimtypes.h"

extern const struct nlop_s* nlop_mse_create(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t mean_dims);
extern const struct nlop_s* nlop_mse_scaled_create(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t mean_dims);
extern const struct nlop_s* nlop_mpsnr_create(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t mean_dims);
extern const struct nlop_s* nlop_mssim_create(int N, const bart_dim_t dims[__VLA(N)], const bart_dim_t kdims[__VLA(N)], bart_flags_t conv_dims);
extern const struct nlop_s* nlop_patched_cross_correlation_create(int N, const bart_dim_t dims[__VLA(N)], const bart_dim_t kdims[__VLA(N)], bart_flags_t conv_dims, float epsilon);
extern const struct nlop_s* nlop_cce_create(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t batch_flag);
extern const struct nlop_s* nlop_weighted_cce_create(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t batch_flag);
extern const struct nlop_s* nlop_accuracy_create(int N, const bart_dim_t dims[__VLA(N)], int class_index);

extern const struct nlop_s* nlop_nmse_create(int N, const bart_dim_t dims[N], bart_flags_t batch_flags);
extern const struct nlop_s* nlop_nrmse_create(int N, const bart_dim_t dims[N], bart_flags_t batch_flags);

extern const struct nlop_s* nlop_dice_generic_create(int N, const bart_dim_t dims[N], bart_flags_t label_flag, bart_flags_t independent_flag, float weighting_exponent, bool square_denominator);
extern const struct nlop_s* nlop_dice_create(int N, const bart_dim_t dims[N], bart_flags_t label_flag, bart_flags_t mean_flag, float weighting_exponent, bool square_denominator);
