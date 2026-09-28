#ifndef _ITER_ASL_H
#define _ITER_ASL_H

#include "misc/dimtypes.h"

extern void get_asl_dims(int N, int asl_dim, bart_dim_t asl_dims[N], const bart_dim_t in_dims[N]);
extern const struct linop_s* linop_asl_create(int N, const bart_dim_t img_dims[N], int asl_dim);

extern void get_teasl_label_dims(int N, int teasl_dim, bart_dim_t teasl_label_dims[N], const bart_dim_t in_dims[N]);
extern void get_teasl_pwi_dims(int N, int teasl_dim, bart_dim_t teasl_pwi_dims[N], const bart_dim_t in_dims[N]);
extern const struct linop_s* linop_teasl_extract_label(int N, const bart_dim_t img_dims[N], int teasl_dim);
extern const struct linop_s* linop_teasl_extract_pwi(int N, const bart_dim_t img_dims[N], int teasl_dim);

#endif // _ITER_ASL_H
