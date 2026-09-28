#ifndef _ITER_TGV_H
#define _ITER_TGV_H

#include "misc/dimtypes.h"
#include "linops/linop.h"
struct reg {

	const struct linop_s* linop;
	const struct operator_p_s* prox;
};

struct reg2 {

	const struct linop_s* linop[2];
	const struct operator_p_s* prox[2];
};

struct reg4 {

	const struct linop_s* linop[4];
	const struct operator_p_s* prox[4];
};

extern struct reg tv_reg(bart_flags_t flags, bart_flags_t jflags, float lambda, int N, const bart_dim_t img_dims[N], int tvscales_N, const float tvscales[tvscales_N], const struct linop_s* lop_trafo);
extern struct reg2 tgv_reg(bart_flags_t flags, bart_flags_t jflags, float lambda, int N, const bart_dim_t in_dims[N], bart_dim_t isize, bart_dim_t* ext_shift, const float alpha[2], int tvscales_N, const float tvscales[tvscales_N], const struct linop_s* lop_trafo);
extern struct reg2 ictv_reg(bart_flags_t flags, bart_flags_t jflags, float lambda, int N, const bart_dim_t in_dims[N], bart_dim_t isize, bart_dim_t* ext_shift, const float gamma[2], int tvscales_N, const float tvscales[tvscales_N], int tvscales2_N, const float tvscales2[tvscales2_N], const struct linop_s* lop_trafo);
extern struct reg4 ictgv_reg(bart_flags_t flags, bart_flags_t jflags, float lambda, int N, const bart_dim_t in_dims[N], bart_dim_t isize, bart_dim_t* ext_shift, const float alpha[2], const float gamma[2], int tvscales_N, const float tvscales[tvscales_N], int tvscales2_N, const float tvscales2[tvscales2_N], const struct linop_s* lop_trafo);

#endif // _ITER_TGV_H

