
#ifndef _ITER_VEC_H
#define _ITER_VEC_H

#include "misc/dimtypes.h"

struct vec_iter_s {

	float* (*allocate)(bart_dim_t N);
	void (*del)(float* x);
	void (*clear)(bart_dim_t N, float* x);
	void (*copy)(bart_dim_t N, float* a, const float* x);
	void (*swap)(bart_dim_t N, float* a, float* x);

	double (*norm)(bart_dim_t N, const float* x);
	double (*dot)(bart_dim_t N, const float* x, const float* y);

	void (*sub)(bart_dim_t N, float* a, const float* x, const float* y);
	void (*add)(bart_dim_t N, float* a, const float* x, const float* y);

	void (*smul)(bart_dim_t N, float alpha, float* a, const float* x);
	void (*xpay)(bart_dim_t N, float alpha, float* a, const float* x);
	void (*axpy)(bart_dim_t N, float* a, float alpha, const float* x);
	void (*axpbz)(bart_dim_t N, float* out, const float a, const float* x, const float b, const float* z);
	void (*fmac)(bart_dim_t N, float* a, const float* x, const float* y);

	void (*mul)(bart_dim_t N, float* a, const float* x, const float* y);
	void (*div)(bart_dim_t N, float* a, const float* x, const float* y);
	void (*sqrt)(bart_dim_t N, float* a, const float* x);

	void (*smax)(bart_dim_t N, float alpha, float* a, const float* x);
	void (*smin)(bart_dim_t N, float alpha, float* a, const float* x);
	void (*sadd)(bart_dim_t N, float* x, float y);
	void (*sdiv)(bart_dim_t N, float* a, float x, const float* y);
	void (*le)(bart_dim_t N, float* a, const float* x, const float* y);

	void (*zmul)(bart_dim_t N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
	void (*zsmax)(bart_dim_t N, float val, _Complex float* dst, const _Complex float* src1);

	void (*rand)(bart_dim_t N, float* dst);
	void (*uniform)(bart_dim_t N, float* dst);

	void (*xpay_bat)(bart_dim_t Bi, bart_dim_t N, bart_dim_t Bo, const float* beta, float* a, const float* x);
	void (*dot_bat)(bart_dim_t Bi, bart_dim_t N, bart_dim_t Bo, float* dst, const float* src1, const float* src2);
	void (*axpy_bat)(bart_dim_t Bi, bart_dim_t N, bart_dim_t Bo, float* a, const float* alpha, const float* x);
};

#ifdef USE_GPU
extern const struct vec_iter_s gpu_iter_ops;
#endif
extern const struct vec_iter_s cpu_iter_ops;

extern const struct vec_iter_s* select_vecops(const float* x);


#endif

