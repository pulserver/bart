
#ifndef _ITER_ITALGOS_H
#define _ITER_ITALGOS_H

#include "misc/dimtypes.h"
#include "misc/cppwrap.h"

enum IN_TYPE { IN_UNDEFINED, IN_STATIC, IN_BATCH, IN_OPTIMIZE, IN_BATCH_GENERATOR, IN_BATCHNORM, IN_GAUSSIAN_RAND, IN_UNIFORM_RAND };
enum OUT_TYPE { OUT_UNDEFINED, OUT_STATIC, OUT_OPTIMIZE, OUT_BATCHNORM };

#ifndef NUM_INTERNAL
// #warning "Use of private interfaces"
#endif

#include "misc/types.h"
#include "misc/nested.h"

struct vec_iter_s;
struct iter_dump_s;

#ifndef MD_IS_SET
#define MD_BIT(x) (UINT64_C(1) << (x))
#define MD_IS_SET(x, y)	((x) & MD_BIT(y))
#define MD_CLEAR(x, y) ((x) & ~MD_BIT(y))
#define MD_SET(x, y)	((x) | MD_BIT(y))
#endif

#ifndef ITER_OP_DATA_S
#define ITER_OP_DATA_S
typedef struct iter_op_data_s { TYPEID* TYPEID; } iter_op_data;
#endif
typedef void (*iter_op_fun_t)(iter_op_data* data, float* dst, const float* src);
typedef void (*iter_nlop_fun_t)(iter_op_data* data, int OO, int II, float* args[OO + II], _Bool der_out[OO], _Bool der_in[II]);
typedef void (*iter_op_p_fun_t)(iter_op_data* data, float rho, float* dst, const float* src);
typedef void (*iter_op_arr_fun_t)(iter_op_data* data, int NO, float* dst[NO], int NI, const float* src[NI]);

struct iter_op_s {

	iter_op_fun_t fun;
	iter_op_data* data;
};

struct iter_nlop_s {

	iter_nlop_fun_t fun;
	iter_op_data* data;
};

struct iter_op_p_s {

	iter_op_p_fun_t fun;
	iter_op_data* data;
};

struct iter_op_arr_s {

	iter_op_arr_fun_t fun;
	iter_op_data* data;
};

inline void iter_op_call(struct iter_op_s op, float* dst, const float* src)
{
	op.fun(op.data, dst, src);
}

inline void iter_nlop_call(struct iter_nlop_s op, int OO, int II, float* args[OO + II])
{
	op.fun(op.data, OO, II, args, NULL, NULL);
}

inline void iter_nlop_call_select_der(struct iter_nlop_s op, int OO, int II, float* args[OO + II], _Bool der_out[OO], _Bool der_in[II])
{
	op.fun(op.data, OO, II, args, der_out, der_in);
}

inline void iter_op_p_call(struct iter_op_p_s op, float rho, float* dst, const float* src)
{
	op.fun(op.data, rho, dst, src);
}

inline void iter_op_arr_call(struct iter_op_arr_s op, int NO, float* dst[NO], int NI, const float* src[NI])
{
	op.fun(op.data, NO, dst, NI, src);
}


struct iter_monitor_s;
struct monitor_iter6_s;

float conjgrad(int maxiter, float l2lambda, float epsilon,
	bart_dim_t N,
	const struct vec_iter_s* vops,
	struct iter_op_s linop,
	float* x, const float* b,
	struct iter_monitor_s* monitor);

void conjgrad_batch(int maxiter, float l2lambda, float* l2lambda_batch, float epsilon,
	bart_dim_t N, bart_dim_t Bi, bart_dim_t Bo,
	const struct vec_iter_s* vops,
	struct iter_op_s linop,
	float* x, const float* b,
	struct iter_monitor_s* monitor);


void landweber(int maxiter, float epsilon, float alpha,
	bart_dim_t N, bart_dim_t M,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	struct iter_op_s adj,
	float* x, const float* b,
	struct iter_op_s callback,
	struct iter_monitor_s* monitor);

void landweber_sym(int maxiter, float epsilon, float alpha,
	bart_dim_t N,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	float* x, const float* b,
	struct iter_monitor_s* monitor);

void eulermaruyama(int maxiter, float alpha, float step,
	bart_dim_t N,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	struct iter_op_p_s* thresh,
	float* x, const float *b,
	struct iter_monitor_s* monitor);

void eulermaruyama_precond(int maxiter, float alpha,
	float step, bart_dim_t N,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	struct iter_op_p_s* thresh,
	float* x, const float* b,
	bart_dim_t M,
	struct iter_op_s prec_adj,
	struct iter_op_s prec_inormal,
	float diag_prec,
	int max_prec_iter,
	float prec_tol,
	bart_dim_t batchsize,
	struct iter_monitor_s* monitor);

void sgd(int epochs, int batches,
	float learning_rate, float batchnorm_momentum,
	const float (*learning_rate_schedule)[epochs][batches],
	int NI, bart_dim_t isize[NI], enum IN_TYPE in_type[NI], float* x[NI],
	int NO, bart_dim_t osize[NO], enum OUT_TYPE out_type[NI],
	int N_batch, int N_total,
	const struct vec_iter_s* vops,
	struct iter_nlop_s nlop,
	struct iter_op_arr_s adj,
	struct iter_op_p_s update[NI],
	struct iter_op_p_s prox[NI],
	struct iter_nlop_s nlop_batch_gen,
	struct iter_op_s callback,
	struct monitor_iter6_s* monitor,
	const struct iter_dump_s* dump);

/**
 * Store information about iterative algorithm.
 * Used to flexibly modify behavior, e.g. continuation
 *
 * @param rsnew current residual
 * @param rsnot initial residual
 * @param iter current iteration
 * @param maxiter maximum iteration
 * @param tau tau
 * @param scale scaling of regularization
 */
struct ist_data {

	double rsnew;
	double rsnot;
	int iter;
	const int maxiter;
	float tau;
	float scale;
};

typedef void CLOSURE_TYPE(ist_continuation_t)(struct ist_data* itrdata);


void ist(int maxiter, float epsilon, float tau, _Bool last,
	bart_dim_t N,
	const struct vec_iter_s* vops,
	ist_continuation_t ist_continuation,
	struct iter_op_s op,
	struct iter_op_p_s thresh,
	float* x, const float* b,
	struct iter_monitor_s* monitor);


struct ravine_conf {
	float p;
	float q;
	float r;
};

extern struct ravine_conf ravine_classical;
extern struct ravine_conf ravine_mod;

void fista(int maxiter, float epsilon, float tau, float alpha,
	_Bool last,
	struct ravine_conf,
	bart_dim_t N,
	const struct vec_iter_s* vops,
	ist_continuation_t ist_continuation,
	struct iter_op_s op,
	struct iter_op_p_s thresh,
	float* x, const float* b,
	struct iter_monitor_s* monitor);


void irgnm(int iter, float alpha, float alpha_min, float redu,
	bart_dim_t N, bart_dim_t M,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	struct iter_op_s adj,
	struct iter_op_p_s inv,
	float* x, const float* x0, const float* y,
	struct iter_op_s callback,
	struct iter_monitor_s* monitor);

void irgnm2(int iter, float alpha, float alpha_min, float alpha0, float redu,
	bart_dim_t N, bart_dim_t M,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	struct iter_op_s der,
	struct iter_op_s adj,
	struct iter_op_p_s lsqr,
	float* x, const float* xref, const float* y,
	struct iter_op_s callback,
	struct iter_monitor_s* monitor);

void levenberg_marquardt(int maxiter, int cgiter, float l2lambda, float redu,
	bart_dim_t N, bart_dim_t M, bart_dim_t Bi, bart_dim_t Bo,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	struct iter_op_s adj,
	struct iter_op_s nrm,
	float* x, const float* y,
	struct iter_op_s callback,
	struct iter_monitor_s* monitor);

void altmin(int iter, float alpha, float redu,
	bart_dim_t N,
	const struct vec_iter_s* vops,
	int NI,
	struct iter_nlop_s op,
	struct iter_op_p_s min_ops[__VLA(NI)],
	float* x[__VLA(NI)], const float* y,
	struct iter_nlop_s callback);

void pocs(int maxiter,
	int D, struct iter_op_p_s proj_ops[__VLA(D)],
	const struct vec_iter_s* vops,
	bart_dim_t N, float* x,
	struct iter_monitor_s* monitor);

double power(int maxiter,
	bart_dim_t N,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	float* u, float*b);

void chambolle_pock(float alpha, int maxiter, float epsilon, float tau, float sigma,
	float sigma_tau_ratio, float theta,
	float decay, _Bool adapt_stepsize,
	int O, bart_dim_t N, bart_dim_t M[O],
	const struct vec_iter_s* vops,
	struct iter_op_s op_norm,
	struct iter_op_s op_forw[O],
	struct iter_op_s op_adj[O],
	struct iter_op_p_s prox1[O],
	struct iter_op_p_s prox2,
	float* x, const float* xadj,
	struct iter_monitor_s* monitor);

void iPALM(	bart_dim_t NI, bart_dim_t isize[__VLA(NI)], enum IN_TYPE in_type[__VLA(NI)], float* x[__VLA(NI)], float* x_old[__VLA(NI)],
		bart_dim_t NO, bart_dim_t osize[__VLA(NO)], enum OUT_TYPE out_type[__VLA(NO)],
		int numbatches, int epoch_start, int epoch_end,
		const struct vec_iter_s* vops,
		float alpha[__VLA(NI)], float beta[__VLA(NI)], _Bool convex[__VLA(NI)], _Bool trivial_stepsize, _Bool reduce_momentum,
		float L[__VLA(NI)], float Lmin, float Lmax, float Lshrink, float Lincrease,
		struct iter_nlop_s nlop,
		struct iter_op_arr_s adj,
		struct iter_op_p_s prox[__VLA(NI)],
		float batchnorm_momentum,
		struct iter_nlop_s nlop_batch_gen,
		struct iter_op_s callback, struct monitor_iter6_s* monitor, const struct iter_dump_s* dump);

void lbfgs(int maxiter, int M, float step, float ftol, float gtol, float c1, float c2, struct iter_op_s op, struct iter_op_s adj, int N, float *x, const struct vec_iter_s* vops);

#include "misc/cppwrap.h"

#endif // _ITER_ITALGOS_H
