
struct nlop_s;
extern float nlop_test_derivative(const struct nlop_s* op);
extern float nlop_test_derivative_at(const struct nlop_s* op, const complex float* in);

extern float nlop_test_adj_derivatives(const struct nlop_s* op, bool real);
extern float nlop_test_derivatives(const struct nlop_s* op);
extern bool nlop_test_derivatives_reduce(const struct nlop_s* op, int iter_max, int reduce_target, float val_target);

extern bool compare_nlops(const struct nlop_s* nlop1, const struct nlop_s* nlop2, bool shape, bool der, bool adj, float tol);

extern float nlop_test_affine_at(const struct nlop_s* op, const _Complex float* in);
extern float nlop_test_affine(const struct nlop_s* op);
