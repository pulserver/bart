#include "misc/dimtypes.h"

struct nlop_s;
struct linop_s;

typedef struct nlop_data_s nlop_data_t;

typedef void (*nlop_del_diag_fun_t)(const nlop_data_t* _data);

typedef void (*nlop_zrblock_diag_generic_fun_t)(const nlop_data_t* _data, int N, int OO, const bart_dim_t odims[OO][N], _Complex float* dst[OO], int II, const bart_dim_t idims[II][N], const _Complex float* src[II], const bart_dim_t ddims[OO][II][N], _Complex float* jac[OO][II], _Complex float* jacc[OO][II]);
typedef void (*nlop_zblock_diag_generic_fun_t)(const nlop_data_t* _data, int N, int OO, const bart_dim_t odims[OO][N], _Complex float* dst[OO], int II, const bart_dim_t idims[II][N], const _Complex float* src[II], const bart_dim_t ddims[OO][II][N], _Complex float* jac[OO][II]);
typedef void (*nlop_rblock_diag_generic_fun_t)(const nlop_data_t* _data, int N, int OO, const bart_dim_t odims[OO][N], float* dst[OO], int II, const bart_dim_t idims[II][N], const float* src[II], const bart_dim_t ddims[OO][II][N], float* jac[OO][II]);

extern struct nlop_s* nlop_zrblock_diag_generic_create(nlop_data_t* data, int N,
						int OO, const bart_dim_t odims[OO][N],
						int II, const bart_dim_t idims[II][N],
						bart_flags_t diag_flags [OO][II],
						bool zr_flags [OO][II],
						nlop_zrblock_diag_generic_fun_t forward, nlop_del_diag_fun_t del);

extern struct nlop_s* nlop_zblock_diag_generic_create(nlop_data_t* data, int N,
						int OO, const bart_dim_t odims[OO][N],
						int II, const bart_dim_t idims[II][N],
						bart_flags_t diag_flags [OO][II],
						nlop_zblock_diag_generic_fun_t forward, nlop_del_diag_fun_t del);

extern struct nlop_s* nlop_rblock_diag_generic_create(nlop_data_t* data, int N,
						int OO, const bart_dim_t rodims[OO][N],
						int II, const bart_dim_t ridims[II][N],
						bart_flags_t diag_flags [OO][II],
						nlop_rblock_diag_generic_fun_t forward, nlop_del_diag_fun_t del);

extern bool nlop_block_diag_der_available(const struct nlop_s* op, int o, int i);

typedef void (*nlop_zrdiag_fun_t)(const nlop_data_t* _data, int N, const bart_dim_t dims[N], _Complex float* dst, const _Complex float* src, _Complex float* jac, _Complex float* jacc);
typedef void (*nlop_zdiag_fun_t)(const nlop_data_t* _data, int N, const bart_dim_t dims[N], _Complex float* dst, const _Complex float* src, _Complex float* jac);
typedef void (*nlop_rdiag_fun_t)(const nlop_data_t* _data, int N, const bart_dim_t dims[N], float* dst, const float* src, float* jac);

extern struct nlop_s* nlop_zrdiag_create(int N, const bart_dim_t dims[N], nlop_data_t* data, nlop_zrdiag_fun_t forward, nlop_del_diag_fun_t del);
extern struct nlop_s* nlop_zdiag_create(int N, const bart_dim_t dims[N], nlop_data_t* data, nlop_zdiag_fun_t forward, nlop_del_diag_fun_t del);
extern struct nlop_s* nlop_rdiag_create(int N, const bart_dim_t dims[N], nlop_data_t* data, nlop_rdiag_fun_t forward, nlop_del_diag_fun_t del);

typedef void (*nlop_zrblock_diag_fun_t)(const nlop_data_t* _data, int N, const bart_dim_t odims[N], _Complex float* dst, const bart_dim_t idims[N], const _Complex float* src, const bart_dim_t ddims[N], _Complex float* jac, _Complex float* jacc);
typedef void (*nlop_zblock_diag_fun_t)(const nlop_data_t* _data, int N, const bart_dim_t odims[N], _Complex float* dst, const bart_dim_t idims[N], const _Complex float* src, const bart_dim_t ddims[N], _Complex float* jac);
typedef void (*nlop_rblock_diag_fun_t)(const nlop_data_t* _data, int N, const bart_dim_t odims[N], float* dst, const bart_dim_t idims[N], const float* src, const bart_dim_t ddims[N], float* jac);

extern struct nlop_s* nlop_zrblock_diag_create(nlop_data_t* data, int N, const bart_dim_t odims[N], const bart_dim_t idims[N], const bart_dim_t ddims[N], nlop_zrblock_diag_fun_t forward, nlop_del_diag_fun_t del);
extern struct nlop_s* nlop_zblock_diag_create(nlop_data_t* data, int N, const bart_dim_t odims[N], const bart_dim_t idims[N], const bart_dim_t ddims[N], nlop_zblock_diag_fun_t forward, nlop_del_diag_fun_t del);
extern struct nlop_s* nlop_rblock_diag_create(nlop_data_t* data, int N, const bart_dim_t odims[N], const bart_dim_t idims[N], const bart_dim_t ddims[N], nlop_rblock_diag_fun_t forward, nlop_del_diag_fun_t del);


extern void linop_compute_matrix_zblock_diag_fwd(const struct linop_s* lop, int N, const bart_dim_t dims[N], _Complex float* out);
extern void linop_compute_matrix_zblock_diag_bwd(const struct linop_s* lop, int N, const bart_dim_t dims[N], _Complex float* out);
extern void linop_compute_matrix_zblock_diag(const struct linop_s* lop, int N, const bart_dim_t dims[N], _Complex float* out);

extern void linop_compute_matrix_rblock_diag_fwd(const struct linop_s* lop, int N, const bart_dim_t dims[N], float* out);
extern void linop_compute_matrix_rblock_diag_bwd(const struct linop_s* lop, int N, const bart_dim_t dims[N], float* out);
extern void linop_compute_matrix_rblock_diag(const struct linop_s* lop, int N, const bart_dim_t dims[N], float* out);

extern void linop_compute_matrix_zrblock_diag(const struct linop_s* lop, int N, const bart_dim_t dims[N], _Complex float* out, _Complex float* outc);

extern struct nlop_s* nlop_zprecomp_jacobian_F(const struct nlop_s* nlop);
extern struct nlop_s* nlop_zrprecomp_jacobian_F(const struct nlop_s* nlop);

extern bool nlop_is_zblock_diag(const struct nlop_s* nlop);
extern const struct nlop_data_s* nlop_zblock_diag_get_data(const struct nlop_s* nlop);
extern void nlop_zblock_diag_get_dims(const struct nlop_s* nlop, int N, int OO, bart_dim_t odims[OO][N], int II, bart_dim_t idims[II][N], bart_dim_t ddims[OO][II][N]);
extern void nlop_zblock_diag_apply(const struct nlop_s* nlop, int N, int OO, const bart_dim_t odims[OO][N], _Complex float* dst[OO], int II, const bart_dim_t idims[II][N], const _Complex float* src[II], const bart_dim_t ddims[OO][II][N], _Complex float* jac[OO][II]);

