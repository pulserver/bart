#include "misc/dimtypes.h"


void mri_ops_activate_multigpu(void);
void mri_ops_deactivate_multigpu(void);

struct nufft_conf_s;
struct iter_conjgrad_conf;

struct config_nlop_mri_s;
extern struct config_nlop_mri_s* sense_model_config_cart_create(int N, const bart_dim_t ksp_dims[N], const bart_dim_t img_dims[N], const bart_dim_t col_dims[N], const bart_dim_t pat_dims[N]);
extern struct config_nlop_mri_s* sense_model_config_noncart_create(int N,
	const bart_dim_t trj_dims[N], const bart_dim_t wgh_dims[N], const bart_dim_t ksp_dims[N],
	const bart_dim_t cim_dims[N],	const bart_dim_t img_dims[N], const bart_dim_t col_dims[N],
	const bart_dim_t bas_dims[N], const _Complex float* basis,
	struct nufft_conf_s conf);

extern int sense_model_get_N(struct config_nlop_mri_s* model);
extern void sense_model_get_img_dims(struct config_nlop_mri_s* model, int N, bart_dim_t img_dims[N]);
extern void sense_model_get_col_dims(struct config_nlop_mri_s* model, int N, bart_dim_t col_dims[N]);
extern void sense_model_get_cim_dims(struct config_nlop_mri_s* model, int N, bart_dim_t cim_dims[N]);
extern void sense_model_get_ksp_dims(struct config_nlop_mri_s* model, int N, bart_dim_t ksp_dims[N]);
extern bool sense_model_get_noncart(const struct config_nlop_mri_s* model);

extern void sense_model_config_free(const struct config_nlop_mri_s* x);


struct sense_model_s;

extern void sense_model_free(const struct sense_model_s* x);

extern struct sense_model_s* sense_model_create(const struct config_nlop_mri_s* config);
extern struct sense_model_s* sense_model_normal_create(const struct config_nlop_mri_s* config);

extern struct sense_model_s* sense_cart_normal_create(int N, const bart_dim_t max_dims[N], const struct config_nlop_mri_s* conf);
extern struct sense_model_s* sense_noncart_normal_create(int N, const bart_dim_t max_dims[N], int ND, const bart_dim_t psf_dims[ND], const struct config_nlop_mri_s* conf);

extern const struct nlop_s* nlop_sense_model_set_data_batch_create(int N, const bart_dim_t dims[N], int Nb, struct sense_model_s* models[Nb]);

extern const struct nlop_s* nlop_sense_adjoint_create(int Nb, struct sense_model_s* models[Nb], bool output_psf);

extern const struct nlop_s* nlop_sense_normal_create(int Nb, struct sense_model_s* models[Nb]);
extern const struct nlop_s* nlop_sense_normal_inv_create(int Nb, struct sense_model_s* models[Nb], struct iter_conjgrad_conf* iter_conf, bart_flags_t lambda_flags);
extern const struct nlop_s* nlop_sense_dc_prox_create(int Nb, struct sense_model_s* models[Nb], struct iter_conjgrad_conf* iter_conf, bart_flags_t lambda_flags);
extern const struct nlop_s* nlop_sense_dc_grad_create(int Nb, struct sense_model_s* models[Nb], bart_flags_t lambda_flags);
extern const struct nlop_s* nlop_sense_scale_maxeigen_create(int Nb, struct sense_model_s* models[Nb], int N, const bart_dim_t dims[N]);

extern const struct nlop_s* nlop_mri_loss_create(bool fft, int Nb, struct sense_model_s* models[Nb]);

extern const struct nlop_s* nlop_mri_normal_create(int Nb, const struct config_nlop_mri_s* conf);
extern const struct nlop_s* nlop_mri_normal_inv_create(int N, const bart_dim_t lam_dims[N], int Nb, const struct config_nlop_mri_s* conf, struct iter_conjgrad_conf* iter_conf);
extern const struct nlop_s* nlop_mri_dc_prox_create(int N, const bart_dim_t lam_dims[N], int Nb, const struct config_nlop_mri_s* conf, struct iter_conjgrad_conf* iter_conf);

extern const struct nlop_s* nlop_mri_normal_max_eigen_create(int Nb, const struct config_nlop_mri_s* conf);
extern const struct nlop_s* nlop_mri_scale_rss_create(int Nb, const struct config_nlop_mri_s* conf);