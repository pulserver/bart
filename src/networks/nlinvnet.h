
#include "misc/dimtypes.h"
#include "linops/someops.h"
#include <iso646.h>

struct nn_weights_s;
struct loss_config_s;

struct nlinvnet_s {

	// Training configuration
	struct iter6_conf_s* train_conf;
	struct loss_config_s* train_loss;
	struct loss_config_s* valid_loss;
	float l2loss_reg;
	bart_dim_t time_mask[2];
	bart_dim_t avg_coils_loss;

	// Self-Supervised k-Space
	bool ksp_training;
	float ksp_split;
	bart_flags_t ksp_shared_dims;
	float ksp_leaky;
	const char* use_reco_file; 

	// Network block
	struct network_s* network;
	struct nn_weights_s* weights;
	bool share_weights;
	float lambda;
	float lambda_sens;
	bart_flags_t filter_flags;
	const _Complex float* filter;
	
	int conv_time;
	enum PADDING conv_padding;
	
	// NLINV configuration
	struct noir2_conf_s* conf;
	struct noir2_net_config_s* model;
	struct iter_conjgrad_conf* iter_conf;
	struct iter_conjgrad_conf* iter_conf_net;
	float cgtol;
	int iter_net;		//# of iterations with network
	float oversampling_coils;
	bart_dim_t senssize;

	bool fix_coils;
	bool ref_init_img;
	bool ref_init_col;
	bool ref_init_col_rt;
	float scaling;
	bool real_time_init;
	float temp_damp;

	bool debug;

	bool normalize_rss;
};

extern struct nlinvnet_s nlinvnet_config_opts;

struct network_data_s;

extern void nlinvnet_init_varnet_default(struct nlinvnet_s* nlinvnet);
extern void nlinvnet_init_varnet_test_default(struct nlinvnet_s* nlinvnet);
extern void nlinvnet_init_resnet_default(struct nlinvnet_s* nlinvnet);

void nlinvnet_init(struct nlinvnet_s* nlinvnet, int N,
	const bart_dim_t trj_dims[__VLA2(N)],
	const bart_dim_t pat_dims[__VLA(N)],
	const bart_dim_t bas_dims[__VLA2(N)], const _Complex float* basis,
	const bart_dim_t ksp_dims[__VLA(N)],
	const bart_dim_t cim_dims[__VLA(N)],
	const bart_dim_t img_dims[__VLA(N)],
	const bart_dim_t col_dims[__VLA(N)]);


enum nlinvnet_out { NLINVNET_OUT_CIM, NLINVNET_OUT_KSP, NLINVNET_OUT_IMG_COL };

struct named_data_list_s;
void train_nlinvnet(struct nlinvnet_s* nlinvnet, int Nb, struct named_data_list_s* train_data, struct named_data_list_s* valid_data);

void apply_nlinvnet(struct nlinvnet_s* nlinvnet, int N,
	const bart_dim_t img_dims[N], _Complex float* img,
	const bart_dim_t col_dims[N], _Complex float* col,
	const bart_dim_t ksp_dims[N], const _Complex float* ksp,
	const bart_dim_t pat_dims[N], const _Complex float* pat,
	const bart_dim_t trj_dims[N], const _Complex float* trj);
