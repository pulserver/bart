

struct nn_weights_s;
struct loss_config_s;

enum BOOL_SELECT {BOOL_DEFAULT, BOOL_TRUE, BOOL_FALSE};

struct reconet_s {

	struct network_s* network;
	long Nt;

	enum BOOL_SELECT share_weights_select;
	enum BOOL_SELECT share_lambda_select;
	bool share_weights;
	bool share_lambda;

	struct config_nlop_mri_s* sense_config;
	bool one_channel_per_map;

	bool external_initialization;	//initialize network with precomputed reconstruction

	//data consistency config
	float dc_lambda_fixed;
	float dc_lambda_init;
	bool dc_gradient;
	bool dc_scale_max_eigen;
	bool dc_proxmap;
	int dc_max_iter;

	//network initialization
	bool normalize;
	bool sense_init;
	int init_max_iter;
	float init_lambda_fixed;
	float init_lambda_init;

	struct nn_weights_s* weights;
	struct iter6_conf_s* train_conf;

	struct loss_config_s* train_loss;
	struct loss_config_s* valid_loss;

	bool low_mem;
	bool gpu;

	const char* graph_file;

	bool coil_image;
	bool ref_is_kspace;

	bool normalize_rss;

	bool ksp_training;

	bool precomp;
};

extern struct reconet_s reconet_config_opts;

struct named_data_list_s;

extern void reconet_init_modl_default(struct reconet_s* reconet);
extern void reconet_init_varnet_default(struct reconet_s* reconet);
extern void reconet_init_unet_default(struct reconet_s* reconet);

extern void reconet_init_modl_test_default(struct reconet_s* reconet);
extern void reconet_init_varnet_test_default(struct reconet_s* reconet);
extern void reconet_init_unet_test_default(struct reconet_s* reconet);

extern void apply_reconet(	struct reconet_s* config,
				struct named_data_list_s* data);

extern void train_reconet(	struct reconet_s* config,
				long Nb_train, struct named_data_list_s* train_data,
				long Nb_valid, struct named_data_list_s* valid_data);

extern void eval_reconet(	struct reconet_s* config,
				struct named_data_list_s* data);