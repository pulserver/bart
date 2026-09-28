#include "misc/dimtypes.h"



struct loss_config_s;
struct nn_weights_s;
struct network_s;
struct iter6_conf_s;
struct nnet_s;

typedef int (*nnet_get_no_odims_t)(const struct nnet_s* config, int NI, const bart_dim_t idims[NI]);
typedef void (*nnet_get_odims_t)(const struct nnet_s* config, int NO, bart_dim_t odims[NO], int NI, const bart_dim_t idims[NI]);

struct nnet_s {

	struct network_s* network;

	struct nn_weights_s* weights;
	struct iter6_conf_s* train_conf;

	struct loss_config_s* train_loss;
	struct loss_config_s* valid_loss;

	bool low_mem;
	bool gpu;

	nnet_get_no_odims_t get_no_odims;
	nnet_get_odims_t get_odims;

	const char* graph_file;

	bart_dim_t N_segm_labels;
};

extern struct nnet_s nnet_init;

struct network_data_s;

extern void nnet_init_mnist_default(struct nnet_s* nnet);
extern void nnet_init_unet_segm_default(struct nnet_s* nnet, bart_dim_t N_unet_segm_labels, bart_dim_t N_nnunet_segm_labels);

extern void apply_nnet(	const struct nnet_s* nnet,
			int NO, const bart_dim_t odims[NO], _Complex float* out,
			int NI, const bart_dim_t idims[NI], const _Complex float* in);

extern void apply_nnet_batchwise(
			const struct nnet_s* nnet,
			int NO, const bart_dim_t odims[NO], _Complex float* out,
			int NI, const bart_dim_t idims[NI], const _Complex float* in,
			bart_dim_t Nb);

extern void train_nnet(	struct nnet_s* nnet,
			int NO, const bart_dim_t odims[NO], const _Complex float* out,
			int NI, const bart_dim_t idims[NI], const _Complex float* in,
			bart_dim_t Nb, const struct nn_weights_s* valid_files);

extern void eval_nnet(	struct nnet_s* nnet,
			int NO, const bart_dim_t odims[NO], const _Complex float* out,
			int NI, const bart_dim_t idims[NI], const _Complex float* in,
			bart_dim_t Nb);

