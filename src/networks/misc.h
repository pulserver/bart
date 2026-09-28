
#include "misc/dimtypes.h"
#include "misc/mri.h"

struct network_data_s {

	struct config_nlop_mri_s* conf;

	int N;
	int ND;

	bart_dim_t ksp_dims[DIMS];
	bart_dim_t col_dims[DIMS];
	bart_dim_t psf_dims[DIMS + 1];
	bart_dim_t img_dims[DIMS];
	bart_dim_t max_dims[DIMS];
	bart_dim_t cim_dims[DIMS];
	bart_dim_t out_dims[DIMS];
	bart_dim_t pat_dims[DIMS];
	bart_dim_t trj_dims[DIMS];
	bart_dim_t bas_dims[DIMS];
	bart_dim_t scl_dims[DIMS];

	const char* filename_trajectory;
	const char* filename_pattern;
	const char* filename_kspace;
	const char* filename_coil;
	const char* filename_basis;
	const char* filename_out;

	_Bool export;
	const char* filename_adjoint;
	const char* filename_psf;

	_Complex float* kspace;
	_Complex float* adjoint;
	_Complex float* initialization;
	_Complex float* coil;
	_Complex float* psf;
	_Complex float* out;
	_Complex float* pattern;
	_Complex float* trajectory;
	_Complex float* basis;
	_Complex float* scale;

	struct nufft_conf_s* nufft_conf;

	_Bool create_out;
	_Bool load_mem;
	_Bool gpu;
	_Bool precomp;

	bart_flags_t batch_flags;
};

extern struct network_data_s network_data_empty;

extern void load_network_data(struct network_data_s* network_data);
extern void free_network_data(struct network_data_s* network_data);

extern void network_data_normalize(struct network_data_s* nd);
extern void network_data_compute_init(struct network_data_s* nd, _Complex float lambda, int cg_iter);
extern void network_data_slice_dim_to_batch_dim(struct network_data_s* nd);

extern void network_data_check_simple_dims(struct network_data_s* network_data);
extern bart_dim_t network_data_get_tot(struct network_data_s* network_data);

extern struct named_data_list_s* network_data_get_named_list(struct network_data_s* nd);