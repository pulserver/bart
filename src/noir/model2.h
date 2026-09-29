#include "misc/dimtypes.h"


struct linop_s;
struct nlop_s;


struct noir2_model_conf_s {

	bool noncart;

	bart_flags_t fft_flags;
	bart_flags_t cfft_flags;
	bart_flags_t ufft_flags;
	bart_flags_t wght_flags;

	bool rvc;
	bool sos;
	float a;
	float b;
	float c;

	float oversampling_coils;

	struct nufft_conf_s* nufft_conf;

	bool asymmetric;
	bool ret_os_coils;
};

extern struct noir2_model_conf_s noir2_model_conf_defaults;


struct noir2_s {

	struct noir2_model_conf_s model_conf;

	const struct nlop_s* model;		// nlop holding the model
	const struct linop_s* lop_asym;		// for asymmetric reconstruction
						// use adjoint to grid data

	// linops to construct model: lop_fft(tenmul(lop_im, lop_coil))
	const struct linop_s* lop_fft;		// fft/nufft from coil images to kspace
	const struct linop_s* lop_coil;		// kspace coils to img-coils
	const struct linop_s* lop_im;		// masking/resizing of image


	const struct linop_s* lop_coil2;	// kspace coils to img-coils for postptocessing

	// references to linops to update model parameters
	const struct linop_s* lop_nufft;	// for retrospectively changing trajectory
	const struct linop_s* lop_pattern;	// for retrospectively changing pattern
	const struct linop_s* lop_basis;	// for retrospectively changing basis (cartesian)


	int N;
	bart_dim_t* pat_dims;
	bart_dim_t* bas_dims;
	bart_dim_t* msk_dims;
	bart_dim_t* ksp_dims;
	bart_dim_t* cim_dims;
	bart_dim_t* img_dims;
	bart_dim_t* col_dims;
	bart_dim_t* col_ten_dims;	// col dims as input of tenmul
	bart_dim_t* trj_dims;

	struct multiplace_array_s* basis;	// this is used in nlinv-net to store basis for trajectory update
};

extern struct noir2_s noir2_noncart_create(int N,
	const bart_dim_t trj_dims[N], const _Complex float* traj,
	const bart_dim_t wgh_dims[N], const _Complex float* weights,
	const bart_dim_t bas_dims[N], const _Complex float* basis,
	const bart_dim_t msk_dims[N], const _Complex float* mask,
	const bart_dim_t ksp_dims[N],
	const bart_dim_t cim_dims[N],
	const bart_dim_t img_dims[N],
	const bart_dim_t kco_dims[N],
	const bart_dim_t col_dims[N],
	const struct noir2_model_conf_s* conf);

extern struct noir2_s noir2_noncart_optimized_create(int N,
	const bart_dim_t trj_dims[N], const _Complex float* traj,
	const bart_dim_t wgh_dims[N], const _Complex float* weights,
	const bart_dim_t bas_dims[N], const _Complex float* basis,
	const bart_dim_t msk_dims[N], const _Complex float* mask,
	const bart_dim_t ksp_dims[N],
	const bart_dim_t cim_dims[N],
	const bart_dim_t img_dims[N],
	const bart_dim_t kco_dims[N],
	const bart_dim_t col_dims[N],
	const struct noir2_model_conf_s* conf);

extern struct noir2_s noir2_cart_create(int N,
	const bart_dim_t pat_dims[N], const _Complex float* pattern,
	const bart_dim_t bas_dims[N], const _Complex float* basis,
	const bart_dim_t msk_dims[N], const _Complex float* mask,
	const bart_dim_t ksp_dims[N],
	const bart_dim_t cim_dims[N],
	const bart_dim_t img_dims[N],
	const bart_dim_t kco_dims[N],
	const bart_dim_t col_dims[N],
	const struct noir2_model_conf_s* conf);

extern void noir2_noncart_update(struct noir2_s* model, int N,
	const bart_dim_t trj_dims[N], const _Complex float* traj,
	const bart_dim_t wgh_dims[N], const _Complex float* weights,
	const bart_dim_t bas_dims[N], const _Complex float* basis);

extern void noir2_cart_update(struct noir2_s* model, int N,
	const bart_dim_t pat_dims[N], const _Complex float* pattern,
	const bart_dim_t bas_dims[N], const _Complex float* basis);

#ifdef __GNUC__
#if __GNUC__ <= 11
#undef N
#endif
#endif

extern void noir2_free(struct noir2_s* model);

extern void noir2_orthogonalize(int N, const bart_dim_t col_dims[N], _Complex float* coils);


