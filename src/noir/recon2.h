#include "misc/dimtypes.h"
#include "misc/cppwrap.h"

#include "misc/mri.h"


struct noir2_conf_s {

	unsigned int iter;
	bool rvc;
	float alpha;
	float alpha_min;
	float redu;
	float a;
	float b;
	float c;

	float oversampling_coils;
	bool ret_os_coils;

	unsigned long fft_flags;
	unsigned long ucfft_flags;
	unsigned long wgh_flags;

	int phasepoles;

	bool sms;
	bool sos;

	float scaling;
	bool undo_scaling;
	bool normalize_lowres;

	bool noncart;
	struct nufft_conf_s* nufft_conf;

	struct opt_reg_s* regs;

	bool gpu;

	int cgiter;
	float cgtol;

	bool realtime;
	float temp_damp;

	bool legacy_early_stoppping;

	bool optimized;

	int iter_reg;
	int liniter;
	float lintol;
};

extern const struct noir2_conf_s noir2_defaults;

struct noir2_s;
extern void noir2_recon(const struct noir2_conf_s* conf, struct noir2_s* noir_ops,
			int N,
			const bart_dim_t img_dims[N], _Complex float* img, const _Complex float* img_ref,
			const bart_dim_t col_dims[N], _Complex float* sens,
			const bart_dim_t kco_dims[N], _Complex float* ksens, const _Complex float* sens_ref,
			const bart_dim_t ksp_dims[N], const _Complex float* kspace);

extern void noir2_recon_noncart(
	const struct noir2_conf_s* conf, int N,
	const bart_dim_t img_dims[N], _Complex float* img,
	const bart_dim_t col_dims[N], _Complex float* sens,
	const bart_dim_t kco_dims[N], _Complex float* ksens,
	const bart_dim_t ksp_dims[N], const _Complex float* kspace,
	const bart_dim_t trj_dims[N], const _Complex float* traj,
	const bart_dim_t wgh_dims[N], const _Complex float* weights,
	const bart_dim_t bas_dims[N], const _Complex float* basis,
	const bart_dim_t msk_dims[N], const _Complex float* mask,
	const bart_dim_t cim_dims[N]);

extern void noir2_recon_cart(
	const struct noir2_conf_s* conf, int N,
	const bart_dim_t img_dims[N], _Complex float* img,
	const bart_dim_t col_dims[N], _Complex float* sens,
	const bart_dim_t kco_dims[N], _Complex float* ksens,
	const bart_dim_t ksp_dims[N], const _Complex float* kspace,
	const bart_dim_t pat_dims[N], const _Complex float* pattern,
	const bart_dim_t bas_dims[N], const _Complex float* basis,
	const bart_dim_t msk_dims[N], const _Complex float* mask,
	const bart_dim_t cim_dims[N]);

#include "misc/cppwrap.h"

