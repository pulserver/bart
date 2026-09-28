
#include "misc/dimtypes.h"
#include "misc/cppwrap.h"

struct operator_s;
struct linop_s;

struct nufft_conf_s {

	_Bool toeplitz; ///< Toeplitz embedding boolean for A^T A
	_Bool pcycle; 	/// < Phase cycling
	_Bool periodic;
	_Bool lowmem;
	bart_flags_t flags;
	bart_flags_t cfft;
	_Bool decomp;
	_Bool nopsf;
	_Bool upper_triag;
	_Bool real;
	_Bool compress_psf;
	_Bool decomposed_psf;

	_Bool precomp;
	_Bool precomp_linphase;
	_Bool precomp_fftmod;
	_Bool precomp_roll;
	_Bool zero_overhead;

	float width;
	float os;
};

extern struct nufft_conf_s nufft_conf_defaults;
extern struct nufft_conf_s nufft_conf_options;

#include "misc/opts.h"
extern struct opt_s nufft_conf_opts[];
extern int N_nufft_conf_opts;


extern struct linop_s* nufft_create(int N,				///< Number of dimensions
				    const bart_dim_t ksp_dims[__VLA(N)],	///< Kspace dimension
				    const bart_dim_t coilim_dims[__VLA(N)],	///< Coil image dimension
				    const bart_dim_t traj_dims[__VLA(N)],	///< Trajectory dimension
				    const _Complex float* traj,		///< Trajectory
				    const _Complex float* weights,	///< Weights, ex, density-compensation
				    struct nufft_conf_s conf);		///< NUFFT configuration

extern struct linop_s* nufft_create2(int N,
			     const bart_dim_t ksp_dims[N],
			     const bart_dim_t cim_dims[N],
			     const bart_dim_t traj_dims[N],
			     const _Complex float* traj,
			     const bart_dim_t wgh_dims[N],
			     const _Complex float* weights,
			     const bart_dim_t bas_dims[N],
			     const _Complex float* basis,
			     struct nufft_conf_s conf);

extern _Complex float* compute_psf(int N,
				   const bart_dim_t img2_dims[__VLA(N)],
				   const bart_dim_t trj_dims[__VLA(N)],
				   const _Complex float* traj,
				   const bart_dim_t bas_dims[__VLA2(N)],
				   const _Complex float* basis,
				   const bart_dim_t wgh_dims[__VLA2(N)],
				   const _Complex float* weights,
				   _Bool periodic,
				   _Bool lowmem);

extern _Complex float* compute_psf2(int N, const bart_dim_t psf_dims[__VLA(N + 1)], bart_flags_t flags,
				const bart_dim_t trj_dims[__VLA(N + 1)], const _Complex float* traj,
				const bart_dim_t bas_dims[__VLA2(N + 1)], const _Complex float* basis,
				const bart_dim_t wgh_dims[__VLA2(N + 1)], const _Complex float* weights,
				_Bool periodic, _Bool lowmem, _Bool upper_triag);

extern _Complex float* compute_psf2_decomposed(int N, const bart_dim_t psf_dims[__VLA(N + 1)], bart_flags_t flags,
				const bart_dim_t trj_dims[__VLA(N + 1)], const _Complex float* traj,
				const bart_dim_t bas_dims[__VLA2(N + 1)], const _Complex float* basis,
				const bart_dim_t wgh_dims[__VLA2(N + 1)], const _Complex float* weights,
				_Bool periodic, _Bool lowmem, _Bool upper_triag);

extern const struct operator_s* nufft_precond_create(const struct linop_s* nufft_op);

extern struct linop_s* nufft_create_normal(int N, const bart_dim_t cim_dims[__VLA(N)],
					   int ND, const bart_dim_t psf_dims[__VLA(ND)], const _Complex float* psf,
					   _Bool basis, struct nufft_conf_s conf);

extern void nufft_update_traj(const struct linop_s* nufft, int N,
			const bart_dim_t trj_dims[__VLA(N)], const _Complex float* traj,
			const bart_dim_t wgh_dims[__VLA2(N)], const _Complex float* weights,
			const bart_dim_t bas_dims[__VLA2(N)], const _Complex float* basis);
extern void nufft_update_psf(const struct linop_s* nufft, int ND, const bart_dim_t psf_dims[__VLA(ND)], const _Complex float* psf);
extern void nufft_update_psf2(const struct linop_s* nufft, int ND, const bart_dim_t psf_dims[__VLA(ND)], const bart_stride_t psf_strs[__VLA(ND)], const _Complex float* psf);

extern int nufft_get_psf_dims(const struct linop_s* nufft, int N, bart_dim_t psf_dims[N]);
extern void nufft_get_psf2(const struct linop_s* nufft, int N, const bart_dim_t psf_dims[N], const bart_stride_t psf_strs[N], _Complex float* psf);
extern void nufft_get_psf(const struct linop_s* nufft, int N, const bart_dim_t psf_dims[N], _Complex float* psf);

#include "misc/cppwrap.h"

