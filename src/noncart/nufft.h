
#ifndef _NUFFT_H
#define _NUFFT_H

#include "misc/cppwrap.h"

struct operator_s;
struct linop_s;

struct nufft_conf_s {

	bool toeplitz; ///< Toeplitz embedding boolean for A^T A
	bool pcycle; 	/// < Phase cycling
	bool periodic;
	bool lowmem;
	bool dft;

	unsigned long flags;
	unsigned long cfft;
	bool decomp;
	bool nopsf;
	bool upper_triag;
	bool real;
	bool compress_psf;
	bool decomposed_psf;

	bool precomp;
	bool precomp_linphase;
	bool precomp_fftmod;
	bool precomp_roll;
	bool zero_overhead;

	float width;
	float os;
};

extern const struct nufft_conf_s nufft_conf_defaults;
extern struct nufft_conf_s nufft_conf_options;

#include "misc/opts.h"
extern struct opt_s nufft_conf_opts[];
extern int N_nufft_conf_opts;


extern struct linop_s* nufft_create(int N,				///< Number of dimensions
				    const long ksp_dims[__VLA(N)],	///< Kspace dimension
				    const long coilim_dims[__VLA(N)],	///< Coil image dimension
				    const long traj_dims[__VLA(N)],	///< Trajectory dimension
				    const _Complex float* traj,		///< Trajectory
				    const _Complex float* weights,	///< Weights, ex, density-compensation
				    struct nufft_conf_s conf);		///< NUFFT configuration

extern struct linop_s* nufft_create2(int N,
				const long ksp_dims[N],
				const long cim_dims[N],
				const long traj_dims[N], const complex float* traj,
				const long wgh_dims[N], const complex float* weights,
				const long bas_dims[N], const complex float* basis,
				const long fm_dims[N], const complex float* fieldmap,
				const long tm_dims[N], const complex float* timemap,
				struct nufft_conf_s conf);

extern _Complex float* compute_psf(int N,
				   const long img2_dims[__VLA(N)],
				   const long trj_dims[__VLA(N)],
				   const _Complex float* traj,
				   const long bas_dims[__VLA2(N)],
				   const _Complex float* basis,
				   const long wgh_dims[__VLA2(N)],
				   const _Complex float* weights,
				   bool periodic,
				   bool lowmem);

extern _Complex float* compute_psf2(int N, const long psf_dims[__VLA(N + 1)], unsigned long flags,
				const long trj_dims[__VLA(N + 1)], const _Complex float* traj,
				const long bas_dims[__VLA2(N + 1)], const _Complex float* basis,
				const long wgh_dims[__VLA2(N + 1)], const _Complex float* weights,
				bool periodic, bool lowmem, bool upper_triag);

extern _Complex float* compute_psf2_decomposed(int N, const long psf_dims[__VLA(N + 1)], unsigned long flags,
				const long trj_dims[__VLA(N + 1)], const _Complex float* traj,
				const long bas_dims[__VLA2(N + 1)], const _Complex float* basis,
				const long wgh_dims[__VLA2(N + 1)], const _Complex float* weights,
				bool periodic, bool lowmem, bool upper_triag);

extern const struct operator_s* nufft_precond_create(const struct linop_s* nufft_op);

extern struct linop_s* nufft_create_normal(int N, const long cim_dims[__VLA(N)],
					   int ND, const long psf_dims[__VLA(ND)], const _Complex float* psf,
					   bool basis, struct nufft_conf_s conf);

extern void nufft_update_traj(const struct linop_s* nufft, int N,
			const long trj_dims[__VLA(N)], const _Complex float* traj,
			const long wgh_dims[__VLA2(N)], const _Complex float* weights,
			const long bas_dims[__VLA2(N)], const _Complex float* basis);
extern void nufft_update_psf(const struct linop_s* nufft, int ND, const long psf_dims[__VLA(ND)], const _Complex float* psf);
extern void nufft_update_psf2(const struct linop_s* nufft, int ND, const long psf_dims[__VLA(ND)], const long psf_strs[__VLA(ND)], const _Complex float* psf);

extern int nufft_get_psf_dims(const struct linop_s* nufft, int N, long psf_dims[N]);
extern void nufft_get_psf2(const struct linop_s* nufft, int N, const long psf_dims[N], const long psf_strs[N], _Complex float* psf);
extern void nufft_get_psf(const struct linop_s* nufft, int N, const long psf_dims[N], _Complex float* psf);

#include "misc/cppwrap.h"

#endif // __NUFFT_H

