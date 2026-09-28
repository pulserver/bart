
#include "misc/dimtypes.h"
#include "noncart/grid.h"

struct multiplace_array_s;

/**
 *
 * NUFFT internal data structure
 *
 */
struct nufft_data {

	linop_data_t super;

	struct nufft_conf_s conf;	///< NUFFT configuration structure
	struct grid_conf_s grid_conf;

	int N;				///< Number of dimension
	bart_flags_t flags;

	struct multiplace_array_s* linphase;	///< Linear phase for pruned FFT
	struct multiplace_array_s* traj;	///< Trajectory
	struct multiplace_array_s* roll;	///< Roll-off factor
	struct multiplace_array_s* psf;		///< Point-spread function (2x size)
	struct multiplace_array_s* fftmod;	///< FFT modulation for centering
	struct multiplace_array_s* weights;	///< Weights, ex, density compensation
	struct multiplace_array_s* basis;
	struct multiplace_array_s* compress;	///< Compression index

	float width;			///< Interpolation kernel width
	double beta;			///< Kaiser-Bessel beta parameter

	const struct linop_s* fft_op;	///< FFT operator

	bart_dim_t* ksp_dims;			///< Kspace dimension
	bart_dim_t* cim_dims;			///< Coil image dimension
	bart_dim_t* cml_dims;			///< Coil + linear phase dimension
	bart_dim_t* img_dims;			///< Image dimension
	bart_dim_t* trj_dims;			///< Trajectory dimension
	bart_dim_t* lph_dims;			///< Linear phase dimension
	bart_dim_t* psf_dims;			///< Point spread function dimension
	bart_dim_t* wgh_dims;			///< Weights dimension
	bart_dim_t* bas_dims;
	bart_dim_t* out_dims;
	bart_dim_t* ciT_dims;			///< Coil image dimension
	bart_dim_t* cmT_dims;			///< Coil + linear phase dimension
	bart_dim_t* com_dims;			///< Compression index dimensions

	//!
	bart_dim_t* cm2_dims;			///< 2x oversampled coil image dimension
	bart_dim_t* factors;

	bart_stride_t* ksp_strs;
	bart_stride_t* cim_strs;
	bart_stride_t* cml_strs;
	bart_stride_t* img_strs;
	bart_stride_t* trj_strs;
	bart_stride_t* lph_strs;
	bart_stride_t* psf_strs;
	bart_stride_t* wgh_strs;
	bart_stride_t* bas_strs;
	bart_stride_t* out_strs;
	bart_stride_t* com_strs;			///< Compression index dimensions

	const struct linop_s* cfft_op;   ///< Pcycle FFT operator
	int cycle;
};




