/* Copyright 2013. The Regents of the University of California.
 * Copyright 2019-2022. Uecker Lab, University Medical Center Goettingen.
 * Copyright 2021-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Xiaoqing Wang, Martin Uecker, Nick Scholand
 */

#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"
#include "num/filter.h"
#include "num/rand.h"

#include "misc/mri.h"
#include "misc/mri2.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/utils.h"
#include "misc/opts.h"
#include "misc/debug.h"
#include "misc/version.h"

#include "seq/pulse.h"

#include "simu/simulation.h"

#include "noncart/nufft.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "iter/iter2.h"

#include "grecon/optreg.h"
#include "grecon/italgo.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "moba/optreg.h"
#include "moba/recon.h"
#include "moba/moba.h"
#include "moba/meco.h"

static const char help_str[] = "Model-based nonlinear inverse reconstruction";


static void edge_filter1(const long map_dims[DIMS], complex float* dst, float lambda)
{
	float sc[DIMS];
	for (int i = 0; i < DIMS; i++)
		sc[i] = 1. / (float)map_dims[i];

	klaplace_scaled(DIMS, map_dims, READ_FLAG|PHS1_FLAG, sc, dst);
	md_zreal(DIMS, map_dims, dst, dst);
	md_zsqrt(DIMS, map_dims, dst, dst);

	md_zsmul(DIMS, map_dims, dst, dst, -2.);
	md_zsadd(DIMS, map_dims, dst, dst, 1.);
	md_zatanr(DIMS, map_dims, dst, dst);

	md_zsmul(DIMS, map_dims, dst, dst, -1. / M_PI);
	md_zsadd(DIMS, map_dims, dst, dst, 1.);
	md_zsmul(DIMS, map_dims, dst, dst, lambda);
}

static void edge_filter2(const long map_dims[DIMS], complex float* dst, float lambda)
{
	float beta = 100.;

	float sc[DIMS];
	for (int i = 0; i < DIMS; i++)
		sc[i] = 1. / (float)map_dims[i];

	klaplace_scaled(DIMS, map_dims, READ_FLAG|PHS1_FLAG, sc, dst);
	md_zspow(DIMS, map_dims, dst, dst, 0.5);

	md_zsmul(DIMS, map_dims, dst, dst, -beta * 2.);
	md_zsadd(DIMS, map_dims, dst, dst, beta);

	md_zatanr(DIMS, map_dims, dst, dst);
	md_zsmul(DIMS, map_dims, dst, dst, -(50. * lambda) / M_PI);
	md_zsadd(DIMS, map_dims, dst, dst, 25. * lambda);
}


int main_moba(int argc, char* argv[argc])
{
	double start_time = timestamp();

	const char* ksp_file = NULL;
	const char* TI_file = NULL;
	const char* out_file = NULL;
	const char* sens_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &ksp_file, "kspace"),
		ARG_INFILE(true, &TI_file, "TI/TE"),
		ARG_OUTFILE(true, &out_file, "output"),
		ARG_OUTFILE(false, &sens_file, "sensitivities"),
	};

	float restrict_fov = -1.;
	float oversampling = 1.f;

	float kfilter_strength = 2e-3;

	bool normalize_scaling = false;
	float scaling = 1.;
	float scaling_psf = 1.;

	const char* psf_file = NULL;
	const char* traj_file = NULL;
	const char* init_file = NULL;
	const char* input_b1 = NULL;
	const char* input_b0 = NULL;
	const char* input_sens = NULL;
	const char* fixed_sens = NULL;
	const char* input_TE = NULL;

	struct moba_conf conf = moba_defaults;
	struct opt_reg_s ropts;
	conf.ropts = &ropts;

	long img_vec[3] = { };

	struct moba_conf_s data;

	data.sim.seq = simdata_seq_defaults;
	data.sim.voxel = simdata_voxel_defaults;
	data.sim.pulse = simdata_pulse_defaults;
	data.sim.pulse.sinc = pulse_sinc_defaults;
	data.sim.pulse.hs = pulse_hypsec_defaults;
	data.sim.grad = simdata_grad_defaults;
	data.sim.other = simdata_other_defaults;
	data.other = moba_other_defaults;

	// FIXME: Move to separate function to reuse it for sim.c
	struct opt_s seq_opts[] = {

		/* Sequence Specific Parameters */
		OPTL_SELECT(0, "BSSFP", enum sim_seq, &(data.sim.seq.seq_type), SEQ_BSSFP, "bSSFP"),
		OPTL_SELECT(0, "IR-BSSFP", enum sim_seq, &(data.sim.seq.seq_type), SEQ_IRBSSFP, "Inversion-Recovery bSSFP"),
		OPTL_SELECT(0, "FLASH", enum sim_seq, &(data.sim.seq.seq_type), SEQ_FLASH, "FLASH"),
		OPTL_SELECT(0, "IR-FLASH", enum sim_seq, &(data.sim.seq.seq_type), SEQ_IRFLASH, "Inversion-Recovery FLASH"),
		OPTL_FLOAT(0, "TR", &(data.sim.seq.tr), "float", "Repetition time [s]"),
		OPTL_FLOAT(0, "TE", &(data.sim.seq.te), "float", "Echo time [s]"),
		OPTL_PINT(0, "Nspins", &(data.sim.seq.spin_num), "int", "Number of averaged spins"),
		OPTL_PINT(0, "Nrep", &(data.sim.seq.rep_num), "int", "Number of repetitions"),
		OPTL_SET(0, "pinv", &(data.sim.seq.perfect_inversion), "Use perfect inversions"),
		OPTL_FLOAT(0, "ipl", &(data.sim.seq.inversion_pulse_length), "float", "Inversion Pulse Length [s]"),
		OPTL_FLOAT(0, "isp", &(data.sim.seq.inversion_spoiler), "float", "Inversion Spoiler Gradient Length [s]"),
		OPTL_FLOAT(0, "ppl", &(data.sim.seq.prep_pulse_length), "float", "Preparation Pulse Length [s]"),
		OPTL_PINT(0, "av-spokes", &(data.sim.seq.averaged_spokes), "", "Number of averaged consecutive spokes"),

		/* Pulse Specific Parameters */
		OPTL_FLOAT(0, "Trf", &(data.sim.pulse.rf_end), "float", "Pulse Duration [s]"), /* Assumes to start at t=0 */
		OPTL_FLOAT(0, "FA", &(CAST_UP(&data.sim.pulse.sinc)->flipangle), "float", "Flipangle [deg]"),
		OPTL_FLOAT(0, "BWTP", &(data.sim.pulse.sinc.bwtp), "float", "Bandwidth-Time-Product"),

		/* Voxel Specific Parameters */
		OPTL_FLOAT(0, "off", &(data.sim.voxel.w), "float", "Off-Resonance [rad/s]"),

		/* Slice Profile Parameters */
		OPTL_FLOAT(0, "sl-grad", &(data.sim.grad.sl_gradient_strength), "float", "Strength of slice-selection gradient [T/m]"),
		OPTL_FLOAT(0, "slice-thickness", &(data.sim.seq.slice_thickness), "float", "Thickness of simulated slice. [m]"),
		OPTL_FLOAT(0, "nom-slice-thickness", &(data.sim.seq.nom_slice_thickness), "float", "Nominal thickness of simulated slice. [m]"),
	};

	struct opt_s sim_opts[] = {

		OPTL_SELECT(0, "ODE", enum sim_type, &(data.sim.seq.type), SIM_ODE, "full ordinary differential equation solver based simulation"),
 		OPTL_SELECT(0, "STM", enum sim_type, &(data.sim.seq.type), SIM_STM, "state-transition matrix based simulation (default)"),
	};


	int tvscales_N = 4;
	float tvscales[4] = { 0. };

	struct opt_s other_opts[] = {

		// FIXME: MGRE can have 5 parameters
		OPTL_FLVECN(0, "pscale", data.other.scale,"Scaling of parameters in model-based reconstruction"),
		OPTL_FLVECN(0, "pinit", data.other.initval, "Initial values of parameters in model-based reconstruction"),
		OPTL_INFILE(0, "b1map", &input_b1, "[deg]", "Input B1 map as cfl file"),
		OPTL_INFILE(0, "b0map", &input_b0, "[rad/s]", "Input B0 map as cfl file"),
		OPTL_INFILE(0, "ksp-sens", &input_sens, "", "Input kspace sensitivities"),
		OPTL_INFILE(0, "echo", &input_TE, "", "Input Echo times for IR multi-echo gradient-echo [ms]"), // FIXME: SI units here!
		OPTL_FLVEC4(0, "tvscale", &tvscales, "s1:s2:s3:s4", "Scaling of derivatives in TV penalty"),
		OPTL_FLOAT(0, "b1-sobolev-a", &(data.other.b1_sobolev_a), "", "(a in 1 + a * \\Laplace^-b/2)"),
		OPTL_FLOAT(0, "b1-sobolev-b", &(data.other.b1_sobolev_b), "", "(a in 1 + a * \\Laplace^-b/2)"),
		OPTL_FLOAT(0, "ode-tol", &(data.sim.other.ode_tol), "f", "ODE tolerance value [def: 1e-5]"),
		OPTL_FLOAT(0, "stm-tol", &(data.sim.other.stm_tol), "f", "STM tolerance value [def: 1e-6]"),
		OPTL_SET(0,"no-sens-l2", &data.other.no_sens_l2, "(Turn off l2 regularization on coils)"),
		OPTL_SET(0,"no-sens-deriv", &data.other.no_sens_deriv, "(Turn off coil updates)"),
		OPTL_SET(0,"export-ksp-sens", &data.other.export_ksp_coils, "(Export coil sensitivities in ksp)"),
	};

	opt_reg_init(&ropts);

	bool t2_old_flag = false;

	const struct opt_s opts[] = {

		// FIXME: Sort options into optimization and others interface
		{ 'r', NULL, true, OPT_SPECIAL, opt_reg_moba, &ropts, "<T>:A:B:C", "generalized regularization options (-rh for help)" },
		OPT_SELECT('L', enum mdb_t, &conf.mode, MDB_T1, "T1 mapping using model-based look-locker"),
		OPT_SELECT('P', enum mdb_t, &conf.mode, MDB_T1_PHY, "T1 mapping using reparameterized (M0, R1, alpha) model-based look-locker (TR required!)"),
		OPT_SET('F', &t2_old_flag, "(T2 mapping using model-based Fast Spin Echo)"),
		OPT_SELECT('T', enum mdb_t, &conf.mode, MDB_T2, "T2 mapping using model-based Fast Spin Echo"),
		OPT_SELECT('G', enum mdb_t, &conf.mode, MDB_MGRE, "T2* mapping using model-based multiple gradient echo"),
		OPT_SELECT('D', enum mdb_t, &conf.mode, MDB_IR_MGRE, "Joint T1 and T2* mapping using model-based IR multiple gradient echo"),
		OPTL_SELECT(0, "bloch", enum mdb_t, &conf.mode, MDB_BLOCH, "Bloch model-based reconstruction"),
		OPT_UINT('m', &conf.mgre_model, "model", "Select the MGRE model from enum { WF = 0, WFR2S, WF2R2S, R2S, PHASEDIFF, PI, T1_R2S, W_T1_F_T1_RS2 } [default: WFR2S]"),
		OPT_PINT('l', &conf.opt_reg, "\b1/-l2", "  toggle l1-wavelet or l2 regularization."), // extra spaces needed because of backspace \b earlier
		OPT_PINT('i', &conf.iter, "iter", "Number of Newton steps"),
		OPTL_FLOAT('R', "reduction", &conf.redu, "redu", "reduction factor"),
		OPT_FLOAT('j', &conf.alpha_min, "minreg", "Minimum regularization parameter"),
		OPT_FLOAT('u', &conf.rho, "rho", "ADMM rho [default: 0.01]"),
		OPT_PINT('C', &conf.inner_iter, "iter", "inner iterations"),
		OPTL_FLOAT(0, "tol", &conf.tolerance, "tol", "tolerance for fista early stopping (default: 0.01)"),
		OPT_FLOAT('s', &conf.step, "step", "step size"),
		OPT_FLOAT('B', &conf.lower_bound, "bound", "lower bound for relaxation"),
		OPT_FLVEC2('b', &conf.scale_fB0, "a:b", "B0 field: sobolev parameter (a=0 means no sobolev) [default: 222.; 32.]"),
		OPT_INT('d', &debug_level, "level", "Debug level"),
		OPT_SET('N', &conf.auto_norm, "(normalize)"),
		OPT_FLOAT('f', &restrict_fov, "FOV", ""),
		OPT_INFILE('p', &psf_file, "PSF", ""),
		OPT_SET('J', &conf.stack_frames, "Stack frames for joint recon"),
		OPT_SET('M', &conf.sms, "Simultaneous Multi-Slice reconstruction"),
		OPT_SET('O', &conf.out_origin_maps, "(Output original maps from reconstruction without post processing)"),
		OPT_SET('g', &bart_use_gpu, "use gpu"),
		OPTL_ULONG(0, "positive-maps", &conf.constrained_maps, "flag", "Maps with positivity constraint as FLAG!"),
		OPTL_PINT(0, "not-wav-maps", &conf.not_wav_maps, "d", "Maps removed from wavelet denoising (counted from back!)"),
		OPTL_ULONG(0, "l2-on-parameters", &conf.l2para, "flag", "Flag for parameter maps with l2 norm"),
		OPTL_PINT(0, "pusteps", &conf.pusteps, "ud", "Number of partial update steps for IRGNM"),
		OPTL_FLOAT(0, "ratio", &conf.ratio, "f:[0;1]", "Ratio of partial updates: ratio*<updated-map> + (1-ratio)*<previous-map>"),
		OPTL_FLOAT(0, "l1val", &conf.l1val, "f", "Regularization scaling of l1 wavelet (default: 1.)"),
		OPTL_FLOAT(0, "temporal_damping", &conf.damping, "f", "Temporal damping factor."),
		OPTL_INT(0, "multi-gpu", &(conf.num_gpu), "num", "(number of gpus to use)"),
		OPT_INFILE('I', &init_file, "init", "File for initialization"),
		OPT_INFILE('t', &traj_file, "traj", "K-space trajectory"),
		OPTL_INFILE(0, "sens", &fixed_sens, "sens", "Use precomputed sensitivities"),
		OPT_FLOAT('o', &oversampling, "os", "Oversampling factor for gridding [default: 1.]"),
		OPTL_VEC3('x', "img_dims", &img_vec, "x:y:z", "dimensions"),
		OPT_SET('k', &conf.k_filter, "k-space edge filter for non-Cartesian trajectories"),
		OPTL_SELECT(0, "kfilter-1", enum edge_filter_t, &conf.k_filter_type, EF1, "k-space edge filter 1"),
		OPTL_SELECT(0, "kfilter-2", enum edge_filter_t, &conf.k_filter_type, EF2, "k-space edge filter 2"),
		OPT_FLOAT('e', &kfilter_strength, "kfilter_strength", "strength for k-space edge filter [default: 2e-3]"),
		OPT_CLEAR('n', &conf.auto_norm, "(disable normalization of parameter maps for thresholding)"),
		OPTL_CLEAR(0, "no_alpha_min_exp_decay", &conf.alpha_min_exp_decay, "(Use hard minimum instead of exponential decay towards alpha_min)"),
		OPTL_FLOAT(0, "sobolev_a", &conf.sobolev_a, "", "(a in 1 + a * \\Laplace^-b/2)"),
		OPTL_FLOAT(0, "sobolev_b", &conf.sobolev_b, "", "(b in 1 + a * \\Laplace^-b/2)"),
		OPTL_SELECT(0, "fat_spec_0", enum fat_spec, &conf.fat_spec, FAT_SPEC_0, "select fat spectrum from ISMRM fat-water tool"),
		OPTL_FLOAT(0, "scale_data", &scaling, "", "scaling factor for data"),
		OPTL_FLOAT(0, "scale_psf", &scaling_psf, "", "(scaling factor for PSF)"),
		OPTL_SET(0, "normalize_scaling", &normalize_scaling, "(normalize scaling by data / PSF)"),
		OPTL_SUBOPT(0, "seq", "...", "configure sequence parameters", ARRAY_SIZE(seq_opts), seq_opts),
		OPTL_SUBOPT(0, "sim", "...", "configure simulation parameters", ARRAY_SIZE(sim_opts), sim_opts),
		OPTL_SUBOPT(0, "other", "...", "configure other parameters", ARRAY_SIZE(other_opts), other_opts),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	if (0 != conf.num_gpu)
		error("Multi-GPU only supported by MPI!\n");

	num_init_gpu_support();
	num_rand_init(0ULL);

	if (MDB_T1_PHY == conf.mode)
		debug_printf(DP_INFO, "The TR for MDB_T1_PHY is %f s!\n", data.sim.seq.tr);

	// debug_sim(&(data.sim));
	// debug_other(&(data.other));
	if (use_compat_to_version("v0.6.00"))
		conf.scaling_M0 = 2.;

	if (t2_old_flag)
		conf.mode = MDB_T2;

	data.model = conf.mode;

	if (conf.ropts->r > 0)
		conf.algo = ALGO_ADMM;

	while ((0 < tvscales_N) && (0. == tvscales[tvscales_N - 1]))
		tvscales_N--;

	data.other.tvscales_N = tvscales_N;

	for (int i = 0; i < tvscales_N; i++)
		data.other.tvscales[i] = tvscales[i];



	long ksp_dims[DIMS];
	complex float* kspace_data = load_cfl(ksp_file, DIMS, ksp_dims);

	long TI_dims[DIMS];
	complex float* TI = load_cfl(TI_file, DIMS, TI_dims);

	if (t2_old_flag)
		md_zsmul(DIMS, TI_dims, TI, TI, 10.);

	assert(TI_dims[TE_DIM] == ksp_dims[TE_DIM]);
	assert(1 == ksp_dims[MAPS_DIM]);

	long grid_dims[DIMS];
	md_copy_dims(DIMS, grid_dims, ksp_dims);

	complex float* cim = NULL;

	complex float* pattern = NULL;
	long pat_dims[DIMS];


	if (NULL != psf_file) {

		complex float* tmp_psf = load_cfl(psf_file, DIMS, pat_dims);

		pattern = anon_cfl("", DIMS, pat_dims);

		md_copy(DIMS, pat_dims, pattern, tmp_psf, CFL_SIZE);

		unmap_cfl(DIMS, pat_dims, tmp_psf);

		cim = md_alloc_sameplace(DIMS, grid_dims, CFL_SIZE, kspace_data);

		ifftuc(DIMS, grid_dims, FFT_FLAGS, cim, kspace_data);

		unmap_cfl(DIMS, ksp_dims, kspace_data);

		if (!md_check_compat(DIMS, COIL_FLAG, ksp_dims, pat_dims))
			error("pattern not compatible with kspace dimensions\n");

		if (-1 == restrict_fov)
			restrict_fov = 0.5;

		conf.noncartesian = true;

	} else if (NULL != traj_file) {

		long traj_dims[DIMS];
		complex float* traj = load_cfl(traj_file, DIMS, traj_dims);

		md_zsmul(DIMS, traj_dims, traj, traj, oversampling);

		if (0 == md_calc_size(3, img_vec)) {

			long tmp_dims[DIMS];
			estimate_im_dims(DIMS, FFT_FLAGS, tmp_dims, traj_dims, traj);
			md_copy_dims(3, img_vec, tmp_dims);
			debug_printf(DP_INFO, "Est. image size: %ld %ld %ld\n", img_vec[0], img_vec[1], img_vec[2]);
		}

		float scl_trj = 1.;
		float scl_psf = 1.;

		NESTED(long, dbl, (long x)) { return (x > 1) ? (2 * x) : 1; };

		if (use_compat_to_version("v0.7.00")) {

			long grid_size = ksp_dims[1] * oversampling;
			grid_dims[READ_DIM] = grid_size;
			grid_dims[PHS1_DIM] = grid_size;
			grid_dims[PHS2_DIM] = 1L;

		} else if (use_compat_to_version("v1.0.00")) {

			md_zsmul(DIMS, traj_dims, traj, traj, 2.);

			for (int i = 0; i < 3; i++)
				grid_dims[i] = dbl(img_vec[i]);

		} else {

			scl_trj = 2.;
			scl_psf = powf(2., bitcount(md_nontriv_dims(3, img_vec)));
			data.other.sobolev_os = 2.;

			for (int i = 0; i < 3; i++)
				grid_dims[i] = img_vec[i];
		}

		if (-1 == restrict_fov)
			restrict_fov = 0.5 * scl_trj;

		conf.noncartesian = true;

		// Gridding raw data

		struct nufft_conf_s nufft_conf = nufft_conf_defaults;
		nufft_conf.toeplitz = false;

		const struct linop_s* nufft_op_k = nufft_create(DIMS, ksp_dims, grid_dims, traj_dims, traj, NULL, nufft_conf);

		cim = md_alloc_sameplace(DIMS, grid_dims, CFL_SIZE, kspace_data);

		linop_adjoint(nufft_op_k, DIMS, grid_dims, cim, DIMS, ksp_dims, kspace_data);

		linop_free(nufft_op_k);

		md_select_dims(DIMS, FFT_FLAGS|TE_FLAG|CSHIFT_FLAG|TIME_FLAG|SLICE_FLAG|TIME2_FLAG, pat_dims, grid_dims);

		if (2. == scl_trj)
			for (int i = 0; i < 3; i++)
				pat_dims[i] = dbl(pat_dims[i]);

		md_zsmul(DIMS, traj_dims, traj, traj, scl_trj);

		pattern = anon_cfl("", DIMS, pat_dims);

		// Gridding sampling pattern

		complex float* psf = NULL;

		long wgh_dims[DIMS];
		md_select_dims(DIMS, ~COIL_FLAG, wgh_dims, ksp_dims);

		complex float* wgh = md_alloc(DIMS, wgh_dims, CFL_SIZE);

		estimate_pattern(DIMS, ksp_dims, COIL_FLAG, wgh, kspace_data);

		psf = compute_psf(DIMS, pat_dims, traj_dims, traj, traj_dims, NULL, wgh_dims, wgh, false, false);

		md_zsmul(DIMS, pat_dims, psf, psf, scl_psf);

		fftuc(DIMS, pat_dims, FFT_FLAGS, pattern, psf);

		md_free(wgh);
		md_free(psf);

		unmap_cfl(DIMS, ksp_dims, kspace_data);
		unmap_cfl(DIMS, traj_dims, traj);

	} else {

		md_select_dims(DIMS, ~COIL_FLAG, pat_dims, grid_dims);

		pattern = anon_cfl("", DIMS, pat_dims);

		estimate_pattern(DIMS, ksp_dims, COIL_FLAG, pattern, kspace_data);

		cim = md_alloc_sameplace(DIMS, grid_dims, CFL_SIZE, kspace_data);

		ifftuc(DIMS, grid_dims, FFT_FLAGS, cim, kspace_data);

		unmap_cfl(DIMS, ksp_dims, kspace_data);
	}

	if (conf.sms) {

		debug_printf(DP_INFO, "SMS Model-based reconstruction. Multiband factor: %ld\n", ksp_dims[SLICE_DIM]);
		ifft(DIMS, grid_dims, SLICE_FLAG, cim, cim);

		// FIXME: maybe this can go, but before sclaing was normalized with respect to k-space
		if (normalize_scaling)
			scaling *= sqrt((float)ksp_dims[SLICE_DIM]);
	}

	long img_dims[DIMS];

	md_select_dims(DIMS, FFT_FLAGS|MAPS_FLAG|COEFF_FLAG|TIME_FLAG|SLICE_FLAG|TIME2_FLAG, img_dims, grid_dims);
	img_dims[COEFF_DIM] = moba_get_nr_of_coeffs(&conf, grid_dims[TE_DIM]); // grid_dims[TE_DIM] is only used for MECO_PI == conf.mgre_model

	long img_strs[DIMS];
	md_calc_strides(DIMS, img_strs, img_dims, CFL_SIZE);

	complex float* img = create_cfl(out_file, DIMS, img_dims);
	md_zfill(DIMS, img_dims, img, 1.);

	long dims[DIMS];
	md_copy_dims(DIMS, dims, grid_dims);

	dims[COEFF_DIM] = img_dims[COEFF_DIM];

	long coil_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|COIL_FLAG|MAPS_FLAG|TIME_FLAG|SLICE_FLAG|TIME2_FLAG, coil_dims, grid_dims);

	complex float* sens = NULL;

	if (NULL != fixed_sens) {

		long tmp_dims[DIMS];

		sens = load_cfl(fixed_sens, DIMS, tmp_dims);

		assert(md_check_equal_dims(DIMS, tmp_dims, coil_dims, ~0UL));
		assert(NULL == sens_file);

		md_copy_dims(DIMS, tmp_dims, grid_dims);
		md_select_dims(DIMS, ~COIL_FLAG, grid_dims, grid_dims);

		complex float* adj = md_alloc_sameplace(DIMS, grid_dims, CFL_SIZE, cim);

		md_ztenmulc(DIMS, grid_dims, adj, tmp_dims, cim, coil_dims, sens);

		md_free(cim);
		cim = adj;

		data.other.fixed_coil = true;

	} else if (NULL != input_sens) {

		sens = ((NULL != sens_file) ? create_cfl : anon_cfl)(sens_file, DIMS, coil_dims);

		long in_sens_dims[DIMS];

		const complex float* in_sens = load_cfl(input_sens, DIMS, in_sens_dims);

		assert(md_check_equal_dims(DIMS, coil_dims, in_sens_dims, ~0UL));

		md_copy(DIMS, coil_dims, sens, in_sens, CFL_SIZE);

		unmap_cfl(DIMS, in_sens_dims, in_sens);

	} else {

		sens = ((NULL != sens_file) ? create_cfl : anon_cfl)(sens_file, DIMS, coil_dims);

		md_clear(DIMS, coil_dims, sens, CFL_SIZE);
	}


	if (conf.k_filter) {

		long map_dims[DIMS];
		md_select_dims(DIMS, FFT_FLAGS, map_dims, pat_dims);

		long map_strs[DIMS];
		md_calc_strides(DIMS, map_strs, map_dims, CFL_SIZE);

		long pat_strs[DIMS];
		md_calc_strides(DIMS, pat_strs, pat_dims, CFL_SIZE);

		complex float* filter = md_alloc(DIMS, map_dims, CFL_SIZE);

		switch (conf.k_filter_type) {

		case EF1:
			edge_filter1(map_dims, filter, kfilter_strength);
			break;

		case EF2:
			edge_filter2(map_dims, filter, kfilter_strength);
			break;
		}

		md_zadd2(DIMS, pat_dims, pat_strs, pattern, pat_strs, pattern, map_strs, filter);

		md_free(filter);
	}

	// read initialization file

	long init_dims[DIMS] = { [0 ... DIMS-1] = 1 };
	complex float* init = NULL;

	if (NULL != init_file) {

		init = load_cfl(init_file, DIMS, init_dims);

		for (int i = 0; i < (int)ARRAY_SIZE(data.other.initval); i++)
			if (1. != data.other.initval[i])
				error("Cannot provide initialization value and initialization file!\n");

		if (!md_check_equal_dims(DIMS, img_dims, init_dims, ~0UL))
			error("Initialization dimensions do not match image dimensions!\n");
	}

	// Load passed B1

	const complex float* b1 = NULL;
	long b1_dims[DIMS];

	if (NULL != input_b1) {

		b1 = load_cfl(input_b1, DIMS, b1_dims);

		assert(md_check_compat(DIMS, ~FFT_FLAGS, grid_dims, b1_dims));
	}

	// Load passed B0

        const complex float* b0 = NULL;
	long b0_dims[DIMS];

	if (NULL != input_b0) {

		b0 = load_cfl(input_b0, DIMS, b0_dims);

		assert(md_check_compat(DIMS, ~FFT_FLAGS, grid_dims, b0_dims));
	}

	// Load TE for IR MGRE

	const complex float* TE_IR_MGRE = NULL;
	long TE_IR_MGRE_dims[DIMS];

	if (MDB_IR_MGRE == conf.mode)
		TE_IR_MGRE = load_cfl(input_TE, DIMS, TE_IR_MGRE_dims);

	// scaling

	if (normalize_scaling) {

		scaling /= md_znorm(DIMS, grid_dims, cim);
		scaling_psf /= md_znorm(DIMS, pat_dims, pattern);
	}

	if (1. != scaling) {

		debug_printf(DP_INFO, "Scaling: %f\n", scaling);
		md_zsmul(DIMS, grid_dims, cim, cim, scaling);
	}

	if (1. != scaling_psf) {

		debug_printf(DP_INFO, "Scaling_psf: %f\n", scaling_psf);
		md_zsmul(DIMS, pat_dims, pattern, pattern, scaling_psf);
	}


	// mask

	if (-1. != restrict_fov) {

		// mask is not needed since we compute the model only on the restricted FOV

		float restrict_dims[DIMS] = { [0 ... DIMS - 1] = 1. };
		restrict_dims[0] = restrict_fov;
		restrict_dims[1] = restrict_fov;
		restrict_dims[2] = restrict_fov;

		long msk_dims[DIMS];
		md_select_dims(DIMS, FFT_FLAGS, msk_dims, img_dims);

		complex float* mask = compute_mask(DIMS, msk_dims, restrict_dims);

		data.other.fov_reduction_factor = restrict_fov;

		//FIXME: this may be bad for any map regularized by Sobolev,
		// 	 as it will create sharp edges in the initialization
		if (MDB_BLOCH != conf.mode)
		        md_zmul2(DIMS, img_dims, img_strs, img, img_strs, img, MD_STRIDES(DIMS, msk_dims, CFL_SIZE), mask);

		md_free(mask);
	}

	// Scale parameter maps

	long tmp_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|MAPS_FLAG|TIME_FLAG|SLICE_FLAG|TIME2_FLAG, tmp_dims, grid_dims);

	complex float* tmp = md_alloc(DIMS, tmp_dims, CFL_SIZE);

	long pos[DIMS] = { [0 ... DIMS - 1] = 0 };

	assert(img_dims[COEFF_DIM] <= (long)ARRAY_SIZE(data.other.scale));

	// Transform B1 map from image to k-space and add k-space to initialization array (img)

	unsigned long sobolev_flag = 0;

	sobolev_flag |= (MDB_T1_PHY == conf.mode) ? MD_BIT(2) : 0;
	sobolev_flag |= (MDB_BLOCH == conf.mode) ? MD_BIT(3) : 0;

	for (int i = 0; i < img_dims[COEFF_DIM]; i++) {

		pos[COEFF_DIM] = i;

		md_copy_block(DIMS, pos, tmp_dims, tmp, img_dims, (NULL != init) ? init : img, CFL_SIZE);

		md_zsmul(DIMS, tmp_dims, tmp, tmp, data.other.initval[i] / (data.other.scale[i] ?: 1));

		if (MD_IS_SET(sobolev_flag, i) && (NULL == init)) {

			fftuc(DIMS, tmp_dims, FFT_FLAGS, tmp, tmp);

			float scl = powf(data.other.sobolev_os, bitcount(md_nontriv_dims(DIMS, tmp_dims) & FFT_FLAGS) / 2.);
			md_zsmul(DIMS, tmp_dims, tmp, tmp, scl);
		}

		md_copy_block(DIMS, pos, img_dims, img, tmp_dims, tmp, CFL_SIZE);
	}

#ifdef  USE_CUDA
	if (bart_use_gpu) {

		complex float* cim_gpu = md_gpu_move(DIMS, grid_dims, cim, CFL_SIZE);

		md_free(cim);

		cim = cim_gpu;
	}
#endif

	moba_recon(&conf, &data, dims, img_dims, img, coil_dims, sens, pat_dims, pattern, TI, TE_IR_MGRE, b1, b0, grid_dims, cim, init);

	// Rescale estimated parameter maps

	for (int i = 0; i < img_dims[COEFF_DIM]; i++) {

		pos[COEFF_DIM] = i;

		md_copy_block(DIMS, pos, tmp_dims, tmp, img_dims, img, CFL_SIZE);

		md_zsmul(DIMS, tmp_dims, tmp, tmp, (data.other.scale[i] ?: 1.));

		md_copy_block(DIMS, pos, img_dims, img, tmp_dims, tmp, CFL_SIZE);
	}

	md_free(tmp);
	md_free(cim);

	unmap_cfl(DIMS, coil_dims, sens);
	unmap_cfl(DIMS, pat_dims, pattern);
	unmap_cfl(DIMS, img_dims, img);
	unmap_cfl(DIMS, TI_dims, TI);
	unmap_cfl(DIMS, init_dims, init);
	unmap_cfl(DIMS, b1_dims, b1);
	unmap_cfl(DIMS, TE_IR_MGRE_dims, TE_IR_MGRE);
	unmap_cfl(DIMS, b0_dims, b0);

	double recosecs = timestamp() - start_time;

	debug_printf(DP_DEBUG2, "Total Time: %.2f s\n", recosecs);

	return 0;
}

