/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef __CALIB_H
#define __CALIB_H

#include "misc/dimtypes.h"
#include "misc/cppwrap.h"
#include "misc/mri.h"

struct ecalib_conf {

	bart_dim_t kdims[3];
	float threshold;
	int numsv;
	float percentsv;
	_Bool weighting;
	_Bool softcrop;
	float crop;
	_Bool orthiter;
	int num_orthiter;
	_Bool usegpu;
	float perturb;
	_Bool intensity;
	_Bool rotphase;
	float var;
	_Bool automate;
	_Bool phase_normalize;
	int econdim;
	_Bool nystroem;
	bart_dim_t nystroem_os;
	bart_dim_t nystroem_K;
};

extern const struct ecalib_conf ecalib_defaults;

extern void calib(const struct ecalib_conf* conf, const bart_dim_t out_dims[DIMS], _Complex float* out_data, _Complex float* eptr,
			int SN, float svals[__VLA2(SN)], const bart_dim_t calreg_dims[DIMS], const _Complex float* calreg_data);

extern void calib2(const struct ecalib_conf* conf, const bart_dim_t out_dims[DIMS], _Complex float* out_data, _Complex float* eptr, int SN, float svals[__VLA2(SN)], const bart_dim_t calreg_dims[DIMS], const _Complex float* data, const bart_dim_t msk_dims[3], const _Bool* msk);

extern void eigenmaps(const bart_dim_t out_dims[DIMS], _Complex float* out_data, _Complex float* eptr, const _Complex float* imgcov, const bart_dim_t msk_dims[3], const _Bool* msk, _Bool orthiter, int num_orthiter, _Bool usegpu);


extern void crop_sens(const bart_dim_t dims[DIMS], _Complex float* ptr, bool soft, float crth, const _Complex float* map);

extern void calone_dims(const struct ecalib_conf* conf, bart_dim_t cov_dims[4], bart_dim_t channels);
extern void calone(const struct ecalib_conf* conf, const bart_dim_t cov_dims[4], _Complex float* cov, int SN, float svals[__VLA2(SN)], const bart_dim_t calreg_dims[DIMS], const _Complex float* cal_data);
extern void caltwo(const struct ecalib_conf* conf, const bart_dim_t out_dims[DIMS], _Complex float* out_data, _Complex float* emaps, const bart_dim_t in_dims[4], _Complex float* in_data, const bart_dim_t msk_dims[3], const _Bool* msk);
extern void compute_imgcov(const bart_dim_t cov_dims[4], _Complex float* imgcov, const bart_dim_t nskerns_dims[5], const _Complex float* nskerns);
extern void compute_kernels(const struct ecalib_conf* conf, bart_dim_t nskerns_dims[5], _Complex float** nskerns_ptr, int SN, float svals[__VLA2(SN)], const bart_dim_t caldims[DIMS], const _Complex float* caldata);

#include "misc/cppwrap.h"
#endif	// __CALIB_H
