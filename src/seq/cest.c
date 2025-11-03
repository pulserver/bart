/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Huemer M, Stilianu C, Scholand N, Mackner D, Uecker M, Zaiss M, Stollberger R.
 * Dynamic Transitions for Fast Joint Acquisition and Reconstruction of CEST-Rex and T1.
 * Magn Reson Med 2025; 95:2153-68.
 *
 * Mennecke A, Khakzar KM, German A, Herz K, Fabian MS, Liebert A, Blümcke I, Kasper BS,
 * Nagel AM, Laun FB, Schmidt M, Winkler J, Doerfler A, Zaiss M.
 * 7 tricks for 7 T CEST: Improving the reproducibility of multipool evaluation provides
 * insights into the effects of age and the early stages of Parkinson's disease.
 * NMR in Biomed 2026; 36:e4717.
 */

#include <math.h>

#include "seq/config.h"
#include "seq/event.h"
#include "seq/adc_rf.h"
#include "seq/pulse.h"
#include "seq/gradient.h"
#include "seq/flash.h"

#include "cest.h"


#define SEQ_CEST_NUM_OFFSETS_PHA 58

static const double cest_offsets_pha[SEQ_CEST_NUM_OFFSETS_PHA] = { // Huemer et al.

	-1500, -5, -4.75, -4.5, -4.25, -4, -3.75, -3.5, -3.25, -3, -2.75, 
	-2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25,
	0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25,
	2.5, 2.75, 3, 3.25, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 
	3.9, 4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8,
	4.9, 5, 5.1, 5.2, 5.5, 5.75, 6
};


#define SEQ_CEST_NUM_OFFSETS_INVIVO 57

static const double cest_offsets_invivo[SEQ_CEST_NUM_OFFSETS_INVIVO] = { // Mennecke et al.

	-1500, -300, -100, -50, -20, -12, -9, -7.25, -6.25, -5.5, -4.7,
	-4, -3.3, -2.7, -2, -1.7, -1.5, -1.1, -0.9, -0.6, -0.4, 
	0, 0.4, 0.6, 0.95, 1.1, 1.25, 1.4, 1.55, 1.7, 1.85,
	2, 2.15, 2.3, 2.45, 2.6, 2.75, 2.9, 3.05, 3.2, 3.35,
	3.5, 3.65, 3.8, 3.95, 4.1, 4.25, 4.4, 4.7, 5.25, 6.25,
	8, 12, 20, 50, 100, 300
};


long cest_offsets(const struct seq_config *seq_config)
{
	if (SEQ_CEST_NONE == seq_config->cest.sat_type)
		return 1;

	if (SEQ_CEST_OFFSET_EQUIDISTANT == seq_config->cest.offset_type)
		return (long)((fabs(seq_config->cest.offset_first) + fabs(seq_config->cest.offset_last)) / seq_config->cest.offset_increment) + 1;
	else if (SEQ_CEST_OFFSET_PHANTOM == seq_config->cest.offset_type)
		return SEQ_CEST_NUM_OFFSETS_PHA;
	else if (SEQ_CEST_OFFSET_INVIVO == seq_config->cest.offset_type)
		return SEQ_CEST_NUM_OFFSETS_INVIVO;

	return 1;
}

static int prep_grad_spoiler(struct grad_trapezoid* grad, double moment, double duration, struct grad_limits limits)
{
	*grad = (struct grad_trapezoid){ 0 };

	grad_soft(grad, duration, moment, limits);

	return 1;
}

static double calc_offset(long pos_cshift, const struct seq_config* seq_config)
{
	double ppm = 0.;

	if (SEQ_CEST_OFFSET_EQUIDISTANT == seq_config->cest.offset_type)
		ppm = seq_config->cest.offset_first + pos_cshift * seq_config->cest.offset_increment;
	else if (SEQ_CEST_OFFSET_PHANTOM == seq_config->cest.offset_type)
		ppm = cest_offsets_pha[pos_cshift];
	else if (SEQ_CEST_OFFSET_INVIVO == seq_config->cest.offset_type)
		ppm = cest_offsets_invivo[pos_cshift];
	return 1E-6 * ppm * seq_config->sys.b0 * seq_config->sys.gamma;
}

static int prep_rf_cest(struct seq_event* rf_ev, double start, const struct seq_state* seq_state, const struct seq_config* seq_config)
{
	if (SEQ_CEST_NONE == seq_config->cest.sat_type)
		return 0;

	rf_ev->type = SEQ_EVENT_PULSE;

	rf_ev->pulse.shape_id = seq_config->geom.mb_factor;

	if ((SEQ_PREP_IR_NONSELECTIVE == seq_config->magn.mag_prep) || (SEQ_PREP_IR_SELECTIVE == seq_config->magn.mag_prep))
		rf_ev->pulse.shape_id += 1;

	rf_ev->start = start;

	rf_ev->pulse.type = SEQ_RF_EXCITATION;

	rf_ev->pulse.freq = calc_offset(seq_state->pos[CSHIFT_DIM], seq_config);
	rf_ev->pulse.phase = 0.;

	double fa = 0.;
	double dur = 0.;
	
	if (SEQ_CEST_GAUSS == seq_config->cest.sat_type) {

		fa = seq_config->cest.gauss_pulse_fa;
		dur = seq_config->cest.gauss_pulse_duration;
	}
	else if (SEQ_CEST_OC == seq_config->cest.sat_type) {

		struct pulse_arb arb = pulse_arb_oc_cest_sat_defaults;

		fa = arb.super.flipangle;
		dur = arb.super.duration;
	}

	rf_ev->pulse.fa = fa;

	const double asym_pulse = 0.5;
	rf_ev->end = rf_ev->start + dur;
	rf_ev->mid = rf_ev->start + (rf_ev->end - rf_ev->start) * asym_pulse;

	return 1;
}

int cest_block(struct seq_event ev[6], const struct seq_state* seq_state, const struct seq_config* seq_config)
{
	struct grad_trapezoid spoiler;
	double projSPOIL[3] = { 1. , 1. , 0. };
	
	int i = 0;
	if (((seq_config->loop_dims[COEFF2_DIM] - 2) == seq_state->pos[COEFF2_DIM])) {

		if (!prep_grad_spoiler(&spoiler, 0.1E-3, 10E-3, seq_config->sys.grad))
			return ERROR_PREP_GRAD_SP_READ;

		i += seq_grad_to_event(ev + i, 0., &spoiler, projSPOIL);
		return i;
	}

	double start_rf = seq_config->sys.coil_control_lead;

	if ((0 < seq_state->pos[CSHIFT_DIM]) && (2 == seq_state->pos[COEFF2_DIM]) && 0 < seq_config->cest.offset_pause) {

		i += wait_time_to_event(ev, start_rf, seq_config->cest.offset_pause);
		start_rf = ev[i - 1].end;
	}

	i += prep_rf_cest(ev + i, start_rf, seq_state, seq_config);

	i += wait_time_to_event(ev + i, ev[i - 1].end, seq_config->cest.sat_pulse_pause + seq_config->sys.coil_control_lead);

	return i;
}
