/* Copyright 2025-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "seq/config.h"
#include "seq/event.h"
#include "seq/adc_rf.h"
#include "seq/gradient.h"
#include "seq/helpers.h"

#include "mag_prep.h"

int mag_prep(struct seq_event ev[6], const struct seq_config* seq)
{
	if ((SEQ_PREP_IR_NONSELECTIVE != seq->magn.mag_prep) &&  (SEQ_PREP_IR_SELECTIVE != seq->magn.mag_prep))
		return 0;

	int i = 0;
	double proj_slice[3] = { 0., 0., 1. };

	double inv_start = 800.E-6;

	if (SEQ_PREP_IR_SELECTIVE == seq->magn.mag_prep) {

		struct grad_trapezoid slice = {
			.ampl = 18.E-6 / seq->geom.slice_thickness, // FIXME: empirical
			.rampup = 0.8E-3,
			.flat = 10.E-3,
			.rampdown = 0.8E-3,
		};

		i += seq_grad_to_event(ev + i, 0., &slice, proj_slice);
		inv_start = ev[i - 1].mid;
	}

	i += prep_rf_inversion(ev + i, inv_start, seq);

	struct grad_trapezoid spoil = {

		.ampl = .008,
		.rampup = 800E-6,
		.flat = 8200E-6,
		.rampdown = 600E-6,
	};

	i += seq_grad_to_event(ev + i, ev[i - 1].end, &spoil, proj_slice);
	i += wait_time_to_event(ev + i, ev[i - 1].end, seq->magn.ti);

	return i;
}

