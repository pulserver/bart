/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>
#include <assert.h>

#include "misc/mri.h"
#include "misc/misc.h"

#include "seq/event.h"

#include "gradient.h"

double grad_duration(const struct grad_trapezoid* grad)
{
	return grad->rampup + grad->flat;
}

double grad_total_time(const struct grad_trapezoid* grad)
{
	return (grad->rampup + grad->flat + grad->rampdown);
}



double grad_momentum(struct grad_trapezoid* grad)
{
	return grad->ampl * (0.5 * grad->rampup + grad->flat + 0.5 * grad->rampdown);
}

/*
 * arbitrary gradient support points to triangular event list
 *
 */
int seq_grad_support_to_event(int E, struct seq_event ev[E], double start_offset, int N, const struct grad_support grad[N], double proj[3])
{
	if (2 > E)
		return -1;

	int e = 0;

	for (int n = 0; n < N; n++) {

		if (grad[MIN(n, n + 1)].time < grad[n].time)
			return -1;

		if (0. == grad[n].ampl)
			continue;

		ev[e].type = SEQ_EVENT_GRADIENT;
		ev[e].start = start_offset + grad[MAX(0, n - 1)].time;
		ev[e].mid = start_offset + grad[n].time;
		ev[e].end = start_offset + grad[MIN(N - 1, n + 1)].time;

		for (int a = 0; a < 3; a++)
			ev[e].grad.ampl[a] = proj[a] * grad[n].ampl;

		if (E <= e++)
			return -1;
	}

	return e;
}


/* compute parameters for softest symmetric trapezoid
 * */
bool grad_soft(struct grad_trapezoid* grad, double dur, double moment, struct grad_limits sys)
{
	if (0 > dur)
		return false;

	double rise = sys.max_amplitude * sys.inv_slew_rate;
	double flat = dur - 2. * rise;
	double ampl = moment / (rise + flat);

	if (0. > flat)
		return false;

	if (rise < (fabs(ampl) * sys.inv_slew_rate))
		return false;

	if ((fabs(ampl) > sys.max_amplitude))
		return false;

	grad->flat = flat;
	grad->ampl = ampl;
	grad->rampdown = rise;
	grad->rampup = rise;

	return true;
}


/* compute parameters for hardest symmetric trapezoid
 * */
bool grad_hard(struct grad_trapezoid* grad, double moment, struct grad_limits sys)
{
	double rise = sys.max_amplitude * sys.inv_slew_rate;
	double flat = 0.;

	if (rise < fabs(moment) / sys.max_amplitude) {

		flat = fabs(moment) / sys.max_amplitude - rise;

	} else {

		rise = sqrt(fabs(moment) * sys.inv_slew_rate);
	}

	grad->flat = flat;
	grad->rampdown = rise;
	grad->rampup = rise;
	grad->ampl = moment / (flat + rise);

	return true;
}

