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


/*
 * Find degenerated triangular events and merge them if possible.
 */
int seq_event_merge_degenerated_triangles(int E, struct seq_event ev[__VLA(E)])
{
	int ctr = 0;

	for (int e = 0; e < E; e++) {

		if (SEQ_EVENT_GRADIENT != ev[e].type)
			continue;

		if (ev[e].mid == ev[e].end) {

			for (int f = 0; f < E; f++) {

				if (SEQ_EVENT_GRADIENT != ev[f].type)
					continue;

				if (   (ev[f].start == ev[f].mid) 
				&& (ev[f].start == ev[e].end)
				&& (ev[f].grad.ampl[0] == ev[e].grad.ampl[0])
				&& (ev[f].grad.ampl[1] == ev[e].grad.ampl[1])
				&& (ev[f].grad.ampl[2] == ev[e].grad.ampl[2])) {

					ctr++;
					ev[e].end = ev[f].end;
					for (int a = 0; a < 3; a++)
						ev[f].grad.ampl[a] = 0.;
					break;
				}
			}
		}
	}

	return ctr;
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

	if (0. > flat) {

		ampl = 2. * moment / dur;

		if ((fabs(ampl) > sys.max_amplitude))
			return false;

		rise = dur / 2.;
		flat = 0.;
	}

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


static bool grad_support_check(const struct grad_support* support, struct grad_limits sys)
{
	if (0. > support->time)
		return false;

	if (fabs(support->ampl) > sys.max_amplitude)
		return false;

	return true;
}


static bool grad_support_check_interval(const struct grad_support* start, const struct grad_support* end, struct grad_limits sys)
{
	if (!grad_support_check(start, sys))
		return false;

	if (!grad_support_check(end, sys))
		return false;

	if (0. > end->time - start->time)
		return false;

	if (sys.inv_slew_rate > fabs((end->time - start->time) / (end->ampl - start->ampl)))
		return false;

	return true;
}


static int sign(float x)
{
	return (x > 0) - (x < 0);
}


static int grad_support_with_offsets_3point(struct grad_support support[4], double moment, struct grad_support start, struct grad_support end, struct grad_limits sys)
{
	support[0] = start;
	support[1] = (struct grad_support){ 0 };
	support[2] = end;
	support[3] = (struct grad_support){ 0 };

	double available_time = end.time - start.time;

	if (0. >= available_time)
		return -1;

	if ((start.ampl > sys.max_amplitude) || (end.ampl > sys.max_amplitude))
		return -2;

	if (sys.inv_slew_rate  > fabs((end.time - start.time) / (end.ampl - start.ampl)))
		return -3;

	int sign_sm = sign(moment * start.ampl);

	for (int i = 0; i < 10; i++) {

		double spacing = 0.;
		if (0 < sign_sm)
			spacing = 1 - powf(0.5, i + 1);
		else
			spacing = powf(0.5, i + 1);

		support[1].time = start.time + spacing * (end.time - start.time);
		support[1].ampl = 2 * moment / (end.time - start.time) - spacing * start.ampl - (1 - spacing) * end.ampl;

		if (   grad_support_check_interval(&start, &support[1], sys)
		    && grad_support_check_interval(&support[1], &end, sys))
			return 3;
	}

	return -4;
}


static int grad_support_with_offsets_4point(struct grad_support support[4], double moment, struct grad_support start, struct grad_support end, struct grad_limits sys)
{
	support[0] = start;
	support[1] = (struct grad_support){ 0 };
	support[2] = (struct grad_support){ 0 };
	support[3] = end;

	struct grad_support start2;
	struct grad_support end2;
	double moment2;

	if (sign(start.ampl) != sign(moment)) {

		support[1].ampl = sign(moment) * sys.max_amplitude;
		support[1].time = start.time + fabs((support[1].ampl - start.ampl) * sys.inv_slew_rate);
		moment2 = moment - (support[1].ampl + start.ampl) / 2. * fabs((support[1].ampl - start.ampl) * sys.inv_slew_rate);
		start2 = support[1];
		end2 = end;

	} else {

		support[2].ampl = sign(moment) * sys.max_amplitude;
		support[2].time = end.time - fabs((end.ampl - support[2].ampl) * sys.inv_slew_rate);
		moment2 = moment - (support[2].ampl + end.ampl) / 2. * fabs((end.ampl - support[2].ampl) * sys.inv_slew_rate);
		start2 = start;
		end2 = support[2];
	}

	double available_time = end2.time - start2.time;

	if (   fabs(moment2) > fabs((available_time) * (end2.ampl + start2.ampl) / 2.)
	    && (sys.max_amplitude < 2 * fabs(moment2) / available_time - start2.ampl)
	    && (sys.max_amplitude < 2 * fabs(moment2) / available_time - end2.ampl))
		return -11;

	struct grad_support support2[4] = { { 0 } };

	if (!grad_support_with_offsets_3point(support2, moment2, start2, end2, sys))
		return -12;

	if (sign(start.ampl) != sign(moment))
		support[2] = support2[1];
	else
		support[1] = support2[1];

	return 4;
}


/*
 * compute support points for with start/end gradient support point
 * returns number (n) of support points, where 0 is start, n-1 is end
 * 1 .. n-2 intermediate support points
 */
int grad_support_with_offsets(struct grad_support support[4], double moment, struct grad_support start, struct grad_support end, struct grad_limits sys)
{
	int s = grad_support_with_offsets_3point(support, moment, start, end, sys);

	if (0 < s)
		return s;

	return grad_support_with_offsets_4point(support, moment, start, end, sys);
}
