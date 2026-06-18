/* Copyright 2025-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <assert.h>
#include <math.h>

#include "misc/nested.h"

#include "seq/config.h"
#include "seq/event.h"
#include "seq/kernel.h"
#include "seq/seq.h"

#include "checks.h"


void seq_rf_count(int N, long calls[N], int E, const struct seq_event ev[E])
{
	int rfs = events_counter(SEQ_EVENT_PULSE, E, ev);

	for (int i = 0; i < rfs; i++) {

		int id = ev[events_idx(i, SEQ_EVENT_PULSE, E, ev)].pulse.shape_id;

		assert(N > id);

		calls[id]++;
	}
}


/*
 * check limits of sequence events
 */
bool seq_check_gradients(int N, const struct seq_event ev[N], const struct seq_sys* sys)
{
	for (int i = 0; i < N; i++) {

		NESTED(int, check, (double x[/*3*/], double lim))
		{
			for (int k = 0; k < 3; k++)
				if (fabs(x[k]) > lim)
					return 0;

			return 1;
		};

		const double eps = 1.E-9; // numerical stability
		double t[3] = {	ev[i].start + eps, ev[i].mid + eps, ev[i].end - eps};
		double m[3];
		double d[3];

		for (int j = 0; j < 3; j++) {

			seq_gradient(m, t[j], N, ev);
			seq_slew(d, t[j], N, ev);

			if (!check(m, sys->grad.max_amplitude))
				return false;

			if (!check(d, (1 + eps) / sys->grad.inv_slew_rate))
				return false;
		}
	}
	
	return true;
}



/*
 * check ordering of sequence timings; RF and ADC dead times
 */
bool seq_check_timing(int N, const struct seq_event ev[N], const struct seq_sys* sys)
{
	int rfs = 0;
	int adcs = 0;

	for (int i = 0; i < N; i++) {

		if (ev[i].mid < ev[i].start)
			return false;

		if (ev[i].end < ev[i].mid)
			return false;

		if (SEQ_EVENT_ADC == ev[i].type)
			adcs++;
		else if (SEQ_EVENT_PULSE == ev[i].type)
			rfs++;
	}

	for (int r = 0; r < rfs; r++) {

		double time = 0.;
		int rf_idx = events_idx(r, SEQ_EVENT_PULSE, N, ev);

		if (ev[rf_idx].start - time < sys->coil_control_lead)
			return false;

		time = ev[rf_idx].end;

		for (int a = 0; a < adcs; a++)
			if (ev[events_idx(a, SEQ_EVENT_ADC, N, ev)].start - time < sys->min_duration_ro_rf)
				return false;

		time = ev[rf_idx].start;
	}

	return true;
}
