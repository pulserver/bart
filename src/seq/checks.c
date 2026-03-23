/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <stdbool.h>
#include <assert.h>
#include <math.h>

#include "misc/nested.h"

#include "seq/config.h"
#include "seq/event.h"
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

